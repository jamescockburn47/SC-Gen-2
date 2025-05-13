# text_extraction_utils.py

import logging
import re
import json
from io import BytesIO
from typing import Tuple, Optional, Callable, Dict, Union

from PyPDF2 import PdfReader, errors as PyPDF2Errors
from pdfminer.high_level import extract_text as pdfminer_extract
from pdfminer.pdftypes import PDFException as PDFMinerException
from bs4 import BeautifulSoup

from config import MIN_MEANINGFUL_TEXT_LEN, logger

# Type alias for the OCR handler function
# Takes (pdf_bytes, company_no_for_logging)
# Returns (extracted_text, pages_processed, error_message_or_none)
OCRHandlerType = Callable[[bytes, str], Tuple[str, int, Optional[str]]]

def _reconstruct_text_from_ch_json(doc_content: Dict, company_no_for_logging: str) -> str:
    """
    Basic reconstruction of text from Companies House JSON content.
    This is a heuristic-based approach for iXBRL JSON.
    """
    logger.info(f"{company_no_for_logging}: Reconstructing text from CH JSON content.")
    text_parts = []

    def extract_values_from_node(node: Union[Dict, list], current_prefix: str = ""):
        if isinstance(node, dict):
            for key, value in node.items():
                # Heuristic: try to get meaningful keys, avoid purely technical ones.
                # Focus on longer string values that are not URLs or simple references.
                if isinstance(value, str) and len(value) > 20 and not value.startswith("http"):
                    if "contextRef" not in key and "unitRef" not in key and "dimension" not in key:
                        # Attempt to make key more readable for context
                        readable_key = key.replace("Value", "").replace("TextBlock", "")
                        # Remove common iXBRL prefixes for brevity if they exist
                        prefixes_to_remove = ["uk-gaap:", "uk-bus:", "core:", "ref:", "nonNumeric:", "num:", "link:", "xbrli:"]
                        for prefix in prefixes_to_remove:
                            if readable_key.startswith(prefix):
                                readable_key = readable_key[len(prefix):]
                        
                        # CamelCase/PascalCase to Title Case for better readability
                        readable_key = re.sub(r'([a-z])([A-Z])', r'\1 \2', readable_key).replace("_", " ").title()
                        
                        # Avoid adding overly generic or purely structural keys if value is simple
                        if not (readable_key.lower() in ["value", "text"] and len(value.split()) < 5):
                             text_parts.append(f"{current_prefix}{readable_key.strip()}: {value.strip()}")
                        else:
                            text_parts.append(f"{current_prefix}{value.strip()}") # Just add the value

                elif isinstance(value, (dict, list)):
                    extract_values_from_node(value, current_prefix) # Recursive call

        elif isinstance(node, list):
            for item in node:
                extract_values_from_node(item, current_prefix) # Process items in a list

    try:
        # Start extraction from known common top-level keys or the whole document
        # Common iXBRL data might be under 'facts', 'instance', or directly at root
        if 'facts' in doc_content and isinstance(doc_content['facts'], dict):
            for fact_group in doc_content['facts'].values(): # facts are often grouped by concept
                extract_values_from_node(fact_group)
        elif 'instance' in doc_content and isinstance(doc_content['instance'], dict) :
             extract_values_from_node(doc_content['instance'])
        else:
            extract_values_from_node(doc_content) # Process the whole JSON if no obvious entry points
            
        if not text_parts:
            extracted_text = "JSON content was available but yielded no reconstructable text with current generic logic."
            logger.warning(f"{company_no_for_logging}: CH JSON processing (generic) yielded no text. Full JSON sample: {json.dumps(doc_content)[:500]}")
        else:
            extracted_text = "\n".join(text_parts)
            # Further clean up: reduce excessive blank lines that might result from structure
            extracted_text = re.sub(r'\n\s*\n', '\n', extracted_text).strip()
            logger.info(f"{company_no_for_logging}: Textual representation generated from CH JSON ({len(extracted_text)} chars).")
            
    except Exception as e:
        logger.error(f"{company_no_for_logging}: Failed to process CH JSON content for text reconstruction: {e}", exc_info=True)
        extracted_text = f"Error: Could not process CH JSON content for text. Details: {str(e)}"
    return extracted_text


def _extract_text_from_xhtml(xhtml_content: str, company_no_for_logging: str) -> str:
    """Extracts text from XHTML content using BeautifulSoup."""
    logger.info(f"{company_no_for_logging}: Extracting text from XHTML content.")
    try:
        if not isinstance(xhtml_content, str):
            logger.warning(f"{company_no_for_logging}: XHTML content not string (type: {type(xhtml_content)}), converting.")
            xhtml_content = str(xhtml_content)
        
        soup = BeautifulSoup(xhtml_content, "html.parser") # html.parser is generally more lenient
        
        # Remove common non-content tags
        tags_to_remove = ["script", "style", "head", "meta", "link", "title", "header", "footer", "nav", "aside", "form", "button", "input", "noscript"]
        for tag_name in tags_to_remove:
            for tag_element in soup.find_all(tag_name):
                tag_element.decompose()
        
        # Try to find a main body or content div, otherwise use the whole soup
        body_tag = soup.find("body")
        if body_tag:
            text_content = body_tag.get_text(separator=" ", strip=True)
        else:
            text_content = soup.get_text(separator=" ", strip=True)

        # Clean up whitespace
        text_content = re.sub(r'\s\s+', ' ', text_content).strip() # Consolidate multiple spaces
        text_content = re.sub(r'(\n\s*){2,}', '\n\n', text_content).strip() # Consolidate multiple newlines

        if len(text_content) < MIN_MEANINGFUL_TEXT_LEN:
            logger.warning(f"{company_no_for_logging}: XHTML parsing yielded short text ({len(text_content)} chars). Preview: '{text_content[:100]}'")
        else:
            logger.info(f"{company_no_for_logging}: Text extracted from XHTML ({len(text_content)} chars).")
        return text_content
    except Exception as e_parse_xhtml:
        logger.error(f"{company_no_for_logging}: Failed to parse XHTML: {e_parse_xhtml}", exc_info=True)
        return f"Error: Could not parse XHTML content. Details: {str(e_parse_xhtml)}"


def _extract_text_from_pdf_std_libs(
    pdf_bytes: bytes,
    company_no_for_logging: str
) -> Tuple[str, Optional[str]]:
    """
    Attempts to extract text from PDF using PyPDF2 and then pdfminer.six.
    Returns (extracted_text, error_message_if_significant_failure).
    """
    text_from_pypdf2 = ""
    pypdf2_error = None
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        if reader.is_encrypted: # PyPDF2 can sometimes detect encryption
            try:
                if reader.decrypt("") == 0: # Try empty password
                     logger.warning(f"{company_no_for_logging}: PDF was encrypted but decrypted with empty password (PyPDF2).")
                else: # Decryption failed
                    pypdf2_error = "PyPDF2: PDF is encrypted and decryption failed."
                    logger.warning(f"{company_no_for_logging}: {pypdf2_error}")
                    # No point proceeding with PyPDF2 if encrypted and not decrypted
                    return "", pypdf2_error
            except Exception as e_decrypt:
                pypdf2_error = f"PyPDF2: Error during decryption attempt: {e_decrypt}"
                logger.warning(f"{company_no_for_logging}: {pypdf2_error}")
                return "", pypdf2_error


        text_parts_pypdf = [page.extract_text() or "" for page in reader.pages if hasattr(page, 'extract_text')]
        text_from_pypdf2 = "\n".join(filter(None, text_parts_pypdf)).strip()
        
        if len(text_from_pypdf2) >= MIN_MEANINGFUL_TEXT_LEN:
            logger.info(f"{company_no_for_logging}: Text extracted using PyPDF2 ({len(text_from_pypdf2)} chars).")
            return text_from_pypdf2, None
        elif text_from_pypdf2: # Got some text, but not much
             logger.info(f"{company_no_for_logging}: PyPDF2 extracted short text ({len(text_from_pypdf2)} chars). Will try pdfminer.")
        # else: no text from pypdf2

    except PyPDF2Errors.PdfReadError as e_pypdf_read: # More specific PyPDF2 read errors
        pypdf2_error = f"PyPDF2 ReadError: {e_pypdf_read}."
        logger.warning(f"{company_no_for_logging}: {pypdf2_error} Trying pdfminer.")
    except Exception as e_pypdf2: # Generic PyPDF2 errors
        pypdf2_error = f"PyPDF2 failed: {e_pypdf2}."
        logger.warning(f"{company_no_for_logging}: {pypdf2_error} Trying pdfminer.")
    
    # Try pdfminer.six if PyPDF2 failed or produced insufficient text
    text_from_pdfminer = ""
    pdfminer_error = None
    try:
        text_from_pdfminer = pdfminer_extract(BytesIO(pdf_bytes)).strip()
        if len(text_from_pdfminer) >= MIN_MEANINGFUL_TEXT_LEN:
            logger.info(f"{company_no_for_logging}: Text extracted using pdfminer.six ({len(text_from_pdfminer)} chars).")
            return text_from_pdfminer, None
        elif text_from_pdfminer: # Got some text, but not much
             logger.info(f"{company_no_for_logging}: pdfminer.six extracted short text ({len(text_from_pdfminer)} chars).")
        # else: no text from pdfminer
    except PDFMinerException as e_pdfminer_lib: # Specific pdfminer exceptions
        pdfminer_error = f"pdfminer.six library error: {e_pdfminer_lib}."
        logger.warning(f"{company_no_for_logging}: {pdfminer_error}")
    except Exception as e_pdfminer: # Generic pdfminer errors
        pdfminer_error = f"pdfminer.six failed: {e_pdfminer}."
        logger.warning(f"{company_no_for_logging}: {pdfminer_error}")

    # Determine best text from standard libs or combined error
    current_best_text = text_from_pdfminer if len(text_from_pdfminer) > len(text_from_pypdf2) else text_from_pypdf2
    
    if current_best_text:
        # If we have some text, even if short, return it. OCR can be decided by caller.
        logger.info(f"{company_no_for_logging}: Best text from std PDF libs ({len(current_best_text)} chars) - Preview: '{current_best_text[:50]}...'")
        return current_best_text, None # No critical error if some text extracted
    else:
        # Both failed to get any text
        combined_error = f"Both PyPDF2 ({pypdf2_error or 'no text'}) and pdfminer.six ({pdfminer_error or 'no text'}) failed to extract text."
        logger.warning(f"{company_no_for_logging}: {combined_error}")
        return "", combined_error # Return empty string and the combined error message


def extract_text_from_document(
    doc_content_input: Union[bytes, str, Dict],
    content_type_input: str,
    company_no_for_logging: str = "N/A_DocExtract",
    ocr_handler: Optional[OCRHandlerType] = None
) -> Tuple[str, int, Optional[str]]:
    """
    Extracts text from various document content types.
    Uses standard libraries and an optional OCR handler for PDFs.

    Args:
        doc_content_input: The document content (bytes for PDF, str for XHTML, dict for JSON).
        content_type_input: The type of content ("pdf", "xhtml", "json").
        company_no_for_logging: Identifier for logging.
        ocr_handler: Optional function to call for PDF OCR if standard methods fail.
                     Expected signature: ocr_handler(pdf_bytes, log_id) -> (text, pages_processed, error_msg_or_None)

    Returns:
        A tuple: (extracted_text_str, pages_processed_by_ocr_int, error_message_str_or_None).
        pages_processed_by_ocr_int is 0 if OCR was not used or failed.
        error_message_str_or_None contains an error message if a significant failure occurred.
    """
    extracted_text = ""
    pages_ocrd = 0
    error_msg = None

    if content_type_input == "json":
        if isinstance(doc_content_input, dict):
            extracted_text = _reconstruct_text_from_ch_json(doc_content_input, company_no_for_logging)
        else:
            error_msg = f"Expected dict for JSON content, got {type(doc_content_input)}. Cannot process."
            logger.error(f"{company_no_for_logging}: {error_msg}")
            extracted_text = f"Error: {error_msg}" # Return error in text if this happens
        # No pages processed for JSON

    elif content_type_input == "xhtml":
        if isinstance(doc_content_input, str):
            extracted_text = _extract_text_from_xhtml(doc_content_input, company_no_for_logging)
        else:
            error_msg = f"Expected str for XHTML content, got {type(doc_content_input)}. Cannot process."
            logger.error(f"{company_no_for_logging}: {error_msg}")
            extracted_text = f"Error: {error_msg}"
        # No pages processed for XHTML

    elif content_type_input == "pdf":
        if not isinstance(doc_content_input, bytes):
            error_msg = f"Expected bytes for PDF content, got {type(doc_content_input)}. Cannot process."
            logger.error(f"{company_no_for_logging}: {error_msg}")
            extracted_text = f"Error: {error_msg}"
            return extracted_text, 0, error_msg # Early exit

        # Try standard PDF text extraction first
        std_text, std_err = _extract_text_from_pdf_std_libs(doc_content_input, company_no_for_logging)
        
        if len(std_text) >= MIN_MEANINGFUL_TEXT_LEN:
            logger.info(f"{company_no_for_logging}: Sufficient text from standard PDF libs ({len(std_text)} chars). OCR skipped.")
            extracted_text = std_text
        else:
            if std_text: # Some text, but not much
                logger.info(f"{company_no_for_logging}: Standard PDF libs yielded short text ({len(std_text)} chars). Preview: '{std_text[:100]}...'")
            else: # No text from standard libs
                logger.warning(f"{company_no_for_logging}: No text from standard PDF libs. Error (if any): {std_err}")

            if ocr_handler:
                logger.info(f"{company_no_for_logging}: Attempting OCR for PDF using provided handler.")
                ocr_text, pages_ocrd_by_handler, ocr_err = ocr_handler(doc_content_input, company_no_for_logging)
                pages_ocrd = pages_ocrd_by_handler # Store pages processed by OCR

                if ocr_err: # OCR process itself reported an error
                    logger.warning(f"{company_no_for_logging}: OCR handler reported an error: {ocr_err}")
                    error_msg = ocr_err # OCR error becomes the main error if std libs also failed badly
                
                if ocr_text and len(ocr_text) >= MIN_MEANINGFUL_TEXT_LEN / 2: # Use OCR text if it's somewhat substantial
                    extracted_text = ocr_text
                    logger.info(f"{company_no_for_logging}: Using text from OCR ({len(ocr_text)} chars).")
                elif std_text: # OCR failed or insufficient, but std libs had *some* text
                    extracted_text = std_text
                    logger.warning(f"{company_no_for_logging}: OCR text insufficient or failed; falling back to short text from standard libs ('{std_text[:50]}...'). OCR error (if any): {ocr_err}")
                elif ocr_text: # OCR produced very short text, and std_text was empty
                    extracted_text = ocr_text
                    logger.warning(f"{company_no_for_logging}: OCR text very short ({len(ocr_text)} chars), std libs empty. Using OCR text. OCR error (if any): {ocr_err}")
                else: # Both std libs and OCR failed to get any meaningful text
                    extracted_text = "" # Ensure it's an empty string
                    logger.warning(f"{company_no_for_logging}: All PDF text extraction methods (standard and OCR) failed or yielded no text. Std error: {std_err}, OCR error: {ocr_err}")
                    if not error_msg: error_msg = std_err or ocr_err or "All PDF extraction methods failed."
            else: # No OCR handler provided
                extracted_text = std_text # Use whatever standard libs produced
                if not extracted_text:
                    logger.warning(f"{company_no_for_logging}: No text from standard PDF libs and no OCR handler provided. Std error (if any): {std_err}")
                    error_msg = std_err or "Standard PDF extraction failed, no OCR handler."
    else:
        error_msg = f"Unknown content_type '{content_type_input}' for text extraction."
        logger.error(f"{company_no_for_logging}: {error_msg}")
        extracted_text = f"Error: {error_msg}"

    # Final check on extracted_text before returning
    if "Error:" in extracted_text and not error_msg : # If text contains "Error:" but error_msg is not set
        error_msg = extracted_text # Promote the text's error message

    if not extracted_text.strip() and not error_msg:
        # If text is empty/whitespace but no explicit error was flagged,
        # it means extraction happened but found nothing. This isn't an "error" in the process.
        logger.info(f"{company_no_for_logging}: Text extraction resulted in empty content for type '{content_type_input}'.")
        
    return extracted_text, pages_ocrd, error_msg