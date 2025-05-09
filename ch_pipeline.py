# ch_pipeline.py

from __future__ import annotations

import csv
import time
import datetime
import logging
from pathlib import Path
import tempfile
import json
from typing import List, Tuple, Dict, Optional, Any, Union

import pandas as pd

# Configuration and Core Utilities
from config import (
    logger, # Use the centrally configured logger
    MIN_MEANINGFUL_TEXT_LEN,
    MAX_DOCS_TO_PROCESS_PER_COMPANY
)

# CH API Interactions
from ch_api_utils import (
    get_ch_documents_metadata,
    _fetch_document_content_from_ch # Renamed from _fetch_document_content
)

# Text Extraction
from text_extraction_utils import extract_text_from_document, OCRHandlerType

# AI Summarization
from ai_utils import gpt_summarise_ch_docs, gemini_summarise_ch_docs

# --- Optional AWS Textract Import ---
# We'll try to import it and set up a handler if requested.
try:
    from aws_textract_utils import perform_textract_ocr, get_textract_cost_estimation, _initialize_aws_clients as initialize_textract_aws_clients
    TEXTRACT_AVAILABLE = True
    logger.info("aws_textract_utils.py found and imported successfully.")
except ImportError:
    perform_textract_ocr = None
    get_textract_cost_estimation = None
    initialize_textract_aws_clients = None
    TEXTRACT_AVAILABLE = False
    logger.warning("aws_textract_utils.py not found or failed to import. Textract OCR will not be available.")


def find_group_companies(parent_co_no: str) -> list[str]:
    """
    Placeholder for group company discovery.
    Currently, it only returns the parent company itself.
    Future enhancement could involve querying APIs or databases for subsidiary information.
    """
    logger.info(f"Group discovery for parent company {parent_co_no} currently returns only the parent. (Placeholder)")
    return [parent_co_no]


def _save_raw_document_content(
    doc_content: Union[bytes, str, Dict],
    doc_type_str: str, # 'pdf', 'xhtml', 'json'
    company_no: str,
    ch_doc_code: str, # e.g., AA, CS01
    doc_year: int,
    scratch_dir: Path
) -> Optional[Path]:
    """Saves raw document content to the scratch directory."""
    file_extension_map = {"pdf": "pdf", "xhtml": "xhtml", "json": "json"}
    file_extension = file_extension_map.get(doc_type_str, "dat")
    
    # Create a more structured filename
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    doc_filename_prefix = f"{company_no}_{ch_doc_code}_{doc_year}_{timestamp_str}"
    doc_save_path = scratch_dir / f"{doc_filename_prefix}.{file_extension}"

    try:
        if isinstance(doc_content, bytes):
            doc_save_path.write_bytes(doc_content)
        elif isinstance(doc_content, str):
            doc_save_path.write_text(doc_content, encoding='utf-8')
        elif isinstance(doc_content, dict) and doc_type_str == "json":
            with open(doc_save_path, "w", encoding="utf-8") as f_json:
                json.dump(doc_content, f_json, indent=2)
        else:
            logger.error(f"Cannot save document: unsupported content type '{type(doc_content)}' for '{doc_type_str}'.")
            return None
        logger.debug(f"Saved fetched {doc_type_str.upper()} document to: {doc_save_path.name}")
        return doc_save_path
    except IOError as e_save:
        logger.error(f"Failed to save fetched {doc_type_str.upper()} document ({doc_save_path.name}): {e_save}")
        return None


def _cleanup_scratch_directory(scratch_dir: Path, keep_days: int):
    """Removes old files from the scratch directory."""
    if keep_days < 0:
        logger.info(f"Scratch cleanup skipped (keep_days < 0). Path: {scratch_dir}")
        return

    timestamp_cutoff = time.time() - (keep_days * 86400)
    num_files_cleaned = 0
    try:
        for item_path in scratch_dir.iterdir():
            if item_path.is_file():
                try:
                    if item_path.stat().st_mtime < timestamp_cutoff or keep_days == 0:
                        item_path.unlink()
                        num_files_cleaned += 1
                except OSError as e_unlink:
                    logger.warning(f"Could not delete old file {item_path.name} from scratch: {e_unlink}")
        if num_files_cleaned > 0:
            logger.info(f"Cleaned up {num_files_cleaned} old files from scratch directory: {scratch_dir}")
    except Exception as e_cleanup:
        logger.error(f"Error during scratch directory cleanup ({scratch_dir}): {e_cleanup}")


def run_ch_document_pipeline_for_company(
    company_no: str,
    selected_categories: List[str],
    start_year: int,
    end_year: int,
    scratch_dir: Path,
    use_textract: bool = False # New parameter
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Runs the document fetching and text extraction part of the pipeline for a single company.

    Args:
        company_no: The company registration number.
        selected_categories: List of CH document categories to fetch.
        start_year: Start year for filtering documents.
        end_year: End year for filtering documents.
        scratch_dir: Directory to save temporary files.
        use_textract: Whether to attempt Textract OCR for PDFs.

    Returns:
        A tuple:
            - list_of_extracted_text_data (each dict contains 'company_no', 'ch_doc_type', 'year', 'source_format', 'text', 'error')
            - total_pages_processed_by_ocr (int)
            - total_pdfs_sent_to_ocr (int)
    """
    logger.info(f"Starting document pipeline for company: {company_no} (Textract OCR: {'Enabled' if use_textract else 'Disabled'})")
    
    all_extracted_texts_data: List[Dict[str, Any]] = []
    total_pages_ocrd_for_company = 0
    pdfs_sent_to_ocr_for_company = 0

    # 1. Get document metadata from CH
    doc_metadata_items, meta_error = get_ch_documents_metadata(company_no, selected_categories, start_year, end_year)
    if meta_error:
        logger.error(f"Failed to get document metadata for {company_no}: {meta_error}")
        all_extracted_texts_data.append({
            "company_no": company_no, "ch_doc_type": "METADATA_ERROR", "year": 0,
            "source_format": "N/A", "text": "", "error": meta_error
        })
        return all_extracted_texts_data, 0, 0
    
    if not doc_metadata_items:
        logger.warning(f"No document metadata found for {company_no} matching criteria.")
        return [], 0, 0

    # Determine OCR handler based on user preference and availability
    ocr_handler_to_use: Optional[OCRHandlerType] = None
    if use_textract:
        if TEXTRACT_AVAILABLE and perform_textract_ocr and initialize_textract_aws_clients:
            if initialize_textract_aws_clients(): # Ensure AWS clients are ready
                ocr_handler_to_use = perform_textract_ocr
                logger.info(f"Textract OCR is enabled and will be used for PDFs for {company_no} if needed.")
            else:
                logger.warning(f"Textract OCR was requested for {company_no}, but AWS clients could not be initialized. OCR will be skipped.")
        else:
            logger.warning(f"Textract OCR was requested for {company_no}, but 'aws_textract_utils' is not available or configured. OCR will be skipped.")

    processed_doc_transaction_ids = set()
    docs_processed_count = 0

    # Sort by date again just in case, and limit processing
    doc_metadata_items.sort(key=lambda x: x.get("date", "0000-00-00"), reverse=True)

    for item_meta in doc_metadata_items:
        if docs_processed_count >= MAX_DOCS_TO_PROCESS_PER_COMPANY:
            logger.info(f"{company_no}: Reached document processing limit ({MAX_DOCS_TO_PROCESS_PER_COMPANY}).")
            break
        
        # Deduplicate based on transaction ID or metadata link
        unique_doc_identifier = item_meta.get("transaction_id") or item_meta.get("links", {}).get("document_metadata")
        if not unique_doc_identifier or unique_doc_identifier in processed_doc_transaction_ids:
            if unique_doc_identifier: logger.debug(f"{company_no}: Skipping already processed doc (ID: {unique_doc_identifier})")
            continue
        
        ch_doc_type_code = item_meta.get("type", "UNKNOWN_CH_TYPE")
        doc_date_str = item_meta.get("date", "")
        try:
            doc_year_val = int(doc_date_str[:4])
        except ValueError:
            logger.warning(f"{company_no}: Could not parse year from date '{doc_date_str}' for doc {unique_doc_identifier}. Skipping.")
            continue
        
        # Year filtering should ideally be done by get_ch_documents_metadata, but double-check
        if not (start_year <= doc_year_val <= end_year):
            continue

        logger.info(f"{company_no}: Processing document: Type '{ch_doc_type_code}', Date '{doc_date_str}', Desc '{item_meta.get('description', 'N/A')}' (ID: {unique_doc_identifier})")

        # 2. Fetch actual document content
        doc_content_data, fetched_content_type, fetch_err = _fetch_document_content_from_ch(company_no, item_meta)

        if fetch_err or not doc_content_data or fetched_content_type == "none":
            logger.warning(f"{company_no}: Failed to fetch content for doc {unique_doc_identifier}. Error: {fetch_err or 'No content returned.'}")
            all_extracted_texts_data.append({
                "company_no": company_no, "ch_doc_type": ch_doc_type_code, "year": doc_year_val,
                "source_format": "FETCH_ERROR", "text": "", "error": fetch_err or 'No content returned.'
            })
            if unique_doc_identifier: processed_doc_transaction_ids.add(unique_doc_identifier)
            docs_processed_count += 1
            continue

        # Optionally save raw fetched document
        _save_raw_document_content(doc_content_data, fetched_content_type, company_no, ch_doc_type_code, doc_year_val, scratch_dir)

        # 3. Extract text from the document content
        # Pass the selected ocr_handler (if any) to extract_text_from_document
        extracted_text, pages_ocrd, extract_err = extract_text_from_document(
            doc_content_data, fetched_content_type, company_no,
            ocr_handler=ocr_handler_to_use if fetched_content_type == 'pdf' else None
        )
        
        if pages_ocrd > 0: # If OCR was used and processed pages
            total_pages_ocrd_for_company += pages_ocrd
            pdfs_sent_to_ocr_for_company += 1

        # Store result even if there was an error or no text
        result_data = {
            "company_no": company_no,
            "ch_doc_type": ch_doc_type_code,
            "year": doc_year_val,
            "source_format": fetched_content_type.upper(),
            "text": extracted_text,
            "error": extract_err # This will be None if extraction was successful
        }
        all_extracted_texts_data.append(result_data)

        if extract_err:
            logger.warning(f"{company_no}: Error extracting text from doc {unique_doc_identifier} (Type: {fetched_content_type}): {extract_err}")
        elif not extracted_text or len(extracted_text.strip()) < MIN_MEANINGFUL_TEXT_LEN / 2:
            logger.warning(f"{company_no}: No significant text extracted for doc {unique_doc_identifier} (Type: {fetched_content_type}). Preview: '{extracted_text[:100]}...'")
        else:
            logger.info(f"{company_no}: Successfully extracted text from doc {unique_doc_identifier} (Type: {fetched_content_type}, Length: {len(extracted_text)} chars).")
            
        if unique_doc_identifier: processed_doc_transaction_ids.add(unique_doc_identifier)
        docs_processed_count += 1

    logger.info(f"Finished document pipeline for {company_no}. Extracted data for {len(all_extracted_texts_data)} items. OCR Pages: {total_pages_ocrd_for_company}, PDFs to OCR: {pdfs_sent_to_ocr_for_company}")
    return all_extracted_texts_data, total_pages_ocrd_for_company, pdfs_sent_to_ocr_for_company


def run_batch_company_analysis(
    csv_path: Path,
    selected_categories: List[str],
    start_year: int,
    end_year: int,
    ai_model_identifier: str, # e.g., "gpt-4o-mini" or "gemini-1.5-pro-latest"
    specific_ai_instructions: str = "",
    base_scratch_dir: Optional[Path] = None, # Base for creating timestamped run folders
    keep_days: int = 7,
    use_textract_ocr: bool = False # To control Textract usage
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    Main batch processing function for Companies House analysis.
    Orchestrates fetching, text extraction, and AI summarization for companies in a CSV.

    Args:
        csv_path: Path to the input CSV file with company numbers.
        selected_categories: List of CH document categories.
        start_year: Start year for document filtering.
        end_year: End year for document filtering.
        ai_model_identifier: Name of the AI model to use for summarization.
        specific_ai_instructions: Additional instructions for the AI.
        base_scratch_dir: Optional base directory for scratch files. A timestamped subfolder will be created.
        keep_days: How long to keep files in the run-specific scratch folder.
        use_textract_ocr: Boolean flag to enable Textract OCR.

    Returns:
        A tuple: (output_csv_digest_path_or_None, batch_processing_metrics_dict).
    """
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_scratch_dir:
        run_scratch_dir = base_scratch_dir / f"ch_run_{run_timestamp}"
    else:
        run_scratch_dir = Path(tempfile.gettempdir()) / f"ch_pipeline_scratch_{run_timestamp}"
    
    try:
        run_scratch_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Batch run using scratch directory: {run_scratch_dir}")
    except OSError as e_mkdir:
        logger.error(f"CRITICAL: Could not create scratch directory {run_scratch_dir}: {e_mkdir}. Aborting batch.")
        return None, {"error": "Failed to create scratch directory.", "notes": str(e_mkdir)}

    output_data_rows: List[Dict[str, Any]] = []
    parent_company_numbers: List[str] = []

    # 1. Read Company Numbers from CSV
    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8-sig') as fh_csv:
            reader = csv.reader(fh_csv)
            header = next(reader, None) # Skip header
            if header: logger.info(f"Input CSV header: {header}")
            
            for i, row in enumerate(reader):
                if row and row[0].strip():
                    company_no_cleaned = row[0].strip().upper().replace(" ", "").zfill(8)
                    # Basic validation for 8 alphanumeric chars
                    if len(company_no_cleaned) == 8 and company_no_cleaned.isalnum():
                        parent_company_numbers.append(company_no_cleaned)
                    else:
                        logger.warning(f"Skipping invalid company number on input CSV row {i+2}: '{row[0]}'")
        logger.info(f"Loaded {len(parent_company_numbers)} valid company numbers from {csv_path.name}.")
    except FileNotFoundError:
        logger.error(f"CRITICAL: Input CSV file not found: {csv_path}. Aborting batch.")
        return None, {"error": "Input CSV not found.", "notes": str(csv_path)}
    except Exception as e_read_csv:
        logger.error(f"CRITICAL: Error reading input CSV {csv_path.name}: {e_read_csv}.", exc_info=True)
        return None, {"error": "Failed to read input CSV.", "notes": str(e_read_csv)}

    if not parent_company_numbers:
        logger.warning("No valid company numbers loaded from CSV. Batch processing cannot continue.")
        # Create an empty digest for consistency
        empty_digest_filename = f"digest_empty_input_{run_timestamp}.csv"
        empty_digest_path = run_scratch_dir / empty_digest_filename
        try:
            pd.DataFrame([]).to_csv(empty_digest_path, index=False, encoding='utf-8-sig')
        except Exception as e_write_empty: logger.error(f"Failed to write empty input digest: {e_write_empty}")
        return empty_digest_path, {
            "notes": "No companies processed due to empty or invalid input CSV.",
            "total_companies_processed": 0,
            "aws_ocr_costs": {} # Empty cost dict
        }

    # --- Batch Processing Metrics ---
    batch_total_pages_ocrd = 0
    batch_total_pdfs_to_ocr = 0
    companies_successfully_summarized = 0
    companies_with_extraction_errors = 0

    # 2. Process each parent company
    for parent_co_no in parent_company_numbers:
        logger.info(f">>> Processing parent company: {parent_co_no}")
        # Placeholder: In a real scenario, find_group_companies would be more complex.
        group_member_co_numbers = find_group_companies(parent_co_no)
        
        all_texts_for_group_summary: List[str] = []
        member_processing_logs: List[str] = []
        has_critical_extraction_error_for_group = False

        for member_co_no in group_member_co_numbers:
            extracted_data, pages_ocrd_member, pdfs_to_ocr_member = run_ch_document_pipeline_for_company(
                member_co_no, selected_categories, start_year, end_year, run_scratch_dir, use_textract_ocr
            )
            batch_total_pages_ocrd += pages_ocrd_member
            batch_total_pdfs_to_ocr += pdfs_to_ocr_member
            
            num_docs_with_text = 0
            num_docs_with_errors = 0

            for data_item in extracted_data:
                if data_item.get("error"):
                    num_docs_with_errors += 1
                    has_critical_extraction_error_for_group = True # Flag if any doc in group had extraction error
                    # Log the specific error for the document
                    logger.warning(f"Error for {data_item['company_no']} Doc: {data_item['ch_doc_type']} Yr: {data_item['year']} - {data_item['error']}")

                # Check if text is meaningful (not just an error message and meets length)
                text_content = data_item.get("text", "")
                if text_content and "Error:" not in text_content and len(text_content.strip()) > MIN_MEANINGFUL_TEXT_LEN / 2:
                    all_texts_for_group_summary.append(
                        f"[Text from Company: {data_item['company_no']}, CH Doc Type: {data_item['ch_doc_type']}, "
                        f"Year: {data_item['year']}, Source Format: {data_item['source_format']}]\n{text_content}"
                    )
                    num_docs_with_text +=1
            
            member_processing_logs.append(
                f"{member_co_no} ({num_docs_with_text}/{len(extracted_data)} docs yielded usable text; {num_docs_with_errors} errors)"
            )

        # 3. AI Summarization for the group
        final_summary_text = "No processable documents or text extracted that meets criteria for summarization."
        full_text_char_count = 0

        if all_texts_for_group_summary:
            full_text_for_ai = "\n\n===[END OF DOCUMENT/SECTION]===\n\n".join(all_texts_for_group_summary)
            full_text_char_count = len(full_text_for_ai)
            logger.info(f"Combined text for AI summary (Parent Co: {parent_co_no}) has {full_text_char_count:,} chars from {len(all_texts_for_group_summary)} text blocks.")

            if ai_model_identifier.startswith("gemini"):
                final_summary_text = gemini_summarise_ch_docs(
                    full_text_for_ai, parent_co_no, specific_ai_instructions, ai_model_identifier
                )
            elif ai_model_identifier.startswith("gpt"):
                final_summary_text = gpt_summarise_ch_docs(
                    full_text_for_ai, parent_co_no, specific_ai_instructions, ai_model_identifier
                )
            else:
                logger.warning(f"Unknown AI model identifier prefix: '{ai_model_identifier}'. Skipping summarization.")
                final_summary_text = f"Error: Unknown AI model ('{ai_model_identifier}') specified for summarization."
            
            if "Error:" not in final_summary_text and "No content provided" not in final_summary_text:
                companies_successfully_summarized +=1
        else:
            logger.warning(f"No text was extracted for any member of parent group {parent_co_no}. No AI summary will be generated.")
            if has_critical_extraction_error_for_group:
                companies_with_extraction_errors +=1


        output_data_rows.append({
            "parent_company_no": parent_co_no,
            "processed_group_members_log": "; ".join(member_processing_logs),
            "summary_of_findings": final_summary_text.replace("\n", "  "), # Flatten for CSV
            "combined_text_char_count": full_text_char_count,
        })
        if len(parent_company_numbers) > 1: # Small delay if processing multiple companies
            time.sleep(0.5)

    # 4. Finalize batch results and costs
    aws_cost_metrics = {}
    if use_textract_ocr and TEXTRACT_AVAILABLE and get_textract_cost_estimation:
        aws_cost_metrics = get_textract_cost_estimation(batch_total_pages_ocrd, batch_total_pdfs_to_ocr)
        logger.info(f"Final AWS Cost Estimation for CH Batch OCR: {aws_cost_metrics}")
    elif use_textract_ocr: # Requested but not available/used
        aws_cost_metrics = {"notes": "Textract OCR was requested but not available or not used. No OCR costs."}
    else: # Not requested
        aws_cost_metrics = {"notes": "Textract OCR was not requested. No OCR costs."}

    batch_metrics = {
        "total_parent_companies_processed": len(parent_company_numbers),
        "companies_successfully_summarized": companies_successfully_summarized,
        "companies_with_extraction_errors_in_group": companies_with_extraction_errors,
        "total_docs_considered_for_extraction_across_all_members": "N/A (Track per company in logs)", # More complex to sum up here
        "total_textract_pages_processed": batch_total_pages_ocrd,
        "total_pdfs_sent_to_textract": batch_total_pdfs_to_ocr,
        "aws_ocr_costs": aws_cost_metrics,
        "ai_model_used_for_summaries": ai_model_identifier,
        "run_timestamp": run_timestamp
    }

    # 5. Write output digest CSV
    output_csv_path = None
    if output_data_rows:
        digest_filename = f"digest_CH_analysis_report_{run_timestamp}.csv"
        output_csv_path = run_scratch_dir / digest_filename
        try:
            df_output = pd.DataFrame(output_data_rows)
            # Ensure consistent column order
            output_cols = ["parent_company_no", "processed_group_members_log", "summary_of_findings", "combined_text_char_count"]
            df_output = df_output.reindex(columns=output_cols)
            df_output.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"Batch processing complete. Output digest: {output_csv_path}")
        except Exception as e_write_digest:
            logger.error(f"Failed to write final output digest CSV to {output_csv_path}: {e_write_digest}", exc_info=True)
            output_csv_path = None # Indicate failure
            batch_metrics["error"] = "Failed to write output digest CSV."
    elif parent_company_numbers: # Processed companies but no rows (e.g., all failed very early)
        logger.warning("No data rows generated, though companies were processed. Creating an empty digest with notes.")
        no_data_digest_filename = f"digest_processing_yielded_no_data_{run_timestamp}.csv"
        output_csv_path = run_scratch_dir / no_data_digest_filename
        try:
            pd.DataFrame([{"parent_company_no": "N/A", 
                           "processed_group_members_log": "All processed companies yielded no usable data for summary or encountered critical errors.", 
                           "summary_of_findings": "No summary generated.", 
                           "combined_text_char_count": 0}]).to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        except Exception as e_write_no_data: logger.error(f"Failed to write 'no data' digest: {e_write_no_data}")
        batch_metrics["notes"] = (batch_metrics.get("notes","") + " Processing yielded no data rows.").strip()
    # (Empty input CSV case already handled and returns early)

    # 6. Cleanup scratch directory
    _cleanup_scratch_directory(run_scratch_dir, keep_days)

    return output_csv_path, batch_metrics


# --- Standalone Test Block ---
if __name__ == '__main__':
    logger.info("Running ch_pipeline.py as standalone test.")
    
    # Check for necessary environment variables for a full test
    # Note: AWS keys are not strictly required if use_textract_ocr is False for the test.
    required_env_vars = ['OPENAI_API_KEY', 'CH_API_KEY'] # GEMINI_API_KEY optional for specific model test
    if not all(os.getenv(var) for var in required_env_vars):
        logger.critical(f"Standalone test requires missing env vars: {', '.join(var for var in required_env_vars if not os.getenv(var))}. Some tests may fail or be skipped.")
    
    # Create a dummy CSV for testing
    test_csv_dir = Path(tempfile.mkdtemp(prefix="ch_pipeline_test_"))
    standalone_test_csv_path = test_csv_dir / "test_ch_pipeline_input.csv"
    # Use a known company number, e.g., Google UK or BP
    test_company_no = os.getenv("TEST_COMPANY_NUMBER_CH", "03977602") # Google UK Ltd.

    with open(standalone_test_csv_path, "w", newline="", encoding="utf-8") as f_test_csv:
        csv_writer = csv.writer(f_test_csv)
        csv_writer.writerow(["CompanyNumber", "OptionalOtherColumn"])
        csv_writer.writerow([test_company_no, "Test Data"])
    logger.info(f"Created test input CSV: {standalone_test_csv_path} for company {test_company_no}")

    test_scratch_base = Path(tempfile.mkdtemp(prefix="ch_pipeline_test_scratch_base_"))
    logger.info(f"Using temporary base scratch for standalone test runs: {test_scratch_base}")

    test_categories = ['accounts'] # Focus on accounts for testing
    current_test_year = datetime.datetime.now().year
    test_start_year = current_test_year - 3
    test_end_year = current_test_year -1 # Use previous years as current year might not have filings yet

    test_instructions = "Highlight any changes in Total Assets and Net Profit After Tax year on year."

    # --- Test 1: OpenAI without Textract ---
    logger.info(f"\n--- Test Run with OpenAI (default model), Textract Disabled ---")
    try:
        result_path_o_no_ocr, metrics_o_no_ocr = run_batch_company_analysis(
            csv_path=standalone_test_csv_path,
            selected_categories=test_categories,
            start_year=test_start_year,
            end_year=test_end_year,
            ai_model_identifier="gpt-4o-mini", # Or use OPENAI_MODEL_DEFAULT from config
            specific_ai_instructions=test_instructions,
            base_scratch_dir=test_scratch_base,
            keep_days=0,
            use_textract_ocr=False
        )
        if result_path_o_no_ocr: logger.info(f"OpenAI (No OCR) Test Output Digest: {result_path_o_no_ocr}")
        else: logger.error("OpenAI (No OCR) Test Run DID NOT produce an output digest path.")
        logger.info(f"OpenAI (No OCR) Test Metrics: {json.dumps(metrics_o_no_ocr, indent=2)}")
    except Exception as e_test1:
        logger.error(f"Error during OpenAI (No OCR) test: {e_test1}", exc_info=True)

    # --- Test 2: OpenAI with Textract (if available) ---
    if TEXTRACT_AVAILABLE and os.getenv("S3_TEXTRACT_BUCKET"): # Also check S3 bucket for good measure
        logger.info(f"\n--- Test Run with OpenAI (default model), Textract Enabled ---")
        try:
            result_path_o_ocr, metrics_o_ocr = run_batch_company_analysis(
                csv_path=standalone_test_csv_path,
                selected_categories=test_categories,
                start_year=test_start_year,
                end_year=test_end_year,
                ai_model_identifier="gpt-4o-mini",
                specific_ai_instructions=test_instructions,
                base_scratch_dir=test_scratch_base,
                keep_days=0,
                use_textract_ocr=True
            )
            if result_path_o_ocr: logger.info(f"OpenAI (OCR) Test Output Digest: {result_path_o_ocr}")
            else: logger.error("OpenAI (OCR) Test Run DID NOT produce an output digest path.")
            logger.info(f"OpenAI (OCR) Test Metrics: {json.dumps(metrics_o_ocr, indent=2)}")
        except Exception as e_test2:
            logger.error(f"Error during OpenAI (OCR) test: {e_test2}", exc_info=True)
    else:
        logger.warning("\n--- Skipping Textract enabled test as TEXTRACT_AVAILABLE is False or S3_TEXTRACT_BUCKET not set. ---")

    # --- Test 3: Gemini (if available) without Textract ---
    if os.getenv("GEMINI_API_KEY"):
        logger.info(f"\n--- Test Run with Gemini (default model), Textract Disabled ---")
        try:
            result_path_g_no_ocr, metrics_g_no_ocr = run_batch_company_analysis(
                csv_path=standalone_test_csv_path,
                selected_categories=test_categories,
                start_year=test_start_year,
                end_year=test_end_year,
                ai_model_identifier="gemini-1.5-flash-latest", # Or use GEMINI_MODEL_DEFAULT
                specific_ai_instructions="Focus on reported turnover and profit before tax figures for the last two available years.",
                base_scratch_dir=test_scratch_base,
                keep_days=0,
                use_textract_ocr=False
            )
            if result_path_g_no_ocr: logger.info(f"Gemini (No OCR) Test Output Digest: {result_path_g_no_ocr}")
            else: logger.error("Gemini (No OCR) Test Run DID NOT produce an output digest path.")
            logger.info(f"Gemini (No OCR) Test Metrics: {json.dumps(metrics_g_no_ocr, indent=2)}")
        except Exception as e_test3:
            logger.error(f"Error during Gemini (No OCR) test: {e_test3}", exc_info=True)

    else:
        logger.warning("\n--- Skipping Gemini test as GEMINI_API_KEY not set. ---")
        
    # Cleanup test files and dirs
    try:
        if standalone_test_csv_path.exists(): standalone_test_csv_path.unlink()
        if test_csv_dir.exists(): test_csv_dir.rmdir() # Remove dir only if empty
        # Scratch sub-folders will be cleaned by _cleanup_scratch_directory if keep_days=0
        # To remove the base scratch dir for testing (if it's empty):
        # if test_scratch_base.exists() and not any(test_scratch_base.iterdir()): test_scratch_base.rmdir()
        logger.info(f"Test cleanup: Removed {standalone_test_csv_path.name} and {test_csv_dir.name}. Base scratch: {test_scratch_base} (subfolders cleaned by pipeline).")
    except Exception as e_cleanup_main:
        logger.warning(f"Error during main test cleanup: {e_cleanup_main}")