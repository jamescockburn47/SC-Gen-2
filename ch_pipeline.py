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
    logger,
    MIN_MEANINGFUL_TEXT_LEN,
    MAX_DOCS_TO_PROCESS_PER_COMPANY,
    GEMINI_API_KEY,
    GEMINI_MODEL_DEFAULT,
    OPENAI_MODEL_DEFAULT # Added for fallback
)

# CH API Interactions
from ch_api_utils import (
    get_ch_documents_metadata,
    _fetch_document_content_from_ch
)

# Text Extraction
# Ensure text_extraction_utils.py is present in your project directory
try:
    from text_extraction_utils import extract_text_from_document #, OCRHandlerType # OCRHandlerType removed from here
except ImportError:
    logger.error("text_extraction_utils.py not found. Text extraction will fail.")
    # Define a placeholder if not found to prevent outright crash at import time,
    # but it will fail at runtime.
    def extract_text_from_document(*args, **kwargs) -> Tuple[str, int, Optional[str]]:
        return "Error: text_extraction_utils.py not found.", 0, "text_extraction_utils.py is missing."
    # OCRHandlerType = Any # This was causing an issue, use Any directly in type hints if needed or specific Callable


# AI Summarization (these now return token counts)
from ai_utils import gpt_summarise_ch_docs, gemini_summarise_ch_docs

# Optional AWS Textract Import
try:
    from aws_textract_utils import perform_textract_ocr, get_textract_cost_estimation, _initialize_aws_clients as initialize_textract_aws_clients
    TEXTRACT_AVAILABLE = True
    logger.info("aws_textract_utils.py found and imported successfully.")
except ImportError:
    perform_textract_ocr = None
    get_textract_cost_estimation = None
    initialize_textract_aws_clients = None
    TEXTRACT_AVAILABLE = False
    logger.warning("aws_textract_utils.py not found. Textract OCR will not be available.")


def find_group_companies(parent_co_no: str) -> list[str]:
    logger.info(f"Group discovery for parent company {parent_co_no} currently returns only the parent. (Placeholder)")
    return [parent_co_no]


def _save_raw_document_content(
    doc_content: Union[bytes, str, Dict],
    doc_type_str: str,
    company_no: str,
    ch_doc_code: str,
    doc_year: int,
    scratch_dir: Path
) -> Optional[Path]:
    file_extension_map = {"pdf": "pdf", "xhtml": "xhtml", "json": "json"}
    file_extension = file_extension_map.get(doc_type_str, "dat")
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
    ch_api_key: str, 
    selected_categories: List[str],
    start_year: int,
    end_year: int,
    target_docs_per_category_in_date_range: int, 
    max_docs_to_scan_per_category: int, 
    scratch_dir: Path,
    filter_keywords: Optional[List[str]] = None,
    use_textract: bool = False
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], int, int, Optional[str]]: 
    logger.info(f"Starting document pipeline for company: {company_no} (Textract OCR: {'Enabled' if use_textract else 'Disabled'}, Keywords: {filter_keywords if filter_keywords else 'None'})")
    all_extracted_texts_data: List[Dict[str, Any]] = []
    total_pages_ocrd_for_company = 0
    pdfs_sent_to_ocr_for_company = 0
    company_profile_data: Optional[Dict[str, Any]] = None

    doc_metadata_items, company_profile_data, meta_error = get_ch_documents_metadata(
        company_no=company_no,
        api_key=ch_api_key,
        categories=selected_categories,
        items_per_page=100, 
        max_docs_to_fetch_meta=max_docs_to_scan_per_category,
        target_docs_per_category_in_date_range=target_docs_per_category_in_date_range,
        year_range=(start_year, end_year)
    )

    if meta_error:
        logger.error(f"Failed to get document metadata for {company_no}: {meta_error}")
        all_extracted_texts_data.append({
            "company_no": company_no, "ch_doc_type": "METADATA_ERROR", "year": 0,
            "source_format": "N/A", "text": "", "error": meta_error,
            "prompt_tokens": 0, "completion_tokens": 0
        })
        return all_extracted_texts_data, company_profile_data, 0, 0, meta_error

    if not doc_metadata_items:
        logger.warning(f"No document metadata found for {company_no} matching criteria.")
        return [], company_profile_data, 0, 0, "No document metadata found matching criteria."

    ocr_handler_to_use: Optional[Any] = None # Changed OCRHandlerType to Any
    if use_textract:
        if TEXTRACT_AVAILABLE and perform_textract_ocr and initialize_textract_aws_clients:
            if initialize_textract_aws_clients():
                ocr_handler_to_use = perform_textract_ocr
                logger.info(f"Textract OCR is enabled for PDFs for {company_no} if needed.")
            else:
                logger.warning(f"Textract OCR requested for {company_no}, but AWS clients failed to initialize. OCR skipped.")
        else:
            logger.warning(f"Textract OCR requested for {company_no}, but 'aws_textract_utils' unavailable/unconfigured. OCR skipped.")

    processed_doc_transaction_ids = set()
    docs_processed_count = 0
    doc_metadata_items.sort(key=lambda x: x.get("date", "0000-00-00"), reverse=True)

    for item_meta in doc_metadata_items:
        if docs_processed_count >= MAX_DOCS_TO_PROCESS_PER_COMPANY:
            logger.info(f"{company_no}: Reached document processing limit ({MAX_DOCS_TO_PROCESS_PER_COMPANY}).")
            break

        unique_doc_identifier = item_meta.get("transaction_id") or item_meta.get("links", {}).get("document_metadata")
        if not unique_doc_identifier or unique_doc_identifier in processed_doc_transaction_ids:
            if unique_doc_identifier: logger.debug(f"{company_no}: Skipping already processed doc (ID: {unique_doc_identifier})")
            continue

        ch_doc_type_code = item_meta.get("type", "UNKNOWN_CH_TYPE")
        doc_date_str = item_meta.get("date", "")
        try:
            doc_year_val = int(doc_date_str[:4])
        except ValueError:
            logger.warning(f"{company_no}: Could not parse year from '{doc_date_str}' for doc {unique_doc_identifier}. Skipping.")
            continue

        if not (start_year <= doc_year_val <= end_year):
            continue

        logger.info(f"{company_no}: Processing document: Type '{ch_doc_type_code}', Date '{doc_date_str}', Desc '{item_meta.get('description', 'N/A')}' (ID: {unique_doc_identifier})")
        doc_content_data, fetched_content_type, fetch_err = _fetch_document_content_from_ch(company_no, item_meta)

        current_doc_data = {
            "company_no": company_no, "ch_doc_type": ch_doc_type_code, "year": doc_year_val,
            "source_format": "FETCH_ERROR", "text": "", "error": None,
            "prompt_tokens": 0, "completion_tokens": 0 # Ensure these keys are present
        }

        if fetch_err or not doc_content_data or fetched_content_type == "none":
            err_msg = fetch_err or 'No content returned.'
            logger.warning(f"{company_no}: Failed to fetch content for doc {unique_doc_identifier}. Error: {err_msg}")
            current_doc_data["error"] = err_msg
        else:
            current_doc_data["source_format"] = fetched_content_type.upper()
            _save_raw_document_content(doc_content_data, fetched_content_type, company_no, ch_doc_type_code, doc_year_val, scratch_dir)

            extracted_text, pages_ocrd, extract_err = extract_text_from_document(
                doc_content_data, fetched_content_type, company_no,
                ocr_handler=ocr_handler_to_use if fetched_content_type == 'pdf' else None
            )
            current_doc_data["text"] = extracted_text
            current_doc_data["error"] = extract_err

            if pages_ocrd > 0:
                total_pages_ocrd_for_company += pages_ocrd
                pdfs_sent_to_ocr_for_company += 1

            if extract_err:
                logger.warning(f"{company_no}: Error extracting text from doc {unique_doc_identifier} (Type: {fetched_content_type}): {extract_err}")
            elif not extracted_text or len(extracted_text.strip()) < MIN_MEANINGFUL_TEXT_LEN / 2:
                logger.warning(f"{company_no}: No significant text from doc {unique_doc_identifier} (Type: {fetched_content_type}). Preview: '{extracted_text[:100]}...'")
            else:
                logger.info(f"{company_no}: Successfully extracted text from doc {unique_doc_identifier} (Type: {fetched_content_type}, Length: {len(extracted_text)} chars).")

                # --- Placeholder for Keyword Filtering Logic ---
                if filter_keywords and extracted_text:
                    text_lower = extracted_text.lower()
                    found_keywords = [kw for kw in filter_keywords if kw in text_lower]
                    if not found_keywords:
                        logger.info(f"{company_no}: Doc {unique_doc_identifier} - Text did not contain focus keywords. Not using for focused summary (full text still available if needed).")
                        # Potentially set a flag or modify 'extracted_text' for the aggregator
                        # For now, we'll just log. The aggregator will receive all text.
                        # A more advanced implementation might return only relevant excerpts.
                    else:
                        logger.info(f"{company_no}: Doc {unique_doc_identifier} - Text contains focus keywords: {found_keywords}.")
                # --- End Placeholder ---


        all_extracted_texts_data.append(current_doc_data)
        if unique_doc_identifier: processed_doc_transaction_ids.add(unique_doc_identifier)
        docs_processed_count += 1

    logger.info(f"Finished document pipeline for {company_no}. Extracted data for {len(all_extracted_texts_data)} items. OCR Pages: {total_pages_ocrd_for_company}, PDFs to OCR: {pdfs_sent_to_ocr_for_company}")
    return all_extracted_texts_data, company_profile_data, total_pages_ocrd_for_company, pdfs_sent_to_ocr_for_company, None


def _get_default_model_prices_gbp() -> Dict[str, float]:
    # Directly use imported model names from config
    return {
        OPENAI_MODEL_DEFAULT: 0.0004, 
        GEMINI_MODEL_DEFAULT: 0.0028, 
        # Add other models and their prices as needed
        "gpt-4o-mini": 0.00012, # Cost per 1k input tokens in GBP (example)
        "gpt-4-turbo": 0.008,   # Cost per 1k input tokens in GBP (example)
        "gpt-4": 0.024,         # Cost per 1k input tokens in GBP (example)
        "gemini-1.5-flash-latest": 0.00028, # Cost per 1k input tokens in GBP (example)
    }

def run_batch_company_analysis(
    csv_path: Path,
    ch_api_key_batch: str, 
    selected_categories: List[str],
    start_year: int,
    end_year: int,
    target_docs_per_category_in_date_range_batch: int, 
    max_docs_to_scan_per_category_batch: int, 
    model_prices_gbp: Dict[str, float],
    specific_ai_instructions: str = "",
    filter_keywords_str: Optional[str] = None, # New parameter from app.py
    base_scratch_dir: Optional[Path] = None,
    keep_days: int = 7,
    use_textract_ocr: bool = False
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    Main batch processing function. CH Summaries will prioritize Gemini.
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

    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8-sig') as fh_csv:
            reader = csv.reader(fh_csv)
            header = next(reader, None)
            if header: logger.info(f"Input CSV header: {header}")
            for i, row in enumerate(reader):
                if row and row[0].strip():
                    company_no_cleaned = row[0].strip().upper().replace(" ", "").zfill(8)
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
        logger.warning("No valid company numbers loaded. Batch processing cannot continue.")
        empty_digest_filename = f"digest_empty_input_{run_timestamp}.csv"
        empty_digest_path = run_scratch_dir / empty_digest_filename
        try: pd.DataFrame([]).to_csv(empty_digest_path, index=False, encoding='utf-8-sig')
        except Exception as e_write_empty: logger.error(f"Failed to write empty input digest: {e_write_empty}")
        return empty_digest_path, {
            "notes": "No companies processed due to empty or invalid input CSV.",
            "total_companies_processed": 0, "aws_ocr_costs": {}, "ai_ch_summary_costs": {} # Ensure keys exist
        }

    parsed_filter_keywords: Optional[List[str]] = None
    if filter_keywords_str:
        parsed_filter_keywords = [kw.strip().lower() for kw in filter_keywords_str.split(',') if kw.strip()]
        if not parsed_filter_keywords: # If string was just commas or spaces
            parsed_filter_keywords = None
    if parsed_filter_keywords:
        logger.info(f"Keyword filtering will be attempted with: {parsed_filter_keywords}")
    else:
        logger.info("No keyword filtering requested or keywords were empty.")


    batch_total_pages_ocrd = 0
    batch_total_pdfs_to_ocr = 0
    companies_successfully_summarized = 0
    companies_with_extraction_errors = 0
    total_ch_summary_prompt_tokens = 0
    total_ch_summary_completion_tokens = 0
    total_pipeline_errors = 0 # Initialize total_pipeline_errors

    ch_summary_model_to_use = GEMINI_MODEL_DEFAULT
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not found. Falling back to OpenAI default for CH summaries.")
        ch_summary_model_to_use = OPENAI_MODEL_DEFAULT
    logger.info(f"CH Document Summaries will use AI model: {ch_summary_model_to_use}")


    for company_idx, company_no in enumerate(parent_company_numbers):
        company_start_time = time.time()
        logger.info(f"Processing company: {company_no} ({company_idx + 1}/{len(parent_company_numbers)})")
        
        # Call the document pipeline
        extracted_docs_data, company_profile, ocr_pages, ocr_pdfs_count, pipeline_error_msg = run_ch_document_pipeline_for_company(
            company_no=company_no,
            ch_api_key=ch_api_key_batch, 
            selected_categories=selected_categories,
            start_year=start_year,
            end_year=end_year,
            target_docs_per_category_in_date_range=target_docs_per_category_in_date_range_batch, 
            max_docs_to_scan_per_category=max_docs_to_scan_per_category_batch, 
            scratch_dir=run_scratch_dir,
            filter_keywords=parsed_filter_keywords,
            use_textract=use_textract_ocr
        )
        
        current_company_total_cost_gbp = 0.0
        current_company_prompt_tokens = 0
        current_company_completion_tokens = 0
        # ... existing cost calculation and AI summarization logic ...

        if company_profile:
            logger.debug(f"Company profile data available for {company_no}: {company_profile.get('company_name', 'N/A')}")
        
        if pipeline_error_msg:
            logger.error(f"Pipeline error for {company_no}: {pipeline_error_msg}")
            output_data_rows.append({
                "Company Number": company_no,
                "Company Name": company_profile.get("company_name", "N/A") if company_profile else "N/A",
                "Status": "Pipeline Error",
                "Summary": pipeline_error_msg,
                "Document Category": "N/A",
                "Document Date": "N/A",
                "Document Type": "N/A",
                "Source Format": "N/A",
                "Text Length": 0,
                "AI Model Used": "N/A",
                "Prompt Tokens": 0,
                "Completion Tokens": 0,
                "Estimated Cost (GBP)": 0.0,
                "Keywords Found": "N/A",
                "File Path (if saved)": "N/A",
                "Error Message": pipeline_error_msg
            })
            total_pipeline_errors +=1
            continue # Move to the next company

        if not extracted_docs_data:
            logger.warning(f"No documents processed or extracted for {company_no}. Skipping summarization.")
            output_data_rows.append({
                "Company Number": company_no,
                "Company Name": company_profile.get("company_name", "N/A") if company_profile else "N/A",
                "Status": "No Documents Processed",
                "Summary": "No documents found or extracted matching criteria.",
                 # ... fill other fields as N/A or 0 ...
                "Document Category": "N/A",
                "Document Date": "N/A",
                "Document Type": "N/A",
                "Source Format": "N/A",
                "Text Length": 0,
                "AI Model Used": "N/A",
                "Prompt Tokens": 0,
                "Completion Tokens": 0,
                "Estimated Cost (GBP)": 0.0,
                "Keywords Found": "N/A",
                "File Path (if saved)": "N/A",
                "Error Message": "No documents processed"
            })
            continue
        # ... rest of the loop for summarization and output row creation ...

    ai_summary_cost_gbp = 0.0
    price_per_1k_tokens_ch_model = model_prices_gbp.get(ch_summary_model_to_use, 0.0)

    if price_per_1k_tokens_ch_model > 0:
        total_tokens_for_ch_summaries = total_ch_summary_prompt_tokens + total_ch_summary_completion_tokens
        ai_summary_cost_gbp = (total_tokens_for_ch_summaries / 1000) * price_per_1k_tokens_ch_model

    ai_summary_cost_metrics = {
        "model_used_for_ch_summaries": ch_summary_model_to_use,
        "total_prompt_tokens": total_ch_summary_prompt_tokens,
        "total_completion_tokens": total_ch_summary_completion_tokens,
        "estimated_cost_gbp": round(ai_summary_cost_gbp, 5)
    }

    aws_cost_metrics = {}
    if use_textract_ocr and TEXTRACT_AVAILABLE and get_textract_cost_estimation:
        aws_cost_metrics = get_textract_cost_estimation(batch_total_pages_ocrd, batch_total_pdfs_to_ocr)
        logger.info(f"Final AWS Cost Estimation for CH Batch OCR: {aws_cost_metrics}")
    elif use_textract_ocr:
        aws_cost_metrics = {"notes": "Textract OCR requested but not available/used. No OCR costs."}
    else:
        aws_cost_metrics = {"notes": "Textract OCR was not requested. No OCR costs."}

    batch_metrics = {
        "total_parent_companies_processed": len(parent_company_numbers),
        "companies_successfully_summarized": companies_successfully_summarized,
        "companies_with_extraction_errors_in_group": companies_with_extraction_errors,
        "total_pipeline_errors": total_pipeline_errors, # Add to batch_metrics
        "total_textract_pages_processed": batch_total_pages_ocrd,
        "total_pdfs_sent_to_textract": batch_total_pdfs_to_ocr,
        "aws_ocr_costs": aws_cost_metrics,
        "ai_ch_summary_costs": ai_summary_cost_metrics,
        "run_timestamp": run_timestamp,
        "keywords_applied_to_batch": ", ".join(parsed_filter_keywords) if parsed_filter_keywords else "N/A"
    }

    output_csv_path = None
    if output_data_rows:
        digest_filename = f"digest_CH_analysis_report_{run_timestamp}.csv"
        output_csv_path = run_scratch_dir / digest_filename
        try:
            df_output = pd.DataFrame(output_data_rows)
            output_cols = ["parent_company_no", "processed_group_members_log", "summary_of_findings",
                           "combined_text_char_count", "summary_prompt_tokens", "summary_completion_tokens",
                           "keywords_used_for_filtering"]
            df_output = df_output.reindex(columns=output_cols)
            df_output.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"Batch processing complete. Output digest: {output_csv_path}")
        except Exception as e_write_digest:
            logger.error(f"Failed to write final output digest CSV: {e_write_digest}", exc_info=True)
            output_csv_path = None
            batch_metrics["error"] = "Failed to write output digest CSV."
    elif parent_company_numbers: # Processed companies but no rows (e.g., all errors or no text)
        logger.warning("No data rows generated, though companies were processed. Creating empty digest with notes.")
        no_data_digest_filename = f"digest_processing_yielded_no_data_{run_timestamp}.csv"
        output_csv_path = run_scratch_dir / no_data_digest_filename
        try:
            pd.DataFrame([{"parent_company_no": "N/A",
                           "processed_group_members_log": "All processed companies yielded no usable data for summary or encountered critical errors.",
                           "summary_of_findings": "No summary generated.",
                           "combined_text_char_count": 0,
                           "summary_prompt_tokens":0, "summary_completion_tokens":0,
                           "keywords_used_for_filtering": batch_metrics["keywords_applied_to_batch"]
                           }]).to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        except Exception as e_write_no_data: logger.error(f"Failed to write 'no data' digest: {e_write_no_data}")
        batch_metrics["notes"] = (batch_metrics.get("notes","") + " Processing yielded no data rows.").strip()


    _cleanup_scratch_directory(run_scratch_dir, keep_days)
    return output_csv_path, batch_metrics


if __name__ == '__main__':
    # Setup for standalone test run
    import os # Ensure os is imported for getenv
    logger.info("Running ch_pipeline.py as standalone test.")
    
    # Ensure required environment variables are available for testing
    required_env_vars_for_test = ['OPENAI_API_KEY', 'CH_API_KEY'] # GEMINI_API_KEY check is intrinsic
    if not all(os.getenv(var) for var in required_env_vars_for_test):
        logger.critical(f"Standalone test requires API keys (OPENAI_API_KEY, CH_API_KEY). Some tests may fail or be skipped if not set.")

    test_model_prices = {
        config.OPENAI_MODEL_DEFAULT: 0.0004, # Example: gpt-3.5-turbo if it's the default
        "gpt-4o-mini": 0.00012,
        config.GEMINI_MODEL_DEFAULT: 0.0028,
        "gemini-1.5-flash-latest": 0.00028
    }

    test_csv_dir = Path(tempfile.mkdtemp(prefix="ch_pipeline_test_input_dir_"))
    standalone_test_csv_path = test_csv_dir / "test_ch_pipeline_input.csv"
    # Use an environment variable for the test company number, or a common default
    test_company_no = os.getenv("TEST_COMPANY_NUMBER_CH", "00445790") # Example: ROLLS-ROYCE PLC

    with open(standalone_test_csv_path, "w", newline="", encoding="utf-8") as f_test_csv:
        csv_writer = csv.writer(f_test_csv)
        csv_writer.writerow(["CompanyNumber"])
        csv_writer.writerow([test_company_no])
    logger.info(f"Created test input CSV: {standalone_test_csv_path} for company {test_company_no}")

    test_scratch_base = Path(tempfile.mkdtemp(prefix="ch_pipeline_test_scratch_base_"))
    logger.info(f"Using temporary base scratch for standalone test runs: {test_scratch_base}")

    test_categories = ['accounts'] # Focus on accounts for potentially larger text
    current_test_year = datetime.datetime.now().year
    # Look back further for more chance of documents for a test company
    test_start_year, test_end_year = current_test_year - 5, current_test_year - 1
    test_instructions = "Summarize key financial trends and any mention of strategic direction."
    test_keywords = "profit, revenue, strategy"

    logger.info(f"\n--- Test Run: CH Summaries (Default AI), Textract Disabled, No Keywords ---")
    try:
        result_path_gemini, metrics_gemini = run_batch_company_analysis(
            csv_path=standalone_test_csv_path, ch_api_key_batch=os.getenv("CH_API_KEY"), selected_categories=test_categories,
            start_year=test_start_year, end_year=test_end_year, target_docs_per_category_in_date_range_batch=5, max_docs_to_scan_per_category_batch=50,
            model_prices_gbp=test_model_prices,
            specific_ai_instructions=test_instructions,
            filter_keywords_str=None, # Test without keywords first
            base_scratch_dir=test_scratch_base,
            keep_days=0, use_textract_ocr=False
        )
        if result_path_gemini: logger.info(f"CH (Default AI) Test Output Digest: {result_path_gemini}")
        else: logger.error("CH (Default AI) Test Run DID NOT produce an output digest path.")
        logger.info(f"CH (Default AI) Test Metrics: {json.dumps(metrics_gemini, indent=2)}")
    except Exception as e_test_default: logger.error(f"Error during CH (Default AI) test: {e_test_default}", exc_info=True)

    logger.info(f"\n--- Test Run: CH Summaries (Default AI), Textract Disabled, WITH Keywords: '{test_keywords}' ---")
    try:
        result_path_keywords, metrics_keywords = run_batch_company_analysis(
            csv_path=standalone_test_csv_path, ch_api_key_batch=os.getenv("CH_API_KEY"), selected_categories=test_categories,
            start_year=test_start_year, end_year=test_end_year, target_docs_per_category_in_date_range_batch=5, max_docs_to_scan_per_category_batch=50,
            model_prices_gbp=test_model_prices,
            specific_ai_instructions=test_instructions,
            filter_keywords_str=test_keywords, # Test with keywords
            base_scratch_dir=test_scratch_base,
            keep_days=0, use_textract_ocr=False
        )
        if result_path_keywords: logger.info(f"CH (Keywords) Test Output Digest: {result_path_keywords}")
        else: logger.error("CH (Keywords) Test Run DID NOT produce an output digest path.")
        logger.info(f"CH (Keywords) Test Metrics: {json.dumps(metrics_keywords, indent=2)}")
    except Exception as e_test_keywords: logger.error(f"Error during CH (Keywords) test: {e_test_keywords}", exc_info=True)


    if TEXTRACT_AVAILABLE and os.getenv("S3_TEXTRACT_BUCKET") and os.getenv("AWS_ACCESS_KEY_ID"): # Add key check for Textract
        logger.info(f"\n--- Test Run: CH Summaries (Default AI), Textract Enabled, No Keywords ---")
        try:
            result_path_ocr, metrics_ocr = run_batch_company_analysis(
                csv_path=standalone_test_csv_path, ch_api_key_batch=os.getenv("CH_API_KEY"), selected_categories=test_categories,
                start_year=test_start_year, end_year=test_end_year, target_docs_per_category_in_date_range_batch=5, max_docs_to_scan_per_category_batch=50,
                model_prices_gbp=test_model_prices,
                specific_ai_instructions=test_instructions,
                filter_keywords_str=None,
                base_scratch_dir=test_scratch_base,
                keep_days=0, use_textract_ocr=True
            )
            if result_path_ocr: logger.info(f"CH (OCR) Test Output Digest: {result_path_ocr}")
            else: logger.error("CH (OCR) Test Run DID NOT produce an output digest path.")
            logger.info(f"CH (OCR) Test Metrics: {json.dumps(metrics_ocr, indent=2)}")
        except Exception as e_test_ocr: logger.error(f"Error during CH (OCR) test: {e_test_ocr}", exc_info=True)
    else:
        logger.warning("\n--- Skipping Textract enabled test: TEXTRACT_AVAILABLE is False or S3_TEXTRACT_BUCKET/AWS Keys not set. ---")

    try:
        if standalone_test_csv_path.exists(): standalone_test_csv_path.unlink()
        if test_csv_dir.exists(): test_csv_dir.rmdir() # Remove the directory itself
        # Deleting the base scratch dir and its contents needs more care if runs are parallel
        # For simple test, can try to remove it. For production, keep_days should handle it.
        # import shutil
        # if test_scratch_base.exists(): shutil.rmtree(test_scratch_base)
        logger.info(f"Test cleanup: Removed {standalone_test_csv_path.name} and {test_csv_dir.name}. Base scratch path for manual review if needed: {test_scratch_base}.")
    except Exception as e_cleanup_main: logger.warning(f"Error during main test cleanup: {e_cleanup_main}")