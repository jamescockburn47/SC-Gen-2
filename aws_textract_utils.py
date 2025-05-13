# aws_textract_utils.py

import logging
import time
import uuid
from typing import Tuple, Optional

import boto3
from botocore.exceptions import ClientError as BotoClientError, NoCredentialsError, PartialCredentialsError

# Import AWS configuration from config.py
from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION_DEFAULT,
    S3_TEXTRACT_BUCKET,
    AWS_PRICING_CONFIG, # For cost estimation if needed here, or pass costs back
    logger # Use the centrally configured logger
)

_textract_client = None
_s3_client = None
_aws_clients_initialized = False

def _initialize_aws_clients():
    global _textract_client, _s3_client, _aws_clients_initialized
    if _aws_clients_initialized:
        return True

    if not AWS_REGION_DEFAULT:
        logger.error("AWS_DEFAULT_REGION not set. Cannot initialize AWS Textract/S3 clients.")
        return False
    if not S3_TEXTRACT_BUCKET:
        logger.error("S3_TEXTRACT_BUCKET not set. Cannot initialize S3 client for Textract.")
        return False

    try:
        session_params = {"region_name": AWS_REGION_DEFAULT}
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY: # Only use explicit creds if both are provided
            session_params["aws_access_key_id"] = AWS_ACCESS_KEY_ID
            session_params["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
        
        boto_session = boto3.Session(**session_params)
        
        _textract_client = boto_session.client("textract")
        _s3_client = boto_session.client("s3")
        
        # Perform a simple test to check credentials and connectivity (optional)
        # For S3: _s3_client.list_buckets() 
        # For Textract: (no simple lightweight test, usually just proceed)
        logger.info(f"AWS Textract and S3 clients initialized for region {AWS_REGION_DEFAULT} and bucket {S3_TEXTRACT_BUCKET}.")
        _aws_clients_initialized = True
        return True
    except (NoCredentialsError, PartialCredentialsError) as e_creds:
        logger.error(f"AWS credentials not found or incomplete: {e_creds}. Textract functionality will be unavailable.")
    except BotoClientError as e_boto_init:
        logger.error(f"BotoClientError initializing AWS clients: {e_boto_init}. Textract functionality may be unavailable.")
    except Exception as e_aws_init:
        logger.error(f"Failed to initialize AWS clients: {e_aws_init}. Textract functionality may be unavailable.", exc_info=True)
    
    _aws_clients_initialized = False
    return False


def perform_textract_ocr(
    pdf_bytes: bytes,
    company_no_for_logging: str = "N/A_Textract"
) -> Tuple[str, int, Optional[str]]:
    """
    Performs OCR on PDF bytes using AWS Textract via S3.
    Returns (extracted_text, pages_processed, error_message).
    Error_message is None if successful.
    """
    if not _initialize_aws_clients() or not _textract_client or not _s3_client:
        err_msg = "AWS Textract/S3 clients not available or not initialized. OCR skipped."
        logger.warning(f"{company_no_for_logging}: {err_msg}")
        return "", 0, err_msg

    logger.info(f"{company_no_for_logging}: AWS Textract: Initiating OCR (Bucket: {S3_TEXTRACT_BUCKET}).")
    extracted_text = ""
    textract_pages_processed = 0
    s3_object_key = f"temp-textract-inputs/{company_no_for_logging.replace('/', '_')}_{uuid.uuid4()}.pdf"
    textract_job_id = None
    upload_successful = False
    error_occurred_msg = None

    try:
        logger.debug(f"{company_no_for_logging}: AWS Textract: Uploading PDF ({len(pdf_bytes)} bytes) to S3: s3://{S3_TEXTRACT_BUCKET}/{s3_object_key}")
        _s3_client.put_object(Bucket=S3_TEXTRACT_BUCKET, Key=s3_object_key, Body=pdf_bytes)
        upload_successful = True
        logger.info(f"{company_no_for_logging}: AWS Textract: PDF uploaded to S3.")

        logger.debug(f"{company_no_for_logging}: AWS Textract: Starting job for S3 object: {s3_object_key}")
        start_response = _textract_client.start_document_text_detection(
            DocumentLocation={'S3Object': {'Bucket': S3_TEXTRACT_BUCKET, 'Name': s3_object_key}}
        )
        textract_job_id = start_response['JobId']
        logger.info(f"{company_no_for_logging}: AWS Textract: Job started: {textract_job_id}")

        textract_text_parts = []
        next_token_for_textract_results = None
        job_final_status = "UNKNOWN"
        
        max_poll_attempts_config = 240
        polling_interval_seconds_config = 1.5 # Increased slightly
        logger.info(f"{company_no_for_logging}: AWS Textract: Polling job {textract_job_id} up to {max_poll_attempts_config} times...")

        for poll_attempt in range(max_poll_attempts_config):
            time.sleep(polling_interval_seconds_config)
            get_params_textract = {'JobId': textract_job_id}
            if next_token_for_textract_results:
                get_params_textract['NextToken'] = next_token_for_textract_results
            
            try:
                get_response_textract = _textract_client.get_document_text_detection(**get_params_textract)
                job_status_textract = get_response_textract['JobStatus']
            except BotoClientError as e_get_boto:
                logger.warning(f"{company_no_for_logging}: Textract BotoClientError during poll (Attempt {poll_attempt + 1}): {e_get_boto}. Retrying.")
                if 'AccessDeniedException' in str(e_get_boto) or 'InvalidS3ObjectException' in str(e_get_boto) : # Non-retryable
                    error_occurred_msg = f"Textract job failed due to S3 access/object issue: {e_get_boto}"
                    logger.error(f"{company_no_for_logging}: {error_occurred_msg}")
                    job_final_status = "POLLING_S3_ERROR"
                    break
                continue
            except Exception as e_poll_gen:
                error_occurred_msg = f"Textract generic error during poll (Attempt {poll_attempt + 1}): {e_poll_gen}"
                logger.error(f"{company_no_for_logging}: {error_occurred_msg}", exc_info=True)
                job_final_status = "POLLING_EXCEPTION"
                break

            if job_status_textract == 'SUCCEEDED':
                blocks_textract = get_response_textract.get('Blocks', [])
                for block in blocks_textract:
                    if block.get('BlockType') == 'LINE':
                        textract_text_parts.append(block.get('Text', ''))
                
                if get_response_textract.get('DocumentMetadata') and isinstance(get_response_textract['DocumentMetadata'].get('Pages'), int):
                    # Accumulate pages if job is paginated (though text detection usually gives all pages at once if successful)
                    textract_pages_processed = max(textract_pages_processed, get_response_textract['DocumentMetadata']['Pages']) 
                
                next_token_for_textract_results = get_response_textract.get('NextToken')
                if not next_token_for_textract_results:
                    extracted_text = "\n".join(filter(None, textract_text_parts)).strip()
                    job_final_status = 'SUCCEEDED'
                    logger.info(f"{company_no_for_logging}: Textract: Job {textract_job_id} SUCCEEDED. Extracted {len(extracted_text)} chars. Processed {textract_pages_processed} pages.")
                    break
                else:
                    logger.debug(f"{company_no_for_logging}: Textract: Job {textract_job_id} SUCCEEDED (partial), fetching next page...")
            elif job_status_textract == 'FAILED':
                job_final_status = 'FAILED'
                status_msg_textract = get_response_textract.get('StatusMessage', 'No error message from Textract.')
                error_occurred_msg = f"Textract: Job {textract_job_id} FAILED. StatusMessage: {status_msg_textract}"
                logger.error(f"{company_no_for_logging}: {error_occurred_msg}")
                break
            elif job_status_textract == 'IN_PROGRESS':
                if (poll_attempt + 1) % 30 == 0:
                    logger.debug(f"{company_no_for_logging}: Textract: Job {textract_job_id} IN_PROGRESS (Poll {poll_attempt + 1}/{max_poll_attempts_config})...")
            else:
                job_final_status = job_status_textract
                error_occurred_msg = f"Textract: Job {textract_job_id} unhandled status: {job_status_textract}."
                logger.warning(f"{company_no_for_logging}: {error_occurred_msg} Response: {get_response_textract}")
                break
        else: # Loop finished without break (timeout)
            error_occurred_msg = f"Textract: Job {textract_job_id} timed out polling after {max_poll_attempts_config} attempts."
            logger.warning(f"{company_no_for_logging}: {error_occurred_msg}")
            job_final_status = 'TIMEOUT_POLLING'

    except (NoCredentialsError, PartialCredentialsError) as e_creds_runtime:
        error_occurred_msg = f"AWS credentials error during Textract operation: {e_creds_runtime}"
        logger.error(f"{company_no_for_logging}: {error_occurred_msg}", exc_info=True)
    except BotoClientError as e_aws_boto_outer:
        error_code_aws = e_aws_boto_outer.response.get('Error', {}).get('Code', 'UnknownAWSBotoClientError')
        error_message_aws = e_aws_boto_outer.response.get('Error', {}).get('Message', str(e_aws_boto_outer))
        error_occurred_msg = f"Textract BotoClientError (S3/Textract): {error_code_aws} - {error_message_aws} (Job ID: {textract_job_id})"
        logger.error(f"{company_no_for_logging}: {error_occurred_msg}", exc_info=True)
        if error_code_aws in ('AccessDenied', 'UnauthorizedOperation', 'AccessDeniedException'):
            logger.error(f"CRITICAL CHECK: AWS Permissions for S3 bucket '{S3_TEXTRACT_BUCKET}' (GetObject, PutObject, DeleteObject) and Textract (StartDocumentTextDetection, GetDocumentTextDetection) might be insufficient or misconfigured!")
    except Exception as e_generic_outer_textract:
        error_occurred_msg = f"Textract: Unexpected generic error: {e_generic_outer_textract}"
        logger.error(f"{company_no_for_logging}: {error_occurred_msg}", exc_info=True)
    finally:
        if upload_successful and S3_TEXTRACT_BUCKET and _s3_client:
            try:
                logger.debug(f"{company_no_for_logging}: Textract: Deleting temporary S3 object: s3://{S3_TEXTRACT_BUCKET}/{s3_object_key}")
                _s3_client.delete_object(Bucket=S3_TEXTRACT_BUCKET, Key=s3_object_key)
                logger.info(f"{company_no_for_logging}: Textract: Temp S3 object {s3_object_key} deleted.")
            except Exception as e_del_s3_final:
                logger.error(f"{company_no_for_logging}: Textract: FAILED to delete S3 object {s3_object_key}: {e_del_s3_final}. Manual cleanup may be required in '{S3_TEXTRACT_BUCKET}'.")

    if not extracted_text and not error_occurred_msg: # No text but no explicit error during job
        if job_final_status != 'SUCCEEDED':
            error_occurred_msg = f"Textract: OCR yielded no text. Final job status for {textract_job_id} was '{job_final_status}'."
            logger.warning(f"{company_no_for_logging}: {error_occurred_msg}")
        else: # Succeeded but no text (e.g. blank image PDF)
            logger.info(f"{company_no_for_logging}: Textract: OCR job {textract_job_id} SUCCEEDED but returned no text (possibly a blank or image-only PDF with no detectable text).")
            # Not strictly an error, could be a valid outcome.

    return extracted_text, textract_pages_processed, error_occurred_msg

def get_textract_cost_estimation(pages_processed: int, num_pdfs_processed: int) -> dict:
    """Estimates AWS Textract and related S3 costs."""
    if not _aws_clients_initialized: # Or based on actual usage flags
        return {
            "notes": "AWS clients not initialized, or Textract was not used. Cost estimation skipped.",
            "total_estimated_aws_cost_usd_for_ocr": 0.0,
            "total_estimated_aws_cost_gbp_for_ocr": 0.0
        }

    s3_put_cost_usd = num_pdfs_processed * AWS_PRICING_CONFIG["s3_put_request_per_pdf_to_textract"]
    textract_page_processing_cost_usd = pages_processed * AWS_PRICING_CONFIG["textract_per_page"]
    
    total_aws_cost_usd = s3_put_cost_usd + textract_page_processing_cost_usd
    total_aws_cost_gbp = total_aws_cost_usd * AWS_PRICING_CONFIG["usd_to_gbp_exchange_rate"]

    return {
        "num_pdfs_sent_to_textract_s3": num_pdfs_processed,
        "num_pages_processed_by_textract": pages_processed,
        "estimated_s3_put_cost_usd_for_textract_pdfs": round(s3_put_cost_usd, 5),
        "estimated_textract_page_processing_cost_usd": round(textract_page_processing_cost_usd, 5),
        "total_estimated_aws_cost_usd_for_ocr": round(total_aws_cost_usd, 5),
        "total_estimated_aws_cost_gbp_for_ocr": round(total_aws_cost_gbp, 5),
        "notes": "AWS costs cover Textract page processing and S3 PUTs for PDFs sent to Textract. Other S3 costs (storage, GETs) not included."
    }