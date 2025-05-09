# ch_api_utils.py

import logging
import time
import json
from typing import List, Tuple, Dict, Any, Optional, Union

import requests

from config import (
    get_ch_session,
    CH_API_BASE_URL,
    CH_DOCUMENT_API_BASE_URL,
    logger
)

def _fetch_document_content_from_ch(
    company_no: str,
    item_details: Dict[str, Any]
) -> Tuple[Optional[Union[bytes, str, Dict]], str, Optional[str]]:
    """
    Fetches the content of a single document from Companies House.
    Tries JSON, then XHTML, then PDF.

    Returns:
        A tuple: (content_data, content_type_fetched, error_message_or_None).
        content_type_fetched can be "json", "xhtml", "pdf", or "none".
    """
    ch_session = get_ch_session()
    doc_meta_link = item_details.get("links", {}).get("document_metadata", "")
    if not doc_meta_link:
        err_msg = f"No document_metadata link for item: {item_details.get('transaction_id', 'N/A')}"
        logger.warning(f"{company_no}: {err_msg}")
        return None, "none", err_msg
    
    # Ensure doc_meta_link is absolute
    if doc_meta_link.startswith("/document/"):
        doc_id = doc_meta_link.split("/")[-1]
    elif "/document/" in doc_meta_link: # Check if it's already a full document API link path
        doc_id = doc_meta_link.split("/document/")[-1].split("/")[0] # Get the ID part
    else: # Fallback assuming the last part is the ID if structure is unexpected
        doc_id = doc_meta_link.split("/")[-1]
        logger.warning(f"{company_no}: Document metadata link '{doc_meta_link}' has unexpected format. Using last part as ID: '{doc_id}'.")


    content_url = f"{CH_DOCUMENT_API_BASE_URL}/document/{doc_id}/content"
    doc_ch_type = item_details.get("type", "").upper() # e.g., "AA", "CS01"
    doc_description = item_details.get('description', 'N/A')
    
    request_delay = 0.35 # Seconds between attempts for different content types

    # 1. Attempt to fetch JSON (primarily for iXBRL Annual Accounts)
    # Only certain document types are likely to have JSON (e.g., modern accounts)
    if doc_ch_type in ["AA", "ACCOUNTS TYPE FULL", "ACCOUNTS TYPE MEDIUM", "ACCOUNTS TYPE SMALL", "ACCOUNTS TYPE MICROENTITY", "ACCOUNTS TYPE GROUP", "ACCOUNTS TYPE INTERIM", "ACCOUNTS TYPE INITIAL", "ACCOUNTS TYPE DORMANT"]: # Add more if known
        logger.debug(f"{company_no}: Attempting JSON for doc {doc_id} (Type: {doc_ch_type}, Desc: {doc_description})")
        try:
            time.sleep(request_delay)
            resp = ch_session.get(content_url, headers={"Accept": "application/json"}, timeout=45)
            resp.raise_for_status()
            if "application/json" in resp.headers.get("Content-Type", "").lower():
                logger.info(f"{company_no}: Successfully fetched JSON for doc {doc_id}.")
                return resp.json(), "json", None
            else:
                logger.debug(f"{company_no}: JSON request for doc {doc_id} got Content-Type: {resp.headers.get('Content-Type', '')}. Text preview: {resp.text[:200]}")
        except requests.exceptions.HTTPError as e_http:
            if e_http.response.status_code in [404, 406, 503]: # 406 Not Acceptable is common if JSON isn't available
                logger.debug(f"{company_no}: JSON not available (HTTP {e_http.response.status_code}) for doc {doc_id}.")
            else:
                logger.warning(f"{company_no}: JSON fetch HTTP error for doc {doc_id}: {e_http}.")
        except json.JSONDecodeError as e_json_decode:
            logger.error(f"{company_no}: Failed to decode JSON for doc {doc_id} (Desc: {doc_description}): {e_json_decode}. Response text: {getattr(resp, 'text', 'N/A')[:200]}")
        except requests.exceptions.RequestException as e_req:
            logger.warning(f"{company_no}: JSON fetch RequestException for doc {doc_id}: {e_req}.")
        # No need for generic Exception catch here if specific ones cover requests/JSON issues

    # 2. Attempt to fetch XHTML (common for accounts)
    # Accounts, insolvency docs might have XHTML
    if doc_ch_type in ["AA", "CONLIQ", "AMENDED ACCS", "ACCOUNTS TYPE FULL", "ACCOUNTS TYPE MEDIUM", "ACCOUNTS TYPE SMALL", "ACCOUNTS TYPE MICROENTITY", "ACCOUNTS TYPE GROUP", "ACCOUNTS TYPE INTERIM", "ACCOUNTS TYPE INITIAL", "ACCOUNTS TYPE DORMANT", "LIQ10", "LIQ13", "LIQ14", "WURESOLUTION"]:
        logger.debug(f"{company_no}: Attempting XHTML for doc {doc_id} (Type: {doc_ch_type}, Desc: {doc_description})")
        try:
            time.sleep(request_delay)
            resp = ch_session.get(content_url, headers={"Accept": "application/xhtml+xml"}, timeout=45)
            resp.raise_for_status()
            if "application/xhtml+xml" in resp.headers.get("Content-Type", "").lower():
                logger.info(f"{company_no}: Successfully fetched XHTML for doc {doc_id}.")
                return resp.text, "xhtml", None
            else:
                logger.debug(f"{company_no}: XHTML request for doc {doc_id} got Content-Type: {resp.headers.get('Content-Type', '')}. Text preview: {resp.text[:200]}")
        except requests.exceptions.HTTPError as e_http:
            if e_http.response.status_code in [404, 406, 503]:
                logger.debug(f"{company_no}: XHTML not available (HTTP {e_http.response.status_code}) for doc {doc_id}.")
            else:
                logger.warning(f"{company_no}: XHTML fetch HTTP error for doc {doc_id}: {e_http}.")
        except requests.exceptions.RequestException as e_req:
            logger.warning(f"{company_no}: XHTML fetch RequestException for doc {doc_id}: {e_req}.")

    # 3. Attempt to fetch PDF (fallback for most other documents)
    logger.debug(f"{company_no}: Attempting PDF for doc {doc_id} (Type: {doc_ch_type}, Desc: {doc_description})")
    try:
        time.sleep(request_delay)
        # Longer timeout for PDFs as they can be larger
        resp = ch_session.get(content_url, headers={"Accept": "application/pdf"}, timeout=120)
        resp.raise_for_status()
        if "application/pdf" in resp.headers.get("Content-Type", "").lower():
            logger.info(f"{company_no}: Successfully fetched PDF for doc {doc_id} (size: {len(resp.content)} bytes).")
            return resp.content, "pdf", None
        else:
            # This case might indicate an issue with CH API or an unexpected response type
            content_sample = resp.content[:200] if isinstance(resp.content, bytes) else resp.text[:200]
            err_msg = f"PDF request for doc {doc_id} got Content-Type {resp.headers.get('Content-Type', '')} instead of PDF. Sample: {content_sample}"
            logger.warning(f"{company_no}: {err_msg}")
            return None, "none", err_msg
    except requests.exceptions.RequestException as e_req:
        err_msg = f"PDF fetch failed for doc {doc_id}: {e_req}"
        logger.error(f"{company_no}: {err_msg}")
        return None, "none", err_msg
    except Exception as e_pdf_other: # Catch any other unexpected error during PDF fetch
        err_msg = f"Unexpected error fetching PDF for doc {doc_id}: {e_pdf_other}"
        logger.error(f"{company_no}: {err_msg}", exc_info=True)
        return None, "none", err_msg

    final_err_msg = f"All attempts to fetch content for doc {doc_id} (Type: {doc_ch_type}, Desc: {doc_description}) failed."
    logger.warning(f"{company_no}: {final_err_msg}")
    return None, "none", final_err_msg


def get_ch_documents_metadata(
    company_no: str,
    categories: List[str],
    start_year: int,
    end_year: int,
    max_docs_to_fetch_meta: int = 100 # Max metadata items to pull per category initially
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Fetches filing history metadata from Companies House for specified categories and year range.

    Returns:
        A tuple: (list_of_document_metadata_items, error_message_or_None).
    """
    ch_session = get_ch_session()
    logger.info(f"Fetching CH document metadata for {company_no} | Categories: {categories} | Years: {start_year}-{end_year}")

    if not categories:
        logger.warning(f"No categories specified for {company_no}. Skipping metadata fetch.")
        return [], "No categories specified."

    # Get the filing history link first
    company_profile_url = f"{CH_API_BASE_URL}/company/{company_no}"
    try:
        time.sleep(0.25) # Rate limit
        company_resp = ch_session.get(company_profile_url, timeout=30)
        company_resp.raise_for_status()
        company_data = company_resp.json()
        filings_url_path = company_data.get("links", {}).get("filing_history")
        if not filings_url_path:
            err_msg = f"No 'filing_history' link found for company {company_no}."
            logger.warning(f"{company_no}: {err_msg}")
            return [], err_msg
        
        # Ensure filing_url_path is absolute
        full_filings_url = filings_url_path if filings_url_path.startswith("http") else f"{CH_API_BASE_URL}{filings_url_path}"

    except requests.exceptions.RequestException as e_req_profile:
        err_msg = f"API request failed for company profile {company_no}: {e_req_profile}"
        logger.error(f"{company_no}: {err_msg}")
        return [], err_msg
    except json.JSONDecodeError as e_json_profile:
        err_msg = f"Failed to decode JSON for company profile {company_no}: {e_json_profile}"
        logger.error(f"{company_no}: {err_msg}")
        return [], err_msg

    all_items_from_api: List[Dict[str, Any]] = []
    processed_categories = set()
    total_api_calls = 0

    for category in categories:
        cat_lower = category.lower().strip()
        if not cat_lower or cat_lower in processed_categories:
            continue
        processed_categories.add(cat_lower)
        logger.debug(f"{company_no}: Fetching metadata for category '{cat_lower}'...")

        # CH API has a max of 100 items_per_page.
        # We might need pagination if a category has > 100 filings,
        # but usually we're interested in recent ones.
        # Let's fetch a reasonable number and then filter by year locally.
        # A `start_index` can be used for pagination if needed.
        # For now, fetch up to `max_docs_to_fetch_meta` per category.
        
        items_per_page_api = 100 # CH API max
        start_index = 0
        current_category_items = []
        
        # Make up to a few calls per category if needed to reach max_docs_to_fetch_meta
        # (e.g., if max_docs_to_fetch_meta = 200, this would be 2 calls)
        # More robust pagination would check 'total_count' from API response.
        max_api_calls_per_category = (max_docs_to_fetch_meta + items_per_page_api -1) // items_per_page_api

        for _ in range(max_api_calls_per_category):
            if len(current_category_items) >= max_docs_to_fetch_meta:
                break

            params = {"category": cat_lower, "items_per_page": items_per_page_api, "start_index": start_index}
            try:
                time.sleep(0.25) # Rate limit before each API call
                total_api_calls +=1
                filings_resp = ch_session.get(full_filings_url, params=params, timeout=30)
                
                if filings_resp.status_code == 404: # Category might not exist or no filings
                    logger.debug(f"{company_no}: Category '{cat_lower}' not found or no filings (404) at start_index {start_index}.")
                    break 
                filings_resp.raise_for_status() # For other HTTP errors
                
                filings_data = filings_resp.json()
                page_items_list = filings_data.get("items", [])
                if not page_items_list: # No more items for this category
                    break
                
                current_category_items.extend(page_items_list)
                start_index += len(page_items_list)

                # If total_count is available and we've fetched them all for this category
                if filings_data.get("total_count", 0) > 0 and start_index >= filings_data.get("total_count", 0):
                    break

            except requests.exceptions.RequestException as e_req_cat:
                err_msg = f"API request failed for category '{cat_lower}' (company {company_no}): {e_req_cat}"
                logger.error(f"{company_no}: {err_msg}")
                # Potentially break or return partial results with error
                return all_items_from_api, err_msg # Return what we have so far with error
            except json.JSONDecodeError as e_json_decode_filings:
                err_msg = f"Failed to decode JSON from filing history for '{cat_lower}' (company {company_no}). Error: {e_json_decode_filings}. Response text: {filings_resp.text[:200]}"
                logger.error(f"{company_no}: {err_msg}")
                return all_items_from_api, err_msg # Return what we have so far with error
        
        all_items_from_api.extend(current_category_items)
        logger.info(f"{company_no}: Fetched {len(current_category_items)} metadata items for category '{cat_lower}'.")

    logger.info(f"{company_no}: Total {len(all_items_from_api)} metadata items fetched across all specified categories (made {total_api_calls} API calls for filings).")
    
    # Filter by year and sort
    filtered_items: List[Dict[str, Any]] = []
    for item in all_items_from_api:
        date_str = item.get("date", "")
        if not date_str:
            continue
        try:
            year_val = int(date_str[:4])
            if start_year <= year_val <= end_year:
                filtered_items.append(item)
        except ValueError:
            logger.warning(f"{company_no}: Could not parse year from date '{date_str}' for doc item {item.get('transaction_id','N/A')}.")
            
    # Sort by date descending (most recent first)
    filtered_items.sort(key=lambda x: x.get("date", "0000-00-00"), reverse=True)
    
    logger.info(f"{company_no}: Returning {len(filtered_items)} metadata items after year filtering ({start_year}-{end_year}).")
    return filtered_items, None