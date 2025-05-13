# ch_api_utils.py

import logging
import time
import json
from typing import List, Tuple, Dict, Any, Optional, Union
from datetime import datetime # Added import

import requests

from config import (
    get_ch_session,
    CH_API_BASE_URL,
    CH_DOCUMENT_API_BASE_URL,
    logger
)

# Cache for company profiles to reduce redundant API calls within a run
_company_profile_cache: Dict[str, Dict[str, Any]] = {}

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
    elif "/document/" in doc_meta_link: 
        doc_id = doc_meta_link.split("/document/")[-1].split("/")[0]
    else: 
        doc_id = doc_meta_link.split("/")[-1]
        logger.warning(f"{company_no}: Document metadata link '{doc_meta_link}' has unexpected format. Using last part as ID: '{doc_id}'.")

    content_url = f"{CH_DOCUMENT_API_BASE_URL}/document/{doc_id}/content"
    doc_ch_type = item_details.get("type", "").upper() 
    doc_description = item_details.get('description', 'N/A')
    
    request_delay = 0.35 

    # 1. Attempt to fetch JSON 
    if doc_ch_type in ["AA", "ACCOUNTS TYPE FULL", "ACCOUNTS TYPE MEDIUM", "ACCOUNTS TYPE SMALL", "ACCOUNTS TYPE MICROENTITY", "ACCOUNTS TYPE GROUP", "ACCOUNTS TYPE INTERIM", "ACCOUNTS TYPE INITIAL", "ACCOUNTS TYPE DORMANT"]:
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
            if e_http.response.status_code in [404, 406, 503]: 
                logger.debug(f"{company_no}: JSON not available (HTTP {e_http.response.status_code}) for doc {doc_id}.")
            else:
                logger.warning(f"{company_no}: JSON fetch HTTP error for doc {doc_id}: {e_http}.")
        except json.JSONDecodeError as e_json_decode:
            logger.error(f"{company_no}: Failed to decode JSON for doc {doc_id} (Desc: {doc_description}): {e_json_decode}. Response text: {getattr(resp, 'text', 'N/A')[:200]}")
        except requests.exceptions.RequestException as e_req:
            logger.warning(f"{company_no}: JSON fetch RequestException for doc {doc_id}: {e_req}.")

    # 2. Attempt to fetch XHTML
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

    # 3. Attempt to fetch PDF
    logger.debug(f"{company_no}: Attempting PDF for doc {doc_id} (Type: {doc_ch_type}, Desc: {doc_description})")
    try:
        time.sleep(request_delay)
        resp = ch_session.get(content_url, headers={"Accept": "application/pdf"}, timeout=120)
        resp.raise_for_status()
        if "application/pdf" in resp.headers.get("Content-Type", "").lower():
            logger.info(f"{company_no}: Successfully fetched PDF for doc {doc_id} (size: {len(resp.content)} bytes).")
            return resp.content, "pdf", None
        else:
            content_sample = resp.content[:200] if isinstance(resp.content, bytes) else resp.text[:200]
            err_msg = f"PDF request for doc {doc_id} got Content-Type {resp.headers.get('Content-Type', '')} instead of PDF. Sample: {content_sample}"
            logger.warning(f"{company_no}: {err_msg}")
            return None, "none", err_msg
    except requests.exceptions.RequestException as e_req:
        err_msg = f"PDF fetch failed for doc {doc_id}: {e_req}"
        logger.error(f"{company_no}: {err_msg}")
        return None, "none", err_msg
    except Exception as e_pdf_other: 
        err_msg = f"Unexpected error fetching PDF for doc {doc_id}: {e_pdf_other}"
        logger.error(f"{company_no}: {err_msg}", exc_info=True)
        return None, "none", err_msg

    final_err_msg = f"All attempts to fetch content for doc {doc_id} (Type: {doc_ch_type}, Desc: {doc_description}) failed."
    logger.warning(f"{company_no}: {final_err_msg}")
    return None, "none", final_err_msg


def get_ch_documents_metadata(
    company_no: str,
    api_key: str, # Passed from app.py, used by get_ch_session if env var not set
    categories: List[str],
    items_per_page: int,
    max_docs_to_fetch_meta: int,
    target_docs_per_category_in_date_range: int,
    year_range: Tuple[int, int]
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetches filing history metadata from Companies House for specified categories,
    filtering by year range as data is paged, until a target number of documents
    within the date range is met for each category or scan limits are reached.

    Args:
        company_no: The company registration number.
        api_key: Companies House API key.
        categories: List of filing categories to fetch.
        items_per_page: Number of items to request per API call (max 100 for CH).
        max_docs_to_fetch_meta: Maximum number of raw documents to scan per category (scan limit).
        target_docs_per_category_in_date_range: Target number of documents to find within the date range for each category.
        year_range: A tuple (start_year, end_year) for filtering documents.

    Returns:
        A tuple: (filtered_document_metadata_list, company_profile_data, error_message_or_None).
    """
    ch_session = get_ch_session(api_key) # Pass api_key to session setup
    start_filter_year, end_filter_year = year_range
    logger.info(
        f"Fetching CH document metadata for {company_no} | Categories: {categories} | "
        f"Years: {start_filter_year}-{end_filter_year} | Target/cat: {target_docs_per_category_in_date_range} | "
        f"Scan limit/cat: {max_docs_to_fetch_meta}"
    )

    if not categories:
        logger.warning(f"No categories specified for {company_no}. Skipping metadata fetch.")
        return [], None, "No categories specified."

    company_profile_data: Optional[Dict[str, Any]] = None
    if company_no in _company_profile_cache:
        company_profile_data = _company_profile_cache[company_no]
        logger.debug(f"Using cached profile for company {company_no}.")
    else:
        company_profile_url = f"{CH_API_BASE_URL}/company/{company_no}"
        try:
            time.sleep(0.3) # Rate limit general CH calls
            profile_resp = ch_session.get(company_profile_url, timeout=30)
            profile_resp.raise_for_status()
            company_profile_data = profile_resp.json()
            _company_profile_cache[company_no] = company_profile_data
            logger.debug(f"Fetched and cached profile for company {company_no}.")
        except requests.exceptions.RequestException as e:
            err_msg = f"API request failed for company profile {company_no}: {e}"
            logger.error(err_msg)
            return [], None, err_msg # No profile data if fetch fails
        except json.JSONDecodeError as e:
            err_msg = f"Failed to decode JSON for company profile {company_no}: {e}"
            logger.error(err_msg)
            return [], None, err_msg # No profile data if JSON decode fails
    
    if not company_profile_data: # Should be caught by exceptions, but defensive
        return [], None, "Failed to retrieve company profile data unexpectedly."

    filings_url_path = company_profile_data.get("links", {}).get("filing_history")
    if not filings_url_path:
        err_msg = f"No 'filing_history' link found for company {company_no}."
        logger.warning(f"{company_no}: {err_msg}")
        # Return profile even if filings link is missing, as profile itself might be useful
        return [], company_profile_data, err_msg 

    full_filings_url = filings_url_path if filings_url_path.startswith("http") else f"{CH_API_BASE_URL}{filings_url_path}"

    all_docs_in_date_range: List[Dict[str, Any]] = []
    processed_categories = set()
    total_api_calls_filings = 0

    for category in categories:
        cat_lower = category.lower().strip()
        if not cat_lower or cat_lower in processed_categories:
            continue
        processed_categories.add(cat_lower)
        
        logger.info(f"{company_no}: Category '{cat_lower}': Targeting {target_docs_per_category_in_date_range} docs in range {start_filter_year}-{end_filter_year}.")

        start_index = 0
        current_category_docs_in_date_range: List[Dict[str, Any]] = []
        total_items_scanned_for_category_api = 0 

        while True: 
            if total_items_scanned_for_category_api >= max_docs_to_fetch_meta:
                logger.info(f"{company_no}: Category '{cat_lower}': Reached scan limit ({max_docs_to_fetch_meta} API items) before checking API for more.")
                break

            params = {
                "category": cat_lower,
                "items_per_page": items_per_page,
                "start_index": start_index,
            }
            try:
                time.sleep(0.51) 
                total_api_calls_filings += 1
                logger.debug(f"{company_no}: Filings API call {total_api_calls_filings} to {full_filings_url} with params: {params}")
                resp = ch_session.get(full_filings_url, params=params, timeout=45)
                resp.raise_for_status()
                api_response_data = resp.json()
            except requests.exceptions.RequestException as e_req_filings:
                err_msg = f"API request failed for filings (category: {cat_lower}, start_index: {start_index}) for {company_no}: {e_req_filings}"
                logger.error(err_msg)
                return all_docs_in_date_range, company_profile_data, err_msg 
            except json.JSONDecodeError as e_json_filings:
                err_msg = f"Failed to decode JSON for filings (category: {cat_lower}, company: {company_no}): {e_json_filings}. Response text: {getattr(resp, 'text', 'N/A')[:200]}"
                logger.error(err_msg)
                return all_docs_in_date_range, company_profile_data, err_msg

            items_on_page = api_response_data.get("items", [])
            api_total_count_for_category = api_response_data.get("total_count", 0) # Total for this category in CH
            api_filing_history_status = api_response_data.get("filing_history_status") # e.g. 'filing-history-available'

            if not items_on_page:
                logger.info(f"{company_no}: Category '{cat_lower}': No more items from API at start_index {start_index} (status: {api_filing_history_status}).")
                break 

            page_items_in_date_range_count = 0
            for item in items_on_page:
                # Check if target for this category is already met before processing more items on the page
                if len(current_category_docs_in_date_range) >= target_docs_per_category_in_date_range:
                    break 

                item_date_str = item.get("date")
                if item_date_str:
                    try:
                        item_datetime = datetime.strptime(item_date_str, "%Y-%m-%d")
                        item_year = item_datetime.year
                        if start_filter_year <= item_year <= end_filter_year:
                            # Add a parsed datetime object to the item for easier sorting/use later if needed
                            item['_parsed_date'] = item_datetime 
                            current_category_docs_in_date_range.append(item)
                            page_items_in_date_range_count +=1
                    except ValueError:
                        logger.warning(f"{company_no}: Category '{cat_lower}': Could not parse date '{item_date_str}' for item {item.get('transaction_id', 'N/A')}. Skipping date filter for this item.")
                else:
                    logger.warning(f"{company_no}: Category '{cat_lower}': Item {item.get('transaction_id', 'N/A')} has no date. Skipping date filter for this item.")
            
            total_items_scanned_for_category_api += len(items_on_page)

            logger.debug(
                f"{company_no}: Category '{cat_lower}': Page start_index {start_index}. Scanned {len(items_on_page)} API items. "
                f"Found {page_items_in_date_range_count} in date range this page. "
                f"Total for category in range: {len(current_category_docs_in_date_range)}/{target_docs_per_category_in_date_range}. "
                f"Total API items scanned for category: {total_items_scanned_for_category_api}/{max_docs_to_fetch_meta} (API reports {api_total_count_for_category} total for cat)."
            )

            if len(current_category_docs_in_date_range) >= target_docs_per_category_in_date_range:
                logger.info(f"{company_no}: Category '{cat_lower}': Met target of {target_docs_per_category_in_date_range} docs in date range.")
                break 
            # Check scan limit for this category (total_items_scanned_for_category_api already incremented)
            if total_items_scanned_for_category_api >= max_docs_to_fetch_meta:
                logger.info(f"{company_no}: Category '{cat_lower}': Reached scan limit of {max_docs_to_fetch_meta} API items.")
                break
            # Check if all available items for the category have been scanned from the API
            if start_index + items_per_page >= api_total_count_for_category and api_total_count_for_category > 0:
                logger.info(f"{company_no}: Category '{cat_lower}': Scanned all ~{api_total_count_for_category} available API items (or reached end of paged results).")
                break
            
            start_index += items_per_page # Prepare for next page

        all_docs_in_date_range.extend(current_category_docs_in_date_range)
        logger.info(f"{company_no}: Category '{cat_lower}': Added {len(current_category_docs_in_date_range)} docs. Total docs in date range so far: {len(all_docs_in_date_range)}.")

    # Optional: Sort all collected documents by date if desired, e.g., most recent first
    # all_docs_in_date_range.sort(key=lambda x: x.get('_parsed_date', datetime.min), reverse=True)
    # Removing the _parsed_date key if it's not needed downstream, or keep it if useful
    # for item in all_docs_in_date_range:
    #     item.pop('_parsed_date', None)

    logger.info(
        f"{company_no}: Completed metadata fetch. Found {len(all_docs_in_date_range)} docs in date range across all categories. "
        f"Total CH Filings API calls: {total_api_calls_filings} (excluding profile call)."
    )
    return all_docs_in_date_range, company_profile_data, None # Success