# app_utils.py

import json
import re
import logging
from typing import Tuple, Optional, List, Dict

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document # Assuming python-docx is installed
import io

from config import get_openai_client, get_ch_session, PROTO_TEXT_FALLBACK # Assuming PROTO_TEXT is here or passed

logger = logging.getLogger(__name__)

# Fallback protocol text if not loaded from file in app.py
PROTO_TEXT_FALLBACK = "You are a helpful AI assistant."


def _word_cap(word_count: int) -> int:
    """Determines a reasonable word cap for summaries based on input word count."""
    if word_count <= 2000:
        return max(150, int(word_count * 0.15))
    elif word_count <= 10000:
        return 300
    else:
        return min(500, int(word_count * 0.05))

def summarise_with_title(
    text: str,
    model_name_selected: str, # This implies the main app's selected model
    topic: str, # For context, not used in current prompt
    protocol_text: str = PROTO_TEXT_FALLBACK
) -> Tuple[str, str]:
    """
    Generates a short title and summary for UI display of uploaded documents.
    Currently hardcoded to use a specific OpenAI model for this task.
    """
    if not text or not text.strip():
        return "Empty Content", "No text was provided for summarization."

    word_count = len(text.split())
    summary_word_cap = _word_cap(word_count)
    # Truncate text if extremely long, to protect this specific quick summarizer
    text_to_summarise = text[:15000] # Increased slightly from 12k
    max_tokens_for_response = int(summary_word_cap * 1.8) # Allow more tokens for JSON structure and content

    # This specific summarizer is for short titles/summaries for UI display.
    # It will use a cost-effective OpenAI model by default for this limited task.
    # Consider making this configurable or using the main selected model if consistency is key.
    openai_model_for_this_task = "gpt-4o-mini"
    openai_client = get_openai_client()

    if not openai_client:
        logger.error(f"OpenAI client not available for summarise_with_title (topic: {topic}).")
        return "Summarization Error", "OpenAI client not configured."

    prompt = (
        f"Return ONLY valid JSON in the format {{\"title\": \"<A concise title of less than 12 words>\", "
        f"\"summary\": \"<A summary of approximately {summary_word_cap} words, capturing the essence of the text>\"}}.\n\n"
        f"Analyze the following text:\n---\n{text_to_summarise}\n---"
    )
    raw_response_content = ""
    title = "Error in Summarization"
    summary = "Could not generate summary due to an issue."

    try:
        response = openai_client.chat.completions.create(
            model=openai_model_for_this_task,
            temperature=0.2,
            max_tokens=max_tokens_for_response,
            messages=[
                {"role": "system", "content": protocol_text}, # Use provided or fallback protocol
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        raw_response_content = response.choices[0].message.content.strip()
        data = json.loads(raw_response_content)
        title = str(data.get("title", "Title Missing"))
        summary = str(data.get("summary", "Summary Missing"))
        logger.info(f"Successfully generated title/summary for topic '{topic}' using {openai_model_for_this_task}.")
    except json.JSONDecodeError as e_json:
        logger.error(f"JSONDecodeError in summarise_with_title (topic: {topic}, model: {openai_model_for_this_task}): {e_json}. Raw response: {raw_response_content[:200]}")
        title = "Summarization Format Error"
        summary = f"Failed to parse AI response as JSON. Preview: {raw_response_content[:150]}"
    except Exception as e:
        logger.error(f"Exception in summarise_with_title (topic: {topic}, model: {openai_model_for_this_task}): {e}. Raw response: {raw_response_content[:200]}")
        first_part = raw_response_content.split(".")[0][:75].strip() if raw_response_content else ""
        title = first_part if first_part else f"Summarization Failed ({type(e).__name__})"
        summary = raw_response_content if raw_response_content else "No response content."
    return title, summary


def fetch_url_content(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetches and extracts text content from a URL. Returns (text, error_message)."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch URL '{url}': {e}")
        return None, f"Fetch URL Error: {e.__class__.__name__} for {url}"

    try:
        soup = BeautifulSoup(response.content, "html.parser")
        for s_tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button", "input"]):
            s_tag.decompose()

        main_content_tags = soup.find_all(['main', 'article', 'div.content', 'div.main', 'div.post', 'section'])
        content_text = ""
        if main_content_tags:
            content_text = " ".join(tag.get_text(" ", strip=True) for tag in main_content_tags)
        
        if not content_text.strip() or len(content_text.split()) < 20 : # If main content is too short or empty, try body
            body_tag = soup.find("body")
            if body_tag:
                content_text = body_tag.get_text(" ", strip=True)
            else: # Fallback to all text if no body
                content_text = soup.get_text(" ", strip=True)
        
        # Basic cleaning
        content_text = re.sub(r'\s\s+', ' ', content_text).strip()
        content_text = re.sub(r'(\n\s*){3,}', '\n\n', content_text).strip() # Reduce multiple blank lines

        if not content_text.strip():
            logger.info(f"No significant text extracted from URL '{url}' after parsing.")
            return None, f"No text content found at {url} after parsing."
        logger.info(f"Successfully extracted text from URL '{url}' ({len(content_text)} chars).")
        return content_text, None
    except Exception as e:
        logger.error(f"Failed to process content from URL '{url}': {e}")
        return None, f"Process URL Error: {str(e)} for {url}"


def find_company_number(query: str, ch_api_key: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
    """
    Searches Companies House for a company number.
    Returns (company_number, error_message, first_match_details).
    """
    ch_session = get_ch_session() # Get the configured session
    if not ch_api_key: # CH_API_KEY from config is used by get_ch_session, this is an extra check
        return None, "Companies House API Key is missing or not configured.", None

    if not query or not query.strip():
        return None, "Please enter a company name or number to search.", None

    query_cleaned = query.strip().upper()
    
    # Regex for typical UK company numbers (allows for variations like SC, NI prefixes)
    # Standard: 8 digits, or 2 letters + 6 digits. Also allows for just 8 alphanumeric.
    company_no_match = re.fullmatch(r"([A-Z]{2})?([0-9]{6,8})|[A-Z0-9]{8}", query_cleaned.replace(" ", ""))
    
    if company_no_match:
        # Attempt to format it correctly, especially if it's just numbers
        potential_no = query_cleaned.replace(" ", "")
        if potential_no.isdigit() and len(potential_no) <= 8:
            formatted_no = potential_no.zfill(8)
            if re.fullmatch(r"[0-9]{8}", formatted_no): # Check if it's purely 8 digits
                 logger.info(f"Using provided/formatted company number: {formatted_no}")
                 return formatted_no, None, {"company_number": formatted_no, "title": "Input as Number"}

        # If it already matches a more complex pattern (e.g., SC123456 or 8 mixed chars)
        if re.fullmatch(r"[A-Z0-9]{8}|[A-Z]{2}[0-9]{6}", potential_no): # Strict check after potential zfill
            logger.info(f"Using provided company number: {potential_no}")
            return potential_no, None, {"company_number": potential_no, "title": "Input as Number/Code"}


    logger.info(f"Searching Companies House for name/number: '{query}'")
    search_url = "https://api.company-information.service.gov.uk/search/companies"
    params = {'q': query_cleaned, 'items_per_page': 5} # Fetch a few for user to see if direct match fails

    try:
        response = ch_session.get(search_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Companies House search API error for '{query}': {e}")
        return None, f"Companies House Search Error: {e}", None
    except json.JSONDecodeError as e_json:
        logger.error(f"Failed to decode JSON from CH search for '{query}': {e_json}")
        return None, "Companies House Search Error: Could not parse response.", None

    if not items:
        logger.warning(f"No company found for query '{query}'.")
        return None, f"No company found for '{query}'.", None

    # Prioritize exact match on company number if present in results
    for item in items:
        if item.get("company_number") == query_cleaned:
            logger.info(f"Exact company number match found: {item.get('title')} ({item.get('company_number')})")
            return item.get("company_number"), None, item

    # Fallback to the first result if no exact number match
    first_match = items[0]
    company_number = first_match.get("company_number")
    company_name = first_match.get("title", "N/A")
    
    if company_number:
        logger.info(f"Found via search: {company_name} ({company_number}). Using this number.")
        return company_number, None, first_match
    else:
        logger.warning(f"First match for '{query}' ({company_name}) has no company number.")
        return None, "First match found but no company number available in the result.", first_match


def extract_text_from_uploaded_file(file_obj: io.BytesIO, file_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts text from an uploaded file object (PDF, DOCX, TXT). Returns (text, error_message)."""
    file_ext = file_name.split(".")[-1].lower()
    text_content = None
    error_message = None
    
    try:
        file_obj.seek(0) # Ensure buffer is at the beginning
        if file_ext == "pdf":
            reader = PdfReader(file_obj)
            text_parts = [page.extract_text() or "" for page in reader.pages if hasattr(page, 'extract_text')]
            text_content = "\n".join(filter(None, text_parts)).strip()
        elif file_ext == "docx":
            doc = Document(file_obj)
            text_content = "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())
        elif file_ext == "txt":
            text_content = file_obj.getvalue().decode("utf-8", "ignore").strip()
        else:
            error_message = f"Unsupported file type: {file_ext}"
            logger.warning(error_message)

        if text_content is not None and not text_content.strip():
            text_content = None # Treat empty extraction as None
            # error_message = f"No text extracted from {file_name}." # Optional: report as error
            logger.info(f"No text content extracted from {file_name} (empty after extraction).")
        elif text_content:
             logger.info(f"Successfully extracted text from uploaded file: {file_name} ({len(text_content)} chars).")

    except Exception as e:
        logger.error(f"Error reading or processing uploaded file {file_name}: {e}", exc_info=True)
        error_message = f"File Read/Process Error for {file_name}: {e}"
        text_content = None
        
    return text_content, error_message