# config.py

import os
import logging
import warnings
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import openai
import requests

# Attempt to import Google Generative AI
try:
    import google.generativeai as genai
except ImportError:
    genai = None # Set to None if not installed

load_dotenv()

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logs from libraries
logging.getLogger("PyPDF2").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)

# Suppress specific warnings
from PyPDF2 import errors as PyPDF2Errors
warnings.filterwarnings("ignore", message="incorrect startxref pointer.*")
warnings.filterwarnings("ignore", category=PyPDF2Errors.PdfReadWarning)


# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CH_API_KEY = os.getenv("CH_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID") # For Textract
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") # For Textract
AWS_REGION_DEFAULT = os.getenv("AWS_DEFAULT_REGION", "eu-west-2")
S3_TEXTRACT_BUCKET = os.getenv("S3_TEXTRACT_BUCKET") # For Textract

# --- Model Configuration ---
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL_DEFAULT = "gemini-1.5-pro-latest" # Or load from env

# --- Application Constants ---
MIN_MEANINGFUL_TEXT_LEN = 200
MAX_DOCS_TO_PROCESS_PER_COMPANY = int(os.getenv("MAX_DOCS_PER_COMPANY_PIPELINE", "20"))
CH_API_BASE_URL = "https://api.company-information.service.gov.uk"
CH_DOCUMENT_API_BASE_URL = "https://document-api.company-information.service.gov.uk"

# --- Protocol Text Fallback ---
# This will be the default. app.py will try to load strategic_protocols.txt
# and can update this value if successful.
PROTO_TEXT_FALLBACK = "You are a helpful AI assistant. Please provide concise and factual responses."

# --- AWS Pricing (relevant if Textract is used) ---
AWS_PRICING_CONFIG = {
    "textract_per_page": 0.0015,
    "s3_put_request_per_pdf_to_textract": 0.000005,
    "usd_to_gbp_exchange_rate": 0.80
}

# --- Initialize API Clients ---
_openai_client = None
_ch_session = None
_ch_api_key_used = None # To track which key the current session is using
_gemini_generative_model = None

# Removed direct initialization of _openai_client, genai.configure, and _ch_session here

def get_openai_client():
    global _openai_client
    if not _openai_client:
        if OPENAI_API_KEY:
            try:
                _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
                logger.info("OpenAI client initialized.")
            except Exception as e_openai_init:
                logger.error(f"Error setting up OpenAI client: {e_openai_init}")
        else:
            logger.warning("OPENAI_API_KEY not found. OpenAI calls will fail.")
    return _openai_client

def get_ch_session(api_key: Optional[str] = None) -> requests.Session:
    global _ch_session, _ch_api_key_used
    
    # Determine the API key to use: passed argument takes precedence over environment variable
    effective_api_key = api_key if api_key else CH_API_KEY

    if not effective_api_key:
        logger.warning("CH_API_KEY not provided as argument or found in environment. Companies House calls may fail or be rate-limited.")
        # If no key at all, decide if we should raise error or return a key-less session
        # For now, let's return a session that might work for non-authenticated endpoints or rely on IP rate limits
        if _ch_session is None or _ch_api_key_used is not None: # If session exists but had a key, or no session
            _ch_session = requests.Session()
            _ch_api_key_used = None
            logger.info("Initialized Companies House session without an API key.")
        return _ch_session

    # If session doesn't exist, or if the key to be used is different from the one the current session uses
    if _ch_session is None or _ch_api_key_used != effective_api_key:
        logger.info(f"Initializing new Companies House session. API key source: {'argument' if api_key else 'environment'}.")
        _ch_session = requests.Session()
        _ch_session.auth = (effective_api_key, "")
        # Add any other default headers or configurations for the session here
        # e.g., _ch_session.headers.update({'User-Agent': 'MyApplication/1.0'})
        _ch_api_key_used = effective_api_key
        logger.info(f"Companies House session configured with API key (ending with ...{effective_api_key[-4:] if len(effective_api_key) > 4 else effective_api_key}).")
    return _ch_session

def get_gemini_model(model_name: str):
    # global _gemini_generative_model # Not strictly needed as we return a new model instance
    if not genai:
        logger.warning("google-generativeai library not installed. Gemini calls will not be available.")
        return None
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not found. Gemini calls will fail.")
        return None
    
    # Configure genai here, only when first Gemini model is requested
    # This assumes genai.configure is safe to call multiple times if needed,
    # or that this function is the primary entry point for Gemini usage.
    # For simplicity, let's assume it's okay or that get_gemini_model is called once per model type.
    try:
        # Check if already configured to avoid re-configuring if not necessary,
        # though genai.configure itself might be idempotent.
        # This is a simple check; a more robust way might involve a global flag.
        if not getattr(get_gemini_model, "_configured_genai", False):
             genai.configure(api_key=GEMINI_API_KEY)
             logger.info("Google Generative AI SDK configured.")
             get_gemini_model._configured_genai = True # Mark as configured
    except Exception as e_gemini_config:
        logger.error(f"Error configuring Google Generative AI SDK: {e_gemini_config}")
        return None # Stop if configuration fails

    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model \'{model_name}\': {e}")
        return None

# Base path for the application (useful for file operations in app.py)
APP_BASE_PATH = Path(__file__).resolve().parent

