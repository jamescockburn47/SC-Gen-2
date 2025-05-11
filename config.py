# config.py

import os
import logging
import warnings
from pathlib import Path

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
if OPENAI_API_KEY:
    try:
        _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized.")
    except Exception as e_openai_init:
        logger.error(f"Error setting up OpenAI client: {e_openai_init}")
else:
    logger.warning("OPENAI_API_KEY not found. OpenAI calls will fail.")

_gemini_generative_model = None # This is not used directly; get_gemini_model is preferred
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Google Generative AI SDK configured.")
    except Exception as e_gemini_config:
        logger.error(f"Error configuring Google Generative AI SDK: {e_gemini_config}")
elif not genai:
    logger.warning("google-generativeai library not installed. Gemini calls will not be available.")
elif not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found. Gemini calls will fail.")

_ch_session = requests.Session()
if CH_API_KEY:
    _ch_session.auth = (CH_API_KEY, "")
    logger.info("Companies House session configured with API key.")
else:
    logger.warning("CH_API_KEY not found. Companies House calls will fail.")

def get_openai_client():
    if not _openai_client:
        logger.error("OpenAI client requested but not initialized (API key likely missing).")
    return _openai_client

def get_ch_session():
    return _ch_session

def get_gemini_model(model_name: str):
    if not genai or not GEMINI_API_KEY:
        logger.error("Gemini model requested but SDK or API Key not configured.")
        return None
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model '{model_name}': {e}")
        return None

# Base path for the application (useful for file operations in app.py)
APP_BASE_PATH = Path(__file__).resolve().parent

