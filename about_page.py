# about_page.py
"""
Contains the content and rendering logic for the 'About' tab in the Strategic Counsel application.
Version 2: Revised emphasis on core AI strategic counsel capabilities.
"""
import streamlit as st
import config # To access APP_BASE_PATH and CH_PIPELINE_TEXTRACT_FLAG
# The following global variables are expected to be set in app.py before this module is used
# if they are to be referenced directly here.
# For now, this module relies on config for shared constants.
# PROTO_TEXT = config.PROTO_TEXT_FALLBACK # Example if needed directly, but app.py handles actual loading
# PROTO_PATH = config.APP_BASE_PATH / "strategic_protocols.txt" # Example

def render_about_page():
    """Renders all the content for the 'About' tab."""

    st.markdown("## ℹ️ About Strategic Counsel")
    st.markdown("*(Version 3.0 - Modular Refactor, Optional OCR)*")
    st.markdown("---")
    st.markdown("""
    **Strategic Counsel (SC)** is an advanced AI-powered workspace prototype, meticulously engineered to
    serve as an intelligent partner for legal, financial, and strategic professionals. Its core purpose is to
    facilitate **AI-assisted drafting, in-depth research, and nuanced strategic exploration**.

    SC achieves this by integrating sophisticated AI models with a powerful context injection system,
    allowing users to leverage bespoke protocols, matter-specific documents, web-based information, and curated
    memories to guide and inform the AI. Specialized tools, such as the UK Companies House analysis module,
    further augment these capabilities by providing structured data and insights.

    The application is designed with a focus on modularity, efficiency, and empowering users to derive
    actionable intelligence from complex information landscapes.

    *Disclaimer: This application is currently a demonstration and development prototype. It should not be
    used for making real-world legal or financial decisions without independent verification by qualified
    professionals. All data processing and AI outputs are subject to the inherent limitations of the
    underlying models and data sources.*
    """)

    st.subheader("Core Philosophy & Design Principles")
    st.markdown("""
    - **Intelligence Augmentation:** SC aims to be a 'cognitive partner,' enhancing the user's ability to draft,
      research, analyze, and strategize more effectively, rather than replacing human expertise.
    - **Context is Key:** The system's primary strength lies in its ability to maintain and inject relevant context
      (session digests, document summaries, web links, user-defined memories, and operational protocols)
      into AI interactions, leading to highly pertinent and tailored outputs.
    - **Strategic Exploration:** Provide a flexible environment for users to explore complex scenarios, test hypotheses,
      and generate diverse strategic options with AI assistance.
    - **Modularity & Maintainability:** The codebase is structured into distinct, manageable modules for
      easier development, testing, and future enhancements.
    - **Transparency & Control:** Users are provided with insights into the AI models being used, cost
      estimations (where applicable), and options to control aspects like data injection and specialized tool usage.
    - **Objective Analysis (for specialized tools):** When employing data-specific tools like the Companies House analyzer,
      the AI summarization prompts are designed to enforce rigorous, objective, and factual extraction of information.
    """)

    st.subheader("Key Features & Capabilities")

    with st.expander("Core AI-Assisted Counsel: Drafting, Research & Strategy", expanded=True):
        st.markdown("""
        The heart of Strategic Counsel lies in its 'Consult Counsel' tab, providing an interactive environment for:
        - **AI-Powered Drafting:** Generate initial drafts, refine existing text, create summaries, or formulate responses based on complex instructions and injected context.
        - **In-Depth Research Assistance:** Leverage AI to process and synthesize information from provided documents, URLs, and internal knowledge bases (memories/digests) to answer questions or identify key insights.
        - **Strategic Exploration & Scenario Planning:** Use the AI as a sounding board to explore different strategic avenues, analyze potential outcomes, and identify risks or opportunities based on the provided context.
        - **Protocol-Driven Behavior:** A master "protocol" file (`strategic_protocols.txt`) guides the AI's persona, tone, analytical framework, and primary objectives, ensuring its responses align with professional standards and user needs.
        - **Dynamic Context Injection:** This is crucial. Users can dynamically select and inject various information sources into the AI's working memory for each interaction:
            - **Session Digest:** A running, AI-updated summary of the current matter/topic.
            - **User-Defined Memories:** Persistent, topic-specific snippets of key facts, instructions, or client preferences.
            - **Uploaded Documents & Summaries:** Textual content and AI-generated summaries from PDFs, DOCX, TXT files.
            - **Web Links:** Content extracted and summarized from provided URLs.
        This rich contextual grounding ensures AI outputs are highly relevant, tailored, and directly applicable to the user's specific task and matter.
        """)

    with st.expander("Intelligent Document & Web Intake", expanded=False):
        st.markdown("""
        SC supports a versatile intake system to build the contextual basis for AI interaction:
        - **File Uploads:** Accepts PDF, DOCX, and TXT files. Documents are processed locally to extract textual content.
        - **URL Ingestion:** Users can paste URLs, and the application attempts to fetch and parse the primary textual content from web pages.
        - **AI-Powered Summarization:** Extracted text is summarized using an AI model. These summaries can be:
            - Displayed for quick review.
            - Cached locally per topic to avoid re-processing.
            - Selected for injection as context into the main AI consultation.
        """)

    with st.expander("Specialized Data Analysis: UK Companies House Tool", expanded=False):
        st.markdown("""
        As a powerful supporting feature, SC includes a specialized tool for analysis of UK company filings:
        - **Targeted Data Retrieval:** Fetches filings (Accounts, Confirmation Statements, etc.) for specified companies within a defined year range.
        - **Multi-Format Processing:** Handles JSON, XHTML, and PDF documents from Companies House.
        - **Advanced Text Extraction:** Includes standard PDF text extraction and optional AWS Textract OCR for scanned/image-based documents.
        - **Objective AI Summarization:** Extracted filing data is summarized by an AI (OpenAI or Gemini) using a rigorous, fact-focused prompt.
        - **Actionable Output:** Results are provided as a CSV digest and can feed into the broader strategic analysis within the 'Consult Counsel' tab by providing factual background on entities involved.
        """)

    with st.expander("Modular Architecture & Extensibility", expanded=False):
        st.markdown("""
        The application's backend has been significantly refactored into a series of focused Python modules
        (e.g., `config.py`, `app_utils.py`, `ai_utils.py`, `ch_pipeline.py`, `text_extraction_utils.py`).
        This modular design enhances code readability, simplifies maintenance, and allows for easier integration
        of new features, data sources, or AI models in the future.
        """)

    with st.expander("Session Management & Persistence", expanded=False):
        st.markdown("""
        - **Topic-Based Workspaces:** All work is siloed by a user-defined 'Matter / Topic ID'.
        - **Session History & Digests:** Interactions are logged and can be consolidated into a persistent 'Digest' for the topic.
        - **Caching:** Document summaries are cached to improve performance and reduce costs.
        """)

    with st.expander("Export & Logging", expanded=False):
        st.markdown("""
        - **DOCX Export:** AI responses from 'Consult Counsel' can be exported.
        - **CSV Digest Export:** CH analysis results are downloadable.
        - **Detailed Logging:** AI interactions and CH analysis parameters are logged for review.
        """)

    st.markdown("---")
    st.subheader("Technology Stack Highlights")
    st.markdown(f"""
    - **Frontend Framework:** Streamlit
    - **Core Language:** Python 3.x
    - **AI Model APIs:** OpenAI API (GPT series), Google Gemini API (Flash & Pro)
    - **Document Processing Libraries:** PyPDF2, pdfminer.six, python-docx, BeautifulSoup4
    - **Specialized Data APIs:** Companies House Public Data API
    - **Optional Cloud OCR Service:** AWS Textract (via Boto3)
    - **Data Handling & Utilities:** Pandas, Requests, python-dotenv
    """)

    st.markdown("---")
    # Accessing config attributes directly. Ensure app.py has loaded PROTO_TEXT and PROTO_PATH for these to be accurate if config isn't updated.
    # For robustness, these might be better passed as arguments to render_about_page if they are dynamic.
    # However, since config.py is imported, its static values are available.
    # The dynamic PROTO_TEXT loaded in app.py is not directly visible here unless config.PROTO_TEXT_FALLBACK was updated.

    # For CH_PIPELINE_TEXTRACT_FLAG, it's set in ch_pipeline.py and imported by app.py.
    # To make it accessible here via config, app.py would need to set it on the config object,
    # or ch_pipeline.py itself could set a flag on the config object.
    # For simplicity, if app.py imports ch_pipeline, it can then set a flag on config.
    # Assuming app.py might do: import config; from ch_pipeline import TEXTRACT_AVAILABLE; config.CH_PIPELINE_TEXTRACT_FLAG = TEXTRACT_AVAILABLE

    # Safely access attributes that might be set by app.py on the config module
    app_base_path_str = str(config.APP_BASE_PATH) if hasattr(config, 'APP_BASE_PATH') else "N/A"
    
    # Accessing CH_PIPELINE_TEXTRACT_FLAG: This flag is defined in ch_pipeline.py.
    # If app.py imports it and wants to make it available globally via config, it would do:
    # import config
    # from ch_pipeline import TEXTRACT_AVAILABLE as CH_PIPELINE_TEXTRACT_FLAG
    # config.CH_PIPELINE_TEXTRACT_FLAG = CH_PIPELINE_TEXTRACT_FLAG # Add this line in app.py after imports
    textract_available_str = str(getattr(config, 'CH_PIPELINE_TEXTRACT_FLAG', "Unknown (Not set in config by app.py)"))

    # Accessing PROTO_PATH and PROTO_TEXT (loaded by app.py)
    # app.py loads PROTO_TEXT and PROTO_PATH. If we want to display them here from config,
    # app.py should set them on the config object.
    # Example in app.py: config.LOADED_PROTO_PATH_NAME = PROTO_PATH.name; config.LOADED_PROTO_TEXT = PROTO_TEXT
    proto_path_name_str = getattr(config, 'LOADED_PROTO_PATH_NAME', "strategic_protocols.txt (default)")
    proto_text_status_str = "(using fallback if an error occurred or not specifically set in config by app.py)"
    if hasattr(config, 'LOADED_PROTO_TEXT') and hasattr(config, 'PROTO_TEXT_FALLBACK'):
        if config.LOADED_PROTO_TEXT != config.PROTO_TEXT_FALLBACK:
            proto_text_status_str = "(successfully loaded)"


    st.markdown(f"**Application Base Path (determined at runtime):** `{app_base_path_str}`")
    st.markdown(f"**AWS Textract OCR Utilities Available (backend check):** `{textract_available_str}`")
    st.markdown(f"**Protocol File Referenced:** `'{proto_path_name_str}'` {proto_text_status_str}")


    st.markdown("---")
    st.caption("Strategic Counsel | AI-Augmented Professional Workspace")

