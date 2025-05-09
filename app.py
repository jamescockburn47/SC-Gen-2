#!/usr/bin/env python3
"""Strategic Counsel v3.0 - Modular Refactor

Key Changes:
- Codebase significantly refactored into multiple utility modules.
- AWS Textract OCR is now optional and managed via aws_textract_utils.py.
- UI option added to enable/disable Textract OCR for CH Analysis.
- Improved clarity and organization.
"""

from __future__ import annotations

import streamlit as st # Import Streamlit early
st.set_page_config( # Must be the first Streamlit command
    page_title="Strategic Counsel",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "# Strategic Counsel v3.0\nModular AI Legal Assistant Workspace."}
)

# Load environment variables and configurations first
# config.py handles dotenv loading and basic logging setup
try:
    import config
    logger = config.logger # Use the logger configured in config.py
except ImportError:
    st.error("Fatal Error: Could not import 'config.py'. Ensure the file exists and has no errors.")
    st.stop()
except Exception as e_conf:
    st.error(f"Fatal Error during config.py import or setup: {e_conf}")
    st.stop()


# Core application imports after config
import datetime as _dt
import hashlib as _hashlib
import io
import json
import os
import pathlib as _pl
import tempfile # For CH Analysis scratch directory base
import csv # For creating single company CSV for pipeline
from typing import List, Tuple, Dict, Optional

import pandas as pd
# OpenAI client is now typically fetched via config.get_openai_client()
# Requests session for CH is via config.get_ch_session()
from docx import Document # For exporting

# Application-specific utilities
try:
    from app_utils import (
        summarise_with_title,
        fetch_url_content,
        find_company_number,
        extract_text_from_uploaded_file
    )
except ImportError as e_app_utils:
    st.error(f"Fatal Error: Could not import utilities from 'app_utils.py': {e_app_utils}")
    st.stop()

# Companies House pipeline
try:
    from ch_pipeline import run_batch_company_analysis
    # Check if Textract is available via ch_pipeline's check, to inform UI
    from ch_pipeline import TEXTRACT_AVAILABLE as CH_PIPELINE_TEXTRACT_FLAG
except ImportError as e_ch_pipe:
    st.error(f"Fatal Error: Could not import from 'ch_pipeline.py': {e_ch_pipe}")
    st.stop()


# --- Constants & Initial Setup from Config (if not already used directly) ---
APP_BASE_PATH: _pl.Path = config.APP_BASE_PATH
OPENAI_API_KEY_PRESENT = bool(config.OPENAI_API_KEY and config.OPENAI_API_KEY.startswith("sk-"))
CH_API_KEY_PRESENT = bool(config.CH_API_KEY)
GEMINI_API_KEY_PRESENT = bool(config.GEMINI_API_KEY) # For UI warnings or model filtering

# Ensure required directories exist (relative to APP_BASE_PATH)
REQUIRED_DIRS_REL = ("memory", "memory/digests", "summaries", "exports", "logs", "static")
for rel_p in REQUIRED_DIRS_REL:
    abs_p = APP_BASE_PATH / rel_p
    try:
        abs_p.mkdir(parents=True, exist_ok=True)
    except OSError as e_mkdir:
        st.error(f"Fatal Error creating directory {abs_p.name}: {e_mkdir}")
        st.stop()

# --- UI Constants ---
# Pricing (Approx GBP per 1k INPUT tokens @ $1=£0.80)
# Last checked: May 9, 2025 (as per original comment). ALWAYS VERIFY CURRENT PRICING.
MODEL_PRICES_PER_1K_TOKENS_GBP: Dict[str, float] = {
    "gpt-4o": 0.0040,
    "gpt-4-turbo": 0.0080,
    "gpt-3.5-turbo": 0.0004,
    "gpt-4o-mini": 0.00012,
    "gemini-1.5-pro-latest": 0.0028, # Gemini API pricing: $3.50/1M input tokens for 1.5 Pro up to 128K context
    "gemini-1.5-flash-latest": 0.00028 # Gemini API pricing: $0.35/1M input tokens for 1.5 Flash up to 128K context
    # Note: Prices for >128K context for Gemini are higher, adjust if handling such large single inputs.
    # For simplicity, using the <128K context prices here, as summarization often chunks.
}
MODEL_ENERGY_WH_PER_1K_TOKENS: Dict[str, float] = { # Placeholder values
    "gpt-4o": 0.15, "gpt-4-turbo": 0.4, "gpt-3.5-turbo": 0.04, "gpt-4o-mini": 0.02,
    "gemini-1.5-pro-latest": 0.2, "gemini-1.5-flash-latest": 0.05
}
KETTLE_WH: int = 360 # Wh to boil a kettle once (approx)

# --- Protocol File ---
PROTO_PATH = APP_BASE_PATH / "strategic_protocols.txt"
PROTO_TEXT = ""
PROTO_HASH = ""
if not PROTO_PATH.exists():
    st.error(f"‼️ Protocol file '{PROTO_PATH.name}' not found in application directory! Please create this file.")
    st.stop()
try:
    PROTO_TEXT = PROTO_PATH.read_text(encoding="utf-8")
    PROTO_HASH = _hashlib.sha256(PROTO_TEXT.encode()).hexdigest()[:8]
    config.PROTO_TEXT_FALLBACK = PROTO_TEXT # Update the fallback in app_utils if it uses it
except Exception as e_proto:
    st.error(f"‼️ Error loading protocol file '{PROTO_PATH.name}': {e_proto}")
    st.stop()

# --- CH Categories ---
CH_CATEGORIES: Dict[str, str] = {
    "Accounts": "accounts", "Confirmation Stmt": "confirmation-statement",
    "Officers": "officers", "Capital": "capital", "Charges": "mortgage",
    "Insolvency": "insolvency", "PSC": "persons-with-significant-control", # Added PSC
    "Name Change": "change-of-name", "Reg. Office": "registered-office-address",
}


# ── Session State Initialization ───────────────────────────────────
def init_session_state():
    defaults = {
        "current_topic": "general_default_topic", # Ensure a default topic
        "session_history": [], # List of strings: "Instruction: ...\n\nResponse: ..."
        "loaded_memories": [], # List of strings (selected memory snippets)
        "processed_summaries": [], # List of tuples: (source_id, title, summary_text)
        "selected_summary_texts": [], # List of summary_text strings for injection
        "latest_digest_content": "", # String content of the current topic's digest
        "document_processing_complete": True, # Flag for async-like doc processing
        "ch_last_digest_path": None, # Path object to the last CH analysis CSV
        "ch_last_df": None, # Pandas DataFrame of the last CH analysis
        "ch_last_narrative": None, # AI-generated narrative from CH df
        "ch_last_batch_metrics": {} # Metrics from the last CH run
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ── Sidebar UI ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")
    
    # Topic Management
    current_topic_input = st.text_input(
        "Matter / Topic ID",
        st.session_state.current_topic, # Uses the initialized default
        key="topic_input_sidebar", # Unique key
        help="Changing topic resets session history, summaries, and CH results for this session."
    )
    if current_topic_input != st.session_state.current_topic:
        st.session_state.current_topic = current_topic_input
        # Reset relevant parts of session state when topic changes
        st.session_state.session_history = []
        st.session_state.processed_summaries = []
        st.session_state.selected_summary_texts = []
        st.session_state.loaded_memories = [] # Reset selected memories too
        st.session_state.ch_last_digest_path = None
        st.session_state.ch_last_df = None
        st.session_state.ch_last_narrative = None
        st.session_state.ch_last_batch_metrics = {}
        # latest_digest_content will be reloaded based on new topic below
        st.rerun()

    def _topic_color_style(topic_str: str) -> str:
        # Simple hash-based color for visual feedback on topic
        color_hue = int(_hashlib.sha1(topic_str.encode()).hexdigest(), 16) % 360
        return f"background-color:hsl({color_hue}, 70%, 90%); padding:8px 12px; border-radius:8px; margin:8px 0; text-align:center; color:#333;"
    st.markdown(f"<div style='{_topic_color_style(st.session_state.current_topic)}'>Topic: <strong>{st.session_state.current_topic}</strong></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### AI Settings")

    if not OPENAI_API_KEY_PRESENT:
        st.error("‼️ OpenAI API Key missing/invalid in your .env file. Most AI functions will fail.")
    if not GEMINI_API_KEY_PRESENT: # Added check for Gemini key for UI feedback
        st.warning("⚠️ Gemini API Key missing from .env. Gemini models will not be available.")

    # AI Model Selection
    available_models = list(MODEL_PRICES_PER_1K_TOKENS_GBP.keys())
    if not GEMINI_API_KEY_PRESENT: # Filter out Gemini models if key is missing
        available_models = [m for m in available_models if not m.startswith("gemini-")]
    if not OPENAI_API_KEY_PRESENT: # Filter out OpenAI models if key is missing
        available_models = [m for m in available_models if not m.startswith("gpt-")]
    
    if not available_models:
        st.error("‼️ No AI models available due to missing API keys. Application cannot function properly.")
        st.stop()

    default_model_index = 0
    if "selected_ai_model_sidebar" in st.session_state and st.session_state.selected_ai_model_sidebar in available_models:
        default_model_index = available_models.index(st.session_state.selected_ai_model_sidebar)
    
    selected_model_name = st.selectbox(
        "AI Model (for Consultation & CH Summary)",
        available_models,
        index=default_model_index,
        key="selected_ai_model_sidebar" # Unique key
    )
    price_per_1k = MODEL_PRICES_PER_1K_TOKENS_GBP.get(selected_model_name, 0.0)
    st.metric("Est. Input Cost / 1K Tokens", f"£{price_per_1k:.5f}")
    ai_temp = st.slider("AI Creativity (Temperature)", 0.0, 1.0, 0.2, 0.05, key="ai_temp_slider_sidebar")

    st.markdown("---")
    st.markdown("### Context Injection")

    # Memory File Handling
    memory_file_path = APP_BASE_PATH / "memory" / f"{st.session_state.current_topic}.json"
    loaded_memories_from_file: List[str] = []
    if memory_file_path.exists():
        try:
            mem_data = json.loads(memory_file_path.read_text(encoding="utf-8"))
            if isinstance(mem_data, list): # Expecting a list of strings
                loaded_memories_from_file = [str(item) for item in mem_data if isinstance(item, str)]
        except Exception as e_mem_load:
            st.warning(f"Could not load/parse memory file {memory_file_path.name}: {e_mem_load}")

    # Use st.session_state.loaded_memories for multiselect default, which is updated upon selection
    selected_mem_snippets = st.multiselect(
        "Inject Memories (from topic file)",
        loaded_memories_from_file,
        default=st.session_state.loaded_memories, # Persists selection across reruns
        key="mem_multiselect_sidebar"
    )
    st.session_state.loaded_memories = selected_mem_snippets # Update session state with current selection

    # Digest File Handling
    digest_file_path = APP_BASE_PATH / "memory" / "digests" / f"{st.session_state.current_topic}.md"
    if digest_file_path.exists():
        try:
            st.session_state.latest_digest_content = digest_file_path.read_text(encoding="utf-8")
        except Exception as e_digest_load:
            st.warning(f"Could not load digest {digest_file_path.name}: {e_digest_load}")
            st.session_state.latest_digest_content = "" # Clear if error
    else: # If file doesn't exist for current topic, clear content
        st.session_state.latest_digest_content = ""

    inject_digest_checkbox = st.checkbox(
        "Inject Digest (if available for topic)",
        value=bool(st.session_state.latest_digest_content), # Default to checked if content exists
        key="inject_digest_checkbox_sidebar",
        disabled=not bool(st.session_state.latest_digest_content)
    )

    st.markdown("---")
    st.markdown("### Document Intake (for Context)")
    uploaded_docs_list = st.file_uploader(
        "Upload Docs (PDF, DOCX, TXT)",
        ["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="doc_uploader_sidebar"
    )
    urls_input_str = st.text_area("Paste URLs (one per line)", key="url_textarea_sidebar", height=80)
    urls_to_process = [u.strip() for u in urls_input_str.splitlines() if u.strip().startswith("http")]

    # --- Document Processing Logic ---
    # Identifiers for current sources (uploaded files by name, URLs by themselves)
    current_source_identifiers = {f.name for f in uploaded_docs_list} | set(urls_to_process)
    # Identifiers of summaries already processed and stored in session state
    processed_summary_ids_in_session = {s_tuple[0] for s_tuple in st.session_state.processed_summaries}
    
    sources_needing_processing = current_source_identifiers - processed_summary_ids_in_session
    
    # If there are new sources AND previous processing is complete, start new processing
    if sources_needing_processing and st.session_state.document_processing_complete:
        st.session_state.document_processing_complete = False # Set flag before starting
        
        # Prepare cache directory for current topic's summaries
        summaries_cache_dir_for_topic = APP_BASE_PATH / "summaries" / st.session_state.current_topic
        summaries_cache_dir_for_topic.mkdir(parents=True, exist_ok=True)
        
        newly_processed_summaries_for_this_run: List[Tuple[str, str, str]] = []
        
        with st.spinner(f"Processing {len(sources_needing_processing)} new document(s)/URL(s)..."):
            progress_bar_docs = st.progress(0.0)
            
            for idx, src_id in enumerate(list(sources_needing_processing)): # Iterate over a list copy
                title, summary = "Error", "Processing Failed"
                # Cache file name based on hash of source identifier
                cache_file_name = f"summary_{_hashlib.sha256(src_id.encode()).hexdigest()[:16]}.json"
                summary_cache_file = summaries_cache_dir_for_topic / cache_file_name

                if summary_cache_file.exists():
                    try:
                        cached_data = json.loads(summary_cache_file.read_text(encoding="utf-8"))
                        title, summary = cached_data.get("t", "Cache Title Error"), cached_data.get("s", "Cache Summary Error")
                        logger.info(f"Loaded summary for '{src_id}' from cache.")
                    except Exception as e_cache:
                        logger.warning(f"Cache read error for {src_id}: {e_cache}. Regenerating.")
                        # Fall through to re-processing
                
                # If not cached or cache read failed, process the source
                if title == "Error" or "Cache" in title : # Needs processing
                    raw_content, error_msg = None, None
                    if src_id in {f.name for f in uploaded_docs_list}: # It's an uploaded file
                        # Find the file object again
                        file_obj_to_process = next((f for f in uploaded_docs_list if f.name == src_id), None)
                        if file_obj_to_process:
                            # Reading file content needs to be done carefully with BytesIO
                            file_bytes_io = io.BytesIO(file_obj_to_process.getvalue())
                            raw_content, error_msg = extract_text_from_uploaded_file(file_bytes_io, src_id)
                    elif src_id in urls_to_process: # It's a URL
                        raw_content, error_msg = fetch_url_content(src_id)
                    
                    if error_msg:
                        title, summary = f"Error: {src_id[:40]}...", error_msg
                    elif not raw_content or not raw_content.strip():
                        title, summary = f"Empty Content: {src_id[:40]}...", "No text extracted."
                    else:
                        # Use the main selected AI model for these UI summaries too for consistency
                        # Or, stick to a fast/cheap model like gpt-4o-mini if that's preferred for quick summaries
                        # For this refactor, let's use `summarise_with_title` which is hardcoded for gpt-4o-mini
                        # but now takes PROTO_TEXT.
                        title, summary = summarise_with_title(raw_content, "gpt-4o-mini", st.session_state.current_topic, PROTO_TEXT)
                        # Cache the new summary if successfully generated (not an error state)
                        if "Error" not in title and "Empty" not in title :
                            try:
                                summary_cache_file.write_text(json.dumps({"t":title,"s":summary,"src":src_id}),encoding="utf-8")
                            except Exception as e_cache_write:
                                logger.warning(f"Failed to write summary cache for {src_id}: {e_cache_write}")
                
                newly_processed_summaries_for_this_run.append((src_id, title, summary))
                progress_bar_docs.progress((idx + 1) / len(sources_needing_processing))

            # Update session state: add new summaries, keep existing ones not in current_source_identifiers
            # This logic ensures that if a file is removed from upload, its summary is also removed.
            final_processed_summaries = newly_processed_summaries_for_this_run + \
                [s_tuple for s_tuple in st.session_state.processed_summaries if s_tuple[0] not in sources_needing_processing and s_tuple[0] in current_source_identifiers]
            st.session_state.processed_summaries = final_processed_summaries
            
            progress_bar_docs.empty()
        st.session_state.document_processing_complete = True # Reset flag
        st.rerun() # Rerun to update the displayed summaries list

    # Display available summaries and allow selection for injection
    st.session_state.selected_summary_texts = [] # Clear and rebuild based on checkbox state
    if st.session_state.processed_summaries:
        st.markdown("---")
        st.markdown("### Available Summaries (Select to Inject)")
        
        # Make displayed summaries persistent using their source_id in checkbox key
        for idx, (s_id, title, summary_text) in enumerate(st.session_state.processed_summaries):
            checkbox_key = f"sum_sel_{_hashlib.md5(s_id.encode()).hexdigest()}" # Stable key for checkbox
            # Default to True if newly processed, otherwise use existing state
            default_checked = s_id in sources_needing_processing or st.session_state.get(checkbox_key, True)

            is_injected = st.checkbox(
                f"{idx+1}. {title[:40]}...",
                value=default_checked,
                key=checkbox_key, # Use the stable key
                help=f"Source: {s_id}\nSummary (preview): {summary_text[:200]}..."
            )
            if is_injected:
                st.session_state.selected_summary_texts.append(summary_text)


    st.markdown("---")
    if st.button("End Session & Update Digest", key="end_session_button_sidebar"):
        if not st.session_state.session_history:
            st.warning("No new interactions in this session to add to digest.")
        else:
            with st.spinner("Updating Digest..."):
                new_interactions_block = "\n\n---\n\n".join(st.session_state.session_history)
                existing_digest_text = st.session_state.latest_digest_content

                update_digest_prompt = (
                    f"Consolidate the following notes. Integrate the NEW interactions into the EXISTING digest, "
                    f"maintaining a coherent and concise summary. Aim for a maximum of around 2000 words for the entire updated digest. "
                    f"Preserve key facts and decisions.\n\n"
                    f"EXISTING DIGEST (for topic: {st.session_state.current_topic}):\n{existing_digest_text}\n\n"
                    f"NEW INTERACTIONS (to integrate for topic: {st.session_state.current_topic}):\n{new_interactions_block}"
                )
                try:
                    openai_client = config.get_openai_client()
                    if not openai_client:
                        st.error("OpenAI client not available for digest update.")
                        st.stop()

                    response = openai_client.chat.completions.create(
                        model=selected_model_name, # Use the main selected model
                        temperature=0.1,
                        max_tokens=3000, # Allow ample space for updated digest
                        messages=[
                            {"role": "system", "content": PROTO_TEXT},
                            {"role": "user", "content": update_digest_prompt}
                        ]
                    )
                    updated_digest_text = response.choices[0].message.content.strip()
                    digest_file_path.write_text(updated_digest_text, encoding="utf-8")
                    
                    # Append to a historical digest log as well
                    historical_digest_path = APP_BASE_PATH / "memory" / "digests" / f"history_{st.session_state.current_topic}.md"
                    with historical_digest_path.open("a", encoding="utf-8") as fp_hist:
                        fp_hist.write(f"\n\n### Update: {_dt.datetime.now():%Y-%m-%d %H:%M} (Model: {selected_model_name})\n{updated_digest_text}\n---\n")
                    
                    st.success(f"Digest for '{st.session_state.current_topic}' updated successfully.")
                    st.session_state.session_history = [] # Clear history for this session
                    st.session_state.latest_digest_content = updated_digest_text # Update in-session content
                    st.rerun()
                except Exception as e_digest_update:
                    st.error(f"Digest update failed: {e_digest_update}")

# ── Main Application Area UI (Using Tabs) ─────────────────────────
st.markdown(f"## 🏛️ Strategic Counsel: {st.session_state.current_topic}")
tab_consult, tab_ch_analysis, tab_about = st.tabs(["💬 Consult Counsel", "🇬🇧 Companies House Analysis", "ℹ️ About"])

with tab_consult:
    st.markdown("Provide instructions and context (using sidebar options) for drafting, analysis, or advice.")
    user_instruction_main = st.text_area(
        "Your Instruction:",
        height=200,
        key="main_instruction_area_consult_tab",
        placeholder="Example: Draft a response to the client email regarding their upcoming board meeting, considering points from their annual report summary and our discussion digest."
    )
    
    if st.button("✨ Consult Counsel", type="primary", key="run_ai_button_consult_tab"):
        if not user_instruction_main.strip():
            st.warning("Please enter your instructions.")
        elif not selected_model_name:
            st.error("No AI model selected or available. Please check sidebar configuration and API keys.")
        else:
            with st.spinner(f"Consulting {selected_model_name}..."):
                # Construct messages for AI
                messages_for_ai = [{"role": "system", "content": PROTO_TEXT + f"\n[Protocol Hash:{PROTO_HASH}]"}]
                
                context_parts_for_ai = []
                if inject_digest_checkbox and st.session_state.latest_digest_content:
                    context_parts_for_ai.append(f"CURRENT DIGEST FOR TOPIC '{st.session_state.current_topic}':\n{st.session_state.latest_digest_content}")
                if st.session_state.loaded_memories: # These are the selected memory snippets
                    context_parts_for_ai.append("INJECTED MEMORIES (selected by user):\n" + "\n---\n".join(st.session_state.loaded_memories))
                if st.session_state.selected_summary_texts: # These are from selected document summaries
                    context_parts_for_ai.append("SELECTED DOCUMENT SUMMARIES (from uploaded/URL sources):\n" + "\n---\n".join(st.session_state.selected_summary_texts))
                
                if context_parts_for_ai:
                    messages_for_ai.append({"role": "system", "content": "ADDITIONAL CONTEXT:\n\n" + "\n\n".join(context_parts_for_ai)})
                
                messages_for_ai.append({"role": "user", "content": user_instruction_main})
                
                try:
                    # Determine which client to use based on model name
                    ai_response_text = "Error: AI response could not be generated."
                    usage_info = None

                    if selected_model_name.startswith("gpt-"):
                        openai_client = config.get_openai_client()
                        if not openai_client: raise ValueError("OpenAI client not available.")
                        ai_api_response = openai_client.chat.completions.create(
                            model=selected_model_name,
                            temperature=ai_temp,
                            messages=messages_for_ai,
                            max_tokens=3500 # Max tokens for completion
                        )
                        ai_response_text = ai_api_response.choices[0].message.content.strip()
                        usage_info = ai_api_response.usage
                    elif selected_model_name.startswith("gemini-"):
                        gemini_model_client = config.get_gemini_model(selected_model_name)
                        if not gemini_model_client: raise ValueError("Gemini client/model not available.")
                        if not config.genai_sdk: raise ValueError("Gemini SDK not loaded.")

                        # Gemini API expects contents as a list of parts or a simple string.
                        # For chat-like interaction, usually a list of turns.
                        # Here, we'll send the constructed prompt history.
                        gemini_api_response = gemini_model_client.generate_content(
                            contents=messages_for_ai, # This structure should work for Gemini chat history
                            generation_config=config.genai_sdk.types.GenerationConfig(
                                temperature=ai_temp,
                                max_output_tokens=3500 # Max tokens for completion
                            )
                        )
                        ai_response_text = gemini_api_response.text.strip() # Assuming response.text for Gemini
                        # Gemini API doesn't return token usage in the same way as OpenAI in the basic response.
                        # For more accurate Gemini token counting, you might need to use model.count_tokens() before the call.
                        # For simplicity, we'll estimate or omit for Gemini here in UI.
                        # usage_info = {"prompt_tokens": "N/A for Gemini (estimate)", "completion_tokens": "N/A (estimate)"}
                    else:
                        raise ValueError(f"Unsupported model type: {selected_model_name}")

                    st.session_state.session_history.append(f"Instruction:\n{user_instruction_main}\n\nResponse ({selected_model_name}):\n{ai_response_text}")
                    
                    with st.chat_message("assistant", avatar="⚖️"):
                        st.markdown(ai_response_text)
                    
                    # --- Run Details & Export Expander ---
                    with st.expander("📊 Run Details & Export"):
                        prompt_tokens, completion_tokens = 0, 0
                        if usage_info and hasattr(usage_info, 'prompt_tokens') and hasattr(usage_info, 'completion_tokens'):
                            prompt_tokens = usage_info.prompt_tokens
                            completion_tokens = usage_info.completion_tokens
                        total_tokens = prompt_tokens + completion_tokens
                        
                        cost = (total_tokens / 1000) * price_per_1k if total_tokens > 0 else 0.0
                        energy_model_wh = MODEL_ENERGY_WH_PER_1K_TOKENS.get(selected_model_name, 0.0)
                        energy_wh = (total_tokens / 1000) * energy_model_wh if total_tokens > 0 else 0.0

                        st.metric("Total Tokens (approx.)", f"{total_tokens:,}", f"{prompt_tokens:,} prompt + {completion_tokens:,} completion" if total_tokens > 0 else "N/A for Gemini in this view")
                        st.metric("Est. Cost", f"£{cost:.5f}" if cost > 0 else "N/A for Gemini in this view")
                        if energy_model_wh > 0 and energy_wh > 0:
                            st.metric("Est. Energy", f"{energy_wh:.3f}Wh", f"~{(energy_wh / KETTLE_WH * 100):.1f}% of Kettle Boil" if KETTLE_WH > 0 else "")
                        
                        # Export options
                        ts_now_str = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                        # DOCX Export
                        docx_filename = f"{st.session_state.current_topic}_{ts_now_str}_response.docx"
                        docx_export_path = APP_BASE_PATH / "exports" / docx_filename
                        try:
                            doc = Document()
                            doc.add_heading(f"AI Consultation: {st.session_state.current_topic}", level=1)
                            doc.add_paragraph(f"Instruction: {user_instruction_main}\n")
                            doc.add_paragraph(f"Response ({selected_model_name} at {ts_now_str}):\n{ai_response_text}")
                            doc.save(docx_export_path)
                            with open(docx_export_path, "rb") as fp_docx:
                                st.download_button("Download Response (.docx)", fp_docx, docx_filename, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                        except Exception as e_docx_export:
                            st.error(f"DOCX export failed: {e_docx_export}")
                        
                        # Log interaction
                        log_filename = f"{st.session_state.current_topic}_{ts_now_str}_log.json"
                        log_export_path = APP_BASE_PATH / "logs" / log_filename
                        log_data_to_save = {
                            "topic": st.session_state.current_topic, "timestamp": ts_now_str,
                            "model_used": selected_model_name, "temperature": ai_temp,
                            "tokens": {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens},
                            "estimated_cost_gbp": cost, "estimated_energy_wh": energy_wh,
                            "messages_sent_to_api": [m["content"] for m in messages_for_ai if m["role"]=="user"], # Log user parts
                            "system_prompts_summary": [m["content"][:100]+"..." for m in messages_for_ai if m["role"]=="system"],
                            "answer_received_preview": ai_response_text[:200]+"..."
                        }
                        try:
                            log_export_path.write_text(json.dumps(log_data_to_save, indent=2), encoding="utf-8")
                        except Exception as e_log_save:
                            st.error(f"Log saving failed: {e_log_save}")

                except Exception as e_ai_consult:
                    st.error(f"AI Consultation Error with {selected_model_name}: {e_ai_consult}", icon="🚨")
                    logger.error(f"AI Consultation Error ({selected_model_name}): {e_ai_consult}", exc_info=True)
    
    # Display session history
    if st.session_state.session_history:
        st.markdown("---")
        st.subheader("📜 Current Session History (Newest First)")
        history_display_container = st.container(height=400) # Scrollable container
        for i, entry_text in enumerate(reversed(st.session_state.session_history)):
            history_display_container.markdown(f"**Interaction {len(st.session_state.session_history)-i}:**\n---\n{entry_text}\n\n")


with tab_ch_analysis:
    st.markdown("Fetch, process, and summarize UK company filings using the AI model selected in the sidebar.")
    if not CH_API_KEY_PRESENT:
        st.warning("⚠️ Companies House API Key missing from .env file. This tab's functionality will be limited or non-functional.", icon="🔑")

    col1_ch_input, col2_ch_results = st.columns([1, 2]) # Input on left, results on right

    with col1_ch_input:
        st.subheader("Inputs & Configuration")
        ch_input_method = st.radio(
            "1. Input Method:",
            ("Upload CSV (Company Numbers)", "Single Company Lookup"),
            key="ch_input_method_radio",
            horizontal=True,
            disabled=not CH_API_KEY_PRESENT # Disable if no CH key
        )
        
        uploaded_ch_csv_file_ui = None
        single_company_query_str_ui = ""

        if ch_input_method == "Upload CSV (Company Numbers)":
            uploaded_ch_csv_file_ui = st.file_uploader(
                "Upload CSV with Company Numbers", ["csv"],
                key="ch_csv_uploader_ui",
                help="CSV should have a header row, with company registration numbers in the first column."
            )
        else: # Single Company Lookup
            single_company_query_str_ui = st.text_input(
                "Enter Company Name or Number",
                key="ch_single_company_input_ui",
                help="Enter a UK company registration number or full company name."
            )
        
        st.markdown("---")
        st.markdown("##### Document Selection")
        selected_ch_categories_display = st.multiselect(
            "Document Categories",
            list(CH_CATEGORIES.keys()),
            default=["Accounts", "Confirmation Stmt", "Charges"], # Sensible defaults
            key="ch_categories_multiselect_ui",
            help="Select filing categories to retrieve."
        )
        api_categories_for_pipeline = [CH_CATEGORIES[cat_name] for cat_name in selected_ch_categories_display if cat_name in CH_CATEGORIES]

        current_system_year = _dt.date.today().year
        # Default year range: last 3 full years of filings
        ch_default_start_year = current_system_year - 4 
        ch_default_end_year = current_system_year -1 # Filings for current year might be incomplete
        
        selected_year_range_ch = st.slider(
            "Filing Year Range (inclusive)",
            2000, current_system_year,
            (ch_default_start_year, ch_default_end_year),
            key="ch_year_range_slider_ui"
        )
        start_year_for_pipeline, end_year_for_pipeline = selected_year_range_ch
        
        st.markdown("---")
        st.markdown("##### Analysis & Output Options")
        additional_ai_instructions_ch = st.text_area(
            "Additional AI Summary Instructions for Filings",
            placeholder="Example: Pay specific attention to changes in liabilities and any mention of legal proceedings.",
            key="ch_ai_instructions_textarea_ui", height=100,
            help="Provide specific focus points for AI summarization. Base prompt already covers core financial/legal areas."
        )

        # Option to use Textract OCR
        use_textract_ocr_ch = st.checkbox(
            "Use AWS Textract for PDF OCR (if needed for scanned PDFs)",
            value=False, # Default to False (cost-saving, relies on standard PDF extraction)
            key="ch_use_textract_checkbox_ui",
            help="Enable for better text from image-based PDFs. Requires AWS credentials and S3 bucket in .env.",
            disabled=not CH_PIPELINE_TEXTRACT_FLAG # Disable if aws_textract_utils not available
        )
        if not CH_PIPELINE_TEXTRACT_FLAG and use_textract_ocr_ch: # User checked it but it's not available
            st.warning("Textract OCR selected, but the necessary AWS utilities are not available in the backend. OCR will be skipped.")
            use_textract_ocr_ch = False # Force disable

        keep_temp_files_days_ch = st.slider(
            "Temporary File Retention (Days in Scratch)", 0, 30, 1, # Default to 1 day for inspection
            key="ch_keep_temp_files_slider_ui",
            help="Days to keep downloaded/processed files in the temporary run folder. 0 deletes immediately."
        )
        
        run_ch_analysis_button_ui = st.button(
            "🚀 Run Companies House Analysis",
            key="ch_run_analysis_button_ui",
            use_container_width=True,
            disabled=not CH_API_KEY_PRESENT or not selected_model_name # Also disable if no AI model selected
        )

    with col2_ch_results:
        st.subheader("Analysis Results")
        ch_results_display_container = st.container(border=True, height=600) # Main container for results

        if run_ch_analysis_button_ui:
            # Clear previous results from session state before starting a new run
            st.session_state.ch_last_digest_path = None
            st.session_state.ch_last_df = None
            st.session_state.ch_last_narrative = None
            st.session_state.ch_last_batch_metrics = {}

            with ch_results_display_container: # Display updates within this container
                st.info("Preparing Companies House analysis run...")
                
                # Prepare input CSV for the pipeline
                temp_csv_for_pipeline_path: Optional[_pl.Path] = None
                # Create a unique scratch directory for this run's artifacts
                ch_run_scratch_dir = _pl.Path(tempfile.gettempdir()) / f"ch_app_run_{_dt.datetime.now():%Y%m%d%H%M%S}"
                try:
                    ch_run_scratch_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e_mkdir_ch_run:
                    st.error(f"Error creating run scratch directory {ch_run_scratch_dir}: {e_mkdir_ch_run}")
                    st.stop()

                if ch_input_method == "Upload CSV (Company Numbers)":
                    if not uploaded_ch_csv_file_ui:
                        st.warning("Please upload a CSV file with company numbers.")
                        st.stop()
                    # Save uploaded CSV to the run's scratch directory to pass to pipeline
                    temp_csv_for_pipeline_path = ch_run_scratch_dir / uploaded_ch_csv_file_ui.name
                    try:
                        temp_csv_for_pipeline_path.write_bytes(uploaded_ch_csv_file_ui.getbuffer())
                    except IOError as e_csv_save_ui:
                        st.error(f"Error saving uploaded CSV to scratch: {e_csv_save_ui}")
                        st.stop()
                else: # Single Company Lookup
                    if not single_company_query_str_ui.strip():
                        st.warning("Please enter a company name or number for lookup.")
                        st.stop()
                    
                    # Use app_utils.find_company_number
                    found_co_no, find_err, _ = find_company_number(single_company_query_str_ui, config.CH_API_KEY)
                    if find_err:
                        st.error(f"Company Lookup Error: {find_err}")
                        st.stop()
                    if not found_co_no: # Should be caught by find_err, but double check
                        st.error("Could not resolve company name/number.")
                        st.stop()
                    
                    st.success(f"Found company: Using number {found_co_no} for analysis.")
                    # Create a temporary CSV for this single company
                    temp_csv_for_pipeline_path = ch_run_scratch_dir / f"single_co_{found_co_no}.csv"
                    try:
                        with open(temp_csv_for_pipeline_path, "w", newline="", encoding="utf-8") as f_csv_single:
                            writer = csv.writer(f_csv_single)
                            writer.writerow(["CompanyNumber"]) # Header expected by pipeline
                            writer.writerow([found_co_no])
                    except IOError as e_temp_csv_single:
                        st.error(f"Error creating temporary CSV for single company: {e_temp_csv_single}")
                        st.stop()
                
                if not temp_csv_for_pipeline_path or not temp_csv_for_pipeline_path.exists():
                    st.error("Input file for CH pipeline could not be prepared. Analysis cannot proceed.")
                    st.stop()
                if not api_categories_for_pipeline:
                    st.warning("Please select at least one document category for analysis.")
                    st.stop()

                # Now run the pipeline
                with st.spinner(f"Running Companies House analysis (Model: {selected_model_name}, Textract: {'Enabled' if use_textract_ocr_ch else 'Disabled'})... This may take time."):
                    output_digest_file_path, batch_metrics_from_run = run_batch_company_analysis(
                        csv_path=temp_csv_for_pipeline_path,
                        selected_categories=api_categories_for_pipeline,
                        start_year=start_year_for_pipeline,
                        end_year=end_year_for_pipeline,
                        ai_model_identifier=selected_model_name, # Pass the selected model
                        specific_ai_instructions=additional_ai_instructions_ch,
                        base_scratch_dir=ch_run_scratch_dir.parent, # Pass the parent of run-specific dir
                        keep_days=keep_temp_files_days_ch,
                        use_textract_ocr=use_textract_ocr_ch # Pass the UI choice
                    )
                    # Store results in session state
                    st.session_state.ch_last_batch_metrics = batch_metrics_from_run
                    if output_digest_file_path and output_digest_file_path.exists():
                        st.session_state.ch_last_digest_path = output_digest_file_path
                        try:
                            st.session_state.ch_last_df = pd.read_csv(output_digest_file_path)
                        except Exception as e_read_output_csv:
                            st.error(f"Error reading CH analysis digest CSV into session state: {e_read_output_csv}")
                            st.session_state.ch_last_df = None # Clear if error
                    elif batch_metrics_from_run.get("error"): # If pipeline reported an error in metrics
                         st.error(f"CH Analysis pipeline reported an error: {batch_metrics_from_run['error']}")
                         st.session_state.ch_last_df = None
                    else: # No file, no explicit error in metrics (e.g. empty input CSV led to empty digest)
                        if "No companies processed" in batch_metrics_from_run.get("notes","") :
                             st.warning("CH Analysis: No companies were processed (e.g. empty input CSV).")
                        else:
                             st.error("CH Analysis ran, but the output digest CSV was not found or is empty.")
                        st.session_state.ch_last_df = None
            st.rerun() # Rerun to display the results stored in session state

        # Display results from session state (if any) - this part runs after the rerun
        if st.session_state.ch_last_df is not None and not st.session_state.ch_last_df.empty:
            with ch_results_display_container:
                st.success("✅ Companies House Batch Analysis Complete!")
                st.markdown("---")
                st.markdown("##### Last Analysis Digest Table")
                st.dataframe(st.session_state.ch_last_df, use_container_width=True, height=250)
                
                if st.session_state.ch_last_digest_path and _pl.Path(st.session_state.ch_last_digest_path).exists():
                    try:
                        with open(st.session_state.ch_last_digest_path, "rb") as fp_digest_dl_ui:
                            st.download_button(
                                label="Download Full Digest CSV",
                                data=fp_digest_dl_ui,
                                file_name=_pl.Path(st.session_state.ch_last_digest_path).name,
                                mime="text/csv",
                                key="ch_download_digest_button_ui"
                            )
                    except Exception as e_dl_btn: st.warning(f"Download button error: {e_dl_btn}")
                
                # Display Batch Metrics, including AWS costs if Textract was used
                if st.session_state.ch_last_batch_metrics:
                    with st.expander("Batch Processing Metrics (Last Run)"):
                        metrics = st.session_state.ch_last_batch_metrics
                        st.text(f"Run Timestamp: {metrics.get('run_timestamp', 'N/A')}")
                        st.text(f"AI Model for Summaries: {metrics.get('ai_model_used_for_summaries', 'N/A')}")
                        st.text(f"Parent Companies Processed: {metrics.get('total_parent_companies_processed', 0)}")
                        st.text(f"Companies Successfully Summarized: {metrics.get('companies_successfully_summarized', 0)}")
                        st.text(f"Companies with Group Extraction Errors: {metrics.get('companies_with_extraction_errors_in_group',0)}")
                        
                        aws_costs = metrics.get("aws_ocr_costs", {})
                        if aws_costs and any(val for key, val in aws_costs.items() if key != "notes"): # Check if there are actual cost figures
                            st.markdown("###### AWS Textract OCR Cost Estimation (if used):")
                            st.text(f"  Textract Pages Processed: {metrics.get('total_textract_pages_processed',0)}")
                            st.text(f"  PDFs Sent to Textract: {metrics.get('total_pdfs_sent_to_textract',0)}")
                            for key, value in aws_costs.items():
                                if key != "notes": # Don't display the generic note here
                                    st.text(f"  {key.replace('_', ' ').title()}: {value}")
                            if "notes" in aws_costs: st.caption(f"  Cost Notes: {aws_costs['notes']}")
                        elif "notes" in aws_costs: # Only notes exist (e.g. Textract not used/available)
                             st.caption(f"OCR Cost Notes: {aws_costs['notes']}")


                st.markdown("---")
                if st.button("🧠 Generate AI Narrative Summary from Digest", key="ch_narrative_button_ui"):
                    if st.session_state.ch_last_df is not None and not st.session_state.ch_last_df.empty:
                        with st.spinner("Generating AI narrative summary..."):
                            df_for_narrative = st.session_state.ch_last_df.copy()
                            # Filter out rows where summary indicates an error or no useful content
                            if 'summary_of_findings' in df_for_narrative.columns:
                                valid_summaries_df = df_for_narrative[
                                    ~df_for_narrative['summary_of_findings'].astype(str).str.contains(
                                        "No processable|Error:|No significant text|not configured|No content provided",
                                        case=False, na=False
                                    ) & (df_for_narrative['summary_of_findings'].astype(str).str.len() > 50) # Min length for a summary
                                ]
                            else:
                                valid_summaries_df = pd.DataFrame()

                            if not valid_summaries_df.empty:
                                relevant_cols = ["parent_company_no", "summary_of_findings"]
                                existing_cols = [col for col in relevant_cols if col in valid_summaries_df.columns]
                                
                                if existing_cols:
                                    # Convert relevant parts to JSON string for the prompt
                                    data_for_prompt_str = valid_summaries_df[existing_cols].to_json(orient="records", indent=2)
                                    
                                    # Truncate if too long for the narrative model context
                                    max_narrative_input_chars = 30000 # Adjust as needed
                                    if len(data_for_prompt_str) > max_narrative_input_chars:
                                        st.warning(f"Narrative input data truncated from {len(data_for_prompt_str)} to {max_narrative_input_chars} chars.")
                                        data_for_prompt_str = data_for_prompt_str[:max_narrative_input_chars]

                                    narrative_sys_prompt = (
                                        "You are an expert financial analyst. Based on the following JSON data, which contains "
                                        "summaries of findings for several companies, write a concise 2-3 paragraph narrative. "
                                        "Your narrative should synthesize the information, highlighting common themes, significant "
                                        "findings, or any apparent red flags across the companies. Focus on the content within "
                                        "'summary_of_findings'. Avoid simply listing data; create a flowing, insightful overview."
                                    )
                                    narrative_user_prompt = f"DATA:\n{data_for_prompt_str}\n\nNARRATIVE:"
                                    
                                    narrative_text_response = "Error: Narrative generation failed."
                                    try:
                                        if selected_model_name.startswith("gpt-"):
                                            openai_client_narr = config.get_openai_client()
                                            if not openai_client_narr: raise ValueError("OpenAI client not ready for narrative.")
                                            narr_resp = openai_client_narr.chat.completions.create(
                                                model=selected_model_name, temperature=0.25, max_tokens=1000,
                                                messages=[{"role": "system", "content": narrative_sys_prompt}, {"role": "user", "content": narrative_user_prompt}]
                                            )
                                            narrative_text_response = narr_resp.choices[0].message.content.strip()
                                        elif selected_model_name.startswith("gemini-"):
                                            gemini_model_narr = config.get_gemini_model(selected_model_name)
                                            if not gemini_model_narr or not config.genai_sdk : raise ValueError("Gemini client/SDK not ready for narrative.")
                                            # Gemini expects a list of turns or simple string. We'll combine.
                                            full_narr_prompt = f"{narrative_sys_prompt}\n\n{narrative_user_prompt}"
                                            narr_resp = gemini_model_narr.generate_content(
                                                full_narr_prompt,
                                                generation_config=config.genai_sdk.types.GenerationConfig(temperature=0.25, max_output_tokens=1000)
                                            )
                                            narrative_text_response = narr_resp.text.strip()
                                        
                                        st.session_state.ch_last_narrative = narrative_text_response
                                    except Exception as e_narr_gen:
                                        st.warning(f"AI Narrative generation failed: {e_narr_gen}")
                                        st.session_state.ch_last_narrative = "Error: Narrative generation failed."
                                else: # No relevant columns in the valid_summaries_df
                                    st.info("No data with 'summary_of_findings' found in the filtered digest for narrative generation.")
                                    st.session_state.ch_last_narrative = None
                            else: # No valid summaries after filtering
                                st.info("No valid summaries found in the digest to generate an AI narrative. Ensure analysis produced meaningful findings without errors.")
                                st.session_state.ch_last_narrative = None
                        st.rerun() # Rerun to display the narrative
                    else: # No dataframe in session state
                        st.info("No CH analysis data available to generate a narrative summary.")
                        st.session_state.ch_last_narrative = None
                
                if st.session_state.ch_last_narrative:
                    st.markdown("##### AI Narrative Summary of Digest")
                    st.markdown(st.session_state.ch_last_narrative)

        elif run_ch_analysis_button_ui: # Button was pressed, but df is None (error occurred before display)
            with ch_results_display_container:
                st.warning("CH Analysis was run, but no data was generated or an error occurred. Check console/log for details if needed.")
                # Display metrics if they exist, as they might contain error info from pipeline
                if st.session_state.ch_last_batch_metrics and st.session_state.ch_last_batch_metrics.get("error"):
                    st.error(f"Pipeline Error: {st.session_state.ch_last_batch_metrics['error']}")
                elif st.session_state.ch_last_batch_metrics and st.session_state.ch_last_batch_metrics.get("notes"):
                     st.info(f"Pipeline Notes: {st.session_state.ch_last_batch_metrics['notes']}")


with tab_about:
    st.markdown("## ℹ️ About Strategic Counsel")
    st.markdown("*(Version 3.0 - Modular Refactor, Optional OCR)*")
    st.markdown("---")
    st.markdown("""
    **Strategic Counsel (SC)** is a prototype AI-powered workspace designed to augment legal and strategic professionals.
    It provides tools for document intake, summarization, contextual AI consultation, and specialized analysis of UK company filings from Companies House.
    
    *This application is for demonstration and development purposes.*
    """)
    st.subheader("Key Features & Capabilities")
    with st.expander("Modular Architecture"):
        st.markdown("The codebase has been refactored into distinct modules for configuration, API interactions, text extraction, AI utilities, and application-specific helpers. This improves maintainability and scalability.")
    with st.expander("Optional Advanced OCR"):
        st.markdown("Integration with AWS Textract for PDF OCR is now optional and can be enabled via the UI. This allows for robust text extraction from scanned or image-based PDFs when needed, while standard extraction methods are used by default.")
    with st.expander("Protocol-Driven Workflows & Contextual AI"):
        st.markdown("Utilizes a base 'protocol' (system prompt) to guide AI behavior. Injects session digests, selected memories, and document summaries for context-aware AI responses in the 'Consult Counsel' tab.")
    with st.expander("Document Intake & Intelligent Summarization"):
        st.markdown("Supports upload of PDF, DOCX, TXT files and ingestion of URLs. Documents are processed, and AI-generated summaries can be used as context.")
    with st.expander("Companies House (UK) Analysis Tool"):
        st.markdown("Fetches company filings based on selected categories and year ranges. Employs advanced text extraction and AI (OpenAI GPT or Google Gemini models) to summarize findings according to a rigorous, objective framework and user-defined instructions.")
    with st.expander("Session Management & Persistence"):
        st.markdown("Manages context per 'Matter/Topic ID'. Session interactions can be saved to a persistent 'Digest' for future reference. Document summaries and CH analysis results (CSVs) are cached or stored.")
    with st.expander("Export & Logging"):
        st.markdown("Allows exporting AI responses to DOCX. Detailed logs of AI interactions and CH analysis parameters are saved for review.")

    st.markdown("---")
    st.subheader("Technology Stack Highlights")
    st.markdown("- **Frontend:** Streamlit")
    st.markdown("- **AI Models:** OpenAI API (GPT series), Google Gemini API (Flash & Pro)")
    st.markdown("- **UK Company Data:** Companies House API")
    st.markdown("- **PDF Processing:** PyPDF2, pdfminer.six")
    st.markdown("- **Optional OCR:** AWS Textract (via Boto3)")
    st.markdown("- **Core Language & Libraries:** Python, Pandas, Requests")
    
    st.markdown("---")
    st.markdown(f"Application Base Path (from config): `{config.APP_BASE_PATH}`")
    st.markdown(f"Textract OCR Available (backend check): `{CH_PIPELINE_TEXTRACT_FLAG}`")

# --- End of Main App Area ---