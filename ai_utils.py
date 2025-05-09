# ai_utils.py

import logging
import json
import re # For the duplicated logic fix if needed for gemini_generate
from typing import Optional

# Attempt to import Google Generative AI
try:
    import google.generativeai as genai_sdk
    from google.api_core importexceptions as GoogleAPICoreExceptions # For specific Gemini errors
except ImportError:
    genai_sdk = None
    GoogleAPICoreExceptions = None


from config import (
    get_openai_client,
    get_gemini_model, # Function to get a configured model instance
    OPENAI_MODEL_DEFAULT,
    GEMINI_MODEL_DEFAULT,
    logger
)

OBJECTIVE_CH_SUMMARIZATION_BASE_PROMPT = """
You are an expert financial and legal analyst AI. Your task is to perform a rigorous and objective analysis of excerpts from UK Companies House filings.
The provided text is derived from official company documents such as Annual Accounts (AA), Confirmation Statements (CS01), director change forms (AP01, TM01), charge registrations (MR01), etc.
Extract and synthesize key factual information pertaining *only* to the content of the provided text. Adhere strictly to the data presented.

MANDATORY FOCUS AREAS:

1.  **Financial Performance & Position (Primarily from Annual Accounts):**
    * Identify and state key financial figures: Revenue/Turnover, Profit/Loss Before Tax (PBT), Operating Profit, Total Assets, Net Assets (Total Assets minus Total Liabilities or Shareholders' Funds), Total Liabilities, Current/Non-Current Assets & Liabilities breakdown if detailed, and key Cash Flow figures (Operating, Investing, Financing) if available.
    * For each figure, state the value and the corresponding financial year or period end date clearly. Use 'N/A' if a specific figure is not found.
    * Note any significant year-on-year percentage changes or absolute changes for core metrics if data for multiple periods is present and allows such comparison.
    * Mention any stated accounting policies crucial for understanding the figures, if briefly detailed (e.g., basis of consolidation, revenue recognition).

2.  **Company Structure & Capital:**
    * Describe the company's legal structure if stated (e.g., private limited, PLC). If identified as a subsidiary, note the Parent Company Name and Registration Number if provided in the text.
    * Report any changes in share capital (e.g., statement of capital in CS01), share classes, or significant share allotments/transfers if detailed.

3.  **Governance & Officers:**
    * List any director appointments or resignations/terminations, including full names and effective dates if provided.
    * Report changes in Persons with Significant Control (PSCs), including names and dates of change if mentioned.
    * Note the name of the appointed auditor and any change in auditor if specified.

4.  **Material Obligations, Risks & Commitments:**
    * List any new or existing registered charges (mortgages), including the amount secured, date registered, charge code, and persons entitled, if detailed.
    * Identify and quote or closely paraphrase any explicitly stated material risks to the business (e.g., from Strategic Report or Directors' Report), 'going concern' statements (especially if adverse or with material uncertainties highlighted by directors or auditors), or adverse audit opinions/qualifications from the Auditor's Report (e.g., disclaimer of opinion, qualified opinion, emphasis of matter related to going concern).
    * Report significant legal proceedings or contingent liabilities if disclosed.

5.  **Significant Corporate Events:**
    * Detail any mentions of mergers, acquisitions, disposals of significant assets or business segments, or major restructuring activities, including dates and parties involved if available.
    * Note any changes to the company's registered office address or name, including dates.

STYLE AND TONE REQUIREMENTS:
* Output must be objective, factual, precise, and technical. Use formal business language.
* Financial figures, names, and dates must be extracted accurately.
* Avoid speculation, inference beyond the explicit text, or subjective commentary.
* The summary should be well-organized, using bullet points for lists (e.g., directors, charges, key financial figures per year) and concise paragraphs for narrative elements (e.g., audit opinion discussion, going concern). Structure with clear headings for each focus area.

EXPLICIT EXCLUSIONS (CRITICAL - DO NOT INCLUDE THE FOLLOWING):
* DO NOT mention generic corporate social responsibility (CSR) statements, environmental initiatives ('green credentials'), community engagement, employee welfare programs, corporate culture, or general market commentary UNLESS these are explicitly linked to a material financial impact, a stated significant risk by the directors/auditors, or a major operational change clearly detailed in the filing.
* DO NOT include any marketing language, PR-style statements, or aspirational goals found in the documents.
* DO NOT infer positive or negative sentiment unless it's a direct quote or a formally stated opinion from an auditor or director regarding a material issue (e.g., an adverse audit opinion).
* Focus strictly on the verifiable, substantive information as per a formal regulatory filing. This is a technical analysis, not a public relations summary.

If additional specific instructions are provided by the user (appended below), address those *within the same rigorous and objective framework* outlined above, integrating them logically into the relevant sections.
"""

# Max characters per chunk for summarization (adjust based on typical model token limits and text density)
# ~100k chars is roughly 25k-35k tokens. Good for models with >=128k token context windows.
MAX_CHARS_PER_AI_CHUNK = 100_000
# Max tokens for the AI to generate in a single (chunk or final) summarization call
MAX_TOKENS_FOR_SUMMARY_RESPONSE = 4000 # For OpenAI
MAX_TOKENS_FOR_GEMINI_SUMMARY_RESPONSE = 8192 # For Gemini (supports larger outputs)


def _prepare_full_prompt(base_prompt: str, specific_instructions: str, company_no: str, text_chunk: str, chunk_info: str = "") -> str:
    """Helper to construct the full prompt for the AI."""
    final_system_prompt = base_prompt
    if specific_instructions and specific_instructions.strip():
        final_system_prompt += f"\n\nADDITIONAL USER-SPECIFIC INSTRUCTIONS (Apply within the objective framework above):\n{specific_instructions.strip()}"
    
    user_content = (
        f"Company Reference (for context only, data is below): {company_no} {chunk_info}\n\n"
        f"Document Text (derived from JSON, XHTML, or OCR'd PDF):\n---\n{text_chunk}\n---"
    )
    return final_system_prompt, user_content


def gpt_summarise_ch_docs(
    text_to_summarize: str,
    company_no: str,
    specific_instructions: str = "",
    model_to_use: Optional[str] = None
) -> str:
    """Summarizes text using an OpenAI GPT model, handling chunking if necessary."""
    openai_client = get_openai_client()
    if not openai_client:
        return "Error: OpenAI client not configured for summarization."
    if not text_to_summarize or not text_to_summarize.strip():
        logger.warning(f"No text provided to summarise (OpenAI) for {company_no}.")
        return "No content provided to AI for summarization."

    current_model_for_summary = model_to_use or OPENAI_MODEL_DEFAULT
    logger.info(f"{company_no}: Starting OpenAI summarization with model {current_model_for_summary}. Total text length: {len(text_to_summarize)} chars.")

    # If text is larger than a single chunk, split and summarize, then aggregate
    if len(text_to_summarize) > MAX_CHARS_PER_AI_CHUNK:
        sub_summaries = []
        num_chunks = (len(text_to_summarize) + MAX_CHARS_PER_AI_CHUNK - 1) // MAX_CHARS_PER_AI_CHUNK
        logger.info(f"{company_no}: Text exceeds {MAX_CHARS_PER_AI_CHUNK} chars, splitting into {num_chunks} chunks for OpenAI.")

        for i in range(0, len(text_to_summarize), MAX_CHARS_PER_AI_CHUNK):
            chunk = text_to_summarize[i:i + MAX_CHARS_PER_AI_CHUNK]
            chunk_label = f"(Chunk {i // MAX_CHARS_PER_AI_CHUNK + 1}/{num_chunks})"
            system_prompt, user_content = _prepare_full_prompt(
                OBJECTIVE_CH_SUMMARIZATION_BASE_PROMPT, specific_instructions, company_no, chunk, chunk_label
            )
            try:
                logger.debug(f"{company_no}: Summarizing chunk {chunk_label} ({len(chunk)} chars) with {current_model_for_summary}.")
                response = openai_client.chat.completions.create(
                    model=current_model_for_summary,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=MAX_TOKENS_FOR_SUMMARY_RESPONSE 
                )
                sub_summary = response.choices[0].message.content.strip()
                sub_summaries.append(sub_summary)
                logger.info(f"{company_no}: OpenAI sub-summary {chunk_label} complete ({len(sub_summary)} chars).")
            except Exception as e_chunk:
                error_msg = f"OpenAI chunk summarization failed for {company_no} at {chunk_label}: {e_chunk}"
                logger.error(error_msg)
                sub_summaries.append(f"Error summarizing {chunk_label}: {e_chunk}") # Add error to list to indicate failure

        # Aggregate the sub-summaries
        if not any("Error summarizing" not in s for s in sub_summaries): # All chunks failed
            return "Error: All text chunks failed during OpenAI summarization."

        logger.info(f"{company_no}: Aggregating {len(sub_summaries)} OpenAI sub-summaries.")
        # Ensure errors are clearly marked in the aggregation input
        aggregation_input_text = "\n\n---\n\n".join(
            f"Section from Chunk {idx+1}:\n{summary_text}" if "Error summarizing" not in summary_text
            else f"Section from Chunk {idx+1} (encountered an error during its summarization):\n{summary_text}"
            for idx, summary_text in enumerate(sub_summaries)
        )

        # The aggregation prompt needs to be robust
        aggregation_system_prompt = (
            f"{OBJECTIVE_CH_SUMMARIZATION_BASE_PROMPT}\n\n"
            "You are now in an AGGREGATION step. The following text consists of multiple pre-summarized sections from a larger document. "
            "Your task is to synthesize these sections into a single, coherent, and comprehensive summary that adheres to all the original "
            "formatting and content requirements. Ensure all key financial figures, governance changes, risks, and corporate events "
            "are consolidated accurately from all provided sections. If some sections indicate errors, acknowledge this and focus on valid data."
        )
        # User content for aggregation is the combined sub-summaries
        aggregation_user_content = (
             f"Company Reference (for context only, data is below): {company_no} (AGGREGATED VIEW)\n\n"
             f"Pre-summarized Document Sections:\n---\n{aggregation_input_text}\n---"
        )
        try:
            agg_response = openai_client.chat.completions.create(
                model=current_model_for_summary, # Use same model or a more powerful one for aggregation
                temperature=0.1,
                messages=[
                    {"role": "system", "content": aggregation_system_prompt},
                    {"role": "user", "content": aggregation_user_content}
                ],
                max_tokens=MAX_TOKENS_FOR_SUMMARY_RESPONSE # Allow ample space
            )
            final_summary = agg_response.choices[0].message.content.strip()
            logger.info(f"{company_no}: OpenAI aggregated summary created from {len(sub_summaries)} sub-sections ({len(final_summary)} chars).")
            return final_summary
        except Exception as e_agg:
            logger.error(f"{company_no}: OpenAI aggregation step failed: {e_agg}. Returning concatenated sub-summaries.")
            # Fallback: return concatenated summaries with a note if aggregation fails
            return f"Error during aggregation step: {e_agg}.\n\nConcatenated Sub-summaries:\n{aggregation_input_text}"
    else:
        # Single pass summarization (text is within chunk limit)
        system_prompt, user_content = _prepare_full_prompt(
            OBJECTIVE_CH_SUMMARIZATION_BASE_PROMPT, specific_instructions, company_no, text_to_summarize
        )
        try:
            logger.debug(f"{company_no}: Summarizing text ({len(text_to_summarize)} chars) in a single pass with {current_model_for_summary}.")
            response = openai_client.chat.completions.create(
                model=current_model_for_summary,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=MAX_TOKENS_FOR_SUMMARY_RESPONSE
            )
            summary_content = response.choices[0].message.content.strip()
            logger.info(f"OpenAI summary received for {company_no} (output length: {len(summary_content)} chars).")
            return summary_content
        except Exception as e_openai_api: # Catching a more general Exception from openai client
            logger.error(f"{company_no}: OpenAI API error during summarization: {e_openai_api}", exc_info=True)
            return f"Error: GPT summarization failed due to OpenAI API error: {str(e_openai_api)}"


def _gemini_generate_content_with_retry(
    gemini_model_client,
    prompt_parts: list, # Gemini API prefers a list of parts for complex prompts
    generation_config,
    company_no: str,
    context_label: str, # For logging (e.g., "chunk X", "aggregation")
    max_retries: int = 2,
    initial_delay: float = 5.0
) -> str:
    """Helper to call Gemini's generate_content with retries for specific errors."""
    if not genai_sdk or not GoogleAPICoreExceptions: # Ensure library is loaded
        return "Error: Gemini SDK or Google API Core Exceptions not available."

    for attempt in range(max_retries + 1):
        try:
            response = gemini_model_client.generate_content(
                contents=prompt_parts, # Pass as list of parts
                generation_config=generation_config
            )
            
            # Check for blocked content
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_str = genai_sdk.types.BlockedReason(response.prompt_feedback.block_reason).name
                logger.error(f"{company_no} ({context_label}): Gemini content blocked. Reason: {block_reason_str}, Details: {response.prompt_feedback.safety_ratings}")
                return f"Error: Gemini blocked content generation ({block_reason_str}). Review safety ratings/prompt."

            # Check for valid text part
            if response.parts:
                text_parts = [part.text for part in response.parts if hasattr(part, 'text')]
                if text_parts:
                    return "".join(text_parts).strip() # Join if multiple text parts returned

            # If no text and not blocked, it's an unusual response
            logger.error(f"{company_no} ({context_label}): Gemini returned empty or malformed response. Full response: {response}")
            return "Error: Gemini response did not contain usable text output and was not explicitly blocked."

        except GoogleAPICoreExceptions.ResourceExhausted as e_rate_limit: # Specific to Google API rate limits
            if attempt < max_retries:
                delay = initial_delay * (2 ** attempt) # Exponential backoff
                logger.warning(f"{company_no} ({context_label}): Gemini API rate limit hit (Attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay:.1f}s. Error: {e_rate_limit}")
                time.sleep(delay)
            else:
                logger.error(f"{company_no} ({context_label}): Gemini API rate limit hit after {max_retries + 1} attempts. Error: {e_rate_limit}")
                return f"Error: Gemini API rate limit exceeded after retries: {e_rate_limit}"
        except GoogleAPICoreExceptions.DeadlineExceeded as e_timeout: # Request timed out
             if attempt < max_retries:
                delay = initial_delay * (1.5 ** attempt) # Slightly less aggressive backoff for timeouts
                logger.warning(f"{company_no} ({context_label}): Gemini API call timed out (Attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay:.1f}s. Error: {e_timeout}")
                time.sleep(delay)
             else:
                logger.error(f"{company_no} ({context_label}): Gemini API call timed out after {max_retries + 1} attempts. Error: {e_timeout}")
                return f"Error: Gemini API call timed out after retries: {e_timeout}"
        except Exception as e: # General errors
            logger.error(f"{company_no} ({context_label}): Gemini generate_content() failed (Attempt {attempt + 1}/{max_retries + 1}): {e}", exc_info=True)
            if attempt == max_retries : # Last attempt
                 return f"Error: Gemini generate_content() failed after retries: {e}"
            time.sleep(initial_delay) # Wait before retrying general errors too
    return "Error: Gemini summarization failed after all retries (unknown reason)." # Should not be reached if logic is correct


def gemini_summarise_ch_docs(
    text_to_summarize: str,
    company_no: str,
    specific_instructions: str = "",
    model_name: Optional[str] = None
) -> str:
    """Summarizes text using a Google Gemini model, handling chunking if necessary."""
    if not genai_sdk:
        return "Error: Google Generative AI SDK (google-generativeai) not installed or available."

    current_model_name = model_name or GEMINI_MODEL_DEFAULT
    gemini_model_client = get_gemini_model(current_model_name) # Fetches an initialized model

    if not gemini_model_client:
        return f"Error: Could not initialize Gemini model '{current_model_name}'."
    if not text_to_summarize or not text_to_summarize.strip():
        logger.warning(f"No text provided to summarise (Gemini) for {company_no}.")
        return "No content provided to AI for summarization."

    logger.info(f"{company_no}: Starting Gemini summarization with model {current_model_name}. Total text length: {len(text_to_summarize)} chars.")

    generation_config = genai_sdk.types.GenerationConfig(
        temperature=0.1,
        max_output_tokens=MAX_TOKENS_FOR_GEMINI_SUMMARY_RESPONSE
        # Consider adding stop_sequences if needed, or top_p, top_k
    )

    # If text is larger than a single chunk, split and summarize, then aggregate
    if len(text_to_summarize) > MAX_CHARS_PER_AI_CHUNK:
        sub_summaries = []
        num_chunks = (len(text_to_summarize) + MAX_CHARS_PER_AI_CHUNK - 1) // MAX_CHARS_PER_AI_CHUNK
        logger.info(f"{company_no}: Text exceeds {MAX_CHARS_PER_AI_CHUNK} chars, splitting into {num_chunks} chunks for Gemini.")

        for i in range(0, len(text_to_summarize), MAX_CHARS_PER_AI_CHUNK):
            chunk = text_to_summarize[i:i + MAX_CHARS_PER_AI_CHUNK]
            chunk_label = f"Chunk {i // MAX_CHARS_PER_AI_CHUNK + 1}/{num_chunks}"
            
            # For Gemini, the "system" prompt and "user" prompt are often combined.
            # The API takes a list of 'parts' or a single string.
            # We'll construct the full prompt text here.
            system_prompt_part, user_content_part = _prepare_full_prompt(
                OBJECTIVE_CH_SUMMARIZATION_BASE_PROMPT, specific_instructions, company_no, chunk, chunk_label
            )
            full_prompt_for_chunk = f"{system_prompt_part}\n\n{user_content_part}"
            
            logger.debug(f"{company_no}: Summarizing {chunk_label} ({len(chunk)} chars) with {current_model_name}.")
            sub_summary = _gemini_generate_content_with_retry(
                gemini_model_client, [full_prompt_for_chunk], generation_config, company_no, chunk_label
            )
            sub_summaries.append(sub_summary)
            if "Error:" not in sub_summary:
                logger.info(f"{company_no}: Gemini sub-summary {chunk_label} complete ({len(sub_summary)} chars).")
            else:
                logger.error(f"{company_no}: Gemini sub-summary {chunk_label} FAILED. Error: {sub_summary}")


        # Aggregate the sub-summaries
        if not any("Error:" not in s for s in sub_summaries): # All chunks failed
            return "Error: All text chunks failed during Gemini summarization."
        
        logger.info(f"{company_no}: Aggregating {len(sub_summaries)} Gemini sub-summaries.")
        aggregation_input_text = "\n\n---\n\n".join(
            f"Section from Chunk {idx+1}:\n{summary_text}" if "Error:" not in summary_text
            else f"Section from Chunk {idx+1} (encountered an error during its summarization):\n{summary_text}"
            for idx, summary_text in enumerate(sub_summaries)
        )
        
        aggregation_system_prompt_part = (
            f"{OBJECTIVE_CH_SUMMARIZATION_BASE_PROMPT}\n\n"
            "You are now in an AGGREGATION step. The following text consists of multiple pre-summarized sections from a larger document. "
            "Your task is to synthesize these sections into a single, coherent, and comprehensive summary that adheres to all the original "
            "formatting and content requirements. Ensure all key financial figures, governance changes, risks, and corporate events "
            "are consolidated accurately from all provided sections. If some sections indicate errors, acknowledge this and focus on valid data."
        )
        aggregation_user_content_part = (
             f"Company Reference (for context only, data is below): {company_no} (AGGREGATED VIEW)\n\n"
             f"Pre-summarized Document Sections:\n---\n{aggregation_input_text}\n---"
        )
        full_prompt_for_aggregation = f"{aggregation_system_prompt_part}\n\n{aggregation_user_content_part}"

        final_summary = _gemini_generate_content_with_retry(
            gemini_model_client, [full_prompt_for_aggregation], generation_config, company_no, "Aggregation"
        )
        
        if "Error:" not in final_summary:
            logger.info(f"{company_no}: Gemini aggregated summary created from {len(sub_summaries)} sub-sections ({len(final_summary)} chars).")
        else: # Aggregation itself failed
            logger.error(f"{company_no}: Gemini aggregation step failed. Error: {final_summary}. Returning concatenated sub-summaries as fallback.")
            # Fallback: return concatenated summaries with a note if aggregation fails
            return f"Error during Gemini aggregation step: {final_summary}.\n\nConcatenated Sub-summaries:\n{aggregation_input_text}"
        return final_summary
    else:
        # Single pass summarization
        system_prompt_part, user_content_part = _prepare_full_prompt(
            OBJECTIVE_CH_SUMMARIZATION_BASE_PROMPT, specific_instructions, company_no, text_to_summarize
        )
        full_prompt_single_pass = f"{system_prompt_part}\n\n{user_content_part}"
        
        logger.debug(f"{company_no}: Summarizing text ({len(text_to_summarize)} chars) in a single pass with {current_model_name}.")
        summary_content = _gemini_generate_content_with_retry(
            gemini_model_client, [full_prompt_single_pass], generation_config, company_no, "Single Pass"
        )

        if "Error:" not in summary_content:
            logger.info(f"Gemini summary received for {company_no} (output length: {len(summary_content)} chars).")
        else:
             logger.error(f"{company_no}: Gemini single-pass summarization FAILED. Error: {summary_content}")
        return summary_content