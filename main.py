import streamlit as st
import logging
from langflow.load import run_flow_from_json
from docx import Document
import pandas as pd
from io import StringIO
import sys

# Local modules
import helpers
import openai_module  # Still imported in case other functionality is used
from openai_module import generate_text_basic
import prompts

# ---------------------------------------------------------------------------
# Configure logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Meeting Summary & Calculator (Original "main" snippet)
# ---------------------------------------------------------------------------

# Function to clean output text
def clean_output(output_text):
    """
    Clean the output text to standardize formatting and remove excessive markers.
    """
    output_text = output_text.replace("**", "")  # Remove excessive bold markers
    output_text = output_text.replace("- ", "• ")  # Standardize bullet points
    return output_text

# ---------------------------------------------------------------------------
# CSV Filter Tool (Second snippet) - encapsulated into a function
# ---------------------------------------------------------------------------

def chunk_dataframe(df, chunk_size=500):
    """Yields successive chunks of the DataFrame of size `chunk_size`."""
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i : i + chunk_size]

def summarize_row(row, index, columns_to_summarize):
    """Produce a concise summary string for one row, including the row index."""
    col_summaries = []
    for col in columns_to_summarize:
        # row.get(col, "N/A") handles missing columns gracefully
        value = row.get(col, "N/A")
        col_summaries.append(f"{col}: {value}")
    summary = f"RowIndex: {index}, " + ", ".join(col_summaries)
    return summary

def make_chunk_summaries(chunk_df, columns_to_summarize):
    """Build a list of row summaries for the given chunk."""
    row_summaries = []
    for i, row in chunk_df.iterrows():
        row_summaries.append(summarize_row(row, i, columns_to_summarize))
    return row_summaries

def build_user_instructions(column_keywords_dict):
    """
    Build a text block telling the LLM what columns to look at and
    which keywords/concepts to exclude, using AND logic.
    """
    if not column_keywords_dict:
        return "No special exclude instructions provided."

    instructions = (
        "Exclude a row if and ONLY IF it meets ALL of the following column-based criteria.\n"
        "That means every selected column must match one of the user's exclude keywords.\n"
        "If any column does not match, KEEP the row.\n\n"
        "Here are the columns and their exclude keywords:\n"
    )

    for col, keywords in column_keywords_dict.items():
        instructions += f"- Column '{col}': exclude if it matches any of ({', '.join(keywords)})\n"

    instructions += (
        "\nRemember, it's an AND condition across columns: all must match for EXCLUDE.\n"
        "If even one column doesn't match, keep the row.\n"
    )
    return instructions

def build_conceptual_text(slider_value):
    """
    Return additional instructions depending on the conceptual reasoning strictness.
    1 = Very strict; 5 = Very broad.
    """
    if slider_value == 1:
        return """
        Level 1: Be extremely strict. 
        Only exclude rows if they explicitly match the user's keywords 
        or a very direct synonym. Slight rewordings do NOT count.
        """
    elif slider_value == 5:
        return """
        Level 5: Be very broad. 
        Exclude rows if they even loosely or thematically align 
        with the user's exclude concepts, including synonyms or tangential references. 
        Include abbreviated or spelled out references to the same concepts, e.g. VP = Vice President, 
        CE0 = Chief Executive Officer, etc.
        """
    else:
        return f"""
        Level {slider_value}: Use a moderate approach. 
        Exclude rows if they match or strongly relate to the user's exclude concepts. 
        Avoid excluding rows that are only faintly or coincidentally related.
        """

def build_llm_prompt(
    row_summaries: list,
    min_idx: int,
    max_idx: int,
    user_instructions_text: str,
    reasoning_text: str,
    debug: bool
) -> str:
    """
    Construct the LLM prompt for a single chunk.
    If debug=False, we omit references to the Reason column to save tokens.
    """
    system_instructions = f"""
    You are an expert data-cleaning assistant.

    The user wants to exclude a row ONLY if it meets ALL the specified
    column-based exclude keywords (AND-logic). If even one column does not match,
    you must KEEP the row.

    {reasoning_text}

    IMPORTANT:
    - Do not include any row index larger than {max_idx} or smaller than {min_idx}.
    - Do not include row indices that are not listed in the summaries.
    """

    if debug:
        format_instructions = """
        Return CSV with three columns (no header): 
        RowIndex,Decision,Reason

        Where:
        - RowIndex: integer
        - Decision: KEEP or EXCLUDE
        - Reason: a short explanation referencing *each column* and 
                  how it matched or did not match the user's exclude keywords.
        """
    else:
        format_instructions = """
        Return CSV with two columns (no header): 
        RowIndex,Decision

        Where:
        - RowIndex: integer
        - Decision: KEEP or EXCLUDE
        """

    prompt_for_llm = f"""
    {system_instructions}

    {format_instructions}

    [User Instructions]
    {user_instructions_text}

    [Row Summaries]
    {chr(10).join(row_summaries)}
    """
    return prompt_for_llm.strip()

def parse_llm_decisions(llm_response: str, df_indices: set, chunk_idx: int, debug: bool) -> dict:
    """
    Parse CSV from the LLM response into a dict of row_index -> (Decision, Reason).
    - If debug=False, we expect two columns: RowIndex, Decision.
    - If debug=True, we expect three columns: RowIndex, Decision, Reason.
    """
    decisions = {}
    try:
        if debug:
            response_df = pd.read_csv(StringIO(llm_response), header=None, names=["RowIndex", "Decision", "Reason"])
        else:
            response_df = pd.read_csv(StringIO(llm_response), header=None, names=["RowIndex", "Decision"])
            response_df["Reason"] = ""
    except Exception as e:
        st.error(f"Error parsing the LLM response for chunk {chunk_idx}: {e}")
        st.write("Raw response from LLM:")
        st.write(llm_response)
        sys.exit(1)

    for _, row in response_df.iterrows():
        try:
            row_index = int(row["RowIndex"])
            decision = str(row["Decision"]).strip().upper()
            reason = str(row["Reason"]) if pd.notnull(row["Reason"]) else ""

            if row_index in df_indices:
                decisions[row_index] = (decision, reason)
            else:
                st.warning(f"LLM returned out-of-range or unknown index {row_index}. Skipping...")
        except ValueError:
            st.warning(f"Invalid RowIndex '{row['RowIndex']}'. Skipping...")
            continue

    return decisions

def display_llm_debug_info(decisions: dict, max_display: int = 50):
    """
    Displays LLM debugging information (Reasons) up to max_display rows.
    Only meaningful if we actually requested a Reason in the LLM.
    """
    st.subheader("LLM Debugging Information (Reasoning)")
    display_count = 0
    for idx, (dec, rsn) in decisions.items():
        st.write(f"**Row {idx}** → **Decision**: {dec}")
        st.write(f"**Reason**: {rsn}")
        st.write("---")
        display_count += 1
        if display_count >= max_display:
            st.write(f"(Stopped after {max_display} rows for brevity...)")
            break

def filter_df_via_llm_summaries(
    df: pd.DataFrame,
    user_instructions_text: str,
    columns_to_summarize: list,
    model: str = "gpt-4o-mini",
    chunk_size: int = 500,
    conceptual_slider: int = 3,
    temperature: float = 0.0,
    top_p: float = 1.0,
    debug: bool = True,
    max_debug_display: int = 50
) -> pd.DataFrame:
    """
    Filters the DataFrame using an LLM-based conceptual exclusion approach.
    If debug=False, we won't request or parse the "Reason" column from the LLM, saving tokens.
    """

    reasoning_text = build_conceptual_text(conceptual_slider)
    decisions = {}
    total_rows = len(df)
    total_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)
    progress_bar = st.progress(0)
    current_chunk = 0

    for chunk_idx, chunk_df in enumerate(chunk_dataframe(df, chunk_size=chunk_size)):
        current_chunk += 1
        min_idx = chunk_df.index.min()
        max_idx = chunk_df.index.max()

        row_summaries = make_chunk_summaries(chunk_df, columns_to_summarize)
        prompt_for_llm = build_llm_prompt(
            row_summaries=row_summaries,
            min_idx=min_idx,
            max_idx=max_idx,
            user_instructions_text=user_instructions_text,
            reasoning_text=reasoning_text,
            debug=debug
        )

        st.write(f"Processing chunk {current_chunk} of {total_chunks}...")

        llm_response = generate_text_basic(
            prompt_for_llm,
            model=model,
            temperature=temperature,
            top_p=top_p
        )

        valid_indices_set = set(chunk_df.index)
        chunk_decisions = parse_llm_decisions(llm_response, valid_indices_set, chunk_idx, debug=debug)
        decisions.update(chunk_decisions)

        progress_fraction = current_chunk / total_chunks
        progress_bar.progress(progress_fraction)

    if debug:
        display_llm_debug_info(decisions, max_display=max_debug_display)

    keep_indices = [idx for idx, (dec, _) in decisions.items() if dec == "KEEP"]
    final_filtered_df = df.loc[keep_indices]
    return final_filtered_df

def filter_tool():
    """
    The UI logic for the CSV Filter Tool tab.
    (Originally the `main()` function in the second snippet, now encapsulated.)
    """
    st.title("LLM-Powered Conceptual CSV Filter (Detailed Reasoning)")

    uploaded_file = st.file_uploader("Upload a CSV file to filter", type=["csv"])
    if not uploaded_file:
        st.stop()

    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded CSV")
    st.write(df.head())

    st.subheader("Select Columns for Conceptual Exclusion")
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Pick one or more columns:", all_columns)

    if not selected_columns:
        st.info("No columns selected. Please select at least one to proceed.")
        st.stop()

    st.subheader("Exclude Keywords/Concepts for Each Selected Column (comma-separated)")
    column_keywords = {}
    for col in selected_columns:
        user_input = st.text_input(f"Enter exclude concepts for '{col}' (comma-separated)", key=f"kw_{col}")
        if user_input.strip():
            keywords_list = [kw.strip() for kw in user_input.split(",") if kw.strip()]
            column_keywords[col] = keywords_list

    chunk_size = st.number_input("Chunk Size for LLM Processing", value=500, min_value=1)

    conceptual_slider = st.slider(
        "Conceptual Reasoning Strictness",
        min_value=1,
        max_value=5,
        value=3,
        help="1=Very strict. 5=Very broad conceptual reasoning."
    )

    st.subheader("LLM Hyperparameters")
    temperature = st.slider("Temperature (0=deterministic, 1=creative)", 0.0, 1.0, 0.5, 0.1)
    top_p = st.slider("Top_p (0=small nucleus, 1=full distribution)", 0.0, 1.0, 1.0, 0.05)

    debug_mode = st.checkbox("Show debugging info (include 'Reason' column from LLM)?", value=True)
    max_debug_rows = st.number_input("Number of rows to display in debugging info", min_value=1, value=50)

    user_instructions_text = build_user_instructions(column_keywords)

    if st.button("Filter CSV with LLM"):
        st.write("Filtering in progress. This may take a while for large files...")

        filtered_df = filter_df_via_llm_summaries(
            df=df,
            user_instructions_text=user_instructions_text,
            columns_to_summarize=selected_columns,
            model="gpt-4o-mini",
            chunk_size=chunk_size,
            conceptual_slider=conceptual_slider,
            temperature=temperature,
            top_p=top_p,
            debug=debug_mode,
            max_debug_display=max_debug_rows
        )

        st.success(f"Filtering complete! {len(filtered_df)} rows remain out of {len(df)}.")
        st.write("Preview of filtered data:")
        st.write(filtered_df.head(50))

        csv_buffer = StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Filtered CSV",
            data=csv_buffer.getvalue(),
            file_name="filtered_output.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------------------------
# Combined Main App: meeting summary, calculator, and filter tabs
# ---------------------------------------------------------------------------

def main():
    """
    Streamlit UI for all three tools:
    - Meeting Summary Tool
    - Calculator
    - CSV Filter Tool
    """

    st.set_page_config(page_title="Wyspur Business Intelligence Dashboard", layout="wide")

    # Sidebar configuration with company logo
    with st.sidebar:
        st.image("wyspur.png", use_container_width=True)
        st.title("Wyspur Dashboard")

        # Initialize session states
        if "show_summary_tool" not in st.session_state:
            st.session_state.show_summary_tool = False
        if "show_calculator" not in st.session_state:
            st.session_state.show_calculator = False
        if "show_filter_tool" not in st.session_state:
            st.session_state.show_filter_tool = False

        # Sidebar buttons for navigation
        if st.button("Meeting Summary Tool"):
            st.session_state.show_summary_tool = True
            st.session_state.show_calculator = False
            st.session_state.show_filter_tool = False

        if st.button("Calculator"):
            st.session_state.show_summary_tool = False
            st.session_state.show_calculator = True
            st.session_state.show_filter_tool = False

        if st.button("CSV Filter Tool"):
            st.session_state.show_summary_tool = False
            st.session_state.show_calculator = False
            st.session_state.show_filter_tool = True

    # Main content area
    if st.session_state.show_summary_tool and not st.session_state.show_calculator and not st.session_state.show_filter_tool:
        st.header("Meeting Summary Tool")

        # File inputs for onboarding and transcript documents
        onboarding_file = st.file_uploader("Upload Onboarding Document", type=["docx"], key="onboarding")
        transcript_file = st.file_uploader("Upload Transcript Document", type=["docx"], key="transcript")

        if onboarding_file is None or transcript_file is None:
            st.warning("Please upload both onboarding and transcript documents to proceed.")
            return

        try:
            onboarding_text = helpers.docx_to_string(onboarding_file)
            logger.debug("Successfully read onboarding document.")
        except Exception as e:
            st.error(f"Error reading onboarding document: {e}")
            logger.error(f"Error reading onboarding document: {e}")
            onboarding_text = ""

        try:
            transcript_text = helpers.docx_to_string(transcript_file)
            logger.debug("Successfully read transcript document.")
        except Exception as e:
            st.error(f"Error reading transcript document: {e}")
            logger.error(f"Error reading transcript document: {e}")
            transcript_text = ""

        # Dynamically load prompt templates from prompts.py
        try:
            prompt_options = {
                name: getattr(prompts, name)
                for name in dir(prompts)
                if isinstance(getattr(prompts, name), str) and not name.startswith("__")
            }
            logger.debug(f"Loaded prompt templates: {list(prompt_options.keys())}")

            if "selected_prompt_name" not in st.session_state:
                st.session_state["selected_prompt_name"] = list(prompt_options.keys())[0]
                logger.debug("Initialized selected_prompt_name in session state.")

            def update_selected_prompt():
                selected_prompt_name = st.session_state["prompt_selectbox"]
                st.session_state["selected_prompt_name"] = selected_prompt_name
                logger.debug(f"Updated selected_prompt_name to: {selected_prompt_name}")

            selected_prompt_name = st.selectbox(
                "Select a Prompt Template",
                list(prompt_options.keys()),
                index=list(prompt_options.keys()).index(
                    st.session_state.get("selected_prompt_name", list(prompt_options.keys())[0])
                ),
                key="prompt_selectbox",
                on_change=update_selected_prompt
            )
            logger.debug(f"Selected prompt: {st.session_state.get('selected_prompt_name', '')}")

            # Use the selected prompt template
            selected_prompt_name = st.session_state["selected_prompt_name"]
            prompt_template = prompt_options.get(selected_prompt_name, "")
            logger.debug(f"Prompt template content for '{selected_prompt_name}': {prompt_template[:100]}...")
        except Exception as e:
            st.error(f"Error loading prompt templates: {e}")
            prompt_template = ""
            logger.error(f"Error loading prompt templates: {e}")

        # API key override input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value="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"
        )
        logger.debug(f"API Key entered: {'***' if api_key else 'Not provided'}")

        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, key="temperature_slider")
        logger.debug(f"Temperature selected: {temperature}")

        # Define TWEAKS based on user inputs
        TWEAKS = {
            "Prompt-Q6LyS": {
                "template": prompt_template,
                "transcript": transcript_text,
                "onboarding": onboarding_text
            },
            "OpenAIModel-Ecrye": {
                "api_key": api_key,
                "input_value": "Prompt-Q6LyS.output_text",
                "json_mode": False,
                "max_tokens": None,
                "model_kwargs": {},
                "model_name": "gpt-4o-mini",
                "openai_api_base": "",
                "output_schema": {},
                "seed": 1,
                "stream": False,
                "system_message": "You are an assistant that creates well-structured outputs with emojis and headings. Avoid overuse of bold formatting and ensure clarity and professionalism.",
                "temperature": temperature
            },
            "ChatOutput-FhPOl": {
                "background_color": "",
                "chat_icon": "",
                "data_template": "{text}",
                "input_value": "OpenAIModel-Ecrye.output_text",
                "sender": "Machine",
                "sender_name": "AI",
                "session_id": "",
                "should_store_message": True,
                "text_color": ""
            }
        }
        logger.debug(f"TWEAKS: {TWEAKS}")

        if "run_flow_clicked" not in st.session_state:
            st.session_state.run_flow_clicked = False

        if st.button("Run Flow") or st.session_state.run_flow_clicked:
            st.session_state.run_flow_clicked = True
            with st.spinner("Running the LangFlow process..."):
                try:
                    # Run the flow with the provided TWEAKS
                    result = run_flow_from_json(
                        flow="tranonbflow.json",
                        session_id="",
                        fallback_to_env_vars=True,
                        input_value="Hi",  # Dummy input
                        tweaks=TWEAKS
                    )
                    logger.debug(f"Flow result: {result}")

                    if isinstance(result, list) and len(result) > 0:
                        run_outputs = result[0]
                        outputs = run_outputs.outputs if hasattr(run_outputs, "outputs") else []
                        if len(outputs) > 0 and hasattr(outputs[0], "results"):
                            message_data = outputs[0].results.get("message", {}).data
                            if hasattr(message_data, "get"):
                                output_text = message_data.get("text", "No output text found")
                            else:
                                output_text = "No valid message data found in the output."
                        else:
                            output_text = "No valid outputs found in the result."
                    else:
                        output_text = "No valid result returned from LangFlow."

                    cleaned_output = clean_output(output_text)
                    st.success("Flow completed successfully!")
                    st.text_area("Output Text", value=cleaned_output, height=300)

                except Exception as e:
                    st.error(f"An error occurred while running the flow: {e}")
                    logger.error(f"Error during flow execution: {e}")

    elif st.session_state.show_calculator and not st.session_state.show_summary_tool and not st.session_state.show_filter_tool:
        st.header("Calculator")

        num1 = st.number_input("Enter first number", value=0.0, format="%.2f", key="num1_input")
        num2 = st.number_input("Enter second number", value=0.0, format="%.2f", key="num2_input")
        operation = st.selectbox("Select operation", ["Add", "Subtract", "Multiply", "Divide"], key="operation_selectbox")

        if operation == "Add":
            result = num1 + num2
        elif operation == "Subtract":
            result = num1 - num2
        elif operation == "Multiply":
            result = num1 * num2
        elif operation == "Divide":
            result = num1 / num2 if num2 != 0 else "Error: Division by zero"

        st.write(f"Result: {result}", key="result_display")

    elif st.session_state.show_filter_tool:
        # Show the CSV Filter tool UI
        filter_tool()

if __name__ == "__main__":
    main()
