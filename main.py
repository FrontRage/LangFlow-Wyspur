import streamlit as st
import helpers
import openai_module
import prompts
from langflow.load import run_flow_from_json
from docx import Document
import logging
import pandas as pd
import io
from io import StringIO
import sys

# Import your existing function that calls the LLM
# Make sure generate_text_basic can handle temperature and top_p if you're using them.
from openai_module import generate_text_basic

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Function to clean output text
def clean_output(output_text):
    """
    Clean the output text to standardize formatting and remove excessive markers.
    """
    output_text = output_text.replace("**", "")  # Remove excessive bold markers
    output_text = output_text.replace("- ", "\u2022 ")  # Standardize bullet points
    return output_text

# Define a Streamlit app for the script functionality
def main():
    """
    Streamlit UI for overriding fields in TWEAKS and running the flow.
    """

    # Set the title of the Streamlit app
    st.set_page_config(page_title="Wyspur Business Intelligence Dashboard", layout="wide")

    # Sidebar configuration with company logo
    with st.sidebar:
        st.image("wyspur.png", use_container_width=True)
        st.title("Wyspur Dashboard")

        # Sidebar buttons for navigation
        if "show_summary_tool" not in st.session_state:
            st.session_state.show_summary_tool = False
        if "show_calculator" not in st.session_state:
            st.session_state.show_calculator = False
        if "show_llm_csv_filter" not in st.session_state:
            st.session_state.show_llm_csv_filter = False

        if st.button("Meeting Summary Tool", key="summary_tool_button"):
            st.session_state.show_summary_tool = True
            st.session_state.show_calculator = False
            st.session_state.show_llm_csv_filter = False
        if st.button("Calculator", key="calculator_button"):
            st.session_state.show_summary_tool = False
            st.session_state.show_calculator = True
            st.session_state.show_llm_csv_filter = False
        if st.button("LLM-Powered Conceptual CSV Filter", key="llm_csv_filter_button"):
            st.session_state.show_summary_tool = False
            st.session_state.show_calculator = False
            st.session_state.show_llm_csv_filter = True

    # Main content area switches based on selected button
    if st.session_state.show_summary_tool and not st.session_state.show_calculator and not st.session_state.show_llm_csv_filter:
        st.header("Meeting Summary Tool")

        # File inputs for onboarding and transcript documents
        onboarding_file = st.file_uploader("Upload Onboarding Document", type=["docx"], key="onboarding")
        transcript_file = st.file_uploader("Upload Transcript Document", type=["docx"], key="transcript")

        # Ensure both files are uploaded
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

            # Ensure session state is used to store selected prompt
            if "selected_prompt_name" not in st.session_state:
                st.session_state["selected_prompt_name"] = list(prompt_options.keys())[0]
                logger.debug("Initialized selected_prompt_name in session state.")

            # Dropdown to select a prompt template
            def update_selected_prompt():
                selected_prompt_name = st.session_state["prompt_selectbox"]
                st.session_state["selected_prompt_name"] = selected_prompt_name
                logger.debug(f"Updated selected_prompt_name to: {selected_prompt_name}")

            selected_prompt_name = st.selectbox(
                "Select a Prompt Template",
                list(prompt_options.keys()),
                index=list(prompt_options.keys()).index(st.session_state.get("selected_prompt_name", list(prompt_options.keys())[0])),
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

        # Temperature slider for the model
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

        # Button to run the flow and display the output
        if "run_flow_clicked" not in st.session_state:
            st.session_state.run_flow_clicked = False

        if st.button("Run Flow", key="run_flow_button") or st.session_state.run_flow_clicked:
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

                    # Extract output text
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

                    # Clean and display the output
                    cleaned_output = clean_output(output_text)
                    st.success("Flow completed successfully!")
                    st.text_area("Output Text", value=cleaned_output, height=300)

                except Exception as e:
                    st.error(f"An error occurred while running the flow: {e}")
                    logger.error(f"Error during flow execution: {e}")

    # Calculator Tool
    elif st.session_state.show_calculator:
        st.header("Calculator")
        
        # User inputs for calculation
        num1 = st.number_input("Enter first number", value=0.0, format="%.2f", key="num1_input")
        num2 = st.number_input("Enter second number", value=0.0, format="%.2f", key="num2_input")
        operation = st.selectbox("Select operation", ["Add", "Subtract", "Multiply", "Divide"], key="operation_selectbox")

        # Perform calculation
        if operation == "Add":
            result = num1 + num2
        elif operation == "Subtract":
            result = num1 - num2
        elif operation == "Multiply":
            result = num1 * num2
        elif operation == "Divide":
            result = num1 / num2 if num2 != 0 else "Error: Division by zero"

        # Display result
        st.write(f"Result: {result}", key="result_display")

    # LLM-Powered Conceptual CSV Filter
    elif st.session_state.show_llm_csv_filter:
        st.header("LLM-Powered Conceptual CSV Filter (with Progress)")

        # 1. File uploader
        uploaded_file = st.file_uploader("Upload a CSV file to filter", type=["csv"])
        if not uploaded_file:
            st.stop()

        # 2. Read the CSV
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded CSV")
        st.write(df.head())

        # 3. Column Selection
        st.subheader("Select Columns for Conceptual Exclusion")
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Pick one or more columns:", all_columns)

        if not selected_columns:
            st.info("No columns selected. Please select at least one to proceed.")
            st.stop()

        # 4. Gather exclude 'concepts' for each selected column
        st.subheader("Exclude Keywords/Concepts for Each Selected Column (comma-separated)")
        column_keywords = {}
        for col in selected_columns:
            user_input = st.text_input(f"Enter exclude concepts for '{col}' (comma-separated)", key=f"kw_{col}")
            if user_input.strip():
                keywords_list = [kw.strip() for kw in user_input.split(",") if kw.strip()]
                column_keywords[col] = keywords_list

        # 5. Let user pick chunk size or keep default
        chunk_size = st.number_input("Chunk Size for LLM Processing", value=500, min_value=1)

        # 6. Add a slider for conceptual reasoning
        conceptual_slider = st.slider(
            "Conceptual Reasoning Strictness",
            min_value=1,
            max_value=5,
            value=3,
            help="1=Very strict. 5=Very broad conceptual reasoning."
        )

        # 7. Add sliders for temperature and top_p
        st.subheader("LLM Hyperparameters")
        temperature = st.slider("Temperature (0=deterministic, 1=creative)", 0.0, 1.0, 0.5, 0.1)
        top_p = st.slider("Top_p (0=small nucleus, 1=full distribution)", 0.0, 1.0, 1.0, 0.05)

        # Build user instructions
        def build_user_instructions(column_keywords_dict):
            """
            Build a text block that tells the LLM what columns to look at
            and which keywords/concepts to exclude if they appear (conceptually).
            """
            if not column_keywords_dict:
                return "No special exclude instructions provided."

            instructions = "Exclude a row if it conceptually aligns with one of these columns & concepts:\n"
            for col, keywords in column_keywords_dict.items():
                instructions += f"- Column: {col}, Concepts: {', '.join(keywords)}\n"
            return instructions

        user_instructions_text = build_user_instructions(column_keywords)

        # 8. Filtering Button
        if st.button("Filter CSV with LLM"):
            st.write("Filtering in progress. This may take a while for large files...")

            def chunk_dataframe(df, chunk_size=500):
                """
                Yields successive chunks of the DataFrame of size `chunk_size`.
                """
                for i in range(0, len(df), chunk_size):
                    yield df.iloc[i:i + chunk_size]

            def summarize_row(row, index, columns_to_summarize):
                """
                Produce a concise summary string for one row, using only the selected columns.
                Include the row index so we can map LLM decisions back to the original DataFrame.
                """
                col_summaries = []
                for col in columns_to_summarize:
                    value = row.get(col, "N/A")
                    col_summaries.append(f"{col}: {value}")

                summary = f"RowIndex: {index}, " + ", ".join(col_summaries)
                return summary

            def build_conceptual_text(slider_value):
                """
                Return additional instructions depending on the conceptual reasoning strictness.
                1 = Very strict; 5 = Very broad.
                """
                if slider_value == 1:
                    return """
                    Be extremely strict: only exclude rows if they explicitly match the user's keywords
                    or a very direct synonym. Slight rewordings or thematic hints do NOT count.
                    """
                elif slider_value == 5:
                    return """
                    Be very broad: exclude rows if they even loosely or thematically align with the user's
                    exclude concepts. Consider synonyms, related slang, or tangential references.
                    """
                else:
                    return f"""
                    Use a moderate approach (level {slider_value}). Exclude rows if they match or strongly
                    relate to the user's exclude concepts. Avoid excluding rows that are only faintly or
                    coincidentally related.
                    """

            def filter_df_via_llm_summaries(
                df: pd.DataFrame,
                user_instructions_text: str,
                columns_to_summarize: list,
                model: str = "gpt-4o-mini",
                chunk_size: int = 500,
                conceptual_slider: int = 3,
                temperature: float = 0.0,
                top_p: float = 1.0
            ) -> pd.DataFrame:
                """
                Filters the DataFrame using an LLM-based conceptual exclusion approach on selected columns.
                
                Steps:
                  1. Create short row summaries of only the chosen columns.
                  2. Send summaries + instructions to LLM in chunks.
                  3. LLM outputs which rows to KEEP/EXCLUDE in CSV format.
                  4. Combine partial results into a final DataFrame of KEPT rows.
                  
                The conceptual_slider adjusts how broadly the LLM interprets the exclude instructions.
                temperature and top_p are LLM hyperparameters for controlling output randomness & sampling.
                """

                decisions = {}
                reasoning_text = build_conceptual_text(conceptual_slider)

                # Calculate how many chunks we'll have, for progress tracking
                total_rows = len(df)
                total_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size != 0 else 0)

                
                # Create a progress bar in Streamlit
                progress_bar = st.progress(0)
                current_chunk = 0  # To track progress

                for chunk_idx, chunk_df in enumerate(chunk_dataframe(df, chunk_size=chunk_size)):
                    current_chunk += 1

                    # Summaries for this chunk
                    row_summaries = []
                    min_idx = chunk_df.index.min()
                    max_idx = chunk_df.index.max()

                    for i, row in chunk_df.iterrows():
                        summary = summarize_row(row, index=i, columns_to_summarize=columns_to_summarize)
                        row_summaries.append(summary)

                    system_instructions = f"""
                    You are an expert data-cleaning assistant.

                    For each row summary below, decide whether to KEEP or EXCLUDE it
                    based on the user's instructions (keywords, concepts, or themes).
                    
                    {reasoning_text}

                    IMPORTANT:
                    - Do not include any row index larger than {max_idx} or smaller than {min_idx}.
                    - Do not include row indices that are not listed in the summaries.

                    Return only two columns in CSV format:
                    RowIndex,Decision
                    (Decision is either KEEP or EXCLUDE)

                    Do not include extra commentary or text.
                    """

                    prompt_for_llm = f"""
                    {system_instructions}

                    [User Instructions]
                    {user_instructions_text}

                    [Row Summaries]
                    {chr(10).join(row_summaries)}

                    Please output a CSV (without a header) with columns: RowIndex,Decision
                    Example:
                    {min_idx},KEEP
                    {min_idx+1},EXCLUDE
                    """

                    # Show a status message each time we start a chunk
                    st.write(f"Processing chunk {current_chunk} of {total_chunks}...")

                    llm_response = generate_text_basic(
                        prompt_for_llm,
                        model=model,
                        temperature=temperature,
                        top_p=top_p
                    )

                    try:
                        response_df = pd.read_csv(StringIO(llm_response), header=None, names=["RowIndex", "Decision"])
                    except Exception as e:
                        st.error(f"Error parsing the LLM response for chunk {chunk_idx}: {e}")
                        st.write("Raw response from LLM:")
                        st.write(llm_response)
                        sys.exit(1)

                    for _, row in response_df.iterrows():
                        try:
                            row_index = int(row["RowIndex"])
                            decision = str(row["Decision"]).strip().upper()
                            if row_index in df.index:
                                decisions[row_index] = decision
                            else:
                                st.warning(f"LLM returned out-of-range index {row_index}. Skipping...")
                        except ValueError:
                            st.warning(f"Invalid RowIndex '{row['RowIndex']}'. Skipping...")
                            continue

                    # Update progress bar
                    progress_fraction = current_chunk / total_chunks
                    progress_bar.progress(progress_fraction)

                keep_indices = [idx for idx, dec in decisions.items() if dec == "KEEP"]
                final_filtered_df = df.loc[keep_indices]

                return final_filtered_df

            # The function below will update the progress bar each chunk
            filtered_df = filter_df_via_llm_summaries(
                df=df,
                user_instructions_text=user_instructions_text,
                columns_to_summarize=selected_columns,
                model="gpt-4o-mini",
                chunk_size=chunk_size,
                conceptual_slider=conceptual_slider,
                temperature=temperature,
                top_p=top_p
            )

            st.success(f"Filtering complete! {len(filtered_df)} rows remain out of {len(df)}.")
            st.write("Preview of filtered data:")
            st.write(filtered_df.head(50))

            # Download button
            csv_buffer = StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Filtered CSV",
                data=csv_buffer.getvalue(),
                file_name="filtered_output.csv",
                mime="text/csv"
            )
