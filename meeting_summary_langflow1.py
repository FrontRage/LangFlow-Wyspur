import streamlit as st
import helpers
import openai_module
import prompts
from langflow.load import run_flow_from_json
from docx import Document
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Function to clean output text
def clean_output(output_text):
    """venv\Scripts\activate

    Clean the output text to standardize formatting and remove excessive markers.
    """
    output_text = output_text.replace("**", "")  # Remove excessive bold markers
    output_text = output_text.replace("- ", "• ")  # Standardize bullet points
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

        if st.button("Meeting Summary Tool", key="summary_tool_button"):
            st.session_state.show_summary_tool = True
            st.session_state.show_calculator = False
        if st.button("Calculator", key="calculator_button"):
            st.session_state.show_summary_tool = False
            st.session_state.show_calculator = True

    # Main content area switches based on selected button
    if st.session_state.show_summary_tool and not st.session_state.show_calculator:
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

# Run the Streamlit app
if __name__ == "__main__":
    main()
