import streamlit as st
from langflow.load import run_flow_from_json
import logging

# Suppress Langfuse warnings
logging.getLogger("langfuse").setLevel(logging.ERROR)

# Define the tweaks dictionary
TWEAKS = {
    "ChatInput-Y2462": {
        "background_color": "",
        "chat_icon": "",
        "files": "",
        "input_value": "",  # This will be dynamically set
        "sender": "User",
        "sender_name": "User",
        "session_id": "",
        "should_store_message": True,
        "text_color": ""
    },
    "OpenAIModel-cbGR3": {
        "api_key": "",  # Leave blank since it's already set in the flow JSON file
        "input_value": "",
        "json_mode": False,
        "max_tokens": None,
        "model_kwargs": {},
        "model_name": "gpt-4o-mini",
        "openai_api_base": "",
        "output_schema": {},
        "seed": 1,
        "stream": False,
        "system_message": "",
        "temperature": 0.1
    },
    "ChatOutput-nSshV": {
        "background_color": "",
        "chat_icon": "",
        "data_template": "{text}",
        "input_value": "",
        "sender": "Machine",
        "sender_name": "AI",
        "session_id": "",
        "should_store_message": True,
        "text_color": ""
    }
}

# Streamlit App
st.title("Chat with Your Langflow Agent")

# Maintain conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input area
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", key="input_text", placeholder="Type your message here...")
    submit_button = st.form_submit_button("Send")

# Add user message to conversation and run the agent if form is submitted
if submit_button and user_input.strip():
    # Display user message in the chat
    st.session_state.messages.append({"role": "User", "content": user_input})

    try:
        # Run the Langflow agent
        result = run_flow_from_json(
            flow="bprompt.json",  # Path to your exported flow JSON file
            input_value=user_input,  # Pass the user input
            session_id="",            # Provide a session ID if needed
            fallback_to_env_vars=True,  # Use environment variables for API keys if needed
            tweaks=TWEAKS
        )

        # Extract chat answer
        if result and result[0].outputs:
            chat_answer = result[0].outputs[0].results["message"].text
        else:
            chat_answer = "No answer received from the agent."

        # Add agent response to the conversation
        st.session_state.messages.append({"role": "AI", "content": chat_answer})

    except Exception as e:
        # Handle errors
        st.session_state.messages.append({"role": "AI", "content": f"Error: {str(e)}"})

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "User":
        st.write(f"**You:** {message['content']}")
    elif message["role"] == "AI":
        st.write(f"**AI:** {message['content']}")
