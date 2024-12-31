from langflow.load import run_flow_from_json
import logging

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

# Prompt the user for input
input_value = input("Enter a message to send to the agent: ")

# Run the flow
try:
    result = run_flow_from_json(
        flow="bprompt.json",  # Path to your exported flow JSON file
        input_value=input_value,  # Dynamically pass the input value
        session_id="",            # Provide a session ID if needed
        fallback_to_env_vars=True,  # Use environment variables for API keys if needed
        tweaks=TWEAKS
    )

    # Extract the chat answer from the result
    if result and result[0].outputs:
        chat_answer = result[0].outputs[0].results["message"].text
        print("Chat Answer:", chat_answer)
    else:
        print("No answer received from the agent.")

except Exception as e:
    print(f"An error occurred: {e}")
