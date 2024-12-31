import helpers
import openai_module
import prompts
from langflow.load import run_flow_from_json

# Define file paths
file_path_onboarding = "onboarding.docx"
file_path_transcript = "provoke_transcript.docx"

# Read the onboarding and transcript documents
onboarding_text = helpers.docx_to_string(file_path_onboarding)
transcript_text = helpers.docx_to_string(file_path_transcript)

# Update the LangFlow TWEAKS with the inputs
TWEAKS = {
    "Prompt-Q6LyS": {
        "template": """You are an expert in extracting actionable insights from meetings. I will provide a meeting transcript and an onboarding questionnaire. Your task is to carefully analyze the conversation and document, and then produce a comprehensive set of outcomes and next steps. Specifically, please only concentrate on tasks from our end around outreach, messaging, research, or event participation, being the B2B lead generation company offering the services, it should include:\n\n1.Identify the Core Objectives:\nSummarize the main goals, targets, or intentions discussed in the meeting.\n\n2.Develop Any Relevant Strategies:\nBased on the meeting’s context, propose strategies or approaches that can help achieve the identified objectives. If there was discussion about outreach, messaging, research, or event participation, outline how to execute these strategies effectively.\n\n3.Identify Potential Contacts, Resources, or References:\nSuggest any types of contacts to research or connect with (e.g., potential clients, partners, experts, common alumni, previous work contacts, etc), as well as resources (e.g., databases, websites, organizations) that could assist in completing the tasks.\n\n4. Propose Communication or Outreach Templates (If Applicable):\n\na.Provide a draft template or messaging framework if the meeting indicates a need for outreach (such as emailing potential partners or drafting proposals). Provide 10 outreach or email variations depending on the strategies outlined in the call. Explain your reasoning for each variation.\n\nb.Additionally provide the relevant filters and keywords to create the target audiences in Linkedin Sales Navigator for each target audience discussed in the call.\n\n\n5. Events and Opportunities (If Applicable):\nIf the meeting mentions or implies events, conferences, or workshops, list any potentially relevant ones, along with details such as location, timing, and why they are important. Provide strategy on how to tackle this even based on the product fit and target audience.\n\nImportant: Do not include the transcript in your final answer. Instead, use the transcript content to inform your response. If certain details are not provided, make reasonable assumptions based on standard business practices or typical approaches in similar scenarios.\n\nMake sure the output formatting follows the numbering provided in the topics, use indentation, bold, underlined, etc as needed to make sure the paragraphs are clear and easy to read.\n\nMeeting Transcript:\n{transcript}\n\nOnboarding document:\n{onboarding}\n""",
        "transcript": transcript_text,
        "onboarding": onboarding_text
    },
    "OpenAIModel-Ecrye": {
        "api_key": "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",
        "input_value": "Prompt-Q6LyS.output_text",
        "json_mode": False,
        "max_tokens": None,
        "model_kwargs": {},
        "model_name": "gpt-4o-mini",
        "openai_api_base": "",
        "output_schema": {},
        "seed": 1,
        "stream": False,
        "system_message": "",
        "temperature": 0.5
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

# Provide a simple dummy input to start the flow
dummy_input = "Hi"

# Ensure inputs are passed correctly
result = run_flow_from_json(
    flow="tranonbflow.json",
    session_id="",
    fallback_to_env_vars=True,
    input_value=dummy_input,  # Provide a simple starting value
    tweaks=TWEAKS
)

# Extract the output text from the result
if isinstance(result, list) and len(result) > 0:
    run_outputs = result[0]  # Access the first item in the result list
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

# Output handling
def handle_output(output_text):
    print("\nThe LangFlow process completed.\n")
    choice = input("Do you want to save the output to a file or display it here? (save/display): ").strip().lower()

    if choice == "save":
        save_path = input("Enter the file name to save the output (e.g., summary.docx): ").strip()
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(output_text)
            print(f"Output successfully saved to {save_path}")
        except Exception as e:
            print(f"Failed to save the output: {e}")
    elif choice == "display":
        print("\nOutput:\n")
        print(output_text)
    else:
        print("Invalid choice. Output will only be displayed here by default.")
        print("\nOutput:\n")
        print(output_text)

# Call the output handler
handle_output(output_text)
