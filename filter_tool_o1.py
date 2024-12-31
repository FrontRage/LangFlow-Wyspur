import streamlit as st
import pandas as pd
from io import StringIO
import sys

from openai_module import generate_text_basic

def chunk_dataframe(df, chunk_size=500):
    """Yields successive chunks of the DataFrame of size `chunk_size`."""
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]

def summarize_row(row, index, columns_to_summarize):
    """Produce a concise summary string for one row, including the row index."""
    col_summaries = []
    for col in columns_to_summarize:
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
    Build a text block that tells the LLM what columns to look at
    and which keywords/concepts to exclude, using AND logic.
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
        with the user's exclude concepts, including synonyms or tangential references. Include abbreviated or spelled out references to the same concepts, as an example VP = Vice President, CE0 = Chief executvie Officer, etc
        """
    else:
        # Lump 2, 3, 4 together as "moderate," or handle each separately
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
    Centralize construction of the LLM prompt for a single chunk.
    If debug=False, we omit references to the Reason column to save tokens.
    """
    # Common system instructions
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

    # If debugging is ON, we request a "Reason" column
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
    Attempt to parse CSV from the LLM response into a dictionary of row_index -> (Decision, Reason).
    - If debug=False, the CSV is expected to have only two columns: RowIndex, Decision.
    - If debug=True, we expect RowIndex, Decision, Reason.
    """
    decisions = {}

    try:
        if debug:
            # We expect three columns
            response_df = pd.read_csv(StringIO(llm_response), header=None, names=["RowIndex", "Decision", "Reason"])
        else:
            # We expect two columns
            response_df = pd.read_csv(StringIO(llm_response), header=None, names=["RowIndex", "Decision"])
            # Create a blank Reason column just so we can store in the same dict structure
            response_df["Reason"] = ""
    except Exception as e:
        st.error(f"Error parsing the LLM response for chunk {chunk_idx}: {e}")
        st.write("Raw response from LLM:")
        st.write(llm_response)
        sys.exit(1)  # or return {} if you prefer not to exit

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
    We can also limit how many debug rows are displayed.
    """

    # Build the conceptual instructions
    reasoning_text = build_conceptual_text(conceptual_slider)

    decisions = {}
    total_rows = len(df)
    total_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)

    # Setup progress bar
    progress_bar = st.progress(0)
    current_chunk = 0

    for chunk_idx, chunk_df in enumerate(chunk_dataframe(df, chunk_size=chunk_size)):
        current_chunk += 1
        min_idx = chunk_df.index.min()
        max_idx = chunk_df.index.max()

        # Build row summaries for the chunk
        row_summaries = make_chunk_summaries(chunk_df, columns_to_summarize)

        # Build the LLM prompt (conditionally includes or omits Reason instructions)
        prompt_for_llm = build_llm_prompt(
            row_summaries=row_summaries,
            min_idx=min_idx,
            max_idx=max_idx,
            user_instructions_text=user_instructions_text,
            reasoning_text=reasoning_text,
            debug=debug  # <--- we pass the debug flag here
        )

        st.write(f"Processing chunk {current_chunk} of {total_chunks}...")

        # Call your LLM
        llm_response = generate_text_basic(
            prompt_for_llm,
            model=model,
            temperature=temperature,
            top_p=top_p
        )

        # Parse decisions from LLM response (conditionally parse Reason)
        valid_indices_set = set(chunk_df.index)
        chunk_decisions = parse_llm_decisions(llm_response, valid_indices_set, chunk_idx, debug=debug)
        decisions.update(chunk_decisions)

        # Update progress bar
        progress_fraction = current_chunk / total_chunks
        progress_bar.progress(progress_fraction)

    # If debugging is turned on, show reasoning info
    if debug:
        display_llm_debug_info(decisions, max_display=max_debug_display)

    # Filter out rows that are EXCLUDE
    keep_indices = [idx for idx, (dec, _) in decisions.items() if dec == "KEEP"]
    final_filtered_df = df.loc[keep_indices]

    return final_filtered_df


def main():
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

    # -- New UI elements for debugging mode and debug rows --
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

if __name__ == "__main__":
    main()
