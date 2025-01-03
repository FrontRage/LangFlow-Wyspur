import streamlit as st
import pandas as pd
from io import StringIO
import sys

# Import your existing function that calls the LLM
# Make sure generate_text_basic can handle temperature and top_p if you're using them.
from openai_module import generate_text_basic

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


def main():
    st.title("LLM-Powered Conceptual CSV Filter (with Progress)")

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
    user_instructions_text = build_user_instructions(column_keywords)

    # 8. Filtering Button
    if st.button("Filter CSV with LLM"):
        st.write("Filtering in progress. This may take a while for large files...")
        
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

if __name__ == "__main__":
    main()
