import pandas as pd
import os
import re

# --- Configuration ---
SCORE_CSV = '/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/Tsuboyama2023_Dataset1_20230416.csv'
# Assuming these three files were successfully created in the step before last
INPUT_SET1 = 'set1_embeddings.csv'
INPUT_SET2 = 'set2_embeddings.csv'
INPUT_SET3 = 'set3_embeddings.csv'

# New output file names to avoid overwriting input files
OUTPUT_SET1 = 'set1_embeddings_with_score.csv'
OUTPUT_SET2 = 'set2_embeddings_with_score.csv'
OUTPUT_SET3 = 'set3_embeddings_with_score.csv'
# ---------------------

def create_merge_key(variant_name):
    """
    Standardizes variant names by replacing file extensions (.graphml, .pdb) 
    with an underscore to create a consistent merge key (e.g., '1A32_A45D').
    This ensures that variants from the .pdb-based score file and 
    the .pdb-based embeddings file align.
    """
    if pd.isna(variant_name):
        return None
    # Replace the file extension (.graphml or .pdb) followed by an underscore
    return re.sub(r'\.(graphml|pdb)', '_', str(variant_name), flags=re.IGNORECASE)

def merge_score_to_embeddings(input_file, output_file, df_scores):
    """Loads an embedding file, merges it with the score dataframe, and saves the result."""
    if not os.path.exists(input_file):
        print(f"Error: Input embeddings file not found at {input_file}. Skipping.")
        return 0

    try:
        # Load the segregated embeddings CSV
        df_embed = pd.read_csv(input_file)
        
        # Get the name of the first column (the variant ID column, which should be WT.pdb_mutation)
        variant_col_name = df_embed.columns[0]
        
        # 1. Create Merge Key
        df_embed['Merge_Key'] = df_embed[variant_col_name].apply(create_merge_key)

        # 2. Merge DataFrames (Left merge keeps all embedding rows)
        df_merged = pd.merge(
            df_embed, 
            df_scores, 
            on='Merge_Key', 
            how='left'
        )
        
        # 3. Clean up and Save
        df_merged = df_merged.drop(columns=['Merge_Key'])
        df_merged.to_csv(output_file, index=False)

        print(f"Successfully merged {len(df_merged)} rows from {input_file} and saved to {output_file}.")
        return len(df_merged)

    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return 0

def add_deltaG_to_embeddings():
    """Manages the loading of the score file and the sequential merging process."""
    print("--- Starting DeltaG Merge into Embeddings ---")

    # 1. Load and Prepare Master Score File
    if not os.path.exists(SCORE_CSV):
        print(f"Error: Score CSV file not found at {SCORE_CSV}. Cannot proceed.")
        return

    try:
        # Load only the 'name' and 'deltaG' columns
        df_scores = pd.read_csv(SCORE_CSV, usecols=['name', 'deltaG'])
        print(f"Score CSV loaded with {len(df_scores)} rows.")
        
        # Prepare Merge Key for the score dataframe
        df_scores['Merge_Key'] = df_scores['name'].apply(create_merge_key)
        
        # Keep only the columns needed for merging and remove duplicates
        df_scores = df_scores[['Merge_Key', 'deltaG']].drop_duplicates(subset=['Merge_Key'], keep='first')
        print(f"Prepared {len(df_scores)} unique score keys for merging.")
        
    except Exception as e:
        print(f"Error reading and preparing score CSV {SCORE_CSV}: {e}")
        return

    # 2. Process and Merge Each Set
    
    # Set 1
    merge_score_to_embeddings(INPUT_SET1, OUTPUT_SET1, df_scores)
    
    # Set 2
    merge_score_to_embeddings(INPUT_SET2, OUTPUT_SET2, df_scores)

    # Set 3
    merge_score_to_embeddings(INPUT_SET3, OUTPUT_SET3, df_scores)
    
    print("--------------------------------------------------")


if __name__ == '__main__':
    add_deltaG_to_embeddings()