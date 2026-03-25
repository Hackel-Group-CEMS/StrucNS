import os
import csv
import re
from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/fasta_files_dataset1_only" # Directory containing FASTA files
OUTPUT_CSV = "unirep_input_data.csv"   # Output CSV file

# Reverse Codon Table (using one common codon for each AA for reverse translation)
# This uses the most common/simple codons where possible, otherwise it's an arbitrary choice.
# Note: UniRep features are generally calculated on the AA sequence, so the exact
# DNA sequence is often used just for record-keeping.
REV_CODON_TABLE = {
    'I': 'ATT', 'M': 'ATG', 'T': 'ACT', 'N': 'AAT',
    'K': 'AAG', 'S': 'TCT', 'R': 'CGT', 'L': 'TTG',
    'P': 'CCT', 'H': 'CAT', 'Q': 'CAG', 'V': 'GTT',
    'A': 'GCT', 'D': 'GAT', 'E': 'GAG', 'G': 'GGT',
    'F': 'TTT', 'Y': 'TAT', '_': 'TAA', 'C': 'TGT',
    'W': 'TGG', '?': 'NNN' # '?' for unknown/unmatched
}

def aa_to_dna(aa_seq):
    """
    Converts an amino acid sequence back to a possible coding DNA sequence.
    This uses the single preferred codon defined in REV_CODON_TABLE.
    """
    dna_seq = ''.join(REV_CODON_TABLE.get(aa, REV_CODON_TABLE['?']) for aa in aa_seq)
    # Add a termination codon (TAA) if the AA sequence doesn't end in '_'
    if not aa_seq.endswith('_'):
        dna_seq += 'TAA'
    return dna_seq

def read_fasta_file(file_path):
    """Reads a single FASTA file and returns the header name and sequence."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract name (header line, excluding the leading '>')
    # Use re.split to handle potential whitespace after the name
    name = re.split(r'\s+', lines[0].strip()[1:])[0] 
    
    # Extract sequence (joining all other lines and removing whitespace)
    sequence = ''.join(line.strip() for line in lines[1:])
    
    return name, sequence

def initialize_csv(filename):
    """Initializes the CSV file with headers if it doesn't exist."""
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'sequence', 'coding_dna'])

def get_existing_names(filename):
    """Reads the existing CSV and returns a set of names (headers) already processed."""
    if not os.path.exists(filename):
        return set()
    
    existing_names = set()
    try:
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f)
            # Skip header row if file is not empty
            header = next(reader, None)
            if header and header[0] == 'name':
                pass
            else:
                # If there was no header, reset the file pointer
                f.seek(0)

            for row in reader:
                if row:
                    existing_names.add(row[0])
    except Exception as e:
        print(f"Warning: Could not read existing CSV file {filename}. Error: {e}")
    return existing_names

# --- Main Processing Logic ---

def process_fasta_files():
    """
    Main function to iterate through FASTA files, check for existing entries, 
    and write new data row-by-row to the CSV.
    """
    # 1. Initialize the CSV file with headers
    initialize_csv(OUTPUT_CSV)

    # 2. Get the set of names already present in the CSV (Resilience Check)
    existing_names = get_existing_names(OUTPUT_CSV)
    print(f"Found {len(existing_names)} existing entries in {OUTPUT_CSV}. Will skip these.")

    # 3. Get all FASTA files to process
    all_files = [
        os.path.join(INPUT_DIR, f) 
        for f in os.listdir(INPUT_DIR) 
        if f.endswith('.fasta')
    ]
    
    print(f"Total FASTA files found: {len(all_files)}")

    # 4. Open the CSV file in append mode for writing new rows
    with open(OUTPUT_CSV, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # 5. Loop through files with a progress bar
        for file_path in tqdm(all_files, desc="Processing FASTA files"):
            try:
                # a. Read the fasta file
                name, aa_sequence = read_fasta_file(file_path)
                
                # b. Check if the name already exists
                if name in existing_names:
                    # Skip the file if the name is already processed
                    continue
                
                # c. Generate the coding DNA sequence
                dna_sequence = aa_to_dna(aa_sequence)
                
                # d. Write the new row immediately (memory efficient)
                writer.writerow([name, aa_sequence, dna_sequence])
                
                # e. Add the name to the set so it's checked instantly
                existing_names.add(name)
                
            except Exception as e:
                # Log any errors and continue to the next file
                print(f"\n❌ Error processing file {file_path}. Skipping. Error: {e}")
                
    print(f"\n✅ Processing complete. Data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    # Ensure the input directory exists
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found. Please create it or check the path.")
    else:
        process_fasta_files()