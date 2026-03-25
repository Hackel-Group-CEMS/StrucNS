import os
import re
import random
import numpy as np
import networkx as nx
from Bio.PDB import PDBParser
from typing import Optional
from scipy.spatial.distance import cosine
import pandas as pd
import seaborn as sns 
import sys 

# --- Directory and Chunk Configuration ---
INPUT_DIRECTORY = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/omegafold_pdbs"  
OUTPUT_DIRECTORY = "graphml_mutants_v2" 
NUM_CHUNKS = 100 # Fixed number of chunks
GLOBAL_CHECKPOINT_FILE = "processed_pdbs.txt" # Using the existing file as requested
# ----------------------------------------

# --- Existing Constants and Functions (Unchanged) ---
HYDROPHOBIC_RESIS = ["ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"]
DISULFIDE_RESIS = ["CYS"]
DISULFIDE_ATOMS = ["SG"]
HBOND_ATOMS = [
    "ND", "NE", "NH", "NZ", "OD1", "OD2", "OE", "OG", "OH", "SD", "SG", "N", "O"
]
HBOND_ATOMS_SULPHUR = ["SD", "SG"]
AROMATIC_RESIS = ["PHE", "TRP", "HIS", "TYR"]
AROMATIC_RING_ATOMS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"]
}

def add_sequence_distance_edges(G: nx.Graph, d: int, name: str = "sequence_edge") -> nx.Graph:
    for chain_id in G.graph["chain_ids"]:
        chain_residues = [(n, v) for n, v in G.nodes(data=True) if v["chain_id"] == chain_id]
        for i, residue in enumerate(chain_residues):
            try:
                if i == len(chain_residues) - d:
                    continue
                cond_1 = (residue[1]["chain_id"] == chain_residues[i + d][1]["chain_id"])
                cond_2 = (abs(residue[1]["residue_number"] - chain_residues[i + d][1]["residue_number"]) == d)
                if (cond_1) and (cond_2):
                    if residue[0] != chain_residues[i + d][0]:
                        if G.has_edge(residue[0], chain_residues[i + d][0]):
                            if 'kind' not in G.edges[residue[0], chain_residues[i + d][0]]:
                                G.edges[residue[0], chain_residues[i + d][0]]['kind'] = set()
                            G.edges[residue[0], chain_residues[i + d][0]]["kind"].add(name)
                        else:
                            G.add_edge(residue[0], chain_residues[i + d][0], kind={name})
            except IndexError:
                continue
    return G

def add_peptide_bonds(G: nx.Graph) -> nx.Graph:
    return add_sequence_distance_edges(G, d=1, name="peptide_bond")

def filter_dataframe(df: pd.DataFrame, column: str, values: list, include: bool = True) -> pd.DataFrame:
    if include:
        return df[df[column].isin(values)]
    else:
        return df[~df[column].isin(values)]

def compute_distmat(df: pd.DataFrame) -> np.ndarray:
    coords = df[['x', 'y', 'z']].to_numpy()
    distmat = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
    return distmat

def get_interacting_atoms(cutoff: float, distmat: np.ndarray) -> np.ndarray:
    interacting_atoms = np.argwhere(distmat < cutoff)
    interacting_atoms = interacting_atoms[interacting_atoms[:, 0] < interacting_atoms[:, 1]]
    return interacting_atoms

def add_interacting_resis(G: nx.Graph, interacting_atoms: np.ndarray, df: pd.DataFrame, edge_types: list):
    for (i, j) in interacting_atoms:
        res1 = df.iloc[i]['node_id']
        res2 = df.iloc[j]['node_id']
        if res1 != res2:
            if G.has_edge(res1, res2):
                if 'kind' not in G.edges[res1, res2]:
                    G.edges[res1, res2]['kind'] = set()
                G.edges[res1, res2]["kind"].update(edge_types)
            else:
                G.add_edge(res1, res2, kind=set(edge_types))

def add_hydrophobic_interactions(G: nx.Graph, rgroup_df: Optional[pd.DataFrame] = None):
    if rgroup_df is None:
        rgroup_df = G.graph["rgroup_df"]
    hydrophobics_df = filter_dataframe(rgroup_df, "residue_name", HYDROPHOBIC_RESIS, True)
    hydrophobics_df = filter_dataframe(hydrophobics_df, "node_id", list(G.nodes()), True)
    distmat = compute_distmat(hydrophobics_df)
    interacting_atoms = get_interacting_atoms(5, distmat)
    add_interacting_resis(G, interacting_atoms, hydrophobics_df, ["hydrophobic"])

def add_disulfide_interactions(G: nx.Graph, rgroup_df: Optional[pd.DataFrame] = None):
    residues = [d["residue_name"] for _, d in G.nodes(data=True)]
    if residues.count("CYS") < 2:
        return

    if rgroup_df is None:
        rgroup_df = G.graph["rgroup_df"]
    disulfide_df = filter_dataframe(rgroup_df, "residue_name", DISULFIDE_RESIS, True)
    disulfide_df = filter_dataframe(disulfide_df, "atom_name", DISULFIDE_ATOMS, True)
    distmat = compute_distmat(disulfide_df)
    interacting_atoms = get_interacting_atoms(2.2, distmat)
    add_interacting_resis(G, interacting_atoms, disulfide_df, ["disulfide"])

def add_hydrogen_bond_interactions(G: nx.Graph, rgroup_df: Optional[pd.DataFrame] = None):
    if rgroup_df is None:
        rgroup_df = G.graph["rgroup_df"]
    rgroup_df = filter_dataframe(rgroup_df, "node_id", list(G.nodes()), True)

    donors = ["ND", "NE", "NH", "NZ", "OG", "OH", "SD", "SG", "N"]
    acceptors = ["OD1", "OD2", "OE", "OG", "OH", "O"]

    donor_df = filter_dataframe(rgroup_df, "atom_name", donors, True)
    acceptor_df = filter_dataframe(rgroup_df, "atom_name", acceptors, True)

    if len(donor_df.index) > 0 and len(acceptor_df.index) > 0:
        donor_coords = donor_df[['x', 'y', 'z']].to_numpy()
        acceptor_coords = acceptor_df[['x', 'y', 'z']].to_numpy()
        distmat = np.linalg.norm(donor_coords[:, np.newaxis] - acceptor_coords[np.newaxis, :], axis=2)
        interacting_pairs = np.argwhere(distmat < 3.5)
        
        for (i, j) in interacting_pairs:
            res1 = donor_df.iloc[i]['node_id']
            res2 = acceptor_df.iloc[j]['node_id']
            if res1 != res2:
                if G.has_edge(res1, res2):
                    if 'kind' not in G.edges[res1, res2]:
                        G.edges[res1, res2]['kind'] = set()
                    G.edges[res1, res2]["kind"].add("hbond")
                else:
                    G.add_edge(res1, res2, kind={"hbond"})

    hbond_df_sulphur = filter_dataframe(rgroup_df, "atom_name", HBOND_ATOMS_SULPHUR, True)
    if len(hbond_df_sulphur.index) > 0:
        distmat = compute_distmat(hbond_df_sulphur)
        interacting_atoms = get_interacting_atoms(4.0, distmat)
        add_interacting_resis(G, interacting_atoms, hbond_df_sulphur, ["hbond"])

def get_ring_atoms(pdb_df: pd.DataFrame, resi: str) -> pd.DataFrame:
    return filter_dataframe(pdb_df, "residue_name", [resi], True).query(f"atom_name in {AROMATIC_RING_ATOMS[resi]}")

def get_ring_centroids(ring_atoms_df: pd.DataFrame) -> pd.DataFrame:
    return ring_atoms_df.groupby("node_id").agg({
        "x": "mean",
        "y": "mean",
        "z": "mean",
        "residue_name": "first"
    }).reset_index()

def add_aromatic_interactions(G: nx.Graph, pdb_df: Optional[pd.DataFrame] = None):
    if pdb_df is None:
        pdb_df = G.graph["raw_pdb_df"]
    dfs = []
    for resi in AROMATIC_RESIS:
        resi_rings_df = get_ring_atoms(pdb_df, resi)
        resi_rings_df = filter_dataframe(resi_rings_df, "node_id", list(G.nodes()), True)
        resi_centroid_df = get_ring_centroids(resi_rings_df)
        dfs.append(resi_centroid_df)
    aromatic_df = pd.concat(dfs).sort_values(by="node_id").reset_index(drop=True)
    distmat = compute_distmat(aromatic_df)
    interacting_atoms = get_interacting_atoms(4.5, distmat)
    add_interacting_resis(G, interacting_atoms, aromatic_df, ["aromatic"])

def generate_network_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    G = nx.Graph()
    chain_ids = set()
    rgroup_data = []

    for model in structure:
        for chain in model:
            chain_ids.add(chain.id)
            for residue in chain:
                resname = residue.resname
                resnum = residue.id[1]
                unique_resname = f"{resname}{resnum}"
                centroid = np.mean([atom.coord for atom in residue if atom.element != 'H'], axis=0)
                G.add_node(unique_resname, pos=centroid, chain_id=chain.id, residue_number=resnum, residue_name=resname)
                for atom in residue:
                    if atom.element != 'H':
                        rgroup_data.append({
                            'node_id': unique_resname,
                            'residue_name': resname,
                            'atom_name': atom.name,
                            'x': atom.coord[0],
                            'y': atom.coord[1],
                            'z': atom.coord[2]
                        })

    G.graph["chain_ids"] = list(chain_ids)
    rgroup_df = pd.DataFrame(rgroup_data)
    G.graph["rgroup_df"] = rgroup_df
    G.graph["raw_pdb_df"] = rgroup_df

    for node1, data1 in G.nodes(data=True):
        for node2, data2 in G.nodes(data=True):
            if node1 != node2:
                distance = np.linalg.norm(data1['pos'] - data2['pos'])
                if distance < 5.0 and not G.has_edge(node1, node2):
                    G.add_edge(node1, node2, kind=set())

    G = add_peptide_bonds(G)
    add_hydrophobic_interactions(G)
    add_disulfide_interactions(G)
    add_hydrogen_bond_interactions(G)
    add_aromatic_interactions(G)
    
    return G

def convert_attributes_to_strings(G):
    for node, data in G.nodes(data=True):
        for key, value in data.items():
            if isinstance(value, (np.ndarray, list, set)):
                G.nodes[node][key] = str(value)
            else:
                G.nodes[node][key] = str(value)
    for u, v, data in G.edges(data=True):
        for key, value in data.items():
            if isinstance(value, (np.ndarray, list, set)):
                G[u][v][key] = str(value)
            else:
                G[u][v][key] = str(value)
    for key, value in G.graph.items():
        if isinstance(value, (np.ndarray, list, set, pd.DataFrame)):
            G.graph[key] = str(value)
        else:
            G.graph[key] = str(value)
    return G
# --- End of Existing Constants and Functions ---


def process_pdb_chunk(chunk_index: int, num_chunks: int, input_dir: str, output_dir: str):
    """
    Processes a specific chunk of PDB files based on the index.
    This function is run by each parallel job array task.
    """
    
    # Chunk-specific checkpoint file (used for thread-safe updates by a single job array)
    CHUNK_CHECKPOINT_FILE = f"processed_pdbs_chunk_{chunk_index}.txt" 
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load ALL PDB files and divide into reproducible chunks
    all_pdb_files = [f for f in os.listdir(input_dir) if f.endswith('.pdb')]
    all_pdb_files.sort() # IMPORTANT: Ensures reproducible chunk distribution
    
    if not all_pdb_files:
        print("No PDB files found in the input directory. Exiting.")
        return

    # Reproducibly divide the sorted files
    chunks = np.array_split(all_pdb_files, num_chunks)
    
    if chunk_index >= len(chunks):
        print(f"Chunk index {chunk_index} is out of bounds. Max index is {len(chunks)-1}. Exiting.")
        return
        
    pdb_files_in_chunk = chunks[chunk_index]
    print(f"Job Array {chunk_index} is processing {len(pdb_files_in_chunk)} files.")


    # 2. Get the list of already processed files
    processed_globally = set()
    if os.path.exists(GLOBAL_CHECKPOINT_FILE):
        try:
            with open(GLOBAL_CHECKPOINT_FILE, 'r') as f:
                # Read from the existing 'processed_pdbs.txt' file
                processed_globally.update({line.strip() for line in f if line.strip().endswith('.pdb')})
        except IOError as e:
            print(f"Warning: Could not read global checkpoint file {GLOBAL_CHECKPOINT_FILE}. Error: {e}")

    processed_in_chunk = set()
    # Check if the chunk's temporary checkpoint file exists and load it
    if os.path.exists(CHUNK_CHECKPOINT_FILE):
        try:
            with open(CHUNK_CHECKPOINT_FILE, 'r') as f:
                processed_in_chunk.update({line.strip() for line in f if line.strip().endswith('.pdb')})
        except IOError as e:
            print(f"Warning: Could not read chunk checkpoint file {CHUNK_CHECKPOINT_FILE}. Error: {e}")
            
    # Combine processed lists for checking
    all_processed = processed_globally.union(processed_in_chunk)


    # 3. Process the files in this chunk
    # The chunk-specific checkpoint file must be created/opened here for appending
    with open(CHUNK_CHECKPOINT_FILE, 'a') as chunk_checkpoint:
        for pdb_file in pdb_files_in_chunk:
            try:
                # Check if the file has already been processed by this job or any prior jobs
                if pdb_file in all_processed:
                    print(f"Skipping (found in checkpoint): {pdb_file}")
                    continue

                graphml_filename = pdb_file.replace('.pdb', '.graphml')
                output_path = os.path.join(output_dir, graphml_filename)

                # Skip if the target GraphML file already exists
                if os.path.exists(output_path):
                     print(f"Skipping (output file exists): {pdb_file}")
                     chunk_checkpoint.write(f"{pdb_file}\n") # Write to chunk checkpoint for future runs
                     continue

                print(f"Processing: {pdb_file}")
                pdb_path = os.path.join(input_dir, pdb_file)

                # Generate the graph
                G = generate_network_from_pdb(pdb_path)

                # Convert attributes to strings for GraphML compatibility
                G = convert_attributes_to_strings(G)

                # Save the graph to GraphML
                nx.write_graphml(G, output_path)

                # Write the original PDB filename to the chunk-specific checkpoint file
                chunk_checkpoint.write(f"{pdb_file}\n")
                
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")
# -----------------------------------------------------------


if __name__ == "__main__":
    if len(sys.argv) != 2:
        # This branch is now ONLY for error checking/guidance if run incorrectly
        print("Error: This script must be run with a single integer argument (the SLURM array index, 1-100).")
        print("Example: python pdb_to_graphml_mutants.py 1")
        print("Exiting.")
        sys.exit(1)

    else:
        # --- Array Processing Run (Run this with the SLURM job array) ---
        try:
            array_index = int(sys.argv[1])
            # SLURM array index is 1-based, convert to 0-based chunk index
            chunk_idx_0_based = array_index - 1 
            print(f"--- Running Job Array Index: {array_index} (Chunk {chunk_idx_0_based}) ---")
            
            # The calling statement
            process_pdb_chunk(chunk_idx_0_based, NUM_CHUNKS, INPUT_DIRECTORY, OUTPUT_DIRECTORY)
            
        except ValueError:
            print("Error: The first argument must be an integer (the SLURM array index).")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during processing: {e}")
            sys.exit(1)