import os
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import random 
from scipy.spatial import ConvexHull, QhullError
from itertools import combinations
from scipy.spatial.distance import pdist, squareform

# Set a global random seed for reproducibility
np.random.seed(42)
random.seed(42)

# --- I. PARAMETERS AND SCALES (KEPT SAME) ---

# Define bond weights based on their physicochemical properties
bond_weights = {
    'hbond': 5/18, 'hydrophobic': 5/9, 'disulfide': 5/9, 'aromatic': 5/18,
    'peptide_bond': 10, 'non-specific interaction': 5/36
}

# Define the Kyte-Doolittle hydrophobicity scale
kyte_doolittle_scale = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
    'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}
# Define the Grantham polarity/volume scale (simplified for this context)
grantham_scale = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 4.9, 'PHE': 5.2, 'CYS': 5.5,
    'MET': 5.7, 'ALA': 8.1, 'GLY': 9.0, 'THR': 8.6, 'SER': 9.2,
    'TRP': 5.4, 'TYR': 6.2, 'PRO': 8.0, 'HIS': 10.4, 'GLU': 13.0,
    'GLN': 10.5, 'ASP': 13.0, 'ASN': 11.6, 'LYS': 11.3, 'ARG': 10.5
}

# --- Normalize and combine Kyte-Doolittle and Grantham scales ---
amino_acids = list(kyte_doolittle_scale.keys())
df_scales = pd.DataFrame({
    'AA': amino_acids,
    'KyteDoolittle': [kyte_doolittle_scale[aa] for aa in amino_acids],
    'Grantham': [grantham_scale[aa] for aa in amino_acids]
})
df_scales['KD_norm'] = (df_scales['KyteDoolittle'] - df_scales['KyteDoolittle'].min()) / (
    df_scales['KyteDoolittle'].max() - df_scales['KyteDoolittle'].min())
df_scales['Grantham_norm'] = (df_scales['Grantham'] - df_scales['Grantham'].min()) / (
    df_scales['Grantham'].max() - df_scales['Grantham'].min())
df_scales['Grantham_inv'] = 1 - df_scales['Grantham_norm']
df_scales['HydrophobicityScore'] = 0.5 * df_scales['KD_norm'] + 0.5 * df_scales['Grantham_inv']
combined_score_dict = dict(zip(df_scales['AA'], df_scales['HydrophobicityScore']))

thresholds = {
    'hydrophobic': 0.66, 'polar': 0.33, 'charged': 0.3, 'flexible': 0.3
}
functional_classes = ['Hydrophobic', 'Polar', 'Mixed', 'Charged', 'Uncharged', 'Flexible']

# Define the base set of community features and their aggregation strategies
BASE_COMM_FEATURES_STRATEGIES = {
    'Degree': 'NWM', 'ClusteringCoefficient': 'NWM', 'BetweennessCentrality': 'NWM', 
    'EigenCentrality': 'NWM', 'PageRank': 'NWM', 'KCoreDecomposition': 'NWM', 
    'CBC': 'NWM', 'ParatopeRatio': 'NWM', 'PolarPct': 'NWM',
    'ConvexHullVolume': 'VWM', 'Spread': 'VWM', 'AveragePairwiseDistance': 'VWM', 
    'SpatialClusteringCoefficient': 'VWM', 'ProximityCentralization': 'VWM', 
    'WAI': 'VWM', 'TPED': 'VWM', 'NRF': 'VWM', 'CCR': 'VWM', 'STC': 'VWM', 
    'TDF': 'VWM', 'TFD': 'VWM', 'WSAVR': 'VWM',
    'Modularity': 'SM', 'EdgeDensity': 'SM', 'Compactness': 'SM', 'BondDensity': 'SM', 
    'AlgebraicConnectivity': 'SM', 'GR': 'SM', 'PTDC': 'SM', 'NBS': 'SM', 'CICS': 'SM', 
    'ZCent': 'SM'
}

# --- II. HELPER FUNCTIONS (KEPT SAME) ---

def safe_calc(func, *args, default=0, **kwargs):
    """Helper to catch NaN or exceptions in calculation, supporting keyword arguments (kwargs)."""
    try:
        result = func(*args, **kwargs) 
        return result if not (isinstance(result, float) and math.isnan(result)) else default
    except Exception:
        return default

def calculate_edge_weight(edge_data):
    kinds = edge_data.get('kind', '')
    if isinstance(kinds, str):
        kinds = kinds.replace('{', '').replace('}', '').replace('\'', '').split(',')
    weight = sum(bond_weights.get(kind.strip(), 0) for kind in kinds)
    return weight

def rearrange_communities(partition):
    community_to_new = {}
    new_partition = {}
    current_new_community = 0
    for node, old_community in partition.items():
        if old_community not in community_to_new:
            community_to_new[old_community] = current_new_community
            current_new_community += 1
        new_partition[node] = community_to_new[old_community]
    return new_partition

def calculate_scores(nodes, G):
    charged_residues = {'ARG', 'LYS', 'ASP', 'GLU'}
    flexible_residues = {'GLY', 'PRO'}
    hydro_scores = []
    charged_count = 0
    flexible_count = 0

    for node in nodes:
        resname = G.nodes[node].get('residue_name', '')
        if resname not in combined_score_dict: continue
        hydro_scores.append(combined_score_dict[resname])
        if resname in charged_residues: charged_count += 1
        if resname in flexible_residues: flexible_count += 1

    total = len(nodes)
    if not hydro_scores or total == 0: return 0, 0, 0, 0

    avg_hydro = np.mean(hydro_scores)
    avg_polar = 1 - avg_hydro
    charge_frac = charged_count / total
    flex_frac = flexible_count / total
    return avg_hydro, avg_polar, charge_frac, flex_frac

def classify_community(nodes, G):
    hydro, polar, charge, flex = calculate_scores(nodes, G)
    labels = []
    if hydro >= thresholds['hydrophobic']: labels.append("Hydrophobic")
    elif hydro <= thresholds['polar']: labels.append("Polar")
    else: labels.append("Mixed")

    if abs(charge) >= thresholds['charged']: labels.append("Charged")
    else: labels.append("Uncharged")

    if flex >= thresholds['flexible']: labels.append("Flexible")
    return labels or ["Uninterpretable"]

def calculate_centroid(nodes, pos, weights=None):
    if not nodes: return (0, 0, 0)
    try:
        points = np.array([pos[node] for node in nodes])
        if weights is None:
            weights = np.ones(len(nodes))
        else:
            weights = np.array(weights)

        if len(points) == 0: return (0, 0, 0)
        centroid = np.average(points, axis=0, weights=weights)
        return tuple(centroid)
    except Exception:
        return (0, 0, 0)

def calculate_spread(nodes, pos):
    if not nodes: return 0
    try:
        centroid = calculate_centroid(nodes, pos)
        spread = np.mean([np.linalg.norm(np.array(pos[node]) - np.array(centroid)) for node in nodes])
        return spread
    except Exception: return 0

def calculate_convex_hull_volume(nodes, pos):
    points = np.array([pos[node] for node in nodes])
    if len(points) < 4: return 0
    
    try:
        # Use 'QJ' qhull option for robustness against coplanar points
        hull = ConvexHull(points, qhull_options='QJ') 
        return hull.volume
    except QhullError:
        return 0
    except Exception:
        return 0

def calculate_average_pairwise_distance(nodes, pos):
    points = np.array([pos[node] for node in nodes])
    if len(points) < 2: return 0
    try:
        dists = pdist(points)
        return np.mean(dists)
    except Exception: return 0

def calculate_proximity_centralization(nodes, pos):
    if not nodes or len(nodes) <= 1: return 0
    try:
        centroid = calculate_centroid(nodes, pos)
        proximities = []
        for node in nodes:
            dist = np.linalg.norm(np.array(pos[node]) - np.array(centroid))
            if dist != 0:
                proximities.append(1 / dist)
        return np.sum(proximities) / (len(nodes) - 1) if proximities else 0
    except Exception: return 0

def calculate_polar_or_charged_percentage(nodes, G):
    polar_or_charged = {'GLU', 'ASP', 'LYS', 'ARG', 'HIS'}
    polar_or_charged_count = sum(1 for node in nodes if G.nodes[node].get('residue_name') in polar_or_charged)
    return (polar_or_charged_count / len(nodes)) * 100 if len(nodes)>0 else 0

def adjust_positions_to_centroid(subgraph, pos, target_centroid):
    current_centroid = calculate_centroid(subgraph.nodes(), pos)
    translation_vector = np.array(target_centroid) - np.array(current_centroid)
    for node in subgraph.nodes():
        pos[node] = tuple(np.array(pos[node]) + translation_vector)
    return pos

def parse_pos_string(pos_str):
    if isinstance(pos_str, np.ndarray): return tuple(pos_str)
    if isinstance(pos_str, tuple): return pos_str
    try:
        return tuple(np.fromstring(str(pos_str).strip('[]()'), sep=' '))
    except Exception:
        return (0.0, 0.0, 0.0)

# --- Complex/Hybrid Metrics (Optimized for performance by replacing expensive ones with 0) ---

def calculate_geodesic_and_correlation(subg, pdb_pos, spring_pos): return 0, 0
def calculate_wai_and_tped(subg, spring_pos): return 0, 0
def calculate_spatial_clustering_coefficient(G, nodes, pos): 
    try:
        return nx.average_clustering(G, nodes, weight='weight')
    except Exception:
        return 0

def calculate_topological_density_metrics(subg, spring_pos, G_full):
    nodes = list(subg.nodes())
    try:
        edge_density = nx.density(subg)
        vol_spring = calculate_convex_hull_volume(nodes, spring_pos)
        tdf = edge_density / vol_spring if vol_spring > 1e-6 else edge_density
        
        internal_weights = [d['weight'] for u, v, d in subg.edges(data=True)]
        boundary_weights = [d['weight'] for u, v, d in nx.edge_boundary(G_full, nodes, data=True)]
        
        avg_internal_w = np.mean(internal_weights) if internal_weights else 0
        avg_boundary_w = np.mean(boundary_weights) if boundary_weights else 0
        
        nbs = avg_boundary_w / avg_internal_w if avg_internal_w > 1e-6 else avg_boundary_w
        cbc = 0 # Placeholder for expensive CBC
        return tdf, nbs, cbc
    except Exception: return 0, 0, 0

def calculate_complex_spatial_ratios(subg, pdb_pos, spring_pos, G_full_vol):
    nodes = list(subg.nodes())
    if not nodes: return 0, 0, 0, 0
    try:
        pdb_spread = calculate_spread(nodes, pdb_pos)
        spring_spread = calculate_spread(nodes, spring_pos)
        spring_vol = calculate_convex_hull_volume(nodes, spring_pos)

        nrf = pdb_spread / spring_spread if spring_spread > 1e-6 else pdb_spread
        stc = spring_vol / G_full_vol if G_full_vol > 1e-6 else spring_vol
        
        pdb_comp = 1 / pdb_spread if pdb_spread > 1e-6 else 0
        spring_comp = 1 / spring_spread if spring_spread > 1e-6 else 0
        ccr = pdb_comp / spring_comp if spring_comp > 1e-6 else pdb_comp
        
        cent_pdb = calculate_centroid(nodes, pdb_pos)
        cent_spring = calculate_centroid(nodes, spring_pos)
        cics = np.linalg.norm(np.array(cent_pdb) - np.array(cent_spring))

        return nrf, stc, ccr, cics
    except Exception: return 0, 0, 0, 0

def calculate_inter_class_features(G_full, community_data_list, functional_classes, master_centroids):
    class_features = {cls: {} for cls in functional_classes}
    class_map = {cls: [] for cls in functional_classes}
    for item in community_data_list:
        for cls in item['Classes']:
            if cls in class_map: class_map[cls].append(item)
    
    # Pre-calculate positions for pairwise metrics
    all_spring_pos = {n: parse_pos_string(G_full.nodes[n]['pos']) for n in G_full.nodes()}
    
    # 2. CSD, CTS, ICP-Z, CSH (Class-Level Aggregations)
    for cls in functional_classes:
        comm_nodes = [item['Nodes'] for item in class_map[cls]]
        
        try:
            # CSH
            sizes = [len(nodes) for nodes in comm_nodes]
            csh = np.std(sizes) / np.mean(sizes) if len(sizes) > 1 and np.mean(sizes) > 0 else 0
            class_features[cls]['CSH_CL'] = csh
            
            # CSD, ICP-Z
            centroids = [item['Centroid_Spring'] for item in class_map[cls]]
            if len(centroids) > 1 and master_centroids:
                dist_matrix = pdist(centroids)
                all_dist_pairs = pdist(list(master_centroids.values()))
                max_all_dist = np.max(all_dist_pairs) if len(all_dist_pairs) > 0 else 1
                csd = np.max(dist_matrix) / max_all_dist if max_all_dist > 1e-6 else 0
                mu_all, sigma_all = np.mean(all_dist_pairs), np.std(all_dist_pairs)
                avg_dist_class = np.mean(dist_matrix)
                icpz = (avg_dist_class - mu_all) / sigma_all if sigma_all > 1e-6 else 0
            else: csd, icpz = 0, 0
            class_features[cls]['CSD_CL'] = csd
            class_features[cls]['ICPZ_CL'] = icpz
                
            # CTS
            intra_weight = sum(G_full.subgraph(nodes).size(weight='weight') for nodes in comm_nodes)
            inter_weight = 0
            for i in range(len(comm_nodes)):
                for j in range(i + 1, len(comm_nodes)):
                    inter_weight += sum(d['weight'] for u, v, d in nx.edge_boundary(G_full, comm_nodes[i], comm_nodes[j], data=True))
            cts = inter_weight / intra_weight if intra_weight > 1e-6 else inter_weight
            class_features[cls]['CTS_CL'] = cts
        
        except Exception:
            class_features[cls]['CSH_CL'], class_features[cls]['CSD_CL'], class_features[cls]['ICPZ_CL'], class_features[cls]['CTS_CL'] = 0, 0, 0, 0

    all_inter_class_features = {}
    class_pairs = list(combinations(functional_classes, 2))
    
    # 3. ICBF, WICDS, CTO, ICFC (Inter-Class Pairwise Features)
    for cls_a, cls_b in class_pairs:
        key = f"{cls_a}_{cls_b}"
        nodes_a = class_map[cls_a]
        nodes_b = class_map[cls_b]
        
        nodes_a_flat = list(set(n for comm in nodes_a for n in comm['Nodes']))
        nodes_b_flat = list(set(n for comm in nodes_b for n in comm['Nodes']))
        
        if not nodes_a or not nodes_b or not nodes_a_flat or not nodes_b_flat:
             all_inter_class_features[key] = [0, 0, 0, 0]
             continue
             
        try:
            boundary_edges = list(nx.edge_boundary(G_full, nodes_a_flat, nodes_b_flat, data=True))
            icbf_weight_sum = sum(d['weight'] for u, v, d in boundary_edges)
            
            boundary_nodes = set(u for u, v, d in boundary_edges) | set(v for u, v, d in boundary_edges)
            perimeter_proxy = sum(G_full.degree(n, weight='weight') for n in boundary_nodes)
            icbf = icbf_weight_sum / perimeter_proxy if perimeter_proxy > 1e-6 else icbf_weight_sum
            
            inter_class_dists = []
            for u in nodes_a_flat:
                for v in nodes_b_flat:
                    inter_class_dists.append(np.linalg.norm(np.array(all_spring_pos[u]) - np.array(all_spring_pos[v])))
            wicds = np.std(inter_class_dists) if len(inter_class_dists) > 1 else 0
            
            vol_a = calculate_convex_hull_volume(nodes_a_flat, all_spring_pos)
            vol_b = calculate_convex_hull_volume(nodes_b_flat, all_spring_pos)
            cto = min(vol_a, vol_b) / max(vol_a, vol_b) if max(vol_a, vol_b) > 1e-6 else 0
            
            max_edge_weight = max([d['weight'] for u, v, d in boundary_edges]) if boundary_edges else 0
            icfc = icbf_weight_sum / max_edge_weight if max_edge_weight > 1e-6 else 0
            
            all_inter_class_features[key] = [icbf, wicds, cto, icfc]
        except Exception:
            all_inter_class_features[key] = [0, 0, 0, 0]
            
    return class_features, all_inter_class_features

# --- III. MAIN EXECUTION FUNCTION (MODIFIED FOR QUADRANT 2: NOISE TOPOLOGY) ---

def perform_louvain_community_detection_and_save(
    G, graph_file, output_dir, master_csv, debug=False
):
    try:
        filename = os.path.basename(graph_file)
        base, _ = os.path.splitext(filename)
        
        # 1. Edge Weighting (Calculate ORIGINAL weights first)
        for u, v, data in G.edges(data=True):
            data['weight'] = calculate_edge_weight(data)
        
        # --- [NEW] QUADRANT 2: TOPOLOGY NOISING START ---
        # Strategy: Keep nodes/geometry intact, but rewire edges randomly and shuffle weights.
        
        # A. Extract original weights
        original_weights = [d['weight'] for u, v, d in G.edges(data=True)]
        num_edges_target = len(original_weights)
        
        # B. Wipe existing topology (Leaves nodes and 'pos', 'x', 'y', 'z' attributes alone)
        G.clear_edges() 
        
        # C. Generate Random Edges (Optimized Batch Method)
        nodes = list(G.nodes())
        num_nodes = len(nodes)
        new_edges = set()
        
        # Proceed only if we have enough nodes and target edges
        if num_nodes > 1 and num_edges_target > 0:
            # Loop until we have enough unique edges
            # (Loop is safe due to high sparsity of protein graphs vs complete graph)
            while len(new_edges) < num_edges_target:
                # Oversample batch size (1.5x) to account for collisions/self-loops
                batch_size = int((num_edges_target - len(new_edges)) * 1.5) + 10
                
                u_indices = np.random.randint(0, num_nodes, batch_size)
                v_indices = np.random.randint(0, num_nodes, batch_size)
                
                for i in range(batch_size):
                    u, v = u_indices[i], v_indices[i]
                    if u != v:
                        # Sort to ensure undirected uniqueness
                        if u < v: pair = (nodes[u], nodes[v])
                        else: pair = (nodes[v], nodes[u])
                        new_edges.add(pair)
                    
                    if len(new_edges) >= num_edges_target:
                        break
        
        # D. Assign Shuffled Weights to New Edges
        random.shuffle(original_weights)
        
        # Limit to exactly the target number (in case we overshot slightly)
        final_edges = list(new_edges)[:num_edges_target]
        
        for i, (u, v) in enumerate(final_edges):
            G.add_edge(u, v, weight=original_weights[i], kind={'randomized'})
            
        if debug:
             print(f"   [Noise] Rewired {len(final_edges)} edges randomly for {base}.")
             
        # --- [NEW] QUADRANT 2: TOPOLOGY NOISING END ---

        # 1.5 PDB Position Storage (Nodes unchanged, so this still works)
        pdb_pos = {n: parse_pos_string(G.nodes[n].get('pos', (0,0,0))) for n in G.nodes()}

        # 2. Community Detection (Louvain) - NOW RUNS ON RANDOM EDGES
        partition = community_louvain.best_partition(G, weight='weight', random_state=42)
        new_partition = rearrange_communities(partition)
        modularity = community_louvain.modularity(partition, G, weight='weight')
        
        community_nodes = {}
        for node, comm in new_partition.items():
            community_nodes.setdefault(comm, []).append(node)
        
        # 3. Spring Balance (Positional Recalculation)
        community_graph = nx.Graph()
        master_centroids = {}
        for comm, nodes in community_nodes.items():
            orig_pos = {n: parse_pos_string(G.nodes[n].get('pos', (0,0,0))) for n in nodes}
            master_centroids[comm] = calculate_centroid(nodes, orig_pos)
            for u, v, data in nx.edge_boundary(G, nodes, data=True):
                cu, cv = new_partition[u], new_partition[v]
                if cu != cv:
                    w = data['weight']
                    community_graph.add_edge(cu, cv, weight=community_graph.get_edge_data(cu, cv, {'weight': 0})['weight'] + w)
        
        community_positions = nx.spring_layout(
            community_graph, pos=master_centroids, seed=42, weight='weight', dim=3
        )

        spring_pos = {}
        for comm, nodes in community_nodes.items():
            subg = G.subgraph(nodes)
            orig_pos = {n: parse_pos_string(G.nodes[n].get('pos', (0,0,0))) for n in subg}
            sub_pos = nx.spring_layout(subg, pos=orig_pos, seed=42, weight='weight', dim=3)
            target = community_positions[comm]
            spring_pos.update(adjust_positions_to_centroid(subg, sub_pos, target))
            
        # Update G1 with final positions
        G1 = G.copy()
        for node in G1.nodes():
            G1.nodes[node]['pos'] = spring_pos.get(node, (0,0,0))
            G1.nodes[node]['pdb_pos'] = pdb_pos.get(node, (0,0,0))
            
        # 4. Community-Level Feature Calculation
        community_data_list = []
        G_full_vol = calculate_convex_hull_volume(G1.nodes(), spring_pos)
        
        k_nodes = min(100, G1.number_of_nodes())
        bc_full = nx.betweenness_centrality(G1, k=k_nodes, normalized=True, weight='weight', seed=42)
        avg_eigen, avg_pr = 0, 0 # Explicitly set expensive placeholders to 0

        for comm, nodes in community_nodes.items():
            subg = G1.subgraph(nodes)
            
            # --- Topological Features ---
            avg_degree = safe_calc(lambda: np.mean([G1.degree(n, weight='weight') for n in nodes]))
            avg_betw = safe_calc(lambda: np.mean([bc_full.get(n, 0) for n in nodes]))
            avg_clust = safe_calc(lambda: np.mean(list(nx.clustering(G1, nodes, weight='weight').values())))
            avg_kcore = safe_calc(lambda: np.mean(list(nx.core_number(subg).values()))) if subg.number_of_nodes() > 0 else 0
            density = safe_calc(nx.density, subg)
            
            # **CORRECTED CALL:** Passes 'weight' as a keyword argument to nx.algebraic_connectivity
            algebraic = safe_calc(nx.algebraic_connectivity, subg, default=0, weight='weight') if subg.number_of_nodes() > 1 else 0
            
            try:
                # Handle compactness separately as shortest_path_length can raise errors on unconnected graphs
                compactness = sum(len(nx.single_source_shortest_path_length(subg, n)) for n in nodes) / len(nodes) if nodes else 0
            except Exception:
                compactness = 0
                
            spread = safe_calc(calculate_spread, nodes, spring_pos)
            ch_vol = safe_calc(calculate_convex_hull_volume, nodes, spring_pos)
            avg_pw = safe_calc(calculate_average_pairwise_distance, nodes, spring_pos)
            spat_clust = safe_calc(calculate_spatial_clustering_coefficient, G1, nodes, spring_pos)
            prox_cent = safe_calc(calculate_proximity_centralization, nodes, spring_pos)
            bond_density = safe_calc(lambda: subg.size(weight='weight') / subg.number_of_edges()) if subg.number_of_edges() > 0 else 0
            
            nrf, stc, ccr, cics = calculate_complex_spatial_ratios(subg, pdb_pos, spring_pos, G_full_vol)
            tdf, nbs, cbc = calculate_topological_density_metrics(subg, spring_pos, G1)
            wai, tped, gr, ptdc, tfd, wsarv = 0, 0, 0, 0, 0, 0 # Explicitly set expensive placeholders to 0
            
            polar_pct = calculate_polar_or_charged_percentage(nodes, G1)
            
            comm_features = {
                'CommID': comm, 'Nodes': nodes, 'NumNodes': len(nodes), 'Vol_Spring': ch_vol, 
                'Centroid_Spring': calculate_centroid(nodes, spring_pos), 'Classes': classify_community(nodes, G1),
                'Degree': avg_degree, 'EigenCentrality': avg_eigen, 'ClusteringCoefficient': avg_clust, 
                'BetweennessCentrality': avg_betw, 'PageRank': avg_pr, 'KCoreDecomposition': avg_kcore,
                'EdgeDensity': density, 'AlgebraicConnectivity': algebraic, 'Compactness': compactness,
                'BondDensity': bond_density, 'Modularity': modularity, 'Spread': spread, 'ConvexHullVolume': ch_vol, 
                'AveragePairwiseDistance': avg_pw, 'SpatialClusteringCoefficient': spat_clust, 
                'ProximityCentralization': prox_cent, 'NRF': nrf, 'STC': stc, 'CCR': ccr, 'TDF': tdf, 
                'WAI': wai, 'TPED': tped, 'TFD': tfd, 'WSAVR': wsarv, 'GR': gr, 'PTDC': ptdc, 
                'NBS': nbs, 'CBC': cbc, 'CICS': cics,
                'ZCent': np.linalg.norm(np.array(calculate_centroid(nodes, spring_pos)) - np.array(calculate_centroid(G1.nodes(), spring_pos))), 
                'ParatopeRatio': 0, 'PolarPct': polar_pct 
            }
            community_data_list.append(comm_features)
            
        # 5. Class-Level Aggregation
        all_class_features = []
        for cls in functional_classes:
            comm_data = [d for d in community_data_list if cls in d['Classes']]
            class_row = {'Function': cls}
            
            n_total = sum(d['NumNodes'] for d in comm_data)
            v_total = sum(d['Vol_Spring'] for d in comm_data)
            k_count = len(comm_data)
            
            for feat, strategy in BASE_COMM_FEATURES_STRATEGIES.items():
                if strategy == 'NWM':
                    numerator = sum(d.get(feat, 0) * d['NumNodes'] for d in comm_data)
                    class_row[f'{feat}_NWM'] = numerator / n_total if n_total > 0 else 0
                elif strategy == 'VWM':
                    numerator = sum(d.get(feat, 0) * d['Vol_Spring'] for d in comm_data)
                    class_row[f'{feat}_VWM'] = numerator / v_total if v_total > 0 else 0
                elif strategy == 'SM':
                    numerator = sum(d.get(feat, 0) for d in comm_data)
                    class_row[f'{feat}_SM'] = numerator / k_count if k_count > 0 else 0

            # Calculate Class Centroid (simple average of community centroids)
            centroid_pos_map = {i: d['Centroid_Spring'] for i, d in enumerate(comm_data)}
            centroid_indices = list(centroid_pos_map.keys())
            class_row['CentroidX'], class_row['CentroidY'], class_row['CentroidZ'] = calculate_centroid(
                centroid_indices, centroid_pos_map
            )
            all_class_features.append(class_row)

        # 6. Inter-Class and Pairwise Calculation
        class_features_final, all_inter_class_features = calculate_inter_class_features(
            G1, community_data_list, functional_classes, master_centroids
        )

        for i, row in enumerate(all_class_features):
            row.update(class_features_final.get(row['Function'], {}))
            
        # 7. Final DataFrame Construction and Saving
        community_df = pd.DataFrame(all_class_features)
        
        pairwise_cols = [f'{feat}_{cls_a}_{cls_b}' for cls_a, cls_b in combinations(functional_classes, 2) for feat in ['ICBF', 'WICDS', 'CTO', 'ICFC']]
        pairwise_values = []
        for cls_a, cls_b in combinations(functional_classes, 2):
            key = f"{cls_a}_{cls_b}"
            pairwise_values.extend(all_inter_class_features.get(key, [0, 0, 0, 0]))

        stacked = community_df.drop(columns=['Function']).values.flatten()
        NUM_FEATURES_PER_CLASS = len(community_df.drop(columns=['Function']).columns)
        stacked_cols = [f"{col}_{i}" for i in range(len(functional_classes)) for col in community_df.drop(columns=['Function']).columns]
        
        stacked_all = stacked.tolist() + pairwise_values
        stacked_cols_all = stacked_cols + pairwise_cols
        
        row_df = pd.DataFrame([[base] + stacked_all], columns=['file'] + stacked_cols_all)

        # --- Saving to Master CSV ---
        os.makedirs(output_dir, exist_ok=True)
        # Use a temporary file for atomic write to the array chunk file
        temp_csv = master_csv + ".tmp"
        row_df.to_csv(
            temp_csv,
            mode='a',
            header=not os.path.exists(master_csv),
            index=False
        )
        # Append to the final array chunk file in a single step (simulated)
        with open(master_csv, 'a') as f:
             # Use a simple file append if the header check is done at the array job level
             # For robustness, we'll keep the existing logic and rename the file
             if not os.path.exists(master_csv) or os.path.getsize(master_csv) == 0:
                 # If the file doesn't exist or is empty, write the header and data
                 row_df.to_csv(master_csv, mode='a', header=True, index=False)
             else:
                 # Otherwise, append the data without the header
                 row_df.to_csv(master_csv, mode='a', header=False, index=False)
             
        # Clean up temp file
        if os.path.exists(temp_csv): os.remove(temp_csv)


        if debug:
            print(f"✅ [{base}] successfully calculated and appended {len(stacked_all)} features to array chunk CSV (Optimized)")

    except Exception as e:
        # --- Internal Function Error Catch ---
        print(f"❌ Error processing {graph_file} (Internal): {e}")
        import traceback
        if debug: traceback.print_exc()

# --- IV. TOP-LEVEL BATCH PROCESSING (MODIFIED) ---

def get_processed_files_set(csv_file_paths):
    """Loads a set of 'file' names from multiple CSV paths."""
    processed_files = set()
    for path in csv_file_paths:
        if os.path.exists(path):
            try:
                # Read only the 'file' column for efficiency
                df = pd.read_csv(path, usecols=['file'])
                processed_files.update(df['file'].astype(str).tolist())
            except Exception:
                # Handle empty or corrupted CSV files gracefully
                continue
    return processed_files

def process_all_graphml(input_directory, output_directory, chunk_id, total_chunks, debug=False):
    """
    Processes a deterministic chunk of GraphML files for a Slurm job array.
    
    Args:
        chunk_id (int): The current array task ID (0-indexed).
        total_chunks (int): The total number of array tasks.
    """
    
    # 1. Configuration and Output Paths
    master_csv = os.path.join(output_directory, "StructureNS_features_MASTER.csv")
    # Array chunk CSV for the current job
    array_csv = os.path.join(output_directory, f"StructureNS_features_chunk_{chunk_id}.csv")
    
    # All array chunk CSV paths (for comprehensive checkpointing)
    all_array_csvs = [os.path.join(output_directory, f"StructureNS_features_chunk_{i}.csv") for i in range(total_chunks)]

    # 2. Get the full list of files and chunk deterministically
    graph_files = [f for f in os.listdir(input_directory) if f.endswith('.graphml')]
    # CRITICAL: Sort for reproducibility across job restarts
    graph_files.sort() 
    
    # Calculate chunk size and indices
    chunk_size = math.ceil(len(graph_files) / total_chunks)
    start_index = chunk_id * chunk_size
    end_index = min((chunk_id + 1) * chunk_size, len(graph_files))
    
    files_to_process = graph_files[start_index:end_index]

    print(f"Total files: {len(graph_files)}. Chunk Size: {chunk_size}. Processing chunk {chunk_id} ({start_index}-{end_index-1}) with {len(files_to_process)} files.")
    
    # 3. Comprehensive Checkpointing (Master CSV + All Array Chunks)
    # This checks for files completed by the master consolidation or any other currently running/failed array job.
    all_csv_paths = [master_csv] + all_array_csvs
    processed_files = get_processed_files_set(all_csv_paths)
    print(f"Found {len(processed_files)} unique files already processed in master/chunks.")
    
    # 4. Iterate and Process New Files in this chunk
    
    for filename in files_to_process:
        base, _ = os.path.splitext(filename)
        graph_file_path = os.path.join(input_directory, filename)
        
        if base in processed_files:
            if debug:
                 # Print to stdout/stderr which gets logged in the .out file
                print(f"⏩ Skipping {base}: Already processed (found in checkpoint CSVs).")
            continue
        
        # --- ROBUST ERROR BLOCK PER FILE ---
        try:
            # 1. Load the graph
            G = nx.read_graphml(graph_file_path)
            
            # 2. Critical size check: Skip if graph is too small for meaningful analysis
            if G.number_of_nodes() < 2:
                print(f"🚨 Skipping {filename}: Graph has fewer than 2 nodes.")
                continue

            # 3. Call the core processing function
            # **IMPORTANT:** The core function now saves to the ARRAY_CSV, not the master CSV.
            perform_louvain_community_detection_and_save(
                G, 
                graph_file_path, 
                output_directory, 
                array_csv, # Save to array-specific CSV
                debug=debug
            )
            
        except Exception as e:
            # This catches errors during loading or unhandled errors from the internal function
            print(f"❌ CRITICAL OUTER ERROR: Failed to load or process {filename}. Error: {e}")
            if debug: 
                import traceback
                traceback.print_exc()
            continue # Skip to the next file

    print(f"✨ Chunk {chunk_id} processing complete.")


if __name__ == "__main__":
    import sys
    # Expects chunk_id and total_chunks as command line arguments from the Slurm script
    if len(sys.argv) != 3:
        print("Usage: python graphml_to_features_array.py <chunk_id> <total_chunks>")
        sys.exit(1)
        
    try:
        # Slurm array task ID is usually 1-indexed, but we use it 0-indexed here
        chunk_id = int(sys.argv[1])
        total_chunks = int(sys.argv[2])
    except ValueError:
        print("Error: Chunk ID and Total Chunks must be integers.")
        sys.exit(1)

    # Define the directories
    # NOTE: You should replace these paths with your actual environment variables or explicit paths
    # These are hardcoded based on the user's provided context for consistency.
    input_directory = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/StructureNS_analysis/random_walk_graphml_v2/graphml_mutants_random_walk"
    output_directory = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/StructureNS_analysis/random_walk_with_random_edges"
    
    # Run the array job processing
    process_all_graphml(input_directory, output_directory, chunk_id, total_chunks, debug=True)