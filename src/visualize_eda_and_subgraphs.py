import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data

# Set professional publication styling for IEEE
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

OUTPUT_DIR = "reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_figure1_temporal_distribution(csv_path="data/processed/ibm_aml_sliced_1m.csv"):
    """
    Figure 1: Temporal Transaction Volume and Laundering Occurrence over Time (EDA).
    """
    print("[*] Generating Figure 1: Temporal Transaction Distribution...")
    df = pd.read_csv(csv_path, sep=';', usecols=['Timestamp', 'Is Laundering'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    
    # Resample by 1-hour intervals
    df = df.set_index('Timestamp')
    hourly = df.resample('1h').agg(
        total_tx=('Is Laundering', 'count'),
        illicit_tx=('Is Laundering', 'sum')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=300)
    
    color_total = '#1f77b4'
    color_illicit = '#d62728'

    # Bar plot for total transactions
    ax1.bar(hourly.index, hourly['total_tx'], color=color_total, alpha=0.6, label='Normal Transactions', width=0.8)
    ax1.set_xlabel('Timeline Interval (1-Hour Timesteps)')
    ax1.set_ylabel('Total Transaction Volume', color=color_total)
    ax1.tick_params(axis='y', labelcolor=color_total)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Twin axis for illicit cases
    ax2 = ax1.twinx()
    ax2.plot(hourly.index, hourly['illicit_tx'], color=color_illicit, linewidth=2, marker='o', markersize=3, label='Illicit (Money Laundering)')
    ax2.set_ylabel('Confirmed Laundering Cases', color=color_illicit)
    ax2.tick_params(axis='y', labelcolor=color_illicit)

    plt.title('Figure 1: Temporal Transaction Volume and Illicit Activity Distribution (1M Dataset)')
    fig.tight_layout()

    out_file = os.path.join(OUTPUT_DIR, "fig1_temporal_transaction_distribution.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  [+] Saved successfully: {out_file}")

def generate_figure2_window_statistics(meta_csv="data/processed/subgraphs_metadata.csv"):
    """
    Figure 2: Subgraph Structural Statistics across the 42 Temporal Windows.
    """
    print("[*] Generating Figure 2: Temporal Subgraph Structural Statistics...")
    if not os.path.exists(meta_csv):
        print(f"  [!] Metadata not found: {meta_csv}")
        return

    df_meta = pd.read_csv(meta_csv)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=300)

    # Subplot A: Nodes & Edges
    ax1.plot(df_meta['window_id'], df_meta['num_edges'], color='#2ca02c', marker='s', markersize=4, label='Edges (Transactions)')
    ax1.plot(df_meta['window_id'], df_meta['num_nodes'], color='#1f77b4', marker='o', markersize=4, label='Nodes (Accounts)')
    ax1.set_ylabel('Count (Log Scale)')
    ax1.set_yscale('log')
    ax1.set_title('Figure 2: Micro-Subgraph Characteristics across 42 Overlapping Windows (50% Overlap)')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right')

    # Subplot B: Illicit Cases per window
    ax2.bar(df_meta['window_id'], df_meta['num_illicit'], color='#d62728', alpha=0.75, label='Laundering Transactions')
    ax2.set_xlabel('Temporal Window Index (1-Hour Duration, 30-Min Slide)')
    ax2.set_ylabel('Laundering Count')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "fig2_window_structural_statistics.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  [+] Saved successfully: {out_file}")

def generate_figure3_subgraph_topology(sample_pt="data/processed/subgraphs/subgraph_win_002.pt"):
    """
    Figure 3: Topological Network Visualization of a Sample Micro-Subgraph.
    """
    print("[*] Generating Figure 3: Sample Micro-Subgraph Network Topology...")
    if not os.path.exists(sample_pt):
        print(f"  [!] Sample graph file not found: {sample_pt}")
        return

    data = torch.load(sample_pt, weights_only=False)
    
    # Extract illicit edges and their connected nodes
    edge_index = data.edge_index.numpy()
    y_edge = data.y_edge.numpy()

    illicit_indices = np.where(y_edge == 1)[0]
    
    # We build an ego-network around the illicit nodes + a subset of background transactions
    illicit_nodes = set(edge_index[0, illicit_indices]).union(set(edge_index[1, illicit_indices]))
    
    # Filter 2-hop neighbor edges for readable visualization
    G = nx.DiGraph()
    
    # Add illicit edges
    for idx in illicit_indices:
        u, v = edge_index[0, idx], edge_index[1, idx]
        G.add_edge(u, v, is_fraud=True)

    # Sample a small background context (100 normal edges around those nodes)
    normal_indices = np.where(y_edge == 0)[0]
    context_count = 0
    for idx in normal_indices:
        u, v = edge_index[0, idx], edge_index[1, idx]
        if (u in illicit_nodes or v in illicit_nodes) and context_count < 120:
            G.add_edge(u, v, is_fraud=False)
            context_count += 1

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    pos = nx.spring_layout(G, seed=42, k=0.35)
    
    # Draw normal edges
    normal_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('is_fraud', False)]
    fraud_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('is_fraud', False)]
    
    nx.draw_networkx_nodes(G, pos, node_size=35, node_color='#1f77b4', alpha=0.7, ax=ax)
    
    # Highlight fraud nodes
    fraud_nodes_in_g = [n for n in G.nodes() if n in illicit_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=fraud_nodes_in_g, node_size=60, node_color='#d62728', ax=ax)

    # Normal edges
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='#a6a6a6', alpha=0.4, arrows=True, arrowsize=8, ax=ax)
    # Laundering edges
    nx.draw_networkx_edges(G, pos, edgelist=fraud_edges, edge_color='#d62728', width=2.0, alpha=0.9, arrows=True, arrowsize=12, ax=ax)

    ax.set_title('Figure 3: Topology of a Micro-Subgraph (Red: Circular/Illicit Flow, Gray: Normal Flow)')
    ax.axis('off')

    out_file = os.path.join(OUTPUT_DIR, "fig3_sample_subgraph_topology.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved successfully: {out_file}")

if __name__ == "__main__":
    print("=" * 65)
    print(" [DOCUMENTATION] IEEE-Standard Data & Subgraph Visualization")
    print("=" * 65)
    generate_figure1_temporal_distribution()
    generate_figure2_window_statistics()
    generate_figure3_subgraph_topology()
    print("\n[+] All IEEE figures successfully generated in 'reports/figures/'.\n")
