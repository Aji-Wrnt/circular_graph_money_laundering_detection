import os
import time
import glob
import torch
import numpy as np
import pandas as pd
from torch_geometric.nn import GAE, SAGEConv
from torch_geometric.utils import negative_sampling

class GraphSAGEEncoder(torch.nn.Module):
    """
    Two-layer GraphSAGE Encoder for Inductive Unsupervised Subgraph Representation Learning.
    Maps local node topology and features (dim=6) to a dense latent space (dim=32).
    """
    def __init__(self, in_channels=6, hidden_channels=64, out_channels=32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.act1 = torch.nn.PReLU()
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.act1(h)
        h = self.dropout(h)
        z = self.conv2(h, edge_index)
        return z

def train_gae_on_subgraph(model, optimizer, data, device, epochs=25):
    """
    Trains the Graph Autoencoder on a single temporal micro-subgraph using unsupervised reconstruction loss.
    """
    model.train()
    x = data.x.to(device)
    pos_edge_index = data.edge_index.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(x, pos_edge_index)
        loss = model.recon_loss(z, pos_edge_index)
        loss.backward()
        optimizer.step()

    # Inference Mode: Extract Latent Embeddings & Reconstruction Anomaly Scores
    model.eval()
    with torch.no_grad():
        z = model.encode(x, pos_edge_index)
        
        # Inner-product edge probability: sigma(z_u^T * z_v)
        src = pos_edge_index[0]
        dst = pos_edge_index[1]
        z_src = z[src]
        z_dst = z[dst]

        dot_product = torch.sum(z_src * z_dst, dim=-1)
        reconstructed_prob = torch.sigmoid(dot_product)
        
        # Reconstruction Error (Anomaly Score): 1.0 - reconstructed_prob
        # For atypical / circular patterns, reconstruction error surges
        anomaly_score = (1.0 - reconstructed_prob).cpu().numpy()

        # Structural Edge Representation via Hadamard Product: z_u (*) z_v
        edge_embedding = (z_src * z_dst).cpu().numpy()

    return anomaly_score, edge_embedding

def run_unsupervised_gae_pipeline(
    subgraph_dir="data/processed/subgraphs",
    output_features_csv="data/processed/gae_extracted_features.csv",
    epochs_per_window=25,
    learning_rate=0.005,
    embedding_dim=16
):
    print("=" * 65)
    print(" [STAGE 3] UNSUPERVISED GRAPH AUTOENCODER (GAE) PIPELINE")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Compute Device : {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    subgraph_files = sorted(glob.glob(os.path.join(subgraph_dir, "subgraph_win_*.pt")))
    if not subgraph_files:
        raise FileNotFoundError(f"No subgraph files (.pt) found in: {subgraph_dir}")

    print(f"[*] Found {len(subgraph_files)} micro-subgraphs for representation learning.")
    print(f"[*] Hyperparameters: Epochs/window={epochs_per_window}, LR={learning_rate}, Latent Dim={embedding_dim}")

    # Initialize GAE with GraphSAGE encoder
    encoder = GraphSAGEEncoder(in_channels=6, hidden_channels=32, out_channels=embedding_dim).to(device)
    model = GAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    total_start_time = time.time()
    extracted_records = []

    emb_cols = [f"emb_{i}" for i in range(embedding_dim)]

    print("\n[*] Processing sequential temporal representation learning...")
    for idx, pt_file in enumerate(subgraph_files, 1):
        t_win_start = time.time()
        data = torch.load(pt_file, weights_only=False)

        # Train GAE and compute edge-level representations
        anomaly_scores, edge_embeddings = train_gae_on_subgraph(
            model=model,
            optimizer=optimizer,
            data=data,
            device=device,
            epochs=epochs_per_window
        )

        win_id = data.window_id
        tx_ids = data.edge_tx_id
        num_edges = len(tx_ids)

        # Build window DataFrame
        win_df = pd.DataFrame(edge_embeddings, columns=emb_cols)
        win_df.insert(0, 'TxID', tx_ids)
        win_df.insert(1, 'window_id', win_id)
        win_df.insert(2, 'anomaly_score', anomaly_scores)

        extracted_records.append(win_df)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed = time.time() - t_win_start

        if idx <= 5 or idx % 5 == 0 or idx == len(subgraph_files):
            mean_anomaly = float(np.mean(anomaly_scores))
            print(f"  [+] Window {win_id:02d}/{len(subgraph_files):02d} | "
                  f"Edges: {num_edges:,} | Mean Anomaly Score: {mean_anomaly:.4f} | "
                  f"Duration: {elapsed:.2f}s")

    print("\n[*] Concatenating extracted features across all temporal windows...")
    df_all_features = pd.concat(extracted_records, ignore_index=True)

    print(f"[*] Saving extracted graph features to: {output_features_csv}...")
    os.makedirs(os.path.dirname(output_features_csv), exist_ok=True)
    df_all_features.to_csv(output_features_csv, sep=';', index=False)

    total_time = time.time() - total_start_time
    file_size_mb = os.path.getsize(output_features_csv) / (1024 * 1024)

    print("\n" + "=" * 65)
    print("                    STAGE 3 SUMMARY REPORT                    ")
    print("=" * 65)
    print(f"Total Subgraphs Processed : {len(subgraph_files)}")
    print(f"Total Edge Instances      : {len(df_all_features):,} rows (including 50% overlap)")
    print(f"Unique Transactions (TxID): {df_all_features['TxID'].nunique():,}")
    print(f"Output Feature Dimension  : {df_all_features.shape[1]} columns (1 Score + {embedding_dim} Embeddings)")
    print(f"File Size                 : {file_size_mb:.2f} MB")
    print(f"Total Computation Time    : {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Target Output Path        : {output_features_csv}")
    print("=" * 65)
    print("[+] Stage 3 Completed! Ready for Stage 4: Feature Fusion & Aggregation.\n")

if __name__ == "__main__":
    run_unsupervised_gae_pipeline()
