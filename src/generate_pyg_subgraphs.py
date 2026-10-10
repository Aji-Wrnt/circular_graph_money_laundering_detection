import os
import time
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from datetime import timedelta

def build_temporal_pyg_subgraphs(
    input_csv="data/processed/ibm_aml_sliced_1m.csv",
    output_dir="data/processed/subgraphs",
    metadata_csv="data/processed/subgraphs_metadata.csv",
    lebar_jendela_menit=60,
    geser_menit=30,
    min_transaksi=50
):
    print("=" * 65)
    print(" [TAHAP 2] GENERASI SUB-GRAF TEMPORAL PYTORCH GEOMETRIC (PyG)")
    print("           (Konfigurasi: 1 Jam Window, Geser 30 Menit - Overlap 50%)")
    print("=" * 65)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File transaksi tidak ditemukan: {input_csv}")

    # Bersihkan file subgraf lama jika ada
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.endswith(".pt"):
            os.remove(os.path.join(output_dir, f))

    print(f"[*] 1. Membaca dataset transaksi: {input_csv}...")
    t_start_total = time.time()
    df = pd.read_csv(input_csv, sep=';')
    
    # Parsing timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    df = df.sort_values(by='Timestamp').reset_index(drop=True)

    waktu_mulai = df['Timestamp'].min()
    waktu_akhir = df['Timestamp'].max()

    print(f"    - Rentang waktu: {waktu_mulai} s/d {waktu_akhir}")
    print(f"    - Total transaksi: {len(df):,} baris")
    print(f"    - Konfigurasi: Window {lebar_jendela_menit} Menit, Geser {geser_menit} Menit (Overlap 50%)")

    print("\n[*] 2. Memproses Micro-Graph Construction...")
    waktu_saat_ini = waktu_mulai
    w_len = timedelta(minutes=lebar_jendela_menit)
    w_stride = timedelta(minutes=geser_menit)
    window_id = 1
    metadata_records = []

    while waktu_saat_ini < waktu_akhir:
        batas_akhir = waktu_saat_ini + w_len
        
        # Filter transaksi dalam rentang jendela 1 jam
        mask = (df['Timestamp'] >= waktu_saat_ini) & (df['Timestamp'] < batas_akhir)
        sub = df[mask].copy()

        if len(sub) >= min_transaksi:
            t_win_start = time.time()

            # 1. Pemetaan Node Unik ke Indeks Integer Lokal (0 s/d N-1)
            nodes = pd.concat([sub['From_Node'], sub['To_Node']]).unique()
            node2idx = {node: idx for idx, node in enumerate(nodes)}
            num_nodes = len(nodes)

            src_indices = sub['From_Node'].map(node2idx).values
            dst_indices = sub['To_Node'].map(node2idx).values
            edge_index = torch.tensor(np.vstack([src_indices, dst_indices]), dtype=torch.long)

            # 2. Rekayasa Fitur Topologi Node Lokal (X in R^[N, 6])
            # In/Out degree & Balance Flow untuk mendeteksi perputaran uang lokal
            in_deg = np.bincount(dst_indices, minlength=num_nodes).astype(np.float32)
            out_deg = np.bincount(src_indices, minlength=num_nodes).astype(np.float32)

            amt = sub['Amount Paid'].values.astype(np.float32)
            amt_in = np.bincount(dst_indices, weights=amt, minlength=num_nodes)
            amt_out = np.bincount(src_indices, weights=amt, minlength=num_nodes)

            amt_in_log = np.log1p(amt_in)
            amt_out_log = np.log1p(amt_out)
            net_flow = amt_in_log - amt_out_log
            flow_ratio = amt_in_log / (amt_out_log + 1e-4)

            x_raw = np.column_stack([in_deg, out_deg, amt_in_log, amt_out_log, net_flow, flow_ratio])
            
            # Normalisasi z-score sederhana per subgraf
            x_mean = np.mean(x_raw, axis=0, keepdims=True)
            x_std = np.std(x_raw, axis=0, keepdims=True) + 1e-6
            x_norm = (x_raw - x_mean) / x_std
            x_tensor = torch.tensor(x_norm, dtype=torch.float32)

            # 3. Fitur Edge (Atribut Transaksi)
            amt_edge_log = np.log1p(sub['Amount Paid'].values)
            durasi_jendela_detik = (batas_akhir - waktu_saat_ini).total_seconds()
            rel_time = ((sub['Timestamp'] - waktu_saat_ini).dt.total_seconds() / durasi_jendela_detik).values
            edge_attr = torch.tensor(np.column_stack([amt_edge_log, rel_time]), dtype=torch.float32)

            # 4. Label dan Join Key untuk Tracking
            y_edge = torch.tensor(sub['Is Laundering'].values, dtype=torch.long)
            edge_tx_id = sub['TxID'].tolist()
            num_illicit = int(sub['Is Laundering'].sum())

            # 5. Membangun Objek PyG Data
            subgraph_data = Data(
                x=x_tensor,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y_edge=y_edge,
                num_nodes=num_nodes,
                edge_tx_id=edge_tx_id,
                window_id=window_id,
                start_time=str(waktu_saat_ini),
                end_time=str(batas_akhir)
            )

            # 6. Simpan File Objek PyG (.pt) ke Disk
            pt_filename = f"subgraph_win_{window_id:03d}.pt"
            pt_path = os.path.join(output_dir, pt_filename)
            torch.save(subgraph_data, pt_path)

            file_size_mb = os.path.getsize(pt_path) / (1024 * 1024)
            durasi_proses = time.time() - t_win_start

            if window_id <= 10 or window_id % 5 == 0:
                print(f"  [+] Window {window_id:02d} | Waktu: {waktu_saat_ini.strftime('%m-%d %H:%M')} s/d {batas_akhir.strftime('%m-%d %H:%M')} | "
                      f"Nodes: {num_nodes:,} | Edges: {len(sub):,} | Fraud: {num_illicit:2d} | Size: {file_size_mb:.1f}MB | {durasi_proses:.2f}s")

            metadata_records.append({
                'window_id': window_id,
                'start_time': str(waktu_saat_ini),
                'end_time': str(batas_akhir),
                'num_nodes': num_nodes,
                'num_edges': len(sub),
                'num_illicit': num_illicit,
                'file_name': pt_filename,
                'file_size_mb': round(file_size_mb, 2)
            })

            window_id += 1

        waktu_saat_ini += w_stride

    # 7. Simpan Ringkasan Metadata Subgraf
    df_meta = pd.DataFrame(metadata_records)
    df_meta.to_csv(metadata_csv, index=False)

    print("\n" + "=" * 65)
    print("                    RINGKASAN TAHAP 2                    ")
    print("=" * 65)
    print(f"Total Subgraf Dihasilkan   : {len(df_meta)} file (.pt)")
    print(f"Total Edges Terproses      : {df_meta['num_edges'].sum():,} edges (termasuk overlap 50%)")
    print(f"Rata-rata Node per Jendela : {df_meta['num_nodes'].mean():,.0f} nodes")
    print(f"Rata-rata Edge per Jendela : {df_meta['num_edges'].mean():,.0f} edges")
    print(f"Jendela Berisi Kasus Fraud : {(df_meta['num_illicit'] > 0).sum()} dari {len(df_meta)} jendela ({(df_meta['num_illicit'] > 0).sum()/len(df_meta)*100:.1f}%)")
    print(f"Folder Subgraf PyG         : {output_dir}")
    print(f"File Metadata              : {metadata_csv}")
    print(f"Total Waktu Komputasi      : {(time.time() - t_start_total):.2f} detik")
    print("=" * 65)
    print("[+] Tahap 2 Selesai! Sub-graf PyG siap disuapkan ke Graph Autoencoder (GAE).\n")

if __name__ == "__main__":
    build_temporal_pyg_subgraphs()
