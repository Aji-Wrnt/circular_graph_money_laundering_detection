import os
import pandas as pd

def slice_and_prepare_ibm(
    file_input="data/raw/HI-Small_Trans.csv",
    file_output="data/processed/ibm_aml_sliced_1m.csv",
    n_rows=1000000
):
    print("=" * 60)
    print(" [TAHAP 1] PEMANGKASAN TEMPORAL KONTINU (1 JUTA TRANSAKSI)")
    print("=" * 60)
    
    if not os.path.exists(file_input):
        raise FileNotFoundError(f"File mentah tidak ditemukan: {file_input}")

    print(f"[*] 1. Membaca {n_rows:,} baris temporal pertama secara kontinu...")
    df = pd.read_csv(file_input, nrows=n_rows)
    df.columns = df.columns.str.strip()

    print("[*] 2. Membersihkan duplikasi nama kolom akun...")
    if 'Account.1' in df.columns:
        df = df.rename(columns={'Account': 'From_Account', 'Account.1': 'To_Account'})
    else:
        df.columns.values[2] = 'From_Account'
        df.columns.values[4] = 'To_Account'

    print("[*] 3. Membuat Node ID Komposit (Bank_Account)...")
    df['From_Node'] = df['From Bank'].astype(str) + "_" + df['From_Account'].astype(str)
    df['To_Node'] = df['To Bank'].astype(str) + "_" + df['To_Account'].astype(str)

    print("[*] 4. Memastikan urutan kronologis berdasarkan Timestamp...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp').reset_index(drop=True)

    print("[*] 5. Menyuntikkan 'TxID' unik sebagai Join Key...")
    df['TxID'] = [f"TX_IBM_{i:07d}" for i in range(len(df))]

    os.makedirs(os.path.dirname(file_output), exist_ok=True)

    print(f"[*] 6. Menyimpan data bersih ke: {file_output}")
    df.to_csv(file_output, sep=';', index=False)

    print("\n" + "=" * 60)
    print("                 RINGKASAN HASIL PEMANGKASAN                ")
    print("=" * 60)
    print(f"Total Baris Dipangkas     : {len(df):,} baris")
    print(f"Rentang Waktu Kontinu     : {df['Timestamp'].min()} s/d {df['Timestamp'].max()}")
    print(f"Jumlah Node Pengirim Unik : {df['From_Node'].nunique():,} akun")
    print(f"Jumlah Node Penerima Unik : {df['To_Node'].nunique():,} akun")
    print(f"Total Kasus Pencucian Uang: {int(df['Is Laundering'].sum()):,} transaksi")
    print(f"Proporsi Kelas Minoritas  : {(df['Is Laundering'].sum() / len(df) * 100):.4f}%")
    print(f"File Hasil Tersimpan      : {file_output}")
    print("=" * 60 + "\n")
    print("[+] Tahap 1 Selesai! Data 1 Juta Transaksi siap digunakan.\n")

if __name__ == "__main__":
    slice_and_prepare_ibm()
