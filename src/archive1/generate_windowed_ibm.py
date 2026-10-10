import os
import pandas as pd
from datetime import timedelta

def proses_window_ibm(
    file_input, 
    file_output, 
    lebar_jendela_jam=24, 
    geser_jam=12, 
    min_transaksi=3
):
    print(f"[*] 1. Membaca dataset IBM AML: {file_input}")
    if not os.path.exists(file_input):
        raise FileNotFoundError(f"File tidak ditemukan di: {file_input}")

    # Dataset IBM AML menggunakan pemisah koma standar (,)
    df = pd.read_csv(file_input)
    df.columns = df.columns.str.strip()

    # Mengatasi duplikasi kolom 'Account' -> Ubah jadi From_Account dan To_Account
    # Urutan kolom asli: [Timestamp, From Bank, Account, To Bank, Account.1, ...]
    kolom_baru = list(df.columns)
    if 'Account.1' in kolom_baru:
        df = df.rename(columns={'Account': 'From_Account', 'Account.1': 'To_Account'})
    else:
        # Penanganan fallback jika dibaca via index urutan kolom
        df.columns.values[2] = 'From_Account'
        df.columns.values[4] = 'To_Account'

    # Buat Node ID Komposit (Bank_Account) agar rekening antar bank tidak bertabrakan
    print("[*] 2. Membentuk Node ID Pengirim dan Penerima...")
    df['From_Node'] = df['From Bank'].astype(str) + "_" + df['From_Account'].astype(str)
    df['To_Node'] = df['To Bank'].astype(str) + "_" + df['To_Account'].astype(str)

    # Parsing waktu (format YYYY/MM/DD HH:MM)
    print("[*] 3. Mengonversi timestamp dan mengurutkan secara kronologis...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp').reset_index(drop=True)

    waktu_mulai = df['Timestamp'].min().floor('D')
    waktu_akhir = df['Timestamp'].max()

    print(f"    - Rentang transaksi : {waktu_mulai} s/d {waktu_akhir}")
    print(f"    - Total transaksi   : {len(df):,} baris")

    # Pemotongan sliding window
    print(f"[*] 4. Memproses Overlapping Window ({lebar_jendela_jam} jam, geser {geser_jam} jam)...")
    daftar_jendela = []
    waktu_saat_ini = waktu_mulai
    window_id = 1

    while waktu_saat_ini < waktu_akhir:
        batas_akhir = waktu_saat_ini + timedelta(hours=lebar_jendela_jam)
        mask = (df['Timestamp'] >= waktu_saat_ini) & (df['Timestamp'] < batas_akhir)
        potongan = df.loc[mask].copy()

        if len(potongan) >= min_transaksi:
            potongan['window_id'] = window_id
            daftar_jendela.append(potongan)
            window_id += 1

        waktu_saat_ini += timedelta(hours=geser_jam)

    if not daftar_jendela:
        print("[!] Tidak ada jendela yang memenuhi syarat minimal transaksi.")
        return

    print("[*] 5. Menggabungkan potongan jendela dan menyimpan ke file fisik...")
    df_windowed = pd.concat(daftar_jendela, ignore_index=True)
    os.makedirs(os.path.dirname(file_output), exist_ok=True)
    
    df_windowed.to_csv(file_output, sep=';', index=False)

    print("\n" + "="*50)
    print("        RINGKASAN WINDOWING DATASET IBM AML       ")
    print("="*50)
    print(f"Total baris mentah        : {len(df):,} baris")
    print(f"Total baris ber-window    : {len(df_windowed):,} baris")
    print(f"Total jendela terbentuk   : {df_windowed['window_id'].nunique():,} jendela")
    print(f"Total kasus Laundering    : {int(df_windowed['Is Laundering'].sum()):,} transaksi")
    print(f"File tersimpan di         : {file_output}")
    print("="*50 + "\n")

if __name__ == "__main__":
    RAW_FILE = "data/raw/HI-Small_Trans.csv"
    PROCESSED_FILE = "data/processed/ibm_transactions_windowed.csv"

    proses_window_ibm(
        file_input=RAW_FILE,
        file_output=PROCESSED_FILE,
        lebar_jendela_jam=24,
        geser_jam=12,
        min_transaksi=3
    )