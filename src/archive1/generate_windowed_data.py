import os
import pandas as pd
from datetime import timedelta

def potong_dan_labeli_jendela(
    file_input, 
    file_output, 
    lebar_jendela_jam=24, 
    geser_jam=12, 
    min_transaksi=3
):
    print(f"[*] 1. Membaca dataset mentah: {file_input}")
    if not os.path.exists(file_input):
        raise FileNotFoundError(f"File tidak ditemukan di: {file_input}")
        
    df = pd.read_csv(file_input, sep=';')
    df.columns = df.columns.str.strip()
    
    # Konversi waktu dan urutkan secara kronologis
    print("[*] 2. Mengonversi timestamp dan mengurutkan transaksi...")
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], dayfirst=True)
    df = df.sort_values(by='TimeStamp').reset_index(drop=True)
    
    waktu_mulai = df['TimeStamp'].min().floor('D')
    waktu_akhir = df['TimeStamp'].max()
    
    print(f"    - Rentang transaksi : {waktu_mulai} s/d {waktu_akhir}")
    print(f"    - Total transaksi   : {len(df):,} baris")
    
    # Pemotongan sliding window
    print(f"[*] 3. Memproses Overlapping Window ({lebar_jendela_jam} jam, geser {geser_jam} jam)...")
    daftar_jendela = []
    waktu_saat_ini = waktu_mulai
    window_id = 1
    
    while waktu_saat_ini < waktu_akhir:
        batas_akhir = waktu_saat_ini + timedelta(hours=lebar_jendela_jam)
        
        # Filter transaksi di dalam rentang jendela
        mask = (df['TimeStamp'] >= waktu_saat_ini) & (df['TimeStamp'] < batas_akhir)
        potongan = df.loc[mask].copy()
        
        # Hanya simpan jendela yang memenuhi syarat minimum transaksi
        if len(potongan) >= min_transaksi:
            potongan['window_id'] = window_id
            daftar_jendela.append(potongan)
            window_id += 1
            
        waktu_saat_ini += timedelta(hours=geser_jam)
        
    if not daftar_jendela:
        print("[!] Tidak ada jendela yang memenuhi syarat.")
        return
        
    # Gabungkan seluruh potongan jendela menjadi 1 DataFrame
    print("[*] 4. Menggabungkan seluruh potongan jendela ke satu DataFrame...")
    df_windowed = pd.concat(daftar_jendela, ignore_index=True)
    
    # Buat direktori tujuan jika belum ada
    os.makedirs(os.path.dirname(file_output), exist_ok=True)
    
    print(f"[*] 5. Menyimpan hasil ke: {file_output}")
    df_windowed.to_csv(file_output, sep=';', index=False)
    
    print("\n" + "="*45)
    print("           RINGKASAN HASIL EKSEKUSI          ")
    print("="*45)
    print(f"Total baris mentah        : {len(df):,} baris")
    print(f"Total baris ber-window    : {len(df_windowed):,} baris (termasuk overlap)")
    print(f"Total jendela terbentuk   : {df_windowed['window_id'].nunique():,} jendela")
    print(f"Rata-rata transaksi/jendela: {len(df_windowed) / df_windowed['window_id'].nunique():.1f}")
    print(f"Distribusi Fraud/Scam      : {int(df_windowed['Class'].sum()):,} transaksi anomali")
    print("="*45)
    print("[+] File siap digunakan untuk training GAE dan Random Forest!\n")

if __name__ == "__main__":
    RAW_PATH = "data/raw/datasetBlockChain.csv"
    PROCESSED_PATH = "data/processed/eth_transactions_windowed.csv"
    
    potong_dan_labeli_jendela(
        file_input=RAW_PATH,
        file_output=PROCESSED_PATH,
        lebar_jendela_jam=24,
        geser_jam=12,
        min_transaksi=3
    )