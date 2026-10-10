import pandas as pd
from data_loader import potong_data_overlap

nama_file = "data/raw/datasetBlockChain.csv"

print(f"[*] Membaca dataset blockchain: {nama_file}")
# 1. Baca dengan separator titik koma
df_eth = pd.read_csv(nama_file, sep=';')

# 2. Bersihkan spasi gaib pada nama kolom
df_eth.columns = df_eth.columns.str.strip()

# 3. Ubah format waktu (DD/MM/YYYY HH:MM)
df_eth['TimeStamp'] = pd.to_datetime(df_eth['TimeStamp'], dayfirst=True)

print("\n--- RINGKASAN DATASET ---")
print(f"Total Transaksi : {len(df_eth):,} baris")
print(f"Distribusi Label: \n{df_eth['Class'].value_counts()}")
print("\nContoh 3 Baris Data:")
print(df_eth[['TimeStamp', 'From', 'To', 'Value', 'Class']].head(3))

# 4. Uji coba potong data (Jendela 2 jam, geser 1 jam)
print("\n--- MENCOBA SLIDING WINDOW ---")
kumpulan_jendela = potong_data_overlap(
    df_eth, 
    kolom_waktu='TimeStamp', 
    lebar_jendela_jam=24, 
    geser_jam=12,
    min_transaksi=3
)

import os

os.makedirs("data/processed", exist_ok=True)

# Simpan 3 sampel jendela pertama ke file CSV terpisah
for i in range(min(3, len(kumpulan_jendela))):
    nama_file_jendela = f"data/processed/subgraph_window_{i+1}.csv"
    kumpulan_jendela[i].to_csv(nama_file_jendela, index=False)
    print(f"[+] Berhasil menyimpan fisik: {nama_file_jendela}")