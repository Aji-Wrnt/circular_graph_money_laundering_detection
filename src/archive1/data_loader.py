import pandas as pd
from datetime import timedelta

def potong_data_overlap(df, kolom_waktu, lebar_jendela_jam=24, geser_jam=12, min_transaksi=3):
    print(f"[*] Memulai pemotongan data. Total transaksi mentah: {len(df)} baris")
    
    df[kolom_waktu] = pd.to_datetime(df[kolom_waktu])
    df = df.sort_values(by=kolom_waktu)
    
    waktu_mulai = df[kolom_waktu].min().floor('D')
    waktu_akhir = df[kolom_waktu].max()
    
    kumpulan_jendela = []
    waktu_saat_ini = waktu_mulai
    no_jendela = 1
    
    while waktu_saat_ini < waktu_akhir:
        batas_akhir_jendela = waktu_saat_ini + timedelta(hours=lebar_jendela_jam)
        
        # Ambil data transaksi di rentang jendela
        data_potongan = df[(df[kolom_waktu] >= waktu_saat_ini) & (df[kolom_waktu] < batas_akhir_jendela)]
        
        # FILTER: Hanya simpan jendela yang punya minimal transaksi tertentu
        if len(data_potongan) >= min_transaksi:
            kumpulan_jendela.append(data_potongan)
            no_jendela += 1
            
        waktu_saat_ini += timedelta(hours=geser_jam)
        
    print(f"[*] Selesai! Dari ribuan jam, tersaring {len(kumpulan_jendela)} potongan jendela aktif yang siap diproses.\n")
    return kumpulan_jendela

# ==========================================
# UJI COBA MENGGUNAKAN DATA DUMMY DULU
# ==========================================
# if __name__ == "__main__":
#     # Kita bikin simulasi 4 transaksi.
#     # Perhatikan: Transaksi B->C (12.55) dan C->A (13.05) adalah pencucian uang sirkular (A->B->C->A)
    
#     data_dummy = {
#         'waktu_transaksi': ['2026-10-01 12:15:00', '2026-10-01 12:55:00', '2026-10-01 13:05:00', '2026-10-01 13:45:00'],
#         'pengirim': ['A', 'B', 'C', 'D'],
#         'penerima': ['B', 'C', 'A', 'E'],
#         'nominal': [500000, 500000, 500000, 1000000]
#     }
    
#     df_dummy = pd.DataFrame(data_dummy)
#     print("=== DATA TRANSAKSI AWAL (Murni Tabular) ===")
#     print(df_dummy[['waktu_transaksi', 'pengirim', 'penerima', 'nominal']])
#     print("="*43)
    
#     # Kita eksekusi fungsi pemotong datanya
#     # Lebar jendela 2 jam, tapi digeser per 1 jam (biar tumpang tindih)
#     hasil = potong_data_overlap(df_dummy, kolom_waktu='waktu_transaksi', lebar_jendela_jam=2, geser_jam=1)
    
#     # Kita cek isi Jendela ke-2 (Index 1) untuk membuktikan alat ini bekerja
#     print("=== CEK ISI JENDELA 2 (Bukti Pola Sirkular Terselamatkan) ===")
#     print(hasil[1][['waktu_transaksi', 'pengirim', 'penerima']])