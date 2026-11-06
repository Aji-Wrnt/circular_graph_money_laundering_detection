import os
import time
import pandas as pd

def run_feature_fusion(
    tabular_csv="data/processed/ibm_aml_sliced_1m.csv",
    gae_features_csv="data/processed/gae_extracted_features.csv",
    output_fused_csv="data/processed/ibm_aml_fused_features.csv"
):
    print("Starting Stage 4: Feature Fusion and Overlap Aggregation")
    start_time = time.time()

    if not os.path.exists(tabular_csv):
        raise FileNotFoundError(f"Tabular file not found: {tabular_csv}")
    if not os.path.exists(gae_features_csv):
        raise FileNotFoundError(f"GAE features file not found: {gae_features_csv}")

    print(f"Reading GAE extracted features from {gae_features_csv}...")
    df_gae = pd.read_csv(gae_features_csv, sep=';')
    raw_gae_rows = len(df_gae)

    emb_cols = [col for col in df_gae.columns if col.startswith("emb_")]
    agg_rules = {"anomaly_score": "max"}
    for col in emb_cols:
        agg_rules[col] = "mean"

    print("Aggregating overlapping window instances per TxID (max anomaly score, mean embeddings)...")
    df_gae_unique = df_gae.groupby("TxID", as_index=False).agg(agg_rules)
    del df_gae

    print(f"Reading original tabular transactions from {tabular_csv}...")
    df_tab = pd.read_csv(tabular_csv, sep=';')

    print("Merging tabular features with aggregated graph features on TxID...")
    df_fused = pd.merge(df_tab, df_gae_unique, on="TxID", how="inner")
    del df_tab, df_gae_unique

    os.makedirs(os.path.dirname(output_fused_csv), exist_ok=True)
    print(f"Saving fused dataset to {output_fused_csv}...")
    df_fused.to_csv(output_fused_csv, sep=';', index=False)

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_fused_csv) / (1024 * 1024)
    fraud_count = int(df_fused["Is Laundering"].sum())

    print("\n___STAGE 4 SUMMARY REPORT___")
    print(f"Raw GAE Edge Instances    : {raw_gae_rows:,} rows")
    print(f"Unique Aggregated TxIDs   : {len(df_fused):,} transactions")
    print(f"Confirmed Laundering Cases: {fraud_count:,} transactions")
    print(f"Total Columns in Fused Set: {df_fused.shape[1]} columns")
    print(f"Output File Size          : {file_size_mb:.2f} MB")
    print(f"Computation Time          : {elapsed:.2f} seconds")
    print(f"Saved Path                : {output_fused_csv}")
    print("Stage 4 completed. Ready for Stage 5 (Random Forest Classification).")

if __name__ == "__main__":
    run_feature_fusion()
