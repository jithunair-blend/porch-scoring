import os
import time
import pandas as pd
from config.config import LOCAL_DIR

# Helpers
from helpers.preprocessing_acxiom import preprocess_acx
from helpers.preprocessing_epsilon import preprocess_eps
from helpers.read_s3 import get_asset
from helpers.scoring import scoring_acx, scoring_eps
from helpers.write_s3 import put_output


def run_acxiom_pipeline(original_df):
    """Run full Acxiom preprocessing → scoring → upload pipeline."""
    start_total = time.time()
    print("\n==================== ACXIOM PIPELINE START ====================")

    # 1. Preprocess Acxiom
    start = time.time()
    print("[ACX STEP: preprocessing] Fetching and preprocessing Acxiom data...")
    scoring_df, df_scoring_ids = preprocess_acx("acxiom_data")
    print(f"[ACX STEP: preprocessing] Completed in {time.time() - start:.2f} seconds")

    # 2. Scoring
    start = time.time()
    print("[ACX STEP: scoring] Running model predictions...")
    scored_df = scoring_acx(scoring_df, df_scoring_ids, original_df)
    print(f"[ACX STEP: scoring] Completed in {time.time() - start:.2f} seconds")

    # 3. Write output
    start = time.time()
    local_outfile = os.path.join(LOCAL_DIR, "scored_acxiom_output.csv")
    scored_df.to_csv(local_outfile, index=False)
    s3_uri = put_output("acxiom_output", local_outfile)
    print(f"[ACX STEP: write_upload] Completed in {time.time() - start:.2f} seconds")
    print(f"[ACX RESULT] Scored output written to {s3_uri}")

    print(f"[ACXIOM PIPELINE] Finished in {time.time() - start_total:.2f} seconds")
    print("==============================================================\n")


def run_epsilon_pipeline(original_df):
    """Run full Epsilon preprocessing → scoring → upload pipeline."""
    start_total = time.time()
    print("\n==================== EPSILON PIPELINE START ====================")

    # 1. Preprocess Epsilon
    start = time.time()
    print("[EPS STEP: preprocessing] Fetching and preprocessing Epsilon data...")
    scoring_df, df_scoring_ids = preprocess_eps("epsilon_data")
    print(f"[EPS STEP: preprocessing] Completed in {time.time() - start:.2f} seconds")

    # 2. Scoring
    start = time.time()
    print("[EPS STEP: scoring] Running model predictions...")
    scored_df = scoring_eps(scoring_df, df_scoring_ids, original_df)
    print(f"[EPS STEP: scoring] Completed in {time.time() - start:.2f} seconds")

    # 3. Write output
    start = time.time()
    local_outfile = os.path.join(LOCAL_DIR, "scored_epsilon_output.csv")
    scored_df.to_csv(local_outfile, index=False)
    s3_uri = put_output("epsilon_output", local_outfile)
    print(f"[EPS STEP: write_upload] Completed in {time.time() - start:.2f} seconds")
    print(f"[EPS RESULT] Scored output written to {s3_uri}")

    print(f"[EPSILON PIPELINE] Finished in {time.time() - start_total:.2f} seconds")
    print("===============================================================\n")


def main():
    """Main entry — runs both Acxiom and Epsilon scoring pipelines sequentially."""
    pipeline_start = time.time()
    print("==================== PORCH SCORING PIPELINE ====================\n")

    # 1. Read original CSV once (shared by both)
    start = time.time()
    print("[STEP: original_csv] Fetching original CSV from S3...")
    orig_path = get_asset("original_csv")
    original_df = pd.read_csv(orig_path)

    # Add synthetic CUSTOMER_ID for merging if missing
    if "CUSTOMER_ID" not in original_df.columns:
        original_df["CUSTOMER_ID"] = range(len(original_df))

    print(f"[STEP: original_csv] Completed in {time.time() - start:.2f} seconds")

    # 2. Run both pipelines
    run_acxiom_pipeline(original_df)
    run_epsilon_pipeline(original_df)

    total = time.time() - pipeline_start
    print(f"\n==================== ALL PIPELINES COMPLETED ====================")
    print(f"Total end-to-end runtime: {total:.2f} seconds")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
