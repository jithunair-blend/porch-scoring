import os
from config.config import LOCAL_DIR
import time
import pandas as pd

from helpers.preprocessing_acxiom import preprocess_acx
from helpers.read_s3 import get_asset
from helpers.scoring import scoring_acx
from helpers.write_s3 import put_output


def main():
    pipeline_start = time.time()

    # 1. Read and preprocess Acxiom file from S3
    start = time.time()
    print("[STEP: preprocessing] Fetching and preprocessing Acxiom data...")
    scoring_df, df_scoring_ids = preprocess_acx("acxiom_data")
    print(f"[STEP: preprocessing] Completed in {time.time() - start:.2f} seconds")

    # 2. Read original CSV for merging later
    start = time.time()
    print("[STEP: original_csv] Fetching original CSV...")
    orig_path = get_asset("original_csv")
    original_df = pd.read_csv(orig_path)
    # Add synthetic CUSTOMER_ID for merging if missing
    if "CUSTOMER_ID" not in original_df.columns:
        original_df["CUSTOMER_ID"] = range(len(original_df))
    print(f"[STEP: original_csv] Completed in {time.time() - start:.2f} seconds")

    # 3. Score the file (model pulled from S3 inside scoring_acx)
    start = time.time()
    print("[STEP: scoring] Running model predictions...")
    scored_df = scoring_acx(scoring_df, df_scoring_ids, original_df)
    print(f"[STEP: scoring] Completed in {time.time() - start:.2f} seconds")

    # 4. Write output back to S3
    start = time.time()
    # local_outfile = "scored_acxiom_output.csv"
    # scored_df.to_csv(local_outfile, index=False)
    local_outfile = os.path.join(LOCAL_DIR, "scored_acxiom_output.csv")
    scored_df.to_csv(local_outfile, index=False)
    s3_uri = put_output("acxiom_output", local_outfile)
    print(f"[STEP: write_upload] Completed in {time.time() - start:.2f} seconds")
    print(f"[RESULT] Scored output written to {s3_uri}")

    # Total runtime
    total = time.time() - pipeline_start
    print(f"[PIPELINE] Finished in {total:.2f} seconds")


if __name__ == "__main__":
    main()
