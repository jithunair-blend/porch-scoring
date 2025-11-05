# from config.config import path
import xgboost as xgb
import pandas as pd
from helpers.read_s3 import get_asset
# from read_s3 import get_asset

def scoring_acx(scoring_df, df_scoring_ids, original_df):
# def scoring_acx(scoring_df, df_scoring_ids):

    # # Load model from downloaded file
    # loaded_model = xgb.XGBClassifier()
    # loaded_model.load_model(path)

    model_path = get_asset("acxiom_model")  # fetch Acxiom model
    model = xgb.XGBClassifier()
    # model.load_model("C:/Users/JithuNair/OneDrive - Blend 360/Documents/Projects/Porch/Codes/Acxiom/final_xgb_model_top_hpt_acxiom.json")
    model.load_model(model_path)

    scoring_df['score'] = model.predict_proba(scoring_df)[:, 1]

    scoring_df['ADDRESS_LINE2'] = df_scoring_ids['ADDRESS_LINE2']
    scoring_df['CONCAT_ADDRESS'] = df_scoring_ids['CONCAT_ADDRESS']
    scoring_df['ZIP_CODE'] = df_scoring_ids['ZIP_CODE']
    scoring_df['ZIP4_CODE'] = df_scoring_ids['ZIP4_CODE']
    scoring_df['STATE'] = df_scoring_ids['STATE']
    scoring_df['RECORD_KEY'] = df_scoring_ids['RECORD_KEY']
    scoring_df["FIRST_NAME"] = df_scoring_ids["FIRST_NAME"]
    scoring_df["LAST_NAME"] = df_scoring_ids["LAST_NAME"]
    scoring_df["ADDRESS_LINE_1"] = df_scoring_ids["ADDRESS_LINE_1"]
    # scoring_df["FIRST_NAME"] = df_scoring_ids["FIRST_NAME"]

    # scoring_df['MERGED_NAME'] = scoring
        #Add the individual IDs and addresses back to the dataframe
    scoring_df['combined_pii'] = (
        df_scoring_ids[['FIRST_NAME', 'LAST_NAME', 'ADDRESS_LINE_1']]
        .fillna('')                              # Replace NaNs with empty string
        .apply(lambda x: x.str.upper())          # Convert each column to uppercase
        .agg(' '.join, axis=1)                   # Join with space
        .str.strip()                             # Remove leading/trailing spaces
    )


    scoring_df = scoring_df.reset_index(drop=True)
    
    original_df = original_df.reset_index(drop=True)
    print("Fetching original CSV...")

    # # Add synthetic CUSTOMER_ID for merging (matches notebook logic)
    # original_df['CUSTOMER_ID'] = range(len(original_df))
    original_df['combined_pii'] = (
        original_df[['FirstName', 'LastName', 'PropertyAddress']]
        .fillna('')                              # Handle NaNs
        .apply(lambda x: x.str.upper())          # Convert each column to uppercase
        .agg(' '.join, axis=1)                   # Join with spaces
        .str.strip()                             # Clean extra spaces
    )
    
    #Splitting into bins/deciles
    scoring_df['score_bin'] = pd.cut(scoring_df['score'], bins=10)
    unique_bins = sorted(scoring_df['score_bin'].unique(), reverse=True)
    # Map each bin to a decile number (1 to 10)
    bin_to_decile = {bin_interval: i + 1 for i, bin_interval in enumerate(unique_bins)}

    # Apply the decile mapping
    scoring_df['decile'] = scoring_df['score_bin'].map(bin_to_decile)

    merged = original_df.merge(scoring_df, on='combined_pii', how='left')
    merged.to_csv("scored_file_acx.csv",index=False)
    merged = merged.reset_index(drop=True)
    # scoring_df.to_csv("scored_file.csv")
    return(merged)

def scoring_eps(scoring_df, df_scoring_ids, original_df):
    model_path = get_asset("epsilon_model")  # fetch Epsilon model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    scoring_df['score'] = model.predict_proba(scoring_df)[:, 1]

    scoring_df['CONTRACTED_ADDRESS'] = df_scoring_ids['CONTRACTED_ADDRESS']
    scoring_df['ZIP'] = df_scoring_ids['ZIP']
    scoring_df['ZIP4'] = df_scoring_ids['ZIP4']
    scoring_df['STATE'] = df_scoring_ids['STATE']
    scoring_df['PERSON_SEQ_NO'] = df_scoring_ids['PERSON_SEQ_NO']
    scoring_df["CASS_FIRST_NAME"] = df_scoring_ids["CASS_FIRST_NAME"]
    scoring_df["CASS_LAST_NAME"] = df_scoring_ids["CASS_LAST_NAME"]
    scoring_df["CASS_ADDRESS_LINE_1"] = df_scoring_ids["CASS_ADDRESS_LINE_1"]
    scoring_df['CASS_ADDRESS_LINE_2'] = df_scoring_ids['CASS_ADDRESS_LINE_2']

    scoring_df['combined_pii'] = (
        df_scoring_ids[['CASS_FIRST_NAME', 'CASS_LAST_NAME', 'CASS_ADDRESS_LINE_1']]
        .fillna('')                              # Replace NaNs with empty string
        .apply(lambda x: x.str.upper())          # Convert each column to uppercase
        .agg(' '.join, axis=1)                   # Join with space
        .str.strip()                             # Remove leading/trailing spaces
    )

    scoring_df = scoring_df.reset_index(drop=True)
    
    original_df = original_df.reset_index(drop=True)
    print("Fetching original CSV...")

    # # Add synthetic CUSTOMER_ID for merging (matches notebook logic)
    # original_df['CUSTOMER_ID'] = range(len(original_df))
    original_df['combined_pii'] = (
        original_df[['FirstName', 'LastName', 'PropertyAddress']]
        .fillna('')                              # Handle NaNs
        .apply(lambda x: x.str.upper())          # Convert each column to uppercase
        .agg(' '.join, axis=1)                   # Join with spaces
        .str.strip()                             # Clean extra spaces
    )

    #Splitting into bins/deciles
    scoring_df['score_bin'] = pd.cut(scoring_df['score'], bins=10)
    unique_bins = sorted(scoring_df['score_bin'].unique(), reverse=True)
    # Map each bin to a decile number (1 to 10)
    bin_to_decile = {bin_interval: i + 1 for i, bin_interval in enumerate(unique_bins)}

    # Apply the decile mapping
    scoring_df['decile'] = scoring_df['score_bin'].map(bin_to_decile)

    merged = original_df.merge(scoring_df, on='combined_pii', how='left')
    merged.to_csv("scored_file_eps.csv",index=False)
    merged = merged.reset_index(drop=True)
    # scoring_df.to_csv("scored_file.csv")
    return(merged)