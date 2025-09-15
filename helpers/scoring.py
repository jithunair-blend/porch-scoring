# from config.config import path
import xgboost as xgb
import pandas as pd
from helpers.read_s3 import get_asset

def scoring_acx(scoring_df, df_scoring_ids, original_df):


    # # Load model from downloaded file
    # loaded_model = xgb.XGBClassifier()
    # loaded_model.load_model(path)

    model_path = get_asset("acxiom_model")  # fetch Acxiom model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    scoring_df['score'] = model.predict_proba(scoring_df)[:, 1]

    #Add the individual IDs and addresses back to the dataframe
    scoring_df['CUSTOMER_ID'] = df_scoring_ids['CUSTOMER_ID']
    scoring_df['ADDRESS_LINE2'] = df_scoring_ids['ADDRESS_LINE2']
    scoring_df['CONCAT_ADDRESS'] = df_scoring_ids['CONCAT_ADDRESS']
    scoring_df['ZIP_CODE'] = df_scoring_ids['ZIP_CODE']
    scoring_df['ZIP4_CODE'] = df_scoring_ids['ZIP4_CODE']
    scoring_df['STATE'] = df_scoring_ids['STATE']
    scoring_df['RECORD_KEY'] = df_scoring_ids['RECORD_KEY']

    scoring_df = scoring_df.reset_index(drop=True)
    
    original_df = original_df.reset_index(drop=True)
    print("Fetching original CSV...")

    # Add synthetic CUSTOMER_ID for merging (matches notebook logic)
    original_df['CUSTOMER_ID'] = range(len(original_df))
    
    #Splitting into bins/deciles
    scoring_df['score_bin'] = pd.cut(scoring_df['score'], bins=10)
    unique_bins = sorted(scoring_df['score_bin'].unique(), reverse=True)
    # Map each bin to a decile number (1 to 10)
    bin_to_decile = {bin_interval: i + 1 for i, bin_interval in enumerate(unique_bins)}

    # Apply the decile mapping
    scoring_df['decile'] = scoring_df['score_bin'].map(bin_to_decile)

    merged = original_df.merge(scoring_df, on='CUSTOMER_ID', how='left')
    merged = merged.reset_index(drop=True)

    return(merged)

def scoring_eps(scoring_df, df_scoring_ids, original_df):
    model_path = get_asset("epsilon_model")  # fetch Epsilon model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    scoring_df['score'] = model.predict_proba(scoring_df)[:, 1]
    # ... rest of your scoring logic ...
    return scoring_df