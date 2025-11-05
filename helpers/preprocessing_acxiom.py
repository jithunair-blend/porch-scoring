import pandas as pd
import numpy as np
from helpers.read_s3 import get_asset
from helpers.generate_snowflake_data import generate_and_upload_acxiom_data

def preprocess_acx(asset_name: str):
# def preprocess_acx():

    use_cols=['FIRST_NAME','LAST_NAME','ADDRESS_LINE_1','ADDRESS_LINE_2','RECORD_KEY', 'ADDRESS_LINE2','CONCAT_ADDRESS','ZIP_CODE','ZIP4_CODE','STATE',
    'AP001241',
    'IBE8707',
    'IBE8579',
    'AP004950',
    'IBE4100',
    'AP009353',
    'AP008717',
    'AP009796',
    'IBE8652',
    'AP009356',
    'AP000920',
    'AP003446',
    'AP000572',
    'IBE8463',
    'IBE8589',
    'IBE2526',
    'AP000903',
    'IBE8850',
    'AP008718',
    'AP004959',
    'AP000953',
    'IBE8702',
    'AP006249_01',
    'AP004988',
    'AP000482',
    'AP009355',
    'IBE2800',
    'AP006334',
    'IBE8592',
    'AP003588',
    'AP009795',
    'AP006246_01',
    'IBE8706',
    'AP002897',
    'AP001242',
    'AP001238',
    'AP009803',
    'IBE8614',
    'AP009354',
    'IBE8436',
    'IBE9356',
    'IBE8584',
    'IBE8586',
    'IBE8671',
    'IBE8567',
    'IBE8763',
    'IBE8440',
    'IBE8437',
    'IBE8642',    
    'IBE3700',
    'IBE7777',
    'IBE3699',
    'IBE8623_03',
    'IBE7801',
    'IBE2062_02',
    'IBE7796',
    'IBE6145',
    'IBE8167_21',
    'IBE2061_02',
    'IBE8578',
    'IBE8271',
    'IBE8611_03',
    'IBE8621_01',
    'IBE9153_12',
    'IBE7823',
    'IBE7720',
    'IBE7817',
    'IBE7753',
    'IBE6505',
    'IBE7829', 
    'HOME_OWNER_SOURCE_FLAG',
    'IBE8688',
    'IBE2360',
    'IBE7622_01',
    'IBE9515',
    'IBE8625',
    'IBE8701',
    'IBE8581',
    'PX011283_01',
    'PX011284_01',
    'PX011282_01',
    "MATCH_SCORE"]

    # local_path = get_asset(asset_name)  
    # scoring_df = pd.read_csv("C:/Users/JithuNair/OneDrive - Blend 360/Documents/Projects/Porch/New/match_acx_results.csv")
    scoring_df = generate_and_upload_acxiom_data()
    # Example: df = pd.read_csv("your_file.csv")

    scoring_df = scoring_df.rename(columns={
        'ACX_RECORD_KEY': 'RECORD_KEY',
        'CASS_FIRST_NAME': 'FIRST_NAME',
        'CASS_LAST_NAME': 'LAST_NAME',
        'CASS_ADDRESS_LINE_1': 'ADDRESS_LINE_1',
        'CASS_ADDRESS_LINE_2': 'ADDRESS_LINE_2',
        'ACX_ADDRESS': 'CONCAT_ADDRESS',
        'CASS_ZIPCODE': 'ZIP_CODE',
        'CASS_STATE': 'STATE'
    })

    scoring_df = scoring_df[use_cols]


    #Drop rows where more than 30 percent of column being null to avoid having the model impute them
    scoring_df = scoring_df[scoring_df["MATCH_SCORE"].notna() & (scoring_df.isnull().sum(axis=1) / scoring_df.shape[1] * 100 <= 30)]
    scoring_df.drop(columns = ['MATCH_SCORE'],inplace=True,axis=1)

    # scoring_df = generate_acxiom_data()

    # Step 1: Separate individual IDs and addresses from the scoring data
    df_scoring_ids = scoring_df[['RECORD_KEY','FIRST_NAME','LAST_NAME','ADDRESS_LINE_1','ADDRESS_LINE_2','ADDRESS_LINE2','CONCAT_ADDRESS','ZIP_CODE','ZIP4_CODE','STATE']].copy() # Edit if the columns change

    top_features_continous=['AP001241',
    'IBE8707',
    'IBE8579',
    'AP004950',
    'IBE4100',
    'AP009353',
    'AP008717',
    'AP009796',
    'IBE8652',
    'AP009356',
    'AP000920',
    'AP003446',
    'AP000572',
    'IBE8463',
    'IBE8589',
    'IBE2526',
    'AP000903',
    'IBE8850',
    'AP008718',
    'AP004959',
    'AP000953',
    'IBE8702',
    'AP006249_01',
    'AP004988',
    'AP000482',
    'AP009355',
    'IBE2800',
    'AP006334',
    'IBE8592',
    'AP003588',
    'AP009795',
    'AP006246_01',
    'IBE8706',
    'AP002897',
    'AP001242',
    'AP001238',
    'AP009803',
    'IBE8614',
    'AP009354']

    top_features_ordinal=['IBE8436',
    'IBE9356',
    'IBE8584',
    'IBE8586',
    'IBE8671',
    'IBE8567',
    'IBE8763',
    'IBE8440',
    'IBE8437',
    'IBE8642']

    top_features_binary=['IBE3700',
    'IBE7777',
    'IBE3699',
    'IBE8623_03',
    'IBE7801',
    'IBE2062_02',
    'IBE7796',
    'IBE6145',
    'IBE8167_21',
    'IBE2061_02',
    'IBE8578',
    'IBE8271',
    'IBE8611_03',
    'IBE8621_01',
    'IBE9153_12',
    'IBE7823',
    'IBE7720',
    'IBE7817',
    'IBE7753',
    'IBE6505',
    'IBE7829']

    top_features_categorical=['HOME_OWNER_SOURCE_FLAG',
    'IBE8688',
    'IBE2360',
    'IBE7622_01',
    'IBE9515',
    'IBE8625',
    'IBE8701',
    'IBE8581',
    'PX011283_01',
    'PX011284_01',
    'PX011282_01']


    # # Loop through each variable and impute based on skewness
    # for var in top_features_continous:
    #     skew_val = scoring_df[var].skew()
    #     if abs(skew_val) > 1:
    #         # Highly skewed → use median
    #         scoring_df[var] = scoring_df[var].fillna(scoring_df[var].median())
    #     else:
    #         # Fairly symmetric → use mean
    #         scoring_df[var] = scoring_df[var].fillna(scoring_df[var].mean())

    for var in top_features_continous:
        # Ensure column is numeric
        scoring_df[var] = pd.to_numeric(scoring_df[var], errors='coerce')

        skew_val = scoring_df[var].skew(skipna=True)
        if abs(skew_val) > 1:
            # Highly skewed → use median
            scoring_df[var] = scoring_df[var].fillna(scoring_df[var].median())
        else:
            # Fairly symmetric → use mean
            scoring_df[var] = scoring_df[var].fillna(scoring_df[var].mean())




    binary_mappings_base = {
        # 'Y' vs NaN
        'IBE1806': {'Y': 1, np.nan: 0},
        'IBE8693': {'Y': 1, np.nan: 0},
        'IBE8653': {'Y': 1, np.nan: 0},
        'IBE8630': {'Y': 1, np.nan: 0},
        'IBE7467': {'Y': 1, np.nan: 0},
        'IBE7468': {'Y': 1, np.nan: 0},

        # 1 vs NaN
        'IBE8621_01': {1: 1, np.nan: 0},
        'IBE8621_02': {1: 1, np.nan: 0},
        'IBE8621_04': {1: 1, np.nan: 0},
        'IBE8621_06': {1: 1, np.nan: 0},
        'IBE2061_02': {1: 1, np.nan: 0},
        'IBE2062_02': {1: 1, np.nan: 0},
        'IBE7825': {1: 1, np.nan: 0},
        'IBE7832': {1: 1, np.nan: 0},
        'IBE7829': {1: 1, np.nan: 0},
        'IBE7826': {1: 1, np.nan: 0},
        'IBE7827': {1: 1, np.nan: 0},
        'IBE7830': {1: 1, np.nan: 0},
        'IBE7828': {1: 1, np.nan: 0},
        'IBE7823': {1: 1, np.nan: 0},
        'IBE7822': {1: 1, np.nan: 0},
        'IBE7824': {1: 1, np.nan: 0},
        'IBE7831': {1: 1, np.nan: 0},
        'IBE2076_01': {1: 1, np.nan: 0},
        'IBE2076_08': {1: 1, np.nan: 0},
        'IBE7733': {1: 1, np.nan: 0},
        'IBE7737': {1: 1, np.nan: 0},
        'IBE7788': {1: 1, np.nan: 0},
        'IBE7796': {1: 1, np.nan: 0},
        'IBE7801': {1: 1, np.nan: 0},
        'IBE7719': {1: 1, np.nan: 0},
        'IBE7739': {1: 1, np.nan: 0},
        'IBE7740': {1: 1, np.nan: 0},
        'IBE7720': {1: 1, np.nan: 0},
        'IBE7753': {1: 1, np.nan: 0},
        'IBE7817': {1: 1, np.nan: 0},
        'IBE7815': {1: 1, np.nan: 0},
        'IBE7816': {1: 1, np.nan: 0},
        'IBE6504': {1: 1, np.nan: 0},
        'IBE2031': {1: 1, np.nan: 0},
        'IBE6503': {1: 1, np.nan: 0},
        'IBE8167_21': {1: 1, np.nan: 0},
        'IBE3699': {1: 1, np.nan: 0},
        'IBE3700': {1: 1, np.nan: 0},
        'IBE6505': {1: 1, np.nan: 0},
        'IBE7851': {1: 1, np.nan: 0},
        'IBE6145': {1: 1, np.nan: 0},
        'IBE9153_12': {1: 1, np.nan: 0},
        'IBE8272': {1: 1, np.nan: 0},
        'IBE8271': {1: 1, np.nan: 0},
        'IBE8322': {1: 1, np.nan: 0},

        # 0 vs NaN
        'IBE8611_03': {0: 1, np.nan: 0},
        'IBE8623_03': {0: 1, np.nan: 0},

        # nan then 1 (same as 1 vs nan)
        'IBE7779': {1: 1, np.nan: 0},
        'IBE7777': {1: 1, np.nan: 0},

        # Special Cases (Single Character Values)
        'IBE9567': {'M': 1, np.nan: 0},
        'IBE8615': {'B': 1, np.nan: 0},
        'IBE8620': {'R': 1, np.nan: 0},
        'IBE9569': {'M': 1, np.nan: 0},
        'IBE9568': {'M': 1, np.nan: 0},
        'IBE8578': {'O': 1, np.nan: 0},
        'IBE8749': {'P': 1, np.nan: 0},
        'IBE9100': {'M': 1, np.nan: 0},
        'AP1999': {'M': 1, np.nan: 0},
    }


    for col in top_features_binary:
        if col in binary_mappings_base:
            scoring_df[col] = scoring_df[col].map(binary_mappings_base[col])


    scoring_df[top_features_categorical] = scoring_df[top_features_categorical].fillna('unknown')
    mode_dict = scoring_df[top_features_ordinal].mode(dropna=True).iloc[0]
    scoring_df.fillna(value=mode_dict.to_dict(), inplace=True)


    scoring_df['IBE8763'] = scoring_df['IBE8763'].map({
        'P': 1,   # 2026 and greater (newest)
        'O': 2,   # 2021 - 2025
        'N': 3,   # 2016 - 2020
        'M': 4,   # 2011 - 2015
        'L': 5,   # 2006 - 2010
        'K': 6,   # 2001 - 2005
        'J': 7,   # 1996 - 2000
        'I': 8,   # 1991 - 1995
        'H': 9,   # 1986 - 1990
        'G': 10,  # 1981 - 1985
        'F': 11,  # 1971 - 1980
        'E': 12,  # 1961 - 1970
        'D': 13,  # 1951 - 1960
        'C': 14,  # 1941 - 1950
        'B': 15,  # 1900 - 1940
        'A': 16   # Before 1900 (oldest)
    })
    scoring_df['IBE8440'] = scoring_df['IBE8440'].map({
        'P': 1,   # 2026 and greater (newest)
        'O': 2,   # 2021 - 2025
        'N': 3,   # 2016 - 2020
        'M': 4,   # 2011 - 2015
        'L': 5,   # 2006 - 2010
        'K': 6,   # 2001 - 2005
        'J': 7,   # 1996 - 2000
        'I': 8,   # 1991 - 1995
        'H': 9,   # 1986 - 1990
        'G': 10,  # 1981 - 1985
        'F': 11,  # 1971 - 1980
        'E': 12,  # 1961 - 1970
        'D': 13,  # 1951 - 1960
        'C': 14,  # 1941 - 1950
        'B': 15,  # 1900 - 1940
        'A': 16   # Before 1900 (oldest)
    })
    scoring_df['IBE8642'] = scoring_df['IBE8642'].map({
        'A': 1,  # $1,000 - $24,999
        'B': 2,  # $25,000 - $49,999
        'C': 3,  # $50,000 - $74,999
        'D': 4,  # $75,000 - $99,999
        'E': 5,  # $100,000 - $124,999
        'F': 6,  # $125,000 - $149,999
        'G': 7,  # $150,000 - $174,999
        'H': 8,  # $175,000 - $199,999
        'I': 9,  # $200,000 - $224,999
        'J': 10, # $225,000 - $249,999
        'K': 11, # $250,000 - $274,999
        'L': 12, # $275,000 - $299,999
        'M': 13, # $300,000 - $349,999
        'N': 14, # $350,000 - $399,999
        'O': 15, # $400,000 - $449,999
        'P': 16, # $450,000 - $499,999
        'Q': 17, # $500,000 - $749,999
        'R': 18, # $750,000 - $999,999
        'S': 19  # $1,000,000 Plus
    })

    scoring_df['IBE8671'] = scoring_df['IBE8671'].map({
        '1': 1,  # Less than $15,000
        '2': 2,  # $15,000 - $19,999
        '3': 3,  # $20,000 - $29,999
        '4': 4,  # $30,000 - $39,999
        '5': 5,  # $40,000 - $49,999
        '6': 6,  # $50,000 - $59,999
        '7': 7,  # $60,000 - $69,999
        '8': 8,  # $70,000 - $79,999
        '9': 9,  # $80,000 - $89,999
        'A': 10, # $90,000 - $99,999
        'B': 11, # $100,000 - $124,999
        'C': 12, # $125,000 - $149,999
        'D': 13  # Greater than $149,999
    })
    scoring_df['IBE9356'] = scoring_df['IBE9356'].map({
        '1': 1,  # Less than or equal to $0
        '2': 2,  # $1 - $4,999
        '3': 3,  # $5,000 - $9,999
        '4': 4,  # $10,000 - $24,999
        '5': 5,  # $25,000 - $49,999
        '6': 6,  # $50,000 - $99,999
        '7': 7,  # $100,000 - $249,999
        '8': 8,  # $250,000 - $499,999
        '9': 9,  # $500,000 - $999,999
        'A': 10, # $1,000,000 - $1,999,999
        'B': 11  # $2,000,000 +
    })

    scoring_df['IBE8436'] = scoring_df['IBE8436'].map({
        'A': 1,  # Less than 1 Year
        'B': 2,  # 1 Year
        'C': 3,  # 2 Years
        'D': 4,  # 3 Years
        'E': 5,  # 4 Years
        'F': 6,  # 5 Years
        'G': 7,  # 6 - 7 Years
        'H': 8,  # 8 - 9 Years
        'I': 9,  # 10 - 14 Years
        'J': 10  # Greater than 14 Years
    })

    scoring_df['IBE8584'] = scoring_df['IBE8584'].map({
        'A': 1,   # 100 - 999%
        'B': 2,   # 95 - 99%
        'C': 3,   # 90 - 94%
        'D': 4,   # 85 - 89%
        'E': 5,   # 80 - 84%
        'F': 6,   # 75 - 79%
        'G': 7,   # 70 - 74%
        'H': 8,   # 60 - 69%
        'I': 9,   # 50 - 59%
        'J': 10,    # 01 - 49%
        'K': 11,  # 0% (No Loan)
    })

    scoring_df['IBE8567'] = scoring_df['IBE8567'].map({
        'A': 18,    # $1 - $4,999
        'B': 17,    # $5,000 - $9,999
        'C': 16,    # $10,000 - $19,999
        'D': 15,    # $20,000 - $29,999
        'E': 14,    # $30,000 - $49,999
        'F': 13,    # $50,000 - $74,999
        'G': 12,    # $75,000 - $99,999
        'H': 11,    # $100,000 - $149,999
        'I': 10,    # $150,000 - $199,999
        'J': 9,   # $200,000 - $249,999
        'K': 8,   # $250,000 - $299,999
        'L': 7,   # $300,000 - $349,999
        'M': 6,   # $350,000 - $399,999
        'N': 5,   # $400,000 - $499,999
        'O': 4,   # $500,000 - $749,999
        'P': 3,   # $750,000 - $999,999
        'Q': 2,   # $1,000,000 - $1,999,999
        'R': 1   # Greater than $1,999,999
    })

    scoring_df['IBE8586'] = scoring_df['IBE8586'].map({
        'A': 1,    # Equity Less than $1
        'B': 2,    # Equity $1 - $4,999
        'C': 3,    # Equity $5,000 - $9,999
        'D': 4,    # Equity $10,000 - $19,999
        'E': 5,    # Equity $20,000 - $29,999
        'F': 6,    # Equity $30,000 - $49,999
        'G': 7,    # Equity $50,000 - $74,999
        'H': 8,    # Equity $75,000 - $99,999
        'I': 9,    # Equity $100,000 - $149,999
        'J': 10,   # Equity $150,000 - $199,999
        'K': 11,   # Equity $200,000 - $249,999
        'L': 12,   # Equity $250,000 - $299,999
        'M': 13,   # Equity $300,000 - $349,999
        'N': 14,   # Equity $350,000 - $399,999
        'O': 15,   # Equity $400,000 - $499,999
        'P': 16,   # Equity $500,000 - $749,999
        'Q': 17,   # Equity $750,000 - $999,999
        'R': 18,   # Equity $1,000,000 - $1,999,999
        'S': 19    # Equity Greater than $1,999,999
    })

    scoring_df['IBE8437'] = scoring_df['IBE8437'].map({
        'A': 1,    # $1 - $4,999
        'B': 2,    # $5,000 - $9,999
        'C': 3,    # $10,000 - $19,999
        'D': 4,    # $20,000 - $29,999
        'E': 5,    # $30,000 - $49,999
        'F': 6,    # $50,000 - $74,999
        'G': 7,    # $75,000 - $99,999
        'H': 8,    # $100,000 - $149,999
        'I': 9,    # $150,000 - $199,999
        'J': 10,   # $200,000 - $249,999
        'K': 11,   # $250,000 - $299,999
        'L': 12,   # $300,000 - $349,999
        'M': 13,   # $350,000 - $399,999
        'N': 14,   # $400,000 - $499,999
        'O': 15,   # $500,000 - $749,999
        'P': 16,   # $750,000 - $999,999
        'Q': 17,   # $1,000,000 - $1,999,999
        'R': 18    # Greater than $1,999,999
    })


    scoring_df = pd.get_dummies(
        scoring_df,
        columns=top_features_categorical,
        drop_first=False,
        dtype='int8'  # or 'uint8' if all values are non-negative
    )

    # scoring_df[['PX011282_01_14.0', 'PX011282_01_69.0', 'PX011282_01_67.0',
    #         'PX011282_01_13.0', 'PX011282_01_70.0', 'PX011282_01_28.0',
    #         'PX011282_01_73.0', 'PX011282_01_72.0']] = np.nan

    X_train_top_columns=['IBE8436',
    'IBE9356',
    'AP001241',
    'IBE8584',
    'IBE8707',
    'IBE8579',
    'IBE7622_01_Y',
    'IBE8701_R',
    'AP004950',
    'IBE4100',
    'AP009353',
    'IBE9515_M',
    'AP008717',
    'IBE3700',
    'IBE7777',
    'IBE8586',
    'IBE8688_M',
    'AP009796',
    'IBE8701_unknown',
    'IBE3699',
    'PX011282_01_14.0',
    'PX011283_01_33.0',
    'IBE8652',
    'IBE8623_03',
    'AP009356',
    'IBE8671',
    'AP000920',
    'IBE8625_C',
    'IBE7801',
    'IBE2062_02',
    'AP003446',
    'AP000572',
    'IBE7796',
    'IBE8463',
    'IBE8589',
    'IBE2526',
    'PX011283_01_38.0',
    'AP000903',
    'PX011282_01_69.0',
    'IBE8850',
    'PX011282_01_67.0',
    'IBE6145',
    'AP008718',
    'IBE8167_21',
    'IBE8688_unknown',
    'IBE2061_02',
    'PX011284_01_13B',
    'IBE8578',
    'PX011284_01_04B',
    'AP004959',
    'IBE2360_I001',
    'IBE8567',
    'AP000953',
    'PX011283_01_73.0',
    'IBE8702',
    'IBE8271',
    'AP006249_01',
    'PX011282_01_13.0',
    'IBE8581_C',
    'IBE8611_03',
    'PX011284_01_03B',
    'PX011284_01_11B',
    'HOME_OWNER_SOURCE_FLAG_V',
    'IBE8763',
    'PX011283_01_62.0',
    'IBE8621_01',
    'IBE9153_12',
    'AP004988',
    'IBE8440',
    'IBE8437',
    'PX011284_01_02A',
    'IBE7823',
    'IBE2360_G002',
    'AP000482',
    'AP009355',
    'IBE2800',
    'AP006334',
    'IBE7720',
    'IBE7817',
    'IBE8592',
    'AP003588',
    'IBE7753',
    'AP009795',
    'PX011282_01_70.0',
    'AP006246_01',
    'PX011282_01_28.0',
    'IBE6505',
    'IBE8706',
    'IBE8625_B',
    'PX011282_01_73.0',
    'AP002897',
    'IBE8642',
    'AP001242',
    'HOME_OWNER_SOURCE_FLAG_unknown',
    'AP001238',
    'AP009803',
    'IBE8614',
    'PX011282_01_72.0',
    'AP009354',
    'IBE7829']

    # ✅ Ensure all expected columns exist, fill missing with NaN (XGBoost handles NaN natively)
    for col in X_train_top_columns:
        if col not in scoring_df.columns:
            scoring_df[col] = np.nan

    # ✅ Reorder columns to match the model's expected schema
    scoring_df = scoring_df[X_train_top_columns]


    scoring_df=scoring_df[X_train_top_columns]
    # scoring_df.head()

    # scoring_df.to_csv("scoring_df.csv")
    df_scoring_ids = df_scoring_ids.loc[:, ~df_scoring_ids.columns.duplicated()]
    # df_scoring_ids.to_csv("scoring_ids.csv")
    # print(df_scoring_ids.columns)
    # print(df_scoring_ids.shape)
    # print(df_scoring_ids.head(2))
    return scoring_df, df_scoring_ids

# if __name__ == "__main__":
#     scoring_df,df_scoring_ids = preprocess_acx()