import pandas as pd
import numpy as np
from helpers.read_s3 import get_asset
from helpers.generate_snowflake_data import generate_and_upload_epsilon_data


def preprocess_eps(asset_name: str):


    use_cols=['CASS_FIRST_NAME','CASS_LAST_NAME','CASS_ADDRESS_LINE_1','CASS_ADDRESS_LINE_2','PERSON_SEQ_NO', 'CONTRACTED_ADDRESS','ZIP','ZIP4','STATE',
    'ACT_NUM_ONLINE_PURCHASE_QUINT',
    'CHANNEL_PREF_RT_CATALOG_QUINT',
    'NUM_GENERATIONS_HH_ENH',
    'AVA_HOME_EQUITY_IN_K',
    'MT_VEHICLE_SERVICE_CENTER_USERS',
    'INDIVIDUAL_EXACT_AGE',
    'MT_MEAL_KIT_DELIVERY_CONSUMERS',
    'MT_GROCERY_STORE_FREQUENTERS',
    'ADV_HH_MARITAL_STATUS',
    'MT_HOME_WARRANTY_PURCHASERS',
    'MT_RETIRED_BUT_STILL_WORKING',
    'ADV_NUM_ADULTS',
    'MORTG_INTEREST_RATE_REFIN',
    'MT_MEDICARE_ADVANTAGE_PLAN_PURCHASERS',
    'CHANNEL_PREF_RT_ONLINE_QUINT',
    '_2020_COMPLETE_CENSUS_GEO',
    'MT_YOGAPILATES_ENTHUSIAST',
    'MT_COMMUNITY_BANK_CUSTR',
    'TGT_NET_WORTH_30',
    'MT_CLICK_TO_CART_HOME_DELIVERY_CUSTOMERS',
    'DWELLING_TYP_LEGACY',
    'MT_MOBILE_PHONE_SERVICE_SWITCHERS',
    'MT_SENIOR_CAREGIVERS',
    'ACT_TOT_DOLLARS_QUINT',
    'MT_SALTY_SNACKS_PURCHASERS',
    'ACT_TOT_ONLINE_DOLLARS_QUINT',
    'MT_LOYAL_FIN_INSTITUTION_CUSTR',
    'MT_WHATS_ON_SALE_SHOPPERS',
    'MT_HEAVY_COUPON_USERS',
    'MT_ONLINE_HOTEL_UPGRADERS',
    'MT_NEW_ROOF_CUSTOMERS',
    'MT_MULTI_RETAILER_SHOPPERS',
    'MT_CASINO_GAMER',
    'MT_VEHICLE_DIYRS',
    'ROOFTOP_LONGITUDE',
    'MT_INSURANCE_FOR_LOAN_PRODUCTS_PURCHASERS',
    'MT_FINANCIALHEALTH_NEWSLETTER_SUBSCRIBERS',
    'NUM_TRADELINES',
    'AGILITY_OCCUPANCY_SCORE',
    'CURRENT_LOAN_TO_VALUE',
    'MT_CABLE_BUNDLE_CABLE_INTERNET_HOME_PHONE',
    'MT_AUTO_INSURANCE_SELF_SERVE_ONLINE_BUYERS',
    '_2010_COMPLETE_CENSUS_GEO_',
    'NUM_DOLLARS_ON_RETURNS',
    'MT_MEAL_PLANNERS',
    'ADV_DWELLING_TYP',
    'MT_AUTO_INSURANCE_CALL_CENTER_SOLD',
    'ROOFTOP_LATITUDE',
    'MT_PRE_SHOP_PLANNERS',
    'MT_MEDICARE_SUPPLEMENT_INSURANCE_PURCHASERS',
    'MT_ATT_CELL_PHONE_CUSTOMER',
    'MT_DO_IT_YOURSELFER',
    'MT_ONLINE_PERSONAL_CARE_PRODUCT_BUYERS',
    'MT_IDENTITY_THEFT_PROTECTION_PURCHASERS',
    'MT_LOW_DOLLAR_DONORS',
    'MT_FRESH_FOOD_DELIVERY_CONSUMERS',
    'MT_REWARDS_CARD_CASH_BACK_USER',
    'MT_HOUSEHOLD_CLEANING_PRODUCTS_BRAND_SWITCHERS',
    'ADVANTAGE_TARGET_NARROW_BAND_INCOME_4_0',
    'MERITSCORE',
    'TARGET_VALUESCORE_20_RETAIL_CARD_MARKETERS',
    'TARGET_VALUESCORE_20_AUTO_FINANCE_MARKETERS',
    'PROPERTY_LOT_SIZE_IN_ACRES',
    'TARGET_VALUESCORE_20_ALL_MARKETERS',
    'TRIGGERVAL_INCOME',
    'NET_WORTH_TIERS',
    'TRIGGERVAL_VALUESCORE',
    'TARGET_VALUESCORE_3_0',
    'ADV_TGT_INCOME_30',
    'TARGET_VALUESCORE_20_BANK_CARD_MARKETERS',
    'ADV_TGT_NARROW_BAND_INCOME_30',
    'HH_PURCHASE_CHANNEL_MO',
    'HH_PURCHASE_CHANNEL_INT',
    'MEMBER_CODE_PERSON',
    'MO_ANY_ALL',
    'OTHER_ONLINE_HOUSEHOLD_ALL',
    'ADV_PREZ_CHILDREN_ENH',
    'ADV_NUM_ADULTS_INDICATOR',
    'ADV_HH_AGE_INDICATOR_ENH',
    'REFIN_INDICATOR',
    'HOME_EQUITY_LOAN_INDICATOR',
    'ADV_IND_MARITAL_STATUS',
    'EMPTY_NESTER_TRIGGER',
    'TYP_CC_ANY_CC',
    'ADV_IND_MARITAL_STATS_INDICATR',
    'ADV_HH_MARITAL_STATUS_INDICATR',
    'MO_BUYER',
    'NICHES_40',
    'MORTG_INTEREST_RATE_TYP_REFIN',
    'MORTG_LOAN_TYP_REFIN',
    'EXTERIOR_WALL_TYP',
    'HOME_HEAT_SOURCE',
    'INCOME_TRIGGER',
    'NICHE_SWITCH_TRIGGER',
    'NICHE_SWITCH_TRIGGER_CHG_TYP',
    'NICHES_50',
    'MATCH_SCORE']

    # local_path = get_asset(asset_name)  
    # scoring_df = pd.read_excel(local_path, usecols=use_cols)
    # local_path = get_asset(asset_name)
    # scoring_df = pd.read_excel(local_path, usecols=use_cols)
    # scoring_df = pd.read_csv("C:/Users/JithuNair/OneDrive - Blend 360/Documents/Projects/Porch/New/match_acx_results.csv")
    scoring_df = generate_and_upload_epsilon_data()
    scoring_df = scoring_df[use_cols]


    #Drop rows where more than 30 percent of column being null to avoid having the model impute them
    scoring_df = scoring_df[scoring_df["MATCH_SCORE"].notna() & (scoring_df.isnull().sum(axis=1) / scoring_df.shape[1] * 100 <= 30)]
    scoring_df.drop(columns = ['MATCH_SCORE'],inplace=True,axis=1)

    # Step 1: Separate individual IDs and addresses from the scoring data
    df_scoring_ids = scoring_df[['CASS_FIRST_NAME','CASS_LAST_NAME','CASS_ADDRESS_LINE_1','CASS_ADDRESS_LINE_2','PERSON_SEQ_NO', 'CONTRACTED_ADDRESS','ZIP','ZIP4','STATE']]  # Edit if the columns change

    top_features_continous=['ACT_NUM_ONLINE_PURCHASE_QUINT',
    'CHANNEL_PREF_RT_CATALOG_QUINT',
    'NUM_GENERATIONS_HH_ENH',
    'AVA_HOME_EQUITY_IN_K',
    'MT_VEHICLE_SERVICE_CENTER_USERS',
    'INDIVIDUAL_EXACT_AGE',
    'MT_MEAL_KIT_DELIVERY_CONSUMERS',
    'MT_GROCERY_STORE_FREQUENTERS',
    'ADV_HH_MARITAL_STATUS',
    'MT_HOME_WARRANTY_PURCHASERS',
    'MT_RETIRED_BUT_STILL_WORKING',
    'ADV_NUM_ADULTS',
    'MORTG_INTEREST_RATE_REFIN',
    'MT_MEDICARE_ADVANTAGE_PLAN_PURCHASERS',
    'CHANNEL_PREF_RT_ONLINE_QUINT',
    '_2020_COMPLETE_CENSUS_GEO',
    'MT_YOGAPILATES_ENTHUSIAST',
    'MT_COMMUNITY_BANK_CUSTR',
    'TGT_NET_WORTH_30',
    'MT_CLICK_TO_CART_HOME_DELIVERY_CUSTOMERS',
    'DWELLING_TYP_LEGACY',
    'MT_MOBILE_PHONE_SERVICE_SWITCHERS',
    'MT_SENIOR_CAREGIVERS',
    'ACT_TOT_DOLLARS_QUINT',
    'MT_SALTY_SNACKS_PURCHASERS',
    'ACT_TOT_ONLINE_DOLLARS_QUINT',
    'MT_LOYAL_FIN_INSTITUTION_CUSTR',
    'MT_WHATS_ON_SALE_SHOPPERS',
    'MT_HEAVY_COUPON_USERS',
    'MT_ONLINE_HOTEL_UPGRADERS',
    'MT_NEW_ROOF_CUSTOMERS',
    'MT_MULTI_RETAILER_SHOPPERS',
    'MT_CASINO_GAMER',
    'MT_VEHICLE_DIYRS',
    'ROOFTOP_LONGITUDE',
    'MT_INSURANCE_FOR_LOAN_PRODUCTS_PURCHASERS',
    'MT_FINANCIALHEALTH_NEWSLETTER_SUBSCRIBERS',
    'NUM_TRADELINES',
    'AGILITY_OCCUPANCY_SCORE',
    'CURRENT_LOAN_TO_VALUE',
    'MT_CABLE_BUNDLE_CABLE_INTERNET_HOME_PHONE',
    'MT_AUTO_INSURANCE_SELF_SERVE_ONLINE_BUYERS',
    '_2010_COMPLETE_CENSUS_GEO_',
    'NUM_DOLLARS_ON_RETURNS',
    'MT_MEAL_PLANNERS',
    'ADV_DWELLING_TYP',
    'MT_AUTO_INSURANCE_CALL_CENTER_SOLD',
    'ROOFTOP_LATITUDE',
    'MT_PRE_SHOP_PLANNERS',
    'MT_MEDICARE_SUPPLEMENT_INSURANCE_PURCHASERS',
    'MT_ATT_CELL_PHONE_CUSTOMER',
    'MT_DO_IT_YOURSELFER',
    'MT_ONLINE_PERSONAL_CARE_PRODUCT_BUYERS',
    'MT_IDENTITY_THEFT_PROTECTION_PURCHASERS',
    'MT_LOW_DOLLAR_DONORS',
    'MT_FRESH_FOOD_DELIVERY_CONSUMERS',
    'MT_REWARDS_CARD_CASH_BACK_USER',
    'MT_HOUSEHOLD_CLEANING_PRODUCTS_BRAND_SWITCHERS']

    top_features_ordinal=['ADVANTAGE_TARGET_NARROW_BAND_INCOME_4_0',
    'MERITSCORE',
    'TARGET_VALUESCORE_20_RETAIL_CARD_MARKETERS',
    'TARGET_VALUESCORE_20_AUTO_FINANCE_MARKETERS',
    'PROPERTY_LOT_SIZE_IN_ACRES',
    'TARGET_VALUESCORE_20_ALL_MARKETERS',
    'TRIGGERVAL_INCOME',
    'NET_WORTH_TIERS',
    'TRIGGERVAL_VALUESCORE',
    'TARGET_VALUESCORE_3_0',
    'ADV_TGT_INCOME_30',
    'TARGET_VALUESCORE_20_BANK_CARD_MARKETERS',
    'ADV_TGT_NARROW_BAND_INCOME_30']

    top_features_binary=['HH_PURCHASE_CHANNEL_MO',
    'HH_PURCHASE_CHANNEL_INT',
    'MEMBER_CODE_PERSON',
    'MO_ANY_ALL',
    'OTHER_ONLINE_HOUSEHOLD_ALL',
    'ADV_PREZ_CHILDREN_ENH',
    'ADV_NUM_ADULTS_INDICATOR',
    'ADV_HH_AGE_INDICATOR_ENH',
    'REFIN_INDICATOR',
    'HOME_EQUITY_LOAN_INDICATOR',
    'ADV_IND_MARITAL_STATUS',
    'EMPTY_NESTER_TRIGGER',
    'TYP_CC_ANY_CC',
    'ADV_IND_MARITAL_STATS_INDICATR',
    'ADV_HH_MARITAL_STATUS_INDICATR',
    'MO_BUYER']

    top_features_categorical=['NICHES_40',
    'MORTG_INTEREST_RATE_TYP_REFIN',
    'MORTG_LOAN_TYP_REFIN',
    'EXTERIOR_WALL_TYP',
    'HOME_HEAT_SOURCE',
    'INCOME_TRIGGER',
    'NICHE_SWITCH_TRIGGER',
    'NICHE_SWITCH_TRIGGER_CHG_TYP',
    'NICHES_50']


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
        # Numeric swaps
        'MEMBER_CODE_PERSON': {1: 1, 3: 0},
        'ADV_IND_MARITAL_STATUS': {1: 1, 2: 0},
        'ADV_PREZ_CHILDREN_ENH': {1: 1, 0: 0},
        'REFIN_INDICATOR': {1: 1, 0: 0},
        'HOME_EQUITY_LOAN_INDICATOR': {1: 1, 0: 0},

        # Y vs NaN or N
        'SELF_REPORTED_RESPONDER': {'Y': 1, np.nan: 0},
        'OCREDIT_FIN_SERVICES_BANKING': {'Y': 1, np.nan: 0},
        'OCREDIT_FIN_SERVICES_INSTALL': {'Y': 1, np.nan: 0},
        'BOOK_BOOKS_READING_ALL': {'Y': 1, np.nan: 0},
        'DONOR_DONATE_CHARIT_CAUSES_ALL': {'Y': 1, np.nan: 0},
        'COLLECT_ANY_ALL': {'Y': 1, np.nan: 0},
        'HOBBIES_COOKING_ALL': {'Y': 1, np.nan: 0},
        'HOBBIES_CRAFTS_ALL': {'Y': 1, np.nan: 0},
        'HOBBIES_GARDENING_ALL': {'Y': 1, np.nan: 0},
        'HOBBIES_ANY_ALL': {'Y': 1, np.nan: 0},
        'HOBBIES_HOME_IMPROV_DIY_ALL': {'Y': 1, np.nan: 0},
        'MO_ANY_ALL': {'Y': 1, np.nan: 0},
        'MO_APPRL_ALL': {'Y': 1, np.nan: 0},
        'MO_BOOKS_ALL': {'Y': 1, np.nan: 0},
        'MO_FOOD_ALL': {'Y': 1, np.nan: 0},
        'MO_HEALTH_BEAUTY_PROD_ALL': {'Y': 1, np.nan: 0},
        'MO_HOME_FURNISHING_ALL': {'Y': 1, np.nan: 0},
        'MO_MAGAZINES_ALL': {'Y': 1, np.nan: 0},
        'MUSIC_ANY_ALL': {'Y': 1, np.nan: 0},
        'NUTRITION_NUTRITION_DIET_ALL': {'Y': 1, np.nan: 0},
        'OTHER_ELECTRONICS_ALL': {'Y': 1, np.nan: 0},
        'OTHER_ONLINE_HOUSEHOLD_ALL': {'Y': 1, np.nan: 0},
        'PETS_PETS_ALL': {'Y': 1, np.nan: 0},
        'SPORTS_FISHING_ALL': {'Y': 1, np.nan: 0},
        'SPORTS_SPORT_PARTICIPATION_ALL': {'Y': 1, np.nan: 0},
        'TRAVEL_TRAVEL_IN_THE_USA_ALL': {'Y': 1, np.nan: 0},
        'TRAVEL_TRAVELER_ALL': {'Y': 1, np.nan: 0},
        'MO_BUYER': {'Y': 1, np.nan: 0},
        'HH_PURCHASE_CHANNEL_INT': {'Y': 1, np.nan: 0},
        'HH_PURCHASE_CHANNEL_MO': {'Y': 1, np.nan: 0},
        'CLUB_CONTINUITY_BUYER': {'Y': 1, np.nan: 0},
        'PAYMENT_METHOD_CC': {'Y': 1, np.nan: 0},
        'TYP_CC_ANY_CC': {'Y': 1, np.nan: 0},
        'TYP_CC_BANK_CARD': {'Y': 1, np.nan: 0},
        'TYP_CC_FIN_CO_CARD': {'Y': 1, np.nan: 0},
        'TYP_CC_STD_RETAIL': {'Y': 1, np.nan: 0},
        'LIFECYCLE_BABY_BOOMERS': {'Y': 1, np.nan: 0},
        'LIFECYCLE_GENERATION_X': {'Y': 1, np.nan: 0},
        'LIFECYCLE_MILLENNIALS': {'Y': 1, np.nan: 0},

        # S vs H
        'ADV_IND_MARITAL_STATS_INDICATR': {'S': 1, 'H': 0},
        'ADV_HH_EDU_INDICATOR_ENH': {'H': 1, 'S': 0},
        'ADV_HH_AGE_INDICATOR_ENH': {'S': 1, 'H': 0},
        'ADV_HH_SIZE_INDICATOR_ENH': {'S': 1, 'H': 0},
        'ADV_PREZ_CHILDREN_INDICATR_ENH': {'H': 1, 'S': 0},
        'ADV_NUM_ADULTS_INDICATOR': {'S': 1, 'H': 0},
        'ADV_HH_MARITAL_STATUS_INDICATR': {'S': 1, 'H': 0},
        'ADV_DWELLING_TYP_INDICATOR': {'S': 1, np.nan: 0},
        'ADV_TGT_INCOME_INDICATOR_30': {'S': 1, 'H': 0},
        'INCOME_TIERS_INDICATOR': {'S': 1, 'H': 0},
        'ADVANTAGE_TARGET_INCOME_INDICATOR_4_0': {'S': 1, 'H': 0},

        # N vs Y
        'COLLEGE_GRAD_TRIGGER': {'Y': 1, 'N': 0},
        'EMPTY_NESTER_TRIGGER': {'Y': 1, 'N': 0},
        'NEW_FIRST_CHILD_0_2_TRIGGER': {'Y': 1, 'N': 0},
        'NEW_ADULT_TO_FILE_TRIGGER': {'Y': 1, 'N': 0},
        'NEW_YNG_ADULT_TO_FILE_TRIGGER': {'Y': 1, 'N': 0},
        'NEW_MARRIED_TRIGGER': {'Y': 1, 'N': 0},
        'NEW_SINGLE_TRIGGER': {'Y': 1, 'N': 0},
    }

    for col in top_features_binary:
        if col in binary_mappings_base:
            scoring_df[col] = scoring_df[col].map(binary_mappings_base[col])


    scoring_df[top_features_categorical] = scoring_df[top_features_categorical].fillna('unknown')

    for col in top_features_ordinal:
        mode_value = scoring_df[col].mode(dropna=True)[0]  # Get the most frequent value
        scoring_df[col].fillna(mode_value, inplace=True)


    scoring_df['ADV_TGT_INCOME_30'] = scoring_df['ADV_TGT_INCOME_30'].map({'1': 1,'2': 2,'3': 3,'4': 4,'5': 5,'6': 6,'7': 7,'8': 8,'9': 9,'A': 10,'B': 11,'C': 12,'D': 13})
    scoring_df['ADV_TGT_NARROW_BAND_INCOME_30'] = scoring_df['ADV_TGT_NARROW_BAND_INCOME_30'].map({
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, 
        '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 
        'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 
        'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 
        'W': 32, 'X': 33, 'Y': 34, 'Z': 35
    })
    scoring_df['MERITSCORE'] = scoring_df['MERITSCORE'].map({
        'A1': 12, 'A2': 11, 
        'B1': 10, 'B2': 9, 
        'C1': 8, 'C2': 7, 
        'D1': 6, 'D2': 5, 'D3': 4,  
        'E1': 3, 'E2':2, 'E3': 1 
    })
    scoring_df['PROPERTY_LOT_SIZE_IN_ACRES'] = scoring_df['PROPERTY_LOT_SIZE_IN_ACRES'].map({
        'A': 1,   # < .26 acres
        'B': 2,   # .26 - .59 acres
        'C': 3,   # .60 - .99 acres
        'D': 4,   # 1 acre
        'E': 5,   # 2 acres
        'F': 6,   # 3 acres
        'G': 7,   # 4 acres
        'H': 8,   # 5 acres
        'I': 9,   # 6 - 10 acres
        'J': 10,  # 11 - 20 acres
        'K': 11,  # 21 - 40 acres
        'L': 12,  # > 40 acres
    })
    scoring_df['TRIGGERVAL_INCOME'] = scoring_df['TRIGGERVAL_INCOME'].map({
        '1': 1,   # $0 - $14,999
        '2': 2,   # $15,000 - $19,999
        '3': 3,   # $20,000 - $29,999
        '4': 4,   # $30,000 - $39,999
        '5': 5,   # $40,000 - $49,999
        '6': 6,   # $50,000 - $74,999
        '7': 7,   # $75,000 - $99,999
        '8': 8,   # $100,000 - $124,999
        '9': 9,   # $125,000 - $149,999
        'A': 10,  # $150,000 - $174,999
        'B': 11,  # $175,000 - $199,999
        'C': 12,  # $200,000 - $249,999
        'D': 13,  # $250,000 and more
    })
    scoring_df['TRIGGERVAL_VALUESCORE'] = scoring_df['TRIGGERVAL_VALUESCORE'].map({
        'A1': 12,   # Best Profit Margin
        'A2': 11,   # Best Profit Margin
        'B1': 10,   # Above Average Profit Margin
        'B2': 9,   # Above Average Profit Margin
        'C1': 8,   # Average Profit Margin
        'C2': 7,   # Average Profit Margin
        'D1': 6,   # Below Average Profit Margin
        'D2': 5,   # Below Average Profit Margin
        'D3': 4,   # Below Average Profit Margin
        'E1': 3,   # Poor Profit Margin
        'E2': 2,   # Poor Profit Margin
        'E3': 1   # Poor Profit Margin
    })

    scoring_df['TARGET_VALUESCORE_20_ALL_MARKETERS'] = scoring_df['TARGET_VALUESCORE_20_ALL_MARKETERS'].map({
        'A1': 12,   # Best Profit Margin
        'A2': 11,   # Best Profit Margin
        'B1': 10,   # Above Average Profit Margin
        'B2': 9,   # Above Average Profit Margin
        'C1': 8,   # Average Profit Margin
        'C2': 7,   # Average Profit Margin
        'D1': 6,   # Below Average Profit Margin
        'D2': 5,   # Below Average Profit Margin
        'D3': 4,   # Below Average Profit Margin
        'E1': 3,   # Poor Profit Margin
        'E2': 2,   # Poor Profit Margin
        'E3': 1   # Poor Profit Margin
    })
    scoring_df['TARGET_VALUESCORE_20_AUTO_FINANCE_MARKETERS'] = scoring_df['TARGET_VALUESCORE_20_AUTO_FINANCE_MARKETERS'].map({
        'A1': 12,   # Best Profit Margin
        'A2': 11,   # Best Profit Margin
        'B1': 10,   # Above Average Profit Margin
        'B2': 9,   # Above Average Profit Margin
        'C1': 8,   # Average Profit Margin
        'C2': 7,   # Average Profit Margin
        'D1': 6,   # Below Average Profit Margin
        'D2': 5,   # Below Average Profit Margin
        'D3': 4,   # Below Average Profit Margin
        'E1': 3,   # Poor Profit Margin
        'E2': 2,   # Poor Profit Margin
        'E3': 1   # Poor Profit Margin
    })
    scoring_df['TARGET_VALUESCORE_20_BANK_CARD_MARKETERS'] = scoring_df['TARGET_VALUESCORE_20_BANK_CARD_MARKETERS'].map({
        'A1': 12,   # Best Profit Margin
        'A2': 11,   # Best Profit Margin
        'B1': 10,   # Above Average Profit Margin
        'B2': 9,   # Above Average Profit Margin
        'C1': 8,   # Average Profit Margin
        'C2': 7,   # Average Profit Margin
        'D1': 6,   # Below Average Profit Margin
        'D2': 5,   # Below Average Profit Margin
        'D3': 4,   # Below Average Profit Margin
        'E1': 3,   # Poor Profit Margin
        'E2': 2,   # Poor Profit Margin
        'E3': 1   # Poor Profit Margin
    })
    scoring_df['TARGET_VALUESCORE_20_RETAIL_CARD_MARKETERS'] = scoring_df['TARGET_VALUESCORE_20_RETAIL_CARD_MARKETERS'].map({
        'A1': 12,   # Best Profit Margin
        'A2': 11,   # Best Profit Margin
        'B1': 10,   # Above Average Profit Margin
        'B2': 9,   # Above Average Profit Margin
        'C1': 8,   # Average Profit Margin
        'C2': 7,   # Average Profit Margin
        'D1': 6,   # Below Average Profit Margin
        'D2': 5,   # Below Average Profit Margin
        'D3': 4,   # Below Average Profit Margin
        'E1': 3,   # Poor Profit Margin
        'E2': 2,   # Poor Profit Margin
        'E3': 1   # Poor Profit Margin
    })

    scoring_df['NET_WORTH_TIERS'] = scoring_df['NET_WORTH_TIERS'].map({
        '0': 0,    # Under $25,000
        '1': 1,    # $25,000 - $49,999
        '2': 2,    # $50,000 - $74,999
        '3': 3,    # $75,000 - $99,999
        '4': 4,    # $100,000 - $149,999
        '5': 5,    # $150,000 - $249,999
        '6': 6,    # $250,000 - $499,999
        '7': 7,    # $500,000 - $749,999
        '8': 8,    # $750,000 - $999,999
        '9': 9,    # $1,000,000 - $1,999,999
        'A': 10,   # $2,000,000 - $2,999,999
        'B': 11   # Over $3,000,000
    })

    scoring_df['ADVANTAGE_TARGET_NARROW_BAND_INCOME_4_0'] = scoring_df['ADVANTAGE_TARGET_NARROW_BAND_INCOME_4_0'].map({
        '0': 1,    # Less than $14,999
        '1': 2,    # $15,000 - $19,999
        '2': 3,    # $20,000 - $24,999
        '3': 4,    # $25,000 - $29,999
        '4': 5,    # $30,000 - $34,999
        '5': 6,    # $35,000 - $39,999
        '6': 7,    # $40,000 - $44,999
        '7': 8,    # $45,000 - $49,999
        '8': 9,    # $50,000 - $54,999
        '9': 10,   # $55,000 - $59,999
        'A': 11,   # $60,000 - $64,999
        'B': 12,   # $65,000 - $69,999
        'C': 13,   # $70,000 - $74,999
        'D': 14,   # $75,000 - $79,999
        'E': 15,   # $80,000 - $84,999
        'F': 16,   # $85,000 - $89,999
        'G': 17,   # $90,000 - $94,999
        'H': 18,   # $95,000 - $99,999
        'I': 19,   # $100,000 - $104,999
        'J': 20,   # $105,000 - $109,999
        'K': 21,   # $110,000 - $114,999
        'L': 22,   # $115,000 - $119,999
        'M': 23,   # $120,000 - $124,999
        'N': 24,   # $125,000 - $129,999
        'O': 25,   # $130,000 - $134,999
        'P': 26,   # $135,000 - $139,999
        'Q': 27,   # $140,000 - $144,999
        'R': 28,   # $145,000 - $149,999
        'S': 29,   # $150,000 - $159,999
        'T': 30,   # $160,000 - $169,999
        'U': 31,   # $170,000 - $174,999
        'V': 32,   # $175,000 - $189,999
        'W': 33,   # $190,000 - $199,999
        'X': 34,   # $200,000 - $249,999
        'Y': 35,   # $250,000 - $499,999
        'Z': 36   # $500,000 or More
    })
    scoring_df['TARGET_VALUESCORE_3_0'] = scoring_df['TARGET_VALUESCORE_3_0'].map({
        'A1': 12,   # Best Profit Margin
        'A2': 11,   # Best Profit Margin
        'B1': 10,   # Above Average Profit Margin
        'B2': 9,   # Above Average Profit Margin
        'C1': 8,   # Average Profit Margin
        'C2': 7,   # Average Profit Margin
        'D1': 6,   # Below Average Profit Margin
        'D2': 5,   # Below Average Profit Margin
        'D3': 4,   # Below Average Profit Margin
        'E1': 3,   # Poor Profit Margin
        'E2': 2,   # Poor Profit Margin
        'E3': 1   # Poor Profit Margin
    })

    # Loop through each variable and impute based on skewness
    for var in top_features_continous:
        skew_val = scoring_df[var].skew()
        if abs(skew_val) > 1:
            # Highly skewed → use median
            scoring_df[var].fillna(scoring_df[var].median(), inplace=True)
        else:
            # Fairly symmetric → use mean
            scoring_df[var].fillna(scoring_df[var].mean(), inplace=True)

    scoring_df = pd.get_dummies(
        scoring_df,
        columns=top_features_categorical,
        drop_first=False,
        dtype='int8'  # or 'uint8' if all values are non-negative
    )


    X_train_top_columns=['ACT_NUM_ONLINE_PURCHASE_QUINT',
    'CHANNEL_PREF_RT_CATALOG_QUINT',
    'HH_PURCHASE_CHANNEL_MO',
    'HH_PURCHASE_CHANNEL_INT',
    'MEMBER_CODE_PERSON',
    'ADVANTAGE_TARGET_NARROW_BAND_INCOME_4_0',
    'MERITSCORE',
    'NUM_GENERATIONS_HH_ENH',
    'AVA_HOME_EQUITY_IN_K',
    'MO_ANY_ALL',
    'MORTG_LOAN_TYP_REFIN_unknown',
    'OTHER_ONLINE_HOUSEHOLD_ALL',
    'MT_VEHICLE_SERVICE_CENTER_USERS',
    'ADV_PREZ_CHILDREN_ENH',
    'INDIVIDUAL_EXACT_AGE',
    'MT_MEAL_KIT_DELIVERY_CONSUMERS',
    'TARGET_VALUESCORE_20_RETAIL_CARD_MARKETERS',
    'MORTG_INTEREST_RATE_TYP_REFIN_unknown',
    'MT_GROCERY_STORE_FREQUENTERS',
    'ADV_HH_MARITAL_STATUS',
    'MT_HOME_WARRANTY_PURCHASERS',
    'MT_RETIRED_BUT_STILL_WORKING',
    'MORTG_LOAN_TYP_REFIN_F',
    'ADV_NUM_ADULTS',
    'TARGET_VALUESCORE_20_AUTO_FINANCE_MARKETERS',
    'MORTG_INTEREST_RATE_REFIN',
    'ADV_NUM_ADULTS_INDICATOR',
    'MT_MEDICARE_ADVANTAGE_PLAN_PURCHASERS',
    'CHANNEL_PREF_RT_ONLINE_QUINT',
    'PROPERTY_LOT_SIZE_IN_ACRES',
    'ADV_HH_AGE_INDICATOR_ENH',
    'MORTG_LOAN_TYP_REFIN_V',
    'EXTERIOR_WALL_TYP_unknown',
    'TARGET_VALUESCORE_20_ALL_MARKETERS',
    '_2020_COMPLETE_CENSUS_GEO',
    'MT_YOGAPILATES_ENTHUSIAST',
    'MT_COMMUNITY_BANK_CUSTR',
    'TGT_NET_WORTH_30',
    'REFIN_INDICATOR',
    'MT_CLICK_TO_CART_HOME_DELIVERY_CUSTOMERS',
    'DWELLING_TYP_LEGACY',
    'TRIGGERVAL_INCOME',
    'MT_MOBILE_PHONE_SERVICE_SWITCHERS',
    'HOME_EQUITY_LOAN_INDICATOR',
    'NET_WORTH_TIERS',
    'ADV_IND_MARITAL_STATUS',
    'EMPTY_NESTER_TRIGGER',
    'HOME_HEAT_SOURCE_unknown',
    'MT_SENIOR_CAREGIVERS',
    'ACT_TOT_DOLLARS_QUINT',
    'TRIGGERVAL_VALUESCORE',
    'TYP_CC_ANY_CC',
    'TARGET_VALUESCORE_3_0',
    'MT_SALTY_SNACKS_PURCHASERS',
    'ACT_TOT_ONLINE_DOLLARS_QUINT',
    'MT_LOYAL_FIN_INSTITUTION_CUSTR',
    'NICHES_50_C3',
    'ADV_IND_MARITAL_STATS_INDICATR',
    'INCOME_TRIGGER_unknown',
    'MT_WHATS_ON_SALE_SHOPPERS',
    'MT_HEAVY_COUPON_USERS',
    'INCOME_TRIGGER_I',
    'MT_ONLINE_HOTEL_UPGRADERS',
    'MT_NEW_ROOF_CUSTOMERS',
    'MT_MULTI_RETAILER_SHOPPERS',
    'MT_CASINO_GAMER',
    'MT_VEHICLE_DIYRS',
    'ROOFTOP_LONGITUDE',
    'MT_INSURANCE_FOR_LOAN_PRODUCTS_PURCHASERS',
    'MT_FINANCIALHEALTH_NEWSLETTER_SUBSCRIBERS',
    'NUM_TRADELINES',
    'ADV_HH_MARITAL_STATUS_INDICATR',
    'AGILITY_OCCUPANCY_SCORE',
    'ADV_TGT_INCOME_30',
    'MO_BUYER',
    'CURRENT_LOAN_TO_VALUE',
    'MT_CABLE_BUNDLE_CABLE_INTERNET_HOME_PHONE',
    'MT_AUTO_INSURANCE_SELF_SERVE_ONLINE_BUYERS',
    '_2010_COMPLETE_CENSUS_GEO_',
    'NUM_DOLLARS_ON_RETURNS',
    'MT_MEAL_PLANNERS',
    'ADV_DWELLING_TYP',
    'MT_AUTO_INSURANCE_CALL_CENTER_SOLD',
    'ROOFTOP_LATITUDE',
    'MT_PRE_SHOP_PLANNERS',
    'MT_MEDICARE_SUPPLEMENT_INSURANCE_PURCHASERS',
    'MT_ATT_CELL_PHONE_CUSTOMER',
    'NICHE_SWITCH_TRIGGER_CHG_TYP_I',
    'MT_DO_IT_YOURSELFER',
    'MT_ONLINE_PERSONAL_CARE_PRODUCT_BUYERS',
    'MT_IDENTITY_THEFT_PROTECTION_PURCHASERS',
    'TARGET_VALUESCORE_20_BANK_CARD_MARKETERS',
    'MT_LOW_DOLLAR_DONORS',
    'MT_FRESH_FOOD_DELIVERY_CONSUMERS',
    'MT_REWARDS_CARD_CASH_BACK_USER',
    'MORTG_INTEREST_RATE_TYP_REFIN_V',
    'MT_HOUSEHOLD_CLEANING_PRODUCTS_BRAND_SWITCHERS',
    'ADV_TGT_NARROW_BAND_INCOME_30']


    scoring_df=scoring_df[X_train_top_columns]
    # scoring_df.head()

    return scoring_df, df_scoring_ids