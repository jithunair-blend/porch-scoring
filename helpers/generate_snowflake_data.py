import os
import pandas as pd
import boto3
import snowflake.connector
from helpers.read_s3 import get_asset
from datetime import datetime
from dotenv import load_dotenv  # <-- add this
from config import config

# ✅ Load environment variables from .env file
load_dotenv()

ctx = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    role=os.getenv("SNOWFLAKE_ROLE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA"),  # <-- was wrong: should not be 'PORCH_ADMIN'
)
cur = ctx.cursor()

def stage_cass_file_to_snowflake(cur):
    """
    Retrieves the CASS CSV from S3 (via get_asset) and uploads it into
    Snowflake stage @DATA_STAGE/porch-test-data for downstream COPY INTO use.
    """
    # 1️⃣ Get local copy of the CASS file (cached if already downloaded)
    local_path = get_asset("original_cass_csv")
    print(f"✅ Local CASS file ready at {local_path}")

    # 2️⃣ Set Snowflake context
    cur.execute("USE ROLE PORCH_ADMIN;")
    cur.execute("USE DATABASE SANDBOX;")
    cur.execute("USE SCHEMA DS;")

    # 3️⃣ Upload file to Snowflake stage
    # Escape Windows backslashes and quote the path
    escaped_path = local_path.replace("\\", "\\\\")
    put_cmd = f"""
        PUT 'file://{escaped_path}' @DATA_STAGE/porch-test-data
        AUTO_COMPRESS = FALSE
        OVERWRITE = TRUE
    """

    print("⏫ Uploading CASS file to Snowflake stage ...")
    cur.execute(put_cmd)
    print("✅ File successfully staged to @DATA_STAGE/porch-test-data.")

def generate_and_upload_acxiom_data():
    # 🔹 Stage CASS file automatically
    stage_cass_file_to_snowflake(cur)
    sql_script = """
    USE ROLE PORCH_ADMIN;
    USE DATABASE SANDBOX;
    USE SCHEMA DS;

    ------------------------------------------------------------
    -- 1) Create and load CASS input with clustering
    ------------------------------------------------------------
    DROP TABLE IF EXISTS CASS_INPUT_NEW;

    CREATE OR REPLACE TABLE CASS_INPUT_NEW (
        RECORD_ID STRING,
        FIRST_NAME STRING,
        LAST_NAME STRING,
        ADDRESS_LINE_1 STRING,
        ADDRESS_LINE_2 STRING,
        CITY STRING,
        STATE STRING,
        ZIPCODE STRING,
        ERROR_CODE STRING
    )
    CLUSTER BY (LEFT(ZIPCODE,5), STATE);

    COPY INTO CASS_INPUT_NEW
    FROM @DATA_STAGE/porch-test-data/AHP-2025-Converters_CASS.csv
    FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY='"' SKIP_HEADER=1);

    ------------------------------------------------------------
    -- 2) Create optimized Acxiom subset with pre-filtering
    ------------------------------------------------------------
    CREATE OR REPLACE TEMP TABLE ACX_SUBSET AS
    SELECT
        FIRST_NAME,
        LAST_NAME,
        RECORD_KEY,
        ADDRESS_LINE2,
        CONCAT_ADDRESS,
        ZIP_CODE,
        ZIP4_CODE,
        STATE,
        UPPER(LEFT(LAST_NAME, 3)) AS LAST_NAME_PREFIX,  -- Pre-compute for blocking
        AP001241, IBE8707, IBE8579, AP004950, IBE4100, AP009353, AP008717, AP009796, IBE8652,
        AP009356, AP000920, AP003446, AP000572, IBE8463, IBE8589, IBE2526, AP000903, IBE8850,
        AP008718, AP004959, AP000953, IBE8702, AP006249_01, AP004988, AP000482, AP009355, IBE2800,
        AP006334, IBE8592, AP003588, AP009795, AP006246_01, IBE8706, AP002897, AP001242, AP001238,
        AP009803, IBE8614, AP009354, IBE8436, IBE9356, IBE8584, IBE8586, IBE8671, IBE8567, IBE8763,
        IBE8440, IBE8437, IBE8642, IBE3700, IBE7777, IBE3699, IBE8623_03, IBE7801, IBE2062_02,
        IBE7796, IBE6145, IBE8167_21, IBE2061_02, IBE8578, IBE8271, IBE8611_03, IBE8621_01,
        IBE9153_12, IBE7823, IBE7720, IBE7817, IBE7753, IBE6505, IBE7829, HOME_OWNER_SOURCE_FLAG,
        IBE8688, IBE2360, IBE7622_01, IBE9515, IBE8625, IBE8701, IBE8581, PX011283_01,
        PX011284_01, PX011282_01
    FROM C2G_LAKE_DEV.VDS_FTP.ACXIOM_LATEST
    WHERE LEFT(ZIP_CODE,5) IN (SELECT DISTINCT LEFT(ZIPCODE,5) FROM CASS_INPUT_NEW)
    AND STATE IN (SELECT DISTINCT STATE FROM CASS_INPUT_NEW)
    AND FIRST_NAME IS NOT NULL
    AND LAST_NAME IS NOT NULL
    AND CONCAT_ADDRESS IS NOT NULL
    AND TRIM(CONCAT_ADDRESS) != '';

    ------------------------------------------------------------
    -- 3) Create MATCH_RESULTS with blocking and optimized scoring
    ------------------------------------------------------------
    CREATE OR REPLACE TABLE MATCH_RESULTS AS
    WITH zip_filtered AS (
        SELECT
            c.RECORD_ID      AS CASS_RECORD_ID,
            c.FIRST_NAME     AS CASS_FIRST_NAME,
            c.LAST_NAME      AS CASS_LAST_NAME,
            c.ADDRESS_LINE_1 AS CASS_ADDRESS_LINE_1,
            c.ADDRESS_LINE_2 AS CASS_ADDRESS_LINE_2,
            c.CITY           AS CASS_CITY,
            c.STATE          AS CASS_STATE,
            c.ZIPCODE        AS CASS_ZIPCODE,
            c.ERROR_CODE     AS CASS_ERROR_CODE,
            a.RECORD_KEY,
            a.FIRST_NAME     AS ACX_FIRST_NAME,
            a.LAST_NAME      AS ACX_LAST_NAME,
            a.CONCAT_ADDRESS AS ACX_ADDRESS,
            a.ZIP_CODE       AS ACX_ZIP_CODE,
            a.STATE          AS ACX_STATE
        FROM CASS_INPUT_NEW c
        JOIN ACX_SUBSET a
        ON LEFT(c.ZIPCODE,5) = LEFT(a.ZIP_CODE,5)
        AND c.STATE = a.STATE
        AND UPPER(LEFT(c.LAST_NAME, 3)) = a.LAST_NAME_PREFIX  -- Blocking: reduce join size by 95%+
    ),
    scored AS (
        SELECT
            zf.*,
        -- Name matching using Jaro-Winkler (returns 0–1 by default)
        ROUND(JAROWINKLER_SIMILARITY(
            UPPER(TRIM(zf.CASS_FIRST_NAME)),
            UPPER(TRIM(zf.ACX_FIRST_NAME))
        ), 2) AS MATCH_FIRST,

        ROUND(JAROWINKLER_SIMILARITY(
            UPPER(TRIM(zf.CASS_LAST_NAME)),
            UPPER(TRIM(zf.ACX_LAST_NAME))
        ), 2) AS MATCH_LAST,

        -- Address match using normalized Edit Distance
        ROUND(
            100 * GREATEST(
                0,
                1 - LEAST(
                    EDITDISTANCE(
                        REGEXP_REPLACE(UPPER(TRIM(zf.CASS_ADDRESS_LINE_1 || ' ' || NVL(zf.CASS_ADDRESS_LINE_2,''))), '[^A-Z0-9 ]', ''),
                        REGEXP_REPLACE(UPPER(TRIM(zf.ACX_ADDRESS)), '[^A-Z0-9 ]', '')
                    )
                    / NULLIF(GREATEST(
                        LENGTH(REGEXP_REPLACE(UPPER(TRIM(zf.CASS_ADDRESS_LINE_1 || ' ' || NVL(zf.CASS_ADDRESS_LINE_2,''))), '[^A-Z0-9 ]', '')),
                        LENGTH(REGEXP_REPLACE(UPPER(TRIM(zf.ACX_ADDRESS)), '[^A-Z0-9 ]', ''))
                    ), 0),
                    1
                )
            ), 2
        ) AS MATCH_ADDR
    FROM zip_filtered zf
    ),
    final_scored AS (
        SELECT
            *,
            ROUND(
                (0.65 * (MATCH_ADDR / 100.0) + 0.25 * (MATCH_LAST/ 100.0) + 0.10 * (MATCH_FIRST/ 100.0)) * 100, 
                2
            ) AS MATCH_SCORE
        FROM scored
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY CASS_RECORD_ID ORDER BY MATCH_SCORE DESC) AS rn
        FROM final_scored
        WHERE MATCH_SCORE >= 70  -- Filter out low scores early
    )
    SELECT
        CASS_RECORD_ID,
        CASS_FIRST_NAME,
        ACX_FIRST_NAME,
        CASS_LAST_NAME,
        ACX_LAST_NAME,
        CASS_ADDRESS_LINE_1,
        CASS_ADDRESS_LINE_2,
        ACX_ADDRESS,
        CASS_CITY,
        CASS_STATE,
        CASS_ZIPCODE,
        CASS_ERROR_CODE,
        RECORD_KEY AS ACX_RECORD_KEY,
        MATCH_FIRST,
        MATCH_LAST,
        MATCH_ADDR,
        MATCH_SCORE,
        CASE
        WHEN MATCH_SCORE >= 88 THEN 'High'
        WHEN MATCH_SCORE >= 78 THEN 'Medium'
        ELSE 'Low'
        END AS MATCH_LEVEL
    FROM ranked
    WHERE rn = 1;

    ------------------------------------------------------------
    -- 4) Join back only the selected Acxiom columns AFTER matching
    ------------------------------------------------------------
    CREATE OR REPLACE TABLE MATCH_RESULTS_ENRICHED AS
    SELECT
        m.*,
        -- a.CUSTOMER_ID,
        a.RECORD_KEY,
        a.ADDRESS_LINE2,
        a.CONCAT_ADDRESS,
        a.ZIP_CODE,
        a.ZIP4_CODE,
        a.STATE,
        a.AP001241, a.IBE8707, a.IBE8579, a.AP004950, a.IBE4100, a.AP009353, a.AP008717, a.AP009796, a.IBE8652,
        a.AP009356, a.AP000920, a.AP003446, a.AP000572, a.IBE8463, a.IBE8589, a.IBE2526, a.AP000903, a.IBE8850,
        a.AP008718, a.AP004959, a.AP000953, a.IBE8702, a.AP006249_01, a.AP004988, a.AP000482, a.AP009355, a.IBE2800,
        a.AP006334, a.IBE8592, a.AP003588, a.AP009795, a.AP006246_01, a.IBE8706, a.AP002897, a.AP001242, a.AP001238,
        a.AP009803, a.IBE8614, a.AP009354, a.IBE8436, a.IBE9356, a.IBE8584, a.IBE8586, a.IBE8671, a.IBE8567, a.IBE8763,
        a.IBE8440, a.IBE8437, a.IBE8642, a.IBE3700, a.IBE7777, a.IBE3699, a.IBE8623_03, a.IBE7801, a.IBE2062_02,
        a.IBE7796, a.IBE6145, a.IBE8167_21, a.IBE2061_02, a.IBE8578, a.IBE8271, a.IBE8611_03, a.IBE8621_01,
        a.IBE9153_12, a.IBE7823, a.IBE7720, a.IBE7817, a.IBE7753, a.IBE6505, a.IBE7829, a.HOME_OWNER_SOURCE_FLAG,
        a.IBE8688, a.IBE2360, a.IBE7622_01, a.IBE9515, a.IBE8625, a.IBE8701, a.IBE8581, a.PX011283_01,
        a.PX011284_01, a.PX011282_01
    FROM MATCH_RESULTS m
    JOIN C2G_LAKE_DEV.VDS_FTP.ACXIOM_LATEST a
    ON a.RECORD_KEY = m.ACX_RECORD_KEY;

    SELECT * FROM MATCH_RESULTS_ENRICHED;
    """
    # 🔹 Split on semicolons and execute one by one
    statements = [stmt.strip() for stmt in sql_script.split(';') if stmt.strip()]

    last_result = None
    # for i, stmt in enumerate(statements, 1):
    #     print(f"Running statement {i}/{len(statements)}: {stmt[:60]}...")
    #     cur.execute(stmt)
    #     if stmt.lower().startswith("select"):
    #         last_result = cur.fetch_pandas_all()

    for i, stmt in enumerate(statements, 1):
        print(f"Running statement {i}/{len(statements)}: {stmt[:60]}...")
        cur.execute(stmt)
        if stmt.strip().lower().startswith("select"):
            last_result = cur.fetch_pandas_all()


    if last_result is not None:
        # last_result.to_csv("match_acx_results.csv", index=False)
        print("✅ Results written to last_result dataframe")
        return last_result
    else:
        print("⚠️ No SELECT result found.")

def generate_and_upload_epsilon_data():
    sql_script = """
    USE ROLE PORCH_ADMIN;
    USE DATABASE SANDBOX;
    USE SCHEMA DS;

    ------------------------------------------------------------
    -- 1) Ensure CASS_INPUT_NEW exists (reuse from Acxiom pipeline)
    ------------------------------------------------------------
    -- (Assumes CASS_INPUT_NEW already loaded and clustered as before from the Acxiom run)

    ------------------------------------------------------------
    -- 2) Create optimized EPS_SUBSET with pre-filtering and blocking
    ------------------------------------------------------------
    CREATE OR REPLACE TEMP TABLE EPS_SUBSET AS
    SELECT
        PERSON_SEQ_NO,
        GIVEN_NAME,
        SURNAME,
        CONTRACTED_ADDRESS,
        ZIP,
        ZIP4,
        STATE,
        UPPER(LEFT(SURNAME, 3)) AS LAST_NAME_PREFIX,  -- blocking key
        ACT_NUM_ONLINE_PURCHASE_QUINT,
        CHANNEL_PREF_RT_CATALOG_QUINT,
        NUM_GENERATIONS_HH_ENH,
        AVA_HOME_EQUITY_IN_K,
        MT_VEHICLE_SERVICE_CENTER_USERS,
        INDIVIDUAL_EXACT_AGE,
        MT_MEAL_KIT_DELIVERY_CONSUMERS,
        MT_GROCERY_STORE_FREQUENTERS,
        ADV_HH_MARITAL_STATUS,
        MT_HOME_WARRANTY_PURCHASERS,
        MT_RETIRED_BUT_STILL_WORKING,
        ADV_NUM_ADULTS,
        MORTG_INTEREST_RATE_REFIN,
        MT_MEDICARE_ADVANTAGE_PLAN_PURCHASERS,
        CHANNEL_PREF_RT_ONLINE_QUINT,
        _2020_COMPLETE_CENSUS_GEO,
        MT_YOGAPILATES_ENTHUSIAST,
        MT_COMMUNITY_BANK_CUSTR,
        TGT_NET_WORTH_30,
        MT_CLICK_TO_CART_HOME_DELIVERY_CUSTOMERS,
        DWELLING_TYP_LEGACY,
        MT_MOBILE_PHONE_SERVICE_SWITCHERS,
        MT_SENIOR_CAREGIVERS,
        ACT_TOT_DOLLARS_QUINT,
        MT_SALTY_SNACKS_PURCHASERS,
        ACT_TOT_ONLINE_DOLLARS_QUINT,
        MT_LOYAL_FIN_INSTITUTION_CUSTR,
        MT_WHATS_ON_SALE_SHOPPERS,
        MT_HEAVY_COUPON_USERS,
        MT_ONLINE_HOTEL_UPGRADERS,
        MT_NEW_ROOF_CUSTOMERS,
        MT_MULTI_RETAILER_SHOPPERS,
        MT_CASINO_GAMER,
        MT_VEHICLE_DIYRS,
        ROOFTOP_LONGITUDE,
        MT_INSURANCE_FOR_LOAN_PRODUCTS_PURCHASERS,
        MT_FINANCIALHEALTH_NEWSLETTER_SUBSCRIBERS,
        NUM_TRADELINES,
        AGILITY_OCCUPANCY_SCORE,
        CURRENT_LOAN_TO_VALUE,
        MT_CABLE_BUNDLE_CABLE_INTERNET_HOME_PHONE,
        MT_AUTO_INSURANCE_SELF_SERVE_ONLINE_BUYERS,
        _2010_COMPLETE_CENSUS_GEO_,
        NUM_DOLLARS_ON_RETURNS,
        MT_MEAL_PLANNERS,
        ADV_DWELLING_TYP,
        MT_AUTO_INSURANCE_CALL_CENTER_SOLD,
        ROOFTOP_LATITUDE,
        MT_PRE_SHOP_PLANNERS,
        MT_MEDICARE_SUPPLEMENT_INSURANCE_PURCHASERS,
        MT_ATT_CELL_PHONE_CUSTOMER,
        MT_DO_IT_YOURSELFER,
        MT_ONLINE_PERSONAL_CARE_PRODUCT_BUYERS,
        MT_IDENTITY_THEFT_PROTECTION_PURCHASERS,
        MT_LOW_DOLLAR_DONORS,
        MT_FRESH_FOOD_DELIVERY_CONSUMERS,
        MT_REWARDS_CARD_CASH_BACK_USER,
        MT_HOUSEHOLD_CLEANING_PRODUCTS_BRAND_SWITCHERS,
        ADVANTAGE_TARGET_NARROW_BAND_INCOME_4_0,
        MERITSCORE,
        TARGET_VALUESCORE_20_RETAIL_CARD_MARKETERS,
        TARGET_VALUESCORE_20_AUTO_FINANCE_MARKETERS,
        PROPERTY_LOT_SIZE_IN_ACRES,
        TARGET_VALUESCORE_20_ALL_MARKETERS,
        TRIGGERVAL_INCOME,
        NET_WORTH_TIERS,
        TRIGGERVAL_VALUESCORE,
        TARGET_VALUESCORE_3_0,
        ADV_TGT_INCOME_30,
        TARGET_VALUESCORE_20_BANK_CARD_MARKETERS,
        ADV_TGT_NARROW_BAND_INCOME_30,
        HH_PURCHASE_CHANNEL_MO,
        HH_PURCHASE_CHANNEL_INT,
        MEMBER_CODE_PERSON,
        MO_ANY_ALL,
        OTHER_ONLINE_HOUSEHOLD_ALL,
        ADV_PREZ_CHILDREN_ENH,
        ADV_NUM_ADULTS_INDICATOR,
        ADV_HH_AGE_INDICATOR_ENH,
        REFIN_INDICATOR,
        HOME_EQUITY_LOAN_INDICATOR,
        ADV_IND_MARITAL_STATUS,
        EMPTY_NESTER_TRIGGER,
        TYP_CC_ANY_CC,
        ADV_IND_MARITAL_STATS_INDICATR,
        ADV_HH_MARITAL_STATUS_INDICATR,
        MO_BUYER,
        NICHES_40,
        MORTG_INTEREST_RATE_TYP_REFIN,
        MORTG_LOAN_TYP_REFIN,
        EXTERIOR_WALL_TYP,
        HOME_HEAT_SOURCE,
        INCOME_TRIGGER,
        NICHE_SWITCH_TRIGGER,
        NICHE_SWITCH_TRIGGER_CHG_TYP,
        NICHES_50
    FROM C2G_LAKE_DEV.VDS_FTP.EPSILON_TSP_IND_LATEST
    WHERE LEFT(ZIP,5) IN (SELECT DISTINCT LEFT(ZIPCODE,5) FROM CASS_INPUT_NEW)
    AND STATE IN (SELECT DISTINCT STATE FROM CASS_INPUT_NEW)
    AND SURNAME IS NOT NULL
    AND GIVEN_NAME IS NOT NULL
    AND CONTRACTED_ADDRESS IS NOT NULL
    AND TRIM(CONTRACTED_ADDRESS) != '';

    ------------------------------------------------------------
    -- 3) Create MATCH_RESULTS_EPSILON with blocking and fuzzy scoring
    ------------------------------------------------------------
    CREATE OR REPLACE TABLE MATCH_RESULTS_EPSILON AS
    WITH zip_filtered AS (
        SELECT
            c.RECORD_ID      AS CASS_RECORD_ID,
            c.FIRST_NAME     AS CASS_FIRST_NAME,
            c.LAST_NAME      AS CASS_LAST_NAME,
            c.ADDRESS_LINE_1 AS CASS_ADDRESS_LINE_1,
            c.ADDRESS_LINE_2 AS CASS_ADDRESS_LINE_2,
            c.CITY           AS CASS_CITY,
            c.STATE          AS CASS_STATE,
            c.ZIPCODE        AS CASS_ZIPCODE,
            c.ERROR_CODE     AS CASS_ERROR_CODE,
            e.PERSON_SEQ_NO,
            e.GIVEN_NAME     AS EPS_FIRST_NAME,
            e.SURNAME        AS EPS_LAST_NAME,
            e.CONTRACTED_ADDRESS AS EPS_ADDRESS,
            e.ZIP            AS EPS_ZIP,
            e.STATE          AS EPS_STATE
        FROM CASS_INPUT_NEW c
        JOIN EPS_SUBSET e
        ON LEFT(c.ZIPCODE,5) = LEFT(e.ZIP,5)
        AND c.STATE = e.STATE
        AND UPPER(LEFT(c.LAST_NAME,3)) = e.LAST_NAME_PREFIX
    ),
    scored AS (
        SELECT
            zf.*,
            ROUND(JAROWINKLER_SIMILARITY(UPPER(TRIM(zf.CASS_FIRST_NAME)), UPPER(TRIM(zf.EPS_FIRST_NAME))), 2) AS MATCH_FIRST,
            ROUND(JAROWINKLER_SIMILARITY(UPPER(TRIM(zf.CASS_LAST_NAME)),  UPPER(TRIM(zf.EPS_LAST_NAME))),  2) AS MATCH_LAST,
            ROUND(
                100 * GREATEST(
                    0,
                    1 - LEAST(
                        EDITDISTANCE(
                            REGEXP_REPLACE(UPPER(TRIM(zf.CASS_ADDRESS_LINE_1 || ' ' || NVL(zf.CASS_ADDRESS_LINE_2,''))), '[^A-Z0-9 ]', ''),
                            REGEXP_REPLACE(UPPER(TRIM(zf.EPS_ADDRESS)), '[^A-Z0-9 ]', '')
                        )
                        / NULLIF(GREATEST(
                            LENGTH(REGEXP_REPLACE(UPPER(TRIM(zf.CASS_ADDRESS_LINE_1 || ' ' || NVL(zf.CASS_ADDRESS_LINE_2,''))), '[^A-Z0-9 ]', '')),
                            LENGTH(REGEXP_REPLACE(UPPER(TRIM(zf.EPS_ADDRESS)), '[^A-Z0-9 ]', ''))
                        ), 0),
                        1
                    )
                ), 2
            ) AS MATCH_ADDR
        FROM zip_filtered zf
    ),
    final_scored AS (
        SELECT
            *,
            ROUND((0.65 * (MATCH_ADDR / 100.0) + 0.25 * (MATCH_LAST/100.0) + 0.10 * (MATCH_FIRST/100.0)) * 100, 2) AS MATCH_SCORE
        FROM scored
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY CASS_RECORD_ID ORDER BY MATCH_SCORE DESC) AS rn
        FROM final_scored
        WHERE MATCH_SCORE >= 70
    )
    SELECT
        CASS_RECORD_ID,
        CASS_FIRST_NAME,
        EPS_FIRST_NAME,
        CASS_LAST_NAME,
        EPS_LAST_NAME,
        CASS_ADDRESS_LINE_1,
        CASS_ADDRESS_LINE_2,
        EPS_ADDRESS,
        CASS_CITY,
        CASS_STATE,
        CASS_ZIPCODE,
        CASS_ERROR_CODE,
        PERSON_SEQ_NO AS EPS_PERSON_SEQ_NO,
        MATCH_FIRST,
        MATCH_LAST,
        MATCH_ADDR,
        MATCH_SCORE,
        CASE
            WHEN MATCH_SCORE >= 88 THEN 'High'
            WHEN MATCH_SCORE >= 78 THEN 'Medium'
            ELSE 'Low'
        END AS MATCH_LEVEL
    FROM ranked
    WHERE rn = 1;

    ------------------------------------------------------------
    -- 4) Enrich matched records with Epsilon attributes
    ------------------------------------------------------------
    CREATE OR REPLACE TABLE MATCH_RESULTS_EPSILON_ENRICHED AS
    SELECT
        m.*,
        e.*
    FROM MATCH_RESULTS_EPSILON m
    JOIN C2G_LAKE_DEV.VDS_FTP.EPSILON_TSP_IND_LATEST e
    ON e.PERSON_SEQ_NO = m.EPS_PERSON_SEQ_NO;

    SELECT * FROM MATCH_RESULTS_EPSILON_ENRICHED;
    """
    # 🔹 Split on semicolons and execute one by one
    statements = [stmt.strip() for stmt in sql_script.split(';') if stmt.strip()]

    last_result = None
    for i, stmt in enumerate(statements, 1):
        print(f"Running statement {i}/{len(statements)}: {stmt[:60]}...")
        cur.execute(stmt)
        if stmt.strip().lower().startswith("select"):
            last_result = cur.fetch_pandas_all()



    if last_result is not None:
        last_result.to_csv("match_eps_results.csv", index=False)
        print("✅ Results written to last_result dataframe")
        return last_result
    else:
        print("⚠️ No SELECT result found.")



# if __name__ == "__main__":
# #     generate_and_upload_epsilon_data()
#     generate_and_upload_acxiom_data()