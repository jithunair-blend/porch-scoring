from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

LOCAL_DIR = os.path.join(os.getcwd(), "data_cache")
os.makedirs(LOCAL_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Central mapping of assets
S3_ASSETS = {
    # Models
    "acxiom_model":  {"bucket": "porchmodels", "key": "final_xgb_model_top_hpt_acxiom.json"},
    "epsilon_model": {"bucket": "porchmodels", "key": "final_xgb_model_top_hpt_epsilon.json"},

    # Raw CSVs
    "original_csv":  {"bucket": "porchsalesdatafiles", "key": "AHP-2025-Converters.csv"},
    "acxiom_data":   {"bucket": "porchsalesdatafiles", "key": "AHP-2025-Converters-ACX-appended.xlsx"},
    "epsilon_data":  {"bucket": "porchsalesdatafiles", "key": "AHP-2025-Converters-EPS-appended.xlsx"},
}

# Where downloaded files will be cached locally
S3_OUTPUTS = {
    "acxiom_output": {
        "bucket": "porchoutput",
        "key": f"acxiom/scored_output_{timestamp}.csv"
    },
    "epsilon_output": {
        "bucket": "porchoutput",
        "key": f"epsilon/scored_output_{timestamp}.csv"
    }
}