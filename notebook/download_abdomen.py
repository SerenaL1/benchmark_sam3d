import os

# 1. Setup Authentication (KAGGLE_API_TOKEN)
# Replace these with your actual username and key from your kaggle.json
os.environ["KAGGLE_USERNAME"] = "oibook13"
os.environ["KAGGLE_KEY"] = "KGAT_2b8fa2a18683ee43748be1302eabe289"

# 2. Indicate Download Directory
# kagglehub manages downloads via a cache system. 
# You must set this environment variable to change the root folder.
os.environ["KAGGLEHUB_CACHE"] = "/PHShome/yl535/project/python/datasets/abdomen"

import kagglehub

# Download latest version
path = kagglehub.dataset_download("lssz1275/abdomen")

print("Path to dataset files:", path)