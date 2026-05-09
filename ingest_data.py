import requests
import json
import os

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
RAW_FILE = "shl_product_catalog.json"
PROCESSED_FILE = "processed_catalog.json"

def fetch_and_process():
    print(f"Downloading from {CATALOG_URL}...")
    response = requests.get(CATALOG_URL)
    response.raise_for_status()
    data = json.loads(response.text, strict=False)
    
    with open(RAW_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"Downloaded {len(data)} total products.")
    
    # We don't know the exact schema yet, let's print a sample to see it
    if len(data) > 0:
        print("Sample product:", json.dumps(data[0], indent=2))

if __name__ == "__main__":
    fetch_and_process()
