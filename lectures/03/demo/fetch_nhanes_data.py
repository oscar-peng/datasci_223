#!/usr/bin/env python3
"""
Fetch NHANES data for SQL demos.
"""

import os
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import sys

# Create data directory if it doesn't exist
DATA_DIR = Path("lectures/03/demo/data")
DATA_DIR.mkdir(exist_ok=True)

def download_file(url, output_path):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_nhanes():
    """Download NHANES data from CDC."""
    print("Downloading NHANES data...")
    
    # NHANES 2017-2018 data URLs (CSV format)
    urls = {
        "demographics": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.csv",
        "examination": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.csv",
        "laboratory": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BIOPRO_J.csv",
        "questionnaire": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DIQ_J.csv"
    }
    
    for name, url in urls.items():
        print(f"\nDownloading {name}...")
        output_file = DATA_DIR / f"{name}.csv"
        if download_file(url, output_file):
            print(f"Saved {name} data to {output_file}")

def main():
    """Main function to download data."""
    print("Starting NHANES data download...")
    
    try:
        download_nhanes()
        print("\nData download complete!")
    except Exception as e:
        print(f"\nError during data download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 