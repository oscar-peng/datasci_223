#!/usr/bin/env python3
"""
Download script for the Wearable Device Dataset from PhysioNet.

This script downloads the Wearable Device Dataset from Induced Stress and Exercise Sessions
from PhysioNet and extracts it to the data directory.

Dataset: https://physionet.org/content/wearable-stress-affect/1.0.0/
"""

import os
import sys
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path

# PhysioNet dataset information
DATASET_URL = "https://physionet.org/static/published-projects/wearable-stress-affect/wearable-device-dataset-from-induced-stress-and-exercise-sessions-1.0.0.zip"
DATASET_SHA256 = "a1d5b1f5d3c97b634a5c69d2efd5c2f9c0e96ac53e9d4e25b31d89d6d5e45e1e"

def download_file(url, destination):
    """
    Download a file from a URL to a destination with a progress bar.
    
    Args:
        url (str): URL to download from
        destination (str): Path to save the file to
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get file size from headers (if available)
        total_size = int(response.headers.get('content-length', 0))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress bar
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def verify_sha256(file_path, expected_sha256):
    """
    Verify the SHA-256 hash of a file.
    
    Args:
        file_path (str): Path to the file
        expected_sha256 (str): Expected SHA-256 hash
    
    Returns:
        bool: True if hash matches, False otherwise
    """
    import hashlib
    
    sha256_hash = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        file_hash = sha256_hash.hexdigest()
        if file_hash == expected_sha256:
            return True
        else:
            print(f"Hash mismatch: Expected {expected_sha256}, got {file_hash}")
            return False
    
    except Exception as e:
        print(f"Error verifying hash: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """
    Extract a zip file to a directory.
    
    Args:
        zip_path (str): Path to the zip file
        extract_to (str): Directory to extract to
    
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(extract_to, exist_ok=True)
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total number of files for progress bar
            total_files = len(zip_ref.infolist())
            
            # Extract with progress bar
            for i, file in enumerate(zip_ref.infolist()):
                zip_ref.extract(file, extract_to)
                progress = (i + 1) / total_files * 100
                sys.stdout.write(f"\rExtracting: {progress:.1f}% complete")
                sys.stdout.flush()
            
            print()  # New line after progress bar
        
        return True
    
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False

def main():
    """Main function to download and extract the dataset."""
    # Set up paths
    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir / "data"
    zip_path = data_dir / "wearable-stress-affect.zip"
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading dataset to {zip_path}...")
    if download_file(DATASET_URL, zip_path):
        print("Download complete.")
        
        print("Verifying download...")
        if verify_sha256(zip_path, DATASET_SHA256):
            print("Verification successful.")
            
            print(f"Extracting to {data_dir}...")
            if extract_zip(zip_path, data_dir):
                print("Extraction complete.")
                print(f"Dataset is now available in {data_dir}")
                return True
    
    return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("Failed to download and extract the dataset.")
        sys.exit(1)