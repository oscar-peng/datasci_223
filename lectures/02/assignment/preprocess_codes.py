#!/usr/bin/env python3
"""
Preprocess HCPCS code files into clean CSV format for assignment testing.
Handles inconsistent delimiters (comma and whitespace) in source data.
"""

import pandas as pd
from pathlib import Path

def preprocess_hcpcs_codes(input_file: Path, output_file: Path) -> None:
    """
    Read HCPCS codes file and write clean CSV with code and short_desc only.
    
    Args:
        input_file: Path to input file (comma or whitespace delimited)
        output_file: Path to output CSV file
    """
    # Read with flexible whitespace delimiter, then split on comma if present
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    codes = []
    descriptions = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try comma delimiter first
        if ',' in line:
            parts = line.split(',', 1)
            code = parts[0].strip()
            desc = parts[1].strip().strip('"') if len(parts) > 1 else ''
        else:
            # Fallback to whitespace
            parts = line.split(maxsplit=1)
            code = parts[0].strip()
            desc = parts[1].strip() if len(parts) > 1 else ''
        
        codes.append(code)
        descriptions.append(desc)
    
    # Create DataFrame
    df = pd.DataFrame({
        'code': codes,
        'short_desc': descriptions
    })
    
    # Write clean CSV
    df.to_csv(output_file, index=False)
    print(f"Processed {len(df)} codes from {input_file.name} -> {output_file.name}")

def main():
    # Process all HCPCS files
    ref_dir = Path('refs')
    output_dir = Path('lectures/02/assignment')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hcpcs_files = [
        'HCPC2024_CONTR_ANESTHESIA.txt',
        'HCPC2024_DME.txt', 
        'HCPC2024_ENTERAL_PARENTAL.txt'
    ]
    
    for filename in hcpcs_files:
        input_path = ref_dir / filename
        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping")
            continue
            
        output_name = filename.replace('.txt', '.csv')
        output_path = output_dir / output_name
        
        preprocess_hcpcs_codes(input_path, output_path)

if __name__ == '__main__':
    main()