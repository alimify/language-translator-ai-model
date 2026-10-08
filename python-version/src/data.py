"""Data loading utilities for the ManyThings Ben-Eng dataset."""
import os
import zipfile
import requests
from typing import List, Tuple


def download_with_requests(url: str, out_path: str) -> None:
    """Download file with proper headers to avoid 403 errors.
    
    Args:
        url: URL to download from
        out_path: Local path to save file
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "*/*"
    }
    with requests.get(url, headers=headers, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def get_raw_data(limit: int = 50000, max_len: int = 64) -> List[Tuple[str, str]]:
    """Download and parse the English-Bengali dataset.
    
    Args:
        limit: Maximum number of sentence pairs to load
        max_len: Maximum sentence length (in words) to include
        
    Returns:
        List of (english, bengali) sentence pairs
    """
    # Download dataset if not present
    if not os.path.exists("ben.txt"):
        print("Downloading dataset...")
        download_with_requests("https://www.manythings.org/anki/ben-eng.zip", "b.zip")
        with zipfile.ZipFile("b.zip") as zf:
            zf.extractall()
    
    # Parse dataset
    pairs = []
    if not os.path.exists("ben.txt"):
        print("Error: ben.txt not found.")
        return []
    
    with open("ben.txt", encoding="utf8") as f:
        for i, line in enumerate(f):
            parts = line.split("\t")
            if len(parts) >= 2:
                en, bn = parts[0].strip(), parts[1].strip()
                # Filter by sentence length
                if len(en.split()) < max_len and len(bn.split()) < max_len:
                    pairs.append((en, bn))
            if i >= limit:
                break
    
    return pairs
