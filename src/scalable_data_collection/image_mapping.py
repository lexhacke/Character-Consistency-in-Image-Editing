"""
URL-to-filename mapping utilities for Open Images ↔ Pico-Banana.

Maps Flickr URLs (from Pico-Banana JSONL) to Open Images hex ImageIDs
(from the tar files) via the train-images-boxable-with-rotation.csv.
"""
import csv
import json
import requests
from typing import Dict, Set, Tuple


def load_jsonl_flickr_urls(
    start_index: int = 0,
    n: int = 200_000,
    jsonl_url: str = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl",
) -> Set[str]:
    """Download Pico-Banana JSONL and extract unique Flickr URLs for the given slice."""
    r = requests.get(jsonl_url)
    all_lines = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    subset = all_lines[start_index : start_index + n]
    return {item['open_image_input_url'] for item in subset}


def load_csv_filtered(csv_path: str, needed_urls: Set[str]) -> Dict[str, str]:
    """
    Stream CSV, retain only rows whose OriginalURL is in needed_urls.
    Returns: {OriginalURL: ImageID} for matched entries only.
    """
    url_to_imageid = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['OriginalURL'] in needed_urls:
                url_to_imageid[row['OriginalURL']] = row['ImageID']
    return url_to_imageid


def build_mappings(
    flickr_urls: Set[str],
    url_to_imageid: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str], Set[str]]:
    """
    Build bidirectional mappings between Flickr filenames and ImageIDs.

    Returns:
        flickr_to_imageid: {"9423051591_cb1bf5c5e1_o.jpg": "abcdef123456"}
        imageid_to_flickr: {"abcdef123456": "9423051591_cb1bf5c5e1_o.jpg"}
        missing_urls:      URLs in JSONL not found in the CSV
    """
    flickr_to_imageid = {}
    missing_urls = set()

    for url in flickr_urls:
        flickr_filename = url.split('/')[-1]
        if url in url_to_imageid:
            flickr_to_imageid[flickr_filename] = url_to_imageid[url]
        else:
            missing_urls.add(url)

    imageid_to_flickr = {v: k for k, v in flickr_to_imageid.items()}
    return flickr_to_imageid, imageid_to_flickr, missing_urls


def get_needed_imageids(flickr_to_imageid: Dict[str, str]) -> Set[str]:
    """Return the set of all ImageIDs that need to be extracted from tars."""
    return set(flickr_to_imageid.values())
