"""
Phase 0: One-time setup — download Open Images source images from AWS S3.

Downloads tar files from s3://open-images-dataset/tar/, selectively extracts
only the images referenced by Pico-Banana JSONL, renames them from hex ImageID
to Flickr filename, and persists to the 'openimages-source' Modal volume.

Usage:
    modal run src/scalable_data_collection/openimages_setup_modal.py [--n 200000] [--start-index 0]
"""
import os
import modal

app = modal.App("openimages-setup")

source_volume = modal.Volume.from_name("openimages-source", create_if_missing=True)
SOURCE_MOUNT = "/vol/source"

LOCAL_SRC = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("awscli")
    .pip_install("requests", "tqdm")
    .add_local_file(
        os.path.join(LOCAL_SRC, "image_mapping.py"),
        "/root/src/image_mapping.py",
    )
)

# Open Images train tars on S3 (public, no auth)
TARS = [
    "train_0",
    "train_1",
    "train_2",
    "train_3",
    "train_4",
    "train_5",
    "train_6",
    "train_7",
    "train_8",
]

CSV_URL = (
    "https://storage.googleapis.com/openimages/2018_04/"
    "train/train-images-boxable-with-rotation.csv"
)


@app.function(
    image=image,
    volumes={SOURCE_MOUNT: source_volume},
    timeout=86400,
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=10.0),
)
def setup_openimages(n: int = 200_000, start_index: int = 0):
    import sys
    sys.path.insert(0, "/root/src")

    import json
    import subprocess
    import tarfile
    import shutil
    from pathlib import Path
    from tqdm import tqdm
    from image_mapping import (
        load_jsonl_flickr_urls,
        load_csv_filtered,
        build_mappings,
        get_needed_imageids,
    )

    out_dir = Path(SOURCE_MOUNT) / "openimage_source_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_path = Path(SOURCE_MOUNT) / "setup_progress.json"

    # Load progress from previous runs (resume after preemption)
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)
        completed_tars = set(progress.get("completed_tars", []))
        total_extracted = progress.get("total_extracted", 0)
        print(f"Resuming: {len(completed_tars)} tars done, {total_extracted} images extracted so far")
    else:
        completed_tars = set()
        total_extracted = 0

    def _save_progress():
        with open(progress_path, "w") as f:
            json.dump({
                "completed_tars": sorted(completed_tars),
                "total_extracted": total_extracted,
            }, f)
        source_volume.commit()

    # ── Step 1: Load JSONL and get needed Flickr URLs ──
    print(f"Loading JSONL index (n={n}, start_index={start_index})...")
    flickr_urls = load_jsonl_flickr_urls(start_index=start_index, n=n)
    print(f"Need images for {len(flickr_urls)} unique Flickr URLs")

    # ── Step 2: Download CSV and build URL→ImageID mapping ──
    csv_path = "/tmp/train-images-boxable-with-rotation.csv"
    if not os.path.exists(csv_path):
        print("Downloading Open Images CSV...")
        import requests
        resp = requests.get(CSV_URL, stream=True)
        resp.raise_for_status()
        with open(csv_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print("CSV downloaded.")
    else:
        print("CSV already exists on disk.")

    print("Parsing CSV (filtering to needed URLs)...")
    url_to_imageid = load_csv_filtered(csv_path, flickr_urls)
    print(f"Found {len(url_to_imageid)} URL→ImageID mappings in CSV")

    flickr_to_imageid, imageid_to_flickr, missing_urls = build_mappings(
        flickr_urls, url_to_imageid
    )
    needed_ids = get_needed_imageids(flickr_to_imageid)
    print(f"Need {len(needed_ids)} unique ImageIDs from tars")
    print(f"Missing from CSV: {len(missing_urls)} URLs")

    # Check which images we already have on the volume
    already_have = set()
    for flickr_name in flickr_to_imageid:
        if (out_dir / flickr_name).exists():
            already_have.add(flickr_to_imageid[flickr_name])
    remaining_ids = needed_ids - already_have
    print(f"Already on volume: {len(already_have)}, still need: {len(remaining_ids)}")

    if not remaining_ids:
        print("All images already extracted! Nothing to do.")
        return

    # ── Step 3: Process each tar ──
    for tar_name in TARS:
        if tar_name in completed_tars:
            print(f"\n=== {tar_name}: already done, skipping ===")
            continue

        if not remaining_ids:
            print("All needed images found, stopping early.")
            break

        s3_url = f"s3://open-images-dataset/tar/{tar_name}.tar.gz"

        print(f"\n=== {tar_name}: streaming from S3 ===")
        extracted_this_tar = 0

        try:
            # Stream tar directly from S3 — no disk write needed
            # r|gz = streaming mode (sequential read, no seeking)
            proc = subprocess.Popen(
                ["aws", "s3", "cp", "--no-sign-request", s3_url, "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue

                    # Files in tar are like "train_0/abcdef123456.jpg"
                    basename = os.path.basename(member.name)
                    image_id = os.path.splitext(basename)[0]

                    if image_id not in remaining_ids:
                        continue

                    # Extract and rename to Flickr filename
                    flickr_name = imageid_to_flickr.get(image_id)
                    if not flickr_name:
                        continue

                    dest_path = out_dir / flickr_name
                    if dest_path.exists():
                        remaining_ids.discard(image_id)
                        continue

                    fileobj = tar.extractfile(member)
                    if fileobj is None:
                        continue

                    with open(dest_path, "wb") as out_f:
                        shutil.copyfileobj(fileobj, out_f)

                    remaining_ids.discard(image_id)
                    extracted_this_tar += 1
                    total_extracted += 1

                    if extracted_this_tar % 1000 == 0:
                        source_volume.commit()
                        print(f"  Committed {extracted_this_tar} images from {tar_name} "
                              f"({len(remaining_ids)} still needed)")

            proc.wait()
            if proc.returncode != 0:
                stderr = proc.stderr.read().decode()
                print(f"Warning: aws s3 cp exited with code {proc.returncode}: {stderr}")
        except Exception as e:
            print(f"Error processing {tar_name}: {e}")
            _save_progress()
            raise

        completed_tars.add(tar_name)
        _save_progress()
        print(f"{tar_name}: extracted {extracted_this_tar} images "
              f"(total: {total_extracted}, remaining: {len(remaining_ids)})")

    # ── Final summary ──
    found_count = sum(1 for f in flickr_to_imageid if (out_dir / f).exists())
    print(f"\nSetup complete.")
    print(f"  Found on volume: {found_count} / {len(flickr_to_imageid)} needed images")
    print(f"  Total extracted this run: {total_extracted}")
    print(f"  Missing from CSV: {len(missing_urls)}")
    _save_progress()


@app.local_entrypoint()
def main(n: int = 200_000, start_index: int = 0):
    print(f"Launching Open Images setup (n={n}, start_index={start_index})...")
    setup_openimages.remote(n=n, start_index=start_index)
    print("Done. Source images saved to Modal Volume 'openimages-source'.")
