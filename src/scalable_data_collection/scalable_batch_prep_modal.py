"""
Phase 1+2: Batch preparation for the scalable pipeline.

Uses local source images (from openimages-source volume) + Apple CDN for edits.
No Flickr downloads — eliminates the rate-limiting bottleneck.

1. Loads images via LocalPicobananaDataset (source from volume, edits from Apple CDN)
2. Caches image pairs to picobanana-dataset volume
3. Uploads to Google File API in chunks
4. Submits Gemini Batch API jobs, polls, downloads results
5. Saves manifest.json + results.json for Phase 3

Usage:
    modal run src/scalable_data_collection/scalable_batch_prep_modal.py [--n 200000] [--start-index 0] [--chunk-size 5000]
"""
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import modal

app = modal.App("scalable-batch-prep")

# Two volumes: source images (read) + output (read/write)
source_volume = modal.Volume.from_name("openimages-source", create_if_missing=True)
output_volume = modal.Volume.from_name("picobanana-dataset", create_if_missing=True)
SOURCE_MOUNT = "/vol/source"
OUTPUT_MOUNT = "/vol/output"

LOCAL_SRC = os.path.dirname(os.path.abspath(__file__))
DATA_COLLECTION_SRC = os.path.join(os.path.dirname(LOCAL_SRC), "data_collection")
api_secret = modal.Secret.from_dotenv(os.path.join(LOCAL_SRC, "..", ".env"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(
        os.path.join(DATA_COLLECTION_SRC, "batch_requirements.txt")
    )
    .add_local_file(
        os.path.join(LOCAL_SRC, "local_dataset.py"),
        "/root/src/local_dataset.py",
    )
    .add_local_file(
        os.path.join(DATA_COLLECTION_SRC, "batch_manager.py"),
        "/root/src/batch_manager.py",
    )
    .add_local_file(
        os.path.join(DATA_COLLECTION_SRC, "system_prompt.py"),
        "/root/src/system_prompt.py",
    )
)


@app.function(
    image=image,
    secrets=[api_secret],
    volumes={
        SOURCE_MOUNT: source_volume,
        OUTPUT_MOUNT: output_volume,
    },
    timeout=86400,
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=10.0),
)
def batch_prep(n: int = 200_000, start_index: int = 0, chunk_size: int = 5000):
    import sys
    sys.path.insert(0, "/root/src")

    import json
    import asyncio
    import PIL.Image
    from local_dataset import LocalPicobananaDataset
    from batch_manager import BatchManager, build_system_prompt

    source_dir = os.path.join(SOURCE_MOUNT, "openimage_source_images")

    # ── Phase 1: Download dataset and cache images to volume ──
    print(f"Loading dataset: n={n}, start_index={start_index}")
    print(f"Source images from: {source_dir}")
    dataset = LocalPicobananaDataset(
        source_dir=source_dir, n=n, start_index=start_index, return_img=True
    )
    asyncio.run(dataset.prepare_data())

    if len(dataset) == 0:
        print("No valid items found. Exiting.")
        return

    # Sort for deterministic ordering between Phase 1 and Phase 3
    dataset.data.sort(key=lambda x: x['output_image'])

    # Accept all edit types
    freq = {edittype: float('inf') for edittype in dataset.edit_types}

    cache_dir = os.path.join(OUTPUT_MOUNT, "batch_cache")
    img_dir = os.path.join(cache_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    download_workers = max(1, int(os.environ.get("BATCH_PREP_DOWNLOAD_WORKERS", "2")))
    print(f"Downloading and caching {len(dataset)} items (workers={download_workers})...")
    manifest = []
    skipped = 0
    prefetch = max(download_workers * 2, 8)

    def _dataset_fetch(index):
        orig_path = os.path.join(img_dir, f"{index}_original.jpg")
        edit_path = os.path.join(img_dir, f"{index}_edited.png")
        if os.path.exists(orig_path) and os.path.exists(edit_path):
            item = dataset.data[index]
            return {
                'prompt': item['text'],
                'edit_type': item['edit_type'],
                'original': None,
                'edited': None,
                '_cached': True,
            }
        try:
            return dataset[index]
        except Exception as exc:
            print(f"[WARN] Dataset fetch failed for idx={index}: {exc}")
            return -1

    def _process_item(idx, item):
        nonlocal skipped
        if item == -1:
            skipped += 1
            return

        if item['edit_type'] not in freq:
            skipped += 1
            return

        orig_path = os.path.join(img_dir, f"{idx}_original.jpg")
        edit_path = os.path.join(img_dir, f"{idx}_edited.png")

        if not item.get('_cached'):
            original = item['original']
            edited = item['edited']
            edited = edited.resize(original.size, PIL.Image.BILINEAR)
            if not os.path.exists(orig_path):
                original.save(orig_path, "JPEG")
            if not os.path.exists(edit_path):
                edited.save(edit_path, "PNG")

        manifest.append({
            'key': f"req-{len(manifest)}",
            'index': idx,
            'prompt': item['prompt'],
            'edit_type': item['edit_type'],
            'original_path': orig_path,
            'edited_path': edit_path,
        })

        if len(manifest) % 100 == 0:
            output_volume.commit()
        if len(manifest) % 500 == 0:
            print(f"  Cached {len(manifest)} items so far ({skipped} skipped)...")

    pending = {}
    buffered = {}
    next_submit = 0
    next_process = 0
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        while next_process < len(dataset):
            while next_submit < len(dataset) and len(pending) < prefetch:
                fut = executor.submit(_dataset_fetch, next_submit)
                pending[fut] = next_submit
                next_submit += 1

            if not pending:
                break

            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                idx = pending.pop(fut)
                buffered[idx] = fut.result()

            while next_process in buffered:
                data_item = buffered.pop(next_process)
                _process_item(next_process, data_item)
                next_process += 1

    print(f"Cached {len(manifest)} items total, skipped {skipped}")

    # Save manifest
    manifest_path = os.path.join(cache_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    output_volume.commit()

    # ── Phase 2: Upload to File API and submit batch in chunks ──
    bm = BatchManager()
    poll_interval = int(os.environ.get("BATCH_POLL_INTERVAL_SECONDS", "60"))
    max_inflight_chunks = max(1, int(os.environ.get("BATCH_MAX_INFLIGHT_CHUNKS", "2")))
    pending_jobs = {}

    results_path = os.path.join(cache_dir, "results.json")
    progress_path = os.path.join(cache_dir, "progress.json")
    failures_path = os.path.join(cache_dir, "failed_chunks.json")
    pending_jobs_path = os.path.join(cache_dir, "pending_jobs.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Resuming: loaded {len(all_results)} existing results")
    else:
        all_results = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            completed_chunks = set(json.load(f))
        print(f"Resuming: chunks already done: {sorted(completed_chunks)}")
    else:
        completed_chunks = set()
    if os.path.exists(failures_path):
        with open(failures_path) as f:
            failed_chunks = set(json.load(f))
        if failed_chunks:
            print(f"Resuming: previously failed chunks {sorted(failed_chunks)}")
    else:
        failed_chunks = set()

    # Restore pending jobs from disk (survives preemption)
    if os.path.exists(pending_jobs_path):
        with open(pending_jobs_path) as f:
            pending_jobs = json.load(f)
        # Convert string keys back to int
        pending_jobs = {int(k): v for k, v in pending_jobs.items()}
        if pending_jobs:
            print(f"Resuming: {len(pending_jobs)} pending job(s) from previous run: "
                  f"chunks {sorted(pending_jobs.keys())}")
    else:
        pending_jobs = {}

    def _persist_state():
        with open(progress_path, 'w') as f:
            json.dump(sorted(completed_chunks), f)
        with open(results_path, 'w') as f:
            json.dump(all_results, f)
        with open(failures_path, 'w') as f:
            json.dump(sorted(failed_chunks), f)
        with open(pending_jobs_path, 'w') as f:
            json.dump(pending_jobs, f)
        output_volume.commit()

    terminal_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED',
                       'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}

    def _finalize_chunk(chunk_num, job_obj, uploaded_files):
        nonlocal all_results
        state = job_obj.state.name
        print(f"Chunk {chunk_num} finished with state: {state}")
        if state == 'JOB_STATE_SUCCEEDED':
            chunk_results = bm.download_results(job_obj)
            all_results.update(chunk_results)
            completed_chunks.add(chunk_num)
            failed_chunks.discard(chunk_num)
            print(f"  Results merged: {len(chunk_results)} rows (total {len(all_results)})")
        else:
            failed_chunks.add(chunk_num)
            print(f"  Chunk {chunk_num} failed ({state}). It will be retried on the next run.")

        if uploaded_files:
            print(f"  Cleaning up {len(uploaded_files)} uploaded files...")
            bm.delete_files(uploaded_files)
        pending_jobs.pop(chunk_num, None)
        _persist_state()

    def _process_pending(block=False):
        if not pending_jobs:
            return

        while True:
            progressed = False
            for chunk_id, info in list(pending_jobs.items()):
                job_obj = bm.get_job(info['job_name'])
                if job_obj.state.name in terminal_states:
                    _finalize_chunk(chunk_id, job_obj, info['uploaded_files'])
                    progressed = True
            if not block or not pending_jobs:
                break
            if not progressed:
                print(f"Waiting {poll_interval}s for {len(pending_jobs)} pending chunk(s)...")
                time.sleep(poll_interval)

    total_chunks = (len(manifest) + chunk_size - 1) // chunk_size
    for chunk_start in range(0, len(manifest), chunk_size):
        chunk = manifest[chunk_start:chunk_start + chunk_size]
        chunk_num = chunk_start // chunk_size

        if chunk_num in completed_chunks:
            print(f"\n=== Chunk {chunk_num}/{total_chunks - 1}: already done, skipping ===")
            continue

        if chunk_num in pending_jobs:
            print(f"\n=== Chunk {chunk_num}/{total_chunks - 1}: job already pending ({pending_jobs[chunk_num]['job_name']}), skipping submit ===")
            continue

        while len(pending_jobs) >= max_inflight_chunks:
            print(f"Reached max inflight chunks ({max_inflight_chunks}). Waiting for a slot...")
            _process_pending(block=True)

        _process_pending(block=False)

        print(f"\n=== Chunk {chunk_num}/{total_chunks - 1}: items {chunk_start}-{chunk_start + len(chunk) - 1} ===")

        entries = []
        uploaded_file_names = []
        total_uploads = len(chunk) * 2

        for i, entry in enumerate(chunk):
            with open(entry['original_path'], 'rb') as f:
                orig_bytes = f.read()
            with open(entry['edited_path'], 'rb') as f:
                edit_bytes = f.read()

            orig_file = bm.upload_image(orig_bytes, "image/jpeg", f"{entry['key']}-orig")
            edit_file = bm.upload_image(edit_bytes, "image/png", f"{entry['key']}-edit")
            uploaded_file_names.extend([orig_file.name, edit_file.name])

            sys_prompt = build_system_prompt(entry['prompt'])
            batch_entry = bm.build_entry(entry['key'], orig_file, edit_file, sys_prompt)
            entries.append(batch_entry)

            if (i + 1) % 100 == 0:
                print(f"  Uploading: {len(uploaded_file_names)}/{total_uploads} files...")

        print(f"Uploaded {len(uploaded_file_names)} files to File API")

        jsonl_path = os.path.join(cache_dir, f"requests_chunk_{chunk_num}.jsonl")
        bm.write_jsonl(entries, jsonl_path)

        job = bm.submit(jsonl_path, f"scalable-chunk-{chunk_num}")
        pending_jobs[chunk_num] = {
            'job_name': job.name,
            'uploaded_files': uploaded_file_names,
        }
        _persist_state()
        print(f"  Job submitted and persisted: {job.name}")

    _process_pending(block=True)

    print(f"\nBatch prep complete. {len(all_results)} / {len(manifest)} results saved.")
    print(f"  Manifest: {manifest_path}")
    print(f"  Results:  {results_path}")


@app.local_entrypoint()
def main(n: int = 200_000, start_index: int = 0, chunk_size: int = 5000):
    print(f"Launching scalable batch prep (n={n}, start_index={start_index}, chunk_size={chunk_size})...")
    batch_prep.remote(n=n, start_index=start_index, chunk_size=chunk_size)
    print("Done. Results saved to Modal Volume 'picobanana-dataset' under batch_cache/.")
