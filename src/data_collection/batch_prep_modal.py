"""
Phase 1+2: Batch preparation and submission for Gemini API.

Runs on a CPU-only Modal instance. No GPU required.

1. Downloads images from PicobananaDataset, saves to volume
2. Uploads images to Google File API in chunks
3. Builds and submits Gemini Batch API jobs
4. Polls for completion, downloads results
5. Saves manifest.json + results.json to volume for Phase 3

Usage:
    modal run batch_prep_modal.py [--n 200000] [--start-index 0] [--chunk-size 5000]
"""
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import modal

app = modal.App("picobanana-batch-prep")

output_volume = modal.Volume.from_name("picobanana-dataset", create_if_missing=True)
VOLUME_MOUNT_PATH = "/vol/output"

LOCAL_SRC = os.path.dirname(os.path.abspath(__file__))
api_secret = modal.Secret.from_dotenv(os.path.join(LOCAL_SRC, "..", ".env"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(os.path.join(LOCAL_SRC, "batch_requirements.txt"))
    .add_local_file(os.path.join(LOCAL_SRC, "dataset.py"), "/root/src/dataset.py")
    .add_local_file(os.path.join(LOCAL_SRC, "batch_manager.py"), "/root/src/batch_manager.py")
    .add_local_file(os.path.join(LOCAL_SRC, "system_prompt.py"), "/root/src/system_prompt.py")
)


@app.function(
    image=image,
    secrets=[api_secret],
    volumes={VOLUME_MOUNT_PATH: output_volume},
    timeout=86400,
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=10.0),
)
def batch_prep(n: int = 200_000, start_index: int = 0, chunk_size: int = 5000):
    import sys
    sys.path.insert(0, "/root/src")

    import json
    import asyncio
    import PIL.Image
    from dataset import PicobananaDataset
    from batch_manager import BatchManager, build_system_prompt

    # ── Phase 1: Download dataset and cache images to volume ──
    print(f"Loading dataset: n={n}, start_index={start_index}")
    dataset = PicobananaDataset(n=n, start_index=start_index, return_img=True)
    asyncio.run(dataset.prepare_data())

    if len(dataset) == 0:
        print("No valid items found. Exiting.")
        return

    # Sort for deterministic ordering between Phase 1 and Phase 3
    dataset.data.sort(key=lambda x: x['output_image'])

    # Accept all edit types (same as generate_dataset __main__)
    freq = {edittype: float('inf') for edittype in dataset.edit_types}

    cache_dir = os.path.join(VOLUME_MOUNT_PATH, "batch_cache")
    img_dir = os.path.join(cache_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    download_workers = max(1, int(os.environ.get("BATCH_PREP_DOWNLOAD_WORKERS", "2")))
    print(f"Downloading and caching {len(dataset)} items (workers={download_workers})...")
    manifest = []
    skipped = 0
    prefetch = max(download_workers * 2, 8)

    def _dataset_fetch(index):
        # Skip download entirely if both images are already cached
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

    # Load existing progress if resuming after a crash
    results_path = os.path.join(cache_dir, "results.json")
    progress_path = os.path.join(cache_dir, "progress.json")
    failures_path = os.path.join(cache_dir, "failed_chunks.json")
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

    def _persist_state():
        with open(progress_path, 'w') as f:
            json.dump(sorted(completed_chunks), f)
        with open(results_path, 'w') as f:
            json.dump(all_results, f)
        with open(failures_path, 'w') as f:
            json.dump(sorted(failed_chunks), f)
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
                    pending_jobs.pop(chunk_id, None)
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

        while len(pending_jobs) >= max_inflight_chunks:
            print(f"Reached max inflight chunks ({max_inflight_chunks}). Waiting for a slot...")
            _process_pending(block=True)

        _process_pending(block=False)

        print(f"\n=== Chunk {chunk_num}/{total_chunks - 1}: items {chunk_start}-{chunk_start + len(chunk) - 1} ===")

        entries = []
        uploaded_file_names = []

        for entry in chunk:
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

        print(f"Uploaded {len(uploaded_file_names)} files to File API")

        jsonl_path = os.path.join(cache_dir, f"requests_chunk_{chunk_num}.jsonl")
        bm.write_jsonl(entries, jsonl_path)

        job = bm.submit(jsonl_path, f"picobanana-chunk-{chunk_num}")
        pending_jobs[chunk_num] = {
            'job_name': job.name,
            'uploaded_files': uploaded_file_names,
        }

    _process_pending(block=True)

    print(f"\nBatch prep complete. {len(all_results)} / {len(manifest)} results saved.")
    print(f"  Manifest: {manifest_path}")
    print(f"  Results:  {os.path.join(cache_dir, 'results.json')}")


@app.local_entrypoint()
def main(n: int = 200_000, start_index: int = 0, chunk_size: int = 5000):
    print(f"Launching batch prep on Modal (n={n}, start_index={start_index}, chunk_size={chunk_size})...")
    batch_prep.remote(n=n, start_index=start_index, chunk_size=chunk_size)
    print("Done. Results saved to Modal Volume 'picobanana-dataset' under batch_cache/.")
