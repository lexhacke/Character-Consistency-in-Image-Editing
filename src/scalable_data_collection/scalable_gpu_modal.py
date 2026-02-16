"""
Phase 3: GPU compositing for the scalable pipeline.

Loads precomputed Gemini results from batch_cache, runs SAM3 segmentation +
DINOv3 scoring + Laplacian blending on GPU. Always runs in batch mode.

Mounts both volumes:
  - openimages-source (read): source images for LocalPicobananaDataset
  - picobanana-dataset (read/write): batch_cache + output data_sample

Usage:
    modal run src/scalable_data_collection/scalable_gpu_modal.py
"""
import os
import modal

app = modal.App("scalable-dataset-gen")

source_volume = modal.Volume.from_name("openimages-source", create_if_missing=True)
output_volume = modal.Volume.from_name("picobanana-dataset", create_if_missing=True)
SOURCE_MOUNT = "/vol/source"
OUTPUT_MOUNT = "/vol/output"

LOCAL_SRC = os.path.dirname(os.path.abspath(__file__))
DATA_COLLECTION_SRC = os.path.join(os.path.dirname(LOCAL_SRC), "data_collection")
api_secret = modal.Secret.from_dotenv(os.path.join(LOCAL_SRC, "..", ".env"))


def download_models():
    """Pre-download all model weights during image build."""
    import sam3, pathlib, requests
    bpe_dir = pathlib.Path(sam3.__file__).parent.parent / "assets"
    bpe_dir.mkdir(parents=True, exist_ok=True)
    bpe_path = bpe_dir / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.exists():
        resp = requests.get(
            "https://github.com/openai/CLIP/raw/refs/heads/main/clip/bpe_simple_vocab_16e6.txt.gz",
            stream=True,
        )
        resp.raise_for_status()
        with open(bpe_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    from sam3.model_builder import build_sam3_image_model
    build_sam3_image_model()

    from transformers import AutoModel, AutoImageProcessor
    AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
    AutoModel.from_pretrained("facebook/dinov3-convnext-small-pretrain-lvd1689m")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install_from_requirements(
        os.path.join(DATA_COLLECTION_SRC, "requirements.txt")
    )
    .run_function(download_models, gpu="any", secrets=[api_secret])
)

# Add source files from data_collection (reuses existing compositing code)
image = (
    image
    .add_local_file(os.path.join(DATA_COLLECTION_SRC, "dataset.py"), "/root/src/dataset.py")
    .add_local_file(os.path.join(DATA_COLLECTION_SRC, "image_compositor.py"), "/root/src/image_compositor.py")
    .add_local_file(os.path.join(DATA_COLLECTION_SRC, "blending.py"), "/root/src/blending.py")
    .add_local_file(os.path.join(DATA_COLLECTION_SRC, "system_prompt.py"), "/root/src/system_prompt.py")
    .add_local_file(os.path.join(DATA_COLLECTION_SRC, "generate_dataset.py"), "/root/src/generate_dataset.py")
)


@app.function(
    image=image,
    gpu="L4",
    secrets=[api_secret],
    volumes={
        SOURCE_MOUNT: source_volume,
        OUTPUT_MOUNT: output_volume,
    },
    timeout=86400,
)
def generate_dataset():
    import sys
    sys.path.insert(0, "/root/src")

    os.environ["SAVE_PATH"] = OUTPUT_MOUNT

    import json
    import torch
    import PIL.Image
    from generate_dataset import save_to

    cache_dir = os.path.join(OUTPUT_MOUNT, "batch_cache")
    manifest_path = os.path.join(cache_dir, "manifest.json")
    results_path = os.path.join(cache_dir, "results.json")

    if not os.path.exists(manifest_path) or not os.path.exists(results_path):
        raise FileNotFoundError(
            "batch_cache/manifest.json or results.json not found. "
            "Run scalable_batch_prep_modal.py first (Phase 1+2)."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(results_path) as f:
        precomputed = json.load(f)

    available_keys = set(precomputed.keys())
    filtered_manifest = [entry for entry in manifest if entry['key'] in available_keys]
    missing = len(manifest) - len(filtered_manifest)
    if missing:
        print(f"Skipping {missing} manifest entries without precomputed results")
    manifest = filtered_manifest

    print(f"Batch mode: {len(manifest)} items, {len(precomputed)} precomputed results")

    edit_types = {e['edit_type'] for e in manifest}
    freq = {et: float("inf") for et in edit_types}

    dataset = _BatchDataset(manifest)

    try:
        with torch.no_grad():
            save_to(dataset, OUTPUT_MOUNT, freq,
                    commit_fn=output_volume.commit, precomputed=precomputed)
    except Exception as e:
        print(f"Error during batch generation: {e}")
        raise
    finally:
        output_volume.commit()

    print("Dataset generation complete.")


class _BatchDataset:
    """Lazy iterator over manifest entries. Loads images from volume on demand."""

    def __init__(self, manifest):
        self.manifest = manifest

    def __len__(self):
        return len(self.manifest)

    def __iter__(self):
        import PIL.Image
        for entry in self.manifest:
            try:
                original = PIL.Image.open(entry['original_path']).convert('RGB')
                edited = PIL.Image.open(entry['edited_path']).convert('RGB')
                yield {
                    'key': entry['key'],
                    'prompt': entry['prompt'],
                    'original': original,
                    'edited': edited,
                    'edit_type': entry['edit_type'],
                }
            except Exception as e:
                print(f"Failed to load images for {entry['key']}: {e}")
                yield -1


@app.local_entrypoint()
def main():
    print("Launching Phase 3 (scalable batch mode) on Modal...")
    generate_dataset.remote()
    print("Done. Data saved to Modal Volume 'picobanana-dataset'.")
