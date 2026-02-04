import os
import modal

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("picobanana-dataset-gen")

# ---------------------------------------------------------------------------
# Volume for persistent output
# ---------------------------------------------------------------------------
output_volume = modal.Volume.from_name("picobanana-dataset", create_if_missing=True)
VOLUME_MOUNT_PATH = "/vol/output"

# ---------------------------------------------------------------------------
# Secret for API keys (reads from src/.env -- must include HF_TOKEN for gated models)
# ---------------------------------------------------------------------------
LOCAL_SRC = os.path.dirname(os.path.abspath(__file__))
api_secret = modal.Secret.from_dotenv(os.path.join(LOCAL_SRC, "..", ".env"))

# ---------------------------------------------------------------------------
# Image: install deps + bake model weights into the layer
# ---------------------------------------------------------------------------

def download_models():
    """Pre-download all model weights during image build."""
    # SAM3 BPE vocab file
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

    # SAM3 checkpoint
    from sam3.model_builder import build_sam3_image_model
    build_sam3_image_model()

    # DINOv3 models
    from transformers import AutoModel, AutoImageProcessor
    AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
    AutoModel.from_pretrained("facebook/dinov3-convnext-small-pretrain-lvd1689m")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install_from_requirements('requirementx.txt')
    .run_function(download_models, gpu="any", secrets=[api_secret])
)

# ---------------------------------------------------------------------------
# Add local source files to image (preserves relative imports)
# ---------------------------------------------------------------------------
image = (
    image
    .add_local_file(os.path.join(LOCAL_SRC, "dataset.py"), "/root/src/dataset.py")
    .add_local_file(os.path.join(LOCAL_SRC, "image_compositor.py"), "/root/src/image_compositor.py")
    .add_local_file(os.path.join(LOCAL_SRC, "blending.py"), "/root/src/blending.py")
    .add_local_file(os.path.join(LOCAL_SRC, "generate_dataset.py"), "/root/src/generate_dataset.py")
)


# ---------------------------------------------------------------------------
# GPU function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4",
    secrets=[api_secret],
    volumes={VOLUME_MOUNT_PATH: output_volume},
    timeout=86400,
)
def generate_dataset(n: int = 200_000, start_index: int = 0):
    import sys
    sys.path.insert(0, "/root/src")

    # Set SAVE_PATH before importing generate_dataset (asserts at import time)
    os.environ["SAVE_PATH"] = VOLUME_MOUNT_PATH

    import asyncio
    import torch
    import shutil
    from dataset import PicobananaDataset
    from generate_dataset import save_to

    print(f"Starting dataset generation: n={n}, start_index={start_index}")

    dataset = PicobananaDataset(n=n, start_index=start_index, return_img=True)
    asyncio.run(dataset.prepare_data())

    freq = {edittype: float("inf") for edittype in dataset.edit_types}

    try:
        with torch.no_grad():
            save_to(dataset, VOLUME_MOUNT_PATH, freq, commit_fn=output_volume.commit)
    except Exception as e:
        print(f"Error during generation: {e}")
        raise
    finally:
        data_sample_path = os.path.join(VOLUME_MOUNT_PATH, "data_sample")
        if os.path.exists(data_sample_path):
            shutil.make_archive(data_sample_path, "zip", data_sample_path)
            print(f"Archive created at {data_sample_path}.zip")
        output_volume.commit()

    print("Dataset generation complete.")


# ---------------------------------------------------------------------------
# CLI entrypoint: modal run generate_dataset_run_modal.py [--n 1000] [--start-index 0]
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(n: int = 200_000, start_index: int = 0):
    print(f"Launching on Modal (n={n}, start_index={start_index})...")
    generate_dataset.remote(n=n, start_index=start_index)
    print("Done. Data saved to Modal Volume 'picobanana-dataset'.")
