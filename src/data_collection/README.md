# Data Collection Pipeline

Generates ground-truth segmentation masks and composites from the [Pico-Banana-300k](https://ml-site.cdn-apple.com/datasets/pico-banana-300k/) dataset. Each sample pairs an original image with an edited version, and the pipeline determines how to stitch them together so unedited regions are preserved pixel-perfectly.

## How It Works

### 1. Download & Validate (`dataset.py`)
`PicobananaDataset` fetches the JSONL index from Apple's CDN and async-validates that both the original (Open Images) and edited (Apple CDN) URLs are still live. Supports local source images to reduce network load.

### 2. Route with Gemini (`image_compositor.py`, `system_prompt.py`)
Each image pair is sent to **Gemini 2.5 Flash** along with the edit prompt. The VLM returns a compositing recipe as JSON:

```json
{
  "base": "original" | "edited",
  "subtract": ["objects to remove from the base"],
  "union": ["objects to paste from the other image"]
}
```

The routing handles three edit types:
- **Standard edits** (e.g. "add a hat") — base = original, union = [new object]
- **Background changes** (e.g. "change background to volcano") — base = edited, union = [subject from original]
- **Global transforms** (e.g. "make it a pencil sketch") — returns the edited image as-is (empty subtract/union)

### 3. Segment with SAM3 (`image_compositor.py`)
SAM3 segments each object listed in the recipe using text prompts. If SAM3 fails, **Moondream** provides a bounding box as fallback input. Multiple masks per image are combined via logical OR.

### 4. Blend & Score (`blending.py`, `generate_dataset.py`)
Masks are dilated, then the base and other images are composited using **Laplacian pyramid blending** for seamless edges. The composite is scored against the target image using **DINOv3** cosine similarity.

## Output Structure

Each processed sample is saved under `$SAVE_PATH/data_sample/{success|fail}/{i}/`:

```
base.jpeg              # Canvas image
other.jpeg             # The other image (source of pasted regions)
mask.png               # Combined segmentation mask (dilated)
subtraction_mask.png   # Objects removed from base
union_mask.png         # Objects pasted from other
composite.jpeg         # Final blended result
meta.json              # Prompt, base choice, seg successes/failures, similarity score
```

Samples are bucketed into `success/` (all segmentations succeeded) or `fail/` (at least one SAM3 query failed).

## Setup

### Environment Variables (`.env`)
```
GOOGLE=<Gemini API key>
MOONDREAM=<Moondream API key>
SAVE_PATH=/path/to/output
```

### Install
```bash
pip install -r requirements.txt
```

## Usage

```bash
python generate_dataset.py
```

This downloads up to 100k items from Pico-Banana, processes each through the pipeline, and saves results to `$SAVE_PATH`. Processing resumes from where it left off if interrupted.

## Files

| File | Purpose |
|------|---------|
| `dataset.py` | `PicobananaDataset` — async download and validation of image pairs |
| `image_compositor.py` | `ImageCompositor` — Gemini routing, SAM3 segmentation, Moondream fallback, DINO scoring |
| `generate_dataset.py` | Main pipeline — orchestrates download, routing, segmentation, blending, and saving |
| `system_prompt.py` | Gemini system prompt defining the compositing logic rules |
| `blending.py` | Mask dilation and Laplacian pyramid blending |
| `visualize.py` | Visualization utilities |
| `batch_manager.py` | Manages batch processing of Gemini API calls |
| `batch_prep_modal.py` | Modal deployment for batch preparation |
| `generate_dataset_run_modal.py` | Modal deployment for running the full pipeline on cloud GPUs |
