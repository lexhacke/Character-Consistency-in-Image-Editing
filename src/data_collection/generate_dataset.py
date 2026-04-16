from image_compositor import ImageCompositor
from dataset import PicobananaDataset
import asyncio, os, PIL, torch, json, shutil
import numpy as np
from dotenv import load_dotenv
from blending import expand_mask, blend

load_dotenv()

assert os.environ.get('SAVE_PATH') is not None, "Please set the SAVE_PATH in the .env file, if its there, you likely need to  cd into src"

def save_to(dataset, path, frequency_table, commit_fn=None, precomputed=None):
    compositor = ImageCompositor()
    for i, item in enumerate(dataset):
        # Resume support: skip already-processed items
        if (os.path.exists(path + f"/data_sample/success/{i}/meta.json") or
            os.path.exists(path + f"/data_sample/fail/{i}/meta.json")):
            continue

        if item == -1: # Failed dataset get request
            continue

        edit_type = item['edit_type']
        if edit_type not in frequency_table:
            continue

        prompt, original, edited = item['prompt'], item['original'], item['edited']

        # Use precomputed Gemini result if available (from batch API), otherwise call online
        dataset_key = item.get('key') if isinstance(item, dict) else None
        lookup_key = dataset_key or f"req-{i}"
        composite_json = None

        if precomputed is not None:
            if lookup_key not in precomputed:
                print(f"[{i}] No precomputed result for key {lookup_key}, skipping")
                continue
            composite_json = precomputed[lookup_key]
            if composite_json is None:
                print(f"[{i}] Batch result was None for key {lookup_key}, skipping")
                continue
            if not isinstance(original, str):
                edited = edited.resize(original.size, PIL.Image.BILINEAR)
        elif isinstance(original, str):
            composite_json = compositor.get_composite_json(edited, original, prompt)
        else:
            edited = edited.resize(original.size, PIL.Image.BILINEAR)
            composite_json = compositor.get_composite_json(compositor._img_to_bytes(edited), compositor._img_to_bytes(original), prompt)

        frequency_table[edit_type] -= 1
        if frequency_table[edit_type] == 0:
            del frequency_table[edit_type]

        # Requests the segmaps from SAM3 given the composite_json
        segmaps = compositor.get_segmaps(edited, original, composite_json)

        # Normalize and log info into metadata
        edited, original = torch.from_numpy(np.array(edited)) / 255, torch.from_numpy(np.array(original)) / 255
        base = original if composite_json['base'] == 'original' else edited
        other = original if composite_json['base'] == 'edited' else edited
        base, other = base.cpu().numpy(), other.cpu().numpy()

        assert base.ndim == 3 and other.ndim == 3, f"From Picobanana dataset, got {base.shape} or {other.shape} not RGB"

        # Simple bucketing "fail" as any instance where SAM3 couldn't segment an object queried by the VLM
        bucket = "fail/" if len(segmaps['failed']['subtraction']) > 0 or len(segmaps['failed']['union']) > 0 else "success/"
        sample_dir = path + f"/data_sample/" + bucket + f"{i}"
        os.makedirs(sample_dir, exist_ok=True)
        PIL.Image.fromarray((base * 255).astype(np.uint8)).save(sample_dir + "/base.jpeg")
        PIL.Image.fromarray((other * 255).astype(np.uint8)).save(sample_dir + "/other.jpeg")

        h, w = original.shape[:2]

        # Process subtraction masks: save all individual masks, no Gemini gate
        sub_meta = {'success': [], 'failed': segmaps['failed']['subtraction']}
        sub_mask = np.zeros((h, w), dtype=bool)
        sub_mask_idx = 0
        for masks_list, obj_prompt in segmaps['segmaps']['subtraction']:
            obj_mask_files = []
            for mask_tensor in masks_list:
                m = mask_tensor[0].cpu().numpy() if mask_tensor.ndim == 3 else mask_tensor.cpu().numpy()
                assert m.ndim == 2, f"Expected H, W segmap got {m.shape}"
                mask_filename = f"sub_{sub_mask_idx}.png"
                PIL.Image.fromarray((m * 255).astype(np.uint8)).save(sample_dir + "/" + mask_filename)
                obj_mask_files.append(mask_filename)
                sub_mask = np.logical_or(sub_mask, m)
                sub_mask_idx += 1
            sub_meta['success'].append({'prompt': obj_prompt, 'masks': obj_mask_files})

        # Process union masks: same pattern
        union_meta = {'success': [], 'failed': segmaps['failed']['union']}
        union_mask = np.zeros((h, w), dtype=bool)
        union_mask_idx = 0
        for masks_list, obj_prompt in segmaps['segmaps']['union']:
            obj_mask_files = []
            for mask_tensor in masks_list:
                m = mask_tensor[0].cpu().numpy() if mask_tensor.ndim == 3 else mask_tensor.cpu().numpy()
                assert m.ndim == 2, f"Expected H, W segmap got {m.shape}"
                mask_filename = f"union_{union_mask_idx}.png"
                PIL.Image.fromarray((m * 255).astype(np.uint8)).save(sample_dir + "/" + mask_filename)
                obj_mask_files.append(mask_filename)
                union_mask = np.logical_or(union_mask, m)
                union_mask_idx += 1
            union_meta['success'].append({'prompt': obj_prompt, 'masks': obj_mask_files})

        # Save union masks
        PIL.Image.fromarray((sub_mask * 255).astype(np.uint8)).save(sample_dir + "/subtraction_mask.png")
        PIL.Image.fromarray((union_mask * 255).astype(np.uint8)).save(sample_dir + "/union_mask.png")

        mask = np.logical_or(union_mask, sub_mask)

        # Ensure broadcastability between H W C images
        base = base[:, :, np.newaxis] if base.ndim == 2 else base
        other = other[:, :, np.newaxis] if other.ndim == 2 else other
        mask = mask[:, :, np.newaxis] if mask.ndim == 2 else mask

        mask = expand_mask(mask)
        assert (mask.ndim == base.ndim and base.ndim == other.ndim and base.ndim == 3), f"Why is this not H,W,C? {mask.shape, base.shape, base.shape}"
        composite = blend(mask, base, other, mode="laplacian")

        PIL.Image.fromarray((mask * 255).astype(np.uint8)).save(sample_dir + "/mask.png")
        composite_pil = PIL.Image.fromarray((composite * 255).astype(np.uint8))
        composite_pil.save(sample_dir + "/composite.jpeg")

        # DINO similarity score
        edited_for_dino = PIL.Image.fromarray(((base if composite_json['base'] == 'edited' else other) * 255).astype(np.uint8))
        v = compositor.dino_forward(composite_pil)
        w_vec = compositor.dino_forward(edited_for_dino)
        v = v / v.norm(dim=-1, keepdim=True)
        w_vec = w_vec / w_vec.norm(dim=-1, keepdim=True)
        sim_score = float(np.dot(v, w_vec))

        # Gemini validation: compare composite vs edited image
        edited_pil = PIL.Image.fromarray((edited * 255).astype(np.uint8)) if not isinstance(edited, PIL.Image.Image) else edited
        passed, reason = compositor.validate_composite(composite_pil, edited_pil, prompt)

        meta = {
            'prompt': prompt,
            'base': composite_json['base'],
            'subtraction': sub_meta,
            'union': union_meta,
            'similarity_score': sim_score,
            'gemini_validation': {'pass': passed, 'reason': reason},
        }

        # Re-bucket to fail if Gemini rejects
        if not passed and bucket == "success/":
            new_dir = path + f"/data_sample/fail/{i}"
            os.makedirs(new_dir, exist_ok=True)
            for f_name in os.listdir(sample_dir):
                shutil.move(sample_dir + "/" + f_name, new_dir + "/" + f_name)
            os.rmdir(sample_dir)
            sample_dir = new_dir
            bucket = "fail/"

        with open(sample_dir + "/meta.json", 'w') as f:
            json.dump(meta, f, indent=4)

        if commit_fn and i % 50 == 0:
            commit_fn()

if __name__ == "__main__":
    dataset = PicobananaDataset(n=100_000, return_img=True)
    asyncio.run(dataset.prepare_data())

    freq = {edittype:float('inf') for edittype in dataset.edit_types}
    try:
        with torch.no_grad():
            save_to(dataset, os.environ['SAVE_PATH'], freq)
    except Exception as e:
        print(e)
        if input('Press e to delete ') == 'e':
            shutil.rmtree(os.environ['SAVE_PATH'] + '/data_sample')
    finally:
        shutil.make_archive(os.environ['SAVE_PATH'] + '/data_sample', 'zip', os.environ['SAVE_PATH'] + '/data_sample')
