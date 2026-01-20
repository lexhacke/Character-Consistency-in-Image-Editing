from image_compositor import ImageCompositor
from dataset import PicobananaDataset
import asyncio, os, PIL, torch, json, shutil
import numpy as np
from dotenv import load_dotenv
from blending import expand_mask, blend

load_dotenv()

assert os.environ.get('SAVE_PATH') is not None, "Please set the SAVE_PATH in the .env file"

def save_to(dataset, path, frequency_table):
    compositor = ImageCompositor()
    for i, item in enumerate(dataset):
        if item == -1: # Failed dataset get request
            continue

        if item['edit_type'] in frequency_table:
            frequency_table[item['edit_type']] -= 1
            if frequency_table[item['edit_type']] == 0:
                del frequency_table[item['edit_type']]
        else:
            continue

        prompt, original, edited = item['prompt'], item['original'], item['edited']

        # Handling if the dataset returns urls or images (This is done because some VLMs only accept image_url input)
        # Requests the composite_json from the VLM
        if isinstance(original, str):
            composite_json = compositor.get_composite_json(edited, original, prompt)
        else:
            edited = edited.resize(original.size, PIL.Image.BILINEAR)
            composite_json = compositor.get_composite_json(compositor._img_to_bytes(edited), compositor._img_to_bytes(original), prompt)
        print(composite_json)

        # Requests the segmaps from SAM3 given the composite_json
        segmaps = compositor.get_segmaps(edited, original, composite_json)

        # Normalize and log info into metadata
        edited, original = torch.from_numpy(np.array(edited)) / 255, torch.from_numpy(np.array(original)) / 255
        base = original if composite_json['base'] == 'original' else edited
        other = original if composite_json['base'] == 'edited' else edited
        base, other = base.cpu().numpy(), other.cpu().numpy()

        # Simple bucketing "fail" as any instance where SAM3 couldn't segment an object queried by the VLM
        bucket = "fail/" if len(segmaps['failed']['subtraction']) > 0 or len(segmaps['failed']['union']) > 0 else "success/"
        os.makedirs(path+f"/data_sample/"+bucket+f"{i}", exist_ok=True)
        PIL.Image.fromarray((base * 255).astype(np.uint8)).save(path+f"/data_sample/"+bucket+f"{i}/base.jpeg")
        PIL.Image.fromarray((other * 255).astype(np.uint8)).save(path+f"/data_sample/"+bucket+f"{i}/other.jpeg")

        meta = {
            'prompt': prompt,
            'base': composite_json['base'], # The image used as the canvas. i.e. what we subtract from.
            'subtraction': {
                'success': [item[1] for item in segmaps['segmaps']['subtraction']], # Only save segmentation prompt
                'failed': segmaps['failed']['subtraction']
            },
            'union': {
                'success': [item[1] for item in segmaps['segmaps']['union']], # Only save segmentation prompt
                'failed': segmaps['failed']['union']
            }
        }

        # Since SAM3 will output many segmentation maps for one image, we will just take the logical_or of all of them.
        sub_mask = np.zeros_like(original)
        for segmap, obj in segmaps['segmaps']['subtraction']:
            segmap = segmap.unsqueeze(-1)[0]
            segmap = segmap.cpu().numpy()
            sub_mask = np.logical_or(segmap, sub_mask)

        union_mask = np.zeros_like(original)
        for segmap, obj in segmaps['segmaps']['union']:
            segmap = segmap.unsqueeze(-1)[0]
            segmap = segmap.cpu().numpy()
            union_mask = np.logical_or(segmap, union_mask)

        PIL.Image.fromarray((sub_mask * 255).astype(np.uint8)).save(path+f"/data_sample/"+bucket+f"{i}/subtraction_mask.png")
        PIL.Image.fromarray((union_mask * 255).astype(np.uint8)).save(path+f"/data_sample/"+bucket+f"{i}/union_mask.png")

        mask = np.logical_or(union_mask, sub_mask)

        # Ensure broadcastability between H W C images
        if len(base.shape) == 2:
            base = base[:, :, np.newaxis]
        if len(other.shape) == 2:
            other = other[:, :, np.newaxis]
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]

        mask = expand_mask(mask)
        composite = blend(mask, base, other, mode="laplacian")

        PIL.Image.fromarray((mask * 255).astype(np.uint8)).save(path+f"/data_sample/"+bucket+f"{i}/mask.png")
        PIL.Image.fromarray((composite * 255).astype(np.uint8)).save(path+f"/data_sample/"+bucket+f"{i}/composite.jpeg")

        v = compositor.dino_forward(PIL.Image.fromarray((composite * 255).astype(np.uint8)))
        w = compositor.dino_forward(PIL.Image.fromarray(((base if meta['base'] == 'edited' else other) * 255).astype(np.uint8)))
        v = v / v.norm(dim=-1, keepdim=True)
        w = w / w.norm(dim=-1, keepdim=True)
        sim_score = np.dot(v,w)
        meta['similarity_score'] = float(sim_score)
        with open(path+f"/data_sample/"+bucket+f"{i}/meta.json", 'w') as f:
            json.dump(meta, f, indent=4)

if __name__ == "__main__":
    dataset = PicobananaDataset(n = 50)
    freq =  {edittype:2 for edittype in dataset.edit_types}
    asyncio.run(dataset.prepare_data())

    save_to(dataset, os.environ['SAVE_PATH'], freq)
    shutil.make_archive(os.environ['SAVE_PATH'] + '/data_sample', 'zip', os.environ['SAVE_PATH'] + '/data_sample')