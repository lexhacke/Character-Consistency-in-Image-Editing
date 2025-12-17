from image_compositor import ImageCompositor
from dataset import PicobananaDataset
import asyncio, os, PIL, torch, json, shutil
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def save_to(path):
    dataset = PicobananaDataset()
    asyncio.run(dataset.prepare_data())

    compositor = ImageCompositor()
    good_types = ['Add a new object to the scene', 
                "Change an object's attribute (e.g., color/material)",
                'Remove an existing object',              
                'Relocate an object (change its position/spatial relation)',
                'Change the size/shape/orientation of an object',
                'Replace one object category with another']

    for i,item in enumerate(dataset):
        if item['edit_type'] not in good_types:
            continue
        os.makedirs(path+f"/data_sample/{i}", exist_ok=True)
        prompt, original, edited = item['prompt'], item['original'], item['edited']
        edited = edited.resize(original.size, PIL.Image.BILINEAR)
        composite_json = compositor.get_composite_json(compositor._img_to_bytes(edited), compositor._img_to_bytes(original), prompt)
        segmaps = compositor.get_segmaps(edited, original, composite_json)
        edited, original = torch.from_numpy(np.array(edited)) / 255, torch.from_numpy(np.array(original)) / 255
        base = original if composite_json['base'] == 'original' else edited
        other = original if composite_json['base'] == 'edited' else edited
        base, other = base.cpu().numpy(), other.cpu().numpy()
        PIL.Image.fromarray((base * 255).astype(np.uint8)).save(path+f"/data_sample/{i}/base.jpeg")
        PIL.Image.fromarray((other * 255).astype(np.uint8)).save(path+f"/data_sample/{i}/other.jpeg")

        meta = {
            'prompt': prompt,
            'base': composite_json['base'],
            'subtraction': {
                'success': [item[1] for item in segmaps['segmaps']['subtraction']], # Only save segmentation prompt
                'failed': segmaps['failed']['subtraction']
            }, 
            'union': {
                'success': [item[1] for item in segmaps['segmaps']['union']], # Only save segmentation prompt
                'failed': segmaps['failed']['union']
            }
        }

        with open(path+f"/data_sample/{i}/meta.json", 'w') as f:
            json.dump(meta, f, indent=4)

        subtraction_union = np.zeros_like(original)
        for segmap, obj in segmaps['segmaps']['subtraction']:
            segmap = segmap.unsqueeze(-1)[0]
            segmap = segmap.cpu().numpy()

            subtraction_union = np.logical_or(segmap, subtraction_union)
            segmented_img = 0.8 * segmap * base + 0.2 * base
            # Save
            segmented_img = segmented_img * 255
            save_image = PIL.Image.fromarray(segmented_img.astype(np.uint8))
            save_image.save(path+f"/data_sample/{i}/{obj}.jpeg")

        union_union = np.zeros_like(original)
        for segmap, obj in segmaps['segmaps']['union']:
            segmap = segmap.unsqueeze(-1)[0]
            segmap = segmap.cpu().numpy()
            
            union_union = np.logical_or(segmap, union_union)
            segmented_img = 0.8 * segmap * other + 0.2 * other
            # Save
            segmented_img = segmented_img * 255
            save_image = PIL.Image.fromarray(segmented_img.astype(np.uint8))
            save_image.save(path+f"/data_sample/{i}/{obj}.jpeg")
        
        # Now we need to overlay union_union * other over 1 - subtraction_union * base
        overlay = union_union * other
        base = (1 - subtraction_union) * (1 - union_union) * base
        underlay = subtraction_union * other
        composite = overlay + base + underlay
        PIL.Image.fromarray((composite * 255).astype(np.uint8)).save(path+f"/data_sample/{i}/composite.jpeg")

if __name__ == "__main__":
    save_to(os.environ['SAVE_PATH'])
    shutil.make_archive(os.environ['SAVE_PATH'] + '/data_sample', 'zip', os.environ['SAVE_PATH'] + '/data_sample')