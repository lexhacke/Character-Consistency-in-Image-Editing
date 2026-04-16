from google import genai
import requests, os, json, io, base64, pathlib, sam3
import numpy as np
from PIL import Image as PILImage
from google.genai import types
from transformers import AutoModel, AutoImageProcessor
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from dotenv import load_dotenv
from system_prompt import build_system_prompt

load_dotenv() # Assumes MOONDREAM and GOOGLE key are set in .env file

assert os.environ.get('GOOGLE') is not None, "Please set the GOOGLE API key in the .env file, if its there, you likely need to  cd into src"
assert os.environ.get('MOONDREAM') is not None, "Please set the MOONDREAM API key in the .env file, if its there, you likely need to  cd into src"

class ImageCompositor:
    def __init__(self):
        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        self.convnext = AutoModel.from_pretrained("facebook/dinov3-convnext-small-pretrain-lvd1689m")
        self.dino_forward = lambda image: self.convnext(self.dino_processor(images=image, return_tensors="pt").pixel_values).last_hidden_state[0][0]
        self.gemini_client = genai.Client(api_key=os.environ['GOOGLE'])
        try:
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
        except FileNotFoundError:
            response = requests.get("https://github.com/openai/CLIP/raw/refs/heads/main/clip/bpe_simple_vocab_16e6.txt.gz", stream=True) # Use stream=True for large files
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # 3. Write the content to the local file
            with open(pathlib.Path(sam3.__file__).parent.parent / r"assets/bpe_simple_vocab_16e6.txt.gz", 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)            

            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
        self.system_prompt = build_system_prompt
        self.schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "base": types.Schema(
                    type=types.Type.STRING,
                    description="The base image to use as the canvas. Must be 'original' or 'edited'.",
                    enum=["original", "edited"]
                ),
                "subtract":  types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                "union":  types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING))
            },
            required=["base", "subtract", "union"]
        )
        self.validation_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "keep": types.Schema(
                    type=types.Type.BOOLEAN,
                    description="True if the mask correctly captures the described object, False otherwise."
                ),
            },
            required=["keep"]
        )

    def get_segmaps(self, edited_img, original_img, composite_json):
        """
        Takes the original, edited image pair along with a composite_json with keys "union" and "subtract" which map to a list of
        prompts for SAM3 to segment away.

        Returns dict with:
            segmaps.subtraction: list of (list_of_masks, prompt) — each mask is [H, W] tensor
            segmaps.union: same format
            failed.subtraction / failed.union: list of prompt strings that failed
        """
        assert composite_json['base'] in {'edited', 'original'}
        assert 'subtract' in composite_json and 'union' in composite_json

        base = original_img if composite_json['base'] == 'original' else edited_img
        other = original_img if composite_json['base'] == 'edited' else edited_img
        subtract = composite_json['subtract']
        union = composite_json['union']
        output = {
            'failed': {'subtraction': [], 'union': []},
            'segmaps': {'subtraction': [], 'union': []}
        }

        # Subtract from base
        inference_state = self._set_image(base)
        for subtracted_object in subtract:
            mask_list = self._call_sam3(inference_state, subtracted_object, base)
            if mask_list is None:
                print("Trying Moondream point fallback")
                bbox = self._call_moondream_bbox(base, subtracted_object)
                if bbox is None:
                    output['failed']['subtraction'].append(subtracted_object)
                    continue
                mask_list = self._call_sam3(inference_state, bbox, base)
                if mask_list is None:
                    output['failed']['subtraction'].append(subtracted_object)
                    continue
            # mask_list is [(mask, score), ...] — keep all individual masks
            masks_only = [m for m, s in mask_list]
            output['segmaps']['subtraction'].append((masks_only, subtracted_object))

        # Union from other
        inference_state = self._set_image(other)
        for union_object in union:
            mask_list = self._call_sam3(inference_state, union_object, other)
            if mask_list is None:
                print("Trying Moondream point fallback")
                bbox = self._call_moondream_bbox(other, union_object)
                if bbox is None:
                    output['failed']['union'].append(union_object)
                    continue
                mask_list = self._call_sam3(inference_state, bbox, other)
                if mask_list is None:
                    output['failed']['union'].append(union_object)
                    continue
            masks_only = [m for m, s in mask_list]
            output['segmaps']['union'].append((masks_only, union_object))

        return output

    def get_composite_json(self, edited_img, original_img, prompt):
        response = self.gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                'Original image:',
                types.Part.from_bytes(data=original_img, mime_type='image/jpeg'),
                'Edited image:',
                types.Part.from_bytes(data=edited_img, mime_type='image/png'),
                self.system_prompt(prompt)
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self.schema
            )
        )
        if response.text is None:
            print(f"Gemini API returned None for prompt: {prompt[:100]}")
            print(f"Response: {response}")
            raise ValueError("Gemini API returned empty response")
        return json.loads(response.text)

    def _url_to_img_bytes(self, url):
        response = requests.get(url)
        response.raise_for_status()
        return response.content

    def _img_to_bytes(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()

    def validate_mask(self, image, union_mask, object_description):
        """
        Ask Gemini whether the union mask correctly captures the described object.
        image: PIL Image the mask was extracted from.
        union_mask: np.ndarray [H, W] bool — union of all SAM3 masks for this object.
        object_description: str — the text prompt used for segmentation.
        Returns True (keep) or False (discard).
        """
        # Create overlay: original image with red mask overlay
        img_arr = np.array(image).copy()
        overlay = img_arr.copy()
        overlay[union_mask > 0] = [255, 0, 0]
        blended = (0.6 * img_arr + 0.4 * overlay).astype(np.uint8)
        overlay_pil = PILImage.fromarray(blended)

        overlay_bytes = io.BytesIO()
        overlay_pil.save(overlay_bytes, format='JPEG')
        overlay_bytes = overlay_bytes.getvalue()

        response = self.gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                f'Does the red highlighted region correctly capture "{object_description}"? '
                f'Answer keep=true if the mask is a reasonable segmentation of the described object. '
                f'Answer keep=false if the mask is clearly wrong, captures the wrong object, or is empty.',
                types.Part.from_bytes(data=overlay_bytes, mime_type='image/jpeg'),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self.validation_schema
            )
        )
        if response.text is None:
            print(f"Gemini validation returned None for: {object_description}")
            return True  # Default to keep if validation fails
        result = json.loads(response.text)
        keep = result.get("keep", True)
        if not keep:
            print(f"Gemini rejected mask for: {object_description}")
        return keep

    def _set_image(self, img):
        return self.processor.set_image(img)

    def _call_moondream_bbox(self, image, prompt: str):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        resp = requests.post(
            "https://api.moondream.ai/v1/segment",
            headers={
                'Content-Type': 'application/json',
                'X-Moondream-Auth': os.environ.get('MOONDREAM')
            },
            json={
                "image_url": f"data:image/png;base64,{encoded_image}", # need
                "object": prompt
            })

        resp.raise_for_status()
        r = resp.json()
        if 'bbox' not in r:
            print(f"Moondream failed to point at {prompt}")
            return None
        box = r['bbox']
        center = ((box['x_max'] + box['x_min'])/2, (box['y_max'] + box['y_min'])/2)
        hw = (box['x_max'] - box['x_min'], box['y_max'] - box['y_min'])
        return [center[0], center[1], hw[0], hw[1]]

    def _call_sam3(self, inference_state, prompt, image):
        """
        Returns a list of (mask, score) tuples for all masks above threshold,
        sorted by descending confidence. Each mask is [H, W] bool tensor.
        Returns None if no masks found.
        """
        if isinstance(prompt, str):
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
        elif isinstance(prompt, list):
            output = self.processor.add_geometric_prompt(box=prompt, label=True, state=inference_state)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        if masks.shape[0] == 0:
            print(f"No {prompt} found in image")
            return None
        # Return all masks sorted by descending score
        masks_cpu = masks.cpu()
        scores_cpu = scores.cpu()
        result = []
        for idx in scores_cpu.argsort(descending=True):
            result.append((masks_cpu[idx], float(scores_cpu[idx])))
        return result

async def main():
    ds = PicobananaDataset(n=50)
    await ds.prepare_data()
    compositor = ImageCompositor()
    item = ds[0]
    print("Prompt:", item['prompt'])
    print("Edit type:", item['edit_type'])
    composite_json = compositor.get_composite_json(
            compositor._img_to_bytes(item['edited']), 
            compositor._img_to_bytes(item['original']), 
            item['prompt']
        )
    print("Composite JSON:", composite_json)
    item['edited'] = item['edited'].resize(item['original'].size, PIL.Image.BILINEAR)
    segmaps = compositor.get_segmaps(item['edited'], item['original'], composite_json)

    other = item['original'] if composite_json['base'] == 'edited' else item['edited']
    base = item['original'] if composite_json['base'] == 'original' else item['edited']

    subtraction_union = np.zeros_like(base)
    for segmap, _ in segmaps['segmaps']['subtraction']:
        segmap = segmap.unsqueeze(-1)[0]
        segmap = segmap.cpu().numpy()
        subtraction_union = np.logical_or(segmap, subtraction_union)

    union_union = np.zeros_like(other)
    for segmap, _ in segmaps['segmaps']['union']:
        segmap = segmap.unsqueeze(-1)[0]
        segmap = segmap.cpu().numpy()
        union_union = np.logical_or(segmap, union_union)
    
    # Now we need to overlay union_union * other over 1 - subtraction_union * base
    other = item['original'] if composite_json['base'] == 'edited' else item['edited']
    base = item['original'] if composite_json['base'] == 'original' else item['edited']
    overlay = union_union * other
    base = (1 - subtraction_union) * (1 - union_union) * base
    underlay = subtraction_union * other
    composite = overlay + base + underlay
    plt.imshow(composite)
    plt.show()

if __name__ == "__main__":
    from dataset import PicobananaDataset
    import numpy as np
    import matplotlib.pyplot as plt
    import asyncio, PIL
    asyncio.run(main())
