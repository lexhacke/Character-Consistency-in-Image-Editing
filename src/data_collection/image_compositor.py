from google import genai
import requests, os, json, io, base64, pathlib, sam3
from google.genai import types
from transformers import AutoModel, AutoImageProcessor
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from dotenv import load_dotenv

load_dotenv() # Assumes MOONDREAM and GOOGLE key are set in .env file

assert os.environ.get('GOOGLE') is not None, "Please set the GOOGLE API key in the .env file"
assert os.environ.get('MOONDREAM') is not None, "Please set the MOONDREAM API key in the .env file"

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
        self.system_prompt = lambda prompt : f"""
You are an Image Compositing Logic engine. Your goal is to construct a "layering recipe" to merge an Original Image and an Edited Image with pixel-perfect precision.

The image was edited with the following prompt:
{prompt}

### FEASIBILITY CHECK (CRITICAL)
Before constructing a recipe, determine if pixel-sharing is possible.
If the edit involves a **Global Transformation**, return
{{
  "base": "edited",
  "subtract": [],
  "union": []
}}
Guarentee that "base" is "edited" and both "subtract" and "union" are empty lists
Global Transformations include:
- **Art Style Changes:** (e.g., "make it a pencil sketch", "Cyberpunk style", "Oil painting")
- **Outpainting/Expansion:** (e.g., "zoom out", "expand the borders")
- **Camera/Perspective Shifts:** (e.g., "view from above", "wide angle")
- **Lighting/Time of Day:** (e.g., "make it night time", "sunset lighting")

If the edit is **Local** (adding, removing, or modifying specific objects), proceed to the JSON schema.

Return a JSON object with this schema:
{{
  "base": "original" | "edited",
  "subtract": ["list of objects to find in the BASE image and remove"],
  "union": ["list of objects to find in the OTHER image and paste on top"]
}}

### CRITICAL SEGMENTATION RULES:
1. **Visual Descriptions over Nouns:** The segmentation model is literal. Use visual descriptors (appearance, color, texture).
   - GOOD: "red glossy plastic chair", "vintage leather brown suitcase"
2. **Logic Rules:**
   - **Standard Edits:** Base = "original". Union = [New object].
   - **Removals:** Base = "original". Subtract = [Object to remove].
   - **Background Changes:** Base = "edited". Union = [Subject from original to preserve].

### Examples:

Input: "Give the hamster a gold chain"
Output: {{
  "base": "original",
  "subtract": [],
  "union": ["thick gold metal chain necklace"]
}}

Input: "Turn this into a 3D Pixar animation"
Output: {{}}

Input: "Zoom out and show the rest of the room"
Output: {{}}

Input: "Change the background to a volcano"
Output: {{
  "base": "edited",
  "subtract": [],
  "union": ["small grey and white fluffy hamster"]
}}

Input: "Make it look like a van gogh painting"
Output: {{}}
"""
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

    def get_segmaps(self, edited_img, original_img, composite_json):
        """
        Takes the original, edited image pair along with a composite_json with keys "union" and "subtract" which map to a list of
        prompts for SAM3 to segment away.
        """
        assert composite_json['base'] in {'edited', 'original'}
        assert 'subtract' in composite_json and 'union' in composite_json

        base = original_img if composite_json['base'] == 'original' else edited_img
        other = original_img if composite_json['base'] == 'edited' else edited_img
        subtract = composite_json['subtract']
        union = composite_json['union']
        output = {
            'failed':
            {
                'subtraction':[],
                'union':[]
            },
            'segmaps':
            {
                'subtraction':[],
                'union':[]
            }
        }

        # Subtract from base
        inference_state = self._set_image(base)
        for subtracted_object in subtract:
            seg = self._call_sam3(inference_state, subtracted_object, base)
            if seg is None:
                # Try moondream point
                print("Trying Moondream point fallback")
                bbox = self._call_moondream_bbox(base, subtracted_object)
                if bbox is None:
                    output['failed']['subtraction'].append(subtracted_object)
                    continue
                seg = self._call_sam3(inference_state, bbox, base)
                if seg is None:
                    output['failed']['subtraction'].append(subtracted_object)
                    continue
            output['segmaps']['subtraction'].append((seg, subtracted_object))

        # Union from other
        inference_state = self._set_image(other)
        for union_object in union:
            seg = self._call_sam3(inference_state, union_object, base)
            if seg is None:
                # Try moondream point
                print("Trying Moondream point fallback")
                bbox = self._call_moondream_bbox(base, union_object)
                if bbox is None:
                    output['failed']['union'].append(union_object)
                    continue
                seg = self._call_sam3(inference_state, bbox, base)
                if seg is None:
                    output['failed']['union'].append(union_object)
                    continue
            output['segmaps']['union'].append((seg, union_object))

        return output

    def get_composite_json(self, edited_img, original_img, prompt):
        response = self.gemini_client.models.generate_content(
            model='gemini-2.5-flash-lite',
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
        return json.loads(response.text)

    def _url_to_img_bytes(self, url):
        response = requests.get(url)
        response.raise_for_status()
        return response.content

    def _img_to_bytes(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()

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
        if isinstance(prompt, str):
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
        elif isinstance(prompt, list):
            output = self.processor.add_geometric_prompt(box=prompt, label=True, state=inference_state)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        if masks.shape[0] == 0:
            print(f"No {prompt} found in image")
            return None
        return masks.cpu()[0]

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