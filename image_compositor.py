from google import genai
import requests, os, json, io
from google.genai import types
from transformers import pipeline
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from dotenv import load_dotenv

load_dotenv()

class ImageCompositor:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ['gemini3'])
        try:
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
        except FileNotFoundError as e:
            response = requests.get("https://github.com/openai/CLIP/raw/refs/heads/main/clip/bpe_simple_vocab_16e6.txt.gz", stream=True) # Use stream=True for large files
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # 3. Write the content to the local file
            with open(e.filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
        self.system_prompt = system_prompt = lambda prompt : f"""
You are an Image Compositing Logic engine. Your goal is to construct a "layering recipe" to merge an Original Image and an Edited Image with pixel-perfect precision.

The image was edited with the following prompt:
{prompt}

You must decide which image is the "Base" (canvas) and which objects to "Subtract" (erase to reveal the other image) or "Union" (paste from the other image).

Return a JSON object with this schema:
{{
  "base": "original" | "edited",
  "subtract": ["list of objects to find in the BASE image and remove"],
  "union": ["list of objects to find in the OTHER image and paste on top"]
}}

### CRITICAL SEGMENTATION RULES:
1. **Visual Descriptions over Nouns:** The segmentation model is literal. Do not just name the object; describe its **appearance, color, and texture**.
   - BAD: "latte"
   - GOOD: "white paper coffee cup"
   - BAD: "durag"
   - GOOD: "purple silky head covering"
   - BAD: "sword"
   - GOOD: "glowing blue laser sword"

2. **Logic Rules:**
   - **Standard Edits (Add/Modify):** Base = "original". Union = [The new/modified object description]. Subtract = [].
   - **Removals:** Base = "original". Subtract = [Description of object to remove]. Union = [].
   - **Background Changes:** Base = "edited". Union = [Description of main subject from original to preserve]. Subtract = [].

### Examples:

Input: "Give the hamster a gold chain"
Output: {{
  "base": "original",
  "subtract": [],
  "union": ["thick gold metal chain necklace"]
}}

Input: "Remove the stop sign"
Output: {{
  "base": "original",
  "subtract": ["red octagonal stop sign"],
  "union": []
}}

Input: "Change the background to a volcano"
Output: {{
  "base": "edited",
  "subtract": [],
  "union": ["small grey and white hamster"]
}}

Input: "Put a durag on him"
Output: {{
  "base": "original",
  "subtract": [],
  "union": ["dark blue velvet head covering"]
}}
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
            seg = self._call_sam3(inference_state, subtracted_object)
            if seg is None:
                output['failed']['subtraction'].append(subtracted_object)
                continue
            output['segmaps']['subtraction'].append((seg, subtracted_object))

        # Union from other
        inference_state = self._set_image(other)
        for union_object in union:
            seg = self._call_sam3(inference_state, union_object)
            if seg is None:
                output['failed']['union'].append(union_object)
                continue
            output['segmaps']['union'].append((seg, union_object))

        return output

    def get_composite_json(self, edited_img, original_img, prompt):
        response = self.client.models.generate_content(
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

    def _call_sam3(self, inference_state, prompt):
        output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        if masks.shape[0] == 0:
            print(f"No {prompt} found in image")
            return None
        return masks.cpu()[0]