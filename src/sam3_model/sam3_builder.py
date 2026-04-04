import requests
import pathlib
import sam3
from sam3 import build_sam3_image_model

def build_sam3_image_model():
    try:
        model = build_sam3_image_model()
    except FileNotFoundError:
        response = requests.get("https://github.com/openai/CLIP/raw/refs/heads/main/clip/bpe_simple_vocab_16e6.txt.gz", stream=True) # Use stream=True for large files
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # 3. Write the content to the local file
        with open(pathlib.Path(sam3.__file__).parent.parent / r"assets/bpe_simple_vocab_16e6.txt.gz", 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)            

        model = build_sam3_image_model()
    return model
