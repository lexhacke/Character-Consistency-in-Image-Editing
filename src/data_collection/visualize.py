import os, dotenv
import json
import numpy as np
import PIL
import matplotlib.pyplot as plt

dotenv.load_dotenv()

folder = "data_sample/success/"

for file in os.listdir(os.environ['SAVE_PATH']+folder):
    try:
        base = np.array(PIL.Image.open(os.environ['SAVE_PATH']+folder+file+'/base.jpeg'))
        comp = np.array(PIL.Image.open(os.environ['SAVE_PATH']+folder+file+'/composite.jpeg'))
        other = np.array(PIL.Image.open(os.environ['SAVE_PATH']+folder+file+'/other.jpeg'))
        meta = json.load(open(os.environ['SAVE_PATH']+folder+file+'/meta.json'))
        if meta['similarity_score'] > 0.94:
           continue
        print(file)
        stitch = np.concat([other, base, comp] if meta['base']=='edited' else [base, other, comp], axis=1)
        plt.imshow(stitch)
        plt.title(f"Original Image, Edited Image, Composite Image (Stitch Result)\nCosine: {meta['similarity_score']:.3}")
        plt.figure(figsize=(10, 6))
        plt.show()
        print("Prompt:", meta['prompt'])
        print("Base:", meta['base'])
        print("Similarity Score:", meta['similarity_score'])
        if len(meta['union']['success']) > 0:
            print(f"Stitched {', '.join(meta['union']['success'])} from other")
        if len(meta['union']['failed']) > 0:
            print(f"Failed to find {', '.join(meta['union']['failed'])} in other")
        if len(meta['subtraction']['success']) > 0:
            print(f"Cut {', '.join(meta['subtraction']['success'])} from {meta['base']}")
        if len(meta['subtraction']['failed']) > 0:
            print(f"Failed to cut {', '.join(meta['subtraction']['failed'])} from {meta['base']}")
    except Exception as e:
        print(e)
        continue
