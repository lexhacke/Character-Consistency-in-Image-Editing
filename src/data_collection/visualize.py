import os, dotenv
import time
import json
import numpy as np
import PIL
import matplotlib.pyplot as plt

dotenv.load_dotenv()

def corr(lex_ct, vit_ct):
    lex, vit = np.array(lex_ct), np.array(vit_ct)
    norm = np.linalg.norm(lex) * np.linalg.norm(vit)
    return np.dot(lex, vit) / norm

def disjoint_ct(lex_ct, vit_ct):
    lex, vit = np.array(lex_ct), np.array(vit_ct)
    lex = lex == 1
    vit = vit > 0.94
    return np.bitwise_xor(lex, vit).sum()

def false_positive(lex_ct, vit_ct):
    """
    Accepted by vit rejected by lex
    """
    lex, vit = np.array(lex_ct), np.array(vit_ct)
    not_lex = lex == 0
    vit = vit > 0.94
    return np.bitwise_and(not_lex, vit).sum()

def false_negative(lex_ct, vit_ct):
    """
    Accepted by lex rejected by vit
    """
    lex, vit = np.array(lex_ct), np.array(vit_ct)
    lex = lex == 1
    not_vit = vit < 0.94
    return np.bitwise_and(lex, not_vit).sum()

n = 0
lex_ct = []
vit_ct = []

folder = "data_sample/success/"

for file in os.listdir(os.environ['SAVE_PATH']+folder):
    plt.figure(figsize=(20, 12))
    try:
        base = np.array(PIL.Image.open(os.environ['SAVE_PATH']+folder+file+'/base.jpeg'))
        comp = np.array(PIL.Image.open(os.environ['SAVE_PATH']+folder+file+'/composite.jpeg'))
        other = np.array(PIL.Image.open(os.environ['SAVE_PATH']+folder+file+'/other.jpeg'))
        meta = json.load(open(os.environ['SAVE_PATH']+folder+file+'/meta.json'))
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

        if n > 0:
            print(f"Lex Pass Rate: {sum(lex_ct)/n:.3}")
            print(f"ViT Pass Rate: {(np.array(vit_ct) > 0.94).sum()/n:.3}")
            print(f"Disjoint: {disjoint_ct(lex_ct, vit_ct)}")
            print(f"ViT False Positive Rate: {false_positive(lex_ct, vit_ct)}")
            print(f"ViT False Negative Rate: {false_negative(lex_ct, vit_ct)}")
            print(f"Correlation: {corr(lex_ct, vit_ct)}")

        time.sleep(1)
        vit_ct.append(meta['similarity_score'])
        if input() == '':
            lex_ct.append(1)
        else:
            lex_ct.append(0)
        n += 1

    except Exception as e:
        print(e)
        continue

print(f"n: {n}")
print(f"Lex Pass Rate: {sum(lex_ct)/n:.3}")
print(f"ViT Pass Rate: {(np.array(vit_ct) > 0.94).sum()/n:.3}")
print(f"Disjoint: {disjoint_ct(lex_ct, vit_ct) / n:.3}")
print(f"ViT False Positive Rate: {false_positive(lex_ct, vit_ct)/n:.3}")
print(f"ViT False Negative Rate: {false_negative(lex_ct, vit_ct)/n:.3}")
print(f"Correlation: {corr(lex_ct, vit_ct):.3}")
