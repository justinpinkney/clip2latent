from PIL import Image
import torch
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

with open("text/dalle.txt", 'rt') as f:
    data = f.readlines()
    data = [x.strip('\n') for x in data]

dalle_data = Path("all-dalle-test")
ims = list(dalle_data.glob("*.png"))

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
similarities = []

with torch.no_grad():
    for im in tqdm(ims):
        image = Image.open(im)
        matches = [x for x in data if x == str(im.name)[29:-4]]

        inputs = processor(text=matches, images=image, return_tensors="pt", padding=True)

        outputs = model(**inputs)
        sim = outputs.logits_per_image/100
        print(sim)
        similarities.append(sim)

print(torch.cat(similarities).mean())