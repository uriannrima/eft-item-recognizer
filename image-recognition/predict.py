from PIL import Image
import requests
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-u", "--URL", help = "Image URL")

args = parser.parse_args()

if args.URL is None:
    print("Please provide an image URL")
    sys.exit(1) 

url = args.URL
image = Image.open(requests.get(url, stream=True).raw)

#repo_location = "./build/vit-base-patch16-224-finetuned-tarkov"
repo_location = "./build/swin-tiny-patch4-window7-224-finetuned-tarkov"

image_processor = AutoImageProcessor.from_pretrained(repo_location)
model = AutoModelForImageClassification.from_pretrained(repo_location)

encoding = image_processor(image.convert("RGB"), return_tensors="pt")
print(encoding.pixel_values.shape)

# forward pass
with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])