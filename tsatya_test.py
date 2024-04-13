import torch
from transformers import ViTFeatureExtractor, GPT2LMHeadModel
from PIL import Image

# Load the ViT-GPT2 model
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# Generate captions for the input image
def generate_captions(image_path):
    inputs = preprocess_image(image_path)
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=3, no_repeat_ngram_size=2)
    captions = [feature_extractor.decode(output, skip_special_tokens=True) for output in outputs]
    return captions

# Path to the input image file
image_path = "path/to/your/image.jpg"

# Generate captions for the input image
captions = generate_captions(image_path)

# Print or display the captions
for i, caption in enumerate(captions, 1):
    print(f"Caption {i}: {caption}")