import os
import torch
import clip
from PIL import Image

# Set the path to the directory containing the images
image_dir = "../data/VITON-HD/train/cloth"

# Load the pre-trained CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Iterate over the images in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust the extensions as needed
        # Load the image using PIL
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        
        # Preprocess the image for CLIP
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Generate the text representation using CLIP
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(clip.tokenize(["a photo of a piece of clothing"]).to(device))
            similarity = torch.cosine_similarity(image_features, text_features)
            text_repr = similarity.item()
        
        print(str(text_repr))
        
        # Save the text representation to a file
        # text_filename = os.path.splitext(filename)[0] + ".txt"
        # text_path = os.path.join(image_dir, text_filename)
        # with open(text_path, "w") as file:
        #     file.write(str(text_repr))

print("Image to text conversion completed.")