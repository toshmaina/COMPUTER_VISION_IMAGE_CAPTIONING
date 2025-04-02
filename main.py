##################################################
## Import required libraries
##################################################

# Transformer models and tokenizers for image captioning
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset
# HTTP requests for image downloading
import requests
# Deep learning framework
import torch
# Numerical and image processing
import numpy as np
from PIL import Image
import pickle
# Visualization
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Suppress warning messages
import warnings
warnings.filterwarnings('ignore')

# Load the pre-trained Vision Transformer (ViT) + GPT2 model for image captioning
# This model combines visual understanding with natural language generation
model_raw = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Initialize the image processor for ViT and tokenizer for GPT2
# Image processor handles the image preprocessing required by the ViT model
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# Tokenizer converts text to tokens that GPT2 can understand
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def show_n_generate(url, greedy=True, model=model_raw):
    """
    Downloads an image from a URL, displays it, and generates a caption.
    
    Args:
        url (str): URL of the image to process
        greedy (bool): If True, uses greedy decoding; if False, uses sampling with top-k
        model: The vision-language model to use for caption generation
    
    Returns:
        None: Displays image and prints generated caption
    """
    # Download and open image from URL
    image = Image.open(requests.get(url, stream=True).raw)
    # Process image into tensor format required by the model
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    # Display the image
    plt.imshow(np.asarray(image))
    plt.show()

    # Generate caption using either greedy or sampling strategy
    if greedy:
        # Greedy decoding: always choose most likely next token
        generated_ids = model.generate(pixel_values, max_new_tokens=30)
    else:
        # Sampling with top-k: randomly sample from k most likely tokens
        generated_ids = model.generate(
            pixel_values,
            do_sample=True,
            max_new_tokens=30,
            top_k=5)
    # Convert generated token IDs back to text
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)

# Example image URL from COCO dataset
#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# Additional example URLs (commented out)
# url = "https://raw.githubusercontent.com/michael-franke/npNLG/main/neural_pragmatic_nlg/pics/06-3DS-example.jpg"
# url = "https://img.welt.de/img/sport/mobile102025155/9292509877-ci102l-w1024/hrubesch-rummenigge-BM-Berlin-Gijon-jpg.jpg"
url = "https://faroutmagazine.co.uk/static/uploads/2021/09/The-Cover-Uncovered-The-severity-of-Rage-Against-the-Machines-political-message.jpg"
# url = "https://media.npr.org/assets/img/2022/03/13/2ukraine-stamp_custom-30c6e3889c98487086d76869f8ba6a8bfd2fd5a1.jpg"

# Generate caption for the image using sampling strategy
show_n_generate(url, greedy=False)