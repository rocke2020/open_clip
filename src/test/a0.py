from pathlib import Path
import os, sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import logging
from PIL import Image
sys.path.append(os.path.abspath('.'))
import open_clip
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))
ic.lineWrapWidth = 120


logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, datefmt='%y-%m-%d %H:%M',
    format='%(asctime)s %(filename)s %(lineno)d: %(message)s')

# logger.info(open_clip.list_pretrained())

# ViT-B-16 laion2b_s34b_b79k
# RN50 openai
model, _, preprocess = open_clip.create_model_and_transforms(
    'RN50', pretrained='openai', cache_dir='/mnt/nas1/models/clip')
tokenizer = open_clip.get_tokenizer('RN50')

image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # t 100, tensor([[0.1775, 0.5906, 0.2319]])
    # t 1, tensor([[0.3317, 0.3357, 0.3326]])
    text_probs = (1 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]