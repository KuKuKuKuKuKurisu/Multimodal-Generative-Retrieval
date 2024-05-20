"""Dataset module."""
import json
from os.path import join, isfile
import re
import torch
from PIL import Image
from torch.utils import data
from config import RecommendTrainConfig
if RecommendTrainConfig.simmc2:
    from config.simmc_dataset_config import SimmcDatasetConfig as DatasetConfig
else:
    from config import DatasetConfig
from utils import get_product_path
from tqdm import tqdm

class Dataset(data.Dataset):
    """Dataset class."""

    def __init__(self, image_paths,
                 image_text,
                 image_taxonomy,
                 processor,
                 replace_token):
        self.image_paths = image_paths
        self.image_text = image_text
        self.image_taxonomy = image_taxonomy
        self.processor = processor
        self.replace_token = replace_token

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img = self.image_paths[index]
        image_path = join(DatasetConfig.image_data_directory, img)
        product_image = Image.open(image_path)
        product_text = self.image_text[index]
        taxonomy = self.image_taxonomy[index]

        image = [product_image]
        prompt = [f'The product image is {self.replace_token}. It depicts a item with following attributes {product_text} '
                   f'Question: generate its taxonomy']
        label = [taxonomy]

        prompt = " ".join(prompt)
        label = " ".join(label)

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        label = self.processor(text=label, return_tensors="pt")['input_ids']

        return inputs, label