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
                 data,
                 processor,
                 replace_token):
        self.image_paths = image_paths
        self.data = data
        self.processor = processor
        self.replace_token = replace_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        images = []
        for img in self.data[index].dialogue_images:
            img_path = self.image_paths[img]
            img_path = join(DatasetConfig.image_data_directory, img_path)
            product_image = Image.open(img_path)
            product_image = self.processor(images=product_image, return_tensors="pt")
            images.append(product_image['pixel_values'])

        text = self.data[index].text
        label = self.data[index].label

        return torch.tensor(text), images, torch.tensor(label)