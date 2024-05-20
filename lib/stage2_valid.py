import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from config.valid_config import RecommendValidConfig
from utils import stage2_collate
from tqdm import tqdm

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def recommend_valid(
        model,
        valid_dataset,
        args,
        sp_token
):
    """Recommend train.

    Args:
        context_text_encoder (TextEncoder): Context text encoder.
        context_image_encoder (ImageEncoder): Context image encoder.
        context_encoder (ContextEncoder): Gat.
        train_dataset (Dataset): Train dataset.
        valid_dataset (Dataset): Valid dataset.
        test_dataset (Dataset): Test dataset.
        model_file (str): Saved model file.

    """

    # Data loader.
    valid_data_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=RecommendValidConfig.batch_size,
        shuffle=True,
        num_workers=RecommendValidConfig.num_data_loader_workers,
        collate_fn = stage2_collate
    )

    if args.use_deepspeed:
        device = model.local_rank
    else:
        device = 0

    sum_loss = 0
    count = 0

    model.eval()

    for batch_id, valid_data in enumerate(tqdm(valid_data_loader)):

        text, attn_mask, imgs, img_mask, label = valid_data

        with torch.no_grad():
            outputs = model(
                imgs.to(device),
                text.to(device),
                attn_mask.to(device),
                img_mask.to(device),
                label.to(device),
                sp_token
            )

        loss = outputs['loss']

        sum_loss += loss.item()
        count += 1

        # if count > RecommendValidConfig.num_batches:
        #     break

    model.train()
    return sum_loss/count
