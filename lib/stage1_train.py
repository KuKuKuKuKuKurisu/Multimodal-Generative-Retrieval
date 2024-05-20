from datetime import datetime
from os.path import isfile
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import deepspeed
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from tensorboardX import SummaryWriter
from config.train_config import RecommendTrainConfig
from config.global_config import GlobalConfig
from config.dataset_config import DatasetConfig
from utils import collate
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import torch.distributed as dist
import argparse
from peft import LoraConfig, get_peft_model, TaskType

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def recommend_train(
        model,
        dataset,
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

    try:
        # Load saved state.
        print('loading stage1 best model...')
        # model.load_state_dict({k.replace('language_model.base_model.model', 'language_model'): v for k, v in
        #                        state['state_dict'].items()})
        model.load_state_dict(torch.load(DatasetConfig.stage1_model_dir)['state_dict'])
        print('best stage1 model loaded...')
    except:
        pass

    model.stage1_setting()

    # # Data loader.
    # data_loader = DataLoader(
    #     dataset=dataset,
    #     batch_size=2,
    #     shuffle=True,
    #     num_workers=RecommendTrainConfig.num_data_loader_workers,
    #     collate_fn = collate
    # )
    # device = 'cuda:0'
    # model.to(device)

    # Switch to deepspeed mode.
    deepspeed.init_distributed()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, data_loader, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=parameters, training_data=dataset, collate_fn = collate)
    device = model.local_rank

    model.train()
    epoch_id = 0
    min_loss = None
    sum_loss = 0
    count = 0
    for epoch_id in range(epoch_id, RecommendTrainConfig.num_iterations):

        for batch_id, train_data in enumerate(tqdm(data_loader)):

            input, label = train_data

            outputs = model(
                pixel_values = input['pixel_values'].to(device),
                input_ids = input['input_ids'].to(device),
                attention_mask = input['attention_mask'].to(device),
                img_mask = input['img_mask'].to(device),
                labels = label.to(device),
                sp_token= sp_token
            )

            # Encode context.
            loss = outputs['loss']
            model.backward(loss)
            model.step()
            sum_loss += loss.item()
            count += 1
            torch.cuda.empty_cache()

            # Print loss every `TrainConfig.print_freq` batches.
            if (batch_id + 1) % RecommendTrainConfig.print_freq == 0:
                print('epoch: {0}  sim loss: {1:.4f}'.format(epoch_id + 1, sum_loss/count))

        if min_loss is None or (sum_loss/count)<min_loss:
            min_loss = sum_loss/count
            model.save_checkpoint(DatasetConfig.dump_dir, tag='best_model')
        print('Best model saved.')
        sum_loss = 0
        count = 0
