import torch
from torch.optim import Adam
import time
from torch.utils.data import DataLoader
import deepspeed
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from config.train_config import RecommendTrainConfig
from config.dataset_config import DatasetConfig
from utils import stage2_collate
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from lib.stage2_valid import recommend_valid

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def recommend_train(
        model,
        train_dataset,
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

    model.stage1_model.stage2_setting()

    if args.use_deepspeed:
        # Switch to deepspeed mode.
        deepspeed.init_distributed()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        model, optimizer, train_data_loader, _ = deepspeed.initialize(
            args=args, model=model, model_parameters=parameters, training_data=train_dataset, collate_fn = stage2_collate)
        device = model.local_rank
        if os.path.isfile(os.path.join(DatasetConfig.dump_dir, 'latest')):
            print('laoding stage2 model...')
            model.load_checkpoint(DatasetConfig.dump_dir)
            print('laoded stage2 model...')
    else:
        # Data loader.
        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=RecommendTrainConfig.batch_size,
            shuffle=True,
            num_workers=RecommendTrainConfig.num_data_loader_workers,
            collate_fn = stage2_collate
        )

        trainable_param = []
        for p in model.parameters():
            if p.requires_grad:
                trainable_param.append(p)
        optimizer = torch.optim.AdamW(trainable_param, lr=RecommendTrainConfig.learning_rate)
        total_steps = len(train_data_loader) * RecommendTrainConfig.num_iterations
        scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=total_steps, num_warmup_steps=total_steps * 0.01)
        device = 0

    epoch_id = 0
    min_loss = None
    sum_loss = 0
    count = 0

    model.train()

    for epoch_id in range(epoch_id, RecommendTrainConfig.num_iterations):

        for batch_id, train_data in enumerate(tqdm(train_data_loader)):

            text, attn_mask, imgs, img_mask, label = train_data

            outputs = model(
                imgs.to(device),
                text.to(device),
                attn_mask.to(device),
                img_mask.to(device),
                label.to(device),
                sp_token
            )

            loss = outputs['loss']

            if args.use_deepspeed:
                model.backward(loss)
                model.step()
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            sum_loss += loss.item()
            count += 1
            torch.cuda.empty_cache()

            # Print loss every `TrainConfig.print_freq` batches.
            if (batch_id + 1) % RecommendTrainConfig.print_freq == 0:
                print('epoch: {0}  sim loss: {1:.4f}'.format(epoch_id + 1, sum_loss/count))
                sum_loss = 0
                count = 0

            # # Valid every `TrainConfig.print_freq` batches.
            # if (batch_id + 1) % RecommendTrainConfig.valid_freq == 0:
            #     valid_loss = recommend_valid(model, valid_dataset, args, sp_token)
            #     print('epoch: {0}  valid loss: {1:.4f}'.format(epoch_id + 1, valid_loss))
            #     if min_loss is None or valid_loss<min_loss:
            #         min_loss = valid_loss
            #         if args.use_deepspeed:
            #             model.save_checkpoint(DatasetConfig.dump_dir, tag='best_model')
            #         else:
            #             save_dict = {
            #                 'epoch_id': epoch_id,
            #                 'min_valid_loss': min_loss,
            #                 'optimizer': optimizer.state_dict(),
            #                 'context_text_encoder':model.state_dict()
            #             }
            #             torch.save(save_dict, DatasetConfig.stage2_model_dir)
            #         print('Best stage2 model saved.')
            # Valid every `TrainConfig.print_freq` batches.


        valid_loss = recommend_valid(model, valid_dataset, args, sp_token)
        print('epoch: {0}  valid loss: {1:.4f}'.format(epoch_id + 1, valid_loss))
        if min_loss is None or valid_loss<min_loss:
            min_loss = valid_loss
            if args.use_deepspeed:
                model.save_checkpoint(DatasetConfig.dump_dir, tag='best_model')
            else:
                save_dict = {
                    'epoch_id': epoch_id,
                    'min_valid_loss': min_loss,
                    'optimizer': optimizer.state_dict(),
                    'context_text_encoder':model.state_dict()
                }
                torch.save(save_dict, DatasetConfig.stage2_model_dir)
            print('Best stage2 model saved.')
