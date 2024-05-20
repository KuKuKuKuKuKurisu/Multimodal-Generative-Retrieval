#!/usr/bin/python3
import argparse
import math
import re
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from nltk.translate.bleu_score import sentence_bleu
from config import RecommendTrainConfig, GlobalConfig, DatasetConfig
from torch.utils.data import DataLoader
from utils import stage2_collate, get_pretrained_processor, get_pretrained_model
from tqdm import tqdm
from os.path import isfile, join
import torch
import deepspeed
from dataset.stage2_dataset import Dataset
from utils import load_pkl
from collections import namedtuple
from model.stage2_model import stage2_model

CommonData = namedtuple('CommonData',
                        ['image_paths'])

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def cal_metric(pred_text, target_text, sep_token, retrive_num):
    pred_text = pred_text.split(sep_token)
    target_text = target_text.split(sep_token)

    recall_list = []
    precision_list = []
    ndcg_list = []

    bleu_recall_list = []
    bleu_precision_list = []
    bleu_ndcg_list = []

    correct_list = []
    bleu_correct_list = []
    for pred in pred_text:
        is_correct = False
        bleu_is_correct = False
        for target in target_text:
            if pred.strip()==target.strip():
                is_correct = True
                bleu_is_correct = True
            else:
                # bleu match
                bleu_format_label = set(re.sub(r"[^\w\s]", " ", target).split())
                bleu_format_pred = set(re.sub(r"[^\w\s]", " ", pred).split())
                bleu_sccore = sentence_bleu([list(bleu_format_label)], list(bleu_format_pred), weights=(1, 0, 0, 0))
                if bleu_sccore>0.5:
                    bleu_is_correct = True
            if is_correct or bleu_is_correct:
                break
        correct_list.append(int(is_correct))
        bleu_correct_list.append(int(bleu_is_correct))

    for k in retrive_num:
        recall_list.append(min(1.0, sum(correct_list[:k])/len(target_text)))
        bleu_recall_list.append(min(1.0, sum(bleu_correct_list[:k])/len(target_text)))

        precision_list.append(min(1.0, sum(correct_list[:k])/k))
        bleu_precision_list.append(min(1.0, sum(bleu_correct_list[:k])/k))

        dcg_sum = 0
        sorted_dcg_sum = 0

        bleu_dcg_sum = 0
        bleu_sorted_dcg_sum = 0

        # dcg
        for i in range(min(k, len(correct_list))):
            dcg_sum += (correct_list[i] / math.log2(i+2))
            bleu_dcg_sum += (bleu_correct_list[i] / math.log2(i+2))

        # sorted dcg
        sorted_correct_list = sorted(correct_list[:k], reverse=True)
        sorted_bleu_correct_list = sorted(bleu_correct_list[:k], reverse=True)
        for i in range(min(k, len(correct_list))):
            sorted_dcg_sum += (sorted_correct_list[i] / math.log2(i+2))
            bleu_sorted_dcg_sum += (sorted_bleu_correct_list[i] / math.log2(i+2))

        ndcg_list.append(dcg_sum/(sorted_dcg_sum + 0.00001))
        bleu_ndcg_list.append(bleu_dcg_sum/(bleu_sorted_dcg_sum + 0.00001))

    return recall_list, precision_list, ndcg_list, bleu_recall_list, bleu_precision_list, bleu_ndcg_list

def recommend_test(
        model,
        dataset,
        processor,
        sep_token,
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

    z = model.parameters()

    # Load saved state.
    print('loading stage2 best model...')
    state = torch.load(DatasetConfig.stage2_model_dir)
    # model.load_state_dict({k.replace('language_model.base_model.model', 'language_model'): v for k, v in
    #                        state['state_dict'].items()})
    model.load_state_dict(state['state_dict'])
    print('best stage2 model loaded...')

    # # Switch to train mode.
    # deepspeed.init_distributed()
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # model, optimizer, data_loader, __ = deepspeed.initialize(
    #     args=args, model=model, model_parameters=parameters, training_data=dataset, collate_fn = collate)
    # model.load_checkpoint(os.path.join(DatasetConfig.dump_dir, 'global_step10116'))
    # device = model.local_rank

    # Data loader.
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=RecommendTrainConfig.batch_size,
        shuffle=False,
        num_workers=RecommendTrainConfig.num_data_loader_workers,
        collate_fn = stage2_collate
    )
    device = GlobalConfig.device
    model.to(device)

    model.eval()

    retrive_num = [1, 5, 10, 20]

    precision = [[], [], [], []]
    recall = [[], [], [], []]
    ndcg = [[], [], [], []]

    bleu_precision = [[], [], [], []]
    bleu_recall = [[], [], [], []]
    bleu_ndcg = [[], [], [], []]

    with torch.no_grad():

        for batch_id, train_data in enumerate(tqdm(data_loader)):

            text, attn_mask, imgs, img_mask, label = train_data
            label = label.to(device)

            outputs = model.generate(imgs.to(device), text.to(device), attn_mask.to(device), img_mask.to(device), sp_token) # [b, l] end with 1
            generated_text = processor.batch_decode(outputs, skip_special_tokens=False)
            label[torch.where(label==-100)] = 0
            gt_text = processor.batch_decode(label, skip_special_tokens=False)

            with open('mmd_data/stage2_result.txt', 'a') as f:
                for i in range(len(generated_text)):
                    for sp in processor.tokenizer.all_special_tokens:
                        if sp!= sep_token:
                            gt_text[i] = gt_text[i].replace(sp, '')
                            generated_text[i] = generated_text[i].replace(sp, '')
                    f.write('label ==> ' + gt_text[i] + '\tpred ==> ' + generated_text[i] + '\n')
                    recall_list, precision_list, ndcg_list, bleu_recall_list, bleu_precision_list, bleu_ndcg_list = \
                        cal_metric(generated_text[i], gt_text[i], sep_token, retrive_num)

                    for k_id, k in enumerate(retrive_num):
                        precision[k_id].append(precision_list[k_id])
                        recall[k_id].append(recall_list[k_id])
                        ndcg[k_id].append(ndcg_list[k_id])

                        bleu_precision[k_id].append(bleu_precision_list[k_id])
                        bleu_recall[k_id].append(bleu_recall_list[k_id])
                        bleu_ndcg[k_id].append(bleu_ndcg_list[k_id])

            for k_id, k in enumerate(retrive_num):
                print('Precision@{}: {}'.format(k, sum(precision[k_id])/len(precision[k_id])))
                print('Recall@{}: {}'.format(k, sum(recall[k_id])/len(recall[k_id])))
                print('NDCG@{}: {}'.format(k, sum(ndcg[k_id])/len(ndcg[k_id])))

    with open('mmd_data/stage2_metric.txt', 'w') as f:
        for k_id, k in enumerate(retrive_num):
            f.write('Precision@{}: {}\n'.format(k, sum(precision[k_id])/len(precision[k_id])))
            f.write('Recall@{}: {}\n'.format(k, sum(recall[k_id])/len(recall[k_id])))
            f.write('NDCG@{}: {}\n'.format(k, sum(ndcg[k_id])/len(ndcg[k_id])))

            f.write('BLEU Precision@{}: {}\n'.format(k, sum(bleu_precision[k_id])/len(bleu_precision[k_id])))
            f.write('BLEU Recall@{}: {}\n'.format(k, sum(bleu_recall[k_id])/len(bleu_recall[k_id])))
            f.write('BLEU NDCG@{}: {}\n'.format(k, sum(bleu_ndcg[k_id])/len(bleu_ndcg[k_id])))

def add_argument():
    parser = argparse.ArgumentParser(description='deepspeed training script')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def main():
    # Dialog data files.
    # common data
    # Check if data exists.
    if not isfile(DatasetConfig.common_raw_data_file):
        raise ValueError('No common data.')
    if not isfile(DatasetConfig.recommend_test_dialog_file):
        raise ValueError('No test dataset.')

    # Load extracted common data.
    image_paths = load_pkl(DatasetConfig.common_raw_data_file).image_paths
    test_data = load_pkl(DatasetConfig.recommend_test_dialog_file)

    processor, replace_token, sep_token = get_pretrained_processor()
    model = stage2_model(processor)

    test_dataset = Dataset(
        image_paths,
        test_data,
        processor,
        replace_token
    )
    recommend_test(
        model,
        test_dataset,
        processor,
        sep_token,
        sp_token = processor.tokenizer.vocab[replace_token[0]]
    )

def convert_checckpoints(save_path, output_path):
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)

if __name__ == '__main__':
    convert = False
    if convert:
        convert_checckpoints(DatasetConfig.dump_dir, DatasetConfig.stage2_model_dir)
    main()