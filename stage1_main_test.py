#!/usr/bin/python3
import argparse
import re
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from nltk.translate.bleu_score import sentence_bleu
from config import RecommendTrainConfig, GlobalConfig, DatasetConfig
from torch.utils.data import DataLoader
from utils import collate, get_pretrained_processor, get_pretrained_model
from tqdm import tqdm
from os.path import isfile, join
import torch
import deepspeed
from dataset.stage1_dataset import Dataset
from utils import load_pkl
from collections import namedtuple

CommonData = namedtuple('CommonData',
                        ['image_paths'])

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def recommend_test(
        model,
        dataset,
        processor,
        output_path,
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
    model.to_peft()

    # Load saved state.
    print('loading best model...')
    state = torch.load(output_path)
    # model.load_state_dict({k.replace('language_model.base_model.model', 'language_model'): v for k, v in
    #                        state['state_dict'].items()})
    model.load_state_dict(state['state_dict'])
    print('best model loaded...')

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
        collate_fn = collate
    )
    device = GlobalConfig.device
    print(device)
    model.to(device)

    model.eval()

    count = 0
    correct = 0
    soft_correct = 0
    blue_5_correct = 0
    blue_7_correct = 0
    blue_9_correct = 0

    image_taxonomy = []
    with torch.no_grad():

        for batch_id, train_data in enumerate(tqdm(data_loader)):

            input, label = train_data
            label = label.to(device)

            outputs = model.generate(
                pixel_values = input['pixel_values'].to(device),
                input_ids = input['input_ids'].to(device),
                attention_mask = input['attention_mask'].to(device),
                img_mask = input['img_mask'].to(device),
                do_sample=False,
                max_length=512,
                min_length=1,
                set_min_padding_size =False,
                sp_token= sp_token
            ) # [b, l] end with 1
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
            label[torch.where(label==-100)] = 0
            gt_text = processor.batch_decode(label, skip_special_tokens=True)

            with open('mmd_data/stage1_result.txt', 'a') as f:
                for i in range(len(generated_text)):
                    f.write('label ==> ' + gt_text[i] + '\tpred ==> ' + generated_text[i] + '\n')
                    count += 1
                    image_taxonomy.append(generated_text[i])
                    # exact match
                    if generated_text[i]==gt_text[i]:
                        correct += 1

                    # soft match
                    if generated_text[i].split('/')[-1]==gt_text[i].split('/')[-1]:
                        soft_correct += 1

                    # bleu match
                    blue_format_label = set(re.sub(r"[^\w\s]", " ", gt_text[i]).split())
                    blue_format_pred = set(re.sub(r"[^\w\s]", " ", generated_text[i]).split())
                    blue_sccore = sentence_bleu([list(blue_format_label)], list(blue_format_pred), weights=(1, 0, 0, 0))
                    if blue_sccore > 0.5:
                        blue_5_correct += 1
                    if blue_sccore > 0.7:
                        blue_7_correct += 1
                    if blue_sccore > 0.9:
                        blue_9_correct += 1

            if (count // RecommendTrainConfig.batch_size) % RecommendTrainConfig.print_freq == 0:
                print("acc: ", correct/count)
                print("soft acc: ", soft_correct/count)
                print("bleu 0.5 acc: ", blue_5_correct/count)
                print("bleu 0.7 acc: ", blue_7_correct/count)
                print("bleu 0.9 acc: ", blue_9_correct/count)

    print("acc: ", correct/count)
    print("soft acc: ", soft_correct/count)
    print("bleu 0.5 acc: ", blue_5_correct/count)
    print("bleu 0.7 acc: ", blue_7_correct/count)
    print("bleu 0.9 acc: ", blue_9_correct/count)

    if not isfile(DatasetConfig.new_image_taxonomy_dir):
        save_pkl(image_taxonomy, 'image_taxonomy', DatasetConfig.new_image_taxonomy_dir)

def add_argument():
    parser = argparse.ArgumentParser(description='deepspeed training script')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def main(output_path):
    # Dialog data files.
    # common data
    # Check if data exists.
    if not isfile(DatasetConfig.img_path_dir):
        raise ValueError('No common raw data.')
    if not isfile(DatasetConfig.image_text_dir):
        raise ValueError('No common raw data.')
    if not isfile(DatasetConfig.image_taxonomy_dir):
        raise ValueError('No common raw data.')
    # Load extracted common data.
    image_paths = load_pkl(DatasetConfig.img_path_dir)
    image_text = load_pkl(DatasetConfig.image_text_dir)
    image_taxonomy = load_pkl(DatasetConfig.image_taxonomy_dir)

    processor, replace_token, sep_token = get_pretrained_processor()
    model = get_pretrained_model(processor)

    dataset = Dataset(
        image_paths,
        image_text,
        image_taxonomy,
        processor,
        replace_token
    )
    # model_file = join(DatasetConfig.dump_dir, args['model_file_name'])
    recommend_test(
        model,
        dataset,
        processor,
        output_path,
        sp_token = processor.tokenizer.vocab[replace_token[0]]
    )

def convert_checckpoints(save_path, output_path):
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)

if __name__ == '__main__':
    save_path = "/home/student2020/shy/MIC-master/mmd_data"
    output_path = "/home/student2020/shy/MIC-master/mmd_data/stage1_model.pt"
    if not isfile(output_path):
        convert_checckpoints(save_path, output_path)
    # main(output_path)