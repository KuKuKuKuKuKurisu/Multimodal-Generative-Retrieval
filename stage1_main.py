#!/usr/bin/python3
from os.path import isfile, join
import torch
import deepspeed
from config import RecommendTrainConfig, GlobalConfig
from config import DatasetConfig
from dataset.stage1_dataset import Dataset
from utils import load_pkl, get_pretrained_processor, get_pretrained_model
from collections import namedtuple
from lib.stage1_train import recommend_train
import argparse

CommonData = namedtuple('CommonData',
                        ['image_paths'])

def train(args):
    """Train model.

    Args:
        task (int): Task.
        model_file_name (str): Model file name (saved or to be saved).

    """

    # Dialog data files.
    # common data
    # Check if data exists.
    if not isfile(DatasetConfig.img_path_dir):
        raise ValueError('No img path data.')
    if not isfile(DatasetConfig.image_text_dir):
        raise ValueError('No img text data.')
    if not isfile(DatasetConfig.image_taxonomy_dir):
        raise ValueError('No img text taxonomy.')
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
    recommend_train(
        model,
        dataset,
        args,
        sp_token = processor.tokenizer.vocab[replace_token[0]]
    )

def parse_cmd():
    """Parse commandline parameters.

    Returns:
        Dict[str, List[str]]: Parse result.

    """

    # Definition of argument parser.
    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument('model_file_name', default='mic.pth', metavar='<model_file_name>')
    parser.add_argument('device', default='cuda:0', metavar='<device>')

    # Namespace -> Dict
    parse_res = vars(parser.parse_args())
    return parse_res


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
    # Parse commandline parameters and standardize.
    args = add_argument()
    args_dict = {
        'device':'cuda:0',
        'model_file_name':'mic.pth'
    }
    GlobalConfig.device = torch.device(args_dict['device'] if torch.cuda.is_available() else "cpu")
    train(args)

if __name__ == '__main__':
    main()
