#!/usr/bin/python3
from os.path import isfile, join
import torch
import deepspeed
from config import RecommendTrainConfig, GlobalConfig
from config import DatasetConfig
from dataset.stage2_dataset import Dataset
from utils import load_pkl, get_pretrained_processor
from collections import namedtuple
from lib.stage2_train import recommend_train
import argparse
from model.stage2_model import stage2_model

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
    if not isfile(DatasetConfig.common_raw_data_file):
        raise ValueError('No common data.')
    if not isfile(DatasetConfig.recommend_train_dialog_file):
        raise ValueError('No train dataset.')
    if not isfile(DatasetConfig.recommend_valid_dialog_file):
        raise ValueError('No valid dataset.')

    # Load extracted common data.
    image_paths = load_pkl(DatasetConfig.common_raw_data_file).image_paths
    train_data = load_pkl(DatasetConfig.recommend_train_dialog_file)
    valid_data = load_pkl(DatasetConfig.recommend_valid_dialog_file)

    processor, replace_token, sep_token = get_pretrained_processor()
    model = stage2_model(processor)

    train_dataset = Dataset(
        image_paths,
        train_data,
        processor,
        replace_token
    )

    valid_dataset = Dataset(
        image_paths,
        valid_data,
        processor,
        replace_token
    )
    # model_file = join(DatasetConfig.dump_dir, args['model_file_name'])
    recommend_train(
        model,
        train_dataset,
        valid_dataset,
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
    parser.add_argument('--use_deepspeed',
                        type=bool,
                        default=True)
    parser.add_argument('--stage1_device',
                        type=int,
                        default=0,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--stage2_device',
                        type=int,
                        default=1,
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
