"""Tidy data module."""
import copy
from os.path import isfile, join

from config import RecommendTrainConfig
if RecommendTrainConfig.simmc2:
    from config.simmc_dataset_config import SimmcDatasetConfig as DatasetConfig
else:
    from config import DatasetConfig
from tqdm import tqdm
from utils import save_pkl, get_product_path, get_pretrained_processor, get_product_text
from dataset.model import TidyUtterance, Utterance, FormatUtterance

from transformers import BertTokenizer
from config.constants import *

import json

def generate_tidy_data_file(raw_data ,image_paths, output_path, image_taxonomy):
    """Generate tidy data file.

    Args:
        raw_data (RawData): Raw data.
        mode (str): A single mode.

    """

    # Get raw data dialogs according to its mode.
    dialogs = raw_data
    assert dialogs is not None

    processor, replace_token, sep_token = get_pretrained_processor()

    # dialog = get_recommend_task_items(raw_data.image_paths, dialogs[0])

    tidy_dialogs = []
    valid_pos_num = []
    valid_neg_num = []
    for item_idx, dialog in enumerate(tqdm(dialogs)):
        # Get items according to different TASKS.
        if dialog == []:
            continue
        cleaned_dialogs, pos_img_num, neg_img_num = get_recommend_task_items(image_paths, image_taxonomy, dialog, processor, replace_token, sep_token)
        tidy_dialogs.extend(cleaned_dialogs)
        valid_pos_num.extend(pos_img_num)
        valid_neg_num.extend(neg_img_num)

    # Save as pickle file.
    save_pkl(tidy_dialogs, 'tidy_dialogs', output_path)

    print('valid pos num: ', sum(valid_pos_num)/len(valid_pos_num))
    print('valid neg num: ', sum(valid_neg_num)/len(valid_neg_num))

def format_utterance(utterances, image_paths, image_taxonomy, processor, replace_token, sep_token):
    # label
    taxonomy_images = utterances[-1].pos_images
    taxonomys = []
    for img in taxonomy_images:
        product_path = get_product_path(image_paths[img])
        taxonomy = image_taxonomy[img]
        product_text, _ = get_product_text(product_path)
        if taxonomy not in taxonomys:
            taxonomys.append(taxonomy)

    sep_token = ' ' + sep_token + ' '
    taxonomy_id = sep_token.join(taxonomys)
    taxonomy_id = processor(text=[taxonomy_id], return_tensors="pt")['input_ids']
    reduce_num = 0
    while taxonomy_id.shape[-1]>DatasetConfig.text_max_len:
        reduce_num += 1
        taxonomy_id = sep_token.join(taxonomys[:-reduce_num])
        taxonomy_id = processor(text=[taxonomy_id], return_tensors="pt")['input_ids']

    label = taxonomy_id[0].tolist()

    # dialogue history
    dialogue_history = ''
    dialogue_images = []
    utterances = utterances[:-1]
    text_id = []
    n = len(utterances)
    for i in range(n):
        # pure text
        text = utterances[n-i-1].speaker + ': ' + utterances[n-i-1].text
        if text[-1].isalpha():
            text = text + '.'

        # utte image
        image_prompt = []
        utte_images = []
        if i < DatasetConfig.with_img_dialog_context_size:
            for j in range(len(utterances[n-i-1].pos_images)):
                image_prompt.append(f'<image{j}>{replace_token}')
                utte_images.append(utterances[n-i-1].pos_images[j])

        if len(image_prompt)!=0:
            if utterances[n-i-1].speaker == 'system':
                text = text + 'Recommend : ' + ', '.join(image_prompt) + '.'
            else:
                text = text + 'Desire : ' + ', '.join(image_prompt) + '.'

        # dialogue history and imgs
        dialogue_history = text + dialogue_history
        output_text = f'The dialogue history is "{dialogue_history}" Question: recommend items by generating taxonomy'
        output_text_id = processor(text=[output_text], return_tensors="pt")['input_ids']
        if len(text_id)==0:
            text_id = output_text_id[0].tolist()
            utte_images.extend(dialogue_images)
            dialogue_images = utte_images
        else:
            if output_text_id.shape[-1]>DatasetConfig.text_max_len:
                break
            else:
                text_id = output_text_id[0].tolist()
                utte_images.extend(dialogue_images)
                dialogue_images = utte_images

    return text_id, dialogue_images, label

def get_recommend_task_items(image_paths, image_taxonomy, dialog, processor, replace_token, sep_token):
    """Get items for recommend task from a single dialog.

    Args:
        image_paths (List[str]): Image paths.
        dialog (Dialog): Dialog.

    Returns:
        List[TidyDialog]: Extracted tidy dialogs.

    """

    dialogs = []
    utterances = []
    context_size = DatasetConfig.dialog_context_size

    pos_img_num = []
    neg_img_num = []

    # Merge
    tmp_dialogs = []
    tmp_utters = []
    for utter in dialog:
        if utter.text is None:
            utter.text = []
        if utter.pos_images is None:
            utter.pos_images = []
        if utter.neg_images is None:
            utter.neg_images = []

        if len(tmp_utters) == 0:
            tmp_utters.append(Utterance(utter.speaker, utter.utter_type, utter.text, utter.pos_images, utter.neg_images, utter.dst))
            continue
        if utter.speaker != tmp_utters[-1].speaker:
            tmp_dialogs.append(tmp_utters[-1])
            tmp_utters.append(Utterance(utter.speaker, utter.utter_type, utter.text, utter.pos_images, utter.neg_images, utter.dst))
        else:
            last_utter = tmp_utters[-1]
            last_utter.text = last_utter.text + ' ' + utter.text
            last_utter.pos_images.extend(utter.pos_images)
            last_utter.neg_images.extend(utter.neg_images)
            last_utter.utter_type = utter.utter_type
            last_utter.dst = utter.dst
            tmp_utters[-1] = Utterance(last_utter.speaker, last_utter.utter_type, last_utter.text, last_utter.pos_images, last_utter.neg_images, last_utter.dst)
    tmp_dialogs.append(tmp_utters[-1])

    for utter in tmp_dialogs:
        utterances.append(TidyUtterance(utter))

        if utter.speaker == 'system' and utter.pos_images and utter.neg_images:
            pos_img_num.append(len(utter.pos_images))
            neg_img_num.append(len(utter.neg_images))

            utterances = utterances[-(context_size + 1):]
            text, dialogue_images, label = format_utterance(utterances, image_paths, image_taxonomy, processor, replace_token, sep_token)
            dialogs.append(FormatUtterance(text, dialogue_images, label))

    return dialogs, pos_img_num, neg_img_num

