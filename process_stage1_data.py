# -*- coding: utf-8 -*-
import json
import os
import re
from os.path import isfile, join
from tqdm import tqdm
from utils import save_pkl, load_pkl, get_product_path, get_product_text
from config import DatasetConfig
from collections import namedtuple
from dataset.tidy_data import generate_tidy_data_file
from dataset.model import Utterance

CommonData = namedtuple('CommonData',
                        ['image_paths'])

def pre_train_word_emb(sentence):
    # remove quotation marks and spaces at begin and end
    ret = sentence.lstrip('‘').rstrip('’').strip()
    # lower characters
    ret = ret.lower()
    return ret

def get_images_path():
    """Get images (URL and filenames of local images mapping).

    URL -> Path => URL -> index & index -> Path

    Returns:
        Dict[str, int]: Image URL to index.
        List[str]: Index to the filename of the local image.

    """
    # Get URL to filename mapping dict.
    with open(DatasetConfig.url2img, 'r') as file:
        url_image_pairs = [line.strip().split(' ') for line in file.readlines()]
    url_image_pairs = [(p[0], p[1]) for p in url_image_pairs]
    url2img = dict(url_image_pairs)

    # Divided it into two steps.
    # URL -> Path => URL -> index & index -> Pathtrain
    # Element of index 0 should be empty image.
    image_url_id = {'': 0}
    full_image_paths = ['']

    image_paths = []
    image_text = []
    image_taxonomy = []

    for url, img in tqdm(url2img.items()):
        image_path = join(DatasetConfig.image_data_directory, img)
        product_path = get_product_path(img)
        if isfile(image_path) and isfile(product_path):
            product_text, taxonomy = get_product_text(product_path)
            if taxonomy is not None and taxonomy != '' and taxonomy != ' ':
                image_url_id[url] = len(image_url_id)
                full_image_paths.append(img)
                image_paths.append(img)
                image_text.append(product_text)
                image_taxonomy.append(taxonomy)
    print('img num: ', len(image_paths))
    return image_url_id, full_image_paths, image_paths, image_text, image_taxonomy

def main():

    stage2 = False
    #common data
    splits = ['train', 'valid', 'test']

    with open(DatasetConfig.image_id_file) as fp:
        url_id = json.load(fp)[0]
    id_url = {x:y for y,x in url_id.items()}

    if not isfile(DatasetConfig.image_url_id_dir):
        image_url_id, full_image_paths, image_paths, image_text, image_taxonomy = get_images_path()
        save_pkl(image_url_id, 'image_url_id', DatasetConfig.image_url_id_dir)
    else:
        image_url_id = load_pkl(DatasetConfig.image_url_id_dir)

    if not isfile(DatasetConfig.common_raw_data_file):
        common_data = CommonData(image_paths=full_image_paths)
        print('saving common_data...')
        save_pkl(common_data, 'common_data',
                 DatasetConfig.common_raw_data_file)
    else:
        full_image_paths = load_pkl(DatasetConfig.common_raw_data_file).image_paths

    if not isfile(DatasetConfig.img_path_dir):
        save_pkl(image_paths, 'image_paths', DatasetConfig.img_path_dir)
    if not isfile(DatasetConfig.image_text_dir):
        save_pkl(image_text, 'image_text', DatasetConfig.image_text_dir)
    if not isfile(DatasetConfig.image_taxonomy_dir):
        save_pkl(image_taxonomy, 'image_taxonomy', DatasetConfig.image_taxonomy_dir)

    if stage2:
        image_taxonomy = load_pkl(DatasetConfig.new_image_taxonomy_dir)

        for split in splits:

            if split == 'train':
                input_path = DatasetConfig.train_dialog_data_directory
                raw_output_path = DatasetConfig.train_raw_data_file
                tidy_output_path = DatasetConfig.recommend_train_dialog_file
            elif split == 'valid':
                input_path = DatasetConfig.valid_dialog_data_directory
                raw_output_path = DatasetConfig.valid_raw_data_file
                tidy_output_path = DatasetConfig.recommend_valid_dialog_file
            else:
                input_path = DatasetConfig.test_dialog_data_directory
                raw_output_path = DatasetConfig.test_raw_data_file
                tidy_output_path = DatasetConfig.recommend_test_dialog_file

            has_raw_data_pkl = isfile(raw_output_path)

            if not has_raw_data_pkl:

                dialogs = []
                for file_id, file in enumerate(tqdm(os.listdir(input_path))):
                    with open(os.path.join(input_path, file), 'r') as f:
                        data = json.load(f)
                    f.close()
                    dialog = []
                    for utterance in data:
                        # get utter attributes
                        speaker = utterance.get('speaker') #保存type
                        utter_type = f"{utterance.get('type')}"

                        utter = utterance.get('utterance')
                        text = utter.get('nlg')
                        images = utter.get('images')
                        false_images = utter.get('false images')

                        # some attributes may be empty
                        if text is None:
                            text = ""
                        if images is None:
                            images = []
                        if false_images is None:
                            false_images = []
                        if utter_type is None:
                            utter_type = ""

                        # Images
                        new_pos_images = [id_url[x] for x in images]
                        new_neg_images = [id_url[x] for x in false_images]

                        pos_images = []
                        for img in new_pos_images:
                            try:
                                pos_images.append(image_url_id[img])
                            except:
                                pass

                        neg_images = []
                        for img in new_neg_images:
                            try:
                                neg_images.append(image_url_id[img])
                            except:
                                pass

                        dialog.append(Utterance(speaker, utter_type, pre_train_word_emb(text), pos_images, neg_images, None))

                    dialogs.append(dialog)

                # Save common data to a .pkl file.
                save_pkl(dialogs, 'raw_{}_dialogs'.format(split), raw_output_path)
            else:
                dialogs = load_pkl(raw_output_path)

            if not isfile(tidy_output_path):
                generate_tidy_data_file(dialogs, full_image_paths, tidy_output_path, image_taxonomy)
            else:
                pass

if __name__ == '__main__':
    main()