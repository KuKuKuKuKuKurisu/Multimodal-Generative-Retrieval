"""Utilities."""

from model.instructblip import InstructBlipConfig, InstructBlipModel, InstructBlipPreTrainedModel,InstructBlipForConditionalGeneration,InstructBlipProcessor
import pickle as pkl
import re
from os.path import splitext, join
from typing import *
from copy import deepcopy
import torch
from config import DatasetConfig, constants
import json

def save_pkl(obj, obj_name, pkl_file):
    """Save object to a .pkl file.

    Args:
        obj (object): Object.
        obj_name (str): Object name.
        pkl_file (str): Pickle file name.

    Returns:
        None

    """
    print('Saving {} to {}...'.format(obj_name, pkl_file))
    with open(pkl_file, 'wb') as file:
        pkl.dump(obj, file)
    print('Saved.')


def load_pkl(pkl_file):
    """Load object from a .pkl file.

    Args:
        pkl_file (str): Pickle file name.

    Returns:
        obj (object): Object.

    """
    print('Loading {} ...'.format(pkl_file))
    with open(pkl_file, 'rb') as file:
        obj = pkl.load(file)
    print('Loaded.')
    return obj

def pad_or_clip_images(images: List[int],
                       max_len: int) -> Tuple[List[int], int]:
    """Pad or clip images.

    Args:
        images (List[int]): Image id list.
        max_len (int): Maximum length.

    Returns:
        Tuple[List[int], int]: Padded image list of length max_len and its
        actual length.

    """

    images = deepcopy(images)
    if len(images) > max_len:
        images = images[:max_len]
    num_images = len(images)
    pads = max_len - len(images)
    images += [0] * pads

    return images, num_images


def get_embed_init(glove: List[List[float]], vocab_size: int):
    """Get initial embedding.

    Args:
        glove (List[List[float]]): GloVe.
        vocab_size (int): Vocabulary size.

    Returns:
        Initial embedding (vocab_size, embed_size).

    """
    embed = [None] * vocab_size
    for idx in range(vocab_size):
        vec = glove[idx]
        if vec is None:
            vec = torch.zeros(300)
            if idx != PAD_ID:
                vec.uniform_(-0.25, 0.25)
        else:
            vec = torch.tensor(vec)
        embed[idx] = vec
    embed = torch.stack(embed)
    return embed

def get_product_path(image_name: str):
    """Get product path from a given image name.

    Args:
        image_name (str): Image name.

    Returns:
        str: Corresponding product file name.

    """
    product_path = join(DatasetConfig.product_data_directory,
                        splitext(image_name)[0] + '.json')
    return product_path

def get_product_text(product_path):
    ignore_key_list = ['details', 'available_sizes', 'bestSellerRank', 'review', 'care', 'avgStars', 'reviewStars', 'taxonomy']

    product_dict = json.load(open(product_path))

    # taxonomy
    taxonomy = product_dict['taxonomy']
    taxonomy = '/'.join(taxonomy.split('/')[-2:]).lower()
    # taxonomy = taxonomy.split('/')[-1].lower().replace(' ', '/')
    # taxonomy_ = re.sub(r"[^\w\s]", " ", taxonomy).split()

    # product text
    texts = []
    for key, value in product_dict.items():
        # Note: Only a space is also empty.
        if key in ignore_key_list:
            continue
        if value is not None and value != '' and value != ' ':
            if value.lower() != 'unk':
                # ################################
                # # ablation preprocess
                # value_ = re.sub(r"[^\w\s]", " ", value.lower()).split()
                # new_value = []
                # for v in value_:
                #     if v not in taxonomy_:
                #         new_value.append(v)
                # if len(new_value)>0:
                #     texts.append(key + ' is ' + ' '.join(new_value))
                # ################################

                texts.append(key + ' is ' + value)

    product_texts = ','.join(texts).replace('/', ' ').lstrip('‘').rstrip('’').strip().lower()
    product_texts = re.sub(r"[ \( \) \[ \] \< \> = : ;]", " ", product_texts)

    return product_texts.lower(), taxonomy

def get_pretrained_model(processor):
    # model
    model_type="instructblip"
    model_ckpt="MMICL-Instructblip-T5-xl"
    config = InstructBlipConfig.from_pretrained(model_ckpt)

    if 'instructblip' in model_type:
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_ckpt,
            config=config).to(dtype=torch.bfloat16)

    if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(processor.qformer_tokenizer):
        model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))
    return model

def get_pretrained_processor():
    processor_ckpt = "instructblip-flan-t5-xxl"
    image_placeholder="图"
    sep_token = '<@ITEM_SEP>'
    sp = [image_placeholder]+[f"<image{i}>" for i in range(20)] + [sep_token]
    processor = InstructBlipProcessor.from_pretrained(
        processor_ckpt
    )
    sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
    processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
    replace_token="".join(32*[image_placeholder])
    return processor, replace_token, sep_token

def mask_nll_loss(inp, target, mask):
    # print('inp = {}'.format(inp))
    # print('target = {}'.format(target))
    # print('mask = {}'.format(mask))
    n_total = mask.sum()
    # print('n_total = {}'.format(n_total))
    if n_total.item() == 0:
        return 0, 0
    cross_entropy = -torch.log(
        torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask.byte()).mean()
    loss = loss.to(GlobalConfig.device)
    return loss, n_total.item()

def process_txt(txt):
    txt = re.sub(r"[ \( \) \[ \] \< \> , = : ;]", " ", txt)
    return txt.lower().strip()

def get_mask(length: int, target_length):
    """Get mask.

    Args:
        length (int): Length.
        target_length: Target length (batch_size, ).

    Returns:
        mask: Mask (batch_size, length).

    """
    return torch.arange(length, device=GlobalConfig.device).expand(
        target_length.size(0), length) < target_length.unsqueeze(1)


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """

    Source: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    ``torch.nn.functional.softmax(vector)`` does not work if some elements
    of ``vector`` should be masked.  This performs a softmax on just the
    non-masked portions of ``vector``.  Passing ``None`` in for the mask is
    also acceptable; you'll just get a regular softmax. ``vector`` can have
    an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions
    than ``vector``, we will unsqueeze on dimension 1 until they match.  If
    you need a different unsqueezing of your mask, do it yourself before
    passing the mask into this function. If ``memory_efficient`` is set to
    true, we will simply use a very large negative number for those masked
    positions so that the probabilities of those positions would be
    approximately 0. This is not accurate in math, but works for most cases
    and consumes less memory. In the case that the input vector is completely
    masked and ``memory_efficient`` is false, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last
    layer of a model that uses categorical cross-entropy loss. Instead,
    if ``memory_efficient`` is true, this function will treat every element
    as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the
            # mask, we zero these out.
            result = torch.nn.functional.softmax(torch.mul(vector, mask),
                                                 dim=dim)
            result = torch.mul(result, mask)
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(),
                                               mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def to_str(o):
    if isinstance(o, dict):
        res = []
        for key, val in o.items():
            res.append(key)
            res.append(to_str(val))
        return ' '.join(res)
    elif isinstance(o, list):
        return ' '.join([to_str(x) for x in o])
    elif isinstance(o, str):
        return o
    return str(o)

# train collate
def collate(batch):
    max_qformer_len = 0
    max_text_len = 0
    max_label_len = 0
    over_all_maxlen = DatasetConfig.text_max_len
    for data in batch:
        input, label = data
        max_qformer_len = max(input['qformer_input_ids'].shape[-1], max_qformer_len)
        max_text_len = max(input['input_ids'].shape[-1], max_text_len)
        max_label_len = max(label.shape[-1], max_label_len)

    final_input = {
        'img_mask':[]
    }
    final_label = []
    for k in batch[0][0].keys():
        final_input[k] = []

    if max_qformer_len>over_all_maxlen:
        max_qformer_len = over_all_maxlen
    if max_text_len>over_all_maxlen:
        max_text_len = over_all_maxlen
    if max_label_len>over_all_maxlen:
        max_label_len = over_all_maxlen

    for data in batch:
        input, label = data
        if label.shape[-1] > max_label_len:
            label = torch.cat(label[:, :max_label_len], dim=-1).to(torch.int64)
            label[:, :max_label_len-1] = 1
        else:
            label = torch.cat([label, -100*torch.ones(1, max_label_len-label.shape[-1]).long()], dim=-1).to(torch.int64)

        input['pixel_values'] = input['pixel_values'].to(torch.bfloat16)
        input['pixel_values'] = input['pixel_values'].unsqueeze(0)
        input['img_mask'] = torch.tensor([[1 for i in range(input['pixel_values'].shape[1])]])

        if input['input_ids'].shape[1] > max_text_len:
            input['input_ids'] = input['input_ids'][:, :max_text_len].to(torch.int64)
            input['attention_mask'] = input['attention_mask'][:, :max_text_len].to(torch.int64)
            input['input_ids'][:, max_text_len-1] = 1
            input['attention_mask'][:, max_text_len-1] = 1
        else:
            input['input_ids'] = torch.cat([input['input_ids'], torch.zeros(1, max_text_len-input['input_ids'].shape[1]).long()], dim = -1).to(torch.int64)
            input['attention_mask'] = torch.cat([input['attention_mask'], torch.zeros(1, max_text_len-input['attention_mask'].shape[1]).long()], dim = -1).to(torch.int64)

        if input['qformer_input_ids'].shape[1] > max_qformer_len:
            input['qformer_input_ids'] = input['qformer_input_ids'][:, :max_qformer_len].to(torch.int64)
            input['qformer_attention_mask'] = input['qformer_attention_mask'][:, :max_qformer_len].to(torch.int64)
            input['qformer_input_ids'][:, max_qformer_len-1] = 1
            input['qformer_attention_mask'][:, max_qformer_len-1] = 1
        else:
            input['qformer_input_ids'] = torch.cat([input['qformer_input_ids'], torch.zeros(1, max_qformer_len-input['qformer_input_ids'].shape[1]).long()], dim = -1).to(torch.int64)
            input['qformer_attention_mask'] = torch.cat([input['qformer_attention_mask'], torch.zeros(1, max_qformer_len-input['qformer_attention_mask'].shape[1]).long()], dim = -1).to(torch.int64)

        final_label.append(label)
        final_input['pixel_values'].append(input['pixel_values'])
        final_input['img_mask'].append(input['img_mask'])
        final_input['input_ids'].append(input['input_ids'])
        final_input['qformer_input_ids'].append(input['qformer_input_ids'])
        final_input['attention_mask'].append(input['attention_mask'])
        final_input['qformer_attention_mask'].append(input['qformer_attention_mask'])

    for k,v in final_input.items():
        final_input[k] = torch.cat(v,dim=0)
    final_label = torch.cat(final_label,dim=0)
    return final_input, final_label


def stage2_collate(batch):
    max_text_len = 0
    max_label_len = 0
    max_img_len = 1

    final_text = []
    final_attn_mask = []
    final_imgs = []
    final_img_mask = []
    final_label = []

    for data in batch:
        text, images, label = data
        max_text_len = max(text.shape[-1], max_text_len)
        max_label_len = max(label.shape[-1], max_label_len)
        max_img_len = max(len(images), max_img_len)

    for data in batch:
        text, images, label = data

        text = text.unsqueeze(0)
        text_attn_mask = torch.cat([torch.ones(1, text.shape[-1]).long(), torch.zeros(1, max_text_len-text.shape[-1]).long()], dim=-1).to(torch.int64)
        text = torch.cat([text, torch.zeros(1, max_text_len-text.shape[-1]).long()], dim=-1).to(torch.int64)

        img_mask = torch.tensor([[1 for i in range(len(images))] + [0 for i in range(max_img_len - len(images))]])
        images = images + [torch.zeros(1,3,224,224) for i in range(max_img_len-len(images))]
        images = torch.cat(images, 0).unsqueeze(0).to(torch.bfloat16)

        label = label.unsqueeze(0)
        label = torch.cat([label, -100*torch.ones(1, max_label_len-label.shape[-1]).long()], dim=-1).to(torch.int64)

        final_text.append(text)
        final_attn_mask.append(text_attn_mask)
        final_imgs.append(images)
        final_img_mask.append(img_mask)
        final_label.append(label)

    return torch.cat(final_text, 0), torch.cat(final_attn_mask, 0), torch.cat(final_imgs, 0), torch.cat(final_img_mask, 0), torch.cat(final_label, 0)