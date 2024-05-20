# For T5 based model
from model.instructblip import InstructBlipConfig, InstructBlipModel, InstructBlipPreTrainedModel,InstructBlipForConditionalGeneration,InstructBlipProcessor
# import datasets
import json
import transformers
from PIL import Image
import torch
model_type="instructblip"
model_ckpt="MMICL-Instructblip-T5-xl"
processor_ckpt = "instructblip-flan-t5-xxl"
config = InstructBlipConfig.from_pretrained(model_ckpt)
from utils import collate, get_pretrained_processor, get_pretrained_model
import argparse
# import deepspeed
#
# def add_argument():
#     parser = argparse.ArgumentParser(description='deepspeed training script')
#     parser.add_argument('--local_rank',
#                         type=int,
#                         default=-1,
#                         help='local rank passed from distributed launcher')
#
#     parser = deepspeed.add_config_arguments(parser)
#     args = parser.parse_args()
#     return args

processor, replace_token, sep_token = get_pretrained_processor()
model = get_pretrained_model(processor)
model.to_peft()

# # Switch to deepspeed mode.
# args = add_argument()
# deepspeed.init_distributed()
# parameters = filter(lambda p: p.requires_grad, model.parameters())
# model, optimizer, _, _ = deepspeed.initialize(
#     args=args, model=model, model_parameters=parameters)
# device = model.local_rank

# device = 'cuda:0'
# model = model.to(device)

model.eval()

image = Image.open ("images/flamingo_photo.png")
image1 = Image.open ("images/flamingo_cartoon.png")
image2 = Image.open ("images/flamingo_3d.png")

images = [image,image1,image2]
prompt = [f'Use the image 0: <image0>{replace_token}, image 1: <image1>{replace_token} and image 2: <image2>{replace_token} as a visual aids to help you answer the question. Question: Give the reason why image 0, image 1 and image 2 are different? Answer:']

prompt = " ".join(prompt)

inputs = processor(images=images, text=prompt, return_tensors="pt")

inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
inputs['img_mask'] = torch.tensor([[1 for i in range(len(images))]])
inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)

outputs = model.generate(
    pixel_values = inputs['pixel_values'].to(device),
    input_ids = inputs['input_ids'].to(device),
    attention_mask = inputs['attention_mask'].to(device),
    img_mask = inputs['img_mask'].to(device),
    do_sample=False,
    max_length=80,
    min_length=50,
    num_beams=8,
    set_min_padding_size =False,
    sp_token= 32100
)
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)

# # example 2
# image = Image.open ("images/chinchilla.png")
# image1 = Image.open ("images/shiba.png")
# image2 = Image.open ("images/flamingo.png")
# images = [image,image1,image2]
# images = [image,image1,image2]
# prompt = [f'image 0 is <image0>{replace_token},image 1 is <image1>{replace_token},image 2 is <image2>{replace_token}. Question: <image0> is a chinchilla. They are mainly found in Chile.\n Question: <image1> is a shiba. They are very popular in Japan.\nQuestion: image 2 is']
#
# prompt = " ".join(prompt)
#
# inputs = processor(images=images, text=prompt, return_tensors="pt")
#
# inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
# inputs['img_mask'] = torch.tensor([[1 for i in range(len(images))]])
# inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
#
# inputs = inputs.to('cuda:0')
# outputs = model.generate(
#     pixel_values = inputs['pixel_values'],
#     input_ids = inputs['input_ids'],
#     attention_mask = inputs['attention_mask'],
#     img_mask = inputs['img_mask'],
#     do_sample=False,
#     max_length=50,
#     min_length=1,
#     set_min_padding_size =False,
# )
# generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
# print(generated_text)


# # example1
# image = Image.open ("images/cal_num1.png")
# image1 = Image.open ("images/cal_num2.png")
# image2 = Image.open ("images/cal_num3.png")
# images = [image,image1,image2]
#
# prompt = [f'Use the image 0: <image0>{replace_token},image 1: <image1>{replace_token} and image 2: <image2>{replace_token} as a visual aid to help you calculate the equation accurately. image 0 is 2+1=3.\nimage 1 is 5+6=11.\nimage 2 is"']
# prompt = " ".join(prompt)
#
# inputs = processor(images=images, text=prompt, return_tensors="pt")
#
# inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
# inputs['img_mask'] = torch.tensor([[1 for i in range(len(images))]])
# inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
#
# inputs = inputs.to('cuda:0')
# outputs = model.generate(
#     pixel_values = inputs['pixel_values'],
#     input_ids = inputs['input_ids'],
#     attention_mask = inputs['attention_mask'],
#     img_mask = inputs['img_mask'],
#     do_sample=False,
#     max_length=50,
#     min_length=1,
#     set_min_padding_size =False,
# )
# generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
# print(generated_text)
# # output: 3x6=18"

# # example 2
# image = Image.open ("images/chinchilla.png")
# image1 = Image.open ("images/shiba.png")
# image2 = Image.open ("images/flamingo.png")
#
# images1 = [image1]
# prompt1 = [f'{replace_token} It depicts a item with following attributes ']
#
# images2 = [image]
# prompt2 = [f'It depicts a item with following attributes <@ITEM_SEP> It depicts a item with following attributes '
#            f'<@ITEM_SEP> It depicts a item with following attributes <@ITEM_SEP> It depicts a item with following attributes']
#
# inputs1 = processor(images=images1, text=prompt1, return_tensors="pt")
# inputs2 = processor(text=prompt2, return_tensors="pt")
# inputs3 = processor(images=image1, return_tensors="pt")
#
# outputs = model(
#     pixel_values = input['pixel_values'],
#     input_ids = input['input_ids'],
#     attention_mask = input['attention_mask'],
#     img_mask = input['img_mask'],
#     labels = label
# )
# lis = []
# ids = inputs2['input_ids'][0]
# pos = torch.where(ids==processor.tokenizer.vocab['<@ITEM_SEP>'])
# start = 0
# for p in pos[0]:
#     p = int(p)
#     lis.append(ids[start:p].tolist())
#     start = p+1
# lis.append(ids[start:].tolist())
# generated_text = processor.batch_decode(lis, skip_special_tokens=True)
# print(generated_text)