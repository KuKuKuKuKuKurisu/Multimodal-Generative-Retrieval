import torch.nn as nn
from utils import get_pretrained_model
import torch
from config import DatasetConfig
import copy
from peft import LoraConfig, get_peft_model, TaskType

class stage2_model(nn.Module):
    def __init__(self, processor):
        super(stage2_model, self).__init__()

        self.stage1_model = get_pretrained_model(processor)

        try:
            # Load saved state.
            print('loading stage1 best model...')
            # model.load_state_dict({k.replace('language_model.base_model.model', 'language_model'): v for k, v in
            #                        state['state_dict'].items()})
            self.stage1_model.load_state_dict(torch.load(DatasetConfig.stage1_model_dir)['state_dict'])
            print('best stage1 model loaded...')
        except:
            pass

    def forward(self, imgs, text, attn_mask, img_mask, label, sp_token):
        outputs = self.stage1_model(
            pixel_values = imgs,
            input_ids = text,
            attention_mask = attn_mask,
            img_mask = img_mask,
            labels = label,
            sp_token= sp_token
        )
        return outputs

    def generate(self, imgs, text, attn_mask, img_mask, sp_token):
        outputs = self.stage1_model.generate(
            pixel_values = imgs,
            input_ids = text,
            attention_mask = attn_mask,
            img_mask = img_mask,
            do_sample=False,
            max_length=512,
            min_length=1,
            set_min_padding_size =False,
            sp_token= sp_token
        )
        return outputs

    # def forward(self, imgs, text, attn_mask, img_mask, label, sp_token):
    #     with torch.no_grad():
    #         inputs_embeds, attention_mask = self.stage1_model(
    #             pixel_values = imgs,
    #             input_ids = text,
    #             attention_mask = attn_mask,
    #             img_mask = img_mask,
    #             labels = label,
    #             sp_token= sp_token,
    #             return_logits_only = True
    #         )
    #
    #     stage2_emb = self.stage2_model(
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=attention_mask,
    #         output_attentions=None,
    #         output_hidden_states=True,
    #         return_dict=True
    #     ).last_hidden_state
    #
    #     outputs = self.stage1_model.stage2_forward(
    #         input_ids = text,
    #         inputs_embeds=stage2_emb,
    #         attention_mask=attention_mask,
    #         decoder_input_ids=None,
    #         decoder_attention_mask=None,
    #         output_attentions=None,
    #         output_hidden_states=None,
    #         return_dict=True,
    #         labels=label
    #     )
    #     return outputs