import torch
import os
from torchsummary import summary
from torchvision.ops import MLP
from torch import nn
from VisualEncoder.VisualEncoder import VisualEncoder
from VL_mapper.get_VL_mapper import get_VL_mapper
from GL_mapper.get_GL_mapper import get_GL_mapper
from mBART.TranslationModel import TranslationModel
from mBART.get_tokenizer import get_tokenizer
from VideoMAE.VideoEncoder import VideoEncoder
from VideoMAE.MAE_mapper import get_MAE_mapper

class Sign2Text(torch.nn.Module):
    def __init__(self, Sign2Text_cfg, VisualEncoder_cfg):
        super(Sign2Text, self).__init__()

        self.device = Sign2Text_cfg.device
        # self.visual_encoder = VisualEncoder(VisualEncoder_cfg)
        self.video_encoder = VideoEncoder()
        self.language_model = TranslationModel(Sign2Text_cfg)
        # self.VL_mapper = get_VL_mapper(Sign2Text_cfg)
        self.MAE_mapper = get_MAE_mapper(Sign2Text_cfg)

    def get_language_params(self):
        return self.language_model.parameters()

    def get_visual_params(self):
        return list(filter(lambda p: p.requires_grad, self.video_encoder.parameters())) \
        + list(self.MAE_mapper.parameters())

    def get_params(self, CFG):
        return [{'params':self.get_language_params(), 'lr': CFG.init_lr_language_model},
            {'params':self.get_visual_params(), 'lr': CFG.init_lr_visual_model}]

    def forward(self, x, trg, ipt_len):
        x = self.video_encoder(x, ipt_len)
        # x = self.MAE_mapper(x)
        out, loss = self.language_model(x, trg, ipt_len)
        return out, loss

    def predict(self, x, ipt_len, skip_special_tokens = True):
        x = self.video_encoder(x, ipt_len)
        # x = self.MAE_mapper(x)
        preds = self.language_model.generate(
            x, 
            ipt_len,
            skip_special_tokens = skip_special_tokens)
        return preds