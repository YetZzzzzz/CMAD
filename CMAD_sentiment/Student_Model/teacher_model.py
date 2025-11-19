import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torch.nn import L1Loss, MSELoss
from torch.autograd import Function
from math import pi, log
from functools import wraps
from torch import nn, einsum
import torch.nn.functional as F
import os
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.activations import gelu, gelu_new
from transformers import BertConfig
import numpy as np
import torch.optim as optim
from itertools import chain
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from modules.transformer import TransformerEncoder
from Perceiver import Perceiver


logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-base-cased",
]


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))
BertLayerNorm = torch.nn.LayerNorm

ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "gelu_new": gelu_new,
    "mish": mish,
}
   
    
class BERT_TM(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.config.output_hidden_states=True
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.d_l = args.d_l
        self.proj_l = nn.Conv1d(args.TEXT_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
       
        last_sequence_output = encoder_outputs[0]# 36*60*768:bsz*msl*dim
        outputs = last_sequence_output.transpose(1,2)# bsz*dim*msl
        outputs_t = self.proj_l(outputs)# 
        
        return outputs_t


class TeacherModel(BertPreTrainedModel):
    def __init__(self, config, args = None):
        super().__init__(config)
        self.num_labels = 1#config.num_labels# here is 1
        self.d_l = args.d_l
        self.proj_a = nn.Conv1d(args.ACOUSTIC_DIM, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(args.VISUAL_DIM, self.d_l, kernel_size=1, padding=0, bias=False)
        self.activation = nn.ReLU()
        self.dropout = args.dropout_prob
        self.num_heads = args.num_heads
        self.te_layers = args.te_layers
        self.num_latents = args.num_latents
        self.latent_dim = args.d_l
        self.input_dim = args.d_l #modify
        self.cross_heads = args.num_heads
        self.depth = args.depth
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_dropout = args.attn_dropout
        
        self.p_attn_dropout = args.p_attn_dropout
        self.p_ff_dropout = args.p_ff_dropout
        self.latent_heads = args.num_heads
        self.latent_dim_head = args.num_heads
        self.cross_dim_head = args.num_heads
        self.bert = BERT_TM(config, args)
        
        self.perceiver_l = Perceiver(self.num_latents, self.latent_dim, self.input_dim, self.depth, self.cross_heads, self.cross_dim_head, self.latent_heads, self.latent_dim_head, self.p_attn_dropout, self.p_ff_dropout)
        self.perceiver_a = Perceiver(self.num_latents, self.latent_dim, self.input_dim, self.depth, self.cross_heads, self.cross_dim_head, self.latent_heads, self.latent_dim_head, self.p_attn_dropout, self.p_ff_dropout)
        self.perceiver_v = Perceiver(self.num_latents, self.latent_dim, self.input_dim, self.depth, self.cross_heads, self.cross_dim_head, self.latent_heads, self.latent_dim_head, self.p_attn_dropout, self.p_ff_dropout)
        
        encoder_layer_all = nn.TransformerEncoderLayer(d_model=self.d_l, nhead=self.num_heads)# modify
        self.transformer_encoder_all = nn.TransformerEncoder(encoder_layer_all, num_layers=self.te_layers) # num_layers
        
        self.fusion_all = nn.Sequential()
        self.fusion_all.add_module('fusion_layer_all', nn.Linear(in_features=self.d_l*3, out_features=self.d_l*1))
        self.fusion_all.add_module('fusion_layer_all_dropout', nn.Dropout(self.dropout))
        self.fusion_all.add_module('fusion_layer_all_activation', self.activation)
        
        self.classifier =  nn.Linear(in_features=self.d_l*1, out_features= self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        label_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        x_l = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        x_a = acoustic.transpose(1, 2)
        x_v = visual.transpose(1, 2)

        proj_x_a = self.proj_a(x_a)
        proj_x_v = self.proj_v(x_v)

        coarse_l = x_l.permute(0, 2, 1)
        coarse_a = proj_x_a.permute(0, 2, 1)
        coarse_v = proj_x_v.permute(0, 2, 1)

        fine_l = self.perceiver_l(coarse_l)
        fine_a = self.perceiver_a(coarse_a)
        fine_v = self.perceiver_v(coarse_v)
        hidden = torch.stack((fine_l, fine_a, fine_v), dim=0)
        hidden = self.transformer_encoder_all(hidden)
        hidden = torch.cat((hidden[0], hidden[1], hidden[2]), dim=1)
        fusion = self.fusion_all(hidden)
        logits = self.classifier(fusion)

        return logits, fusion


    def test(self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,):

        x_l = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,)
        
        x_a = acoustic.transpose(1, 2)
        x_v = visual.transpose(1, 2)

        proj_x_a = self.proj_a(x_a)
        proj_x_v = self.proj_v(x_v)

        coarse_l = x_l.permute(0, 2, 1)
        coarse_a = proj_x_a.permute(0, 2, 1)
        coarse_v = proj_x_v.permute(0, 2, 1)

        fine_l = self.perceiver_l(coarse_l)
        fine_a = self.perceiver_a(coarse_a)
        fine_v = self.perceiver_v(coarse_v)
        hidden = torch.stack((fine_l, fine_a, fine_v), dim=0)
        hidden = self.transformer_encoder_all(hidden)
        hidden = torch.cat((hidden[0], hidden[1], hidden[2]), dim=1)
        fusion = self.fusion_all(hidden)
        logits = self.classifier(fusion)
        
       
        return logits, fusion
  
