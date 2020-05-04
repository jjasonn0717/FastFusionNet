# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa
#
# Modified by Felix Wu: adding RCModelProto, CnnDocReader, FusionNet, BiDAF

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers
from typing import IO, List, Iterable, Tuple
from qa.encoder import *
from qa.rnn_reader import RCModelProto


class FusionNet(RCModelProto):
    """Network for FusionNet."""

    def __init__(self, opt, padding_idx=0, embedding=None):
        super().__init__(opt, padding_idx, embedding)

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=self.doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=2,
            dropout_rate=opt['dropout_rnn'],
            # dropout_output=opt['dropout_rnn_output'],
            variational_dropout=opt['variational_dropout'],
            concat_layers=True,
            rnn_type=opt['rnn_type'],
            padding=opt['rnn_padding'],
            residual=opt['residual'],
            squeeze_excitation=opt['squeeze_excitation'],
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=self.question_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=2,
            dropout_rate=opt['dropout_rnn'],
            # dropout_output=opt['dropout_rnn_output'],
            variational_dropout=opt['variational_dropout'],
            concat_layers=True,
            rnn_type=opt['rnn_type'],
            padding=opt['rnn_padding'],
            residual=opt['residual'],
            squeeze_excitation=opt['squeeze_excitation'],
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * 2 * opt['hidden_size']
        question_hidden_size = doc_hidden_size

        self.question_urnn = layers.StackedBRNN(
            input_size=question_hidden_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['fusion_understanding_layers'],
            dropout_rate=opt['dropout_rnn'],
            variational_dropout=opt['variational_dropout'],
            rnn_type=opt['rnn_type'],
            padding=opt['rnn_padding'],
            residual=opt['residual'],
            squeeze_excitation=opt['squeeze_excitation'],
            concat_layers=False,
        )

        self.multi_level_fusion = layers.FullAttention(
            full_size=self.paired_input_size + doc_hidden_size,
            hidden_size=2 * 3 * opt['hidden_size'],
            num_level=3,
            dropout=opt['dropout_rnn'],
            variational_dropout=opt['variational_dropout'],
        )

        self.doc_urnn = layers.StackedBRNN(
            input_size=2 * 5 * opt['hidden_size'],
            hidden_size=opt['hidden_size'],
            num_layers=opt['fusion_understanding_layers'],
            dropout_rate=opt['dropout_rnn'],
            variational_dropout=opt['variational_dropout'],
            rnn_type=opt['rnn_type'],
            padding=opt['rnn_padding'],
            residual=opt['residual'],
            squeeze_excitation=opt['squeeze_excitation'],
            concat_layers=False,
        )


        self.self_boost_fusions = nn.ModuleList()
        self.doc_final_rnns = nn.ModuleList()
        full_size=self.paired_input_size + 4 * 3 * opt['hidden_size']
        for i in range(self.opt['fusion_self_boost_times']):
            self.self_boost_fusions.append(layers.FullAttention(
                full_size=full_size,
                hidden_size=2 * opt['hidden_size'],
                num_level=1,
                dropout=opt['dropout_rnn'],
                variational_dropout=opt['variational_dropout'],
            ))

            self.doc_final_rnns.append(layers.StackedBRNN(
                input_size=4 * opt['hidden_size'],
                hidden_size=opt['hidden_size'],
                num_layers=opt['fusion_final_layers'],
                dropout_rate=opt['dropout_rnn'],
                variational_dropout=opt['variational_dropout'],
                rnn_type=opt['rnn_type'],
                padding=opt['rnn_padding'],
                residual=opt['residual'],
                squeeze_excitation=opt['squeeze_excitation'],
                concat_layers=False,
            ))
            full_size += 2 * opt['hidden_size']

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.quesiton_merge_attns = nn.ModuleList()

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(2 * opt['hidden_size'])

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            2 * opt['hidden_size'],
            2 * opt['hidden_size'],
        )

        if opt['end_gru']:
            self.end_gru = nn.GRUCell(2 * opt['hidden_size'], 2 * opt['hidden_size'])

        self.end_attn = layers.BilinearSeqAttn(
            2 * opt['hidden_size'],
            2 * opt['hidden_size'],
        )

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, sent_lens, x1_char=None, x2_char=None, logit=False):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """

        def dropout(x, p=self.opt['dropout_rnn']):
            return layers.dropout(x, p=p,
                                  training=self.training, variational=self.opt['variational_dropout'] and x.dim() == 3)

        # Embed both document and question
        x1_paired_emb, x2_paired_emb, x1_full_emb, x2_full_emb, feat_dict = self.forward_emb(x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, sent_lens, x1_char, x2_char)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(x1_full_emb, x1_mask)
        # Encode question with RNN
        question_hiddens = self.question_rnn(x2_full_emb, x2_mask)

        # Question Understanding
        question_u_hiddens = self.question_urnn(question_hiddens, x2_mask)

        # Fully-Aware Multi-level Fusion
        doc_HoW = torch.cat([x1_paired_emb, doc_hiddens], 2)
        question_HoW = torch.cat([x2_paired_emb, question_hiddens], 2)
        question_cat_hiddens = torch.cat([question_hiddens, question_u_hiddens], 2)
        doc_fusions = self.multi_level_fusion(doc_HoW, question_HoW, question_cat_hiddens, x2_mask)

        # Document Understanding
        doc_u_hiddens = self.doc_urnn(torch.cat([doc_hiddens, doc_fusions], 2), x1_mask)

        # Fully-Aware Self-Boosted Fusion
        self_boost_HoW = torch.cat([x1_paired_emb, doc_hiddens, doc_fusions, doc_u_hiddens], 2)

        for i in range(len(self.self_boost_fusions)):
            doc_self_fusions = self.self_boost_fusions[i](self_boost_HoW, self_boost_HoW, doc_u_hiddens, x1_mask)
            
            # Final document representation
            doc_final_hiddens = self.doc_final_rnns[i](torch.cat([doc_u_hiddens, doc_self_fusions], 2), x1_mask)
            if i < len(self.self_boost_fusions) - 1:
                self_boost_HoW = torch.cat([self_boost_HoW, doc_final_hiddens], 2)
                doc_u_hiddens = doc_final_hiddens

        # Encode question with RNN + merge hidden, 2s
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_u_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(dropout(question_u_hiddens), x2_mask)
        question_u_hidden = layers.weighted_avg(question_u_hiddens, q_merge_weights)

        # Predict start and end positions
        start_logits = self.start_attn(dropout(doc_final_hiddens), dropout(question_u_hidden), x1_mask, logit=True)
        if self.opt['sentence_level']:
            start_logits = layers.combine_sentences(start_logits, sent_lens)

        start_scores = F.log_softmax(start_logits, 1) if self.training else F.softmax(start_logits, 1)
        if self.opt['end_gru']:
            weights = start_scores.exp() if self.training else start_scores
            weighted_doc_hidden = layers.weighted_avg(doc_final_hiddens, weights)
            question_v_hidden = self.end_gru(dropout(weighted_doc_hidden), dropout(question_u_hidden))
            # question_v_hidden = layers.dropout(question_v_hidden)
            end_logits = self.end_attn(dropout(doc_final_hiddens), dropout(question_v_hidden), x1_mask, logit=True)
        else:
            end_logits = self.end_attn(doc_final_hiddens, question_u_hidden, x1_mask, logit=True)

        if self.opt['sentence_level']:
            end_logits = layers.combine_sentences(end_logits, sent_lens)

        if logit:
            return start_logits, end_logits
        else:
            end_scores = F.log_softmax(end_logits, 1) if self.training else F.softmax(end_logits, 1)
            return start_scores, end_scores
