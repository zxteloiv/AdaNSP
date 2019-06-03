from typing import Union, Optional, List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn
import allennlp.models
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TokenEmbedder
from allennlp.nn import util
from models.transformer.multi_head_attention import SingleTokenMHAttentionWrapper
from models.stacked_rnn_cell import StackedRNNCell
from utils.nn import AllenNLPAttentionWrapper, filter_cat, filter_sum
from allennlp.training.metrics import BLEU
from models.stacked_encoder import StackedEncoder

class BaseSeq2Seq(allennlp.models.Model):
    def __init__(self,
                 vocab: Vocabulary,
                 encoder: StackedEncoder,
                 decoder: StackedRNNCell,
                 word_projection: torch.nn.Module,
                 source_embedding: TokenEmbedder,
                 target_embedding: TokenEmbedder,
                 target_namespace: str = "target_tokens",
                 start_symbol: str = '<GO>',
                 eos_symbol: str = '<EOS>',
                 max_decoding_step: int = 50,
                 use_bleu: bool = True,
                 label_smoothing: Optional[float] = None,
                 enc_attention: Union[AllenNLPAttentionWrapper, SingleTokenMHAttentionWrapper, None] = None,
                 dec_hist_attn: Union[AllenNLPAttentionWrapper, SingleTokenMHAttentionWrapper, None] = None,
                 scheduled_sampling_ratio: float = 0.,
                 intermediate_dropout: float = .1,
                 concat_attn_to_dec_input: bool = False,
                 ):
        super(BaseSeq2Seq, self).__init__(vocab)
        self._enc_attn = enc_attention
        self._dec_hist_attn = dec_hist_attn

        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._encoder = encoder
        self._decoder = decoder
        self._src_embedding = source_embedding
        self._tgt_embedding = target_embedding

        src_hidden_dim = self._encoder.get_output_dim()
        tgt_hidden_dim = self._decoder.hidden_dim

        self._enc_attn_mapping = torch.nn.Linear(src_hidden_dim, tgt_hidden_dim)
        self._dec_hist_attn_mapping = torch.nn.Linear(tgt_hidden_dim, tgt_hidden_dim)

        self._start_id = vocab.get_token_index(start_symbol, target_namespace)
        self._eos_id = vocab.get_token_index(eos_symbol, target_namespace)
        self._max_decoding_step = max_decoding_step

        self._target_namespace = target_namespace
        self._label_smoothing = label_smoothing

        self._output_projection = word_projection

        self._concat_attn = concat_attn_to_dec_input
        self._dropout = torch.nn.Dropout(intermediate_dropout)

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token, target_namespace)
            self._bleu = BLEU(exclude_indices={pad_index, self._eos_id, self._start_id})
        else:
            self._bleu = None


    def forward(self,
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """Run the network, and dispatch work to helper functions based on the runtime"""

        # source: (batch, source_length), containing the input word IDs
        # target: (batch, target_length), containing the output IDs

        source, source_mask = source_tokens['tokens'], util.get_text_field_mask(source_tokens)
        state, layer_states = self._encode(source, source_mask)
        init_hidden, _ = self._init_hidden_states(layer_states, source_mask)

        if target_tokens is not None:
            target, target_mask = target_tokens['tokens'], util.get_text_field_mask(target_tokens)

            # predictions: (batch, seq_len)
            # logits: (batch, seq_len, vocab_size)
            predictions, logits, others = self._forward_loop(state, source_mask, init_hidden, target, target_mask)
            loss = self._get_loss(target, target_mask.float(), logits, others)
            self._compute_metric(predictions, target[:, 1:])

        else:
            predictions, logits, _ = self._forward_loop(state, source_mask, init_hidden, None, None)
            loss = [-1]

        output = {
            "predictions": predictions,
            "logits": logits,
            "loss": loss,
        }

        return output

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert the predicted word ids into discrete tokens"""
        # predictions: (batch, max_length)
        predictions = output_dict["predictions"]
        if not isinstance(predictions, np.ndarray):
            predictions = predictions.detach().cpu().numpy()
        all_predicted_tokens = []

        for token_ids in predictions:
            if token_ids.ndim > 1:
                token_ids = token_ids[0]

            token_ids = list(token_ids)
            if self._eos_id in token_ids:
                token_ids = token_ids[:token_ids.index(self._eos_id)]
            tokens = [self.vocab.get_token_from_index(token_id, namespace=self._target_namespace)
                      for token_id in token_ids]
            all_predicted_tokens.append(tokens)
        output_dict['predicted_tokens'] = all_predicted_tokens
        return output_dict

    def _encode(self,
                source: torch.LongTensor,
                source_mask: torch.LongTensor):
        # source: (batch, max_input_length), source sequence token ids
        # source_mask: (batch, max_input_length), source sequence padding mask
        # source_embedding: (batch, max_input_length, embedding_sz)
        source_embedding = self._src_embedding(source)
        source_embedding = self._dropout(source_embedding)
        source_hidden, layered_hidden = self._encoder(source_embedding, source_mask)
        return source_hidden, layered_hidden

    def _forward_loop(self,
                      source_state: torch.Tensor,
                      source_mask: Optional[torch.LongTensor],
                      init_hidden: torch.Tensor,
                      target: Optional[torch.LongTensor],
                      target_mask: Optional[torch.LongTensor],
                      ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Do the decoding process for training and prediction

        :param source_state: (batch, max_input_length, hidden_dim),
        :param source_mask: (batch, max_input_length)
        :param target: (batch, max_target_length)
        :param target_mask: (batch, max_target_length)
        :return:
        """

        # shape: (batch, max_input_sequence_length)
        batch = source_state.size()[0]

        if target is not None:
            num_decoding_steps = target.size()[1] - 1
        else:
            num_decoding_steps = self._max_decoding_step

        # Initialize target predictions with the start index.
        # batch_start: (batch_size,)
        batch_start = source_mask.new_full((batch,), fill_value=self._start_id)
        step_hidden, step_output = init_hidden, self._decoder.get_output_state(init_hidden)

        if self._enc_attn is not None:
            enc_attn_fn = lambda out: self._enc_attn_mapping(self._enc_attn(out, source_state, source_mask))
        else:
            enc_attn_fn = None

        # acc_halting_probs: [(batch,)]
        # updated_num_by_step: [(batch,)]
        # step_logits: [(batch, seq_len, vocab_size)]
        # a list of predicted token ids at each step: [(batch,)]
        logits_by_step = []
        output_by_step = []
        others_by_step = []
        predictions_by_step = [batch_start]
        for timestep in range(num_decoding_steps):
            if self.training and np.random.rand(1).item() < self._scheduled_sampling_ratio:
                # use self-predicted tokens for scheduled sampling in training with _scheduled_sampling_ratio
                # step_inputs: (batch,)
                step_inputs = predictions_by_step[-1]
            elif not self.training or target is None:
                # no target present, maybe in validation
                # step_inputs: (batch,)
                step_inputs = predictions_by_step[-1]
            else:
                # gold choice
                # step_inputs: (batch,)
                step_inputs = target[:, timestep]

            # inputs_embedding: (batch, embedding_dim)
            inputs_embedding = self._tgt_embedding(step_inputs)
            inputs_embedding = self._dropout(inputs_embedding)

            if self._dec_hist_attn is None:
                dec_hist_attn_fn = None

            elif len(output_by_step) > 0:
                dec_hist = torch.stack(output_by_step, dim=1)
                dec_hist_mask = target_mask[:, :timestep] if target_mask is not None else None
                dec_hist_attn_fn = lambda out: self._dec_hist_attn_mapping(
                    self._dec_hist_attn(out, dec_hist, dec_hist_mask)
                )

            else:
                dec_hist_attn_fn = lambda out: torch.zeros_like(out)

            dec_out = self._run_decoder(target[:, timestep + 1] if target is not None else None,
                                        inputs_embedding, step_hidden, step_output, enc_attn_fn, dec_hist_attn_fn)
            step_hidden, step_output, step_logit = dec_out[:3]
            if len(dec_out) > 3:
                others_by_step.append(dec_out[3:])

            output_by_step.append(step_output)
            logits_by_step.append(step_logit)

            # greedy decoding
            # step_prediction: (batch, )
            step_prediction = torch.argmax(step_logit, dim=-1)
            predictions_by_step.append(step_prediction)

        # predictions: (batch, seq_len)
        # logits: (batch, seq_len, vocab_size)
        predictions = torch.stack(predictions_by_step[1:], dim=1)
        logits = torch.stack(logits_by_step, dim=1)

        return predictions, logits, others_by_step

    def _run_decoder(self, step_target, inputs_embedding, step_hidden, step_output, enc_attn_fn, dec_hist_attn_fn):
        batch = inputs_embedding.size()[0]
        # compute attention context before the output is updated
        enc_context = enc_attn_fn(step_output) if enc_attn_fn else None
        dec_hist_context = dec_hist_attn_fn(step_output) if dec_hist_attn_fn else None

        # step_hidden: some_hidden_var_with_unknown_internals
        # step_output: (batch, hidden_dim)
        cat_context = []
        if self._concat_attn and enc_context is not None:
            cat_context.append(self._dropout(enc_context))
        if self._concat_attn and dec_hist_context is not None:
            cat_context.append(self._dropout(dec_hist_context))
        dec_output = self._decoder(inputs_embedding, step_hidden, cat_context)
        step_hidden, step_output = dec_output[:2]

        step_logit = self._get_step_projection(step_output, enc_context, dec_hist_context)

        return step_hidden, step_output, step_logit

    def _get_step_projection(self, *inputs):
        # step_logit: (batch, vocab_size)
        if self._concat_attn:
            proj_input = filter_cat(inputs, dim=-1)
        else:
            proj_input = filter_sum(inputs)

        proj_input = self._dropout(proj_input)
        step_logit = self._output_projection(proj_input)
        return step_logit


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu: # and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics

    def _get_loss(self, target, target_mask, logits, other):
        loss_pred = util.sequence_cross_entropy_with_logits(logits, target[:, 1:].contiguous(), target_mask[:, 1:],
                                                            label_smoothing=self._label_smoothing)
        loss = loss_pred
        return loss

    def _init_hidden_states(self, layer_state, source_mask: torch.LongTensor):
        batch, _, hidden_dim = layer_state[0].size()

        last_word_indices = source_mask.sum(1).long() - 1
        expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch, 1, hidden_dim)

        # [(batch, hidden_dim)]
        forward_by_layer = [state.gather(1, expanded_indices).squeeze(1) for state in layer_state]

        if self._encoder.is_bidirectional():
            hidden_dim = hidden_dim // 2
            forward = [state[:, :hidden_dim] for state in forward_by_layer]
            backward = [layer_state[i][:, 0, hidden_dim:] for i in range(len(forward))]

        else:
            forward = forward_by_layer
            backward = None

        return self._decoder.init_hidden_states_by_layer(forward, backward)

    def _compute_metric(self, predictions, labels):
        if self._bleu:
            self._bleu(predictions, labels)

