from typing import Dict, List, Tuple, Mapping, Optional

import numpy
import torch
import torch.nn
import allennlp.models
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TokenEmbedder
from allennlp.nn import util

from allennlp.training.metrics import BLEU

class ParallelSeq2Seq(allennlp.models.Model):
    def __init__(self,
                 vocab: Vocabulary,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 source_embedding: TokenEmbedder,
                 target_embedding: TokenEmbedder,
                 target_namespace: str = "target_tokens",
                 start_symbol: str = '<GO>',
                 eos_symbol: str = '<EOS>',
                 max_decoding_step: int = 50,
                 use_bleu: bool = True,
                 label_smoothing: Optional[float] = None,
                 ):
        super(ParallelSeq2Seq, self).__init__(vocab)
        self._encoder = encoder
        self._decoder = decoder
        self._src_embedding = source_embedding
        self._tgt_embedding = target_embedding

        self._start_id = vocab.get_token_index(start_symbol, target_namespace)
        self._eos_id = vocab.get_token_index(eos_symbol, target_namespace)
        self._max_decoding_step = max_decoding_step

        self._target_namespace = target_namespace
        self._label_smoothing = label_smoothing

        self._output_projection_layer = torch.nn.Linear(decoder.hidden_dim,
                                                        vocab.get_vocab_size(target_namespace))

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)
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
        state = self._encode(source, source_mask)

        if target_tokens is not None and self.training:
            target, target_mask = target_tokens['tokens'], util.get_text_field_mask(target_tokens)

            predictions, logits = self._forward_training(state, target[:, :-1], source_mask, target_mask[:, :-1])
            loss = util.sequence_cross_entropy_with_logits(logits,
                                                           target[:, 1:].contiguous(),
                                                           target_mask[:, 1:].float(),
                                                           label_smoothing=self._label_smoothing)
            if self._bleu:
                self._bleu(predictions, target[:, 1:])

        else:
            predictions, logits = self._forward_prediction(state, source_mask)
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
        if not isinstance(predictions, numpy.ndarray):
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
                source_mask: torch.LongTensor) -> torch.Tensor:
        """
        Do the encoder work: embedding + encoder(which adds positional features and do stacked multi-head attention)

        :param source: (batch, max_input_length), source sequence token ids
        :param source_mask: (batch, max_input_length), source sequence padding mask
        :return: source hidden states output from encoder, which has shape
                 (batch, max_input_length, hidden_dim)
        """

        # source_embedding: (batch, max_input_length, embedding_sz)
        source_embedding = self._src_embedding(source)
        source_hidden = self._encoder(source_embedding, source_mask)
        return source_hidden

    def _forward_training(self,
                          state: torch.Tensor,
                          target: torch.LongTensor,
                          source_mask: Optional[torch.LongTensor],
                          target_mask: Optional[torch.LongTensor]
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run decoder for training, given target tokens as supervision.
        When training, all timesteps are used and computed universally.
        """
        # target_embedding: (batch, max_target_length, embedding_dim)
        # target_hidden:    (batch, max_target_length, hidden_dim)
        # logits:           (batch, max_target_length, vocab_size)
        # predictions:      (batch, max_target_length)
        target_embedding = self._tgt_embedding(target)
        target_hidden = self._decoder(target_embedding, target_mask, state, source_mask)
        logits = self._output_projection_layer(target_hidden)
        predictions = torch.argmax(logits, dim=-1)

        return predictions, logits

    def _forward_prediction(self,
                            state: torch.Tensor,
                            source_mask: Optional[torch.LongTensor],
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run decoder step by step for testing or validation, with no gold tokens available.
        """
        batch_size = state.size()[0]
        # batch_start: (batch,)
        batch_start = torch.ones((batch_size,), dtype=torch.long) * self._start_id

        # step_logits: a list of logits, [(batch, seq_len, vocab_size)]
        logits_by_step = []
        # a list of predicted token ids at each step: [(batch,)]
        predictions_by_step = [batch_start]

        for timestep in range(self._max_decoding_step):
            # step_inputs: (batch, timestep + 1), i.e., at least 1 token at step 0
            # inputs_embedding: (batch, seq_len, embedding_dim)
            # step_hidden:      (batch, seq_len, hidden_dim)
            # step_logit:       (batch, seq_len, vocab_size)
            step_inputs = torch.stack(predictions_by_step, dim=1)
            inputs_embedding = self._tgt_embedding(step_inputs)
            step_hidden = self._decoder(inputs_embedding, None, state, source_mask)
            step_logit = self._output_projection_layer(step_hidden)

            # a list of logits, [(batch, vocab_size)]
            logits_by_step.append(step_logit[:, -1, :])

            # greedy decoding
            # prediction: (batch, seq_len)
            # step_prediction: (batch, )
            prediction = torch.argmax(step_logit, dim=-1)
            step_prediction = prediction[:, -1]
            predictions_by_step.append(step_prediction)

        # predictions: (batch, seq_len)
        # logits: (batch, seq_len, vocab_size)
        predictions = torch.stack(predictions_by_step[1:], dim=1)
        logits = torch.stack(logits_by_step, dim=1)

        return predictions, logits

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics
