from typing import Union, Optional, List, Tuple, Dict
import torch
import torch.nn
import numpy as np
from functools import reduce

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TokenEmbedder
from utils.nn import AllenNLPAttentionWrapper, filter_cat, filter_sum
from models.transformer.multi_head_attention import SingleTokenMHAttentionWrapper
from models.base_seq2seq import BaseSeq2Seq
from training.tree_acc_metric import TreeAccuracy

from models.stacked_encoder import StackedEncoder
from models.stacked_rnn_cell import StackedRNNCell

class UncSeq2Seq(BaseSeq2Seq):
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
                 model_mode: int = 0,
                 max_pondering: int = 3,
                 uncertainty_sample_num: int = 10,
                 uncertainty_loss_weight: int = 1.,
                 reinforcement_discount: float = 0.,
                 skip_loss: bool = False,
                 ):
        super(UncSeq2Seq, self).__init__(vocab, encoder, decoder, word_projection,
                                         source_embedding, target_embedding, target_namespace,
                                         start_symbol, eos_symbol, max_decoding_step,
                                         use_bleu, label_smoothing,
                                         enc_attention, dec_hist_attn,
                                         scheduled_sampling_ratio, intermediate_dropout,
                                         concat_attn_to_dec_input,
                                         )

        # training step in uncertainty training:
        # 0: pretrain the traditional seq2seq model
        # 1: fix the pretrained s2s model, train the additional module to control structures
        # 2: joint training
        self._model_mode = model_mode
        self._tree_acc = TreeAccuracy(lambda x: self.decode({"predictions": x})["predicted_tokens"])
        self._max_pondering = max_pondering

        self._unc_est_num = uncertainty_sample_num

        self._unc_loss_weight = uncertainty_loss_weight

        self._reward_discount = reinforcement_discount

        self.skip_loss = skip_loss

        out_dim = self._decoder.hidden_dim
        self.unc_fn = torch.nn.Sequential(
            torch.nn.Linear(out_dim, out_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(out_dim, 1),
            torch.nn.Sigmoid()
        )

    def _run_decoder(self, step_target, inputs_embedding, last_hidden, last_output, enc_attn_fn, dec_hist_attn_fn):
        # ponder with the output
        batch = inputs_embedding.size()[0]
        enc_context, dec_hist_context = None, None
        # alive: (batch, 1)
        alive = inputs_embedding.new_ones(batch, 1, requires_grad=False)

        # all_action_log_probs = [(batch, 1)]
        # all_action_rewards = [(batch, 1)]
        all_action_log_probs = []
        all_action_rewards = []
        for pondering_step in range(self._max_pondering):
            if pondering_step == 0:
                pondering_flag = inputs_embedding.new_zeros(batch, 1)
            else:
                pondering_flag = inputs_embedding.new_ones(batch, 1)

            # compute attention context before the output is updated
            enc_context = enc_attn_fn(last_output) if enc_attn_fn else None
            dec_hist_context = dec_hist_attn_fn(last_output) if dec_hist_attn_fn else None

            # step_hidden: some_hidden_var_with_unknown_internals
            # step_output: (batch, hidden_dim)
            cat_context = []
            if self._concat_attn and enc_context is not None:
                cat_context.append(self._dropout(enc_context))
            if self._concat_attn and dec_hist_context is not None:
                cat_context.append(self._dropout(dec_hist_context))
            dec_output = self._decoder(inputs_embedding, last_hidden, cat_context + [pondering_flag])
            new_hidden, new_output = dec_output[:2]

            if self._model_mode == 0:
                last_hidden, last_output = new_hidden, new_output
                break

            # compute the probability that we need to continue (1 means we are uncertain, and need to go further)
            # halting_prob: (batch, 1)
            # choice: (batch, 1)
            if self._model_mode == 1:   # mode 1 deals only with the uncertainty part
                new_output = new_output.detach()

            halting_prob = self.unc_fn(new_output)

            # make a choice that is either 0 or 1 at every position
            choice = halting_prob.bernoulli()

            if step_target is not None and not self.skip_loss:
                action_probs = halting_prob * choice + (1 - halting_prob) * (1 - choice)
                all_action_log_probs.append(alive * (action_probs + 1e-20).log())

                # compute for the current step the rewards
                # step_unc: (batch,)
                step_unc = self._get_step_uncertainty(step_target, inputs_embedding,
                                                      last_hidden, cat_context, pondering_flag)
                step_unc = step_unc.unsqueeze(-1)

                new_logit = self._get_step_projection(new_output, enc_context, dec_hist_context)
                new_pred = new_logit.argmax(dim=-1)
                correctness = (new_pred == step_target).float().unsqueeze(-1)

                step_reward = choice * step_unc * (1 - correctness) + (1 - choice) * (1 - step_unc) * correctness
                all_action_rewards.append(step_reward)

            # if an item within this batch is alive, the new_output will be used next time,
            # otherwise, the last_output will be retained.
            # i.e., last_output = new_output if alive == 1, otherwise last_output = last_output if alive == 0
            last_output = last_output * (1 - alive) + new_output * alive
            last_hidden = self._decoder.merge_hidden_list([last_hidden, new_hidden],
                                                          torch.cat([1 - alive, alive], dim=1))

            # udpate survivors for the next time step
            alive = alive * choice

        step_logit = self._get_step_projection(last_output, enc_context, dec_hist_context)

        return last_hidden, last_output, step_logit, all_action_log_probs, all_action_rewards

    def _get_step_uncertainty(self, step_target, inputs_embedding, last_hidden, cat_context, pondering_flag):
        with torch.no_grad():
            is_orignally_training = self.training
            if not self.training:
                self.train()

            # step_target -> (batch, 1)
            if len(step_target.size()) == 1:
                step_target = step_target.unsqueeze(-1)

            all_pass_prob = []
            for _ in range(self._unc_est_num):
                dec_output = self._decoder(inputs_embedding, last_hidden, cat_context + [pondering_flag])
                _, new_output = dec_output[:2]

                if self._concat_attn:
                    proj_input = filter_cat([new_output] + cat_context, dim=-1)
                else:
                    proj_input = filter_sum([new_output] + cat_context)

                proj_input = self._dropout(proj_input)
                step_logit = self._output_projection(proj_input)
                step_prob = step_logit.softmax(-1)

                # (batch, 1)
                one_pass_prob = step_prob.gather(dim=-1, index=step_target)
                all_pass_prob.append(one_pass_prob)

            # concated_probs: (batch, self._unc_est_num)
            # uncertainty: (batch,)
            concated_probs = torch.cat(all_pass_prob, dim=-1)
            uncertainty = concated_probs.var(dim=1)
            # prob_mean = concated_probs.mean(dim=1)

            # the lower uncertainty the better, the greater probability the better
            # reward = (-uncertainty) * prob_mean
            # reward = -uncertainty

            scaled_unc = torch.stack([uncertainty, torch.full_like(uncertainty, 0.15)], dim=1).max(-1)[0] / 0.15

            if not is_orignally_training:
                self.eval()

            return scaled_unc

    def _get_loss(self, target, target_mask, logits, others_by_step):
        if self.skip_loss:
            return 0

        loss_nll = super(UncSeq2Seq, self)._get_loss(target, target_mask, logits, others_by_step)
        if self._model_mode == 0:
            return loss_nll

        action_log_probs_segment, action_rewards_segment = zip(*others_by_step)
        all_action_log_probs = reduce(lambda x, y: x + y, action_log_probs_segment)
        all_action_rewards = reduce(lambda x, y: x + y, action_rewards_segment)

        assert len(all_action_rewards) == len(all_action_log_probs)

        n_steps = len(all_action_rewards)
        if n_steps > 1:
            for step in range(n_steps - 2, -1, -1): # [L - 2, L - 1, ..., 0]
                all_action_rewards[step] += all_action_rewards[step + 1] * self._reward_discount

        action_rewards = torch.cat(all_action_rewards, dim=1)
        action_log_probs = torch.cat(all_action_log_probs, dim=1)

        loss_unc = - action_rewards * action_log_probs * 10
        loss_unc = loss_unc.sum(dim=1).mean(dim=0) # sum for the tokens then average within the batch

        if self._model_mode == 1:
            return loss_unc

        if self._model_mode == 2:
            return loss_unc * self._unc_loss_weight + loss_nll

        return 0

    def _compute_metric(self, predictions, labels):
        super(UncSeq2Seq, self)._compute_metric(predictions, labels)
        if self._tree_acc:
            self._tree_acc(predictions, labels, None)

    def get_metrics(self, reset: bool = False):
        metrics = super(UncSeq2Seq, self).get_metrics(reset)
        if self._tree_acc and not self.training:
            metrics.update(self._tree_acc.get_metric(reset))
        return metrics

