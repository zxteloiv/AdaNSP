from typing import List, Mapping, Dict, Optional, Tuple, Union, Callable, Sequence
import torch
import torch.nn

from utils.nn import AllenNLPAttentionWrapper, filter_cat
from models.transformer.multi_head_attention import SingleTokenMHAttentionWrapper
from models.universal_hidden_state_wrapper import UniversalHiddenStateWrapper
from models.stacked_rnn_cell import StackedRNNCell

class AdaptiveStateMode:
    BASIC = "basic"
    RANDOM = "random"
    MEAN_FIELD = "mean_field"

class AdaptiveRNNCell(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, hidden,
                enc_attn_fn: Optional[Callable],
                dec_hist_attn_fn: Optional[Callable]) -> Tuple[torch.Tensor, torch.Tensor,
                                                               Optional[torch.Tensor], Optional[torch.Tensor]]:
        raise NotImplementedError

    def init_hidden_states(self, source_state, source_mask, is_bidirectional=False):
        raise NotImplementedError

class ACTRNNCell(AdaptiveRNNCell):
    """
    An RNN-based cell, which adaptively computing the hidden states along depth dimension.
    """
    def __init__(self,
                 rnn_cell: Union[UniversalHiddenStateWrapper, StackedRNNCell],
                 halting_fn: Callable,
                 use_act: bool,
                 act_max_layer: int = 10,
                 act_epsilon: float = .1,
                 rnn_input_dropout: float = .2,
                 depth_wise_attention: Union[AllenNLPAttentionWrapper, SingleTokenMHAttentionWrapper, None] = None,
                 state_mode: str = AdaptiveStateMode.BASIC,
                 ):
        """
        :param use_act: use adaptive computing
        :param act_max_layer: the maximum number of layers for adaptive computing
        :param act_epsilon: float in (0, 1), a reserved range of probability to halt
        :param depth_wise_attention: do input attention over the adaptive RNN output
        :param rnn_cell: some basic RNN cell, e.g., vanilla RNN, LSTM, GRU, accepts inputs across depth
        """
        super(ACTRNNCell, self).__init__()

        self._rnn_cell = rnn_cell
        self._use_act = use_act

        self._halting_fn = halting_fn

        self._threshold = 1 - act_epsilon
        self._max_computing_time = act_max_layer

        self._input_dropout = torch.nn.Dropout(rnn_input_dropout)

        self._depth_wise_attention = depth_wise_attention

        self.state_mode = state_mode

    def forward(self,
                inputs: torch.Tensor,
                hidden,
                enc_attn_fn: Optional[Callable] = None,
                dec_hist_attn_fn: Optional[Callable] = None):
        """
        :param inputs: (batch, hidden_dim)
        :param hidden: some hidden states with unknown internals
        :param enc_attn_fn: Attention function computing over encoder states,
                The function maps the state directly to a context vector:
                i.e. (batch, hidden_dim) -> (batch, hidden_dim)
        :param dec_hist_attn_fn: Attention function computing over decoded history,
                The function maps the state directly to a context vector:
                i.e. (batch, hidden_dim) -> (batch, hidden_dim)
        :return: (batch, hidden_dim) or [(batch, hidden_dim)]
        """
        if self._use_act:
            return self._forward_act(inputs, hidden, enc_attn_fn, dec_hist_attn_fn)

        else:
            hidden, out = self._forward_normal(inputs, hidden, enc_attn_fn, dec_hist_attn_fn)
            return hidden, out, None, None

    def _forward_normal(self, inputs: torch.Tensor, hidden, enc_attn_fn, dec_hist_attn_fn):
        # if ACT is disabled, depth is kept 0, depth-wise attn is untouched
        output = self._rnn_cell.get_output_state(hidden)

        # manually dropout these vectors, in order to keep depth flag
        enc_context = self._input_dropout(enc_attn_fn(output)) if enc_attn_fn else None
        dec_hist_context = self._input_dropout(dec_hist_attn_fn(output)) if dec_hist_attn_fn else None

        # depth
        input_aux = [self._input_dropout(enc_context),
                     self._input_dropout(dec_hist_context),
                     self._get_depth_flag(0, inputs)]
        h, o = self._rnn_cell(inputs, hidden, input_aux)
        return h, o

    def _forward_act(self, inputs: torch.Tensor, hidden, enc_attn_fn, dec_hist_attn_fn):
        """
        :param inputs: (batch, hidden_dim)
        :param hidden: some hidden state recognizable by the universal RNN cell wrapper
        :param enc_attn_fn: Attention function, in case or the adaptively computed states need attention manipulation
                The function maps the state directly to a context vector:
                i.e. (batch, hidden_dim) -> (batch, hidden_dim)
        :return: (batch, hidden_dim) or [(batch, hidden_dim)]
        """
        output = self._rnn_cell.get_output_state(hidden) # output initialized as in the last time

        # halting_prob_cumulation: (batch,)
        # halting_prob_list: [(batch,)]
        # hidden_list: [hidden]
        # alive_mask_list: [(batch,)]
        batch = inputs.size()[0]
        halting_prob_acc = inputs.new_zeros(batch).float()
        halting_prob_list = []
        hidden_list = []
        alive_mask_list = []

        depth = 0
        while depth < self._max_computing_time and (halting_prob_acc < 1.).any():
            # current all alive tokens, which need further computation
            # alive_mask: (batch,)
            alive_mask: torch.Tensor = halting_prob_acc < 1.
            alive_mask = alive_mask.float()

            enc_context = enc_attn_fn(output) if enc_attn_fn else None
            dec_hist_context = dec_hist_attn_fn(output) if dec_hist_attn_fn else None
            depth_context = self._get_depth_wise_attention(output, hidden_list)

            halting_input = filter_cat([output, enc_context, dec_hist_context, depth_context], dim=-1)
            # halting_prob: (batch, ) <- (batch, 1)
            halting_prob = self._halting_fn(halting_input).squeeze(-1)

            # mask to the newly halted tokens, which is exhausted at the current timestep of computation.
            # if the depth hits its upper bound, all nodes should be halted
            # new_halted: (batch,)
            new_halted = ((halting_prob * alive_mask + halting_prob_acc) > self._threshold).float()
            if depth == self._max_computing_time - 1:
                new_halted = new_halted.new_ones(batch).float()
            remainder = 1. - halting_prob_acc + 1.e-15

            # all tokens that survives from the current timestep's computation
            # alive_mask: (batch, )
            alive_mask *= (1 - new_halted)

            # cumulations for newly halted positions will reach 1.0 after adding up remainder at the current timestep
            step_halting_prob = halting_prob * alive_mask + remainder * new_halted
            halting_prob_acc = halting_prob_acc + step_halting_prob

            # Every hidden state at present is paired with an alive mask telling if it needs further computation.
            # And the halting probability at the step is either
            #   - the halting probability just computed, if it's not the last computation step of the very token,
            #   - the remainder for tokens which should get halted at this step.
            hidden_list.append(hidden)
            alive_mask_list.append(alive_mask)
            halting_prob_list.append(step_halting_prob)

            # step_inputs: (batch, hidden_dim)
            input_aux = [self._input_dropout(enc_context),
                         self._input_dropout(dec_hist_context),
                         self._get_depth_flag(depth, inputs)]
            hidden, output = self._rnn_cell(inputs, hidden, input_aux)

            depth += 1

        # halting_probs: (batch, max_computing_depth)
        # alive_masks: (batch, max_computing_depth)
        halting_probs = torch.stack(halting_prob_list, dim=-1).float()
        alive_masks = torch.stack(alive_mask_list, dim=-1).float()
        merged_hidden = self._adaptively_merge_hidden_list(hidden_list, halting_probs, alive_masks)
        merged_output = self._rnn_cell.get_output_state(merged_hidden)

        halting_prob_acc = (halting_probs * alive_masks).sum(1)
        num_updated = alive_masks.sum(1)

        return merged_hidden, merged_output, halting_prob_acc, num_updated

    @staticmethod
    def _get_depth_flag(depth: int, inputs: torch.Tensor):
        batch = inputs.size()[0]
        if depth > 1:
            return inputs.new_zeros(batch, 1)
        else:
            return inputs.new_ones(batch, 1)

    def _adaptively_merge_hidden_list(self, hidden_list, halting_probs, alive_masks):
        # halting_probs: (batch, max_computing_depth)
        # alive_masks: (batch, max_computing_depth)
        batch, max_depth = halting_probs.size()

        # weight: (batch, max_computing_depth)
        if self.state_mode == AdaptiveStateMode.BASIC:
            weight = torch.zeros_like(halting_probs)
            weight[torch.arange(batch), alive_masks.sum(1).long()] = 1

        elif self.state_mode == AdaptiveStateMode.MEAN_FIELD:
            weight = halting_probs

        elif self.state_mode == AdaptiveStateMode.RANDOM:
            # samples_index: torch.LongTensor: (batch, )
            samples_index = torch.multinomial(halting_probs, 1).squeeze(-1)
            weight = torch.zeros_like(halting_probs)
            weight[torch.arange(batch), samples_index] = 1

        else:
            raise ValueError(f'Adaptive State Mode {self.state_mode} not supported')

        merged_hidden = self._rnn_cell.merge_hidden_list(hidden_list, weight)
        return merged_hidden

    def _get_depth_wise_attention(self,
                                  output: torch.Tensor,
                                  previous_hidden_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Get depth-wise attention over the previous hiddens

        :param step_inputs: (batch, hidden_dim)
        :param output: (batch, hidden_dim)
        :param previous_output_list: [(batch, hidden_dim)]
        :return: (batch, hidden_dim)
        """
        if self._depth_wise_attention is None:
            return None

        previous_output_list = [self._rnn_cell.get_output_state(hidden)
                                for hidden in previous_hidden_list]

        if len(previous_output_list) == 0:
            return torch.zeros_like(output)

        # attend_over: (batch, steps, hidden_dim)
        # context: (batch, hidden_dim)
        attend_over = torch.stack(previous_output_list, dim=1)
        context = self._depth_wise_attention(output, attend_over)

        return context

    def init_hidden_states(self, source_state, source_mask, is_bidirectional=False):
        batch, _, hidden_dim = source_state.size()

        last_word_indices = source_mask.sum(1).long() - 1
        expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch, 1, hidden_dim)
        final_encoder_output = source_state.gather(1, expanded_indices)
        final_encoder_output = final_encoder_output.squeeze(1)  # (batch_size, hidden_dim)

        if is_bidirectional:
            hidden_dim = hidden_dim // 2
            forward = final_encoder_output[:, :hidden_dim]
            backward = source_state[:, 0, hidden_dim:]

        else:
            forward = final_encoder_output
            backward = None

        return self._rnn_cell.init_hidden_states(forward, backward)

