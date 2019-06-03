from typing import List, Tuple, Dict, Mapping, Optional
import torch

class AdaptiveComputing(torch.nn.Module):
    def __init__(self,
                 halting_fn: torch.nn.Module,
                 max_computing_time: int = 10,
                 epsilon: float = 0.1,
                 mode: str = "basic"
                 ):
        super(AdaptiveComputing, self).__init__()

        self._halting_fn: torch.nn.Module = halting_fn
        self._threshold: float = 1 - epsilon
        self._max_computing_time: int = max_computing_time

        self._mode: str = mode

    def forward(self,
                *args,
                inputs: torch.nn.Module,
                step_fn: torch.nn.Module, ) -> torch.Tensor:
        """
        Adaptively compute the hidden states, and compute the halting probability for each token

        :param inputs: (batch, seq_len, emb_dim)
        :param mask: (batch, seq_len), input padding mask
        :param step_fn: step function to take to compute a new recurrent state,
                        which accepts inputs, padding mask and timestep, then returns another state
        :param halting_prob_cumulation: (batch, seq_len), the previous cumulated halting_probability

        :return: halting probability: (batch, seq_len)
        """
        timestep = 0
        hidden = inputs
        # halting_prob_cumulation: (batch, seq_len)
        halting_prob_cumulation = hidden.new_zeros(hidden.size()[:-1]).float()

        while timestep < self._max_computing_time and "TODO: exit if all place exhausted":
            # current all alive tokens, which need further computation
            # alive_mask: (batch, seq_len)
            alive_mask: torch.Tensor = halting_prob_cumulation < 1.
            alive_mask = alive_mask.float()

            # halting_prob: (batch, seq_len) <- (batch, seq_len, 1)
            halting_prob = self._halting_fn(hidden).squeeze(-1)

            # temp_cumulation: (batch, seq_len)
            temp_cumulation = halting_prob * alive_mask + halting_prob_cumulation

            # mask to the newly halted tokens, which is exhausted at the current timestep of computation
            # new_halted: (batch, seq_len)
            new_halted = (temp_cumulation > self._threshold).float()
            remainder = 1. - halting_prob_cumulation + 1.e-10

            # all tokens that survives from the current timestep's computation
            # alive_mask: (batch, seq_len)
            alive_mask = (1 - new_halted) * alive_mask

            halting_prob_cumulation += halting_prob * alive_mask
            # cumulations for newly halted positions will reach 1.0 after adding up remainder at the current timestep
            halting_prob_cumulation += remainder * new_halted

            step_out = step_fn(hidden, *args, timestep)
            timestep += 1
            state_update_weight = alive_mask.unsqueeze(-1)
            hidden = state_update_weight * step_out + (1 - state_update_weight) * hidden

        return hidden
