from typing import List, Tuple, Dict, Mapping, Optional
import torch

from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.modules.attention import Attention
from allennlp.nn.util import masked_softmax

def add_position_and_timestep_sinusoid(inputs: torch.Tensor,
                                       timestep: Optional[float] = None,
                                       base_num: float = 1.e4) -> torch.Tensor:
    """
    Add positional embedding to inputs, which contains words at every position

    :param inputs: input embedding for entire sequence, a shape of (batch, seq_len, embedding) assumed
    :param timestep: which timestep is the current recurrent block

    :param base_num: the base number used in dimensional power, default to 10^4 as the Attention is All You Need paper.

    :return: same shape as inputs
    """
    _, seq_len, emb_dim = inputs.size()
    assert emb_dim % 2 == 0, "embedding dimension must be even"

    # position: [0, 1, ..., seq_len - 1] with shape (seq_len, )
    # negative_half_dim_index: [-0, -1, ..., -(emb_dim // 2 - 1)] with shape (emb_dim // 2, )
    # inverse_dim: [base_num ^(-0), ..., base_num^(-(emb_dim // 2 - 1))], with shape (emb_dim // 2,)
    position: torch.Tensor = inputs.new_ones(seq_len).cumsum(dim=0).float() - 1
    negative_half_dim_index: torch.Tensor = -(inputs.new_ones(emb_dim // 2).cumsum(dim=0).float() - 1)
    inverse_dim: torch.Tensor = torch.pow(base_num, negative_half_dim_index)

    # x: (seq_len, emb_dim // 2) <- (seq_len, 1) * (1, emb_dim // 2)
    x: torch.Tensor = position.unsqueeze(1) * inverse_dim.unsqueeze(0)

    if timestep is not None:
        # y: (1, emb_dim // 2)
        y: torch.Tensor = inverse_dim.unsqueeze(0) * timestep
        sinusoid_odd, sinusoid_even = x.sin() + y.sin(), x.cos() + y.cos()
    else:
        sinusoid_odd, sinusoid_even = x.sin(), x.cos()

    # sinusoid: (seq_len, emb_dim // 2, 2) -> (1, seq_len, emb_dim)
    sinusoid: torch.Tensor = torch.stack([sinusoid_odd, sinusoid_even], dim=2).reshape(1, seq_len, -1)

    return inputs + sinusoid

def add_positional_features(inputs: torch.Tensor) -> torch.Tensor:
    """
    A wrapper with the same name from AllenNLP
    """
    return add_position_and_timestep_sinusoid(inputs, None)

def add_depth_features_to_single_position(inputs: torch.Tensor, timestep: float) -> torch.Tensor:
    """
    Add depth-wise features to inputs.
    The word ``depth'' is similar to ``timestep'' in Transformer.

    :param inputs: (batch, emb_dim)
    :param timestep: float
    :returns same shape with inputs: (batch, emb_dim)
    """
    return add_position_and_timestep_sinusoid(inputs.unsqueeze(1), timestep).squeeze(1)

class AllenNLPMatrixAttentionWrapper(torch.nn.Module):
    """
    A wrapper for matrix attention in allennlp, fitting the interface of the multi-headed attention
    defined in models.transformer.multi_head_attention
    """
    def __init__(self, attn: MatrixAttention):
        super(AllenNLPMatrixAttentionWrapper, self).__init__()
        self._attn: MatrixAttention = attn

    def forward(self,
                input: torch.Tensor,
                attend_over: torch.Tensor,
                attend_mask: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wrap the Attention in AllenNLP, with sufficient dimension and context value computation

        :param input: (batch, max_input_length, input_dim)
        :param attend_over: (batch, max_attend_length, attend_dim)
        :param attend_mask: (batch, max_attend_length), used to blind out padded tokens
        :return: Tuple of context vector and attention vector:
                   context: (batch, max_input_length, output_dim=attend_dim)
                 attention: (batch, max_input_length, 1, max_attend_length)
        """

        # logits: (batch, max_input_length, max_attend_length)
        logits = self._attn(input, attend_over)

        # attend_mask: (batch, 1, max_attend_length)
        attend_mask = attend_mask.unsqueeze(1)

        # attn: (batch, max_input_length, max_attend_length)
        attn = masked_softmax(logits, attend_mask)

        # context: (batch, max_input_length, attend_dim)
        context = torch.matmul(attn, attend_over)

        return context, attn.unsqueeze(-2)


class AllenNLPAttentionWrapper(torch.nn.Module):
    """
    A wrapper for matrix attention in allennlp, fitting the interface of the multi-headed attention
    defined in models.transformer.multi_head_attention
    """
    def __init__(self, attn: Attention, attn_dropout: float = 0.):
        super(AllenNLPAttentionWrapper, self).__init__()
        self._attn: Attention = attn
        self._dropout = torch.nn.Dropout(attn_dropout)

    def forward(self,
                inputs: torch.Tensor,
                attend_over: torch.Tensor,
                attend_mask: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wrap the Attention in AllenNLP, with sufficient dimension and context value computation

        :param inputs: (batch, input_dim)
        :param attend_over: (batch, max_attend_length, attend_dim)
        :param attend_mask: (batch, max_attend_length), used to blind out padded tokens
        :return: Tuple of context vector and attention vector:
                   context: (batch, max_input_length, output_dim=attend_dim)
                 attention: (batch, max_input_length, 1, max_attend_length)
        """

        # attn: (batch, max_attend_length, -1)
        attn = self._attn(inputs, attend_over, attend_mask)
        attn = self._dropout(attn).unsqueeze(-1)

        # context: (batch, attend_dim)
        context = (attn * attend_over).sum(1)

        return context

def filter_cat(iterable, dim):
    items = [item for item in iterable if item is not None]
    res = torch.cat(items, dim=dim)
    return res

def filter_sum(iterable):
    items = [item for item in iterable if item is not None]
    res = None
    for item in items:
        if res is None:
            res = item
        else:
            res = res + item
    return res
