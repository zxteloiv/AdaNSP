from typing import Union, List, Mapping, Dict, Tuple, Optional
import torch
import torch.nn
import torch.nn.functional
from allennlp.nn.util import masked_softmax


class GeneralMultiHeadAttention(torch.nn.Module):
    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 total_attention_dim: int,
                 total_value_dim: int,
                 attend_to_dim: Optional[int] = None,
                 output_dim: Optional[int] = None,
                 attention_dropout: float = 0.1,
                 use_future_blinding: bool = False,
                 temperature: Optional[float] = None,
                 ):
        """
        The basic multi-head attention, used for the following cases:
            - self attention at encoder side
            - masked self attention at decoder side
            - (non-self) attention at the decoder, attending over the encoder outputs
            - or, if you like, masked self attention at encoder side

        :param num_heads: Heads number
        :param input_dim: the dimension of input embedding
        :param total_attention_dim: the key and query dimension, all heads combined
        :param total_value_dim: the value dimension, all heads combined
        :param attend_to_dim: the dimension of the sequence to which any token from input will attend
        :param output_dim: the output dimension of the attention layer, all heads combined
        :param attention_dropout: dropout ratio of attention dropout, as specified in paper
        :param use_future_blinding: when self attention is used at the decoder,
                                    tokens should not attend to the tokens after itself
        :param temperature: the temperature used in dot attention, set to square root to d_k by default
        """
        super(GeneralMultiHeadAttention, self).__init__()

        self._num_heads = num_heads

        attend_to_dim = attend_to_dim or input_dim
        output_dim = output_dim or input_dim

        self._key_dim = total_attention_dim // num_heads
        self._query_dim = total_attention_dim // num_heads
        self._value_dim = total_value_dim // num_heads

        self._temperature = temperature or (self._key_dim ** 0.5)
        self._use_future_blinding = use_future_blinding

        self._dropout = torch.nn.Dropout(attention_dropout)

        # key and value mapping, w.r.t. all heads, combined
        self._combined_kv = torch.nn.Linear(attend_to_dim, total_attention_dim + total_value_dim)
        # query mapping, w.r.t. all heads, combined
        self._combined_query = torch.nn.Linear(input_dim, total_attention_dim)
        # final output attention mapping
        self._output = torch.nn.Linear(total_value_dim, output_dim)

    def forward(self,
                input: torch.Tensor,
                attend_over: torch.Tensor,
                attend_mask: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Do a multi-head attention for _input_ tokens over the _attend_over_ tokens.
        _attend_mask_ is used to wipe out padded tokens in the corresponding sequences.

        :param input: (batch, max_input_length, input_dim)
        :param attend_over: (batch, max_attend_length, attend_dim)
        :param attend_mask: (batch, max_attend_length), used to blind out padded tokens
        :return: Tuple of context vector and attention vector:
                   context: (batch, max_input_length, output_dim)
                 attention: (batch, max_input_length, num_heads, max_attend_length)
        """
        kv = self._combined_kv(attend_over)
        # keys: (batch, max_attend_length, num_head * key_dim)
        # values: (batch, max_attend_length, num_head * value_dim)
        keys, values = torch.split(kv, [self._key_dim * self._num_heads, self._value_dim * self._num_heads], dim=-1)

        # queries: (batch, max_input_length, num_head * query_dim)
        queries = self._combined_query(input)

        batch, max_input_length, _ = queries.size()
        _, max_attend_length, _ = keys.size()

        # Rearrange the dimensions for key, values, queries,
        # i.e., split out the head dimension
        keys, values, queries = map(lambda x: x.contiguous(), [keys, values, queries])
        #    keys: (batch, max_attend_len, num_heads, key_dim)
        #  values: (batch, max_attend_len, num_heads, value_dim)
        # queries: (batch, max_input_len,  num_heads, key_dim)
        keys = keys.view(batch, max_attend_length, self._num_heads, self._key_dim)
        values = values.view(batch, max_attend_length, self._num_heads, self._value_dim)
        queries = queries.view(batch, max_input_length, self._num_heads, self._query_dim)

        # When coping with mask, the sequence length and attention dimensions will be
        # crucial, all other dimensions will act independently and identically.
        # Therefore, we permute the tensor and put the sequence and attention dimension
        # at the end, which we are taking consideration into.

        #    keys: (batch, num_heads, key_dim,        max_attend_len)
        #  values: (batch, num_heads, max_attend_len, value_dim)
        # queries: (batch, num_heads, max_input_len,  key_dim)
        keys = keys.permute(0, 2, 3, 1)
        values, queries = map(lambda x: x.permute(0, 2, 1, 3), [values, queries])

        # attn: (batch, num_heads, max_input_len, max_attend_length)
        attn = self.dot_attention(queries, keys, attend_mask)

        # context_by_heads: (batch, num_heads, max_input_len, value_dim)
        context_by_heads = torch.matmul(attn, values)
        # context_by_heads: (batch, max_input_len, num_heads, value_dim)
        context_by_heads = context_by_heads.permute(0, 2, 1, 3)

        # context: (batch, max_input_length, output_dim)
        context = self._output(context_by_heads.reshape(batch, max_input_length, -1))

        # transpose attention matrix for future use
        # attn: (batch, max_input_len, num_heads, max_attend_length)
        attn = attn.transpose(1, 2)

        return context, attn

    def dot_attention(self,
                      queries: torch.Tensor,
                      keys: torch.Tensor,
                      attend_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Doing Dot Attention for multi-heads simultaneously

        :param queries: (batch, num_heads, max_input_len,  key_dim)
        :param keys: (batch, num_heads, key_dim, max_attend_len)
        :param attend_mask: (batch, max_attend_length), used to blind out padded tokens
        :return:
        """
        # similarity: (batch, num_heads, max_input_len, max_attend_len)
        similarity = torch.matmul(queries, keys) / self._temperature
        if attend_mask is not None:
            # attend_mask: (batch, 1, 1, max_attend_length)
            attend_mask = attend_mask.unsqueeze(-2).unsqueeze(-2).float()

        if self._use_future_blinding:
            # If we are using future blinding, which wipes out future attention values for every attended token,
            # we start from a lower triangle full of ones, as last two dimensions of the similarity,
            mask = queries.new_ones(similarity.size()[-2:])
            # and finally build the mask matrix manually, which has the shape
            # (    1, 1, max_input_len, max_attend_len)
            mask = mask.tril_(0).unsqueeze(0).unsqueeze(0)

            # use together the padding mask, which has shape
            # (batch, 1,             1, max_attend_len)
            # the multiplication output will have the shape
            # (batch, 1, max_input_len, max_attend_len)
            attend_mask = attend_mask * mask if attend_mask is not None else mask


        # attn: (batch, num_heads, max_input_len, max_attend_length)
        attn = masked_softmax(similarity, mask=attend_mask, dim=-1)
        attn = self._dropout(attn)

        return attn


class MultiHeadSelfAttention(torch.nn.Module):
    """
    Multi-head self attention.

    - attend to oneself
    - no future blinding mask
    """
    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 total_attention_dim: int,
                 total_value_dim: int,
                 attention_dropout: float = 0.1,
                 temperature: Optional[float] = None,
                 ):
        super(MultiHeadSelfAttention, self).__init__()

        self.self_attention = GeneralMultiHeadAttention(
            num_heads=num_heads,
            input_dim=input_dim,
            total_attention_dim=total_attention_dim,
            total_value_dim=total_value_dim,
            attention_dropout=attention_dropout,
            use_future_blinding=False,
            temperature=temperature
        )

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.self_attention(input=input, attend_over=input, attend_mask=mask)


class MaskedMultiHeadSelfAttention(torch.nn.Module):
    """
    Multi-head self attention.

    - attend to oneself
    - no future blinding mask
    """
    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 total_attention_dim: int,
                 total_value_dim: int,
                 attention_dropout: float = 0.1,
                 temperature: Optional[float] = None,
                 ):
        super(MaskedMultiHeadSelfAttention, self).__init__()

        self.self_attention = GeneralMultiHeadAttention(
            num_heads=num_heads,
            input_dim=input_dim,
            total_attention_dim=total_attention_dim,
            total_value_dim=total_value_dim,
            attention_dropout=attention_dropout,
            use_future_blinding=True,
            temperature=temperature
        )

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.self_attention(input=input, attend_over=input, attend_mask=mask)


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-head self attention.

    - attend to oneself
    - no future blinding mask
    """
    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 attend_to_dim: int,
                 total_attention_dim: int,
                 total_value_dim: int,
                 attention_dropout: float = 0.1,
                 temperature: Optional[float] = None,
                 ):
        super(MultiHeadAttention, self).__init__()

        self.attention = GeneralMultiHeadAttention(
            num_heads=num_heads,
            input_dim=input_dim,
            attend_to_dim=attend_to_dim,
            total_attention_dim=total_attention_dim,
            total_value_dim=total_value_dim,
            attention_dropout=attention_dropout,
            use_future_blinding=False,
            temperature=temperature
        )

    def forward(self, input: torch.Tensor, attend_over: torch.Tensor, attend_mask: Optional[torch.Tensor] = None):
        return self.attention(input=input, attend_over=attend_over, attend_mask=attend_mask)

class SingleTokenMHAttentionWrapper(torch.nn.Module):
    def __init__(self, attn: GeneralMultiHeadAttention):
        super(SingleTokenMHAttentionWrapper, self).__init__()
        self._attn = attn

    def forward(self, inputs, attend_over, attend_mask = None):
        """
        Do a multi-head attention for _input_ tokens over the _attend_over_ tokens.
        _attend_mask_ is used to wipe out padded tokens in the corresponding sequences.

        :param inputs: (batch, input_dim)
        :param attend_over: (batch, max_attend_length, attend_dim)
        :param attend_mask: (batch, max_attend_length), used to blind out padded tokens
        :return: Tuple of context vector and attention vector:
                   context: (batch, output_dim)
                 attention: (batch, num_heads, max_attend_length)
        """
        # inputs: (batch, 1, input_dim)
        inputs = inputs.unsqueeze(1)

        c, a = self._attn(inputs, attend_over, attend_mask)

        c = c.squeeze(1)
        a = a.squeeze(1)

        return c

