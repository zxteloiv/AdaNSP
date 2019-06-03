from typing import Dict, List, Tuple, Mapping, Optional
import torch

from allennlp.modules import LayerNorm, FeedForward
from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention
from allennlp.nn import Activation
# from allennlp.nn.util import add_positional_features
from utils.nn import add_position_and_timestep_sinusoid, add_positional_features

from .multi_head_attention import MultiHeadSelfAttention, MaskedMultiHeadSelfAttention, MultiHeadAttention
from .adaptive_computing import AdaptiveComputing

class TransformerDecoder(torch.nn.Module):
    def __init__(self,
                 input_dim: int,  # input embedding dimension
                 num_layers: int = 6,
                 num_heads: int = 8,
                 feedforward_hidden_dim: int = None,
                 feedforward_dropout: float = 0.1,
                 attention_dim: int = None,
                 value_dim: int = None,
                 residual_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 use_positional_embedding: bool = True,
                 ):
        """
        Construct a decoder for transformer, which is in charge of modules in the transformer model
        from the Positional Embedding before the final linear projection.
        The embedding and linear projection should be implemented elsewhere.

        :param num_layers: the number of stack layers of the transformer block
        """
        super(TransformerDecoder, self).__init__()

        self._mask_attention_layers: List[MaskedMultiHeadSelfAttention] = []
        self._mask_attention_norm_layers: List[LayerNorm] = []
        self._attention_layers: List[MultiHeadAttention] = []
        self._attention_norm_layers: List[LayerNorm] = []
        self._feedforward_layers: List[FeedForward] = []
        self._feedforward_norm_layers: List[LayerNorm] = []

        hidden_dim = input_dim  # the hidden states dimension outputted by the decoder module

        attention_dim = attention_dim or (hidden_dim // num_heads)
        value_dim = value_dim or (hidden_dim // num_heads)
        feedforward_hidden_dim = feedforward_hidden_dim or hidden_dim

        for i in range(num_layers):
            masked_attention = MaskedMultiHeadSelfAttention(num_heads,
                                                            hidden_dim,
                                                            attention_dim * num_heads,
                                                            value_dim * num_heads,
                                                            attention_dropout=attention_dropout)
            self.add_module(f'masked_attention_{i}', masked_attention)
            self._mask_attention_layers.append(masked_attention)

            masked_attention_norm = LayerNorm(hidden_dim)
            self.add_module(f'masked_attention_norm_{i}', masked_attention_norm)
            self._mask_attention_norm_layers.append(masked_attention_norm)

            attention = MultiHeadAttention(num_heads,
                                           hidden_dim,
                                           hidden_dim,
                                           attention_dim * num_heads,
                                           value_dim * num_heads,
                                           attention_dropout=attention_dropout)
            self.add_module(f'attention_{i}', attention)
            self._attention_layers.append(attention)

            attention_norm = LayerNorm(hidden_dim)
            self.add_module(f'attention_norm_{i}', attention_norm)
            self._attention_norm_layers.append(attention_norm)

            feedfoward = FeedForward(hidden_dim,
                                     num_layers=2,
                                     hidden_dims=[feedforward_hidden_dim, hidden_dim],
                                     activations=[Activation.by_name('relu')(),
                                                  Activation.by_name('linear')()],
                                     dropout=feedforward_dropout)
            self.add_module(f"feedforward_{i}", feedfoward)
            self._feedforward_layers.append(feedfoward)

            feedforward_norm = LayerNorm(hidden_dim)
            self.add_module(f"feedforward_norm_{i}", feedforward_norm)
            self._feedforward_norm_layers.append(feedforward_norm)

        self._dropout = torch.nn.Dropout(residual_dropout)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._use_positional_embedding = use_positional_embedding

    def forward(self,
                target: torch.Tensor,
                mask: Optional[torch.LongTensor],
                source_hidden: torch.Tensor,
                source_mask: Optional[torch.LongTensor]
                ) -> torch.Tensor:
        """
        Transformer deocder stacked blocks.

        :param target: (batch, max_target_len, output_embedding_dim)
        :param mask: (batch, max_target_len), 0 or 1 as a mask matrix
        :param source_hidden: (batch, max_source_len, source_embedding_dim)
        :param source_mask: (batch, max_source_len)
        :return: (batch, max_target_len, output_embedding_dim)
        """
        output_tensor = add_positional_features(target) if self._use_positional_embedding else target

        for (masked_attention,
             masked_attention_norm,
             attention,
             attention_norm,
             feedforward,
             feedforward_norm) in zip(self._mask_attention_layers,
                                      self._mask_attention_norm_layers,
                                      self._attention_layers,
                                      self._attention_norm_layers,
                                      self._feedforward_layers,
                                      self._feedforward_norm_layers):

            masked_attention_out, _ = masked_attention(output_tensor, mask)
            masked_attention_out = self._dropout(masked_attention_out)           # add residual dropout
            masked_attention_out = masked_attention_norm(masked_attention_out + output_tensor)  # add residual connection

            attention_out, _ = attention(masked_attention_out, source_hidden, source_mask)
            attention_out = self._dropout(attention_out)                         # add residual dropout
            attention_out = attention_norm(attention_out + masked_attention_out) # add residual connection

            feedforward_out = feedforward(attention_out)
            feedforward_out = self._dropout(feedforward_out)                     # add residual dropout
            feedforward_out = feedforward_norm(feedforward_out + attention_out)  # add residual connection

            output_tensor = feedforward_out

        return output_tensor


class UTDecoder(torch.nn.Module):
    def __init__(self,
                 input_dim: int,  # input embedding dimension
                 max_num_layers: int = 6,
                 num_heads: int = 8,
                 feedforward_hidden_dim: int = None,
                 feedforward_dropout: float = 0.1,
                 attention_dim: Optional[int] = None,
                 value_dim: Optional[int] = None,
                 residual_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 use_act: bool = True,  # use adaptive computation time
                 act_dropout: float = 0.1,
                 act_epsilon: float = 0.1,
                 use_vanilla_wiring: bool = False,
                 ):
        super(UTDecoder, self).__init__()
        self.hidden_dim = input_dim
        self._use_act = use_act
        self._max_num_layers = max_num_layers

        if use_act:
            halting_fn = torch.nn.Sequential(
                torch.nn.Linear(input_dim, input_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(act_dropout),
                torch.nn.Linear(input_dim // 2, 1),
                torch.nn.Sigmoid(),
            )
            self._act_fn = AdaptiveComputing(halting_fn,
                                             max_computing_time=max_num_layers,
                                             epsilon=act_epsilon)

        self._ut_block = UTDecBlock(input_dim,
                                    num_heads=num_heads,
                                    attention_dim=attention_dim,
                                    value_dim=value_dim,
                                    feedforward_hidden_dim=feedforward_hidden_dim,
                                    residual_dropout=residual_dropout,
                                    attention_dropout=attention_dropout,
                                    feedforward_dropout=feedforward_dropout,
                                    use_vanilla_wiring=use_vanilla_wiring
                                    )

    def forward(self,
                target: torch.Tensor,
                mask: Optional[torch.LongTensor],
                source_hidden: torch.Tensor,
                source_mask: Optional[torch.LongTensor],
                ) -> torch.Tensor:
        # ACT not enabled, just act like a stacked RNN, up to the maximum number of layers
        if not self._use_act:
            for timestep in range(self._max_num_layers):
                target = self._ut_block(target, mask, source_hidden, source_mask, timestep)
            return target

        output = self._act_fn(mask, source_hidden, source_mask,
                              inputs=target,
                              step_fn=self._ut_block)
        return output


class UTDecBlock(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_heads: int = 8,
                 attention_dim: Optional[int] = None,
                 value_dim: Optional[int] = None,
                 feedforward_hidden_dim: int = None,
                 residual_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 feedforward_dropout: float = 0.1,
                 use_vanilla_wiring: bool = False,):
        super(UTDecBlock, self).__init__()
        hidden_dim = input_dim
        attention_dim = attention_dim or (hidden_dim // num_heads)
        value_dim = value_dim or (hidden_dim // num_heads)
        feedforward_hidden_dim = feedforward_hidden_dim or hidden_dim

        self._masked_attention = MaskedMultiHeadSelfAttention(num_heads,
                                                              hidden_dim,
                                                              attention_dim * num_heads,
                                                              value_dim * num_heads,
                                                              attention_dropout=attention_dropout)
        self._masked_attention_norm = LayerNorm(hidden_dim)

        self._attention = MultiHeadAttention(num_heads,
                                             hidden_dim,
                                             hidden_dim,
                                             attention_dim * num_heads,
                                             value_dim * num_heads,
                                             attention_dropout=attention_dropout)
        self._dropout = torch.nn.Dropout(residual_dropout)
        self._attention_norm = LayerNorm(hidden_dim)

        # use feedforward net as transition function
        self._feedforward = FeedForward(hidden_dim,
                                        num_layers=2,
                                        hidden_dims=[feedforward_hidden_dim, hidden_dim],
                                        activations=[Activation.by_name('relu')(),
                                                     Activation.by_name('linear')()],
                                        dropout=feedforward_dropout)
        self._feedforward_norm = LayerNorm(hidden_dim)

        self._use_vanilla_wiring = use_vanilla_wiring

    def forward(self,
                target: torch.Tensor,
                mask: Optional[torch.LongTensor],
                source_hidden: torch.Tensor,
                source_mask: Optional[torch.LongTensor],
                timestep: int,
                ) -> torch.Tensor:

        target = add_position_and_timestep_sinusoid(target, timestep=timestep)

        masked_att_out, _ = self._masked_attention(target, mask)
        if self._use_vanilla_wiring:
            masked_att_out = self._dropout(masked_att_out)
            masked_att_out += target
        else:
            masked_att_out += target
            masked_att_out = self._dropout(masked_att_out)
        masked_att_out = self._masked_attention_norm(masked_att_out)

        attention_out, _ = self._attention(masked_att_out, source_hidden, source_mask)
        if self._use_vanilla_wiring:
            attention_out = self._dropout(attention_out)
            attention_out += masked_att_out
        else:
            attention_out += masked_att_out
            attention_out = self._dropout(attention_out)
        attention_out = self._attention_norm(attention_out)

        # use feedforward net as transition function
        feedforward_out = self._feedforward(attention_out)
        if self._use_vanilla_wiring:
            feedforward_out = self._dropout(feedforward_out)
            feedforward_out += attention_out
        else:
            feedforward_out += attention_out
            feedforward_out = self._dropout(feedforward_out)
        feedforward_out = self._feedforward_norm(feedforward_out)

        return feedforward_out
