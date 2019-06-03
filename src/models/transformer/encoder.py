from typing import Dict, List, Tuple, Mapping, Optional
import torch

from allennlp.modules import LayerNorm, FeedForward
from utils.nn import add_position_and_timestep_sinusoid, add_positional_features
from allennlp.nn import Activation

from .multi_head_attention import MultiHeadSelfAttention
from .adaptive_computing import AdaptiveComputing

class TransformerEncoder(torch.nn.Module):
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
        super(TransformerEncoder, self).__init__()

        self._attention_layers: List[MultiHeadSelfAttention] = []
        self._attention_norm_layers: List[LayerNorm] = []
        self._feedforward_layers: List[FeedForward] = []
        self._feedforward_norm_layers: List[LayerNorm] = []

        hidden_dim = input_dim
        attention_dim = attention_dim or (hidden_dim // num_heads)
        value_dim = value_dim or (hidden_dim // num_heads)
        feedforward_hidden_dim = feedforward_hidden_dim or hidden_dim

        for i in range(num_layers):
            attention = MultiHeadSelfAttention(num_heads,
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

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output_tensor = add_positional_features(inputs) if self._use_positional_embedding else inputs

        for (attention,
             attention_norm,
             feedforward,
             feedforward_norm) in zip(self._attention_layers,
                                      self._attention_norm_layers,
                                      self._feedforward_layers,
                                      self._feedforward_norm_layers):
            cached_input = output_tensor

            attention_out, _ = attention(output_tensor, mask)
            attention_out = self._dropout(attention_out)
            attention_out = attention_norm(attention_out + cached_input)

            feedforward_out = feedforward(attention_out)
            feedforward_out = self._dropout(feedforward_out)
            feedforward_out = feedforward_norm(feedforward_out + attention_out)

            output_tensor = feedforward_out

        return output_tensor

    def get_output_dim(self) -> int:
        return self.hidden_dim

    def is_bidirectional(self) -> bool:
        return False

    def get_input_dim(self) -> int:
        return self.input_dim


class UTEncoder(torch.nn.Module):
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
        super(UTEncoder, self).__init__()
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

        self._ut_block = UTEncBlock(input_dim,
                                    num_heads=num_heads,
                                    attention_dim=attention_dim,
                                    value_dim=value_dim,
                                    feedforward_hidden_dim=feedforward_hidden_dim,
                                    residual_dropout=residual_dropout,
                                    attention_dropout=attention_dropout,
                                    feedforward_dropout=feedforward_dropout,
                                    use_vanilla_wiring=use_vanilla_wiring
                                    )

    def get_output_dim(self) -> int:
        return self.hidden_dim

    def is_bidirectional(self) -> bool:
        return False

    def get_input_dim(self) -> int:
        return self.input_dim

    def forward(self, inputs: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        # ACT not enabled, just act like a stacked RNN, up to the maximum number of layers
        if not self._use_act:
            for timestep in range(self._max_num_layers):
                inputs = self._ut_block(inputs, mask, timestep)
            return inputs

        output = self._act_fn(mask, inputs=inputs, step_fn=self._ut_block)
        return output


class UTEncBlock(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_heads: int = 8,
                 attention_dim: Optional[int] = None,
                 value_dim: Optional[int] = None,
                 feedforward_hidden_dim: int = None,
                 residual_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 feedforward_dropout: float = 0.1,
                 use_vanilla_wiring: bool = False):
        super(UTEncBlock, self).__init__()
        hidden_dim = input_dim
        attention_dim = attention_dim or (hidden_dim // num_heads)
        value_dim = value_dim or (hidden_dim // num_heads)
        feedforward_hidden_dim = feedforward_hidden_dim or hidden_dim

        self._attention = MultiHeadSelfAttention(num_heads,
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

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, timestep: int) -> torch.Tensor:
        inputs = add_position_and_timestep_sinusoid(inputs, timestep=timestep)

        attention_out, _ = self._attention(inputs, mask)
        if self._use_vanilla_wiring:
            attention_out = self._dropout(attention_out)
            attention_out += inputs
        else:
            attention_out += inputs
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
