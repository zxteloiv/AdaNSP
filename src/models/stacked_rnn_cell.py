from typing import List, Mapping, Dict, Optional, Tuple, Union, Callable, Sequence
import torch
import torch.nn
from models.universal_hidden_state_wrapper import UniversalHiddenStateWrapper, RNNType
from utils.nn import filter_cat

class StackedRNNCell(torch.nn.Module):
    def __init__(self, RNNType, input_dim, hidden_dim, n_layers, intermediate_dropout: float = 0.):
        super(StackedRNNCell, self).__init__()

        self.layer_rnns = torch.nn.ModuleList([
            UniversalHiddenStateWrapper(RNNType(input_dim, hidden_dim)) for _ in range(n_layers)
        ])

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self._input_dropout = torch.nn.Dropout(intermediate_dropout)

    def get_layer_num(self):
        return len(self.layer_rnns)

    def forward(self, inputs, hidden, input_aux:Optional[List] = None):
        # hidden is a list of subhidden
        last_layer_output = inputs

        if inputs.size()[1] < self.input_dim and input_aux is None:
            raise ValueError('Dimension not match')

        updated_hiddens = []
        for i, rnn in enumerate(self.layer_rnns):
            if i > 0:
                last_layer_output = self._input_dropout(last_layer_output)

            if input_aux is None:
                rnn_input = last_layer_output
            else:
                rnn_input = filter_cat([last_layer_output] + input_aux, dim=1)

            layer_hidden, last_layer_output = rnn(rnn_input, hidden[i])
            updated_hiddens.append(layer_hidden)

        return updated_hiddens, last_layer_output

    def get_output_state(self, hidden):
        last_hidden = hidden[-1]
        return self.layer_rnns[-1].get_output_state(last_hidden)

    def merge_hidden_list(self, hidden_list, weight):
        # hidden_list: [hidden_1, ..., hidden_T] along timestep
        # hidden: (layer_hidden_1, ..., layer_hidden_L) along layers
        layered_list = zip(*hidden_list)
        merged = [self.layer_rnns[i].merge_hidden_list(layer_hidden, weight)
                  for i, layer_hidden in enumerate(layered_list) ]
        return merged

    def init_hidden_states(self, forward_out, backward_out: Optional):
        init_hidden = []
        for i in range(len(self.layer_rnns)):
            h, _ = self.layer_rnns[i].init_hidden_states(forward_out, backward_out)
            init_hidden.append(h)
        return init_hidden, self.get_output_state(init_hidden)

    def init_hidden_states_by_layer(self, layer_forward: List, layer_backward: Optional[List]):
        layer_hidden = []
        for i, rnn in enumerate(self.layer_rnns):
            h, _ = rnn.init_hidden_states(layer_forward[i], None if layer_backward is None else layer_backward[i])
            layer_hidden.append(h)

        return layer_hidden, self.get_output_state(layer_hidden)

class StackedLSTMCell(StackedRNNCell):
    def __init__(self, input_dim, hidden_dim, n_layers, intermediate_dropout = 0.):
        super(StackedLSTMCell, self).__init__(RNNType.LSTM, input_dim, hidden_dim, n_layers, intermediate_dropout)

class StackedGRUCell(StackedRNNCell):
    def __init__(self, input_dim, hidden_dim, n_layers, intermediate_dropout = 0.):
        super(StackedGRUCell, self).__init__(RNNType.GRU, input_dim, hidden_dim, n_layers, intermediate_dropout)


if __name__ == '__main__':
    batch, dim, L = 5, 10, 2
    cell = StackedLSTMCell(dim, L, L)
    f_out = torch.randn(batch, dim).float()
    h, o = cell.init_hidden_states(f_out, None)

    assert o.size() == (batch, dim)
    assert len(h) == L

    x = torch.randn(batch, dim)

    hs = []
    T = 5
    for _ in range(T):
        h, o = cell(x, h)
        assert o.size() == (batch, dim)
        assert len(h) == L
        hs.append(h)

    weight: torch.Tensor = torch.randn(batch, T)
    weight = torch.nn.Softmax(dim=1)(weight)

    mh = cell.merge_hidden_list(hs, weight)




