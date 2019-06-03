from typing import Optional
import torch.nn

class StackedEncoder(torch.nn.Module):
    def __init__(self, encs, input_size, output_size, input_dropout=0.):
        super(StackedEncoder, self).__init__()

        self.layer_encs = torch.nn.ModuleList(encs)
        self.input_size = input_size
        self.output_size = output_size
        self.input_dropout = torch.nn.Dropout(input_dropout)

    def forward(self,
                inputs: torch.Tensor,
                mask: Optional[torch.LongTensor]):
        last_output = inputs
        layered_output = []
        for i, enc in enumerate(self.layer_encs):
            if i > 0:
                last_output = self.input_dropout(last_output)

            last_output = enc(last_output, mask)
            layered_output.append(last_output)

        enc_output = layered_output[-1]
        return enc_output, layered_output

    def get_layer_num(self):
        return len(self.layer_encs)

    def get_input_dim(self):
        return self.input_size

    def get_output_dim(self):
        return self.output_size

    def is_bidirectional(self) -> bool:
        return self.layer_encs[-1].is_bidirectional()

