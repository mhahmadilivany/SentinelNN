import torch
import torch.nn as nn

class HardenedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.hardening_ratio = 0
        self.duplicated_channels = 0
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        super(HardenedConv2d, self).__init__(*args, **kwargs)

    def forward(self, 
                input_activation: torch.tensor) -> torch.tensor:
        # forward pass for the conv layer, includes duplicated channels
        out_activation = super(HardenedConv2d, self).forward(input_activation)

        # compares the cuplicated channles
        correction_mask = torch.ge(out_activation[:, :self.duplicated_channels, :, :], 
                                   out_activation[:, self.duplicated_channels:2*self.duplicated_channels, :, :])
        
        # corrects and merges duplicated channels for the output
        corrected_results = out_activation[:, :self.duplicated_channels] * torch.logical_not(correction_mask) + \
                 out_activation[:, self.duplicated_channels:2*self.duplicated_channels] * correction_mask
        
        batch, out_ch, w, h = out_activation.size()
        new_out_activation = torch.zeros((batch, out_ch - self.duplicated_channels, w, h), device=self.device)
        new_out_activation[:, :self.duplicated_channels] = corrected_results.detach()
        new_out_activation[:, self.duplicated_channels:]  = out_activation[:, 2*self.duplicated_channels:].detach()

        return new_out_activation    

    def __repr__(self):
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, duplicated_channels={self.duplicated_channels})"


class RangerReLU(nn.ReLU):
    def __init__(self, *args, **kwargs):
        super(RangerReLU, self).__init__(*args, **kwargs)
        self.clipping_threshold = 6

    def forward(self,
                input_activation: torch.tensor) -> torch.tensor:
        out_activation = super(RangerReLU, self).forward(input_activation)
        out_activation[out_activation > self.clipping_threshold] = 0
        return out_activation
    
    def __repr__(self):
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, threshold={self.clipping_threshold})"