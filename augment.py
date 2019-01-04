import torch
import torch.nn as nn


class AugmentModel(nn.Module):

    def __init__(self, model, input_dim, output_dim):
        super(AugmentModel, self).__init__()
        self.model = model
        self.scale = nn.Parameter(torch.ones(output_dim))
        self.linear = nn.Linear(input_dim, output_dim)


    def forward(self, input):
        model_out = self.model(input)
        neuron_out = self.scale * torch.exp(self.linear(input.view(input.size(0), -1)))
        return model_out + neuron_out


class AugmentLoss(nn.Module):

    def __init__(self, loss, model, l=0.1):
        super(AugmentLoss, self).__init__()
        assert l > 0, "Lambda in augmented loss must be non-negative"
        self.loss = loss
        self.model = model
        self.l = l


    def forward(self, input, target):
        return self.loss(input, target) + self.l * torch.norm(self.model.scale.data)

