import torch
import torch.nn as nn


class AugmentModel(nn.Module):

    def __init__(self, model, input_dim, output_dim):
        super(AugmentedModel, self).__init__()
        self.model = model
        self.scale = torch.randn(output_dim)
        self.bias = torch.randn(ouptut_dim)
        self.weight = torch.randn((output_dim, input_dim)) 


    def forward(self, input):
        model_out = self.model(input)
        neuron_out = self.scale * torch.exp(input.matmul(self.weight.t()) + self.bias)

        return model_out + neuron_out


class AugmentLoss(nn.Module):

    def __init__(self, loss, lambda=0.1):
        assert lambda > 0, "Lambda in augmented loss must be non-negative"
        self.loss = loss
        self.lambda = lambda

    def forward(self, input, target, a):
        return self.loss(input, target) + self.lambda * torch.norm(a)

