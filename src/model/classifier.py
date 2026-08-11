import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(
        self,
        d_model,
        classifier_dim,
        n_class,
    ):
        super().__init__()
        self.d_model = d_model
        self.classifier_dim = classifier_dim
        self.n_class = n_class
        self.classifier = nn.ModuleList(
            [
                nn.Linear(self.d_model, classifier_dim * self.d_model),
                nn.ReLU(),
                nn.Linear(classifier_dim * self.d_model, self.n_class),
            ]
        )

    def forward(self, x):
        x = x[
            :, -self.n_class :, :
        ]  # [B, n_class(사람), n_class(색깔)] -> [B, n_class, n_class]
        for layer in self.classifier:
            x = layer(x)
        return x
