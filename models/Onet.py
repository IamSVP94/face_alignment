from pathlib import Path

import torch
import torchvision
from torch import nn


class EuclideanLoss(torch.nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, predicted_landmarks, target_landmarks):
        return torch.sum(torch.abs(predicted_landmarks - target_landmarks))


class ONet(nn.Module):
    def __init__(self, num_points=68):
        super(ONet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Dropout(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(2, 2), padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 128, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )

        self.output = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2 * num_points),  # we need [x,y]*68 facial landmark localization only
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        out = self.output(out)
        return out


class ResNet18(nn.Module):
    """
        Parameters
        ----------
        pretrained_weights : bool, default = "True"
            Ways of weights initialization.
            If "False", it means random initialization and no pretrained weights,
            If "True" it means resnet34 pretrained weights are used.

        fine_tune: bool, default = "False"
            Allows to choose between two types of transfer learning: fine tuning and feature extraction.
            For more details of the description of each mode,
            read https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

        embedding_size: int, default = 128
            Size of the embedding of the last layer

    """

    def __init__(self, pretrained_weights=None, fine_tune=True, num_points=68):
        super(ResNet18, self).__init__()
        self.pretrained_weights = pretrained_weights
        self.fine_tune = fine_tune

        if self.pretrained_weights:
            if Path(self.pretrained_weights).exists():
                state_dict = torch.load(str(self.pretrained_weights))['state_dict']
                remove_prefix = 'model.'
                state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in
                              state_dict.items()}
                pretrained_model = torchvision.models.resnet18(weights=state_dict)
            else:
                pretrained_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        else:
            pretrained_model = torchvision.models.resnet18(weights=None)

        if not self.fine_tune:
            for param in pretrained_model.parameters():
                param.requires_grad = False

        pretrained_model.fc = torch.nn.Linear(pretrained_model.fc.in_features, num_points * 2)
        pretrained_model = pretrained_model.type(torch.FloatTensor)
        self.model = pretrained_model

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':  # testing
    x = torch.rand((2, 3, 62, 62))
    for model in [ONet(), ResNet18()]:
        out = model(x)
        print(x.shape, out.shape)
