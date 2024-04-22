import torch
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
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(2, 2), padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.LeakyReLU(),
        )

        self.output = nn.Linear(256, 2 * num_points)  # we need [x,y]*68 facial landmark localization only

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        out = self.output(out)
        return out


if __name__ == '__main__':  # testing
    model = ONet()
    x = torch.rand((2, 3, 62, 62))
    out = model(x)
    print(x.shape, out.shape)
