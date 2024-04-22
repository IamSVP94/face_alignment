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
        )  # 23x23x32

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
        )  # 10x10x64

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )  # 4x4x64

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(2, 2), padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
        )  # 3x3x128

        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.LeakyReLU(),
        )  # 256

        self.output = nn.Linear(256, 2 * num_points)  # we need [x,y]*68 facial landmark localization only

    def forward(self, x):  # 48x48x3
        out = self.conv1(x)  # 23x23x32
        out = self.conv2(out)  # 10x10x64
        out = self.conv3(out)  # 4x4x64
        out = self.conv4(out)  # 3x3x128
        out = torch.flatten(out, start_dim=1)  # 1152
        out = self.fc(out)  # 256
        out = self.output(out)  # 2*68
        return out


if __name__ == '__main__':  # testing
    model = ONet()
    x = torch.rand((2, 3, 48, 48))
    out = model(x)
    print(x.shape, out.shape)
