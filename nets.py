from torchvision import models
import torch


class Resnet50(torch.nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        self.base = torch.nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.pool = torch.nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512,47)
        )
    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__=='__main__':
    print(Resnet50(47))