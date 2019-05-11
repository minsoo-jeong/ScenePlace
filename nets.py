from torchvision import models
import torch


class Resnet50(torch.nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        self.base = torch.nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.pool = torch.nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__=='__main__':
    print(Resnet50(47))