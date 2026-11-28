# Train dùng cái jupyter notebook, file này dùng cho cái app (ko phải train)

import torch
import torch.nn as nn


class MyCNN(nn.Module):
    """
    Conv1 -> pool -> conv2 -> pool
    -> <flatten>
    -> FC1 -> FC2 -> FC3

    inputSize = 64 * 64 * 3

    activation là ReLU
    """

    def __init__(self, num_classes=5):
        super().__init__()

        # Tính size sau khi conv: (input - kernel + 2*padding) / stride + 1

        self.features = nn.Sequential(
            # Conv1 -> pool -> conv2 -> pool
            # input là 3 màu RGB, 16 filter size 3x3
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),  # [16, 62, 62]
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2
            ),  # quét kernel 2x2 lấy max trong 4 giá trị, [16, 31, 31]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),  # [32, 27, 27]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [32, 13, 13]
        )

        # <flatten trước khi vào FC>

        self.classifier = nn.Sequential(
            # FC1 -> FC2 -> FC3
            nn.Linear(in_features=32 * 13 * 13, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_classes),
        )

    def forward(self, x):
        # Conv1 -> pool -> conv2 -> pool
        x = self.features(x)

        # flatten từ 2d thành 1d
        x = torch.flatten(input=x, start_dim=1)

        # FC1 -> FC2 -> FC3
        x = self.classifier(x)
        return x
