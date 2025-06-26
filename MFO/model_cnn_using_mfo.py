import torch
import torch.nn as nn
import math
import numpy as np

class SimpleCNN_CIFAR10_MFO(nn.Module):
    def __init__(self, num_class = 10):
        super().__init__()
        self.cnn1 = self._make_block_cnn(3,8)
        self.cnn2 = self._make_block_cnn(8,16)
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, num_class),
            nn.LeakyReLU(),
        )

    def _make_block_cnn(self,in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features= out_channel),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        # x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]) #flatten
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def mfo(function, dimension, n_moths, max_iter, lb, ub):
    moths_position = np.random.uniform(lb, ub, (n_moths, dimension))
    moths_fitness = np.array([function(i) for i in moths_position])
    best_moth = moths_position[np.argmin(moths_fitness)]
    best_score = np.min(moths_fitness)
    flame_num = n_moths
    b = 1

    for l in range(1, max_iter + 1):
        sorted_indices = np.argsort(moths_fitness)
        flames_position = moths_position[sorted_indices][:flame_num]

        for i in range(n_moths):
            t = np.random.uniform(-1, 1)
            distance = np.linalg.norm(moths_position[i] - flames_position[i % flame_num], ord=2)
            moths_position[i] = distance * math.exp(b * t) * math.cos(2 * math.pi * t) + flames_position[i % flame_num]
            moths_position[i] += np.random.normal(0, 0.001, dimension)

        moths_position = np.clip(moths_position, lb, ub)
        moths_fitness = np.array([function(i) for i in moths_position])

        if np.min(moths_fitness) < best_score:
            best_score = np.min(moths_fitness)
            best_moth = moths_position[np.argmin(moths_fitness)]

        flame_num = int(np.round(n_moths - (l * ((n_moths - 1) / max_iter))))

    return best_moth, best_score
