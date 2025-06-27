
import torch.nn as nn
   

class LCNN(nn.Module):
    def __init__(self, opts):
        super(LCNN, self).__init__()

        self.conv = nn.Conv2d(3, 8, 3, 1, 1)
        self.pool = nn.MaxPool2d(8, 8)
        self.relu = nn.ReLU()
        self.fc   = nn.Linear(200, opts.num_classes)
        
    def forward(self, x):
        N, _, _, _ = x.shape

        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)

        x = x.view(N, -1)
        x = self.fc(x)

        return x