import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # (BATCH_SIZE, 32, 64, 64)
        self.conv1 = nn.Conv2d(1, 32, 3, padding="same")
        # (BATCH_SIZE, 64, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # (BATCH_SIZE, 128, 16, 16)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # (BATCH_SIZE, 256, 8, 8)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # (BATCH_SIZE, 256, 4, 4)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(nn.functional.relu(self.conv2(x)))
        x = self.pool1(nn.functional.relu(self.conv3(x)))
        x = self.pool1(nn.functional.relu(self.conv4(x)))

        return self.pool1(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # (BATCH_SIZE, 256, 4, 4)
        self.conv1 = nn.ConvTranspose2d(
            256, 128, 3, padding=1)  # (BATCH_SIZE, 128, 8, 8)
        self.conv2 = nn.ConvTranspose2d(
            128, 64, 3, padding=1)  # (BATCH_SIZE, 64, 16, 16)
        self.conv3 = nn.ConvTranspose2d(
            64, 32, 3, padding=1)  # (BATCH_SIZE, 32, 32, 32)
        self.conv4 = nn.ConvTranspose2d(
            32, 1, 3, padding=1)  # (BATCH_SIZE, 1, 64, 64)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.interpolate(x, scale_factor=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.interpolate(x, scale_factor=2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.interpolate(x, scale_factor=2)
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.interpolate(x, scale_factor=2)

        return x


class ALSTME(nn.Module):
    def __init__(self):
        super(ALSTME, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.lstm = nn.LSTM(256*4*4, 256*4*4, batch_first=True)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        x, (hidden_state, cell_state) = self.lstm(x)
        x = self.decoder(x.view(-1, 256, 4, 4))

        return x
