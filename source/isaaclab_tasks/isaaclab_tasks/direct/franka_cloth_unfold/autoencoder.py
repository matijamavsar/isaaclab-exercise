import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthAutoencoder(nn.Module):
    def __init__(self, device):
        super().__init__()

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 256 -> 128
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 64 -> 32
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 32 -> 16
            nn.ReLU(True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),# 16 -> 8
            nn.ReLU(True),
        ).to(device)

        # ---------- Bottleneck ----------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True)
        ).to(device)

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 8 -> 16
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16 -> 32
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 64 -> 128
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 128 -> 256
            nn.Sigmoid()  # or identity if depth is not scaled to [0,1]
        ).to(device)

    def forward(self, x):
        z = self.encoder(x)
        z = self.bottleneck(z)
        out = self.decoder(z)
        return out, z


# ---- Example usage ----
if __name__ == "__main__":
    model = DepthAutoencoder()
    x = torch.randn(1, 1, 256, 256)  # batch=1, channel=1
    y = model(x)
    print(y.shape)  # should be (1, 1, 256, 256)
