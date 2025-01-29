import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import csv
from datetime import datetime

# --------------------
# Generator
# --------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# --------------------
# Discriminator
# --------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x, y):
        combined = torch.cat([x, y], dim=1)
        return self.model(combined).view(-1, 1)

# --------------------
# Dataset Loader
# --------------------
class FemaleMaleDataset(Dataset):
    def __init__(self, female_dir, male_dir, transform=None):
        self.female_images = sorted(os.listdir(female_dir))
        self.male_images = sorted(os.listdir(male_dir))
        self.female_dir = female_dir
        self.male_dir = male_dir
        self.transform = transform

    def __len__(self):
        return min(len(self.female_images), len(self.male_images))

    def __getitem__(self, idx):
        female_img = self.load_image(os.path.join(self.female_dir, self.female_images[idx]))
        male_img = self.load_image(os.path.join(self.male_dir, self.male_images[idx]))
        return female_img, male_img

    def load_image(self, path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# --------------------
# Training Function
# --------------------
def train(female_dir, male_dir, output_dir, log_dir, epochs=1000, batch_size=16, lr=0.0002, image_size=(176, 218)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"trail2_female_to_male_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Batch", "Loss D", "Loss G", "PSNR", "SSIM"])

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.5)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=20, gamma=0.5)

    transform = Compose([
        Resize(image_size),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = FemaleMaleDataset(female_dir, male_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    try:
        for epoch in range(epochs):
            for i, (female, male) in enumerate(dataloader):
                female, male = female.to(device), male.to(device)

                # Smooth labels
                real_labels = torch.full((female.size(0), 1), 0.9, device=device)
                fake_labels = torch.full((female.size(0), 1), 0.1, device=device)

                # Train Discriminator
                optimizer_d.zero_grad()
                fake_male = generator(female).detach()
                real_loss = criterion_gan(discriminator(female, male), real_labels)
                fake_loss = criterion_gan(discriminator(female, fake_male), fake_labels)
                loss_d = (real_loss + fake_loss) / 2
                loss_d.backward()
                optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()
                fake_male = generator(female)
                gan_loss = criterion_gan(discriminator(female, fake_male), real_labels)
                l1_loss = criterion_l1(fake_male, male) * 100
                loss_g = gan_loss + l1_loss
                loss_g.backward()
                optimizer_g.step()

                # Metrics
                with torch.no_grad():
                    mse = torch.mean((fake_male - male) ** 2)
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
                    ssim_value = ssim(
                        fake_male[0].cpu().permute(1, 2, 0).detach().numpy(),
                        male[0].cpu().permute(1, 2, 0).detach().numpy(),
                        multichannel=True,
                    )

                # Log Results
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, i + 1, loss_d.item(), loss_g.item(), psnr, ssim_value])

                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(dataloader)}], "
                      f"Loss D: {loss_d:.4f}, Loss G: {loss_g:.4f}, PSNR: {psnr:.2f}, SSIM: {ssim_value:.2f}")

            scheduler_g.step()
            scheduler_d.step()

    except KeyboardInterrupt:
        print("Training interrupted. Saving the current state of models...")
        torch.save(generator.state_dict(), os.path.join(output_dir, "generator_interrupted.pth"))
        torch.save(discriminator.state_dict(), os.path.join(output_dir, "discriminator_interrupted.pth"))
        print("Models saved successfully!")

    torch.save(generator.state_dict(), os.path.join(output_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(output_dir, "discriminator.pth"))

if __name__ == "__main__":
    train(
        female_dir="C:/Users/anees/dataset/custom_gans/female_male_gan_project/datasets/train/female",
        male_dir="C:/Users/anees/dataset/custom_gans/female_male_gan_project/datasets/train/male",
        output_dir="C:/Users/anees/dataset/custom_gans/female_male_gan_project/output",
        log_dir="C:/Users/anees/dataset/custom_gans/female_male_gan_project/logs",
    )
