import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from skimage.metrics import structural_similarity as ssim

# --------------------
# Dataset Loader
# --------------------
class GenderDataset(Dataset):
    def __init__(self, male_dir, female_dir, transform=None):
        self.male_images = sorted(os.listdir(male_dir))
        self.female_images = sorted(os.listdir(female_dir))
        self.male_dir = male_dir
        self.female_dir = female_dir
        self.transform = transform

    def __len__(self):
        return len(self.male_images) + len(self.female_images)

    def __getitem__(self, idx):
        if idx < len(self.male_images):
            img_path = os.path.join(self.male_dir, self.male_images[idx])
            label = 0  # Male
        else:
            img_path = os.path.join(self.female_dir, self.female_images[idx - len(self.male_images)])
            label = 1  # Female
        img = self.load_image(img_path)
        return img, torch.tensor(label, dtype=torch.float32)

    def load_image(self, path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# --------------------
# ResNet Architecture
# --------------------
class GenderResNet(nn.Module):
    def __init__(self):
        super(GenderResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# --------------------
# Evaluation Metrics
# --------------------
def calculate_psnr_and_mse(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2).item()
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse != 0 else float("inf")
    return psnr, mse

# --------------------
# Training Function
# --------------------
def train_gender_identifier(output_dir, log_dir, male_dir, female_dir, batch_size=32, lr=0.0001, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data augmentation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # DataLoader
    dataset = GenderDataset(male_dir, female_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    model = GenderResNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Prepare log file
    log_file = os.path.join(log_dir, "metrics.csv")
    with open(log_file, "w") as f:
        f.write("Epoch,Loss,PSNR,MSE\n")

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0
        epoch_mse = 0

        for idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            psnr, mse = calculate_psnr_and_mse(labels, outputs)
            epoch_loss += loss.item()
            epoch_psnr += psnr
            epoch_mse += mse

        avg_loss = epoch_loss / len(dataloader)
        avg_psnr = epoch_psnr / len(dataloader)
        avg_mse = epoch_mse / len(dataloader)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}, MSE: {avg_mse:.4f}")

        # Save metrics
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.4f},{avg_psnr:.2f},{avg_mse:.4f}\n")

        # Save intermediate model
        torch.save(model.state_dict(), os.path.join(output_dir, f"resnet_epoch_{epoch+1}.pth"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "resnet_final.pth"))

# --------------------
# Entry Point
# --------------------
if __name__ == "__main__":
    train_gender_identifier(
        output_dir=r"C:\Users\anees\dataset\custom_gans\gender_identifier\outputs",
        log_dir=r"C:\Users\anees\dataset\custom_gans\gender_identifier\logs",
        male_dir=r"C:\Users\anees\dataset\custom_gans\gender_identifier\dataset\male",
        female_dir=r"C:\Users\anees\dataset\custom_gans\gender_identifier\dataset\female",
        batch_size=32,
        lr=0.0003,
        epochs=20
    )
