import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np


# --------------------
# Dataset Loader
# --------------------
class EyeglassesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = [os.path.join(data_dir, img) for img in os.listdir(data_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img  # Input and target are the same for autoencoders


# --------------------
# Autoencoder Architecture
# --------------------
class EyeglassesAutoencoder(nn.Module):
    def __init__(self):
        super(EyeglassesAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# --------------------
# Metrics Calculation
# --------------------
def calculate_metrics(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2).item()
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse != 0 else float("inf")
    return psnr, mse


# --------------------
# Training Function
# --------------------
def train_eyeglasses_autoencoder(
    data_dir, 
    output_dir, 
    log_dir, 
    batch_size=32, 
    lr=0.001, 
    epochs=50, 
    model_save_path=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = EyeglassesDataset(data_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    model = EyeglassesAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Training loop
    log_file = os.path.join(log_dir, "training_log.csv")
    with open(log_file, "w") as f:
        f.write("Epoch,Loss,PSNR,MSE\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        # Evaluate metrics
        with torch.no_grad():
            psnr, mse = calculate_metrics(images, outputs)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, PSNR: {psnr:.2f}, MSE: {mse:.4f}")

        # Log metrics
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_loss},{psnr},{mse}\n")

        # Save sample output
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            save_image(outputs, os.path.join(output_dir, f"reconstructed_epoch_{epoch+1}.png"))

    # Save the model
    model_save_path = model_save_path or os.path.join(output_dir, "eyeglasses_autoencoder.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")


# --------------------
# Model Loading for Inference
# --------------------
def load_eyeglasses_autoencoder(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = EyeglassesAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# --------------------
# Entry Point
# --------------------
if __name__ == "__main__":
    train_eyeglasses_autoencoder(
        data_dir=r"C:\Users\anees\dataset\custom_gans\male_female_gan_project\raw images\img_align_celeba",
        output_dir=r"C:\Users\anees\dataset\custom_gans\Eyeglasses_Reconstruction\outputs",
        log_dir=r"C:\Users\anees\dataset\custom_gans\Eyeglasses_Reconstruction\logs",
        batch_size=32,
        lr=0.003,
        epochs=20
    )
