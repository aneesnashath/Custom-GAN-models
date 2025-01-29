import os
import torch
from male_to_female_gan import Generator as MaleToFemaleGenerator
from female_to_male_gan import Generator as FemaleToMaleGenerator
from gender_identifier import GenderResNet
from skin_tone_autoencoder import SkinToneAutoencoder
from eyeglasses_autoencoder import EyeglassesAutoencoder
from smiling_autoencoder import SmilingAutoencoder

# Helper function to check if a model file exists
def validate_model_path(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

# 1. Load Male-to-Female GAN
def load_male_to_female_gan(model_path, device):
    validate_model_path(model_path)
    model = MaleToFemaleGenerator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 2. Load Female-to-Male GAN
def load_female_to_male_gan(model_path, device):
    validate_model_path(model_path)
    model = FemaleToMaleGenerator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 3. Load Gender Identifier Model
def load_gender_identifier(model_path, device):
    validate_model_path(model_path)
    model = GenderResNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 4. Load Skin Tone Autoencoder
def load_skin_tone_autoencoder(model_path, device):
    validate_model_path(model_path)
    model = SkinToneAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 5. Load Eyeglasses Autoencoder
def load_eyeglasses_autoencoder(model_path, device):
    validate_model_path(model_path)
    model = EyeglassesAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 6. Load Smiling Autoencoder
def load_smiling_autoencoder(model_path, device):
    validate_model_path(model_path)
    model = SmilingAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
