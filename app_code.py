import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import os

# Load all models
from load_gan import (
    load_male_to_female_gan,
    load_female_to_male_gan,
    load_gender_identifier,
    load_skin_tone_autoencoder,
    load_eyeglasses_autoencoder,
    load_smiling_autoencoder,
)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
male_to_female_model = load_male_to_female_gan(r"C:\Users\anees\dataset\custom_gans\male_female_gan_project\celeba output\generator.pth", device)
female_to_male_model = load_female_to_male_gan(r"C:\Users\anees\dataset\custom_gans\female_male_gan_project\celeba output\generator.pth", device)
gender_identifier_model = load_gender_identifier(r"C:\Users\anees\dataset\custom_gans\gender_identifier\outputs 5\resnet_final.pth", device)
skin_tone_model = load_skin_tone_autoencoder(r"C:\Users\anees\dataset\custom_gans\Skin_Tone_Enhancement\outputs\skin_tone_autoencoder.pth", device)
eyeglasses_model = load_eyeglasses_autoencoder(r"C:\Users\anees\dataset\custom_gans\Eyeglasses_Reconstruction\outputs\eyeglasses_autoencoder.pth", device)
smiling_model = load_smiling_autoencoder(r"C:\Users\anees\dataset\custom_gans\Smiling_Reconstruction\outputs\smiling_autoencoder.pth", device)

# Transform for preprocessing images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Reverse transform for displaying output images
reverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    transforms.ToPILImage()
])

# App title
st.title("GAN and Autoencoder Streamlit App")

# Sidebar for model selection
model_option = st.sidebar.selectbox(
    "Choose a Model",
    [
        "Male to Female GAN",
        "Female to Male GAN",
        "Gender Identifier",
        "Skin Tone Enhancement Autoencoder",
        "Eyeglasses Add/Remove Autoencoder",
        "Smiling Autoencoder"
    ]
)

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # Process based on selected model
    if model_option == "Male to Female GAN":
        with torch.no_grad():
            output_tensor = male_to_female_model(input_tensor)
    elif model_option == "Female to Male GAN":
        with torch.no_grad():
            output_tensor = female_to_male_model(input_tensor)
    elif model_option == "Gender Identifier":
        with torch.no_grad():
            output_tensor = gender_identifier_model(input_tensor)
        prediction = "Male" if output_tensor.item() < 0.5 else "Female"
        st.write(f"Gender Prediction: {prediction}")
    elif model_option == "Skin Tone Enhancement Autoencoder":
        with torch.no_grad():
            output_tensor = skin_tone_model(input_tensor)
    elif model_option == "Eyeglasses Add/Remove Autoencoder":
        with torch.no_grad():
            output_tensor = eyeglasses_model(input_tensor)
    elif model_option == "Smiling Autoencoder":
        with torch.no_grad():
            output_tensor = smiling_model(input_tensor)

    # Post-process and display the output image for all except Gender Identifier
    if model_option != "Gender Identifier":
        output_image = reverse_transform(output_tensor.squeeze(0).cpu())
        st.image(output_image, caption="Processed Image", use_column_width=True)

    # Download button for output image
    if model_option != "Gender Identifier":
        output_path = "output_image.png"
        output_image.save(output_path)
        with open(output_path, "rb") as file:
            btn = st.download_button(
                label="Download Processed Image",
                data=file,
                file_name="processed_image.png",
                mime="image/png"
            )
