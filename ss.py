import os
import cv2
import subprocess
import streamlit as st
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import math

# Import all models
from load_gan import (
    load_male_to_female_gan,
    load_female_to_male_gan,
    load_gender_identifier,
    load_skin_tone_autoencoder,
    load_eyeglasses_autoencoder,
    load_smiling_autoencoder,
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load all models
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

# Helper function to create necessary folders
def create_user_folders(base_dir, user_name):
    user_dir = base_dir / user_name
    input_dir = user_dir / "input_data"
    output_dir = user_dir / "output_data"
    frames_dir = user_dir / "extracted_frames"
    rendered_dir = user_dir / "rendered_frames"
    for folder in [input_dir, output_dir, frames_dir, rendered_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir, frames_dir, rendered_dir

# Function to extract frames from video
def extract_frames(video_path, frames_dir):
    cmd = f"ffmpeg -i {video_path} -q:v 1 {frames_dir}/frame_%04d.png"
    subprocess.run(cmd, shell=True, check=True)

# Function to apply AnimeGANv3 rendering
def apply_animegan(frames_dir, rendered_dir):
    script_path = "C:/Users/anees/dataset/AnimeGANv3/tools/video2anime.py"
    model_path = "C:/Users/anees/dataset/AnimeGANv3/deploy/AnimeGANv3_Hayao_36.onnx"
    for frame in sorted(os.listdir(frames_dir)):
        input_frame = os.path.join(frames_dir, frame)
        output_frame = os.path.join(rendered_dir, frame)
        if not os.path.exists(output_frame):  # Skip already processed frames
            cmd = f"python {script_path} -i {input_frame} -o {output_frame} -m {model_path}"
            subprocess.run(cmd, shell=True, check=True)

# Function to generate video from frames
def create_video(frames_dir, output_video_path, fps=30):
    cmd = f"ffmpeg -r {fps} -i {frames_dir}/frame_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {output_video_path}"
    subprocess.run(cmd, shell=True, check=True)

# App title
st.title("Reenactment Streamlit App")

# Sidebar for model selection
reenactment_option = st.sidebar.selectbox(
    "Choose Reenactment Type",
    ["Image Reenactment", "Video Reenactment"]
)

# Input field for user name
user_name = st.text_input("Enter your name:")
BASE_DIR = Path("C:/Users/anees/dataset/user_data")

if user_name and reenactment_option:
    input_dir, output_dir, frames_dir, rendered_dir = create_user_folders(BASE_DIR, user_name)

    if reenactment_option == "Image Reenactment":
        st.header("GAN and Autoencoder Image Reenactment")
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
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            input_image = Image.open(uploaded_file).convert("RGB")
            st.image(input_image, caption="Uploaded Image", use_column_width=True)
            input_tensor = transform(input_image).unsqueeze(0).to(device)

            with torch.no_grad():
                if model_option == "Male to Female GAN":
                    output_tensor = male_to_female_model(input_tensor)
                elif model_option == "Female to Male GAN":
                    output_tensor = female_to_male_model(input_tensor)
                elif model_option == "Gender Identifier":
                    output_tensor = gender_identifier_model(input_tensor)
                    prediction = "Male" if output_tensor.item() < 0.5 else "Female"
                    st.write(f"Gender Prediction: {prediction}")
                elif model_option == "Skin Tone Enhancement Autoencoder":
                    output_tensor = skin_tone_model(input_tensor)
                elif model_option == "Eyeglasses Add/Remove Autoencoder":
                    output_tensor = eyeglasses_model(input_tensor)
                elif model_option == "Smiling Autoencoder":
                    output_tensor = smiling_model(input_tensor)

            if model_option != "Gender Identifier":
                output_image = reverse_transform(output_tensor.squeeze(0).cpu())
                st.image(output_image, caption="Processed Image", use_column_width=True)
                output_path = output_dir / "processed_image.png"
                output_image.save(output_path)
                with open(output_path, "rb") as file:
                    st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

    elif reenactment_option == "Video Reenactment":
        st.header("AnimeGANv3 Video Stylization")
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        if uploaded_video:
            video_path = input_dir / uploaded_video.name
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())
            st.write(f"Video uploaded to {video_path}")

            st.write("Extracting frames...")
            extract_frames(video_path, frames_dir)
            st.success("Frames extracted.")

            st.write("Applying AnimeGANv3 rendering...")
            apply_animegan(frames_dir, rendered_dir)
            st.success("Rendering completed.")

            output_video_path = output_dir / f"{uploaded_video.name}_stylized.mp4"
            create_video(rendered_dir, output_video_path)
            st.success("Video stylized and saved.")

            st.video(str(output_video_path))
            with open(output_video_path, "rb") as file:
                st.download_button("Download Stylized Video", file, file_name=output_video_path.name, mime="video/mp4")
