import os
import zipfile
import subprocess
from pathlib import Path
from PIL import Image
import torch
import streamlit as st
from torchvision import transforms
import numpy as np

# Import all models dynamically
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

# Base paths
MODEL_DIR = Path("C:/Users/anees/dataset/custom_gans")
USER_DATA_DIR = MODEL_DIR / "user_data"
ANIMEGAN_SCRIPT = MODEL_DIR / "AnimeGANv3/tools/video2anime.py"
ANIMEGAN_MODEL = MODEL_DIR / "AnimeGANv3/deploy/AnimeGANv3_Hayao_36.onnx"

# Load pre-trained models
male_to_female_model = load_male_to_female_gan(
    str(MODEL_DIR / "male_female_gan_project/celeba output/generator.pth"), device
)
female_to_male_model = load_female_to_male_gan(
    str(MODEL_DIR / "female_male_gan_project/celeba output/generator.pth"), device
)
gender_identifier_model = load_gender_identifier(
    str(MODEL_DIR / "gender_identifier/outputs 5/resnet_final.pth"), device
)
skin_tone_model = load_skin_tone_autoencoder(
    str(MODEL_DIR / "Skin_Tone_Enhancement/outputs/skin_tone_autoencoder.pth"), device
)
eyeglasses_model = load_eyeglasses_autoencoder(
    str(MODEL_DIR / "Eyeglasses_Reconstruction/outputs/eyeglasses_autoencoder.pth"), device
)
smiling_model = load_smiling_autoencoder(
    str(MODEL_DIR / "Smiling_Reconstruction/outputs/smiling_autoencoder.pth"), device
)

# Image transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

reverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    transforms.ToPILImage(),
])

# Helper functions
def create_user_dirs(user_name):
    """Create user-specific directories dynamically."""
    user_dir = USER_DATA_DIR / user_name
    dataset_dir = user_dir / "dataset"
    output_dir = user_dir / "outputs"
    logs_dir = user_dir / "logs"
    for folder in [dataset_dir, output_dir, logs_dir]:
        folder.mkdir(exist_ok=True, parents=True)
    return dataset_dir, output_dir, logs_dir

def extract_dataset(zip_file, target_dir):
    """Extract ZIP dataset to the target directory."""
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(target_dir)

def train_model(script_path, dataset_path, logs_dir, epochs, lr, batch_size):
    """Execute model training script."""
    command = [
        "python", script_path,
        "--data_dir", str(dataset_path),
        "--logs_dir", str(logs_dir),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch_size", str(batch_size)
    ]
    subprocess.run(command, check=True)

# Metrics calculation
def calculate_metrics(original, generated):
    mse = torch.mean((original - generated) ** 2).item()
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float("inf")
    return psnr, mse

# Streamlit app
st.title("Reenactment Application with GANs and Autoencoders")

mode = st.sidebar.selectbox("Choose Mode", ["Image Reenactment", "Video Reenactment", "Developer Mode"])

if mode == "Image Reenactment":
    st.header("GAN and Autoencoder Image Processing")
    model_option = st.sidebar.selectbox(
        "Select Model",
        ["Male to Female GAN", "Female to Male GAN", "Gender Identifier",
         "Skin Tone Enhancement", "Eyeglasses Add/Remove", "Smiling Autoencoder"]
    )
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            if model_option == "Male to Female GAN":
                output_tensor = male_to_female_model(tensor)
            elif model_option == "Female to Male GAN":
                output_tensor = female_to_male_model(tensor)
            elif model_option == "Gender Identifier":
                output_tensor = gender_identifier_model(tensor)
                prediction = "Male" if output_tensor.item() < 0.5 else "Female"
                st.write(f"Predicted Gender: {prediction}")
            elif model_option == "Skin Tone Enhancement":
                output_tensor = skin_tone_model(tensor)
            elif model_option == "Eyeglasses Add/Remove":
                output_tensor = eyeglasses_model(tensor)
            elif model_option == "Smiling Autoencoder":
                output_tensor = smiling_model(tensor)

            if model_option != "Gender Identifier":
                output_image = reverse_transform(output_tensor.squeeze(0).cpu())
                st.image(output_image, caption="Processed Image")
                psnr, mse = calculate_metrics(tensor.cpu(), output_tensor.cpu())
                st.write(f"PSNR: {psnr:.2f}, MSE: {mse:.4f}")

elif mode == "Video Reenactment":
    st.header("Video Processing with AnimeGANv3")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        video_path = USER_DATA_DIR / uploaded_video.name
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        st.write(f"Uploaded video saved at {video_path}")

        st.write("Extracting frames...")
        frames_dir = video_path.parent / "frames"
        frames_dir.mkdir(exist_ok=True)
        cmd = f"ffmpeg -i {video_path} -q:v 1 {frames_dir}/frame_%04d.png"
        subprocess.run(cmd, shell=True, check=True)
        st.success("Frames extracted.")

        st.write("Applying AnimeGANv3 rendering...")
        rendered_dir = frames_dir.parent / "rendered_frames"
        rendered_dir.mkdir(exist_ok=True)
        for frame in sorted(os.listdir(frames_dir)):
            input_frame = os.path.join(frames_dir, frame)
            output_frame = os.path.join(rendered_dir, frame)
            cmd = f"python {ANIMEGAN_SCRIPT} -i {input_frame} -o {output_frame} -m {ANIMEGAN_MODEL}"
            subprocess.run(cmd, shell=True, check=True)
        st.success("Rendering completed.")

        output_video_path = video_path.parent / f"{uploaded_video.name}_stylized.mp4"
        cmd = f"ffmpeg -r 30 -i {rendered_dir}/frame_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {output_video_path}"
        subprocess.run(cmd, shell=True, check=True)
        st.success(f"Stylized video saved at {output_video_path}")

        st.video(str(output_video_path))

elif mode == "Developer Mode":
    st.header("Train Your Own Model")

    # Step 1: Enter User Name
    user_name = st.text_input("Enter your name:")
    if user_name:
        dataset_dir, output_dir, logs_dir = create_user_dirs(user_name)

        # Step 2: Upload Dataset
        uploaded_zip = st.file_uploader("Upload Dataset (ZIP)", type="zip")
        if uploaded_zip:
            zip_path = dataset_dir / "uploaded_dataset.zip"
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())
            extract_dataset(zip_path, dataset_dir)
            st.success("Dataset uploaded and extracted successfully.")

            # Step 3: Dataset Attribute Handling (if applicable)
            st.write("If your dataset includes attributes (e.g., labels or metadata), upload the file:")
            attribute_file = st.file_uploader("Upload Attribute File (Optional)", type=["csv", "json"])
            if attribute_file:
                attr_path = dataset_dir / attribute_file.name
                with open(attr_path, "wb") as f:
                    f.write(attribute_file.read())
                st.success(f"Attribute file {attribute_file.name} uploaded successfully.")

            # Step 4: Create Directories for Male and Female (if needed)
            if st.checkbox("Split Dataset by Gender"):
                male_dir = dataset_dir / "male"
                female_dir = dataset_dir / "female"
                male_dir.mkdir(exist_ok=True)
                female_dir.mkdir(exist_ok=True)
                st.success(f"Created directories: {male_dir} and {female_dir}")

            # Step 5: Model Selection and Training Parameters
            model_type = st.selectbox("Select Model Type", ["GAN", "Autoencoder"])
            epochs = st.number_input("Epochs", min_value=1, value=10)
            lr = st.number_input("Learning Rate", min_value=0.0001, value=0.001)
            batch_size = st.number_input("Batch Size", min_value=1, value=16)

            # Step 6: Train Model
            if st.button("Train Model"):
                training_script = BASE_DIR / f"train_{model_type.lower()}.py"
                try:
                    train_model(training_script, dataset_dir, logs_dir, epochs, lr, batch_size)
                    st.success("Model Training Completed.")
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")

            # Step 7: Test Trained Model
            if st.button("Test Trained Model"):
                st.write("Upload an image to test your trained model:")
                test_image = st.file_uploader("Upload Test Image", type=["jpg", "png", "jpeg"])
                if test_image:
                    image = Image.open(test_image).convert("RGB")
                    st.image(image, caption="Uploaded Image")

                    # Load the trained model for testing
                    trained_model_path = logs_dir / f"{model_type.lower()}_trained_model.pth"
                    if trained_model_path.exists():
                        st.write(f"Using model: {trained_model_path}")
                        # Load the model dynamically based on type
                        model = None
                        if model_type == "GAN":
                            model = torch.load(trained_model_path, map_location=device)  # Replace with actual GAN model loading
                        elif model_type == "Autoencoder":
                            model = torch.load(trained_model_path, map_location=device)  # Replace with actual Autoencoder model loading

                        if model:
                            tensor = transform(image).unsqueeze(0).to(device)
                            with torch.no_grad():
                                output_tensor = model(tensor)
                                output_image = reverse_transform(output_tensor.squeeze(0).cpu())
                                st.image(output_image, caption="Test Output")
                                psnr, mse = calculate_metrics(tensor.cpu(), output_tensor.cpu())
                                st.write(f"PSNR: {psnr:.2f}, MSE: {mse:.4f}")
                    else:
                        st.error("Trained model not found. Please ensure the training is completed.")

