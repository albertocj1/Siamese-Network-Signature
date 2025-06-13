import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# Define the SiameseNetwork class
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=9, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(20, 40, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(40, 80, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(80, 160, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(p=0.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(2560, 640),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(640, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(160, 40),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(40, 10)
        )

    def forward_once(self, x):
        x = self.cnn1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Load the model
@st.cache_resource
def load_model(model_path):
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load trained model
model = load_model("models/best_model.pt")

# Image preprocessing
TARGET_SIZE = (270, 650)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Get image embedding
def get_embedding(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model.forward_once(image)
    return embedding

# Streamlit UI
st.title("Signature Verification App")
st.write("Upload two signature images to verify if they are from the same person.")

uploaded_file1 = st.file_uploader("Choose the first signature image...", type=["png", "jpg", "jpeg", "tif"])
uploaded_file2 = st.file_uploader("Choose the second signature image...", type=["png", "jpg", "jpeg", "tif"])

# Default threshold (no slider)
distance_threshold = 0.5

if uploaded_file1 and uploaded_file2:
    image1 = Image.open(uploaded_file1).convert("L")
    image2 = Image.open(uploaded_file2).convert("L")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, caption="Image 1", use_container_width=True)
    with col2:
        st.image(image2, caption="Image 2", use_container_width=True)

    st.subheader("Verification Result:")

    embedding1 = get_embedding(image1)
    embedding2 = get_embedding(image2)
    euclidean_distance = torch.pairwise_distance(embedding1, embedding2).item()

    st.write(f"Euclidean Distance: {euclidean_distance:.4f}")

    if euclidean_distance <= distance_threshold:
        st.success("✅ The signatures are likely GENUINE.")
    else:
        st.error("❌ The signatures are likely FORGED.")
