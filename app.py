import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Page Config & UI Warning
st.set_page_config(page_title="Corn Disease Predictor", page_icon="🌽")
st.title("🌽 Corn Disease Classifier")
st.warning("⚠️ **Crucial Instruction:** Please center the diseased spot directly in the middle of your photo, filling the frame as much as possible. Do not scan large areas of healthy green leaves.")

# 2. Setup the Model Architecture & Load Weights
@st.cache_resource 
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 5)
    
    model.load_state_dict(torch.load("Copy of resnet18_corn_disease_final.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# 3. Image Transforms (Includes the CenterCrop engineering fix)
predict_transform = transforms.Compose([
    transforms.Resize(256),         
    transforms.CenterCrop(224),     
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])
])

class_names = ['Common Rust', 'Corn Leaf Blight', 'Gray Leaf Spot', 'Healthy', 'Insects damage']

# 4. The File Uploader UI
uploaded_file = st.file_uploader("Upload a picture of the corn leaf...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Leaf", use_container_width=True)
    
    st.write("Diagnosing...")
    
    # 5. Process and Predict
    input_tensor = predict_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        
        # Calculate confidence percentage
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_disease = class_names[predicted_idx.item()]
        confidence_score = confidence.item() * 100
        
    st.success(f"**Diagnosis:** {predicted_disease}")
    st.info(f"**Confidence:** {confidence_score:.2f}%")