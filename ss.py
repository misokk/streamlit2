import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import requests
import os

def download_file_from_google_drive(file_id, destination):
    URL = f"https://drive.google.com/uc?id={file_id}"
    session = requests.Session()

    response = session.get(URL, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        URL = f"https://drive.google.com/uc?id={file_id}&confirm={token}"
        response = session.get(URL, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)

    print(f"파일 다운로드 완료: {destination}")

st.title("강아지와 고양이 분류기 🐶🐱")
st.write("이미지를 업로드하면 강아지인지 고양이인지 알려드립니다!")

file_id = "1G15KctjoC7rA2udqrBZs1DBBhf4QWDzU"  
destination = "dogcat.pth"

if not os.path.exists(destination):
    st.write("모델 파일을 다운로드 중입니다...")
    download_file_from_google_drive(file_id, destination)
    st.write("모델 파일 다운로드 완료!")

uploaded_file = st.file_uploader("이미지를 업로드하세요!", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)  
    model.load_state_dict(torch.load(destination, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB") 
    st.image(image, caption="업로드된 이미지", use_column_width=True)
    
    st.write("이미지를 분류하는 중입니다. 잠시만 기다려주세요!")
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
 
    class_names = ["고양이", "강아지"]
    confidence_scores = {class_names[i]: probabilities[i].item() * 100 for i in range(len(class_names))}
    
  
    predicted_class = max(confidence_scores, key=confidence_scores.get)
    st.write(f"이 이미지는 **{predicted_class}**로 분류되었습니다!")
    

    st.write("확신 정도:")
    for label, score in confidence_scores.items():
        st.write(f"{label}: {score:.2f}%")

    st.bar_chart(list(confidence_scores.values()), height=300)

else:
    st.write("이미지를 업로드하세요!")

 
    st.bar_chart(list(confidence_scores.values()), height=300)

else:
    st.write("이미지를 업로드하세요!")
