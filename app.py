import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
from collections import OrderedDict

class_names = ['unknown', 'главная дорога', 'уступи дорогу', 'стоп', 'нет въезда']  

state_dict = torch.load('model.pth', map_location=torch.device('cpu'))

new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if k.startswith('fc.1'):
        new_key = k.replace('fc.1', 'fc')
    else:
        new_key = k
    new_state_dict[new_key] = v

model = models.resnet34()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))  
model.load_state_dict(new_state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Traffic Sign Recognition")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        class_name = class_names[predicted.item()]
        st.write(f'Predicted Class: {class_name}')
