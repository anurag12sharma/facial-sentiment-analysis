import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import os

# Load text emotion model
pipe_lr = joblib.load(open("Models/text_emotion.pkl", "rb"))

# Emotion dictionary
emotions_emoji_dict = {"anger": "😠", "neutral": "😐", "sadness": "😔", "surprise": "😮", "joy": "😊", "fear": "😨", "disgust": "🤮", "shame": "😞"}


def predict_emotions(docx):
    return pipe_lr.predict([docx])[0]

def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])

# Load trained image emotion model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)  # 5 classes
    model.load_state_dict(torch.load("Models/emotion_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Define transformations (matching training preprocessing)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize as per training
])

def predict_image_emotion(image, model):
    image = image.convert("RGB")  # Ensure 3 channels
    image = data_transforms(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    emotions = ['angry', 'happy', 'neutral', 'sad', 'surprise']  # Correct classes
    return emotions[predicted_class]

def main():
    st.title("Emotion Detection App")
    st.subheader("Detect Emotions in Text and Images")

    option = st.radio("Choose Input Type:", ("Text", "Image"))
    
    if option == "Text":
        with st.form(key='text_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write(f"{prediction} {emoji_icon}")
                st.write(f"Confidence: {np.max(probability):.2f}")

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

    elif option == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            model = load_model()
            prediction = predict_image_emotion(image, model)
            
            st.write(f"Predicted Emotion: **{prediction}**")

if __name__ == '__main__':
    main()
