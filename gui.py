import streamlit as st
import numpy as np
import librosa
from keras.models import load_model

# Load the pre-trained model
model = load_model(r'C:\Users\akash\OneDrive\Desktop\speech_emotion\emotion_detection_by_speech.h5')

# Define the emotions
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Function to extract MFCC features from audio file
def extract_mfcc(filename, duration=3, offset=0.5):
    y, sr = librosa.load(filename, duration=duration, offset=offset)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Streamlit UI
st.title('Emotion detection using Audio')

uploaded_file = st.file_uploader("Upload an audio file (.wav only)", type=["wav"])

if uploaded_file is not None:
    with open('temp_audio.wav', 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio('temp_audio.wav')
    
    if st.button('Detect'):
        with st.spinner('Predicting...'):
            # Extract features from the uploaded audio file
            mfcc = extract_mfcc('temp_audio.wav')
            mfcc = mfcc.reshape(1, 40, 1)
            
            # Predict the emotion
            prediction = model.predict(mfcc)
            predicted_emotion = emotions[np.argmax(prediction)]
        
        st.success('Prediction completed!')
        st.write(f'Predicted Emotion: **{predicted_emotion}**')
