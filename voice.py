import streamlit as st
import speech_recognition as sr
from textblob import TextBlob

# Emotion lexicon (expand later for complexity)
emotion_lexicon = {
    "happy": "Joy",
    "joy": "Joy",
    "excited": "Joy",
    "angry": "Anger",
    "mad": "Anger",
    "furious": "Anger",
    "sad": "Sadness",
    "depressed": "Sadness",
    "cry": "Sadness",
    "tired": "Sadness"
}

def analyze_emotions(text):
    emotions = []
    for word in text.lower().split():
        if word in emotion_lexicon:
            emotions.append(emotion_lexicon[word])
    return emotions if emotions else ["Neutral"]

def detect_depression_level(emotions):
    sadness_count = emotions.count("Sadness")
    if sadness_count == 0:
        return "Low 🙂"
    elif sadness_count == 1:
        return "Moderate 😟"
    else:
        return "High 😢"

# Streamlit UI
st.title("🎙️ Voice-Based Emotion Detection")

uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    recognizer = sr.Recognizer()

    with sr.AudioFile(uploaded_file) as source:
        st.info("Transcribing audio...")
        audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio)
            st.success(f"Transcribed Text: {text}")

            emotions = analyze_emotions(text)
            depression = detect_depression_level(emotions)

            st.subheader("Detected Emotions:")
            for e in set(emotions):
                st.write(f"- {e}: {emotions.count(e)}")

            st.subheader("📝 Conclusion:")
            st.write(f"You are experiencing mainly {max(set(emotions), key=emotions.count)}.")

            st.subheader("📊 Depression Level:")
            st.write(depression)

        except sr.UnknownValueError:
            st.error("Sorry, could not understand the audio.")
        except sr.RequestError:
            st.error("Speech Recognition service error.")
