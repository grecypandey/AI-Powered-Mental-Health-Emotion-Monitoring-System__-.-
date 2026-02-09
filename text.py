import streamlit as st
from collections import Counter

# Extended emotion dictionary
emotion_dict = {
    "happy": "Joy",
    "joy": "Joy",
    "smile": "Joy",
    "excited": "Joy",
    "love": "Joy",
    "angry": "Anger",
    "mad": "Anger",
    "furious": "Anger",
    "hate": "Anger",
    "sad": "Sadness",
    "depressed": "Sadness",
    "cry": "Sadness",
    "upset": "Sadness",
    "lonely": "Sadness",
    "broken": "Sadness",
    "fear": "Fear",
    "scared": "Fear",
    "nervous": "Fear"
}

st.title("🧠 Emotion Detector with Conclusion & Depression Level")

user_input = st.text_area("How are you feeling today?")

if st.button("Analyze"):
    words = user_input.lower().split()
    emotions_found = [emotion_dict[word] for word in words if word in emotion_dict]

    if emotions_found:
        counts = Counter(emotions_found)

        # ✅ Show detected emotions
        st.subheader("Detected Emotions:")
        for emo, count in counts.items():
            st.write(f"• {emo}: {count}")

        # ✅ Depression Level Calculation
        sadness_count = counts.get("Sadness", 0)
        if sadness_count == 0:
            depression_level = "None 😊"
        elif sadness_count <= 1:
            depression_level = "Low 😌"
        elif sadness_count <= 3:
            depression_level = "Moderate 😟"
        else:
            depression_level = "High 😢"

        # ✅ Generate Conclusion
        main_emotion = counts.most_common(1)[0][0]
        if len(counts) == 1:
            conclusion = f"You are mainly experiencing **{main_emotion}**."
        else:
            conclusion = f"You are experiencing **mixed emotions**, mainly {main_emotion}."

        # ✅ Display results
        st.subheader("📝 Conclusion:")
        st.write(conclusion)

        st.subheader("📊 Depression Level:")
        st.write(depression_level)

    else:
        st.warning("No emotions detected, please describe your feelings more clearly.")
