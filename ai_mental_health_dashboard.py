# ai_mental_health_dashboard.py
"""
AI-Powered Multimodal Mental Health Dashboard
Run: streamlit run ai_mental_health_dashboard.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import io
import asyncio

# Optional / conditional imports (some heavy packages)
try:
    from fer import FER
except Exception:
    FER = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

# BLE (bleak) optional
try:
    from bleak import BleakScanner, BleakClient
except Exception:
    BleakScanner = None
    BleakClient = None

# ---------------------------
# Helper analysis functions
# ---------------------------

# Simple lexicon for quick text/voice emotion extraction (can extend)
EMOTION_LEXICON = {
    "happy": "Joy", "joy": "Joy", "smile": "Joy", "excited": "Joy", "love": "Joy",
    "angry": "Anger", "mad": "Anger", "furious": "Anger", "hate": "Anger",
    "sad": "Sadness", "depressed": "Sadness", "cry": "Sadness", "tired": "Sadness",
    "fear": "Fear", "scared": "Fear", "nervous": "Fear", "anxious": "Fear",
    "neutral": "Neutral"
}

def analyze_text_simple(text):
    """Return emotion counts and depression indicator from raw text (simple lexicon + textblob sentiment)."""
    if not text:
        return {}, "None", {}
    words = [w.strip(".,!?").lower() for w in text.split()]
    found = [EMOTION_LEXICON[w] for w in words if w in EMOTION_LEXICON]
    counts = Counter(found)
    # sentiment with TextBlob if available
    sentiment = None
    if TextBlob:
        try:
            tb = TextBlob(text).sentiment.polarity  # -1..1
            sentiment = tb
        except Exception:
            sentiment = None
    # depression heuristic (very simple): many sadness keywords -> moderate/high
    sadness = counts.get("Sadness", 0)
    if sadness == 0:
        depression_level = "None"
    elif sadness <= 1:
        depression_level = "Low"
    elif sadness <= 3:
        depression_level = "Moderate"
    else:
        depression_level = "High"
    return counts, depression_level, {"sentiment": sentiment}

def analyze_transcript(text):
    """Wrapper for voice after transcription."""
    return analyze_text_simple(text)

# ---------------------------
# Face capture (local capture for n seconds) — uses FER
# ---------------------------
def capture_face_emotions(duration_seconds=8, fps=5):
    """
    Capture from webcam for duration_seconds and analyze frames using FER detector.
    Returns emotion counts dictionary and a saved CSV-like bytes buffer.
    """
    if FER is None or cv2 is None:
        raise RuntimeError("FER or OpenCV not available. Install 'fer' and 'opencv-python'.")
    detector = FER(mtcnn=True)  # may download models on first run
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible. Check camera and permissions.")
    end_time = datetime.now().timestamp() + duration_seconds
    emotion_log = []
    try:
        while datetime.now().timestamp() < end_time:
            ret, frame = cap.read()
            if not ret:
                break
            # analyze every frame or sampled frames
            results = detector.detect_emotions(frame)
            if results:
                # detect dominant emotion for first face
                emo, conf = detector.top_emotion(frame)
                if emo is not None:
                    emotion_log.append((datetime.now().strftime("%H:%M:%S"), emo, round(conf, 3)))
            # wait small bit
            cv2.waitKey(int(1000 / fps))
    finally:
        cap.release()
        cv2.destroyAllWindows()
    counts = Counter([e[1] for e in emotion_log])
    # prepare CSV bytes
    csv_buf = io.StringIO()
    csv_buf.write("Time,Emotion,Confidence\n")
    for row in emotion_log:
        csv_buf.write(f"{row[0]},{row[1]},{row[2]}\n")
    csv_buf.seek(0)
    return counts, csv_buf

# ---------------------------
# BLE Heart Rate (scan and connect)
# ---------------------------
HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

async def bleak_scan_once(timeout=5.0):
    if BleakScanner is None:
        raise RuntimeError("bleak not installed or unsupported on this system.")
    devices = await BleakScanner.discover(timeout=timeout)
    return devices

async def bleak_connect_and_listen(address, listen_seconds=20):
    if BleakClient is None:
        raise RuntimeError("bleak not installed or unsupported on this system.")
    results = []
    async with BleakClient(address) as client:
        # ensure connected
        if not client.is_connected:
            raise RuntimeError("Could not connect to device.")
        def callback(sender, data):
            # data format depends; heart rate standard: second byte is BPM if flags indicate
            try:
                bpm = int(data[1])
            except Exception:
                bpm = None
            if bpm:
                results.append((datetime.now().strftime("%H:%M:%S"), bpm))
        await client.start_notify(HR_UUID, callback)
        await asyncio.sleep(listen_seconds)
        await client.stop_notify(HR_UUID)
    return results

def run_ble_scan_blocking(timeout=5.0):
    return asyncio.run(bleak_scan_once(timeout))

def run_ble_listen_blocking(address, listen_seconds=20):
    return asyncio.run(bleak_connect_and_listen(address, listen_seconds))

# ---------------------------
# Fusion logic to combine modalities
# ---------------------------
def fuse_modalities(text_counts, voice_counts, face_counts, hr_values):
    """
    Simple rule-based fusion:
    - Convert counts to percentage weights
    - Heart rate: if mean BPM > 100 -> stress indicator
    - Final score: positive vs negative indicators
    """
    combined = {}
    # aggregate counts across modalities
    def top_from_counts(cnts):
        if not cnts:
            return None
        return cnts.most_common(1)[0][0]
    text_top = top_from_counts(text_counts)
    voice_top = top_from_counts(voice_counts)
    face_top = top_from_counts(face_counts)
    # HR analysis
    hr_flag = None
    if hr_values:
        bpm_vals = [v for _, v in hr_values]
        mean_bpm = sum(bpm_vals) / len(bpm_vals)
        if mean_bpm > 100:
            hr_flag = "High"
        elif mean_bpm < 55:
            hr_flag = "Low"
        else:
            hr_flag = "Normal"
    # Build combined decision heuristics
    negatives = 0
    positives = 0
    for emo in [text_top, voice_top, face_top]:
        if emo in ("Sadness", "Anger", "Fear"):
            negatives += 1
        elif emo == "Joy":
            positives += 1
    if hr_flag == "High":
        negatives += 1
    # Determine final label
    if negatives >= 2 and positives == 0:
        final = "Stressed / Negative"
    elif positives > negatives:
        final = "Positive / Relaxed"
    elif positives == negatives:
        final = "Mixed Emotions"
    else:
        final = "Neutral / Monitor"
    details = {
        "text_top": text_top, "voice_top": voice_top, "face_top": face_top,
        "hr_mean": (sum([v for _, v in hr_values]) / len(hr_values)) if hr_values else None,
        "hr_flag": hr_flag
    }
    return final, details

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Mental Health Dashboard", layout="wide")
st.title("🧠 AI-Powered Multimodal Mental Health Dashboard")

# Initialize session state containers
if "text_counts" not in st.session_state: st.session_state["text_counts"] = Counter()
if "text_depression" not in st.session_state: st.session_state["text_depression"] = "None"
if "voice_counts" not in st.session_state: st.session_state["voice_counts"] = Counter()
if "voice_depression" not in st.session_state: st.session_state["voice_depression"] = "None"
if "face_counts" not in st.session_state: st.session_state["face_counts"] = Counter()
if "hr_log" not in st.session_state: st.session_state["hr_log"] = []  # list of (time,bpm)
if "face_csv" not in st.session_state: st.session_state["face_csv"] = None

# Sidebar navigation
page = st.sidebar.radio("Flow (do in order):", ["Text", "Voice", "Face", "Heart Rate", "Final Dashboard"])

# ---------------------------
# Page: Text
# ---------------------------
if page == "Text":
    st.header("✍️ Text Input Analysis")
    st.write("Describe how you feel (one or two sentences).")
    txt = st.text_area("Enter text here", height=150)
    if st.button("Analyze Text"):
        counts, depression_level, extras = analyze_text_simple(txt)
        st.session_state["text_counts"] = counts
        st.session_state["text_depression"] = depression_level
        st.success("Text analyzed.")
        # Show results
        st.subheader("Detected emotions (text):")
        if counts:
            st.write(pd.Series(dict(counts)))
        else:
            st.write("No emotion keywords detected.")
        st.write(f"Depression level (heuristic): **{depression_level}**")
        if extras.get("sentiment") is not None:
            st.write(f"Sentiment polarity (TextBlob): {extras['sentiment']:.3f}")

# ---------------------------
# Page: Voice
# ---------------------------
elif page == "Voice":
    st.header("🎙️ Voice Input Analysis")
    st.write("Upload an audio file (.wav/.mp3). Transcription uses Google Speech Recognition (internet).")
    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
    if uploaded is not None:
        if sr is None:
            st.error("speech_recognition not installed.")
        else:
            if st.button("Transcribe & Analyze"):
                recognizer = sr.Recognizer()
                try:
                    with sr.AudioFile(uploaded) as source:
                        audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio)
                    st.write("**Transcribed Text:**", text)
                    counts, depression_level, _ = analyze_transcript(text)
                    st.session_state["voice_counts"] = counts
                    st.session_state["voice_depression"] = depression_level
                    st.success("Voice analyzed.")
                    st.subheader("Detected emotions (voice):")
                    st.write(pd.Series(dict(counts)))
                    st.write(f"Depression level (heuristic): **{depression_level}**")
                except sr.UnknownValueError:
                    st.error("Could not understand audio.")
                except sr.RequestError:
                    st.error("Speech recognition service error (check internet).")

# ---------------------------
# Page: Face
# ---------------------------
elif page == "Face":
    st.header("🎥 Facial Emotion Capture")
    st.write("Capture emotions from webcam for a short time. Close the camera window with 'q'.")
    duration = st.slider("Capture duration (seconds)", min_value=4, max_value=20, value=8, step=2)
    if st.button("Capture from Webcam"):
        if FER is None or cv2 is None:
            st.error("FER or OpenCV not installed. Install 'fer' and 'opencv-python'.")
        else:
            try:
                with st.spinner("Capturing... Look at the camera"):
                    counts, csv_buf = capture_face_emotions(duration_seconds=duration)
                st.session_state["face_counts"] = counts
                st.session_state["face_csv"] = csv_buf.getvalue()
                st.success("Face capture complete.")
                if counts:
                    st.subheader("Detected emotions (face):")
                    st.write(pd.Series(dict(counts)))
                    # download CSV
                    st.download_button("Download face emotions CSV", data=csv_buf.getvalue(), file_name="face_emotions.csv")
                else:
                    st.write("No emotions detected from face capture.")
            except Exception as e:
                st.error(f"Error capturing face: {e}")

# ---------------------------
# Page: Heart Rate (BLE)
# ---------------------------
elif page == "Heart Rate":
    st.header("❤️ Heart Rate via Smartwatch (BLE)")
    st.write("Scan for nearby BLE devices and select your device to read heart rate (if device broadcasts HR).")
    scan_timeout = st.number_input("Scan timeout (seconds)", min_value=3, max_value=15, value=5)
    if st.button("Scan BLE devices"):
        if BleakScanner is None:
            st.error("bleak not installed or BLE not supported here.")
        else:
            try:
                with st.spinner("Scanning..."):
                    devices = run_ble_scan_blocking(timeout=float(scan_timeout))
                if not devices:
                    st.warning("No BLE devices found. Ensure smartwatch is on and advertising.")
                else:
                    dev_list = []
                    for i, d in enumerate(devices):
                        name = d.name or "Unknown"
                        dev_list.append((i, name, d.address))
                        st.write(f"[{i}] {name} — {d.address}")
                    # let user select
                    choice = st.number_input("Select device index", min_value=0, max_value=len(devices)-1, value=0)
                    if st.button("Connect & Read HR for 20s"):
                        selected = devices[int(choice)]
                        try:
                            with st.spinner(f"Connecting to {selected.address} ..."):
                                readings = run_ble_listen_blocking(selected.address, listen_seconds=20)
                            if readings:
                                # append to session state
                                for r in readings:
                                    st.session_state["hr_log"].append(r)
                                st.success("Heart rate readings collected.")
                                st.write(pd.DataFrame(st.session_state["hr_log"], columns=["Time", "BPM"]))
                            else:
                                st.warning("No heart rate notifications received. Device may not expose standard HR service.")
                        except Exception as e:
                            st.error(f"BLE read error: {e}")
            except Exception as e:
                st.error(f"Scan error: {e}")
    st.write("Manual entry (if BLE not available)")
    manual_bpm = st.number_input("Enter current BPM manually (optional)", min_value=30, max_value=220, value=75)
    if st.button("Save manual BPM as current reading"):
        st.session_state["hr_log"].append((datetime.now().strftime("%H:%M:%S"), int(manual_bpm)))
        st.success("Manual BPM saved.")

# ---------------------------
# Page: Final Dashboard
# ---------------------------
elif page == "Final Dashboard":
    st.header("📊 Final Multimodal Analysis Dashboard")
    st.write("This page combines Text, Voice, Face and Heart Rate results and displays a final conclusion.")

    text_counts = st.session_state.get("text_counts", Counter())
    voice_counts = st.session_state.get("voice_counts", Counter())
    face_counts = st.session_state.get("face_counts", Counter())
    hr_log = st.session_state.get("hr_log", [])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Text Analysis")
        st.write("Emotion counts:", dict(text_counts))
        st.write("Depression (heuristic):", st.session_state.get("text_depression", "None"))
    with col2:
        st.subheader("Voice Analysis")
        st.write("Emotion counts:", dict(voice_counts))
        st.write("Depression (heuristic):", st.session_state.get("voice_depression", "None"))

    st.subheader("Face Analysis")
    st.write("Emotion counts:", dict(face_counts))
    if st.session_state.get("face_csv"):
        st.download_button("Download face capture CSV", data=st.session_state["face_csv"], file_name="face_emotions.csv")

    st.subheader("Heart Rate Readings")
    if hr_log:
        df_hr = pd.DataFrame(hr_log, columns=["Time", "BPM"])
        st.line_chart(df_hr["BPM"])
        st.write(df_hr)
        avg_bpm = df_hr["BPM"].mean()
        st.write(f"Average BPM: {avg_bpm:.1f}")
    else:
        st.write("No heart rate data yet.")

    # Fusion and final conclusion
    final_label, details = fuse_modalities(text_counts, voice_counts, face_counts, hr_log)
    st.markdown("---")
    st.subheader("🔔 Final Conclusion")
    st.write(f"**{final_label}**")
    st.write("Details:", details)

    # Visual summary pie chart of combined emotions (text+voice+face)
    combined = Counter()
    combined.update(text_counts)
    combined.update(voice_counts)
    combined.update(face_counts)
    if combined:
        labels = list(combined.keys())
        values = list(combined.values())
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
        ax.set_title("Combined Emotion Distribution")
        st.pyplot(fig)
    else:
        st.write("No combined emotion data to plot.")

st.sidebar.markdown("---")
st.sidebar.write("Tip: Complete steps in order: Text → Voice → Face → Heart Rate → Final Dashboard")

