from fer import FER
import cv2
import csv
import time
from collections import Counter
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string

# -----------------------
# Flask app setup
# -----------------------
app = Flask(__name__)
emotions_log = []

# -----------------------
# Run detection first
# -----------------------
def run_emotion_detection():
    global emotions_log
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    print("🎥 Press 'q' to quit capture...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotions = detector.detect_emotions(frame)

        if emotions:
            for face in emotions:
                (x, y, w, h) = face["box"]
                dominant_emotion, score = detector.top_emotion(frame)

                if dominant_emotion:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = f"{dominant_emotion} ({score:.2f})"
                    cv2.putText(frame, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    emotions_log.append([time.strftime("%H:%M:%S"),
                                         dominant_emotion, f"{score:.2f}"])

        cv2.imshow("Facial Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save CSV
    with open("emotions_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Emotion", "Confidence"])
        if emotions_log:
            writer.writerows(emotions_log)
        else:
            writer.writerow(["-", "No emotion detected", "0.0"])


# -----------------------
# Generate analysis + chart images
# -----------------------
def analyze_emotions():
    if not emotions_log:
        return "❌ No data captured", "", "", "⚠️ No conclusion available"

    all_emotions = [e[1] for e in emotions_log]
    emotion_counts = Counter(all_emotions)
    total = sum(emotion_counts.values())
    dominant_emotion = emotion_counts.most_common(1)[0]

    # Interpretation + Conclusion
    if emotion_counts["happy"] / total > 0.5:
        interpretation = "😊 You showed mostly positive and healthy emotions."
        conclusion = "✅ Overall mental state looks positive and healthy."
    elif (emotion_counts["sad"] + emotion_counts["angry"] + emotion_counts["fear"]) / total > 0.5:
        interpretation = "⚠️ Negative emotions (sad, angry, fear) dominated."
        conclusion = "❗ Signs of stress or low mood detected. Consider relaxation or support."
    elif emotion_counts["neutral"] / total > 0.6:
        interpretation = "😐 Neutral expressions dominated."
        conclusion = "ℹ️ Your mood appears stable but somewhat flat."
    else:
        interpretation = "ℹ️ Mixed emotions were detected."
        conclusion = "🙂 Your emotional state is balanced with some variations."

    # Bar Chart
    labels = list(emotion_counts.keys())
    values = list(emotion_counts.values())
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("Emotion Frequency Distribution")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    bar_chart = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    # Pie Chart
    plt.figure(figsize=(5, 5))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Emotion Distribution (Pie Chart)")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    pie_chart = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return interpretation, bar_chart, pie_chart, conclusion


# -----------------------
# Flask route
# -----------------------
@app.route("/")
def home():
    interpretation, bar_chart, pie_chart, conclusion = analyze_emotions()

    html = """
    <html>
    <head><title>Emotion Analysis Report</title></head>
    <body style="font-family: Arial; text-align:center;">
        <h1>🧠 Mental Health Analysis Report</h1>
        <p><b>Analysis:</b> {{ interpretation }}</p>

        {% if bar_chart %}
            <h2>📊 Emotion Frequency</h2>
            <img src="data:image/png;base64,{{ bar_chart }}" width="500"/>
            <h2>🥧 Emotion Distribution</h2>
            <img src="data:image/png;base64,{{ pie_chart }}" width="400"/>
            <h2>📌 Conclusion</h2>
            <p style="font-size:18px; font-weight:bold;">{{ conclusion }}</p>
        {% else %}
            <p>No emotions detected.</p>
        {% endif %}
    </body>
    </html>
    """
    return render_template_string(html,
                                  interpretation=interpretation,
                                  bar_chart=bar_chart,
                                  pie_chart=pie_chart,
                                  conclusion=conclusion)


# -----------------------
# Main Run
# -----------------------
if __name__ == "__main__":
    run_emotion_detection()  # run once to capture
    app.run(debug=True, use_reloader=False)

