from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load models and vectorizers
sentiment_model = load_model("models/sentiment_model.h5")
emotion_model = load_model("models/emotion_nn_model.h5")

sentiment_vectorizer = joblib.load("pickled/vectorizer.pkl")
emotion_vectorizer = joblib.load("pickled/emotion_vectorizer.pkl")
emotion_label_encoder = joblib.load("pickled/emotion_label_encoder.pkl")

# For cleaning emotion inputs
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Flask app setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    result = None
    score = None
    if request.method == 'POST':
        user_input = request.form['text']
        features = sentiment_vectorizer.transform([user_input]).toarray()
        score = sentiment_model.predict(features)[0][0]
        score = round(float(score), 2)
        if score < 0.4:
            result = "Negative"
        elif score <= 0.6:
            result = "Neutral"
        else:
            result = "Positive"
    return render_template('sentiment.html', sentiment=result, score=score)

@app.route('/emotion', methods=['GET', 'POST'])
def emotion():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['text']
        cleaned = clean_text(user_input)
        features = emotion_vectorizer.transform([cleaned]).toarray()
        preds = emotion_model.predict(features)
        label_index = np.argmax(preds)
        prediction = emotion_label_encoder.inverse_transform([label_index])[0]
    return render_template('emotion.html', emotion=prediction)

if __name__ == '__main__':
    app.run(debug=True)
