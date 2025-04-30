import os
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


def load_text_files(base_directory):
    texts = []
    labels = []
    for subfolder in os.listdir(base_directory):
        subfolder_path = os.path.join(base_directory, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".txt"):
                    with open(os.path.join(subfolder_path, filename), 'r', encoding='utf-8') as f:
                        text = f.read()
                        if subfolder.lower() == "neg":
                            texts.append(text)
                            labels.append(0)
                        elif subfolder.lower() == "pos":
                            texts.append(text)
                            labels.append(1)
    return texts, labels

# Load data
base_directory = r"E:\Python\Sentiment-Analysis-NN\Sentiment\train"
texts, labels = load_text_files(base_directory)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf_vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

# Save vectorizer
joblib.dump(tfidf_vectorizer, "vectorizer.pkl")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save("sentiment_model.h5")

# Predict probabilities
y_proba = model.predict(X_test).flatten()

# Interpret scores and assign sentiment
for idx, score in enumerate(y_proba):
    if score < 0.4:
        sentiment = "Negative"
    elif 0.4 <= score <= 0.6:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"
    print(f"Text {idx + 1}: Score = {score:.2f}, Sentiment = {sentiment}")

# Predict class labels for evaluation
y_pred = (y_proba >= 0.5).astype(int)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Prediction function
def predict_sentiment(user_input):
    vectorizer = joblib.load("vectorizer.pkl")
    model = load_model("sentiment_model.h5")
    
    input_features = vectorizer.transform([user_input]).toarray()
    score = model.predict(input_features)[0][0]
    
    if score < 0.4:
        sentiment = "Negative"
    elif 0.4 <= score <= 0.6:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"
    
    print(f"\nText: {user_input}")
    print(f"Score: {score:.2f}")
    print(f"Sentiment: {sentiment}")

# User input for prediction
user_input = input("\nEnter a text to analyze sentiment: ")
predict_sentiment(user_input)
