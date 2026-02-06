import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import joblib

# Load dataset
df = pd.read_pickle("merged_training.pkl")

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Encode emotion labels
le = LabelEncoder()
df['emotion_label'] = le.fit_transform(df['emotions'])

# ❗ Downsample dataset to avoid MemoryError
df_small = df.sample(n=30000, random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_small['clean_text'], df_small['emotion_label'], test_size=0.2, random_state=42
)

# TF-IDF vectorization with fewer features
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# One-hot encode the labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Build neural network
model = Sequential()
model.add(Dense(512, input_shape=(X_train_tfidf.shape[1],), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train_encoded.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(
    X_train_tfidf, y_train_encoded,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop]
)

# Evaluate
loss, accuracy = model.evaluate(X_test_tfidf, y_test_encoded)
print(f"Test Accuracy: {accuracy:.4f}")

# Save model and encoders
model.save("emotion_nn_model.h5")
joblib.dump(vectorizer, "emotion_vectorizer.pkl")
joblib.dump(le, "emotion_label_encoder.pkl")
