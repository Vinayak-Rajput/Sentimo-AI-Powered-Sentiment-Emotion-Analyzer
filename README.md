# Sentimo: Customer Sentiment & Analytics Engine

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/framework-Flask-lightgrey.svg)

Sentimo is a sophisticated full-stack NLP web application designed to decode the complexity of human language. By leveraging Deep Neural Networks, Sentimo provides two distinct layers of analysis: identifying general sentiment (Positive/Negative) and pinpointing specific emotional states such as Joy, Sadness, and Anger.

---

## 📖 Table of Contents
- [🚀 Features](#-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [⚙️ Setup & Installation](#️-setup--installation)
- [💻 Usage](#-usage)
- [🧪 Technical Overview](#-technical-overview)
- [🗺️ Future Roadmap](#️-future-roadmap)
- [🤝 Contributing](#-contributing)
- [⚖️ License](#️-license)

---

## 🚀 Features

*   **Dual-Analysis Engine**: Toggle between high-level Sentiment Analysis (Binary/Ternary) and granular Emotion Detection (Multi-class).
*   **Deep Learning Backend**: Built on TensorFlow/Keras architectures featuring Dropout layers to minimize overfitting and ensure robust generalization.
*   **Real-time Web UI**: A clean, responsive interface powered by Flask for instantaneous text processing.
*   **Automated Pre-processing**: A built-in NLTK pipeline that handles tokenization, stop-word removal, and text cleaning seamlessly.

---

## 🛠️ Tech Stack

| Category | Technology |
| :--- | :--- |
| **Frontend** | HTML5, Jinja2 Templates, CSS3 |
| **Backend** | Flask (Python) |
| **Machine Learning** | TensorFlow, Keras, Scikit-learn |
| **Data Processing** | Pandas, NumPy, NLTK, Joblib |

---

## 📁 Project Structure

```plaintext
├── app.py                      # Flask Web Server & API Routing
├── Neural_network_emotion.py   # Training script for Emotion detection
├── Neural_network_sentiment.py # Training script for Sentiment analysis
├── models/
│   ├── sentiment_model.h5      # Pre-trained Sentiment Model
│   └── emotion_nn_model.h5     # Pre-trained Emotion Model
└── pickled/
    ├── emotion_label_encoder.pkl
    ├── emotion_vectorizer.pkl   
    └── vectorizer.pkl           
└── templates/
    ├── index.html              # Main Landing Page
    ├── sentiment.html          # Sentiment Prediction Interface
    └── emotion.html            # Emotion Prediction Interface
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- Pip (Python Package Manager)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/sentimo.git
   cd sentimo
   ```

2. **Install Dependencies**
   ```bash
   pip install flask tensorflow pandas scikit-learn nltk joblib
   ```

3. **Download NLTK Resources**
   Sentimo requires specific NLTK data for text pre-processing:
   ```python
   python -c "import nltk; nltk.download('stopwords')"
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```
   The application will be available at `http://127.0.0.1:5000`.

---

## 💻 Usage

1. Launch the application and navigate to the home page.
2. **Sentiment Analysis**: Input a block of text to determine if the overall tone is Positive, Negative, or Neutral.
3. **Emotion Detection**: Input text to see which specific emotion (e.g., Joy, Anger, Fear) is most prevalent.
4. Results are displayed instantly with the corresponding classification.

---

## 🧪 Technical Overview

Sentimo utilizes a **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization approach combined with a **Multi-Layer Perceptron (MLP)**. 

- **Optimization**: The model uses Dropout layers (0.2 - 0.5) to maintain high accuracy on unseen data.
- **Efficiency**: Models and vectorizers are loaded into memory once during the server startup to ensure low-latency predictions.
- **Data Pipeline**: Raw text undergoes lowercasing, punctuation removal, and stop-word filtering before reaching the neural network.

---

## 🗺️ Future Roadmap

We are constantly working to improve Sentimo's accuracy and feature set. Upcoming improvements include:

- [ ] **Word Embeddings**: Transitioning from TF-IDF to Word2Vec or GloVe to capture contextual meaning and word sequences.
- [ ] **Confidence Visualizer**: Adding a UI component to show the percentage of model confidence (e.g., "Joy: 85%").
- [ ] **API Endpoint**: Developing a `/api/analyze` JSON route for "headless" integration into other services.
- [ ] **Class Balancing**: Implementing `class_weight` during training to handle imbalances between emotion categories (e.g., Joy vs. Surprise).

---

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

---
**Developed with ❤️ by the Sentimo Team.**
