# Social Media Sentiment Analyzer

This project analyzes the sentiment of user-generated comments on Twitter or YouTube posts using a deep learning model.

## 🔍 Features

* Accepts Twitter or YouTube post URLs as input
* Scrapes the top 100 comments automatically
* Predicts sentiment: Positive, Neutral, or Negative
* Displays sentiment distribution as a pie chart
* Shows sample comments for each sentiment class

## 🧠 Model Details

* Built using TensorFlow and LSTM layers
* Trained on combined datasets from Twitter and YouTube
* Uses NLP preprocessing: tokenization, padding, cleaning
* Supports vocabulary size up to 100,000 words\
* Trained on dataset of size 1 Million+ values.

## 💻 Tech Stack

* Python, TensorFlow, NumPy, Streamlit
* YouTube Data API, snscrape for Twitter
* Matplotlib for charting

## 🚀 How to Run

1. Clone the repository
2. Install dependencies.
3. Add your YouTube Data API key in the code
4. Run the app:

   ```bash
   streamlit run streamlit_sentiment_app.py
   ```

## 📂 Files

* `streamlit_sentiment_app.py`: Streamlit frontend and logic
* `sentiment_model.keras`: Trained LSTM model
* `tokenizer.pkl`: Tokenizer used for input text

## ✨ Example Use Case

Paste a Twitter or YouTube post link and instantly get insights into what people are saying — whether the tone is mostly positive, neutral, or negative.
