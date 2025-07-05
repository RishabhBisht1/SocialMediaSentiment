# streamlit_sentiment_app.py
import streamlit as st
import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from googleapiclient.discovery import build
import requests
import snscrape.modules.twitter as sntwitter
import matplotlib.pyplot as plt
from collections import Counter

# ------------------- Setup -------------------
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🔍 Social Media Sentiment Analyzer")

# Load model and tokenizer
model = load_model("sentiment_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = 100000
max_length = 50

# ------------------- Utility -------------------
def detect_platform(link):
    if "youtube.com" in link or "youtu.be" in link:
        return "youtube"
    elif "twitter.com" in link:
        return "twitter"
    return "unknown"

def preprocess_text(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(comments):
    texts = [preprocess_text(c) for c in comments]
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    predictions = model.predict(padded, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels

# ------------------- YouTube Scraper -------------------
def scrape_youtube_comments(link):
    # Extract video ID
    match = re.search(r"v=([\w-]+)", link)
    if not match:
        st.error("Invalid YouTube link.")
        return []
    video_id = match.group(1)

    api_key = "YOUR_YOUTUBE_API_KEY"  # Replace with your API key
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    nextPageToken = None

    while len(comments) < 100:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=nextPageToken
        ).execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            if len(comments) >= 100:
                break

        nextPageToken = response.get('nextPageToken')
        if not nextPageToken:
            break

    return comments

# ------------------- Twitter Scraper -------------------
def scrape_twitter_comments(link):
    match = re.search(r"status/(\d+)", link)
    if not match:
        st.error("Invalid Twitter link.")
        return []
    tweet_id = match.group(1)
    replies = []
    for i, tweet in enumerate(sntwitter.TwitterTweetScraper(tweet_id).get_items()):
        replies.append(tweet.content)
        if i >= 100:
            break
    return replies

# ------------------- Chart -------------------
def show_sentiment_chart(labels):
    counts = Counter(labels)
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    labels = [sentiment_map[i] for i in counts.keys()]
    sizes = [counts[i] for i in counts.keys()]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'gray', 'green'])
    ax.axis('equal')
    st.pyplot(fig)

# ------------------- App -------------------
link = st.text_input("Paste a YouTube or Twitter link 👇")

if link:
    platform = detect_platform(link)
    if platform == "unknown":
        st.warning("Please enter a valid YouTube or Twitter link.")
    else:
        st.info(f"Detected platform: {platform.capitalize()}")

        if st.button("Analyze Sentiment"):
            st.write("⏳ Scraping comments...")
            if platform == "youtube":
                comments = scrape_youtube_comments(link)
            elif platform == "twitter":
                comments = scrape_twitter_comments(link)

            if comments:
                st.success(f"Fetched {len(comments)} comments.")
                labels = predict_sentiment(comments)
                show_sentiment_chart(labels)

                # Show a few example comments per sentiment
                label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
                st.subheader("Example Comments:")
                for sentiment_id in [2, 1, 0]:
                    examples = [c for i, c in enumerate(comments) if labels[i] == sentiment_id][:3]
                    st.markdown(f"**{label_map[sentiment_id]} ({len(examples)} shown)**")
                    for e in examples:
                        st.markdown(f"- {e}")
