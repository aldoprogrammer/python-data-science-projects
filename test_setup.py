# Test script to verify setup
from textblob import TextBlob
import tweepy
import os
from dotenv import load_dotenv

load_dotenv()

# Test TextBlob
print("Testing TextBlob...")
sample_text = "I love this amazing product! It's fantastic!"
blob = TextBlob(sample_text)
print(f"Text: {sample_text}")
print(f"Sentiment: {blob.sentiment}")
print(f"Polarity: {blob.sentiment.polarity}")

# Test Twitter API credentials
print("\nTesting Twitter API credentials...")
api_key = os.getenv('TWITTER_API_KEY')
bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

if api_key and bearer_token:
    print("✓ Credentials loaded from .env file")
    try:
        client = tweepy.Client(bearer_token=bearer_token)
        print("✓ Twitter client created successfully")
    except Exception as e:
        print(f"✗ Error creating Twitter client: {e}")
else:
    print("✗ Missing credentials in .env file")

print("\nSetup test complete!")
