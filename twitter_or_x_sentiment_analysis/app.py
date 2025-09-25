import tweepy
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import os
from dotenv import load_dotenv
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('brown', quiet=True)
except:
    pass

# Load environment variables
load_dotenv()

class TwitterSentimentAnalyzer:
    def __init__(self):
        """Initialize the Twitter API client"""
        # Twitter API credentials (you'll need to set these)
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        # Initialize Tweepy client
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
            wait_on_rate_limit=True
        )
    
    def clean_tweet(self, tweet_text):
        """Clean tweet text for analysis"""
        # Remove URLs, mentions, hashtags, and special characters
        tweet_text = re.sub(r'http\S+|www\S+|https\S+', '', tweet_text, flags=re.MULTILINE)
        tweet_text = re.sub(r'@\w+|#\w+', '', tweet_text)
        tweet_text = re.sub(r'[^A-Za-z0-9\s]', '', tweet_text)
        tweet_text = tweet_text.strip()
        return tweet_text
    
    def get_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    
    def get_sentiment_score(self, text):
        """Get numerical sentiment score"""
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def create_demo_data(self, query="Demo Data"):
        """Create sample data for demonstration when API fails"""
        sample_tweets = [
            "I absolutely love Python programming! It's so powerful and easy to learn.",
            "Machine learning is revolutionizing the world. Amazing technology!",
            "Hate debugging code all day. So frustrating and time consuming.",
            "AI is the future. Excited to see what comes next!",
            "This new framework is terrible. Documentation is confusing.",
            "Great tutorial on data science. Really helpful for beginners.",
            "Python pandas library is incredible for data analysis.",
            "Struggling with this algorithm. Need better resources.",
            "Amazing breakthrough in artificial intelligence research!",
            "Love working with APIs. Makes integration so much easier.",
            "This code is buggy and poorly written. Needs major refactoring.",
            "Fantastic community support in open source projects.",
            "Data visualization with matplotlib is pretty cool.",
            "Annoying syntax errors keep breaking my code.",
            "Beautiful UI design in this new app. Very intuitive.",
            "Performance issues are ruining user experience.",
            "Excellent documentation makes learning so much faster.",
            "Server downtime is affecting productivity today.",
            "Innovation in tech industry never stops amazing me.",
            "Legacy code maintenance is such a nightmare."
        ]
        
        tweets_data = []
        for i, text in enumerate(sample_tweets):
            cleaned_text = self.clean_tweet(text)
            sentiment = self.get_sentiment(cleaned_text)
            sentiment_score = self.get_sentiment_score(cleaned_text)
            
            tweets_data.append({
                'tweet_id': f'demo_{i+1}',
                'text': text,
                'cleaned_text': cleaned_text,
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'created_at': datetime.now(),
                'retweet_count': i * 2,
                'like_count': i * 5
            })
        
        return tweets_data

    def fetch_tweets(self, query, count=100):
        """Fetch tweets based on query"""
        tweets_data = []
        
        try:
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'author_id', 'public_metrics'],
                max_results=min(count, 100)
            ).flatten(limit=count)
            
            for tweet in tweets:
                cleaned_text = self.clean_tweet(tweet.text)
                if cleaned_text:  # Only process non-empty tweets
                    sentiment = self.get_sentiment(cleaned_text)
                    sentiment_score = self.get_sentiment_score(cleaned_text)
                    
                    tweets_data.append({
                        'tweet_id': tweet.id,
                        'text': tweet.text,
                        'cleaned_text': cleaned_text,
                        'sentiment': sentiment,
                        'sentiment_score': sentiment_score,
                        'created_at': tweet.created_at,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'like_count': tweet.public_metrics['like_count']
                    })
        
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            print("Using demo data instead...")
            return self.create_demo_data(query)
        
        return tweets_data
    
    def analyze_sentiment(self, query, count=100):
        """Main function to analyze sentiment"""
        if not query or query.strip() == "":
            print("Empty query provided. Using demo data...")
            query = "Demo Data"
            tweets_data = self.create_demo_data(query)
        else:
            print(f"Fetching {count} tweets for query: '{query}'...")
            tweets_data = self.fetch_tweets(query, count)
        
        if not tweets_data:
            print("No tweets found or error occurred.")
            return None
        
        df = pd.DataFrame(tweets_data)
        
        # Calculate sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        
        print(f"\nAnalyzed {len(df)} tweets")
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{sentiment}: {count} ({percentage:.1f}%)")
        
        return df
    
    def visualize_sentiment(self, df, query):
        """Create visualizations for sentiment analysis"""
        if df is None or df.empty:
            print("No data to visualize.")
            return
        
        # Set up the plotting style - updated for newer seaborn
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Sentiment Analysis for: "{query}"', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Distribution Pie Chart
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green, Red, Gray
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                      autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 0].set_title('Sentiment Distribution')
        
        # 2. Sentiment Score Distribution
        axes[0, 1].hist(df['sentiment_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Sentiment Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Sentiment Score Distribution')
        
        # 3. Sentiment over Time
        df['created_at'] = pd.to_datetime(df['created_at'])
        df_time = df.groupby([df['created_at'].dt.hour, 'sentiment']).size().unstack(fill_value=0)
        df_time.plot(kind='bar', stacked=True, ax=axes[1, 0], color=['#2ecc71', '#e74c3c', '#95a5a6'])
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Number of Tweets')
        axes[1, 0].set_title('Sentiment Distribution by Hour')
        axes[1, 0].legend(title='Sentiment')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Top Words in Positive vs Negative Tweets
        from collections import Counter
        positive_words = []
        negative_words = []
        
        for _, row in df.iterrows():
            words = row['cleaned_text'].lower().split()
            if row['sentiment'] == 'Positive':
                positive_words.extend(words)
            elif row['sentiment'] == 'Negative':
                negative_words.extend(words)
        
        # Get top 10 words for each sentiment
        pos_common = Counter(positive_words).most_common(10)
        neg_common = Counter(negative_words).most_common(10)
        
        if pos_common and neg_common:
            pos_words, pos_counts = zip(*pos_common)
            neg_words, neg_counts = zip(*neg_common)
            
            x_pos = range(len(pos_words))
            axes[1, 1].barh(x_pos, pos_counts, color='green', alpha=0.7, label='Positive')
            axes[1, 1].set_yticks(x_pos)
            axes[1, 1].set_yticklabels(pos_words)
            axes[1, 1].set_xlabel('Frequency')
            axes[1, 1].set_title('Top Words in Positive Tweets')
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, df, query, filename=None):
        """Save results to CSV file"""
        if df is None or df.empty:
            print("No data to save.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_{query.replace(' ', '_')}_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        print(f"Results saved to: {filename}")

def main():
    print("=" * 60)
    print("        TWITTER/X SENTIMENT ANALYSIS TOOL")
    print("=" * 60)
    print("Examples of queries you can try:")
    print("- 'Python programming'")
    print("- '#AI' or '#MachineLearning'")
    print("- '@elonmusk' or '@openai'")
    print("- 'ChatGPT' or 'artificial intelligence'")
    print("- Press Enter for demo data")
    print("-" * 60)
    
    # Initialize analyzer
    try:
        analyzer = TwitterSentimentAnalyzer()
        print("✓ Twitter API client initialized successfully")
    except Exception as e:
        print(f"⚠ Warning: Twitter API initialization failed: {e}")
        print("Demo mode will be used instead.")
        analyzer = TwitterSentimentAnalyzer()
    
    # Get user input with validation
    while True:
        query = input("\nEnter search query: ").strip()
        if query == "":
            query = "Demo Data"
            break
        elif len(query) >= 1:
            break
        else:
            print("Please enter a valid query or press Enter for demo.")
    
    while True:
        try:
            count_input = input("Enter number of tweets to analyze (max 100, default 20): ").strip()
            count = int(count_input) if count_input else 20
            if 1 <= count <= 100:
                break
            else:
                print("Please enter a number between 1 and 100.")
        except ValueError:
            print("Please enter a valid number.")
    
    print("\n" + "=" * 60)
    
    # Analyze sentiment
    df = analyzer.analyze_sentiment(query, count)
    
    if df is not None:
        # Show sample tweets
        print("\nSample tweets:")
        print("-" * 40)
        for i, row in df.head(3).iterrows():
            print(f"\nTweet: {row['text'][:100]}...")
            print(f"Sentiment: {row['sentiment']} (Score: {row['sentiment_score']:.2f})")
        
        # Visualize results
        print(f"\n📊 Generating visualizations...")
        analyzer.visualize_sentiment(df, query)
        
        # Save results
        save_option = input("\n💾 Save results to CSV? (y/n): ").lower()
        if save_option == 'y':
            analyzer.save_results(df, query)
            
        print("\n✅ Analysis complete! Thank you for using the tool.")
    else:
        print("\n❌ Analysis failed. Please try again.")

if __name__ == "__main__":
    main()

