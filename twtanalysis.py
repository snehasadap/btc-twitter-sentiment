import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import ssl

# Set up SSL and download VADER lexicon
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('vader_lexicon')

# Initialize VADER
sid = SentimentIntensityAnalyzer()

# Define function for sentiment analysis
def get_sentiment(text):
    score = sid.polarity_scores(text)['compound']
    if score > 0:
        return 'positive'
    elif score == 0:
        return 'neutral'
    else:
        return 'negative'


columns = [
    'user_name', 'user_location', 'user_description', 'user_created', 'user_followers', 
    'user_friends', 'user_favourites', 'user_verified', 'date', 'text', 'Sentiment'
]

chunk_size = 90000
first_chunk = True

# Process the file in chunks
for chunk in pd.read_csv('Bitcoin_tweets.csv', chunksize=chunk_size):
    chunk.dropna(subset=['text'], inplace=True)  
    chunk['Sentiment'] = chunk['text'].apply(get_sentiment)  
    
    chunk.to_csv('Tweets_With_Sentiments.csv', mode='a', header=first_chunk, index=False, columns=columns)
    first_chunk = False  
