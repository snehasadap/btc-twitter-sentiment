import pandas as pd
import numpy as np
import plotly.graph_objs as go


file_path = '/Users/sneha/Desktop/twitter-sentiment-analysis/bitcoin_tweets_with_sentiments.csv 15-51-37-951.csv'
tweets_df = pd.read_csv(file_path)

tweets_df['date'] = pd.to_datetime(tweets_df['date'], errors='coerce')

tweets_df['datetime'] = tweets_df['date'].dt.floor('H')
hourly_sentiment_counts = tweets_df.groupby([tweets_df['datetime'], 'Sentiment']).size().unstack(fill_value=0)

hourly_sentiment_counts += np.random.normal(0, 0.1, hourly_sentiment_counts.shape)


positive_trace = go.Scatter(
    x=hourly_sentiment_counts.index,
    y=hourly_sentiment_counts['positive'],
    mode='lines',
    name='Positive',
    line=dict(color='green', width=1),
)
negative_trace = go.Scatter(
    x=hourly_sentiment_counts.index,
    y=hourly_sentiment_counts['negative'],
    mode='lines',
    name='Negative',
    line=dict(color='red', width=1),
)
neutral_trace = go.Scatter(
    x=hourly_sentiment_counts.index,
    y=hourly_sentiment_counts['neutral'],
    mode='lines',
    name='Neutral',
    line=dict(color='gray', width=1),
)

fig = go.Figure(data=[positive_trace, negative_trace, neutral_trace])


fig.update_layout(
    title='BTC Sentiment on Twitter From Feb to April 2021',
    yaxis_title='Tweets',
    xaxis=dict(
        tickformat='%b %d %Y',  
        tickmode='array',
        tickvals=pd.date_range(start=hourly_sentiment_counts.index.min(), 
                               end=hourly_sentiment_counts.index.max(), freq='2W'),  
        tickangle=0           
    ),
    yaxis=dict(tickformat='d'),  
    template='plotly_dark'       
)

fig.show()
