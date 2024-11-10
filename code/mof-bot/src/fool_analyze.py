import pandas as pd
import json
import os

# Prompt user for the fool's name
fool_name = input("Enter the fool's name (without a hashtag): ")

# Construct the file path
file_path = f'../data/fools/{fool_name}.json'

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File {file_path} does not exist. Please check the fool's name and try again.")
    exit()

# Load the dataset
data = pd.read_json(file_path)

# Convert 'created_at' to datetime and extract relevant date fields
data['created_at'] = pd.to_datetime(data['created_at'])
data['date'] = data['created_at'].dt.date.astype(str)  # Convert date to string for JSON compatibility

# Group data by date and calculate daily engagement metrics, including tweet count
daily_engagement = data.groupby('date').agg({
    'retweet_count': 'sum',
    'like_count': 'sum',
    'quote_count': 'sum',
    'reply_count': 'sum',
    'text': 'count'  # Count of tweets (or replies/posts) per day
}).reset_index()
daily_engagement = daily_engagement.rename(columns={'text': 'tweet_count'})

# Aggregate hashtags, mentions, and tickers per date
# Explode lists, then group by date and count occurrences
hashtag_counts = data.explode('hashtags').dropna(subset=['hashtags'])
hashtag_counts = hashtag_counts.groupby(['date', 'hashtags']).size().reset_index(name='count')

mention_counts = data.explode('mentions').dropna(subset=['mentions'])
mention_counts = mention_counts.groupby(['date', 'mentions']).size().reset_index(name='count')

ticker_counts = data.explode('tickers').dropna(subset=['tickers'])
ticker_counts = ticker_counts.groupby(['date', 'tickers']).size().reset_index(name='count')

# Convert grouped counts into dictionaries for each date without using apply()
daily_hashtags = {
    date: group[['hashtags', 'count']].to_dict(orient='records')
    for date, group in hashtag_counts.groupby('date')
}

daily_mentions = {
    date: group[['mentions', 'count']].to_dict(orient='records')
    for date, group in mention_counts.groupby('date')
}

daily_tickers = {
    date: group[['tickers', 'count']].to_dict(orient='records')
    for date, group in ticker_counts.groupby('date')
}

# Add hashtags, mentions, and tickers to each date entry in daily_engagement
daily_engagement['hashtags'] = daily_engagement['date'].map(daily_hashtags).fillna('').apply(lambda x: x if x != '' else [])
daily_engagement['mentions'] = daily_engagement['date'].map(daily_mentions).fillna('').apply(lambda x: x if x != '' else [])
daily_engagement['tickers'] = daily_engagement['date'].map(daily_tickers).fillna('').apply(lambda x: x if x != '' else [])

# Create a dictionary to hold the complete output data
output_data = {
    "daily_engagement": daily_engagement.to_dict(orient='records')
}

# Define the output path and save the results in ../data/fools/
output_file = f'../data/fools/daily_engagement_summary_{fool_name}.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Daily engagement summary has been saved to '{output_file}'")