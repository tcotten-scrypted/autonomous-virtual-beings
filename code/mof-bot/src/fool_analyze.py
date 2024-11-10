import pandas as pd
import json
import os

def analyze_fool(fool_name):
    """
    Analyze tweet data for a specified "fool" (user) and save a daily engagement summary to JSON.

    Parameters:
    - fool_name (str): The name of the "fool" (without '@') for whom the analysis is performed.
    
    This function:
    - Loads the tweet data from a JSON file in ../data/fools/
    - Aggregates engagement metrics by date
    - Counts occurrences of hashtags, mentions, and tickers per day
    - Saves the summarized daily engagement data to a new JSON file in ../data/fools/
    """
    
    # Construct the file path to the JSON file
    file_path = f'../data/fools/{fool_name}.json'

    # Check if the specified file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Please check the fool's name and try again.")
        return

    # Load the dataset from the JSON file into a DataFrame
    data = pd.read_json(file_path)

    # Convert 'created_at' to datetime format for time-based analysis
    # Extract the date part as a string for grouping and JSON compatibility
    data['created_at'] = pd.to_datetime(data['created_at'])
    data['date'] = data['created_at'].dt.date.astype(str)  # Convert to string for compatibility in JSON

    # Group data by date and calculate daily engagement metrics, including tweet count
    daily_engagement = data.groupby('date').agg({
        'retweet_count': 'sum',     # Sum of retweets per day
        'like_count': 'sum',        # Sum of likes per day
        'quote_count': 'sum',       # Sum of quotes per day
        'reply_count': 'sum',       # Sum of replies per day
        'text': 'count'             # Count of tweets per day
    }).reset_index()

    # Rename the 'text' column to 'tweet_count' for clarity
    daily_engagement = daily_engagement.rename(columns={'text': 'tweet_count'})

    # Explode lists in hashtags, mentions, and tickers for counting individual items
    # Drop rows with NaN values for each respective field before counting occurrences
    hashtag_counts = data.explode('hashtags').dropna(subset=['hashtags'])
    hashtag_counts = hashtag_counts.groupby(['date', 'hashtags']).size().reset_index(name='count')

    mention_counts = data.explode('mentions').dropna(subset=['mentions'])
    mention_counts = mention_counts.groupby(['date', 'mentions']).size().reset_index(name='count')

    ticker_counts = data.explode('tickers').dropna(subset=['tickers'])
    ticker_counts = ticker_counts.groupby(['date', 'tickers']).size().reset_index(name='count')

    # Convert grouped counts of hashtags, mentions, and tickers into dictionaries by date
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
        "daily_engagement": daily_engagement.to_dict(orient='records')  # Convert DataFrame to list of records for JSON
    }

    # Define the output path and save the results as a JSON file in ../data/fools/
    output_file = f'../data/fools/daily_engagement_summary_{fool_name}.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Daily engagement summary has been saved to '{output_file}'")


if __name__ == "__main__":
    # Prompt user for the fool's name (without '@')
    fool_name = input("Enter the fool's name (without a hashtag): ")
    analyze_fool(fool_name)