import os
import tweepy
from dotenv import load_dotenv
from datetime import datetime, timezone
import time
import csv
import sys

def load_env_variables():
    """Load environment variables from the .env file."""
    load_dotenv()
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    
    if not bearer_token:
        raise ValueError("Missing required Twitter Bearer Token.")
    
    return bearer_token

def initialize_twitter_client(bearer_token):
    """Initialize and return the Twitter API client for API v2."""
    client = tweepy.Client(bearer_token=bearer_token)
    return client

def handle_rate_limit(retry_count=0, max_retries=5):
    """Simple backoff mechanism for rate limit handling."""
    if retry_count >= max_retries:
        print("Maximum retries reached. Exiting.")
        raise Exception("Rate limit exceeded and maximum retries attempted.")
    
    wait_time = 2 ** retry_count  # Exponential backoff
    print(f"Rate limit hit. Retrying in {wait_time} seconds...")
    time.sleep(wait_time)

def format_tweet_data(tweet, author, ticker):
    """
    Format tweet data into a single line with essential metadata.
    Format: [UTC_TIMESTAMP]|$TICKER|@handle(Real Name)|[followers]|{engagement}|"tweet_content"
    Engagement format: L[likes]R[retweets]Q[quotes]C[comments]
    """
    timestamp = tweet.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')
    handle = f"@{author.username}"
    name = author.name if hasattr(author, 'name') else ''
    followers = author.public_metrics['followers_count'] if hasattr(author, 'public_metrics') else 0
    
    # Engagement metrics
    likes = tweet.public_metrics['like_count']
    retweets = tweet.public_metrics['retweet_count']
    quotes = tweet.public_metrics['quote_count']
    replies = tweet.public_metrics['reply_count']
    engagement = f"L{likes}R{retweets}Q{quotes}C{replies}"
    
    # Clean tweet text for CSV
    clean_text = tweet.text.replace('\n', ' ').replace('\r', ' ')
    
    # Related tickers
    related_tickers = []
    if hasattr(tweet, 'entities') and 'hashtags' in tweet.entities:
        related_tickers = [
            tag['tag'][1:]  # Remove the '$' prefix
            for tag in tweet.entities['hashtags']
            if tag['tag'].startswith('$') and tag['tag'][1:].upper() != ticker.upper()
        ]

    return f"{timestamp}|${ticker}|{handle}({name})|{followers}|{engagement}|{','.join(related_tickers) if related_tickers else 'NONE'}|{clean_text}"

def get_ticker_tweets_metadata(ticker, max_results=1000, output_file=None):
    """
    Retrieve tweets containing the specified ticker symbol and output concise metadata.
    """
    bearer_token = load_env_variables()
    client = initialize_twitter_client(bearer_token)
    
    ticker = ticker.upper()
    query = f"{ticker} lang:en -is:retweet"  # Adjusted for plain ticker query
    
    print(f"\nFetching up to {max_results} tweets for {ticker}...")
    total_tweets = 0
    pagination_token = None
    retry_count = 0
    
    # Setup CSV writer if output file is specified
    csv_file = None
    csv_writer = None
    if output_file:
        csv_file = open(output_file, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp', 'Ticker', 'Handle', 'Name', 'Followers', 
                             'Likes', 'Retweets', 'Quotes', 'Replies', 
                             'Related_Tickers', 'Tweet_Content'])

    try:
        while total_tweets < max_results:
            results_per_page = min(100, max_results - total_tweets)
            try:
                response = client.search_recent_tweets(
                    query=query,
                    max_results=results_per_page,
                    next_token=pagination_token,
                    tweet_fields=['created_at', 'public_metrics', 'entities', 'author_id'],
                    user_fields=['name', 'username', 'public_metrics'],
                    expansions=['author_id']
                )
                retry_count = 0  # Reset retry count on success
                
                if not response.data:
                    break
                
                users = {user.id: user for user in response.includes.get('users', [])}
                
                for tweet in response.data:
                    if f"${ticker}" in tweet.text:  # Validate ticker presence
                        total_tweets += 1
                        author = users.get(tweet.author_id)
                        formatted_data = format_tweet_data(tweet, author, ticker)
                        print(formatted_data)
                        
                        if csv_writer:
                            csv_writer.writerow(formatted_data.split('|'))
                
                if 'next_token' not in response.meta:
                    break
                    
                pagination_token = response.meta['next_token']
                time.sleep(1)  # Delay between pages

            except tweepy.TooManyRequests:
                handle_rate_limit(retry_count)
                retry_count += 1

            except Exception as e:
                print(f"Error during processing: {e}")
                break

    finally:
        if csv_file:
            csv_file.close()
        print(f"\nProcessed {total_tweets} tweets. Results saved to {output_file}" if output_file else "Processing complete.")

if __name__ == "__main__":
    ticker = input("Enter the ticker symbol (without $): ")
    output_file = input("Enter output CSV filename (or press Enter for console output only): ").strip()
    output_file = output_file if output_file else None
    get_ticker_tweets_metadata(ticker, output_file=output_file)
