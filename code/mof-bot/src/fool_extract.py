import os
import json
import tweepy
import time
from dotenv import load_dotenv

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

def extract_content_from_fool(handle, max_tweets=1000):
    """
    Extract tweet content, engagement metrics, and other key information from a specified handle's timeline
    and save it as a JSON file.

    Parameters:
    - handle (str): Twitter handle to extract tweets from (without '@').
    - max_tweets (int): Maximum number of tweets to retrieve. Default is 1000.

    This function captures the following data for each tweet:
    - text: Full text of the tweet.
    - hashtags: List of hashtags used in the tweet.
    - mentions: List of mentioned usernames in the tweet.
    - tickers: List of cryptocurrency tickers (e.g., $AVB) referenced in the tweet.
    - retweet_count: Number of retweets.
    - like_count: Number of likes.
    - quote_count: Number of quotes.
    - reply_count: Number of replies.
    - created_at: Timestamp of when the tweet was created in ISO format.

    The function handles pagination and rate limits for large extractions.
    """
    bearer_token = load_env_variables()
    client = initialize_twitter_client(bearer_token)
    
    try:
        # Get user ID by handle
        user = client.get_user(username=handle)
        user_id = user.data.id

        tweet_texts = []
        pagination_token = None

        # Fetch tweets in batches until max_tweets is reached or no more tweets are available
        while len(tweet_texts) < max_tweets:
            try:
                response = client.get_users_tweets(
                    user_id,
                    max_results=100,
                    pagination_token=pagination_token,
                    tweet_fields=['text', 'entities', 'public_metrics', 'created_at']
                )
                
                if response.data:
                    for tweet in response.data:
                        # Extract engagement metrics
                        retweet_count = tweet.public_metrics["retweet_count"]
                        like_count = tweet.public_metrics["like_count"]
                        quote_count = tweet.public_metrics["quote_count"]
                        reply_count = tweet.public_metrics["reply_count"]

                        # Extract entities (hashtags, mentions, cashtags)
                        hashtags, mentions, tickers = [], [], []
                        if tweet.entities:
                            if "hashtags" in tweet.entities:
                                hashtags = [hashtag["tag"] for hashtag in tweet.entities["hashtags"]]
                            if "mentions" in tweet.entities:
                                mentions = [mention["username"] for mention in tweet.entities["mentions"]]
                            if "cashtags" in tweet.entities:
                                tickers = [cashtag["tag"] for cashtag in tweet.entities["cashtags"]]

                        # Construct tweet data dictionary
                        tweet_data = {
                            "text": tweet.text,
                            "hashtags": hashtags,
                            "mentions": mentions,
                            "tickers": tickers,
                            "retweet_count": retweet_count,
                            "like_count": like_count,
                            "quote_count": quote_count,
                            "reply_count": reply_count,
                            "created_at": tweet.created_at.isoformat()  # Convert datetime to string
                        }
                        tweet_texts.append(tweet_data)
                else:
                    break  # No more tweets available

                # Stop if we have reached the max_tweets limit
                if len(tweet_texts) >= max_tweets:
                    break

                # Update the pagination token for the next request
                pagination_token = response.meta.get('next_token')
                if not pagination_token:
                    break  # No more pages available

            except tweepy.errors.TooManyRequests as e:
                print("Rate limit hit. Sleeping for 15 minutes...")
                time.sleep(15 * 60)  # Wait for 15 minutes before retrying
            except tweepy.errors.HTTPException as e:
                print(f"An HTTP error occurred: {e}")
                break

        # Limit the result to the max_tweets specified
        tweet_texts = tweet_texts[:max_tweets]

        # Create the folder if it doesn't exist
        data_folder = f"../data/fools/"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        # Save tweets to a JSON file
        with open(f"{data_folder}{handle}.json", 'w', encoding='utf-8') as f:
            json.dump(tweet_texts, f, ensure_ascii=False, indent=4)
        
        print(f"Extracted {len(tweet_texts)} tweets from @{handle} and saved to {data_folder}{handle}.json")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    handle = input("Enter the Twitter handle (without @): ")
    extract_content_from_fool(handle, max_tweets=1000)