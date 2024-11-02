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
    """Extract full tweet text content from the specified handle's timeline and save it as a JSON file."""
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
                    tweet_fields=['text', 'entities', 'public_metrics'],
                    expansions=['referenced_tweets.id']
                )
                
                if response.data:
                    for tweet in response.data:
                        tweet_texts.append(tweet.text)
                else:
                    break  # No more tweets available

                # Stop if we have reached the max_tweets limit
                if len(tweet_texts) >= max_tweets:
                    break

                # Update the pagination token for the next request
                pagination_token = response.meta.get('next_token')
                if not pagination_token:
                    break  # No more pages

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