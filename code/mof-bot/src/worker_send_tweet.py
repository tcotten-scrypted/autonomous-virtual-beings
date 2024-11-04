import tweepy
from tweepy.errors import TweepyException, TooManyRequests
import os
from dotenv import load_dotenv

def load_env_variables():
    """Load environment variables from the .env file."""
    load_dotenv()
    access_token = os.getenv("ACCESS_TOKEN_SENDER")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET_SENDER")
    consumer_key = os.getenv("TWITTER_API_KEY")
    consumer_secret = os.getenv("TWITTER_API_SECRET")

    if not all([access_token, access_token_secret, consumer_key, consumer_secret]):
        raise ValueError("One or more required Twitter API credentials are missing.")

    return {
        "access_token": access_token,
        "access_token_secret": access_token_secret,
        "consumer_key": consumer_key,
        "consumer_secret": consumer_secret,
    }

def initialize_twitter_client():
    """Initialize and return the Twitter API client for user authentication."""
    env_vars = load_env_variables()
    client = tweepy.Client(
        consumer_key=env_vars["consumer_key"],
        consumer_secret=env_vars["consumer_secret"],
        access_token=env_vars["access_token"],
        access_token_secret=env_vars["access_token_secret"]
    )
    return client

def send_tweet(tweet, log_event=None):
    """Send a tweet using the Twitter API v2 with user authentication."""
    if log_event:
        log_event(f"Sending tweet: {tweet}")
    print(f"Sending tweet: {tweet}")

    client = initialize_twitter_client()
    try:
        response = client.create_tweet(text=tweet)
        if response.data and 'id' in response.data:
            if log_event:
                log_event(f"Tweet successfully sent. Tweet ID: {response.data['id']}")
            print(f"Tweet successfully sent. Tweet ID: {response.data['id']}")
        else:
            if log_event:
                log_event("Tweet sent but response data is missing the Tweet ID.")
            print("Tweet sent but response data is missing the Tweet ID.")
    except TooManyRequests as e:
        if log_event:
            log_event(f"Rate limit error: {e}")
        raise  # Re-raise to allow the caller to handle it
    except TweepyException as e:
        if log_event:
            log_event(f"Failed to send tweet: {e}")
        raise  # Re-raise to allow the caller to handle it
    except Exception as e:
        if log_event:
            log_event(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
        raise  # Re-raise to allow the caller to handle it
