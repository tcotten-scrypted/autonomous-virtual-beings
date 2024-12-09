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

def initialize_clients():
    """Initialize Tweepy v2 Client for tweets and Tweepy v1.1 API for media uploads."""
    env_vars = load_env_variables()

    # v2 Client for creating tweets
    client_v2 = tweepy.Client(
        consumer_key=env_vars["consumer_key"],
        consumer_secret=env_vars["consumer_secret"],
        access_token=env_vars["access_token"],
        access_token_secret=env_vars["access_token_secret"]
    )

    # v1.1 API for media uploads
    api_v1 = tweepy.API(
        tweepy.OAuth1UserHandler(
            env_vars["consumer_key"],
            env_vars["consumer_secret"],
            env_vars["access_token"],
            env_vars["access_token_secret"]
        )
    )

    return client_v2, api_v1

def upload_media(api_v1, image_path):
    """Upload media using Tweepy v1.1 API and return media ID."""
    try:
        media = api_v1.media_upload(image_path)
        print(f"Image uploaded successfully. Media ID: {media.media_id}")
        return media.media_id
    except Exception as e:
        print(f"Media upload failed: {e}")
        raise

def send_tweet(tweet, image_path=None, log_event=None):
    """Send a tweet using the Twitter API v2, optionally attaching an image."""
    if log_event:
        log_event(f"Sending tweet: {tweet}")
    print(f"Sending tweet: {tweet}")

    client_v2, api_v1 = initialize_clients()

    try:
        media_id = None

        # Upload media if an image path is provided
        if image_path:
            if log_event:
                log_event(f"Uploading image: {image_path}")
            print(f"Uploading image: {image_path}")
            media_id = upload_media(api_v1, image_path)

        # Send tweet with or without media
        if media_id:
            response = client_v2.create_tweet(text=tweet, media_ids=[media_id])
        else:
            response = client_v2.create_tweet(text=tweet)

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