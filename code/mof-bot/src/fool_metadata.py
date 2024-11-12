import os
import tweepy
from dotenv import load_dotenv
from datetime import datetime, timezone

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

def get_latest_tweets_metadata(handle, num_tweets=5):
    """
    Retrieve the latest tweets from the specified Twitter handle and analyze metadata
    for timezone and possible location clues.

    Parameters:
    - handle (str): Twitter handle (without '@') to retrieve tweets from.
    - num_tweets (int): Number of recent tweets to fetch (default is 5).
    """
    bearer_token = load_env_variables()
    client = initialize_twitter_client(bearer_token)
    
    try:
        # Get user by handle
        user = client.get_user(username=handle, user_fields=['location'])
        user_id = user.data.id

        # Fetch the latest tweets (limit to num_tweets)
        response = client.get_users_tweets(
            user_id,
            max_results=num_tweets,
            tweet_fields=['created_at', 'geo', 'context_annotations', 'public_metrics', 'entities']
        )

        if response.data:
            print(f"--- Metadata for the last {num_tweets} tweets from @{handle} ---\n")
            for tweet in response.data:
                print(f"Text: {tweet.text}")
                # Analyze tweet creation time
                created_at_utc = tweet.created_at
                print(f"Tweet created at (UTC): {created_at_utc}")

                # Check for location clues
                user_location = user.data.location if 'location' in user.data else None
                if user_location:
                    print(f"User profile location: {user_location}")
                else:
                    print("No location set in user profile.")

                if tweet.geo:
                    print(f"Tweet geo location metadata: {tweet.geo}")
                else:
                    print("No location metadata in tweet.")

                # Context annotations
                if tweet.context_annotations:
                    print("Tweet context annotations:")
                    for annotation in tweet.context_annotations:
                        print(f" - Domain: {annotation['domain']['name']}, Entity: {annotation['entity']['name']}")
                else:
                    print("No context annotations found in tweet.")

                # Engagement metrics
                print("Engagement metrics:")
                print(f" - Retweets: {tweet.public_metrics['retweet_count']}")
                print(f" - Likes: {tweet.public_metrics['like_count']}")
                print(f" - Quotes: {tweet.public_metrics['quote_count']}")
                print(f" - Replies: {tweet.public_metrics['reply_count']}")

                # Entities (hashtags, mentions, cashtags)
                if tweet.entities:
                    hashtags = [hashtag["tag"] for hashtag in tweet.entities.get("hashtags", [])]
                    mentions = [mention["username"] for mention in tweet.entities.get("mentions", [])]
                    cashtags = [cashtag["tag"] for cashtag in tweet.entities.get("cashtags", [])]
                    print(f"Hashtags: {hashtags}")
                    print(f"Mentions: {mentions}")
                    print(f"Cashtags: {cashtags}")
                else:
                    print("No entities found in tweet.")

                print("\n---------------------------------\n")

        else:
            print(f"No tweets found for @{handle}.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    handle = input("Enter the Twitter handle (without @): ")
    get_latest_tweets_metadata(handle, num_tweets=5)
