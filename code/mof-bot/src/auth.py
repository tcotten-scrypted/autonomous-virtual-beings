from flask import Flask, request, redirect, session
import tweepy
import webbrowser
import threading
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
consumer_key = os.getenv("TWITTER_API_KEY")
consumer_secret = os.getenv("TWITTER_API_SECRET")

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for Flask session handling

# Set up OAuth 1.0a user authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret, 'http://localhost:5000/callback')

@app.route('/')
def start_auth():
    """Starts the OAuth process and redirects to Twitter's authorization URL."""
    try:
        redirect_url = auth.get_authorization_url()
        session['request_token'] = auth.request_token
        return redirect(redirect_url)
    except tweepy.TweepError:
        return "Error! Failed to get request token."

@app.route('/callback')
def oauth_callback():
    """Handles the callback from Twitter and finalizes the OAuth flow."""
    request_token = session.get('request_token')
    session.pop('request_token', None)
    auth.request_token = request_token

    verifier = request.args.get('oauth_verifier')
    try:
        auth.get_access_token(verifier)
        return f"Access Token: {auth.access_token}<br>Access Token Secret: {auth.access_token_secret}"
    except tweepy.TweepError:
        return "Error! Failed to get access token."

def open_browser():
    """Opens the web browser to start the OAuth process."""
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    # Start the Flask app in a separate thread
    threading.Timer(1, open_browser).start()
    app.run(port=5000)
