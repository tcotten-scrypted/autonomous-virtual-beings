import time
import signal
import sys
import threading
import queue
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

import setup
import splash
import result
import fools_content

from worker_pick_lore import pick_lore
from worker_pick_foolish_content import pick_two_posts
from worker_pick_random_effects import pick_effects
from worker_mixture_of_fools_llm import try_mixture

TICK = 1000  # 1000ms = 1 second

# Splash display
splash.display()

# Load content
fools_content.load_available_content()

# Running control
running = True

# Scheduler list to hold events
scheduler_list = []

# Signal handler
def signal_handler(sig, frame):
    global running
    running = False
    print("\nInterrupt received, shutting down gracefully...")

signal.signal(signal.SIGINT, signal_handler)

class ScheduledEvent:
    def __init__(self, event_time, action, description="", backoff_time=0):
        self.event_time = event_time
        self.action = action
        self.description = description
        self.completed = False
        self.backoff_time = backoff_time  # Initial backoff time in minutes

    def apply_backoff(self):
        if self.backoff_time == 0:
            self.backoff_time = 5  # Start with 5 minutes if not set
        else:
            self.backoff_time *= 2  # Double the backoff time
        self.event_time += timedelta(minutes=self.backoff_time)
        print(f"Rescheduled with backoff: {self.backoff_time} minute(s)")

def has_time_remaining(time_start):
    time_elapsed = (time.time() - time_start) * 1000  # Convert to milliseconds
    return time_elapsed < TICK

def execute(time_start, job_queue, results_queue):
    now = datetime.now()

    # Check if there is a scheduled post event and if it's time to execute it
    for event in scheduler_list:
        if not event.completed and event.event_time <= now:
            try:
                # Attempt to send a tweet if content is prepared
                tweet = event.action()
                if tweet:
                    send_tweet("fool_handle", tweet)
                    print(f"Tweet sent successfully at {now}.")
                    event.completed = True
                    event.backoff_time = 0  # Reset backoff on successful send
                else:
                    print("No tweet content found, preparing new content.")
                    prepare_tweet_for_scheduling()
            except Exception as e:
                print(f"Error while sending tweet: {e}")
                event.apply_backoff()
    
    # If no scheduled tweet, prepare a new tweet and schedule it
    if not any(event for event in scheduler_list if not event.completed):
        prepare_tweet_for_scheduling()

def prepare_tweet_for_scheduling():
    """
    Prepares a tweet and schedules it for posting.
    """
    # Schedule a new tweet event using a normal distribution
    delay_minutes = int(np.random.normal(loc=25, scale=10))  # Centered at 25 mins, standard deviation of 10
    delay_minutes = max(5, min(45, delay_minutes))  # Clamping between 5 and 45 mins

    event_time = datetime.now() + timedelta(seconds=10) #timedelta(minutes=delay_minutes)
    print(f"Scheduled a new tweet event at {event_time}.")

    def create_tweet_content():
        """
        Generates and returns the tweet content.
        """
        try:
            lore = pick_lore()
            posts = pick_two_posts(fools_content)
            effects = pick_effects()
            tweet = try_mixture(posts, lore, effects)
            print(f"Prepared tweet content:\n\n\t{tweet}\n")
            return tweet
        except Exception as e:
            print(f"Error while preparing tweet content: {e}")
            return None

    # Initialize the event with no backoff initially
    scheduler_list.append(ScheduledEvent(event_time, create_tweet_content, "Scheduled tweet post"))

def send_tweet(handle, tweet):
    """
    Stub function to simulate sending a tweet. Replace with actual implementation.
    """
    print(f"Sending tweet for @{handle}: {tweet}")
    # Replace this with actual API call logic

def tick():
    job_queue = queue.Queue()
    results_queue = queue.Queue()
    
    console = Console()
        
    with Live(console=console, refresh_per_second=4) as live:
        while running:
            time_start = time.time()
            
            # Update the spinner and current epoch time
            current_epoch = int(time.time())
            spinner = Spinner("dots", f" Tick | Epoch Time: {current_epoch}")
            live.update(spinner)
            
            execute(time_start, job_queue, results_queue)
            
            time_elapsed = (time.time() - time_start) * 1000
            time_sleep = max(0, TICK - time_elapsed) / 1000
            time.sleep(time_sleep)

if __name__ == "__main__":
    print("Starting agent...")
    tick()
    print("Agent stopped.")