import time
import signal
import sys
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
    def __init__(self, event_time, description="", backoff_time=0):
        self.event_time = event_time
        self.description = description
        self.completed = False
        self.content = None  # Holds content if generated
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

    # Iterate over scheduled events
    for event in scheduler_list:
        if not event.completed:
            # Immediately create content if it's not already created
            if not event.content:
                print("Generating content for scheduled tweet.")
                event.content = create_tweet_content()

            # Check if the timestamp has been reached and send the tweet if content is ready
            if event.event_time <= now and event.content:
                try:
                    send_tweet("fool_handle", event.content)
                    print(f"Tweet sent successfully at {now}.")
                    event.completed = True
                    event.backoff_time = 0  # Reset backoff after successful send
                except Exception as e:
                    print(f"Error while sending tweet: {e}")
                    event.apply_backoff()

    # If no active events, schedule a new one
    if not any(event for event in scheduler_list if not event.completed):
        prepare_tweet_for_scheduling()

def prepare_tweet_for_scheduling():
    delay_minutes = int(np.random.normal(loc=25, scale=10))
    delay_minutes = max(5, min(45, delay_minutes))

    event_time = datetime.now() + timedelta(seconds=10) #timedelta(minutes=delay_minutes)
    print(f"Scheduled a new tweet event at {event_time}.")
    scheduler_list.append(ScheduledEvent(event_time, "Scheduled tweet post"))

def create_tweet_content():
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

def send_tweet(handle, tweet):
    print(f"Sending tweet for @{handle}: {tweet}")
    # Replace this with actual API call logic

def tick():
    job_queue = queue.Queue()
    results_queue = queue.Queue()
    console = Console()
    
    with Live(console=console, refresh_per_second=4) as live:
        while running:
            time_start = time.time()
            
            # Display the spinner and current epoch time
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