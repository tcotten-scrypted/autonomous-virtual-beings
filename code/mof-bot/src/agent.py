import os
import time
import signal
import sys
import queue
import numpy as np
import tweepy
import asyncio
from datetime import datetime, timedelta
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

import setup
import splash
import result
import fools_content

from dbh import DBH

from cores.avbcore_manager import AVBCoreManager
from cores.avbcore_exceptions import AVBCoreHeartbeatError, AVBCoreRegistryFileError, AVBCoreLoadingError

from worker_pick_lore import pick_lore
from worker_pick_foolish_content import pick_n_posts
from worker_pick_random_effects import pick_effects
from worker_mixture_of_fools_llm import try_mixture
from worker_send_tweet import send_tweet

from logger import EventLogger

from dotenv import load_dotenv

console = Console()

load_dotenv()
DEBUGGING=os.getenv("DEBUGGING")

TICK = 1000  # 1000ms = 1 second

LOG_DIR = os.path.join(os.path.dirname(__file__), "../log/")
LOG_FILE = os.path.join(LOG_DIR, "agent.log")

logger = EventLogger(console, LOG_FILE)

# Ensure log directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Splash display
splash.display("Westworld (v0.0.2)")

# Load content
fools_content.load_available_content()

# Set database handler
dbh = DBH.get_instance()
db_conn = dbh.get_connection()

# Initialize CoreManager
cores = AVBCoreManager()

# Start the heartbeat, ensuring no conflicting instance is running
try:
    cores.start_heartbeat()
except AVBCoreHeartbeatError as e:
    print(f"Error: {e}")
    print("Agent did not start because a heartbeat file was already present. Ensure no other instance is running.")
    sys.exit(1)

# Load cores from the registry
try:
    cores.load_cores()
except AVBCoreRegistryFileError as e:
    print(f"Error: {e}")
    print("Agent did not start due to an issue with the core registry file. Please check its location and syntax.")
    cores.shutdown()
    sys.exit(1)
except AVBCoreLoadingError as e:
    print(f"Error: {e}")
    print("Agent did not start because a core failed to load or initialize. Please check core configurations.")
    cores.shutdown()
    sys.exit(1)

# Start cores and begin the main loop
cores.start_cores()
print("Core manager is now running. Press Ctrl+C to stop.")

# Running control
running = True

# Scheduler list to hold events
scheduler_list = []

# Last post
previous_post = ""

# Signal handler
def signal_handler(sig, frame):
    # Shut down the cores' heartbeat
    cores.shutdown()
    
    # Stop running the main parent tick
    global running
    running = False
    
    # Log the shutdown
    logger.async_log("Interrupt received, shutting down gracefully...")
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
        logger.async_log(f"Rescheduled with backoff: {self.backoff_time} minute(s)")
        print(f"Rescheduled with backoff: {self.backoff_time} minute(s)")

def has_time_remaining(time_start):
    time_elapsed = (time.time() - time_start) * 1000  # Convert to milliseconds
    return time_elapsed < TICK

def execute(time_start, job_queue, results_queue):
    global previous_post

    now = datetime.now()

    # Iterate over scheduled events
    for event in scheduler_list:
        if not event.completed:
            # Immediately create content if it's not already created
            if not event.content:
                logger.async_log("Generating content for scheduled tweet.")
                print("Generating content for scheduled tweet.")
                event.content = create_tweet_content(previous_post)
                
            # Check if the timestamp has been reached and send the tweet if content is ready
            if event.event_time <= now and event.content:
                try:
                    if not DEBUGGING:
                        send_tweet(event.content, logger.async_log)
                        
                    logger.async_log(f"Tweet sent successfully: {event.content}")
                    print(f"Tweet sent successfully at {now}.")
                    event.completed = True
                    event.backoff_time = 0  # Reset backoff after successful send
                    previous_post = event.content
                except tweepy.errors.TooManyRequests as e:
                    logger.async_log(f"Rate limit error while sending tweet: {e}")
                    print(f"Rate limit error while sending tweet: {e}")
                    event.apply_backoff()
                except tweepy.errors.TweepyException as e:
                    logger.async_log(f"Error while sending tweet: {e}")
                    print(f"Error while sending tweet: {e}")
                    event.apply_backoff()
                except Exception as e:
                    logger.async_log(f"Unexpected error while sending tweet: {e}")
                    print(f"Unexpected error while sending tweet: {e}")
                    event.apply_backoff()

    # If no active events, schedule a new one
    if not any(event for event in scheduler_list if not event.completed):
        # DEBUGGING, DISABLED prepare_tweet_for_scheduling()
        pass

def prepare_tweet_for_scheduling():
    delay_minutes = int(np.random.normal(loc=25, scale=10))
    delay_minutes = max(5, min(80, delay_minutes))
    
    if DEBUGGING:
        delay_minutes = 1

    event_time = datetime.now() + timedelta(minutes=delay_minutes)
    logger.async_log(f"Scheduled a new tweet event at {event_time}.")
    print(f"Scheduled a new tweet event at {event_time}.")
    scheduler_list.append(ScheduledEvent(event_time, "Scheduled tweet post"))

def create_tweet_content(post_prev):
    try:
        lore = pick_lore()
        posts = pick_n_posts(3, fools_content)
        effects = pick_effects()
        tweet = try_mixture(posts, post_prev, lore, effects, logger.async_log)
        logger.async_log(f"Prepared tweet content: {tweet}")
        print(f"Prepared tweet content:\n\n\t{tweet}\n")
        return tweet
    except Exception as e:
        logger.async_log(f"Error while preparing tweet content: {e}")
        print(f"Error while preparing tweet content: {e}")
        return None

def tick():
    job_queue = queue.Queue()
    results_queue = queue.Queue()
    
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
    logger.async_log("Starting agent...")
    print("Starting agent...")
    tick()
    logger.async_log("Agent stopped.")
    print("Agent stopped.")