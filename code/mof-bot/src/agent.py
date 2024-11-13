import os
import signal
import time
import sys
import numpy as np
import tweepy
import asyncio
from datetime import datetime, timedelta
from rich.console import Console

import setup
import splash
import result
import fools_content

from dbh import DBH

from cores.avbcore_manager import AVBCoreManager

from worker_pick_lore import pick_lore
from worker_pick_foolish_content import pick_n_posts
from worker_pick_random_effects import pick_effects
from worker_mixture_of_fools_llm import try_mixture
from worker_send_tweet import send_tweet

from logger import EventLogger
from scheduled_event import ScheduledEvent

from tick.manager import TickManager  # Import TickManager
from tick.tick_exceptions import TickManagerHeartbeatError  # Import the heartbeat exception

from dotenv import load_dotenv

console = Console()

load_dotenv()
DEBUGGING = os.getenv("DEBUGGING")

TICK_INTERVAL_MS = 1000  # 1000ms = 1 second

LOG_DIR = os.path.join(os.path.dirname(__file__), "../log/")
LOG_FILE = os.path.join(LOG_DIR, "agent.log")
HEARTBEAT_FILE = os.path.join(LOG_DIR, "heartbeat.log")

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

# Initialize CoreManager instance (to pass into TickManager)
cores = AVBCoreManager()

# Scheduler list to hold events
scheduler_list = []
previous_post = ""

# Asynchronous shutdown function
async def shutdown():
    """Asynchronously stops the TickManager and shuts down cores."""
    logger.async_log("Interrupt received, shutting down gracefully...")
    await tick_manager.stop()  # Gracefully stops the TickManager
    cores.shutdown()
    print("\nInterrupt received, shutting down gracefully...")

# Signal handler that schedules the async shutdown function
def signal_handler(sig, frame):
    asyncio.create_task(shutdown())  # Schedule shutdown as a task on the event loop
    
# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Define the TickManager at a module level so it can be accessed in the signal handler
tick_manager = TickManager(
    tick_interval_ms=TICK_INTERVAL_MS,
    console=console,
    heartbeat_file=HEARTBEAT_FILE,
    logger=logger,
    cores=cores
)

def has_time_remaining(time_start):
    time_elapsed = (time.time() - time_start) * 1000  # Convert to milliseconds
    return time_elapsed < TICK_INTERVAL_MS

def execute(time_start, job_queue, results_queue):
    global previous_post
    now = datetime.now()

    # Iterate over scheduled events
    for event in scheduler_list:
        if not event.completed:
            if not event.content:
                logger.async_log("Generating content for scheduled tweet.")
                event.content = create_tweet_content(previous_post)
                
            if event.event_time <= now and event.content:
                try:
                    if not DEBUGGING:
                        send_tweet(event.content, logger.async_log)
                        
                    logger.async_log(f"Tweet sent successfully: {event.content}")
                    event.completed = True
                    event.backoff_time = 0
                    previous_post = event.content
                except tweepy.errors.TooManyRequests as e:
                    logger.async_log(f"Rate limit error while sending tweet: {e}")
                    event.apply_backoff()
                except tweepy.errors.TweepyException as e:
                    logger.async_log(f"Error while sending tweet: {e}")
                    event.apply_backoff()
                except Exception as e:
                    logger.async_log(f"Unexpected error while sending tweet: {e}")
                    event.apply_backoff()

    if not any(event for event in scheduler_list if not event.completed):
        pass  # DEBUGGING, prepare_tweet_for_scheduling()

def prepare_tweet_for_scheduling():
    delay_minutes = int(np.random.normal(loc=25, scale=10))
    delay_minutes = max(5, min(80, delay_minutes))
    
    if DEBUGGING:
        delay_minutes = 1

    event_time = datetime.now() + timedelta(minutes=delay_minutes)
    logger.async_log(f"Scheduled a new tweet event at {event_time}.")
    scheduler_list.append(ScheduledEvent(event_time, "Scheduled tweet post"))

def create_tweet_content(post_prev):
    try:
        lore = pick_lore()
        posts = pick_n_posts(3, fools_content)
        effects = pick_effects()
        tweet = try_mixture(posts, post_prev, lore, effects, logger.async_log)
        logger.async_log(f"Prepared tweet content: {tweet}")
        return tweet
    except Exception as e:
        logger.async_log(f"Error while preparing tweet content: {e}")
        return None

async def main():
    try:
        # Start the tick manager
        await tick_manager.initialize_and_start()
        logger.async_log("TickManager stopped successfully.")
        
    except TickManagerHeartbeatError as e:
        logger.async_log(f"Agent startup aborted: {e}", color="red")
        sys.exit(1)
    except Exception as e:
        logger.async_log(f"Unexpected startup error: {e}", color="red")
        await tick_manager.stop()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())