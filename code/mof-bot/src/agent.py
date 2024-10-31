import time
import signal
import sys
import threading
import queue

import splash
import result
import fools_content

from worker_pick_foolish_content import pick_two_posts
from worker_pick_random_effects import pick_effects
from worker_mixture_of_fools_llm import try_mixture

TICK = 1000 # 1000ms = 1 second

### SPLASH
splash.display()

### CONFIG
# this is stubbed, will fill in later

### CONTENT
fools_content.load_available_content()

### START

running = True

def signal_handler(sig, frame):
    global running
    running = False
    print("\nInterrupt received, shutting down gracefully...")

signal.signal(signal.SIGINT, signal_handler)

def has_time_remaining(time_start):
    """
    Determines if there is time remaining in the current tick based on TICK interval.
    
    Args:
        time_start (float): The starting time of the current tick in seconds since epoch.
    
    Returns:
        bool: True if there is time left in the tick, False if the tick interval has been exceeded.
    """
    # Calculate elapsed time in milliseconds
    time_elapsed = (time.time() - time_start) * 1000  # Convert to milliseconds
    
    # Check if the elapsed time is less than TICK
    return time_elapsed < TICK

def execute(time_start, job_queue, results_queue):
    """
    Main work performed in each tick: select two posts from distinct fools.
    """
    
    # Process the results_queue first, which may influence the job_queue
    #while not results_queue.empty() and has_time_remaining(time_start):
    #    result = results_queue.get()
    
    ### TESTING STUB: grab two posts
    posts = pick_two_posts(fools_content)
        
    ### TESTING STUB: create random effects
    effects = pick_effects()
    
    ### TESTING STUB: execute LLM attempt using OpenAI GPT-4o
    tweet = try_mixture(posts, effects)
    
    print("I want to tweet:\n\n\t{tweet}\n")

def tick():
    """
    Main agent loop that executes repeatedly based on the TICK interval
    """
    
    job_queue = queue.Queue()
    results_queue = queue.Queue()
    
    while running:
        time_start = time.time()
        
        print("Executing tick...")
        
        execute(time_start, job_queue, results_queue)
        
        time_elapsed = (time.time() - time_start) * 1000
        time_sleep = max(0, TICK - time_elapsed) / 1000
        
        time.sleep(time_sleep)
        
if __name__ == "__main__":
    print("Starting agent...")
    tick()
    print("Agent stopped.")