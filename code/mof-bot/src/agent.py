import time
import signal
import sys

TICK = 1000 # 1000ms = 1 second

running = True

def signal_handler(sig, frame):
    global running
    running = False
    print("\nInterrupt received, shutting down gracefully...")

signal.signal(signal.SIGINT, signal_handler)

def execute():
    """
    Placeholder for the main work being performed in the loop
    """
    print("Performing work...")

def tick():
    """
    Main agent loop that executes repeatedly based on the TICK interval
    """
    
    while running:
        time_start = time.time()
        
        print("Executing tick...")
        
        execute()
        
        time_elapsed = (time.time() - time_start) * 1000
        time_sleep = max(0, TICK - time_elapsed) / 1000
        
        time.sleep(time_sleep)
        
if __name__ == "__main__":
    print("Starting agent...")
    tick()
    print("Agent stopped.")
