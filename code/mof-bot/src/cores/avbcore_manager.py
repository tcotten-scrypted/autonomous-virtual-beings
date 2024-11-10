import json
import time
import threading
from pathlib import Path

class AVBCoreManager:
    """
    AVBCoreManager with a heartbeat for non-graceful termination detection.
    Writes a heartbeat timestamp to a file periodically.
    """
    def __init__(self, registry_path="./core_registry.json", heartbeat_path="../tmp/heartbeat.txt"):
        # Make paths relative to the base directory
        base_dir = Path(__file__).parent.resolve()
        self.registry_path = base_dir / registry_path
        self.heartbeat_path = base_dir / heartbeat_path
        
        self.cores = []
        self.threads = []
        self.shutdown_event = threading.Event()

    def start_heartbeat(self):
        """
        Starts a separate thread to update the heartbeat file every 5 seconds.
        Raises an exception if the heartbeat file already exists.
        """
        if self.heartbeat_path.exists():
            raise RuntimeError("Heartbeat file already exists. Another instance may be running.")
        
        def write_heartbeat():
            while not self.shutdown_event.is_set():
                with self.heartbeat_path.open("w") as f:
                    f.write(str(time.time()))  # Write the current timestamp
                time.sleep(5)  # Update every 5 seconds
        threading.Thread(target=write_heartbeat, daemon=True).start()

    def load_cores(self):
        """
        Load cores as before, initialize and start each core in a thread.
        """
        # (Implementation as in previous examples, loading and initializing cores)
        pass

    def start_cores(self):
        """
        Start cores in separate threads and begin the heartbeat.
        """
        self.start_heartbeat()  # Start heartbeat thread
        # (Rest of the start_cores method as in previous example)
        pass

    def shutdown(self):
        """
        Signal all cores to shut down and wait for threads to finish.
        """
        self.shutdown_event.set()
        # Wait for threads to complete
        for thread in self.threads:
            thread.join()
        print("All cores have shut down.")
        if self.heartbeat_path.exists():
            self.heartbeat_path.unlink()  # Remove heartbeat file upon graceful shutdown