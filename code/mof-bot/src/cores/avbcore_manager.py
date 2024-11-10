import json
import time
import threading
import importlib

from pathlib import Path
from operator import attrgetter

from cores.avbcore_exceptions import AVBCoreHeartbeatError, AVBCoreRegistryFileError, AVBCoreLoadingError

class AVBCoreManager:
    """
    AVBCoreManager with a heartbeat for non-graceful termination detection.
    Writes a heartbeat timestamp to a file periodically and manages
    asynchronous core execution.

    Attributes:
    ------------
    - registry_path : Path
        Path to the core registry JSON file.
    - heartbeat_path : Path
        Path to the heartbeat file for tracking CoreManager status.
    - cores : list
        List of initialized core instances.
    - threads : list
        List of threads handling each core.
    - shutdown_event : threading.Event
        Event used to signal all cores to shut down gracefully.
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
        Raises AVBCoreHeartbeatError if the heartbeat file already exists.
        """
        if self.heartbeat_path.exists():
            raise AVBCoreHeartbeatError("Heartbeat file already exists. Another instance may be running.")

        def write_heartbeat():
            while not self.shutdown_event.is_set():
                with self.heartbeat_path.open("w") as f:
                    f.write(str(time.time()))  # Write the current timestamp
                time.sleep(5)  # Update every 5 seconds

        threading.Thread(target=write_heartbeat, daemon=True).start()

    def load_cores(self):
        """
        Loads core definitions from the JSON registry, initializes each core,
        and sorts them by priority.
        Raises:
        - AVBCoreRegistryFileError if the registry file is missing or invalid.
        - AVBCoreLoadingError if a core cannot be imported or initialized.
        """
        # Check if the registry file exists
        if not self.registry_path.exists():
            raise AVBCoreRegistryFileError(f"Registry file not found at {self.registry_path}")

        try:
            # Load registry file
            with self.registry_path.open("r") as file:
                registry = json.load(file)

            for core_def in registry.get("cores", []):
                # Check for required keys in each core definition
                if not all(key in core_def for key in ("file", "class", "name", "priority")):
                    raise AVBCoreRegistryFileError("Missing required key in core definition: 'file', 'class', 'name', or 'priority'.")

                file_name = core_def["file"].replace(".py", "")
                class_name = core_def["class"]
                name = core_def["name"]
                priority = core_def["priority"]

                # Dynamically import the core module and class
                try:
                    module = importlib.import_module(file_name)
                    core_class = getattr(module, class_name)
                except (ImportError, AttributeError) as e:
                    raise AVBCoreLoadingError(f"Error loading core '{class_name}' from '{file_name}': {e}")

                # Instantiate the core and set its priority and shutdown event
                try:
                    core_instance = core_class()
                    core_instance.priority = priority
                    core_instance.name = name
                    core_instance.shutdown_event = self.shutdown_event
                except Exception as e:
                    raise AVBCoreLoadingError(f"Failed to initialize core '{class_name}': {e}")

                # Add the core to the list
                self.cores.append(core_instance)

            # Sort cores by priority (lower numbers mean higher priority)
            self.cores.sort(key=attrgetter("priority"))
            print(f"Loaded {len(self.cores)} cores in priority order.")

        except json.JSONDecodeError as e:
            raise AVBCoreRegistryFileError(f"Invalid JSON in registry file {self.registry_path}: {e}")

    def start_cores(self):
        """
        Starts each core in a separate thread and begins the heartbeat.
        """
        self.start_heartbeat()  # Start the heartbeat thread
        
        for core in self.cores:
            thread = threading.Thread(target=core.run, name=f"{core.name}_thread")
            thread.start()
            self.threads.append(thread)
            print(f"Started {core.name} core in a separate thread.")

    def shutdown(self):
        """
        Signals all cores to shut down and waits for threads to complete.
        """
        self.shutdown_event.set()  # Signal all cores to stop

        # Wait for each thread to complete
        for thread in self.threads:
            thread.join()
        print("All cores have shut down.")

        # Remove the heartbeat file on graceful shutdown
        if self.heartbeat_path.exists():
            self.heartbeat_path.unlink()
