# tick/manager.py

import asyncio
import os
import sys
from datetime import datetime
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
import aiofiles
from logger import EventLogger
from cores.avbcore_manager import AVBCoreManager
from cores.avbcore_exceptions import AVBCoreRegistryFileError, AVBCoreLoadingError
from .tick_exceptions import TickManagerHeartbeatError

class TickManager:
    def __init__(self, tick_interval_ms, console, heartbeat_file, logger, cores, max_retries=3, retry_delay=0.1):
        """
        Initializes the TickManager with a specified tick interval, heartbeat file, and core manager.

        Parameters:
        ----------
        tick_interval_ms : int
            The tick interval in milliseconds.
        console : Console
            Rich Console instance for displaying Tick updates.
        heartbeat_file : str
            Path to the heartbeat file used to signal system health.
        logger : EventLogger
            An instance of EventLogger to log events asynchronously.
        cores : AVBCoreManager
            Instance of AVBCoreManager to handle core operations.
        max_retries : int, optional
            Maximum number of retries if heartbeat update fails (default is 3).
        retry_delay : float, optional
            Delay in seconds between retry attempts (default is 0.1 seconds).
        """
        self.tick_interval = tick_interval_ms / 1000.0  # Convert ms to seconds
        self.console = console
        self.heartbeat_file = heartbeat_file
        self.logger = logger
        self.cores = cores
        self.running = True
        self.tick_event = asyncio.Event()  # Event to notify agents of a new tick
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def _check_and_create_heartbeat(self):
        """Check if a heartbeat file exists and create one if not, else raise TickManagerHeartbeatError."""
        if os.path.exists(self.heartbeat_file):
            last_heartbeat = await self._read_heartbeat()
            if last_heartbeat and (datetime.now() - last_heartbeat).seconds < self.tick_interval:
                raise TickManagerHeartbeatError("Another instance of TickManager is already running.")
        
        await self._update_heartbeat()  # Create the heartbeat file if it doesn’t exist

    async def _read_heartbeat(self):
        """Reads the last heartbeat timestamp from the file asynchronously."""
        try:
            async with aiofiles.open(self.heartbeat_file, 'r') as f:
                timestamp_str = await f.read()
                if not timestamp_str.strip():  # Check if the string is empty
                    self.logger.async_log("Heartbeat file is empty, no valid timestamp found.")
                    return None  # or use `datetime.now()` if you prefer a fallback timestamp
                return datetime.fromisoformat(timestamp_str.strip())
        except Exception as e:
            await self.logger.async_log(f"Failed to read heartbeat file: {e}", color="red")
            return None

    async def _update_heartbeat(self):
        """Writes the current timestamp to the heartbeat file, signaling system health."""
        success = False
        attempt = 0
        while not success and attempt < self.max_retries:
            try:
                async with aiofiles.open(self.heartbeat_file, 'w') as f:
                    await f.write(datetime.now().isoformat())
                success = True
            except (OSError, IOError) as e:
                attempt += 1
                self.logger.async_log(f"Failed to write to heartbeat file (attempt {attempt}/{self.max_retries}): {e}", color="red")  # Removed 'await'
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.logger.async_log("Max retries reached. Unable to update heartbeat file.", color="red")  # Removed 'await'

    async def initialize_and_start(self, execute):
        """
        Initializes the TickManager by loading cores and starting the tick loop.
        Ensures the heartbeat is checked and written before loading cores to avoid conflicts.
        """
        try:
            # Check and create the heartbeat file
            await self._check_and_create_heartbeat()
            self.logger.async_log("Heartbeat initialized successfully.")  # Removed 'await'

            # Load cores before starting the tick loop
            await self._load_cores()
            self.logger.async_log("Cores loaded successfully.")  # Removed 'await'
            
            # Start the tick loop
            await self.start_tick_loop(execute)

        except TickManagerHeartbeatError as e:
            self.logger.async_log(f"Startup aborted: {e}", color="red")  # Removed 'await'
            sys.exit(1)
        except (AVBCoreRegistryFileError, AVBCoreLoadingError) as e:
            self.logger.async_log(f"Core loading error: {e}", color="red")  # Removed 'await'
            await self.stop()
            sys.exit(1)
        except Exception as e:
            self.logger.async_log(f"Unexpected error during initialization: {e}", color="red")  # Removed 'await'
            await self.stop()
            sys.exit(1)

    async def _load_cores(self):
        """Attempts to load cores using the CoreManager and handles any initialization errors."""
        try:
            self.cores.load_cores()
        except (AVBCoreRegistryFileError, AVBCoreLoadingError) as e:
            self.logger.async_log(f"Core loading failure: {e}", color="red")
            raise

    async def start_tick_loop(self, execute):
        """Begins the Tick loop, broadcasting Tick events and updating the heartbeat."""
        self.logger.async_log("TickManager started.")
        self.running = True
        with Live(console=self.console, refresh_per_second=4) as live:  # Continuous display with Rich's Live
            while self.running:
                time_start = datetime.now()

                # Update heartbeat for each tick
                await self._update_heartbeat()
                
                # Notify agents of the new Tick
                self.tick_event.set()  # Broadcast the tick
                
                # Key logic to execute during the tick
                execute() 
                
                self.tick_event.clear()  # Reset the event for the next tick
                
                # Display a spinner or current tick status on the console
                current_epoch = int(time_start.timestamp())
                spinner = Spinner("dots", f" Tick | Epoch Time: {current_epoch}")
                live.update(spinner)

                # Wait for the next Tick, adjusting for execution time
                await asyncio.sleep(self.tick_interval)

    async def stop(self):
        """Stops the Tick loop and deletes the heartbeat file asynchronously."""
        self.running = False
        if os.path.exists(self.heartbeat_file):
            try:
                async with aiofiles.open(self.heartbeat_file, 'w') as f:
                    await f.write("")  # Clear the heartbeat file before deletion
                os.remove(self.heartbeat_file)
                self.logger.async_log("Heartbeat file deleted successfully.")
            except Exception as e:
                self.logger.async_log(f"Failed to delete heartbeat file: {e}", color="red")