from datetime import datetime
from rich.console import Console
import asyncio
from asyncio import Lock
import aiofiles
import threading

class EventLogger:
    """
    A singleton class for logging events asynchronously, providing a centralized
    way to record actions, messages, and errors across different modules. The class
    maintains a background event loop in a separate thread for non-blocking, concurrent logging.

    Attributes:
    ----------
    console : Console
        A Rich console instance used for colorized output to the terminal.
    log_file : str
        Path to the file where log messages will be saved.
    lock : Lock
        An asyncio lock to ensure only one write operation occurs at a time.
    loop : AbstractEventLoop
        The background event loop dedicated to logging tasks.
    initialized : bool
        A flag to check if the singleton has already been initialized.
    _instance : EventLogger or None
        Holds the singleton instance of EventLogger.
    """

    _instance = None  # Singleton instance

    def __new__(cls, *args, **kwargs):
        """
        Overrides instance creation to implement the singleton pattern.
        Ensures that only one instance of EventLogger exists.

        Returns:
        -------
        EventLogger
            The single instance of EventLogger.
        """
        if cls._instance is None:
            cls._instance = super(EventLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, console: Console, log_file: str):
        """
        Initializes the EventLogger with a Rich console and log file path. Starts a
        background thread that runs an event loop dedicated to handling asynchronous
        logging tasks.

        Parameters:
        ----------
        console : Console
            A Rich Console instance for displaying colored log messages.
        log_file : str
            Path to the log file where events will be saved.
        """
        if not hasattr(self, "initialized"):  # Prevents reinitialization in singleton
            self.console = console
            self.log_file = log_file
            self.lock = Lock()  # Async lock for concurrent writes
            self.loop = asyncio.new_event_loop()  # Background event loop
            threading.Thread(target=self._start_event_loop, daemon=True).start()  # Start loop in background thread
            self.initialized = True  # Marks this instance as initialized

    def _start_event_loop(self):
        """
        Private method to start the background event loop, allowing async logging
        tasks to run concurrently without blocking the main application thread.
        This method is run in a separate thread.
        """
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def log_event(self, message, color="white"):
        """
        Asynchronously logs a message with a timestamp to both a log file and the console.

        Parameters:
        ----------
        message : str
            The message to log, describing the event or action taken.
        color : str, optional
            The color to display the log message in the console. Defaults to 'white'.

        Output:
        ------
        Writes the log message asynchronously to the log file with a timestamp, and prints
        the message to the console with the specified color.
        """
        # Generate a timestamp for the log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Async write to the log file, ensuring exclusive access with a lock
        async with self.lock:
            async with aiofiles.open(self.log_file, "a", encoding='utf-8') as log_file:
                await log_file.write(f"[{timestamp}] {message}\n")

        # Print to console with the specified color
        self.console.print(f"[{timestamp}] {message}", style=color)

    def async_log(self, message, color="white"):
        """
        A wrapper for log_event that schedules it to run on the background event loop.
        Ensures non-blocking behavior by using asyncio.run_coroutine_threadsafe to
        safely execute the coroutine within the background thread's event loop.

        Parameters:
        ----------
        message : str
            The message to log.
        color : str, optional
            The color to display in the console. Defaults to 'white'.
        """
        asyncio.run_coroutine_threadsafe(self.log_event(message, color=color), self.loop)