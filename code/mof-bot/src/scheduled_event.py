from datetime import datetime, timedelta

class ScheduledEvent:
    """
    Represents a scheduled event with timing, type, content, and backoff properties
    for autonomous execution with retry mechanisms.

    Attributes:
    ----------
    event_time : datetime
        The scheduled time for the event to be executed.
    event_type : str
        Identifies the type of this event (e.g., "meme", "tweet", "other").
    description : str
        A brief description of the event's purpose.
    completed : bool
        Status of event completion; True if completed, False otherwise.
    content : dict or None
        Holds the generated content for the event (e.g., {'text': "...", 'image': "..."}).
    backoff_time : int
        The time in minutes to wait before retrying the event on failure.
    logger : EventLogger or None
        A logger instance for logging event-related messages. If None, no logging is performed.

    Methods:
    -------
    apply_backoff():
        Applies an exponential backoff strategy by increasing the backoff time and rescheduling
        `event_time` accordingly, used when the event execution fails.
    """

    def __init__(self, event_time, event_type, description="", backoff_time=0, logger=None):
        """
        Initializes a new scheduled event with the specified time, type, and initial backoff.

        Parameters:
        ----------
        event_time : datetime
            The time when the event is initially scheduled to occur.
        event_type : str
            The type of the event, which helps in determining how to handle it.
        description : str, optional
            A short description of the event (default is an empty string).
        backoff_time : int, optional
            Initial time in minutes to delay retries if the event fails (default is 0).
        logger : EventLogger, optional
            An instance of EventLogger to handle logging; if not provided, no logs will be recorded.
        """
        self.event_time = event_time
        self.event_type = event_type
        self.description = description
        self.completed = False
        self.content = None
        self.backoff_time = backoff_time
        self.logger = logger

    def apply_backoff(self):
        """
        Adjust the event's `event_time` by applying exponential backoff.
        If `backoff_time` is zero, set it to 5 minutes. Otherwise, double the current backoff time.
        Reschedule `event_time` by the new backoff interval.

        Logs the rescheduling with updated backoff timing.
        """
        if self.backoff_time == 0:
            self.backoff_time = 5
        else:
            self.backoff_time *= 2

        self.event_time += timedelta(minutes=self.backoff_time)

        if self.logger:
            self.logger.async_log(
                f"Rescheduled {self.event_type} event with backoff: {self.backoff_time} minute(s)"
            )
        else:
            print(
                f"Rescheduled {self.event_type} event with backoff: {self.backoff_time} minute(s)"
            )