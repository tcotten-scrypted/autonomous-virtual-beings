from datetime import datetime, timedelta

class ScheduledEvent:
    """
    Represents a scheduled event with timing, content, and backoff properties
    for autonomous execution with retry mechanisms.

    Attributes:
    ----------
    event_time : datetime
        The scheduled time for the event to be executed.
    description : str
        A brief description of the event's purpose.
    completed : bool
        Status of event completion; True if completed, False otherwise.
    content : str or None
        Holds content related to the event (e.g., tweet text), set when generated.
    backoff_time : int
        The time in minutes to wait before retrying the event on failure. Initially set to 0,
        but adjusts dynamically based on retry attempts.

    Methods:
    -------
    apply_backoff():
        Applies an exponential backoff strategy by doubling the backoff time and rescheduling 
        `event_time` accordingly.
    """

    def __init__(self, event_time, description="", backoff_time=0, logger=None):
        """
        Initializes a new scheduled event with the specified time, description, and initial backoff.

        Parameters:
        ----------
        event_time : datetime
            The time when the event is initially scheduled to occur.
        description : str, optional
            A short description of the event (default is an empty string).
        backoff_time : int, optional
            Initial time in minutes to delay retries if the event fails (default is 0).
        logger : EventLogger, optional
            An instance of EventLogger to handle logging; if not provided, logging will be skipped.
        """
        self.event_time = event_time
        self.description = description
        self.completed = False
        self.content = None
        self.backoff_time = backoff_time
        self.logger = logger

    def apply_backoff(self):
        """
        Adjusts the event's `event_time` by doubling the backoff interval. If `backoff_time` is 
        initially zero, sets it to 5 minutes before applying exponential backoff.

        This mechanism is used to delay the retry of an event that previously failed,
        thereby reducing the frequency of retries in the case of repeated errors.

        Logs each reschedule attempt, specifying the new backoff interval.
        """
        if self.backoff_time == 0:
            self.backoff_time = 5
        else:
            self.backoff_time *= 2
        self.event_time += timedelta(minutes=self.backoff_time)
        if self.logger:
            self.logger.async_log(f"Rescheduled with backoff: {self.backoff_time} minute(s)")
        else:
            print(f"Rescheduled with backoff: {self.backoff_time} minute(s)")