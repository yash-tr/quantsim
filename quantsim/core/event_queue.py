"""Manages an event queue for the QuantSim system.

This module provides the `EventQueue` class, a wrapper around Python's
standard `queue.Queue` to facilitate event-driven architecture within the
trading simulation.
"""

import queue
from quantsim.core.events import Event  # For type hinting


class EventQueue:
    """A simple wrapper around Python's `queue.Queue` for event handling.

    This class provides a straightforward interface for putting events onto
    and getting events from a thread-safe queue. It is central to the
    event-driven nature of the backtesting system.

    Attributes:
        _queue (queue.Queue[Event]): The underlying standard library queue instance
            used to store and manage events.
    """

    def __init__(self):
        """Initializes the EventQueue with an empty `queue.Queue`."""
        self._queue: queue.Queue[Event] = queue.Queue()

    def put_event(self, event: Event) -> None:
        """Puts an event onto the end of the queue.

        Args:
            event (Event): The event object to be added to the queue.
        """
        self._queue.put(event)

    def get_event(self) -> Event:
        """Retrieves an event from the front of the queue.

        This method blocks by default until an event is available in the queue.

        Returns:
            Event: The next event from the queue.

        Note:
            The underlying `queue.Queue.get()` can raise `queue.Empty` if
            `block=False` and no item is available, but this wrapper currently
            uses the default blocking behavior.
        """
        return self._queue.get()  # Corrected

    def empty(self) -> bool:
        """Checks if the event queue is currently empty.

        Returns:
            bool: True if the queue is empty, False otherwise.
        """
        return self._queue.empty()
