import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
from unittest.mock import patch
import signal
import agent  # Assuming agent.py is the name of your module

class TestAgent(unittest.TestCase):
    def setUp(self):
        # Announce the name of each test before it runs
        print(f"\n[Running {self._testMethodName}...]")

    def test_signal_handler_sets_running_to_false(self):
        # Arrange
        agent.running = True
        
        # Act
        agent.signal_handler(signal.SIGINT, None)
        
        # Assert
        self.assertFalse(agent.running)

    def test_execute_runs_without_errors(self):
        try:
            agent.execute()
        except Exception as e:
            self.fail(f"execute() raised an exception {e}")

    @patch("time.sleep", return_value=None)  # Mock sleep for fast test execution
    def test_tick_loop_stops_on_interrupt(self, mock_sleep):
        # Arrange
        agent.running = True
        
        # Act
        try:
            agent.running = False  # Directly set running to False to simulate stop after first loop
            agent.tick()
        except Exception as e:
            self.fail(f"tick() raised an exception {e}")
        
        # Assert
        self.assertFalse(agent.running)  # Check that loop stops gracefully

    @patch("time.sleep", return_value=None)  # Mock sleep to speed up test
    @patch("time.time")
    def test_tick_respects_TICK_interval(self, mock_time, mock_sleep):
        MAX_TEST_TICKS = 4
        
        # Arrange
        agent.TICK = 1000  # 1 second
        agent.running = True
        
        # Set up a limited side effect for `time.time()` calls to control the loop;
        # the main loop calls the time function twice (time_start, time_elapsed)
        # so we need 2 * MAX_TEST_TICKS values representing timestamps
        mock_time.side_effect = [
            1698654321, 1698654321.1,    # Trial 1: 0.1 second elapsed
            1698654321.1, 1698654321.6,  # Trial 2: 0.5 seconds elapsed
            1698654321.6, 1698654322.6,  # Trial 3: 1 second elapsed
            1698654322.6, 1698654323.6   # Trial 4: 2 seconds elapsed
        ]

        # Count the number of times execute is called to limit iterations
        call_count = 0
        def limited_execute():
            nonlocal call_count
            call_count += 1
            if call_count >= MAX_TEST_TICKS:  # Stop after 4 executions
                agent.running = False

        # Act
        with patch("agent.execute", side_effect=limited_execute) as mock_execute:
            agent.tick()

        # Assert
        self.assertEqual(mock_execute.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 4)
if __name__ == "__main__":
    unittest.main()
