#!/usr/bin/env python3
"""
Test suite for the Timeline Engine
"""
import unittest
import time
import logging
from timeline_engine import TimelineEngine, TemporalEvent

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)

class TestTimelineEngine(unittest.TestCase):
    """Test cases for the TimelineEngine class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.engine = TimelineEngine()
    
    def sample_observer(self, event, timeline_idx):
        """Sample observer function for testing"""
        logging.debug(f"Observer received event: {event.event_type} at {event.timestamp}")
    
    def test_initialization(self):
        """Test that the engine initializes correctly"""
        self.assertEqual(self.engine.master_tick, 0)
        self.assertEqual(self.engine.current_branch_id, "main")
        self.assertIn("main", self.engine.branches)
    
    def test_register_observer(self):
        """Test registering an observer"""
        initial_observers = len(self.engine.observers)
        self.engine.register_observer(self.sample_observer)
        self.assertEqual(len(self.engine.observers), initial_observers + 1)
    
    def test_create_branch(self):
        """Test creating a new branch"""
        branch_id = self.engine.create_branch("test_branch")
        self.assertEqual(branch_id, "test_branch")
        self.assertIn("test_branch", self.engine.branches)
    
    def test_switch_branch(self):
        """Test switching to a different branch"""
        self.engine.create_branch("test_branch")
        self.engine.switch_branch("test_branch")
        self.assertEqual(self.engine.current_branch_id, "test_branch")
    
    def test_add_event(self):
        """Test adding an event to a branch"""
        event = TemporalEvent(
            timestamp=time.time(),
            event_type="test_event",
            data={"test": "data"}
        )
        initial_events = len(self.engine.branches["main"].events)
        self.engine.add_event(event)
        self.assertEqual(len(self.engine.branches["main"].events), initial_events + 1)
    
    def test_advance_time(self):
        """Test advancing time"""
        initial_tick = self.engine.master_tick
        result = self.engine.advance_time()
        self.assertEqual(self.engine.master_tick, initial_tick + 1)
        self.assertIn('master_tick', result)
        self.assertIn('current_time', result)

if __name__ == '__main__':
    unittest.main()