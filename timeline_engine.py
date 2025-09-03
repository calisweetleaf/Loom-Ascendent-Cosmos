# ================================================================
#  LOOM ASCENDANT COSMOS — RECURSIVE SYSTEM MODULE
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
#  SHA-256: f3b2c4e5d6a7b8c9d0e1f2a3b4c5d6e7f8g9h0i1j2k3l4m5n6o7p8q9r0s1t2u3v4w5x6y7z8a9b0c1d2e3f4g5h6i7j8k9l0m1n2o3p4q5r6s7t8u9v0w1x2y3z4a5b6c7d8e9f0g1h2i3j4k5l6m7n8o9p0q1r2s3t4u5v6w7x8y9z0
#  Description: This module implements a timeline engine that manages the flow of time.
# ================================================================
import heapq
from typing import Dict, List, Tuple, Callable, Optional, Set, Any
import numpy as np
from collections import deque
import threading
import time
import logging
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TimelineEngine")

# Custom exceptions
class TemporalConstraintError(Exception):
    """Exception raised when temporal constraints are violated"""
    pass

class TimelineMetrics:
    """Tracks and analyzes timeline performance and integrity"""
    
    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate
        self.coherence_history = deque(maxlen=sampling_rate)
        self.divergence_history = deque(maxlen=sampling_rate)
        self.paradox_count = 0
        self.branch_count = 0
        self.events_processed = 0
        self.recursion_depth = 0
        self.last_update = time.time()
        
    def update(self, timeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update metrics based on current timeline state"""
        self.coherence_history.append(timeline_state.get('coherence', 1.0))
        self.divergence_history.append(timeline_state.get('divergence', 0.0))
        self.paradox_count += timeline_state.get('new_paradoxes', 0)
        self.branch_count += timeline_state.get('new_branches', 0)
        self.events_processed += timeline_state.get('events_processed', 0)
        self.recursion_depth = max(self.recursion_depth, timeline_state.get('recursion_depth', 0))
        self.last_update = time.time()
        
        return self.get_summary()
        
    def get_summary(self) -> Dict[str, Any]:
        """Generate comprehensive metrics summary"""
        return {
            'avg_coherence': sum(self.coherence_history) / len(self.coherence_history) if self.coherence_history else 1.0,
            'avg_divergence': sum(self.divergence_history) / len(self.divergence_history) if self.divergence_history else 0.0,
            'paradox_count': self.paradox_count,
            'branch_count': self.branch_count,
            'events_processed': self.events_processed,
            'max_recursion_depth': self.recursion_depth,
            'last_update': self.last_update
        }
    
    def measure_divergence(self) -> float:
        """Measure current timeline divergence level"""
        return sum(self.divergence_history) / len(self.divergence_history) if self.divergence_history else 0.0
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.coherence_history.clear()
        self.divergence_history.clear()
        self.paradox_count = 0
        self.branch_count = 0
        self.events_processed = 0
        self.recursion_depth = 0

@dataclass
class TemporalEvent:
    """Enhanced event class with built-in validation and metaproperties"""
    timestamp: float
    event_type: str
    quantum_state: np.ndarray = None
    ethical_vectors: np.ndarray = None
    causal_parents: List[Any] = field(default_factory=list)
    recursion_depth: int = 0
    metadata: Dict = field(default_factory=dict)
    entropy_consumed: float = 0.0
    paradox_resolved: bool = False

    # Add new optional fields
    phase: float = 0.0  # Default phase
    amplitude: float = 0.0  # Default amplitude
    is_inhale: bool = False  # Default inhale state

    def __post_init__(self):
        """Validate event properties after initialization"""
        if self.quantum_state is None:
            self.quantum_state = np.ones(10)  # Default quantum state
        if self.ethical_vectors is None:
            self.ethical_vectors = np.zeros(3)  # Default ethical vector

    def __lt__(self, other):
        """Enable comparison for priority queue"""
        return self.timestamp < other.timestamp

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'quantum_state': self.quantum_state,
            'ethical_vectors': self.ethical_vectors,
            'causal_parents': self.causal_parents,
            'recursion_depth': self.recursion_depth,
            'metadata': self.metadata,
            'entropy_consumed': self.entropy_consumed,
            'paradox_resolved': self.paradox_resolved,
            'phase': self.phase,
            'amplitude': self.amplitude,
            'is_inhale': self.is_inhale
        }

    @classmethod
    def from_dict(cls, event_dict: Dict) -> 'TemporalEvent':
        """Create event from dictionary"""
        # Map 'type' to 'event_type' if necessary
        if 'type' in event_dict:
            event_dict['event_type'] = event_dict.pop('type')

        # Ensure required fields are present
        if 'timestamp' not in event_dict:
            logger.warning("Missing 'timestamp' in event data. Defaulting to 0.0.")
            event_dict['timestamp'] = 0.0
        if 'event_type' not in event_dict:
            logger.warning("Missing 'event_type' in event data. Defaulting to 'unknown_event'.")
            event_dict['event_type'] = 'unknown_event'

        # Pass all fields directly to the constructor
        return cls(**event_dict)

class TemporalBranch:
    """Represents a branch in the timeline with its own events and causal relationships"""
    
    def __init__(self, 
                 branch_id: str,
                 parent_branch_id: Optional[str] = None,
                 branch_point: Optional[float] = None,
                 attributes: Dict[str, Any] = None):
        """
        Initialize a new timeline branch
        
        Args:
            branch_id: Unique identifier for this branch
            parent_branch_id: ID of the parent branch (None for primary timeline)
            branch_point: Timestamp where this branch diverged from parent
            attributes: Optional attributes for this branch
        """
        self.id = branch_id
        self.parent_id = parent_branch_id
        self.branch_point = branch_point or time.time()
        self.creation_time = time.time()
        self.attributes = attributes or {}
        self.child_branches = []  # IDs of branches spawned from this one
        self.events = []  # Events specific to this branch
        self.active = True
        self.coherence = 1.0  # Timeline coherence/stability factor
        
    def add_child_branch(self, branch_id: str) -> None:
        """Register a child branch that spawned from this branch"""
        if branch_id not in self.child_branches:
            self.child_branches.append(branch_id)
            
    def add_event(self, event: TemporalEvent) -> None:
        """Add an event to this branch's timeline"""
        self.events.append(event)
        
    def get_events_in_range(self, start_time: float, end_time: float) -> List[TemporalEvent]:
        """Get events in this branch within the specified time range"""
        return [e for e in self.events if start_time <= e.timestamp <= end_time]
        
    def set_active(self, active: bool) -> None:
        """Set whether this branch is actively evolving"""
        self.active = active
        
    def update_coherence(self, new_coherence: float) -> None:
        """Update the coherence/stability of this branch"""
        self.coherence = max(0.0, min(1.0, new_coherence))
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "branch_point": self.branch_point,
            "creation_time": self.creation_time,
            "attributes": self.attributes,
            "child_branches": self.child_branches,
            "event_count": len(self.events),
            "active": self.active,
            "coherence": self.coherence
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalBranch':
        """Create a branch from dictionary representation"""
        branch = cls(
            branch_id=data["id"],
            parent_branch_id=data.get("parent_id"),
            branch_point=data.get("branch_point"),
            attributes=data.get("attributes", {})
        )
        branch.creation_time = data.get("creation_time", time.time())
        branch.child_branches = data.get("child_branches", [])
        branch.active = data.get("active", True)
        branch.coherence = data.get("coherence", 1.0)
        return branch

class TimelineEngine:
    """Enhanced Timeline Engine with improved causality management and multidimensional support"""
    
    def __init__(self, 
                 breath_frequency: float = 1.0,
                 max_recursion_depth: int = 8,
                 num_dimensions: int = 4,
                 ethical_dimensions: int = 3,
                 parallel_timelines: int = 1,
                 auto_stabilize: bool = True):
        """
        Initialize TimelineEngine with enhanced dimensionality and parameters
        """
        # Core temporal parameters
        self.breath_frequency = breath_frequency
        self.temporal_resolution = 1 / (2 * breath_frequency)  # Nyquist Limit
        self.master_tick = 0
        self.phase = 0.0  # Breath cycle phase [0, 2π)
        self.active_timeline = 0  # Default to the first timeline
        
        # Temporal data structures
        self.event_horizons = [[] for _ in range(parallel_timelines)]  # Priority queues by timeline
        self.timestreams = [{} for _ in range(parallel_timelines)]  # Causal graphs by timeline
        self.recursion_stacks = [deque(maxlen=max_recursion_depth) for _ in range(parallel_timelines)]
        self.breath_phase_array = np.zeros(parallel_timelines)  # Track breath phase per timeline
        
        # Advanced temporal structures
        self.ethical_tensors = np.zeros((parallel_timelines, ethical_dimensions))
        self.paradox_buffers = [deque(maxlen=3) for _ in range(parallel_timelines)]
        self.timeline_coherence = np.ones(parallel_timelines)  # Coherence metrics
        self.entanglement_matrix = np.eye(parallel_timelines)  # Timeline entanglement
        
        # Observer mechanism
        self.observers = []  # Temporal observers for event monitoring
        self.sync_lock = threading.RLock()  # Thread safety for parallel processing
        
        # Auto-stabilization
        self.auto_stabilize = auto_stabilize
        self.stability_threshold = 0.3  # Timeline stability threshold
        
        # Temporal cache for optimization
        self.event_cache = {}  # Cache for frequent events
        self.cache_hit_rate = 0.0
        self.cache_miss_count = 0
        
        # Metrics
        self.metrics = {
            'paradoxes_resolved': 0,
            'events_processed': 0,
            'cache_hits': 0,
            'timeline_branches': 0,
            'quantum_collapses': 0,
            'ethical_corrections': 0,
            'runtime_ns': 0
        }
        
        logger.info(f"TimelineEngine initialized with {parallel_timelines} timeline(s) "
                    f"at {breath_frequency}Hz breath frequency")

    def process_tick(self, 
                     inputs: Dict, 
                     rcf_operator: Callable,
                     timeline_idx: int = None) -> Dict:
        """
        Execute one temporal iteration with RCF integration
        
        Args:
            inputs: Input data dictionary
            rcf_operator: Reality Consistency Function operator
            timeline_idx: Specific timeline to process (None = active timeline)
        
        Returns:
            Dict of outputs including event data and metrics
        """
        start_time = time.time_ns()
        
        with self.sync_lock:
            timeline = timeline_idx if timeline_idx is not None else self.active_timeline
            
            # Update breath phase
            self.phase = (self.phase + 2*np.pi*self.temporal_resolution) % (2*np.pi)
            
            # Initialize outputs structure
            outputs = {
                'master_tick': self.master_tick,
                'timeline': timeline,
                'time_tensors': [],
                'causal_validations': [],
                'breath_pulse': self._generate_breath_pulse(timeline),
                'coherence': self.timeline_coherence[timeline],
                'metrics': {}
            }
            
            # Apply RCF ethical stabilization
            stabilized_inputs = rcf_operator(inputs)
            
            # Remove expired events
            self._remove_expired_events(timeline)
            
            # Handle pending events
            while (self.event_horizons[timeline] and 
                   self.event_horizons[timeline][0].timestamp <= self.master_tick):
                
                event = heapq.heappop(self.event_horizons[timeline])
                
                # Check cache for optimized processing
                cache_key = f"{event.event_type}:{event.timestamp}"
                if cache_key in self.event_cache:
                    self.metrics['cache_hits'] += 1
                    resolved_event = self.event_cache[cache_key]
                else:
                    # Validate and process event
                    if not self._validate_causal_chain(event, timeline):
                        if not event.paradox_resolved:  # Prevent infinite recursion
                            self._handle_paradox(event, timeline)
                        continue
                    
                    resolved_event = self._apply_temporal_constraints(event, stabilized_inputs, timeline)
                    self.event_cache[cache_key] = resolved_event
                    self.metrics['cache_miss_count'] += 1
                
                # Notify observers of the resolved event
                self.notify_observers(resolved_event, timeline)
                
                # Update timestream and collect outputs
                outputs['time_tensors'].append(self._generate_time_tensor(resolved_event))
                self._update_timestream(resolved_event, timeline)
                self.metrics['events_processed'] += 1
            
            # Auto-stabilize timeline if needed
            if self.auto_stabilize and self.timeline_coherence[timeline] < self.stability_threshold:
                self._stabilize_timeline(timeline)
            
            # Enforce forward causality at macro scale
            self.master_tick += 1
            
            # Monitor timeline coherence
            self._monitor_timeline_coherence()
            
            # Check timeline synchronization
            self._check_timeline_synchronization()
            
            # Update metrics
            self.metrics['runtime_ns'] = time.time_ns() - start_time
            outputs['metrics'] = self.metrics.copy()
            
            return outputs

    def schedule_event(self, 
                       event: Dict | TemporalEvent, 
                       timeline_idx: int = None,
                       recursion_depth: int = 0) -> TemporalEvent:
        """
        Add event to horizon queue with recursion context
        """
        timeline = timeline_idx if timeline_idx is not None else self.active_timeline
        logger.debug(f"Scheduling event: {event} on timeline {timeline} at recursion depth {recursion_depth}")
        
        # Convert to TemporalEvent if dictionary
        if isinstance(event, dict):
            event_obj = TemporalEvent(**event)
        else:
            event_obj = event
            
        # Add recursion context
        if recursion_depth > len(self.recursion_stacks[timeline]):
            raise TemporalConstraintError(f"Max recursion depth exceeded: {recursion_depth}")
            
        event_obj.recursion_depth = recursion_depth
        
        # Validate event metadata
        self._validate_event_metadata(event_obj)
        
        # Add to horizon queue
        heapq.heappush(self.event_horizons[timeline], event_obj)
        return event_obj

    def create_recursive_loop(self, 
                              entry_event: Dict | TemporalEvent, 
                              exit_condition: Callable,
                              timeline_idx: int = None,
                              max_iterations: int = 100) -> List[TemporalEvent]:
        """
        Implement bounded temporal loop (Symbolic Operator ⟲t)
        
        Args:
            entry_event: Initial event
            exit_condition: Function that returns True when loop should exit
            timeline_idx: Target timeline (None = active timeline)
            max_iterations: Safety limit for iterations
            
        Returns:
            List of generated event objects
        """
        timeline = timeline_idx if timeline_idx is not None else self.active_timeline
        
        # Convert to TemporalEvent if dictionary
        if isinstance(entry_event, dict):
            current_event = TemporalEvent(**entry_event)
        else:
            current_event = entry_event
            
        self.recursion_stacks[timeline].append(current_event)
        generated_events = [current_event]
        
        for i in range(max_iterations):
            if exit_condition(current_event):
                logger.info(f"Recursive loop exited after {i} iterations")
                break
                
            # Create next event with causality chain
            next_event = TemporalEvent(
                timestamp=current_event.timestamp + self.temporal_resolution,
                event_type=current_event.event_type,
                quantum_state=current_event.quantum_state.copy(),
                ethical_vectors=current_event.ethical_vectors.copy(),
                causal_parents=[current_event],
                recursion_depth=len(self.recursion_stacks[timeline]),
                metadata={**current_event.metadata, 'loop_iteration': i}
            )
            
            self.schedule_event(next_event, timeline, len(self.recursion_stacks[timeline]))
            current_event = next_event
            generated_events.append(current_event)
            
        self.recursion_stacks[timeline].pop()
        return generated_events

    def branch_timeline(self, branch_point_event: Dict | TemporalEvent = None) -> int:
        """
        Create a new timeline branch from the current state
        
        Args:
            branch_point_event: Event causing the branch (None = current state)
            
        Returns:
            int: New timeline index
        """
        with self.sync_lock:
            new_idx = len(self.event_horizons)
            
            # Copy current timeline structures
            self.event_horizons.append(self.event_horizons[self.active_timeline].copy())
            self.timestreams.append(self.timestreams[self.active_timeline].copy())
            self.recursion_stacks.append(self.recursion_stacks[self.active_timeline].copy())
            self.paradox_buffers.append(self.paradox_buffers[self.active_timeline].copy())
            
            # Expand tensors and matrices
            self.ethical_tensors = np.vstack([
                self.ethical_tensors, 
                self.ethical_tensors[self.active_timeline]
            ])
            
            # Update entanglement matrix
            old_size = len(self.entanglement_matrix)
            new_matrix = np.eye(old_size + 1)
            new_matrix[:old_size, :old_size] = self.entanglement_matrix
            
            # Set entanglement between parent and child timeline
            new_matrix[self.active_timeline, new_idx] = 0.5
            new_matrix[new_idx, self.active_timeline] = 0.5
            self.entanglement_matrix = new_matrix
            
            # Update coherence metrics
            self.timeline_coherence = np.append(
                self.timeline_coherence, 
                self.timeline_coherence[self.active_timeline] * 0.9  # Slight reduction in coherence
            )
            
            # Record branch event
            if branch_point_event:
                if isinstance(branch_point_event, dict):
                    event_obj = TemporalEvent(**branch_point_event)
                else:
                    event_obj = branch_point_event
                    
                event_obj.metadata['branch_point'] = True
                event_obj.metadata['parent_timeline'] = self.active_timeline
                event_obj.metadata['child_timeline'] = new_idx
                self.schedule_event(event_obj, new_idx)
                
            self.metrics['timeline_branches'] += 1
            logger.info(f"Timeline branched: {self.active_timeline} -> {new_idx}")
            
            return new_idx

    def switch_timeline(self, timeline_idx: int) -> bool:
        """
        Switch active timeline context
        
        Args:
            timeline_idx: Target timeline index
            
        Returns:
            bool: Success status
        """
        if 0 <= timeline_idx < len(self.event_horizons):
            self.active_timeline = timeline_idx
            logger.info(f"Active timeline switched to {timeline_idx}")
            return True
        return False

    def merge_timelines(self, 
                        source_idx: int, 
                        target_idx: int, 
                        merge_strategy: str = 'overlay') -> bool:
        """
        Merge two timelines using specified strategy
        
        Args:
            source_idx: Source timeline to merge from
            target_idx: Target timeline to merge into
            merge_strategy: Strategy ('overlay', 'interleave', 'selective')
            
        Returns:
            bool: Success status
        """
        if not (0 <= source_idx < len(self.event_horizons) and 
                0 <= target_idx < len(self.event_horizons)):
            return False
            
        with self.sync_lock:
            if merge_strategy == 'overlay':
                # Replace target events with source events where timestamps match
                for event in self.event_horizons[source_idx]:
                    # Find and replace matching events in target
                    target_events = [e for e in self.event_horizons[target_idx] 
                                    if abs(e.timestamp - event.timestamp) < self.temporal_resolution]
                    
                    for target_event in target_events:
                        self.event_horizons[target_idx].remove(target_event)
                    
                    # Add source event to target timeline
                    heapq.heappush(self.event_horizons[target_idx], event)
                    
            elif merge_strategy == 'interleave':
                # Interleave events from both timelines
                merged_horizon = []
                while self.event_horizons[source_idx] and self.event_horizons[target_idx]:
                    # Take alternating events weighted by timeline coherence
                    if (np.random.random() < 
                        self.timeline_coherence[source_idx] / 
                        (self.timeline_coherence[source_idx] + self.timeline_coherence[target_idx])):
                        event = heapq.heappop(self.event_horizons[source_idx])
                    else:
                        event = heapq.heappop(self.event_horizons[target_idx])
                    heapq.heappush(merged_horizon, event)
                
                # Add remaining events
                while self.event_horizons[source_idx]:
                    heapq.heappush(merged_horizon, heapq.heappop(self.event_horizons[source_idx]))
                while self.event_horizons[target_idx]:
                    heapq.heappush(merged_horizon, heapq.heappop(self.event_horizons[target_idx]))
                    
                self.event_horizons[target_idx] = merged_horizon
                
            elif merge_strategy == 'selective':
                # Take only non-paradoxical events from source
                stable_events = [e for e in self.event_horizons[source_idx] 
                               if not any(p[1] == e for p in self.paradox_buffers[source_idx])]
                
                for event in stable_events:
                    heapq.heappush(self.event_horizons[target_idx], event)
            
            # Update entanglement and coherence
            self.entanglement_matrix[source_idx, target_idx] = 0.95
            self.entanglement_matrix[target_idx, source_idx] = 0.95
            self.timeline_coherence[target_idx] = (
                self.timeline_coherence[target_idx] * 0.7 + 
                self.timeline_coherence[source_idx] * 0.3
            )
            
            logger.info(f"Timelines merged: {source_idx} -> {target_idx} using {merge_strategy}")
            return True

    def register_observer(self, callback: Callable | str) -> None:
        """
        Register observer function to be called on each event
        
        Args:
            callback: Function to call with (event, timeline_idx) parameters
            or a string identifier for the observer
        """
        self.observers.append(callback)
        if hasattr(callback, '__name__'):
            logger.info(f"Observer registered: {callback.__name__}")
        else:
            logger.info(f"Observer registered: {callback}")

    def notify_observers(self, event: TemporalEvent | Dict | str, timeline_idx: int) -> None:
        """
        Notify all registered observers of a temporal event.

        Args:
            event: The temporal event to notify observers about.
            timeline_idx: The index of the timeline where the event occurred.
        """
        # Handle string events for backward compatibility
        if isinstance(event, str):
            event_data = {
                "event_type": event,
                "timestamp": self.master_tick * self.temporal_resolution
            }
            event = TemporalEvent.from_dict(event_data)
        elif isinstance(event, dict):
            event = TemporalEvent.from_dict(event)

        if not self.observers:
            logger.warning("No observers registered to notify.")
            return

        for observer in self.observers:
            try:
                # Check if observer is a callable or just a string identifier
                if callable(observer):
                    observer(event, timeline_idx)
                else:
                    logger.info(f"Observer {observer} notified of: {event.event_type}")
            except Exception as e:
                observer_name = observer.__name__ if hasattr(observer, '__name__') else str(observer)
                logger.error(f"Error notifying observer {observer_name}: {e}")

    def dump_state(self) -> Dict:
        """
        Dump the current state of the TimelineEngine for debugging
        """
        state = {
            'master_tick': self.master_tick,
            'phase': self.phase,
            'event_horizons': [[e.to_dict() for e in horizon] for horizon in self.event_horizons],
            'timeline_coherence': self.timeline_coherence.tolist(),
            'metrics': self.metrics
        }
        logger.debug(f"Engine state: {state}")
        return state

    def temporal_injection(self, event: TemporalEvent, timeline_idx: int = None) -> None:
        """
        Temporal Injection (→t): Creates a new event at a specified timepoint.
        """
        timeline = timeline_idx if timeline_idx is not None else self.active_timeline
        heapq.heappush(self.event_horizons[timeline], event)
        logger.info(f"Injected event {event.event_type} at t={event.timestamp:.2f} on timeline {timeline}")

    def causal_binding(self, parent_event: TemporalEvent, child_event: TemporalEvent, timeline_idx: int = None) -> None:
        """
        Causal Binding (∞t): Establishes fixed relationships between events.
        """
        timeline = timeline_idx if timeline_idx is not None else self.active_timeline
        parent_id = id(parent_event)
        child_id = id(child_event)
        
        if parent_id not in self.timestreams[timeline]:
            self.timestreams[timeline][parent_id] = []
        self.timestreams[timeline][parent_id].append(child_id)
        logger.info(f"Bound event {parent_event.event_type} to {child_event.event_type} on timeline {timeline}")

    def breath_synchronization(self, timeline_idx: int = None) -> None:
        """
        Breath Synchronization (≈t): Aligns events to breath cycles.
        """
        timeline = timeline_idx if timeline_idx is not None else self.active_timeline
        self.breath_phase_array[timeline] = (self.phase + 2 * np.pi * self.temporal_resolution) % (2 * np.pi)
        logger.info(f"Breath synchronization applied to timeline {timeline}")

    def recursive_closure(self, entry_event: TemporalEvent, exit_condition: Callable, timeline_idx: int = None) -> List[TemporalEvent]:
        """
        Recursive Closure (⟲t): Marks bounded temporal loops with consistent entry/exit states.
        """
        timeline = timeline_idx if timeline_idx is not None else self.active_timeline
        self.recursion_stacks[timeline].append(entry_event)
        generated_events = [entry_event]
        
        current_event = entry_event
        while not exit_condition(current_event):
            next_event = TemporalEvent(
                timestamp=current_event.timestamp + self.temporal_resolution,
                event_type=current_event.event_type,
                causal_parents=[current_event],
                recursion_depth=len(self.recursion_stacks[timeline]),
                metadata={**current_event.metadata, 'loop_iteration': len(generated_events)}
            )
            self.schedule_event(next_event, timeline)
            generated_events.append(next_event)
            current_event = next_event
        
        self.recursion_stacks[timeline].pop()
        return generated_events

    def enforce_boundary_conditions(self, event: TemporalEvent, timeline_idx: int = None) -> TemporalEvent:
        """
        Enforce boundary conditions on an event.
        """
        timeline = timeline_idx if timeline_idx is not None else self.active_timeline
        
        # Apply recursion depth limit
        if event.recursion_depth > len(self.recursion_stacks[timeline]):
            raise ValueError(f"Recursion depth exceeded for event {event.event_type}")
        
        # Apply timeline pruning threshold
        if event.timestamp < self.master_tick * self.temporal_resolution:
            logger.warning(f"Event {event.event_type} pruned due to timeline threshold")
            return None
        
        # Apply breath synchronization
        self.breath_synchronization(timeline)
        
        return event

    def propagate_state(self, state: Dict, inputs: Dict, paradox_resolver: Callable) -> Dict:
        """
        Propagate the system state using the Temporal Propagation Function:
        S' = T(S, B(t), I(t), P(t))
        """
        # Update breath phase
        self.phase = (self.phase + 2 * np.pi * self.temporal_resolution) % (2 * np.pi)
        
        # Apply paradox resolution
        resolved_state = paradox_resolver(state)
        
        # Integrate external inputs
        for key, value in inputs.items():
            if key in resolved_state:
                resolved_state[key] += value
            else:
                resolved_state[key] = value
        
        # Return the updated state
        return resolved_state

    def _generate_breath_pulse(self, timeline_idx: int) -> Dict:
        """
        Create synchronization pulse aligned to breath phase
        
        Args:
            timeline_idx: Timeline index
            
        Returns:
            Dict of pulse parameters
        """
        return {
            'phase': self.phase,
            'amplitude': np.sin(self.phase),
            'frequency': self.breath_frequency,
            'coherence': self.timeline_coherence[timeline_idx],
            'entanglement_map': self._current_entanglement(timeline_idx),
            'timestamp': self.master_tick * self.temporal_resolution,
            'dimensions': self.num_dimensions
        }

    def _validate_causal_chain(self, event: TemporalEvent, timeline_idx: int) -> bool:
        """
        Check historical consistency and causal validity
        
        Args:
            event: Event to validate
            timeline_idx: Timeline context
            
        Returns:
            bool: True if valid, False if paradox detected
        """
        if not self.timestreams[timeline_idx]:  # First event
            return True
            
        # Verify all parent events exist and are earlier
        for parent in event.causal_parents:
            parent_id = id(parent)
            
            if parent_id not in self.timestreams[timeline_idx]:
                self.paradox_buffers[timeline_idx].append(
                    ('orphan_parent', event, parent)
                )
                logger.warning(f"Paradox detected: Orphaned parent at {event.timestamp}")
                return False
                
            if parent.timestamp >= event.timestamp:
                self.paradox_buffers[timeline_idx].append(
                    ('reverse_causality', event, parent)
                )
                logger.warning(f"Paradox detected: Reverse causality at {event.timestamp}")
                return False
                
        # Check for causal loops
        visited = set()
        stack = [id(event)]
        
        while stack:
            current = stack.pop()
            
            if current in visited:
                self.paradox_buffers[timeline_idx].append(
                    ('causal_loop', event, None)
                )
                logger.warning(f"Paradox detected: Causal loop at {event.timestamp}")
                return False
                
            visited.add(current)
            
            # Add parents to stack
            if current in self.timestreams[timeline_idx]:
                for parent_id in self.timestreams[timeline_idx][current]:
                    stack.append(parent_id)
                    
        return True

    def _validate_event_metadata(self, event: TemporalEvent) -> None:
        """
        Validate that the event has all required metadata fields
        """
        required_fields = ['timestamp', 'event_type']
        for field in required_fields:
            if field not in event.metadata:
                logger.error(f"Event {event} missing required metadata field: {field}")
                raise ValueError(f"Missing required metadata field: {field}")

    def _apply_temporal_constraints(self, 
                                  event: TemporalEvent, 
                                  stabilized_inputs: Dict,
                                  timeline_idx: int) -> TemporalEvent:
        """
        Enforce temporal resolution and recursion limits
        
        Args:
            event: Event to constrain
            stabilized_inputs: RCF-stabilized inputs
            timeline_idx: Timeline context
            
        Returns:
            TemporalEvent: Constrained event
        """
        # Apply Nyquist Limit (Design Constraint 3)
        quantized_time = round(event.timestamp / self.temporal_resolution) * self.temporal_resolution
        
        # Manage recursion depth
        recursion_context = {
            'depth': len(self.recursion_stacks[timeline_idx]),
            'inheritance': (self.recursion_stacks[timeline_idx][-1].to_dict() 
                          if self.recursion_stacks[timeline_idx] else {})
        }
        
        # Apply quantum state transformation
        new_quantum_state = self._collapse_probabilities(event)
        
        # Create new event with constraints
        constrained_event = TemporalEvent(
            timestamp=quantized_time,
            event_type=event.event_type,
            quantum_state=new_quantum_state,
            ethical_vectors=stabilized_inputs.get('ethical_vectors', event.ethical_vectors),
            causal_parents=event.causal_parents,
            recursion_depth=event.recursion_depth,
            metadata={
                **event.metadata,
                'recursion_context': recursion_context,
                'timeline_idx': timeline_idx,
                'coherence': self.timeline_coherence[timeline_idx]
            },
            entropy_consumed=event.entropy_consumed,
            paradox_resolved=event.paradox_resolved
        )
        
        return constrained_event

    def _handle_paradox(self, event: TemporalEvent, timeline_idx: int) -> None:
        """
        Implement Paradox Resolution System integration
        
        Args:
            event: Paradoxical event
            timeline_idx: Timeline context
        """
        paradox_type = self.paradox_buffers[timeline_idx][-1][0]
        entropy = self._calculate_paradox_entropy(event)
        
        # Apply RCF-based resolution
        resolved_event = self._rcf_paradox_resolution(event, paradox_type, entropy, timeline_idx)
        
        if resolved_event:
            heapq.heappush(self.event_horizons[timeline_idx], resolved_event)
            
        # Reduce timeline coherence
        self.timeline_coherence[timeline_idx] *= 0.95
        self.metrics['paradoxes_resolved'] += 1
        
        logger.info(f"Paradox handled: {paradox_type} at {event.timestamp}")

    def _rcf_paradox_resolution(self, 
                              event: TemporalEvent, 
                              paradox_type: str, 
                              entropy: float,
                              timeline_idx: int) -> Optional[TemporalEvent]:
        """
        RCF-integrated paradox handling (Auxiliary System 1)
        
        Args:
            event: Paradoxical event
            paradox_type: Type of paradox
            entropy: Calculated entropy
            timeline_idx: Timeline context
            
        Returns:
            Optional[TemporalEvent]: Resolved event or None
        """
        # Apply resolution strategy by paradox type
        if paradox_type == 'orphan_parent':
            # Create a substitute parent event
            return TemporalEvent(
                timestamp=event.timestamp + self.temporal_resolution,
                event_type=event.event_type,
                quantum_state=event.quantum_state * 0.9,  # Attenuated state
                ethical_vectors=self.ethical_tensors[timeline_idx],
                causal_parents=[],  # No parents to avoid recursion
                recursion_depth=event.recursion_depth,
                metadata={**event.metadata, 'paradox_resolved': True, 'resolution': 'orphan_fix'},
                entropy_consumed=entropy,
                paradox_resolved=True
            )
            
        elif paradox_type == 'reverse_causality':
            # Delay event to preserve causality
            return TemporalEvent(
                timestamp=event.timestamp + 2 * self.temporal_resolution,
                event_type=event.event_type,
                quantum_state=event.quantum_state,
                ethical_vectors=self.ethical_tensors[timeline_idx],
                causal_parents=event.causal_parents,
                recursion_depth=event.recursion_depth,
                metadata={**event.metadata, 'paradox_resolved': True, 'resolution': 'time_delay'},
                entropy_consumed=entropy,
                paradox_resolved=True
            )
            
        elif paradox_type == 'causal_loop':
            # Branch timeline to avoid loop
            new_timeline = self.branch_timeline()
            
            # Schedule fixed event in new branch
            new_event = TemporalEvent(
                timestamp=event.timestamp + self.temporal_resolution,
                event_type=event.event_type,
                quantum_state=event.quantum_state,
                ethical_vectors=event.ethical_vectors,
                causal_parents=[],  # Break causal loop
                recursion_depth=0,  # Reset recursion
                metadata={**event.metadata, 'paradox_resolved': True, 'resolution': 'branch_fix'},
                entropy_consumed=entropy,
                paradox_resolved=True
            )
            
            self.schedule_event(new_event, new_timeline)
            return None  # No event returned for original timeline
        
        else:
            # Generic resolution - delay and reduce coupling
            return TemporalEvent(
                timestamp=event.timestamp + self.temporal_resolution,
                event_type=event.event_type,
                quantum_state=event.quantum_state * 0.8,
                ethical_vectors=self.ethical_tensors[timeline_idx],
                causal_parents=event.causal_parents[:1] if event.causal_parents else [],  # Limit parents
                recursion_depth=min(event.recursion_depth, 1),  # Reduce recursion
                metadata={**event.metadata, 'paradox_resolved': True, 'resolution': 'generic'},
                entropy_consumed=entropy,
                paradox_resolved=True
            )

    def _generate_time_tensor(self, event: TemporalEvent) -> np.ndarray:
        """
        Generate multidimensional temporal representation
        
        Args:
            event: Source event
            
        Returns:
            np.ndarray: Tensor representation
        """
        # Create base tensor from event data
        tensor_shape = (self.num_dimensions, self.ethical_dimensions)
        tensor = np.zeros(tensor_shape)
        
        # Populate tensor dimensions
        tensor[0, :] = np.tile(event.timestamp, self.ethical_dimensions)  # Time dimension
        
        if event.ethical_vectors is not None and len(event.ethical_vectors) == self.ethical_dimensions:
            tensor[1, :] = event.ethical_vectors  # Ethical dimension
            
        # Quantum projection to remaining dimensions
        if event.quantum_state is not None:
            quantum_dims = min(len(event.quantum_state), self.num_dimensions - 2)
            for i in range(quantum_dims):
                if i + 2 < self.num_dimensions:
                    tensor[i+2, :] = event.quantum_state[i]
                    
        # Add coherence signature
        coherence_idx = min(self.num_dimensions - 1, 3)
        tensor[coherence_idx, 0] = event.timestamp
        
        self.metrics['quantum_collapses'] += 1
        return tensor

    def _update_timestream(self, event: TemporalEvent, timeline_idx: int) -> None:
        """
        Update causal graph with new event
        
        Args:
            event: Event to add
            timeline_idx: Timeline context
        """
        event_id = id(event)
        self.timestreams[timeline_idx][event_id] = []
        
        # Link parents to children
        for parent in event.causal_parents:
            parent_id = id(parent)
            if parent_id in self.timestreams[timeline_idx]:
                self.timestreams[timeline_idx][parent_id].append(event_id)

    def _collapse_probabilities(self, event: TemporalEvent) -> np.ndarray:
        """
        Apply quantum collapse to event state
        
        Args:
            event: Event with quantum state
            
        Returns:
            np.ndarray: Collapsed quantum state
        """
        if event.quantum_state is None:
            return np.ones(10)
            
        # Normalize quantum state
        quantum_state = event.quantum_state.copy()
        if np.sum(quantum_state) > 0:
            quantum_state = quantum_state / np.sum(quantum_state)
            
        # Apply mild decoherence based on recursion depth
        decoherence = 0.05 * (event.recursion_depth + 1)
        noise = np.random.normal(0, decoherence, size=quantum_state.shape)
        quantum_state = np.clip(quantum_state + noise, 0, 1)
        
        # Renormalize
        if np.sum(quantum_state) > 0:
            quantum_state = quantum_state / np.sum(quantum_state)
            
        return quantum_state

    def _calculate_paradox_entropy(self, event: TemporalEvent) -> float:
        """
        Calculate entropy cost of resolving a paradox
        
        Args:
            event: Paradoxical event
            
        Returns:
            float: Entropy cost
        """
        # Base entropy proportional to quantum state complexity
        base_entropy = 0.1
        
        if event.quantum_state is not None:
            complexity = np.sum(event.quantum_state ** 2)
            return base_entropy + complexity * 0.05
        
        return base_entropy

    def _remove_expired_events(self, timeline_idx: int) -> None:
        """
        Remove expired events from the event horizon
        """
        current_time = self.master_tick * self.temporal_resolution
        self.event_horizons[timeline_idx] = [
            event for event in self.event_horizons[timeline_idx]
            if event.timestamp > current_time
        ]
        logger.debug(f"Expired events removed from timeline {timeline_idx}")

    def _monitor_timeline_coherence(self) -> None:
        """
        Monitor and log coherence metrics for all timelines
        """
        for idx, coherence in enumerate(self.timeline_coherence):
            if coherence < self.stability_threshold:
                logger.warning(f"Timeline {idx} coherence below threshold: {coherence:.2f}")
            else:
                logger.info(f"Timeline {idx} coherence: {coherence:.2f}")

    def _check_timeline_synchronization(self) -> None:
        """
        Check that all timelines are synchronized with the master tick
        """
        for idx, horizon in enumerate(self.event_horizons):
            if horizon and horizon[0].timestamp < self.master_tick * self.temporal_resolution:
                logger.warning(f"Timeline {idx} is lagging behind master tick")

    def _process_timelines_in_parallel(self, inputs: Dict, rcf_operator: Callable) -> None:
        """
        Process all timelines in parallel
        """
        def process_timeline(idx):
            self.process_tick(inputs, rcf_operator, timeline_idx=idx)
        
        threads = []
        for idx in range(len(self.event_horizons)):
            thread = threading.Thread(target=process_timeline, args=(idx,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        logger.info("All timelines processed in parallel")

    def is_active(self):
        """Check if the Timeline Engine is active."""
        return True  # Placeholder implementation


def test_observer(event, timeline_idx):
    print(f"Observer received event: {event.event_type} on timeline {timeline_idx}")


# Initialize the TimelineEngine
timeline = TimelineEngine(
    breath_frequency=1.0,
    max_recursion_depth=8,
    num_dimensions=4,
    ethical_dimensions=3,
    parallel_timelines=1,
    auto_stabilize=True
)

# Register the test observer
timeline.register_observer(test_observer)

# Create a test event and notify observers
test_event = TemporalEvent(timestamp=0.0, event_type="test_event")
timeline.notify_observers(test_event, 0)

event_data = {'event_type': 'breath_pulse', 'timestamp': 0.0}

# Custom exceptions for timeline-related issues
class TimelineParadoxError(Exception):
    """Exception raised when a temporal paradox is detected in the timeline."""
    def __init__(self, message="Temporal paradox detected in timeline", severity=None, 
                affected_branches=None, timeline_point=None):
        self.message = message
        self.severity = severity
        self.affected_branches = affected_branches or []
        self.timeline_point = timeline_point
        super().__init__(self.message)
        
    def __str__(self):
        details = []
        if self.severity is not None:
            details.append(f"Severity: {self.severity:.2f}")
        if self.affected_branches:
            details.append(f"Affected branches: {len(self.affected_branches)}")
        if self.timeline_point:
            details.append(f"Timeline point: {self.timeline_point}")
        
        if details:
            return f"{self.message} - {'; '.join(details)}"
        return self.message

class QuantumDecoherenceError(Exception):
    """Exception raised when quantum coherence falls below critical threshold causing decoherence."""
    def __init__(self, message="Quantum state coherence failure detected", coherence_value=None, 
                affected_patterns=None, location=None):
        self.message = message
        self.coherence_value = coherence_value
        self.affected_patterns = affected_patterns or []
        self.location = location
        super().__init__(self.message)
        
    def __str__(self):
        details = []
        if self.coherence_value is not None:
            details.append(f"Coherence value: {self.coherence_value:.6f}")
        if self.affected_patterns:
            details.append(f"Affected patterns: {len(self.affected_patterns)}")
        if self.location:
            details.append(f"Location: {self.location}")
        
        if details:
            return f"{self.message} - {'; '.join(details)}"
        return self.message

class RealityFragmentationError(Exception):
    """Exception raised when reality begins to fragment due to incompatible timelines or quantum states."""
    def __init__(self, message="Reality fragmentation detected", fragmentation_level=None,
                affected_regions=None, causality_breaks=None):
        self.message = message
        self.fragmentation_level = fragmentation_level
        self.affected_regions = affected_regions or []
        self.causality_breaks = causality_breaks or []
        super().__init__(self.message)
        
    def __str__(self):
        details = []
        if self.fragmentation_level is not None:
            details.append(f"Fragmentation level: {self.fragmentation_level:.6f}")
        if self.affected_regions:
            details.append(f"Affected regions: {len(self.affected_regions)}")
        if self.causality_breaks:
            details.append(f"Causality breaks: {len(self.causality_breaks)}")
        
        if details:
            return f"{self.message} - {'; '.join(details)}"
        return self.message

def initialize(timeline_engine=None):
    """
    Initialize the timeline engine and return the engine instance.
    
    Args:
        timeline_engine: Optional existing timeline engine instance
        
    Returns:
        The initialized timeline engine instance
    """
    logger = logging.getLogger("TimelineEngine")
    logger.info("Initializing Timeline Engine...")
    
    # Create and return the engine instance
    return TimelineEngine()

import logging
import time
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Tuple, Any, Callable, Optional

class BreathPhase(Enum):
    """Breath phases of the universe cycle"""
    INHALE = auto()       # Expansion, possibility generation, superposition
    HOLD_IN = auto()      # Stabilization, coherence maintenance
    EXHALE = auto()       # Contraction, probability collapse, resolution
    HOLD_OUT = auto()     # Void state, potential reset, quantum vacuum

def initialize(**kwargs):
    """
    Initialize the Timeline Engine with specified parameters.
    
    Args:
        **kwargs: Configuration parameters including:
            - start_time: Starting time (default: current time)
            - tick_rate: Number of ticks per second (default: 10)
            - time_scale: Time scaling factor relative to real time (default: 1.0)
            - breath_cycle_ticks: Number of ticks in a complete breath cycle (default: 100)
            - entity_id: The ID of the entity using this timeline engine
            
    Returns:
        Initialized TimelineEngine instance
    """
    logger = logging.getLogger("TimelineEngine")
    logger.info("Initializing Timeline Engine...")
    
    # Extract configuration parameters with defaults
    start_time = kwargs.get('start_time', time.time())
    tick_rate = kwargs.get('tick_rate', 10)
    time_scale = kwargs.get('time_scale', 1.0)
    breath_cycle_ticks = kwargs.get('breath_cycle_ticks', 100)
    entity_id = kwargs.get('entity_id', f"timeline_{int(start_time) % 10000}")
    
    # Create and configure timeline engine
    timeline_engine = TimelineEngine(
        start_time=start_time,
        tick_rate=tick_rate,
        time_scale=time_scale,
        breath_cycle_ticks=breath_cycle_ticks,
        entity_id=entity_id
    )
    
    logger.info(f"Timeline Engine initialized with ID {entity_id}")
    logger.info(f"Configuration: tick_rate={tick_rate}, time_scale={time_scale}, breath_cycle_ticks={breath_cycle_ticks}")
    
    return timeline_engine

class TimelineEngine:
    """
    Manages the timeline and temporal events for the ORAMA Framework.
    """
    
    def __init__(self, start_time: float, tick_rate: int = 10, time_scale: float = 1.0, 
                 breath_cycle_ticks: int = 100, entity_id: str = "main_timeline"):
        """
        Initialize the Timeline Engine.
        
        Args:
            start_time: Starting time (epoch seconds)
            tick_rate: Number of ticks per second
            time_scale: Time scaling factor relative to real time
            breath_cycle_ticks: Number of ticks in a complete breath cycle
            entity_id: The ID of this timeline engine instance
        """
        self.start_time = start_time
        self.tick_rate = tick_rate
        self.time_scale = time_scale
        self.breath_cycle_ticks = breath_cycle_ticks
        self.entity_id = entity_id
        
        # Initialize timeline state
        self.master_tick = 0
        self.current_real_time = start_time
        self.current_system_time = start_time
        self.is_running = False
        self.paused = False
        
        # Breath cycle configuration
        self.breath_phase_duration = {
            BreathPhase.INHALE: int(breath_cycle_ticks * 0.35),    # 35% of cycle
            BreathPhase.HOLD_IN: int(breath_cycle_ticks * 0.15),   # 15% of cycle
            BreathPhase.EXHALE: int(breath_cycle_ticks * 0.35),    # 35% of cycle
            BreathPhase.HOLD_OUT: int(breath_cycle_ticks * 0.15),  # 15% of cycle
        }
        
        # Initialize breath state
        self.current_breath_phase = BreathPhase.INHALE
        self.breath_phase_tick = 0
        self.breath_cycle_count = 0
        
        # Event management
        self.observers = []
        self.scheduled_events = []
        self.temporal_events = []
        self.event_history = []
        
        # Initialize logger
        self.logger = logging.getLogger(f"TimelineEngine_{entity_id}")
        self.logger.info(f"Timeline Engine initialized at time {start_time}")
        
    def start(self):
        """Start the timeline engine"""
        if self.is_running:
            self.logger.warning("Timeline Engine already running")
            return
        
        self.is_running = True
        self.paused = False
        self.logger.info("Timeline Engine started")
        
    def stop(self):
        """Stop the timeline engine"""
        if not self.is_running:
            self.logger.warning("Timeline Engine not running")
            return
        
        self.is_running = False
        self.logger.info("Timeline Engine stopped")
        
    def pause(self):
        """Pause the timeline engine"""
        if not self.is_running or self.paused:
            self.logger.warning("Timeline Engine not running or already paused")
            return
        
        self.paused = True
        self.logger.info("Timeline Engine paused")
        
    def resume(self):
        """Resume the timeline engine after pausing"""
        if not self.is_running or not self.paused:
            self.logger.warning("Timeline Engine not running or not paused")
            return
        
        self.paused = False
        self.logger.info("Timeline Engine resumed")
        
    def tick(self):
        """
        Advance the timeline by one tick.
        
        Returns:
            Event data dictionary for the current tick
        """
        if not self.is_running or self.paused:
            return None
        
        # Update tick count
        self.master_tick += 1
        
        # Update time
        tick_time_delta = 1.0 / self.tick_rate
        self.current_real_time = time.time()
        self.current_system_time += tick_time_delta * self.time_scale
        
        # Update breath cycle
        self._update_breath_cycle()
        
        # Process scheduled events
        events = self._process_events()
        
        # Create tick event data
        tick_data = {
            'type': 'tick',
            'tick': self.master_tick,
            'time': self.current_system_time,
            'real_time': self.current_real_time,
            'breath_phase': self.current_breath_phase.name,
            'breath_progress': self._get_breath_progress(),
            'breath_cycle': self.breath_cycle_count,
            'events': events
        }
        
        # Notify observers
        self._notify_observers(tick_data)
        
        return tick_data
    
    def _update_breath_cycle(self):
        """Update the breath cycle state based on current tick"""
        # Increment breath phase tick
        self.breath_phase_tick += 1
        
        # Check if we need to transition to next phase
        current_phase_duration = self.breath_phase_duration[self.current_breath_phase]
        
        if self.breath_phase_tick >= current_phase_duration:
            # Move to next phase
            self.breath_phase_tick = 0
            
            if self.current_breath_phase == BreathPhase.INHALE:
                self.current_breath_phase = BreathPhase.HOLD_IN
            elif self.current_breath_phase == BreathPhase.HOLD_IN:
                self.current_breath_phase = BreathPhase.EXHALE
            elif self.current_breath_phase == BreathPhase.EXHALE:
                self.current_breath_phase = BreathPhase.HOLD_OUT
            elif self.current_breath_phase == BreathPhase.HOLD_OUT:
                self.current_breath_phase = BreathPhase.INHALE
                # Completed a full breath cycle
                self.breath_cycle_count += 1
            
            # Log phase transition
            self.logger.debug(f"Breath phase changed to {self.current_breath_phase.name}, cycle {self.breath_cycle_count}")
            
            # Create and notify about breath phase change event
            breath_event = {
                'type': 'breath_phase_change',
                'phase': self.current_breath_phase.name,
                'cycle': self.breath_cycle_count,
                'time': self.current_system_time
            }
            
            self._notify_observers(breath_event)
    
    def _get_breath_progress(self):
        """Get the progress within the current breath phase (0.0 to 1.0)"""
        current_phase_duration = self.breath_phase_duration[self.current_breath_phase]
        if current_phase_duration == 0:
            return 0.0
        
        return self.breath_phase_tick / current_phase_duration
    
    def _process_events(self):
        """Process scheduled temporal events for the current tick"""
        current_events = []
        remaining_events = []
        
        # Check for events scheduled for this tick
        for event in self.scheduled_events:
            if event['tick'] <= self.master_tick:
                current_events.append(event)
                # Add to history
                self.event_history.append(event)
            else:
                remaining_events.append(event)
        
        # Update scheduled events list
        self.scheduled_events = remaining_events
        
        # Process current events
        for event in current_events:
            self._notify_observers(event)
            
        return current_events
    
    def schedule_event(self, event_type: str, tick_delay: int, data: Dict[str, Any] = None):
        """
        Schedule an event to occur after a specified number of ticks.
        
        Args:
            event_type: The type of event
            tick_delay: Number of ticks in the future to schedule the event
            data: Additional event data
            
        Returns:
            The scheduled event object
        """
        event_data = data.copy() if data else {}
        event_data.update({
            'type': event_type,
            'tick': self.master_tick + tick_delay,
            'scheduled_time': self.current_system_time + (tick_delay / self.tick_rate * self.time_scale),
            'creation_tick': self.master_tick
        })
        
        self.scheduled_events.append(event_data)
        self.logger.debug(f"Scheduled {event_type} event for tick {event_data['tick']}")
        
        return event_data
    
    def create_temporal_event(self, event_type: str, data: Dict[str, Any] = None):
        """
        Create an immediate temporal event and notify observers.
        
        Args:
            event_type: The type of event
            data: Additional event data
            
        Returns:
            The created event object
        """
        event_data = data.copy() if data else {}
        event_data.update({
            'type': event_type,
            'tick': self.master_tick,
            'time': self.current_system_time,
        })
        
        self.temporal_events.append(event_data)
        self.event_history.append(event_data)
        
        # Notify observers
        self._notify_observers(event_data)
        
        self.logger.debug(f"Created and processed immediate {event_type} event")
        
        return event_data
    
    def register_observer(self, observer_callback: Callable[[Dict[str, Any]], None]):
        """
        Register an observer to receive timeline events.
        
        Args:
            observer_callback: Function to call with event data
        """
        if observer_callback not in self.observers:
            self.observers.append(observer_callback)
            self.logger.debug(f"Registered new observer, total: {len(self.observers)}")
    
    def unregister_observer(self, observer_callback: Callable[[Dict[str, Any]], None]):
        """
        Unregister an observer.
        
        Args:
            observer_callback: Previously registered observer function
        """
        if observer_callback in self.observers:
            self.observers.remove(observer_callback)
            self.logger.debug(f"Unregistered observer, remaining: {len(self.observers)}")
    
    def _notify_observers(self, event_data: Dict[str, Any]):
        """
        Notify all observers about an event.
        
        Args:
            event_data: Event data to send to observers
        """
        for observer in self.observers:
            try:
                observer(event_data)
            except Exception as e:
                self.logger.error(f"Error in observer callback: {str(e)}")
    
    def get_current_time(self):
        """Get the current system time"""
        return self.current_system_time
    
    def get_current_breath_state(self):
        """
        Get the current breath cycle state.
        
        Returns:
            Dictionary with breath state information
        """
        return {
            'phase': self.current_breath_phase.name,
            'progress': self._get_breath_progress(),
            'cycle': self.breath_cycle_count,
            'phase_tick': self.breath_phase_tick,
            'phase_duration': self.breath_phase_duration[self.current_breath_phase]
        }
    
    def create_paradox_event(self, severity: float, description: str, 
                          location: Tuple[float, float, float, float] = None, 
                          affected_entities: List[str] = None):
        """
        Create a temporal paradox event.
        
        Args:
            severity: Paradox severity (0.0 to 1.0)
            description: Description of the paradox
            location: 4D spacetime coordinates of the paradox
            affected_entities: Entities affected by the paradox
            
        Returns:
            The created paradox event
        """
        if location is None:
            location = (0, 0, 0, self.current_system_time)
            
        paradox_data = {
            'severity': severity,
            'description': description,
            'location': location,
            'affected_entities': affected_entities or [],
            'resolution_status': 'unresolved'
        }
        
        # Create and return the event
        return self.create_temporal_event('temporal_paradox', paradox_data)