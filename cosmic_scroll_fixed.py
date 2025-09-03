# ================================================================
#  LOOM ASCENDANT COSMOS — COSMIC SCROLL MODULE
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
#  Integrity Hash (SHA-256): d3ab9688a5a20b8065990cd9b91805e3d892d6e72472f69dd9afe719250c5e37
# ================================================================

"""
Cosmic Scroll - Symbolic Pattern Repository and Motif Management System

This module implements the symbolic substrate layer of the Genesis Cosmos Engine,
managing patterns, motifs, entities, and events that form the foundational 
symbolic reality of the simulation. It integrates with other engines through
the observer pattern and breath synchronization.
"""

import logging
import time
import uuid
import hashlib
import random
import math
from collections import defaultdict, deque
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime

# Configure logging
logger = logging.getLogger("CosmicScroll")

# ================================================================
# CORE ENUMS AND DATA TYPES
# ================================================================

class BreathPhase(Enum):
    """Breath cycle phases for temporal synchronization"""
    INHALE = auto()
    HOLD_IN = auto() 
    EXHALE = auto()
    HOLD_OUT = auto()

class EntityType(Enum):
    """Types of entities in the cosmic simulation"""
    PHYSICAL = auto()      # Material entities with physical properties
    CONCEPTUAL = auto()    # Abstract concepts and ideas
    HYBRID = auto()        # Entities bridging physical and conceptual
    CONSCIOUS = auto()     # Self-aware entities with volition
    COLLECTIVE = auto()    # Groups or emergent collective entities

class EventType(Enum):
    """Types of events that can occur in the simulation"""
    CREATION = auto()      # Birth or formation of new entities
    TRANSFORMATION = auto()  # Change in entity properties or state
    INTERACTION = auto()   # Entities affecting each other
    DISSOLUTION = auto()   # Destruction or dispersion of entities
    AWAKENING = auto()     # Emergence of consciousness or awareness
    CONVERGENCE = auto()   # Multiple entities combining or aligning

class MotifCategory(Enum):
    """Categories of symbolic motifs"""
    ELEMENTAL = auto()     # Basic building blocks and forces
    STRUCTURAL = auto()    # Organizational patterns and hierarchies
    NARRATIVE = auto()     # Story patterns and temporal sequences
    ARCHETYPAL = auto()    # Universal symbolic forms
    HARMONIC = auto()      # Resonance and frequency patterns
    ETHICAL = auto()       # Moral and value-based patterns
    RECURSIVE = auto()     # Self-referential and fractal patterns

class MetabolicProcessType(Enum):
    """Types of ongoing transformation processes"""
    ENERGY_CONVERSION = auto()     # Converting between energy forms
    PATTERN_EVOLUTION = auto()     # Evolution of symbolic patterns
    INFORMATION_PROCESSING = auto()  # Data transformation and learning
    CONSCIOUSNESS_EXPANSION = auto()  # Growth of awareness and volition
    ETHICAL_DEVELOPMENT = auto()   # Moral and ethical evolution
    HARMONIC_RESONANCE = auto()    # Frequency and vibration processes

# ================================================================
# CORE DATA STRUCTURES
# ================================================================

@dataclass
class Motif:
    """
    A symbolic pattern that influences entities and events.
    
    Motifs represent recurring patterns in the symbolic substrate of reality,
    providing thematic influence and coherence to the simulation.
    """
    motif_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: MotifCategory = MotifCategory.ELEMENTAL
    attributes: Dict[str, float] = field(default_factory=dict)
    resonance_frequency: float = 1.0
    creation_time: float = field(default_factory=time.time)
    activity_level: float = 0.5
    
    def calculate_resonance(self, current_tick: int) -> float:
        """Calculate current resonance strength based on tick and frequency"""
        phase = (current_tick * self.resonance_frequency) % (2 * math.pi)
        base_resonance = math.sin(phase) * 0.5 + 0.5  # Normalize to 0-1
        return base_resonance * self.activity_level
    
    def apply_influence(self, target_attributes: Dict[str, float], strength: float = 1.0) -> Dict[str, float]:
        """Apply motif influence to target attributes"""
        influenced_attributes = target_attributes.copy()
        for attr_name, attr_value in self.attributes.items():
            if attr_name in influenced_attributes:
                influenced_attributes[attr_name] += attr_value * strength * 0.1
            else:
                influenced_attributes[attr_name] = attr_value * strength * 0.1
        return influenced_attributes

@dataclass  
class Entity:
    """
    An entity within the simulation that can have properties and participate in events.
    """
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entity_type: EntityType = EntityType.PHYSICAL
    properties: Dict[str, Any] = field(default_factory=dict)
    motifs: List[str] = field(default_factory=list)  # List of motif IDs
    creation_time: float = field(default_factory=time.time)
    last_interaction: float = field(default_factory=time.time)
    
    def add_motif(self, motif_id: str) -> None:
        """Add a motif to this entity"""
        if motif_id not in self.motifs:
            self.motifs.append(motif_id)
            logger.debug(f"Added motif {motif_id} to entity {self.entity_id}")
    
    def remove_motif(self, motif_id: str) -> bool:
        """Remove a motif from this entity"""
        if motif_id in self.motifs:
            self.motifs.remove(motif_id)
            logger.debug(f"Removed motif {motif_id} from entity {self.entity_id}")
            return True
        return False

@dataclass
class Event:
    """
    An occurrence within the simulation that affects entities and the narrative.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.INTERACTION
    description: str = ""
    entities: List[str] = field(default_factory=list)  # List of entity IDs
    properties: Dict[str, Any] = field(default_factory=dict)
    motifs: List[str] = field(default_factory=list)  # List of motif IDs
    timestamp: float = field(default_factory=time.time)
    
    def add_motif(self, motif_id: str) -> None:
        """Add a motif to this event"""
        if motif_id not in self.motifs:
            self.motifs.append(motif_id)

@dataclass
class MetabolicProcess:
    """
    A continuous process that transforms entities or simulation state over time.
    """
    process_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    process_type: MetabolicProcessType = MetabolicProcessType.ENERGY_CONVERSION
    entities: List[str] = field(default_factory=list)  # List of entity IDs
    rate: float = 1.0
    efficiency: float = 0.8
    active: bool = True
    resources_consumed: Dict[str, float] = field(default_factory=dict)
    products_generated: Dict[str, float] = field(default_factory=dict)
    
    def process(self, delta_time: float) -> Dict[str, Any]:
        """Execute one processing cycle"""
        if not self.active or self.rate <= 0:
            return {"processed": False, "reason": "inactive or zero rate"}
        
        # Calculate processing amount based on rate and time
        processing_amount = self.rate * delta_time * self.efficiency
        
        # Generate results
        results = {
            "processed": True,
            "amount": processing_amount,
            "resources_consumed": {k: v * processing_amount for k, v in self.resources_consumed.items()},
            "products_generated": {k: v * processing_amount for k, v in self.products_generated.items()},
            "entities_affected": len(self.entities)
        }
        
        logger.debug(f"Process {self.name} processed {processing_amount:.3f} units")
        return results

# ================================================================
# COSMIC SCROLL CORE CLASSES
# ================================================================

class CosmicScroll:
    """
    The core symbolic pattern repository of the simulation.
    
    Stores and manages the fundamental patterns that give rise to reality,
    serving as the symbolic substrate for the Genesis Cosmos Engine.
    """
    
    def __init__(self):
        # Pattern storage
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.active_threads: Dict[str, bool] = {}
        
        # Symbolic density tracking
        self.pattern_density_history = deque(maxlen=1000)
        self.last_density_calculation = 0.0
        
        # Thread management
        self.thread_priorities = defaultdict(float)
        self.thread_interactions = defaultdict(list)
        
        logger.info("CosmicScroll initialized with empty pattern repository")
    
    def add_pattern(self, pattern_id: str, pattern_data: Dict[str, Any]) -> bool:
        """Add a new pattern to the repository"""
        try:
            # Validate pattern data
            required_fields = ['type', 'data', 'metadata']
            if not all(field in pattern_data for field in required_fields):
                logger.warning(f"Pattern {pattern_id} missing required fields")
                return False
            
            # Store pattern with timestamp
            pattern_data['timestamp'] = time.time()
            pattern_data['access_count'] = 0
            pattern_data['last_accessed'] = time.time()
            
            self.patterns[pattern_id] = pattern_data
            logger.debug(f"Added pattern {pattern_id} to repository")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add pattern {pattern_id}: {e}")
            return False
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a pattern from the repository"""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern['access_count'] += 1
            pattern['last_accessed'] = time.time()
            return pattern
        return None
    
    def activate_thread(self, thread_id: str, priority: float = 1.0) -> bool:
        """Activate a narrative thread with given priority"""
        try:
            self.active_threads[thread_id] = True
            self.thread_priorities[thread_id] = priority
            logger.debug(f"Activated thread {thread_id} with priority {priority}")
            return True
        except Exception as e:
            logger.error(f"Failed to activate thread {thread_id}: {e}")
            return False
    
    def calculate_symbolic_density(self) -> float:
        """Calculate current symbolic density of the pattern space"""
        if not self.patterns:
            return 0.0
        
        try:
            # Calculate density based on pattern count, complexity, and interactions
            pattern_count = len(self.patterns)
            active_thread_count = sum(1 for active in self.active_threads.values() if active)
            
            # Base density from pattern count
            base_density = min(pattern_count / 100.0, 1.0)  # Normalize to 0-1
            
            # Thread activity bonus
            thread_bonus = min(active_thread_count / 10.0, 0.5)
            
            # Recent access activity
            current_time = time.time()
            recent_accesses = sum(1 for p in self.patterns.values() 
                                if current_time - p.get('last_accessed', 0) < 60.0)
            access_bonus = min(recent_accesses / pattern_count, 0.3) if pattern_count > 0 else 0
            
            density = base_density + thread_bonus + access_bonus
            
            # Store for history
            self.pattern_density_history.append(density)
            self.last_density_calculation = density
            
            return min(density, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Failed to calculate symbolic density: {e}")
            return self.last_density_calculation

class CosmicScrollManager:
    """
    Central management system for the Loom Ascendant Cosmos symbolic substrate.
    
    Handles simulation ticks, scroll memory, motif generation, and symbolic narrative 
    progression. Acts as the orchestrator for the symbolic layer of the reality stack.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CosmicScrollManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the manager with all necessary components"""
        # Core components
        self.cosmic_scroll = CosmicScroll()
        
        # Entity and motif management
        self.entities: Dict[str, Entity] = {}
        self.motifs: Dict[str, Motif] = {}
        self.events: List[Event] = []
        self.metabolic_processes: Dict[str, MetabolicProcess] = {}
        
        # Breath synchronization
        self.current_breath_phase = BreathPhase.EXHALE
        self.breath_cycle_time = 0.0
        self.breath_frequency = 1.0  # cycles per second
        
        # Observer callbacks
        self.observers: List[Callable] = []
        
        # Performance metrics
        self.metrics = {
            'entities_created': 0,
            'motifs_created': 0,
            'events_processed': 0,
            'metabolic_cycles': 0,
            'breath_cycles': 0
        }
        
        # Temporal tracking
        self.simulation_time = 0.0
        self.tick_count = 0
        
        logger.info("CosmicScrollManager initialized successfully")
    
    def register_observer(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register an observer for system events"""
        self.observers.append(callback)
        logger.debug(f"Registered observer: {callback.__name__}")
    
    def notify_observers(self, event_data: Dict[str, Any]) -> None:
        """Notify all observers of an event"""
        for callback in self.observers:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Observer callback failed: {e}")
    
    def tick(self, delta_time: float = 1.0) -> Dict[str, Any]:
        """
        Execute one simulation tick, updating all systems.
        
        Args:
            delta_time: Time elapsed since last tick
            
        Returns:
            Dict containing tick results and metrics
        """
        try:
            self.simulation_time += delta_time
            self.tick_count += 1
            
            # Update breath cycle
            self._update_breath_cycle(delta_time)
            
            # Process metabolic systems
            metabolic_results = self._process_metabolic_processes(delta_time)
            
            # Generate spontaneous events
            spontaneous_events = self._generate_spontaneous_events()
            
            # Update motifs
            motif_updates = self._update_motifs()
            
            # Calculate symbolic density
            symbolic_density = self.cosmic_scroll.calculate_symbolic_density()
            
            # Prepare tick results
            results = {
                'success': True,
                'simulation_time': self.simulation_time,
                'tick_count': self.tick_count,
                'breath_phase': self.current_breath_phase,
                'symbolic_density': symbolic_density,
                'metabolic_results': metabolic_results,
                'spontaneous_events': len(spontaneous_events),
                'motif_updates': motif_updates,
                'entity_count': len(self.entities),
                'active_motifs': sum(1 for m in self.motifs.values() if m.activity_level > 0.1),
                'metrics': self.metrics.copy()
            }
            
            # Notify observers
            self.notify_observers({
                'event_type': 'tick_completed',
                'results': results
            })
            
            logger.debug(f"Tick {self.tick_count} completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Tick processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'simulation_time': self.simulation_time,
                'tick_count': self.tick_count
            }
    
    def _update_breath_cycle(self, delta_time: float) -> None:
        """Update the breath synchronization cycle"""
        self.breath_cycle_time += delta_time * self.breath_frequency
        
        # Determine current phase based on cycle time
        cycle_position = (self.breath_cycle_time % 1.0)  # Normalize to 0-1
        
        if cycle_position < 0.25:
            new_phase = BreathPhase.INHALE
        elif cycle_position < 0.5:
            new_phase = BreathPhase.HOLD_IN
        elif cycle_position < 0.75:
            new_phase = BreathPhase.EXHALE
        else:
            new_phase = BreathPhase.HOLD_OUT
        
        # Track phase changes
        if new_phase != self.current_breath_phase:
            self.current_breath_phase = new_phase
            self.metrics['breath_cycles'] += 1
            
            # Notify observers of phase change
            self.notify_observers({
                'event_type': 'breath_phase_changed',
                'new_phase': new_phase,
                'cycle_time': self.breath_cycle_time
            })
    
    def _process_metabolic_processes(self, delta_time: float) -> List[Dict[str, Any]]:
        """Process all active metabolic processes"""
        results = []
        for process in self.metabolic_processes.values():
            if process.active:
                result = process.process(delta_time)
                if result.get('processed', False):
                    results.append({
                        'process_id': process.process_id,
                        'name': process.name,
                        'result': result
                    })
        
        self.metrics['metabolic_cycles'] += len(results)
        return results
    
    def _generate_spontaneous_events(self) -> List[Event]:
        """Generate spontaneous events based on current system state"""
        events = []
        
        # Generate events based on symbolic density and entity interactions
        density = self.cosmic_scroll.calculate_symbolic_density()
        event_probability = density * 0.1  # Higher density = more events
        
        if random.random() < event_probability and self.entities:
            # Create a random interaction event
            entity_ids = list(self.entities.keys())
            selected_entities = random.sample(entity_ids, min(2, len(entity_ids)))
            
            event = Event(
                event_type=EventType.INTERACTION,
                description=f"Spontaneous interaction between entities",
                entities=selected_entities,
                properties={'spontaneous': True, 'density_factor': density}
            )
            
            events.append(event)
            self.events.append(event)
            self.metrics['events_processed'] += 1
        
        return events
    
    def _update_motifs(self) -> Dict[str, float]:
        """Update all motifs and return summary of changes"""
        updates = {}
        
        for motif_id, motif in self.motifs.items():
            old_activity = motif.activity_level
            
            # Calculate resonance for current tick
            resonance = motif.calculate_resonance(self.tick_count)
            
            # Update activity level based on resonance and interactions
            motif.activity_level = 0.9 * motif.activity_level + 0.1 * resonance
            
            # Track significant changes
            activity_change = abs(motif.activity_level - old_activity)
            if activity_change > 0.1:
                updates[motif_id] = activity_change
        
        return updates
    
    # ================================================================
    # PUBLIC API METHODS
    # ================================================================
    
    def create_entity(self, name: str, entity_type: EntityType, 
                     properties: Dict[str, Any] = None) -> str:
        """Create a new entity and add it to the system"""
        entity = Entity(
            name=name,
            entity_type=entity_type,
            properties=properties or {}
        )
        
        self.entities[entity.entity_id] = entity
        self.metrics['entities_created'] += 1
        
        logger.info(f"Created entity '{name}' with ID {entity.entity_id}")
        
        # Notify observers
        self.notify_observers({
            'event_type': 'entity_created',
            'entity_id': entity.entity_id,
            'entity_type': entity_type,
            'name': name
        })
        
        return entity.entity_id
    
    def create_motif(self, name: str, category: MotifCategory, 
                    attributes: Dict[str, float] = None) -> str:
        """Create a new motif and add it to the system"""
        motif = Motif(
            name=name,
            category=category,
            attributes=attributes or {}
        )
        
        self.motifs[motif.motif_id] = motif
        self.metrics['motifs_created'] += 1
        
        logger.info(f"Created motif '{name}' with ID {motif.motif_id}")
        
        # Notify observers
        self.notify_observers({
            'event_type': 'motif_created',
            'motif_id': motif.motif_id,
            'category': category,
            'name': name
        })
        
        return motif.motif_id
    
    def create_metabolic_process(self, name: str, process_type: MetabolicProcessType,
                               entities: List[str] = None, rate: float = 1.0) -> str:
        """Create a new metabolic process"""
        process = MetabolicProcess(
            name=name,
            process_type=process_type,
            entities=entities or [],
            rate=rate
        )
        
        self.metabolic_processes[process.process_id] = process
        
        logger.info(f"Created metabolic process '{name}' with ID {process.process_id}")
        return process.process_id
    
    def associate_motif_with_entity(self, entity_id: str, motif_id: str) -> bool:
        """Associate a motif with an entity"""
        if entity_id in self.entities and motif_id in self.motifs:
            self.entities[entity_id].add_motif(motif_id)
            logger.debug(f"Associated motif {motif_id} with entity {entity_id}")
            return True
        return False
    
    def get_entity_motifs(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all motifs associated with an entity"""
        if entity_id not in self.entities:
            return []
        
        entity = self.entities[entity_id]
        motif_data = []
        
        for motif_id in entity.motifs:
            if motif_id in self.motifs:
                motif = self.motifs[motif_id]
                motif_data.append({
                    'motif_id': motif_id,
                    'name': motif.name,
                    'category': motif.category,
                    'activity_level': motif.activity_level,
                    'resonance': motif.calculate_resonance(self.tick_count)
                })
        
        return motif_data
    
    def get_simulation_state(self) -> Dict[str, Any]:
        """Get comprehensive simulation state information"""
        return {
            'simulation_time': self.simulation_time,
            'tick_count': self.tick_count,
            'breath_phase': self.current_breath_phase,
            'breath_cycle_time': self.breath_cycle_time,
            'entity_count': len(self.entities),
            'motif_count': len(self.motifs),
            'event_count': len(self.events),
            'process_count': len(self.metabolic_processes),
            'symbolic_density': self.cosmic_scroll.calculate_symbolic_density(),
            'metrics': self.metrics.copy(),
            'active_entities': sum(1 for e in self.entities.values() 
                                 if time.time() - e.last_interaction < 300),  # Active in last 5 min
            'active_motifs': sum(1 for m in self.motifs.values() 
                               if m.activity_level > 0.1),
            'active_processes': sum(1 for p in self.metabolic_processes.values() 
                                  if p.active)
        }

# ================================================================
# MODULE INITIALIZATION
# ================================================================

# Create the singleton instance
cosmic_scroll_manager = CosmicScrollManager()

# Module exports
__all__ = [
    'BreathPhase', 'EntityType', 'EventType', 'MotifCategory', 'MetabolicProcessType',
    'Motif', 'Entity', 'Event', 'MetabolicProcess',
    'CosmicScroll', 'CosmicScrollManager',
    'cosmic_scroll_manager'
]

logger.info("Cosmic Scroll module initialized successfully")
