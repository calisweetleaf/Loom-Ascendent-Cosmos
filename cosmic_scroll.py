# ============================================================
# Module: cosmic_scroll.py
# Description: Central symbolic engine for recursive universe simulation
# Author: Morpheus (author), Somnus Development Collective
# License: Proprietary Software License Agreement (Somnus Development Collective)
# Date: 2025-04-13T04:45:23.708211Z
# SHA-256: 979bc49912d26322228638c9c88f6f8f5942f3752d2334b8c160788f7b24aaa4
# ============================================================
# ===== Standard Library =====
import random
import logging
import math
import uuid
import time
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Dict, Any, Union, Optional

# ===== Engine Modules =====
from aether_engine import AetherEngine
from quantum_physics import QuantumPhysics
from quantum_bridge import QuantumBridge
from quantum_and_physics import QuantumAndPhysics
from perception_module import PerceptionModule
from paradox_engine import ParadoxEngine
from harmonic_engine import HarmonicEngine

# Optional: only include this if it's used directly
from main import CoreDispatcher

# ===== Cosmos Scroll Components =====
from cosmic_scroll import CosmicScroll
from motif import Motif, MotifCategory
from entity import Entity, EntityType
from event import Event, EventType
from metabolic_process import MetabolicProcess
from breath import BreathPhase



class CosmicScrollManager:
    """
    Central management system for the Loom Ascendant Cosmos engine.
    
    Handles simulation ticks, scroll memory, motif generation, and symbolic narrative progression.
    Acts as the runtime loop for the entire system.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CosmicScrollManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the Cosmic Scroll Manager"""
        self.entities = {}  # entity_id -> entity object
        self.entity_types = defaultdict(set)  # entity_type -> set of entity_ids
        
        self.motif_library = {}  # motif_id -> motif data
        self.entity_motifs = defaultdict(set)  # entity_id -> set of motif_ids
        
        self.event_history = []  # List of all events
        self.recent_events = deque(maxlen=100)  # Recent events for quick access
        
        self.tick_count = 0
        self.time_scale = 1.0  # Time dilation factor
        self.breath_cycle_length = 12  # Ticks per complete breath cycle
        self.breath_phase = BreathPhase.INHALE
        self.breath_progress = 0.0  # 0.0 to 1.0 within current phase
        
        self.inhale_ratio = 0.3    # Proportion of cycle spent inhaling
        self.hold_in_ratio = 0.2   # Proportion of cycle spent holding in
        self.exhale_ratio = 0.3    # Proportion of cycle spent exhaling
        self.hold_out_ratio = 0.2  # Proportion of cycle spent holding out
        
        self.history = {
            "creation_time": datetime.now(),
            "tick_history": [],
            "significant_events": []
        }
        
        self.motif_feedback_queue = deque(maxlen=50)  # Recent motif data for external systems
        
        logger.info("CosmicScrollManager initialized")
    
    def tick(self, delta_time: float = 1.0) -> Dict:
        """
        Advance the simulation forward one step.
        
        Args:
            delta_time: Time multiplier for this tick
            
        Returns:
            Dict containing information about the current tick
        """
        adjusted_delta = delta_time * self.time_scale
        self.tick_count += 1
        
        # Update breath cycle
        self._update_breath_cycle()
        
        # Process entity evolution
        evolved_entities = self._evolve_entities(adjusted_delta)
        
        # Generate events from entity interactions
        generated_events = self._generate_events()
        
        # Process any pending events
        for event in generated_events:
            self.log_event(event)
        
        # Record tick information
        tick_info = {
            "tick_id": self.tick_count,
            "timestamp": datetime.now(),
            "delta_time": adjusted_delta,
            "breath_phase": self.breath_phase.value,
            "breath_progress": self.breath_progress,
            "entities_evolved": len(evolved_entities),
            "events_generated": len(generated_events)
        }
        
        # Store tick history (limiting to last 100 ticks)
        self.history["tick_history"].append(tick_info)
        if len(self.history["tick_history"]) > 100:
            self.history["tick_history"] = self.history["tick_history"][-100:]
            
        logger.debug(f"Tick {self.tick_count} completed")
        return tick_info
    
    def _update_breath_cycle(self):
        """Update the breath cycle phase and progress"""
        total_progress = (self.tick_count % self.breath_cycle_length) / self.breath_cycle_length
        
        # Determine current phase
        if total_progress < self.inhale_ratio:
            self.breath_phase = BreathPhase.INHALE
            self.breath_progress = total_progress / self.inhale_ratio
        elif total_progress < (self.inhale_ratio + self.hold_in_ratio):
            self.breath_phase = BreathPhase.HOLD_IN
            self.breath_progress = (total_progress - self.inhale_ratio) / self.hold_in_ratio
        elif total_progress < (self.inhale_ratio + self.hold_in_ratio + self.exhale_ratio):
            self.breath_phase = BreathPhase.EXHALE
            self.breath_progress = (total_progress - self.inhale_ratio - self.hold_in_ratio) / self.exhale_ratio
        else:
            self.breath_phase = BreathPhase.HOLD_OUT
            self.breath_progress = (total_progress - self.inhale_ratio - self.hold_in_ratio - self.exhale_ratio) / self.hold_out_ratio
    
    def _evolve_entities(self, delta_time: float) -> List[str]:
        """
        Evolve all entities forward in time.
        
        Args:
            delta_time: Time multiplier for this evolution step
            
        Returns:
            List of entity IDs that were evolved
        """
        evolved_entities = []
        
        for entity_id, entity in self.entities.items():
            if hasattr(entity, 'evolve'):
                try:
                    entity.evolve(delta_time)
                    evolved_entities.append(entity_id)
                except Exception as e:
                    logger.error(f"Error evolving entity {entity_id}: {str(e)}")
        
        return evolved_entities
    
    def _generate_events(self) -> List[Dict]:
        """
        Generate events from entity interactions.
        This is a placeholder for more complex event generation logic.
        
        Returns:
            List of generated events
        """
        # This would be implemented with more sophisticated logic
        # that detects meaningful interactions between entities
        events = []
        
        # Simple example: random events for demonstration
        if random.random() < 0.1:  # 10% chance per tick
            # Get random entities for interaction
            if len(self.entities) >= 2:
                entities = random.sample(list(self.entities.keys()), 2)
                
                event = {
                    "type": random.choice(list(EventType)).value,
                    "timestamp": self.tick_count,
                    "entities_involved": entities,
                    "description": f"Random interaction between {entities[0]} and {entities[1]}",
                    "importance": random.uniform(0.1, 1.0)
                }
                
                events.append(event)
        
        return events
    
    def register_entity(self, entity) -> str:
        """
        Register an entity with the Cosmic Scroll system.
        
        Args:
            entity: The entity object to register
            
        Returns:
            The entity ID
        """
        # Ensure entity has an ID
        if not hasattr(entity, 'entity_id'):
            entity.entity_id = f"{entity.__class__.__name__.lower()}_{uuid.uuid4().hex}"
        
        entity_id = entity.entity_id
        
        # Register the entity
        self.entities[entity_id] = entity
        
        # Register entity type if available
        if hasattr(entity, 'entity_type'):
            entity_type = entity.entity_type
            if isinstance(entity_type, EntityType):
                entity_type = entity_type.value
            self.entity_types[entity_type].add(entity_id)
        
        logger.debug(f"Entity {entity_id} registered")
        return entity_id
    
    def log_event(self, event: Dict):
        """
        Log an event in the cosmic scroll history.
        
        Args:
            event: The event data to log
        """
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = self.tick_count
            
        # Add a unique ID for the event
        event["event_id"] = f"event_{uuid.uuid4().hex}"
        
        # Log the event
        self.event_history.append(event)
        self.recent_events.append(event)
        
        # If it's a significant event, add to significant events history
        if event.get("importance", 0.0) > 0.7:
            self.history["significant_events"].append(event)
            logger.info(f"Significant event: {event['description']}")
        else:
            logger.debug(f"Event logged: {event['description']}")
    
    def get_motif_feedback(self, max_items: int = 10) -> List[Dict]:
        """
        Retrieve motif data for external systems.
        
        Args:
            max_items: Maximum number of motif items to return
            
        Returns:
            List of recent motif data
        """
        # Get the most recent motifs from the feedback queue
        recent_motifs = list(self.motif_feedback_queue)[-max_items:]
        
        # Create a feedback packet
        feedback = {
            "tick_count": self.tick_count,
            "breath_phase": self.breath_phase.value,
            "breath_progress": self.breath_progress,
            "motifs": recent_motifs,
            "motif_count": len(self.motif_library),
            "entity_count": len(self.entities),
            "dominant_categories": self._get_dominant_motif_categories()
        }
        
        return feedback
        
    def _get_dominant_motif_categories(self) -> Dict[str, float]:
        """Calculate the currently dominant motif categories in the system"""
        category_strengths = defaultdict(float)
        
        # Sum the strength of all motifs by category
        for motif in self.motif_library.values():
            category = motif["category"]
            strength = motif["strength"] * motif["resonance"]
            category_strengths[category] += strength
        
        # Normalize to get relative strengths
        total_strength = sum(category_strengths.values()) or 1.0
        return {category: strength/total_strength for category, strength in category_strengths.items()}
class CosmicScroll:
    def __init__(self):
        self.entities = {}
        self.world_state = None
        self.motif_library = {}
        self.motif_feedback_queue = deque(maxlen=50)
        self.entity_motifs = defaultdict(set)
        self.entity_types = defaultdict(set)
        self.event_history = []
        self.tick_count = 0
        self.breath_phase = BreathPhase.INHALE
        self.breath_progress = 0.0
        self.history = {
            "creation_time": datetime.now(),
            "tick_history": [],
            "significant_events": []
        }

    def tick(self):
        # Update lifecycle for entities with 'civilization' in their name
        for entity in self.entities.values():
            if "civilization" in entity.name.lower() and hasattr(entity, 'birth_time'):
                entity.last_update_time = max(
                    getattr(entity, 'last_update_time', 0.0),
                    getattr(self.world_state, 'current_time', 0.0)
                )

                entity.age = max(
                    getattr(entity, 'age', 0.0),
                    entity.last_update_time - entity.birth_time
                )

                entity.growth_cycles_completed = max(
                    getattr(entity, 'growth_cycles_completed', 0),
                    entity.age // entity.growth_cycle_duration
                )

                entity.growth_factor = min(1.0, entity.age / entity.growth_cycle_duration)
                entity.health = max(0.0, 1.0 - (entity.age / entity.lifespan))

                maturation = 1.0 - (entity.age / entity.lifespan)
                entity.maturation_rate = min(1.0, max(0.0, maturation))

    def get_motif_feedback(self, max_items: int = 10) -> List[Dict]:
        """
        Retrieve motif data for external systems.
        
        Args:
            max_items: Maximum number of motif items to return
            
        Returns:
            List of recent motif data
        """
        # Get the most recent motifs from the feedback queue
        recent_motifs = list(self.motif_feedback_queue)[-max_items:]
        
        # Create a feedback packet
        feedback = {
            "tick_count": self.tick_count,
            "breath_phase": self.breath_phase.value,
            "breath_progress": self.breath_progress,
            "motifs": recent_motifs,
            "motif_count": len(self.motif_library),
            "entity_count": len(self.entities),
            "dominant_categories": self._get_dominant_motif_categories()
        }
        
        return feedback
    
    def _get_dominant_motif_categories(self) -> Dict[str, float]:
        """Calculate the currently dominant motif categories in the system"""
        category_strengths = defaultdict(float)
        
        # Sum the strength of all motifs by category
        for motif in self.motif_library.values():
            category = motif["category"]
            strength = motif["strength"] * motif["resonance"]
            category_strengths[category] += strength
        
        # Normalize to get relative strengths
        total_strength = sum(category_strengths.values()) or 1.0
        return {category: strength/total_strength for category, strength in category_strengths.items()}
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Any]:
        """Retrieve an entity by its ID"""
        return self.entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: Union[str, EntityType]) -> List[Any]:
        """Retrieve all entities of a specific type"""
        if isinstance(entity_type, EntityType):
            entity_type = entity_type.value
            
        entity_ids = self.entity_types.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    def get_entity_motifs(self, entity_id: str) -> List[Dict]:
        """Get all motifs associated with an entity"""
        motif_ids = self.entity_motifs.get(entity_id, set())
        return [self.motif_library[mid] for mid in motif_ids if mid in self.motif_library]
    
    def get_events_by_entity(self, entity_id: str, max_events: int = 10) -> List[Dict]:
        """Get events involving a specific entity"""
        events = [
            event for event in self.event_history 
            if entity_id in event.get("entities_involved", [])
        ]
        return sorted(events, key=lambda e: e.get("timestamp", 0), reverse=True)[:max_events]
    
    def get_simulation_stats(self) -> Dict:
        """Get current statistics about the simulation"""
        entity_type_counts = {etype: len(eids) for etype, eids in self.entity_types.items()}
        
        return {
            "tick_count": self.tick_count,
            "entity_count": len(self.entities),
            "entity_types": entity_type_counts,
            "event_count": len(self.event_history),
            "motif_count": len(self.motif_library),
            "breath_phase": self.breath_phase.value,
            "creation_time": self.history["creation_time"],
            "runtime": (datetime.now() - self.history["creation_time"]).total_seconds(),
            "significant_events": len(self.history["significant_events"])
        }
        
    def _generate_motif_name(self, category: MotifCategory) -> str:
        """Generate a thematic name for a motif based on its category"""
        # Dictionary of prefix and suffix options for each category
        name_components = {
            MotifCategory.LUMINOUS: {
                "prefixes": ["radiant", "glowing", "shining", "illuminated", "bright"],
                "roots": ["light", "sun", "star", "dawn", "glow"],
                "suffixes": ["beam", "ray", "flare", "spark", "corona"]
            },
            MotifCategory.ABYSSAL: {
                "prefixes": ["deep", "dark", "void", "hollow", "endless"],
                "roots": ["abyss", "depth", "void", "darkness", "shadow"],
                "suffixes": ["pit", "chasm", "well", "trench", "gulf"]
            },
            MotifCategory.VITAL: {
                "prefixes": ["living", "growing", "thriving", "flourishing", "verdant"],
                "roots": ["life", "growth", "bloom", "pulse", "breath"],
                "suffixes": ["seed", "root", "heart", "core", "essence"]
            },
            MotifCategory.ENTROPIC: {
                "prefixes": ["decaying", "fading", "eroding", "dissolving", "withering"],
                "roots": ["entropy", "decay", "dust", "ash", "rust"],
                "suffixes": ["dissolution", "erosion", "fall", "decline", "end"]
            },
            MotifCategory.CRYSTALLINE: {
                "prefixes": ["ordered", "structured", "patterned", "aligned", "latticed"],
                "roots": ["crystal", "pattern", "form", "structure", "symmetry"],
                "suffixes": ["lattice", "matrix", "grid", "array", "framework"]
            },
            MotifCategory.CHAOTIC: {
                "prefixes": ["wild", "turbulent", "swirling", "disordered", "random"],
                "roots": ["chaos", "storm", "maelstrom", "tempest", "turmoil"],
                "suffixes": ["vortex", "whirl", "tumult", "frenzy", "disorder"]
            },
            MotifCategory.ELEMENTAL: {
                "prefixes": ["primal", "raw", "fundamental", "essential", "primordial"],
                "roots": ["element", "earth", "water", "fire", "air"],
                "suffixes": ["essence", "force", "power", "current", "flow"]
            },
            MotifCategory.HARMONIC: {
                "prefixes": ["resonant", "balanced", "harmonious", "attuned", "aligned"],
                "roots": ["harmony", "resonance", "balance", "chord", "rhythm"],
                "suffixes": ["wave", "pulse", "oscillation", "cycle", "frequency"]
            },
            MotifCategory.RECURSIVE: {
                "prefixes": ["nested", "iterative", "folded", "layered", "self-similar"],
                "roots": ["recursion", "fractal", "loop", "cycle", "pattern"],
                "suffixes": ["iteration", "reflection", "echo", "mirror", "spiral"]
            },
            MotifCategory.TEMPORAL: {
                "prefixes": ["flowing", "passing", "changing", "cycling", "eternal"],
                "roots": ["time", "moment", "epoch", "era", "age"],
                "suffixes": ["flow", "stream", "cycle", "continuity", "progression"]
            },
            MotifCategory.DIMENSIONAL: {
                "prefixes": ["spatial", "volumetric", "expansive", "containing", "vast"],
                "roots": ["space", "dimension", "realm", "domain", "field"],
                "suffixes": ["expanse", "extent", "boundary", "horizon", "frontier"]
            },
            MotifCategory.CONNECTIVE: {
                "prefixes": ["linking", "binding", "joining", "weaving", "connecting"],
                "roots": ["connection", "network", "web", "link", "bond"],
                "suffixes": ["thread", "bridge", "nexus", "junction", "pathway"]
            },
            MotifCategory.SHADOW: {
                "prefixes": ["hidden", "veiled", "obscured", "occluded", "shrouded"],
                "roots": ["shadow", "veil", "mask", "secret", "mystery"],
                "suffixes": ["cloak", "curtain", "shroud", "cover", "fog"]
            },
            MotifCategory.ASCENDANT: {
                "prefixes": ["rising", "ascending", "elevating", "transcending", "surpassing"],
                "roots": ["ascension", "peak", "summit", "zenith", "pinnacle"],
                "suffixes": ["flight", "climb", "journey", "transformation", "evolution"]
            }
        }
        
        components = name_components.get(category, {
            "prefixes": ["mysterious", "unknown", "undefined"],
            "roots": ["pattern", "form", "essence"],
            "suffixes": ["manifestation", "presence", "aspect"]
        })
        
        # Generate a name using the components
        name_structure = random.choice([
            "{prefix}_{root}",
            "{root}_{suffix}",
            "{prefix}_{root}_{suffix}"
        ])
        
        name_parts = {
            "prefix": random.choice(components["prefixes"]),
            "root": random.choice(components["roots"]),
            "suffix": random.choice(components["suffixes"])
        }
        
        return name_structure.format(**name_parts)
```
# ================================================================
#  LOOM ASCENDANT COSMOS — RECURSIVE SYSTEM MODULE
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
#  Integrity Hash (SHA-256): d3ab9688a5a20b8065990cd9b91805e3d892d6e72472f69dd9afe719250c5e37
# ================================================================

CosmicScroll Engine
class CosmicScroll:
    def __init__(self):
        self.entities = {}
        self.world_state = None

    def tick(self):
        # Update lifecycle for entities with 'civilization' in their name
        for entity in self.entities.values():
            if "civilization" in entity.name.lower() and hasattr(entity, 'birth_time'):
                entity.last_update_time = max(
                    getattr(entity, 'last_update_time', 0.0),
                    getattr(self.world_state, 'current_time', 0.0)
                )

                entity.age = max(
                    getattr(entity, 'age', 0.0),
                    entity.last_update_time - entity.birth_time
                )

                entity.growth_cycles_completed = max(
                    getattr(entity, 'growth_cycles_completed', 0),
                    entity.age // entity.growth_cycle_duration
                )

                entity.growth_factor = min(1.0, entity.age / entity.growth_cycle_duration)
                entity.health = max(0.0, 1.0 - (entity.age / entity.lifespan))

                maturation = 1.0 - (entity.age / entity.lifespan)
                entity.maturation_rate = min(1.0, max(0.0, maturation))
process_motifs = {
    MetabolicProcess.SYMBOLIC_ABSORPTION: ["meaning_derivation", "semantic_integration", "symbolic_conversion"],
}
# Process-specific motifs
from enum import Enum

class MetabolicProcess(Enum):
    """Types of metabolic processes that can occur in living entities"""
    PHOTOSYNTHESIS = "photosynthesis"
    RESPIRATION = "respiration"
    CHEMOSYNTHESIS = "chemosynthesis"
    RADIOSYNTHESIS = "radiosynthesis"
    QUANTUM_ENTANGLEMENT = "quantum_entanglement"
    SYMBOLIC_ABSORPTION = "symbolic_absorption"
    MOTIF_CYCLING = "motif_cycling"
    HARMONIC_RESONANCE = "harmonic_resonance"

process_motifs = {
    MetabolicProcess.PHOTOSYNTHESIS: ["light_harvesting", "solar_alchemy", "growth_cycle"],
    MetabolicProcess.RESPIRATION: ["oxidation_rhythm", "energy_extraction", "cellular_breath"],
    MetabolicProcess.CHEMOSYNTHESIS: ["mineral_transmutation", "chemical_cascade", "elemental_binding"],
    MetabolicProcess.RADIOSYNTHESIS: ["radiation_harvest", "particle_weaving", "decay_reversal"],
    MetabolicProcess.QUANTUM_ENTANGLEMENT: ["probability_harvest", "quantum_threading", "uncertainty_mapping"],
    MetabolicProcess.SYMBOLIC_ABSORPTION: ["meaning_derivation", "semantic_integration", "symbolic_conversion"],
    MetabolicProcess.MOTIF_CYCLING: ["pattern_recognition", "motif_amplification", "thematic_resonance"],
    MetabolicProcess.HARMONIC_RESONANCE: ["harmonic_alignment", "frequency_attunement", "wave_synchronization"]
}
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional, Union, Any

class CosmicScroll:
    def get_motif_feedback(self, max_items: int = 10) -> List[Dict]:
        """
        Retrieve motif data for external systems.
        
        Args:
            max_items: Maximum number of motif items to return
            
        Returns:
            List of recent motif data
        """
        # Get the most recent motifs from the feedback queue
        recent_motifs = list(self.motif_feedback_queue)[-max_items:]
        
        # Create a feedback packet
        feedback = {
            "tick_count": self.tick_count,
            "breath_phase": self.breath_phase.value,
            "breath_progress": self.breath_progress,
            "motifs": recent_motifs,
            "motif_count": len(self.motif_library),
            "entity_count": len(self.entities),
            "dominant_categories": self._get_dominant_motif_categories()
        }
        
        return feedback
    
    def _get_dominant_motif_categories(self) -> Dict[str, float]:
        """Calculate the currently dominant motif categories in the system"""
        category_strengths = defaultdict(float)
        
        # Sum the strength of all motifs by category
        for motif in self.motif_library.values():
            category = motif["category"]
            strength = motif["strength"] * motif["resonance"]
            category_strengths[category] += strength
        
        # Normalize to get relative strengths
        total_strength = sum(category_strengths.values()) or 1.0
        return {category: strength/total_strength for category, strength in category_strengths.items()}
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Any]:
        """Retrieve an entity by its ID"""
        return self.entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: Union[str, EntityType]) -> List[Any]:
        """Retrieve all entities of a specific type"""
        if isinstance(entity_type, EntityType):
            entity_type = entity_type.value
            
        entity_ids = self.entity_types.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    def get_entity_motifs(self, entity_id: str) -> List[Dict]:
        """Get all motifs associated with an entity"""
        motif_ids = self.entity_motifs.get(entity_id, set())
        return [self.motif_library[mid] for mid in motif_ids if mid in self.motif_library]
    
    def get_events_by_entity(self, entity_id: str, max_events: int = 10) -> List[Dict]:
        """Get events involving a specific entity"""
        events = [
            event for event in self.event_history 
            if entity_id in event.get("entities_involved", [])
        ]
        return sorted(events, key=lambda e: e.get("timestamp", 0), reverse=True)[:max_events]
    
    def get_simulation_stats(self) -> Dict:
        """Get current statistics about the simulation"""
        entity_type_counts = {etype: len(eids) for etype, eids in self.entity_types.items()}
        
        return {
            "tick_count": self.tick_count,
            "entity_count": len(self.entities),
            "entity_types": entity_type_counts,
            "event_count": len(self.event_history),
            "motif_count": len(self.motif_library),
            "breath_phase": self.breath_phase.value,
            "creation_time": self.history["creation_time"],
            "runtime": (datetime.now() - self.history["creation_time"]).total_seconds(),
            "significant_events": len(self.history["significant_events"])
        }
def _generate_motif_name(self, category: MotifCategory) -> str:
    """Generate a thematic name for a motif based on its category"""
    # Dictionary of prefix and suffix options for each category
    name_components = {
        MotifCategory.LUMINOUS: {
            "prefixes": ["radiant", "glowing", "shining", "illuminated", "bright"],
            "roots": ["light", "sun", "star", "dawn", "glow"],
            "suffixes": ["beam", "ray", "flare", "spark", "corona"]
        },
        MotifCategory.ABYSSAL: {
            "prefixes": ["deep", "dark", "void", "hollow", "endless"],
            "roots": ["abyss", "depth", "void", "darkness", "shadow"],
            "suffixes": ["pit", "chasm", "well", "trench", "gulf"]
        },
        MotifCategory.VITAL: {
            "prefixes": ["living", "growing", "thriving", "flourishing", "verdant"],
            "roots": ["life", "growth", "bloom", "pulse", "breath"],
            "suffixes": ["seed", "root", "heart", "core", "essence"]
        },
        MotifCategory.ENTROPIC: {
            "prefixes": ["decaying", "fading", "eroding", "dissolving", "withering"],
            "roots": ["entropy", "decay", "dust", "ash", "rust"],
            "suffixes": ["dissolution", "erosion", "fall", "decline", "end"]
        },
        MotifCategory.CRYSTALLINE: {
            "prefixes": ["ordered", "structured", "patterned", "aligned", "latticed"],
            "roots": ["crystal", "pattern", "form", "structure", "symmetry"],
            "suffixes": ["lattice", "matrix", "grid", "array", "framework"]
        },
        MotifCategory.CHAOTIC: {
            "prefixes": ["wild", "turbulent", "swirling", "disordered", "random"],
            "roots": ["chaos", "storm", "maelstrom", "tempest", "turmoil"],
            "suffixes": ["vortex", "whirl", "tumult", "frenzy", "disorder"]
        },
        MotifCategory.ELEMENTAL: {
            "prefixes": ["primal", "raw", "fundamental", "essential", "primordial"],
            "roots": ["element", "earth", "water", "fire", "air"],
            "suffixes": ["essence", "force", "power", "current", "flow"]
        },
        MotifCategory.HARMONIC: {
            "prefixes": ["resonant", "balanced", "harmonious", "attuned", "aligned"],
            "roots": ["harmony", "resonance", "balance", "chord", "rhythm"],
            "suffixes": ["wave", "pulse", "oscillation", "cycle", "frequency"]
        },
        MotifCategory.RECURSIVE: {
            "prefixes": ["nested", "iterative", "folded", "layered", "self-similar"],
            "roots": ["recursion", "fractal", "loop", "cycle", "pattern"],
            "suffixes": ["iteration", "reflection", "echo", "mirror", "spiral"]
        },
        MotifCategory.TEMPORAL: {
            "prefixes": ["flowing", "passing", "changing", "cycling", "eternal"],
            "roots": ["time", "moment", "epoch", "era", "age"],
            "suffixes": ["flow", "stream", "cycle", "continuity", "progression"]
        },
        MotifCategory.DIMENSIONAL: {
            "prefixes": ["spatial", "volumetric", "expansive", "containing", "vast"],
            "roots": ["space", "dimension", "realm", "domain", "field"],
            "suffixes": ["expanse", "extent", "boundary", "horizon", "frontier"]
        },
        MotifCategory.CONNECTIVE: {
            "prefixes": ["linking", "binding", "joining", "weaving", "connecting"],
            "roots": ["connection", "network", "web", "link", "bond"],
            "suffixes": ["thread", "bridge", "nexus", "junction", "pathway"]
        },
        MotifCategory.SHADOW: {
            "prefixes": ["hidden", "veiled", "obscured", "occluded", "shrouded"],
            "roots": ["shadow", "veil", "mask", "secret", "mystery"],
            "suffixes": ["cloak", "curtain", "shroud", "cover", "fog"]
        },
        MotifCategory.ASCENDANT: {
            "prefixes": ["rising", "ascending", "elevating", "transcending", "surpassing"],
            "roots": ["ascension", "peak", "summit", "zenith", "pinnacle"],
            "suffixes": ["flight", "climb", "journey", "transformation", "evolution"]
        }
    }
    
    components = name_components.get(category, {
        "prefixes": ["mysterious", "unknown", "undefined"],
        "roots": ["pattern", "form", "essence"],
        "suffixes": ["manifestation", "presence", "aspect"]
    })
    
    # Generate a name using the components
    name_structure = random.choice([
        "{prefix}_{root}",
        "{root}_{suffix}",
        "{prefix}_{root}_{suffix}"
    ])
    
    name_parts = {
        "prefix": random.choice(components["prefixes"]),
        "root": random.choice(components["roots"]),
        "suffix": random.choice(components["suffixes"])
    }
    
    return name_structure.format(**name_parts)
from enum import Enum

class MotifCategory(Enum):
    """Categories of symbolic motifs that can be applied to entities"""
    LUMINOUS = "luminous"     # Light, radiance, illumination
    ABYSSAL = "abyssal"       # Depth, void, darkness
    VITAL = "vital"           # Life, growth, adaptation
    ENTROPIC = "entropic"     # Decay, dissolution, transformation
    CRYSTALLINE = "crystalline"  # Order, structure, pattern
    CHAOTIC = "chaotic"       # Disorder, unpredictability, complexity
    ELEMENTAL = "elemental"   # Fundamental forces and substrates
    HARMONIC = "harmonic"     # Resonance, harmony, balance
    RECURSIVE = "recursive"   # Self-reference, fractals, depth
    TEMPORAL = "temporal"     # Time, memory, anticipation
    DIMENSIONAL = "dimensional"  # Space, boundaries, containment
    CONNECTIVE = "connective" # Relationships, networks, communication
    SHADOW = "shadow"         # Hidden aspects, potentials, mysteries
    ASCENDANT = "ascendant"   # Transcendence, evolution, higher order
from enum import Enum

class BreathPhase(Enum):
    """Phases of the cosmic breath cycle"""
    INHALE = "inhale"
    HOLD_IN = "hold_in"
    EXHALE = "exhale"
    HOLD_OUT = "hold_out"

class EventType(Enum):
    """Types of events that can occur in the cosmic scroll"""
    CREATION = "creation"
    DESTRUCTION = "destruction"
    TRANSFORMATION = "transformation"
    INTERACTION = "interaction"
    DISCOVERY = "discovery"
    CONVERGENCE = "convergence"
    DIVERGENCE = "divergence"
    AWAKENING = "awakening"
    DORMANCY = "dormancy"
    EMERGENCE = "emergence"
# ================================================================
#  LOOM ASCENDANT COSMOS — RECURSIVE SYSTEM MODULE
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
#  Integrity Hash (SHA-256): d3ab9688a5a20b8065990cd9b91805e3d892d6e72472f69dd9afe719250c5e37
# ================================================================
"""
CosmicScroll Engine
------------------
Procedural cosmic entity generation and management system for a comprehensive reality simulation.
Designed to integrate with quantum physics, timeline, aether, universe engines and reality kernel.
"""

import random
import math
import uuid
import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# Physical constants from the Planetary Framework
G = 6.67430e-11  # Gravitational constant
C = 299792458    # Speed of light
H = 6.62607015e-34  # Planck constant
ALPHA = 7.2973525693e-3  # Fine-structure constant
OMEGA_LAMBDA = 0.6889  # Dark energy density parameter

class EntityType(Enum):
    """Classification of cosmic entities for the DRM system"""
    UNIVERSE = "universe"
    GALAXY_CLUSTER = "galaxy_cluster"
    GALAXY = "galaxy"
    STAR = "star"
    PLANET = "planet"
    MOON = "moon"
    ASTEROID = "asteroid"
    CIVILIZATION = "civilization"
    ANOMALY = "anomaly"


```
class CosmicScrollManager:
    """
    Central management system for the Loom Ascendant Cosmos engine.
    
    Handles simulation ticks, scroll memory, motif generation, and symbolic narrative progression.
    Acts as the runtime loop for the entire system.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CosmicScrollManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the Cosmic Scroll Manager"""
        self.entities = {}  # entity_id -> entity object
        self.entity_types = defaultdict(set)  # entity_type -> set of entity_ids
        
        self.motif_library = {}  # motif_id -> motif data
        self.entity_motifs = defaultdict(set)  # entity_id -> set of motif_ids
        
        self.event_history = []  # List of all events
        self.recent_events = deque(maxlen=100)  # Recent events for quick access
        
        self.tick_count = 0
        self.time_scale = 1.0  # Time dilation factor
        self.breath_cycle_length = 12  # Ticks per complete breath cycle
        self.breath_phase = BreathPhase.INHALE
        self.breath_progress = 0.0  # 0.0 to 1.0 within current phase
        
        self.inhale_ratio = 0.3    # Proportion of cycle spent inhaling
        self.hold_in_ratio = 0.2   # Proportion of cycle spent holding in
        self.exhale_ratio = 0.3    # Proportion of cycle spent exhaling
        self.hold_out_ratio = 0.2  # Proportion of cycle spent holding out
        
        self.history = {
            "creation_time": datetime.now(),
            "tick_history": [],
            "significant_events": []
        }
        
        self.motif_feedback_queue = deque(maxlen=50)  # Recent motif data for external systems
        
        logger.info("CosmicScrollManager initialized")
    
    def tick(self, delta_time: float = 1.0) -> Dict:
        """
        Advance the simulation forward one step.
        
        Args:
            delta_time: Time multiplier for this tick
            
        Returns:
            Dict containing information about the current tick
        """
        adjusted_delta = delta_time * self.time_scale
        self.tick_count += 1
        
        # Update breath cycle
        self._update_breath_cycle()
        
        # Process entity evolution
        evolved_entities = self._evolve_entities(adjusted_delta)
        
        # Generate events from entity interactions
        generated_events = self._generate_events()
        
        # Process any pending events
        for event in generated_events:
            self.log_event(event)
        
        # Record tick information
        tick_info = {
            "tick_id": self.tick_count,
            "timestamp": datetime.now(),
            "delta_time": adjusted_delta,
            "breath_phase": self.breath_phase.value,
            "breath_progress": self.breath_progress,
            "entities_evolved": len(evolved_entities),
            "events_generated": len(generated_events)
        }
        
        # Store tick history (limiting to last 100 ticks)
        self.history["tick_history"].append(tick_info)
        if len(self.history["tick_history"]) > 100:
            self.history["tick_history"] = self.history["tick_history"][-100:]
            
        logger.debug(f"Tick {self.tick_count} completed")
        return tick_info
    
    def _update_breath_cycle(self):
        """Update the breath cycle phase and progress"""
        total_progress = (self.tick_count % self.breath_cycle_length) / self.breath_cycle_length
        
        # Determine current phase
        if total_progress < self.inhale_ratio:
            self.breath_phase = BreathPhase.INHALE
            self.breath_progress = total_progress / self.inhale_ratio
        elif total_progress < (self.inhale_ratio + self.hold_in_ratio):
            self.breath_phase = BreathPhase.HOLD_IN
            self.breath_progress = (total_progress - self.inhale_ratio) / self.hold_in_ratio
        elif total_progress < (self.inhale_ratio + self.hold_in_ratio + self.exhale_ratio):
            self.breath_phase = BreathPhase.EXHALE
            self.breath_progress = (total_progress - self.inhale_ratio - self.hold_in_ratio) / self.exhale_ratio
        else:
            self.breath_phase = BreathPhase.HOLD_OUT
            self.breath_progress = (total_progress - self.inhale_ratio - self.hold_in_ratio - self.exhale_ratio) / self.hold_out_ratio
    
    def _evolve_entities(self, delta_time: float) -> List[str]:
        """
        Evolve all entities forward in time.
        
        Args:
            delta_time: Time multiplier for this evolution step
            
        Returns:
            List of entity IDs that were evolved
        """
        evolved_entities = []
        
        for entity_id, entity in self.entities.items():
            if hasattr(entity, 'evolve'):
                try:
                    entity.evolve(delta_time)
                    evolved_entities.append(entity_id)
                except Exception as e:
                    logger.error(f"Error evolving entity {entity_id}: {str(e)}")
        
        return evolved_entities
    
    def _generate_events(self) -> List[Dict]:
        """
        Generate events from entity interactions.
        This is a placeholder for more complex event generation logic.
        
        Returns:
            List of generated events
        """
        # This would be implemented with more sophisticated logic
        # that detects meaningful interactions between entities
        events = []
        
        # Simple example: random events for demonstration
        if random.random() < 0.1:  # 10% chance per tick
            # Get random entities for interaction
            if len(self.entities) >= 2:
                entities = random.sample(list(self.entities.keys()), 2)
                
                event = {
                    "type": random.choice(list(EventType)).value,
                    "timestamp": self.tick_count,
                    "entities_involved": entities,
                    "description": f"Random interaction between {entities[0]} and {entities[1]}",
                    "importance": random.uniform(0.1, 1.0)
                }
                
                events.append(event)
        
        return events
    
    def register_entity(self, entity) -> str:
        """
        Register an entity with the Cosmic Scroll system.
        
        Args:
            entity: The entity object to register
            
        Returns:
            The entity ID
        """
        # Ensure entity has an ID
        if not hasattr(entity, 'entity_id'):
            entity.entity_id = f"{entity.__class__.__name__.lower()}_{uuid.uuid4().hex}"
        
        entity_id = entity.entity_id
        
        # Register the entity
        self.entities[entity_id] = entity
        
        # Register by type if available
        if hasattr(entity, 'entity_type'):
            entity_type = entity.entity_type.value if isinstance(entity.entity_type, Enum) else entity.entity_type
            self.entity_types[entity_type].add(entity_id)
        
        # Initialize entity with default motifs if applicable
        if hasattr(self, '_seed_default_motifs'):
            self._seed_default_motifs(entity)
        
        # Generate creation event
        creation_event = {
            "type": EventType.CREATION.value,
            "timestamp": self.tick_count,
            "entities_involved": [entity_id],
            "description": f"Creation of {entity.__class__.__name__}",
            "importance": 0.7  # Creation is a significant event
        }
        
        self.log_event(creation_event)
        
        logger.info(f"Entity registered: {entity_id}")
        return entity_id
    
    def log_event(self, event: Dict) -> str:
        """
        Record an event in the cosmic scroll.
        
        Args:
            event: Dictionary containing event data
            
        Returns:
            Event ID
        """
        # Add event ID if not present
        if "id" not in event:
            event["id"] = f"event_{uuid.uuid4().hex}"
        
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = self.tick_count
        
        # Store the event
        self.event_history.append(event)
        self.recent_events.append(event)
        
        # Check if this is a significant event
        if event.get("importance", 0) > 0.7:
            self.history["significant_events"].append(event)
        
        # Process event for motif generation
        if event.get("importance", 0) > 0.3:  # Only generate motifs for moderately important events
            self.generate_motif(event)
        
        logger.debug(f"Event logged: {event['id']}")
        return event["id"]
    
    def generate_motif(self, event: Dict) -> Optional[str]:
        """
        Generate a symbolic motif from an event.
        
        Args:
            event: Dictionary containing event data
            
        Returns:
            Motif ID if generated, None otherwise
        """
        # Skip if no entities involved
        if not event.get("entities_involved"):
            return None
        
        # Determine a motif category based on event type
        event_type = event.get("type", "")
        
        # Map event types to likely motif categories (simplified)
        category_mapping = {
            EventType.CREATION.value: [MotifCategory.LUMINOUS, MotifCategory.VITAL, MotifCategory.CRYSTALLINE],
            EventType.DESTRUCTION.value: [MotifCategory.ENTROPIC, MotifCategory.ABYSSAL, MotifCategory.CHAOTIC],
            EventType.TRANSFORMATION.value: [MotifCategory.ENTROPIC, MotifCategory.RECURSIVE, MotifCategory.SHADOW],
            EventType.INTERACTION.value: [MotifCategory.CONNECTIVE, MotifCategory.HARMONIC, MotifCategory.DIMENSIONAL],
            EventType.DISCOVERY.value: [MotifCategory.LUMINOUS, MotifCategory.SHADOW, MotifCategory.DIMENSIONAL],
            EventType.CONVERGENCE.value: [MotifCategory.CONNECTIVE, MotifCategory.HARMONIC, MotifCategory.RECURSIVE],
            EventType.DIVERGENCE.value: [MotifCategory.CHAOTIC, MotifCategory.DIMENSIONAL, MotifCategory.TEMPORAL],
            EventType.AWAKENING.value: [MotifCategory.VITAL, MotifCategory.LUMINOUS, MotifCategory.ASCENDANT],
            EventType.DORMANCY.value: [MotifCategory.ABYSSAL, MotifCategory.TEMPORAL, MotifCategory.SHADOW],
            EventType.EMERGENCE.value: [MotifCategory.VITAL, MotifCategory.RECURSIVE, MotifCategory.ASCENDANT]
        }
        
        potential_categories = category_mapping.get(event_type, list(MotifCategory))
        
        # Select a category weighted by event importance
        chosen_category = random.choice(potential_categories)
        
        # Generate a motif name and description
        motif_name = self._generate_motif_name(chosen_category)
        
        # Create the motif
        motif = {
            "id": f"motif_{uuid.uuid4().hex}",
            "name": motif_name,
            "category": chosen_category.value,
            "source_event": event["id"],
            "entities": event["entities_involved"],
            "strength": event.get("importance", 0.5),
            "creation_tick": self.tick_count,
            "resonance": random.uniform(0.3, 0.9),  # How strongly this motif resonates with the cosmic fabric
            "description": f"A {chosen_category.value} motif generated from {event_type}"
        }
        
        # Store the motif
        self.motif_library[motif["id"]] = motif
        
        # Associate motif with entities
        for entity_id in event["entities_involved"]:
            if entity_id in self.entities:
                self.entity_motifs[entity_id].add(motif["id"])
                
                # If entity has a motifs attribute, add it there too
                entity = self.entities[entity_id]
                if hasattr(entity, 'motifs') and isinstance(entity.motifs, list):
                    entity.motifs.append(motif_name)
        
        # Add to feedback queue for external systems
        self.motif_feedback_queue.append(motif)
        
        logger.debug(f"Motif generated: {motif['name']} ({motif['id']})")
        return motif["id"]
    
    def _generate_motif_name(self, category: MotifCategory) -> str:
        """Generate a thematic name for a motif based on its category"""
        # Dictionary of prefix and suffix options for each category
        name_components = {
            MotifCategory.LUMINOUS: {
                "prefixes": ["radiant", "glowing", "shining", "illuminated", "bright"],
                "roots": ["light", "sun",
            cls._instance = super(DimensionalRealityManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the DRM singleton"""
        self.entities = {}  # id -> entity
        self.entity_sector_map = defaultdict(set)  # entity_id -> set of sectors
        self.sector_entity_map = defaultdict(lambda: defaultdict(set))  # sector -> {entity_type: set(entity_ids)}
        self.query_cache = {}  # (query_params) -> results
        self.time_dilation_factor = 1.0
        self.reality_coherence = 1.0
        self.active_observers = set()
        
    def store_entity(self, entity_id: str, entity: Any, sectors: Optional[List[Tuple]] = None):
        """Register an entity in the DRM system"""
        self.entities[entity_id] = entity
        
        if hasattr(entity, 'entity_type'):
            entity_type = entity.entity_type.value
        else:
            entity_type = EntityType.ANOMALY.value
            
        if sectors:
            for sector in sectors:
                self.entity_sector_map[entity_id].add(sector)
                self.sector_entity_map[sector][entity_type].add(entity_id)
    
    def query_entities(self, entity_type: str, sector: Optional[Tuple] = None) -> List[Any]:
        """Retrieve entities matching criteria"""
        cache_key = (entity_type, sector)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
            
        if sector is None:
            # Return all entities of this type
            results = [e for e in self.entities.values() 
                      if hasattr(e, 'entity_type') and e.entity_type.value == entity_type]
        else:
            # Return entities in the specific sector
            entity_ids = self.sector_entity_map[sector].get(entity_type, set())
            results = [self.entities[eid] for eid in entity_ids if eid in self.entities]
            
        self.query_cache[cache_key] = results
        return results
    
    def invalidate_cache(self, sector: Optional[Tuple] = None):
        """Clear query cache for updated sectors"""
        if sector is None:
            self.query_cache.clear()
        else:
            keys_to_remove = [k for k in self.query_cache if k[1] == sector]
            for key in keys_to_remove:
                del self.query_cache[key]
    
    def get_entity(self, entity_id: str) -> Optional[Any]:
        """Retrieve a specific entity by ID"""
        return self.entities.get(entity_id)
    
    def update_entity_sectors(self, entity_id: str, old_sector: Tuple, new_sector: Tuple):
        """Update entity location when it moves between sectors"""
        if entity_id not in self.entities:
            return False
            
        entity = self.entities[entity_id]
        entity_type = entity.entity_type.value if hasattr(entity, 'entity_type') else EntityType.ANOMALY.value
        
        # Remove from old sector
        if old_sector in self.sector_entity_map and entity_type in self.sector_entity_map[old_sector]:
            self.sector_entity_map[old_sector][entity_type].discard(entity_id)
            self.entity_sector_map[entity_id].discard(old_sector)
            
        # Add to new sector
        self.sector_entity_map[new_sector][entity_type].add(entity_id)
        self.entity_sector_map[entity_id].add(new_sector)
        
        # Invalidate affected caches
        self.invalidate_cache(old_sector)
        self.invalidate_cache(new_sector)
        
        return True
    
    def register_observer(self, observer_id: str, position: Tuple):
        """Register an active observer in the simulation"""
        self.active_observers.add(observer_id)
        # Increase simulation fidelity around observer position
        self._adjust_reality_coherence(position)
    
    def _adjust_reality_coherence(self, focal_point: Tuple, radius: int = 3):
        """Adjust reality coherence based on observer proximity"""
        affected_sectors = self._get_sectors_in_radius(focal_point, radius)
        # Implementation details for coherence adjustment would go here
        pass
        
    def _get_sectors_in_radius(self, center: Tuple, radius: int) -> List[Tuple]:
        """Get all sectors within a radius of a central point"""
        sectors = []
        x, y, z = center
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx*dx + dy*dy + dz*dz <= radius*radius:
                        sectors.append((x + dx, y + dy, z + dz))
        return sectors


# Initialize the DRM singleton
DRM = DimensionalRealityManager()


class QuantumSeedGenerator:
    """
    Generates procedural content using quantum-inspired permutations
    Ensures consistent entity generation across multiple simulation runs
    """
    
    _seeds_cache = {}
    
    @staticmethod
    def generate_planet_seed(galaxy_id: str, sector: tuple) -> int:
        """Generate a deterministic seed for a planet"""
        cache_key = f"planet|{galaxy_id}|{sector}"
        if cache_key in QuantumSeedGenerator._seeds_cache:
            return QuantumSeedGenerator._seeds_cache[cache_key]
            
        seed = hash(f"{galaxy_id}|{sector}") % 2**32
        QuantumSeedGenerator._seeds_cache[cache_key] = seed
        return seed
    
    @staticmethod
    def generate_star_seed(galaxy_id: str, sector: tuple, index: int) -> int:
        """Generate a deterministic seed for a star"""
        cache_key = f"star|{galaxy_id}|{sector}|{index}"
        if cache_key in QuantumSeedGenerator._seeds_cache:
            return QuantumSeedGenerator._seeds_cache[cache_key]
            
        seed = hash(f"{galaxy_id}|{sector}|{index}") % 2**32
        QuantumSeedGenerator._seeds_cache[cache_key] = seed
        return seed
    
    @staticmethod
    def generate_galaxy_seed(universe_id: str, position: tuple) -> int:
        """Generate a deterministic seed for a galaxy"""
        cache_key = f"galaxy|{universe_id}|{position}"
        if cache_key in QuantumSeedGenerator._seeds_cache:
            return QuantumSeedGenerator._seeds_cache[cache_key]
            
        seed = hash(f"{universe_id}|{position[0]}|{position[1]}|{position[2]}") % 2**32
        QuantumSeedGenerator._seeds_cache[cache_key] = seed
        return seed
    
    @staticmethod
    def generate_anomaly_seed(sector: tuple, timestamp: float) -> int:
        """Generate a non-deterministic seed for anomalies"""
        # Anomalies are unique events that shouldn't be cached
        return hash(f"{sector}|{timestamp}|{uuid.uuid4()}") % 2**32
    
    @staticmethod
    def clear_cache():
        """Clear the seed cache"""
        QuantumSeedGenerator._seeds_cache.clear()


class CosmicEntity:
    """
    Base class for all entities in the cosmic simulation.
    
    CosmicEntity serves as the foundation for the entity hierarchy, handling:
    - Unique identity and persistence
    - Trait and motif management
    - Reality registration via the DRM system
    - Temporal evolution through the simulation
    - Scroll memory integration for entity history
    
    All entities derive from this class, inheriting its core capabilities while
    implementing their own domain-specific behaviors and properties.
    """
    
    def __init__(self, entity_id: str = None):
        self.entity_id = entity_id or f"{self.__class__.__name__.lower()}_{uuid.uuid4().hex}"
        self.traits = {}
        self.motifs = []
        self.creation_time = 0
        self.last_update_time = 0
        self.sectors = set()
        self.entity_type = EntityType.ANOMALY  # Default
        
    def evolve(self, time_delta: float):
        """Update entity state based on time progression"""
        self.last_update_time += time_delta
        # Base implementation does nothing, subclasses will override
        
    def add_to_reality(self, sectors: List[Tuple] = None):
        """Register entity with the DRM system"""
        if sectors:
            self.sectors.update(sectors)
        DRM.store_entity(self.entity_id, self, list(self.sectors))
        
    def get_trait(self, trait_name: str, default=None):
        """Get a trait value with optional default"""
        return self.traits.get(trait_name, default)
    
    def set_trait(self, trait_name: str, value):
        """Set a trait value"""
        self.traits[trait_name] = value
        
    def has_motif(self, motif_name: str) -> bool:
        """Check if entity has a specific motif"""
        return motif_name in self.motifs


class GalaxyType(Enum):
    """Classification of galaxy morphologies"""
    SPIRAL = "spiral"
    ELLIPTICAL = "elliptical"
    IRREGULAR = "irregular"
    PECULIAR = "peculiar"
    DWARF = "dwarf"


class Galaxy(CosmicEntity):
    """Representation of a galaxy with star systems"""
    
    def __init__(self, galaxy_id: str = None):
        super().__init__(galaxy_id)
        self.entity_type = EntityType.GALAXY
        self.galaxy_type = None
        self.stars = []
        self.age = 0
        self.size = 0
        self.active_regions = set()
        self.metallicity = 0
        self.black_hole_mass = 0
        self.star_formation_rate = 0
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve galaxy over time - affects star formation"""
        super().evolve(time_delta)
        self.age += time_delta
        
        # Only evolve stars in active regions
        if self.active_regions:
            for region in self.active_regions:
                stars = DRM.query_entities(EntityType.STAR.value, region)
                for star in stars:
                    if star.galaxy_id == self.entity_id:
                        star.evolve(time_delta)
                        
    def add_star(self, star):
        """Add a star to this galaxy"""
        star.galaxy_id = self.entity_id
        self.stars.append(star.entity_id)


class StarType(Enum):
    """Stellar classification system"""
    O = "O"  # Hot, massive, blue
    B = "B"  # Blue-white
    A = "A"  # White
    F = "F"  # Yellow-white
    G = "G"  # Yellow (Sun-like)
    K = "K"  # Orange
    M = "M"  # Red dwarf
    L = "L"  # Brown dwarf
    T = "T"  # Methane dwarf
    Y = "Y"  # Ultra-cool brown dwarf
    NEUTRON = "neutron"
    WHITE_DWARF = "white_dwarf"
    BLACK_HOLE = "black_hole"


class Star(CosmicEntity):
    """Stellar object that can host planetary systems"""
    
    def __init__(self, star_id: str = None):
        super().__init__(star_id)
        self.entity_type = EntityType.STAR
        self.star_type = None
        self.planets = []  # List of planet IDs
        self.luminosity = 0
        self.mass = 0
        self.radius = 0
        self.temperature = 0
        self.age = 0
        self.life_expectancy = 0
        self.color = (0, 0, 0)  # RGB
        self.galaxy_id = None
        self.position = (0, 0, 0)  # Position in galaxy
        self.habitable_zone = (0, 0)  # Inner and outer radii
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve star over time - affects luminosity, temperature"""
        super().evolve(time_delta)
        self.age += time_delta
        
        # Calculate evolution based on stellar models
        self._update_stellar_properties()
        
        # Evolve planets
        for planet_id in self.planets:
            planet = DRM.get_entity(planet_id)
            if planet:
                planet.evolve(time_delta)
                
    def _update_stellar_properties(self):
        """Update star properties based on its age and type"""
        # Simplified stellar evolution model
        if self.age > self.life_expectancy:
            self._initiate_stellar_death()
        else:
            # Main sequence evolution - gradual changes
            age_fraction = self.age / self.life_expectancy
            
            # Stars get slightly hotter and more luminous over time
            luminosity_factor = 1 + (0.1 * age_fraction)
            self.luminosity *= luminosity_factor
            
            # Recalculate habitable zone based on luminosity
            self._update_habitable_zone()
    
    def _update_habitable_zone(self):
        """Update the habitable zone based on current luminosity"""
        # Simplified model based on stellar luminosity
        # HZ proportional to square root of luminosity
        luminosity_factor = math.sqrt(self.luminosity)
        self.habitable_zone = (
            0.95 * luminosity_factor,  # Inner boundary
            1.37 * luminosity_factor   # Outer boundary
        )
        
    def _initiate_stellar_death(self):
        """Handle star's end-of-life transition"""
        if self.mass < 0.5:  # Low mass
            # Becomes white dwarf
            self.star_type = StarType.WHITE_DWARF
            self.temperature *= 0.5
            self.radius *= 0.01
            self.luminosity *= 0.001
        elif self.mass < 8:  # Medium mass
            # Also becomes white dwarf after red giant phase
            self.star_type = StarType.WHITE_DWARF
            self.temperature *= 0.7
            self.radius *= 0.01
            self.luminosity *= 0.01
        elif self.mass < 20:  # High mass
            # Becomes neutron star
            self.star_type = StarType.NEUTRON
            self.radius *= 0.0001
            self.temperature *= 10
            self.luminosity *= 0.0001
        else:  # Very high mass
            # Becomes black hole
            self.star_type = StarType.BLACK_HOLE
            self.radius *= 0.00001
            self.luminosity = 0
            
            # Create accretion disk effects
            if random.random() < 0.3:
                self.motifs.append('event_horizon')
                self.motifs.append('hawking_radiation')


class PlanetType(Enum):
    """Classification of planetary bodies"""
    TERRESTRIAL = "terrestrial"
    GAS_GIANT = "gas_giant"
    ICE_GIANT = "ice_giant"
    DWARF = "dwarf"
    SUPER_EARTH = "super_earth"
    HOT_JUPITER = "hot_jupiter"
    OCEAN_WORLD = "ocean_world"
    DESERT_WORLD = "desert_world"
    ICE_WORLD = "ice_world"
    LAVA_WORLD = "lava_world"


class Planet(CosmicEntity):
    """Planetary body that can host life and civilizations"""
    
    def __init__(self, planet_id: str = None):
        super().__init__(planet_id)
        self.entity_type = EntityType.PLANET
        self.planet_type = None
        self.star_id = None
        self.moons = []
        self.mass = 0
        self.radius = 0
        self.density = 0
        self.gravity = 0
        self.orbital_period = 0
        self.rotation_period = 0
        self.axial_tilt = 0
        self.orbital_eccentricity = 0
        self.albedo = 0
        self.atmosphere = {}  # Component -> percentage
        self.surface = {}  # Feature -> percentage
        self.climate = {}
        self.has_magnetosphere = False
        self.temperature = 0
        self.civilizations = []
        self.habitability_index = 0
        self.biosphere_complexity = 0
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve planet over time - affects climate, geology"""
        super().evolve(time_delta)
        
        # Geological processes
        self._evolve_geology(time_delta)
        
        # Climate evolution
        self._evolve_climate(time_delta)
        
        # Biosphere evolution
        if self.biosphere_complexity > 0:
            self._evolve_biosphere(time_delta)
            
        # Civilization evolution
        for civ_id in self.civilizations:
            civ = DRM.get_entity(civ_id)
            if civ:
                civ.evolve(time_delta)
                
    def _evolve_geology(self, time_delta: float):
        """Evolve geological features over time"""
        # Simplified plate tectonics and erosion model
        if self.has_motif('tectonic_dreams') and random.random() < 0.01 * time_delta:
            # Geological event
            event_type = random.choice(['volcanic', 'earthquake', 'erosion'])
            if event_type == 'volcanic':
                self.surface['volcanic'] = self.surface.get('volcanic', 0) + 0.01
                self.atmosphere['sulfur'] = self.atmosphere.get('sulfur', 0) + 0.005
            elif event_type == 'earthquake':
                if 'mountains' in self.surface:
                    self.surface['mountains'] += 0.005
            elif event_type == 'erosion':
                if 'mountains' in self.surface and self.surface['mountains'] > 0.01:
                    self.surface['mountains'] -= 0.005
                    self.surface['sedimentary'] = self.surface.get('sedimentary', 0) + 0.005
    
    def _evolve_climate(self, time_delta: float):
        """Evolve climate patterns over time"""
        # Get host star
        star = DRM.get_entity(self.star_id)
        if not star:
            return
            
        # Update temperature based on star's luminosity and orbital distance
        star_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, star.position)))
        luminosity_factor = star.luminosity / (star_distance ** 2)
        
        # Basic greenhouse effect from atmosphere
        greenhouse_factor = 1.0
        if 'carbon_dioxide' in self.atmosphere:
            greenhouse_factor += self.atmosphere['carbon_dioxide'] * 10
        if 'methane' in self.atmosphere:
            greenhouse_factor += self.atmosphere['methane'] * 30
            
        # Calculate temperature
        base_temp = -270 + (340 * luminosity_factor)  # Simple model in Kelvin
        self.temperature = base_temp * greenhouse_factor - 273.15  # Convert to Celsius
        
        # Update climate zones based on temperature and axial tilt
        self._update_climate_zones()
        
    def _update_climate_zones(self):
        """Update climate zone distribution based on current parameters"""
        # Reset climate zones
        self.climate = {}
        
        # Temperature-based assignments
        if self.temperature < -50:
            self.climate['polar'] = 0.8
            self.climate['tundra'] = 0.2
        elif self.temperature < -10:
            self.climate['polar'] = 0.4
            self.climate['tundra'] = 0.4
            self.climate['taiga'] = 0.2
        elif self.temperature < 5:
            self.climate['tundra'] = 0.3
            self.climate['taiga'] = 0.4
            self.climate['temperate'] = 0.3
        elif self.temperature < 15:
            self.climate['taiga'] = 0.2
            self.climate['temperate'] = 0.6
            self.climate['mediterranean'] = 0.2
        elif self.temperature < 25:
            self.climate['temperate'] = 0.3
            self.climate['mediterranean'] = 0.3
            self.climate['subtropical'] = 0.4
        elif self.temperature < 35:
            self.climate['mediterranean'] = 0.2
            self.climate['subtropical'] = 0.3
            self.climate['tropical'] = 0.3
            self.climate['desert'] = 0.2
        else:
            self.climate['subtropical'] = 0.1
            self.climate['tropical'] = 0.3
            self.climate['desert'] = 0.6
            
        # Adjust for water coverage (if present in surface)
        water_percentage = self.surface.get('water', 0)
        if water_percentage > 0.3:
            # Reduce desert, increase humid zones
            if 'desert' in self.climate:
                desert_reduction = min(self.climate['desert'], water_percentage * 0.3)
                self.climate['desert'] -= desert_reduction
                self.climate['tropical'] = self.climate.get('tropical', 0) + desert_reduction
                
            # Add ocean climate
            self.climate['oceanic'] = water_percentage * 0.7
            
    def _evolve_biosphere(self, time_delta: float):
        """Evolve planetary biosphere if present"""
        # Check habitability
        if self.habitability_index > 0:
            growth_rate = 0.001 * time_delta * self.habitability_index
            
            # Biosphere complexity grows over time
            self.biosphere_complexity = min(1.0, self.biosphere_complexity + growth_rate)
            
            # Chance for civilization emergence
            if (self.biosphere_complexity > 0.8 and 
                'exogenesis' in self.motifs and 
                not self.civilizations and 
                random.random() < 0.001 * time_delta):
                self._spawn_civilization()
                
    def _spawn_civilization(self):
        """Generate a new civilization on this planet"""
        civ = Civilization()
        civ.planet_id = self.entity_id
        civ.home_sector = list(self.sectors)[0] if self.sectors else None
        
        # Set initial properties based on planet type
        civ.traits['adaptations'] = []
        
        # Adapt to planet conditions
        if self.temperature < -10:
            civ.traits['adaptations'].append('cold_resistance')
        if self.temperature > 30:
            civ.traits['adaptations'].append('heat_resistance')
        if self.gravity > 1.3:
            civ.traits['adaptations'].append('high_gravity')
        
        # Set initial development level
        civ.development_level = 0.1  # Stone age
        
        # Register with DRM
        civ.add_to_reality(list(self.sectors))
        
        # Add to planet's civilization list
        self.civilizations.append(civ.entity_id)


class CivilizationType(Enum):
    """Classification of civilization types"""
    TYPE_0 = "pre_industrial"
    TYPE_1 = "planetary"
    TYPE_2 = "stellar"
    TYPE_3 = "galactic"
    TYPE_4 = "cosmic"
    
    
class DevelopmentArea(Enum):
    """Areas of technological development"""
    ENERGY = "energy"
    COMPUTATION = "computation"
    MATERIALS = "materials"
    BIOLOGY = "biology"
    SPACE_TRAVEL = "space_travel"
    WEAPONRY = "weaponry"
    COMMUNICATION = "communication"
    SOCIAL_ORGANIZATION = "social_organization"


class Civilization(CosmicEntity):
    """An intelligent species and its technological development"""
    
    def __init__(self, civ_id: str = None):
        super().__init__(civ_id)
        self.entity_type = EntityType.CIVILIZATION
        self.planet_id = None
        self.civ_type = CivilizationType.TYPE_0
        self.development_level = 0  # 0 to 1 within its type
        self.population = 0
        self.tech_focus = None
        self.tech_levels = {area: 0 for area in DevelopmentArea}
        self.colonized_planets = []
        self.home_sector = None
        self.communication_range = 0
        self.ftl_capability = False
        self.quantum_understanding = 0
        self.known_civilizations = []
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve civilization over time"""
        super().evolve(time_delta)
        
        # Development rate decreases as civilization advances
        # This creates a logistic growth curve
        dev_rate = 0.01 * time_delta * (1 - self.development_level)
        self.development_level = min(1.0, self.development_level + dev_rate)
        
        # Check for civ type transition
        self._check_advancement()
        
        # Expand population
        self._evolve_population(time_delta)
        
        # Advance technology
        self._evolve_technology(time_delta)
        
        # Check for expansion to other planets
        self._check_expansion()
        
        # Check for contact with other civilizations
        if random.random() < 0.01 * time_delta:
            self._check_contact()
            
    def _check_advancement(self):
        """Check if civilization should advance to next type"""
        if (self.development_level >= 0.99):
            if self.civ_type == CivilizationType.TYPE_0:
                self.civ_type = CivilizationType.TYPE_1
                self.development_level = 0.1
                self.motifs.append('industrial_revolution')
            elif self.civ_type == CivilizationType.TYPE_1:
                self.civ_type = CivilizationType.TYPE_2
                self.development_level = 0.1
                self.motifs.append('stellar_expansion')
                self.ftl_capability = True
            elif self.civ_type == CivilizationType.TYPE_2:
                self.civ_type = CivilizationType.TYPE_3
                self.development_level = 0.1
                self.motifs.append('galactic_network')
            # Type 4 is theoretical and extremely rare
    
    def _evolve_population(self, time_delta: float):
        """Evolve population size and distribution"""
        if self.population == 0:
            # Initialize population for new civilization
            self.population = 1000
        else:
            # Logistic growth model
            growth_rate = 0.02 * time_delta
            carrying_capacity = 10**10 * (1 + len(self.colonized_planets))
            
            growth = growth_rate * self.population * (1 - self.population / carrying_capacity)
            self.population = max(1000, int(self.population + growth))
    
    def _evolve_technology(self, time_delta: float):
        """Advance technology levels"""
        # Choose focus if none exists
        if not self.tech_focus:
            self.tech_focus = random.choice(list(DevelopmentArea))
            
        # Advance all tech areas with focus getting bonus
        for area in DevelopmentArea:
            advance_rate = 0.005 * time_delta
            
            # Focus area advances faster
            if area == self.tech_focus:
                advance_rate *= 2
                
            # Higher civilization types advance faster
            advance_rate *= (1 + 0.5 * self.civ_type.value.count('_'))
            
            self.tech_levels[area] = min(1.0, self.tech_levels[area] + advance_rate)
            
        # Occasionally change focus (every ~100 time units)
        if random.random() < 0.01 * time_delta:
            self.tech_focus = random.choice(list(DevelopmentArea))
            
        # Update FTL capability
        if (self.tech_levels[DevelopmentArea.SPACE_TRAVEL] > 0.7 and
            self.tech_levels[DevelopmentArea.ENERGY] > 0.8):
            self.ftl_capability = True
            
        # Update communication range
        self.communication_range = 10 * self.tech_levels[DevelopmentArea.COMMUNICATION]
        if self.ftl_capability:
            self.communication_range *= 100
            
        # Update quantum understanding
        self.quantum_understanding = (
            self.tech_levels[DevelopmentArea.COMPUTATION] * 0.5 +
            self.tech_levels[DevelopmentArea.ENERGY] * 0.3 +
            self.tech_levels[DevelopmentArea.MATERIALS] * 0.2
        )
    
    def _check_expansion(self):
        """Check if civilization should expand to new planets"""
        # Only Type 1+ civs with space travel can expand
        if (self.civ_type.value != CivilizationType.TYPE_0.value and 
            self.tech_levels[DevelopmentArea.SPACE_TRAVEL] > 0.5 and
            random.random() < 0.05 * self.tech_levels[DevelopmentArea.SPACE_TRAVEL]):
            
            # Get home planet
            home_planet = DRM.get_entity(self.planet_id)
            if not home_planet:
                return
                
            # Get star of home planet
            star = DRM.get_entity(home_planet.star_id)
            if not star:
                return
                
            # Search for habitable planets in the star system first
            habitable_candidates = []
            for planet_id in star.planets:
                planet = DRM.get_entity(planet_id)
                if (planet and 
                    planet.entity_id != self.planet_id and
                    planet.entity_id not in self.colonized_planets and
                    planet.habitability_index > 0.3):
                    habitable_candidates.append(planet)
                    
            # If no candidates in home system and civilization has FTL,
            # search nearby star systems
            if not habitable_candidates and self.ftl_capability:
                nearby_stars = self._find_nearby_stars(star, 10)  # 10 light year radius
                
                for nearby_star in nearby_stars:
                    for planet_id in nearby_star.planets:
                        planet = DRM.get_entity(planet_id)
                        if (planet and 
                            planet.habitability_index > 0.4 and  # Higher threshold for interstellar travel
                            planet.entity_id not in self.colonized_planets):
                            habitable_candidates.append(planet)
            
            # Colonize a random candidate
            if habitable_candidates:
                target_planet = random.choice(habitable_candidates)
                
                # Add to colonized planets
                self.colonized_planets.append(target_planet.entity_id)
                
                # Add civilization to planet
                if self.entity_id not in target_planet.civilizations:
                    target_planet.civilizations.append(self.entity_id)
                    
                # Update sectors covered by civilization
                for sector in target_planet.sectors:
                    if sector not in self.sectors:
                        self.sectors.add(sector)
                        
                # Update DRM
                DRM.store_entity(self.entity_id, self, list(self.sectors))
                
                # Add colonization motif
                self.motifs.append('interplanetary_colonization' if target_planet.star_id == star.entity_id 
                                  else 'interstellar_colonization')
    
    def _find_nearby_stars(self, reference_star, max_distance: float) -> List[Star]:
        """Find stars within max_distance light years of reference_star"""
        nearby_stars = []
        ref_pos = reference_star.position
        
        # Get galaxy
        galaxy = DRM.get_entity(reference_star.galaxy_id)
        if not galaxy:
            return nearby_stars
            
        # Query all stars in galaxy
        all_stars = DRM.query_entities(EntityType.STAR.value)
        
        for star in all_stars:
            if star.entity_id == reference_star.entity_id:
                continue
                
            if star.galaxy_id != reference_star.galaxy_id:
                continue
                
            # Calculate distance
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(ref_pos, star.position)))
            
            if distance <= max_distance:
                nearby_stars.append(star)
                
        return nearby_stars
    
    def _check_contact(self):
        """Check for contact with other civilizations"""
        # Get all civilizations
        all_civs = DRM.query_entities(EntityType.CIVILIZATION.value)
        
        for civ in all_civs:
            if civ.entity_id == self.entity_id:
                continue
                
            if civ.entity_id in self.known_civilizations:
                continue
                
            # Check if in communication range
            home_planet = DRM.get_entity(self.planet_id)
            other_home_planet = DRM.get_entity(civ.planet_id)
            
            if not home_planet or not other_home_planet:
                continue
                
            # Get stars
            home_star = DRM.get_entity(home_planet.star_id)
            other_star = DRM.get_entity(other_home_planet.star_id)
            
            if not home_star or not other_star:
                continue
                
            # Calculate distance between stars
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(home_star.position, other_star.position)))
            
            # Check if in range
            civ_range = self.communication_range + civ.communication_range
            
            if distance <= civ_range:
                # Contact established
                self.known_civilizations.append(civ.entity_id)
                civ.known_civilizations.append(self.entity_id)
                
                # Add motif
                contact_type = 'first_contact' if not self.known_civilizations else 'alien_contact'
                self.motifs.append(contact_type)
                civ.motifs.append(contact_type)


class AnomalyType(Enum):
    """Classification of cosmic anomalies"""
    WORMHOLE = "wormhole"
    BLACK_HOLE = "black_hole"
    NEUTRON_STAR = "neutron_star"
    DARK_MATTER_CLOUD = "dark_matter_cloud"
    QUANTUM_FLUCTUATION = "quantum_fluctuation"
    TIME_DILATION = "time_dilation"
    DIMENSIONAL_RIFT = "dimensional_rift"
    COSMIC_STRING = "cosmic_string"
    STRANGE_MATTER = "strange_matter"
    REALITY_BUBBLE = "reality_bubble"


class Anomaly(CosmicEntity):
    """Rare cosmic phenomena with unique properties"""
    
    def __init__(self, anomaly_id: str = None):
        super().__init__(anomaly_id)
        self.entity_type = EntityType.ANOMALY
        self.anomaly_type = None
        self.intensity = 0
        self.stability = 0
        self.radius = 0
        self.effects = []
        self.danger_level = 0
        self.is_traversable = False
        self.connects_to = None  # For wormholes, dimensional rifts
        self.discovery_chance = 0.1
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve anomaly over time"""
        super().evolve(time_delta)
        
        # Stability changes over time
        stability_change = (random.random() - 0.5) * 0.1 * time_delta
        self.stability = max(0, min(1, self.stability + stability_change))
        
        # Check for collapse if unstable
        if self.stability < 0.2 and random.random() < 0.01 * time_delta:
            self._trigger_collapse()
            
        # Intensity fluctuations
        intensity_change = (random.random() - 0.5) * 0.05 * time_delta
        self.intensity = max(0.1, min(1, self.intensity + intensity_change))
        
        # Effect on nearby entities
        self._apply_proximity_effects()
        
    def _trigger_collapse(self):
        """Handle anomaly collapse or transformation"""
        if self.anomaly_type == AnomalyType.WORMHOLE:
            # Wormhole collapse can create a black hole
            if random.random() < 0.3:
                self.anomaly_type = AnomalyType.BLACK_HOLE
                self.intensity *= 2
                self.stability = 0.7
                self.is_traversable = False
                self.connects_to = None
                self.motifs.append('wormhole_collapse')
            else:
                # Complete dissipation
                self._mark_for_removal()
                
        elif self.anomaly_type == AnomalyType.DIMENSIONAL_RIFT:
            # Rift collapse can cause quantum side effects
            self.anomaly_type = AnomalyType.QUANTUM_FLUCTUATION
            self.intensity *= 0.5
            self.stability = 0.4
            self.motifs.append('dimensional_collapse')
            
        elif self.anomaly_type == AnomalyType.REALITY_BUBBLE:
            # Reality bubble collapse can be catastrophic
            self._create_collapse_shockwave()
            self._mark_for_removal()
            
        else:
            # Generic dissipation
            self._mark_for_removal()
            
    def _mark_for_removal(self):
        """Mark anomaly for removal from simulation"""
        # This would be implemented by the reality engine
        # For now we just add a motif to indicate pending removal
        self.motifs.append('marked_for_removal')
        
    def _create_collapse_shockwave(self):
        """Create a shockwave effect when anomaly collapses"""
        # This would normally spawn a temporary effect entity
        # For now we just add the motif
        self.motifs.append('collapse_shockwave')
        
    def _apply_proximity_effects(self):
        """Apply effects to entities near the anomaly"""
        # Get nearby entities
        nearby_entities = {}
        
        for sector in self.sectors:
            for entity_type in EntityType:
                entities = DRM.query_entities(entity_type.value, sector)
                
                for entity in entities:
                    if entity.entity_id != self.entity_id:
                        nearby_entities[entity.entity_id] = entity
        
        # Apply effects based on anomaly type
        for entity in nearby_entities.values():
            self._apply_anomaly_effect(entity)
            
    def _apply_anomaly_effect(self, entity: CosmicEntity):
        """Apply type-specific anomaly effect to an entity"""
        effect_strength = self.intensity * 0.2  # Base effect strength
        
        if self.anomaly_type == AnomalyType.TIME_DILATION:
            # Time flows differently near this anomaly
            if hasattr(entity, 'time_dilation_factor'):
                entity.time_dilation_factor = 1 + (effect_strength * (random.random() - 0.5) * 2)
                
        elif self.anomaly_type == AnomalyType.DARK_MATTER_CLOUD:
            # Affects gravity and energy calculations
            if hasattr(entity, 'mass'):
                entity.effective_mass = entity.mass * (1 + effect_strength)
                
        elif self.anomaly_type == AnomalyType.QUANTUM_FLUCTUATION:
            # Can cause unpredictable behavior or even mutations
            if random.random() < 0.1 * effect_strength:
                if 'quantum_affected' not in entity.motifs:
                    entity.motifs.append('quantum_affected')
                    
                    # Special effect on civilizations: boost quantum understanding
                    if entity.entity_type == EntityType.CIVILIZATION:
                        if hasattr(entity, 'quantum_understanding'):
                            entity.quantum_understanding = min(1.0, entity.quantum_understanding + 0.1)
                
        elif self.anomaly_type == AnomalyType.DIMENSIONAL_RIFT:
            # Can connect to other parts of space or even realities
            if random.random() < 0.05 * effect_strength:
                if 'dimensional_exposure' not in entity.motifs:
                    entity.motifs.append('dimensional_exposure')


class Universe(CosmicEntity):
    """Top-level container for all cosmic entities"""
    
    def __init__(self, universe_id: str = None):
        super().__init__(universe_id)
        self.entity_type = EntityType.UNIVERSE
        self.galaxies = []
        self.age = 0
        self.expansion_rate = 67.8  # Hubble constant
        self.dark_energy_density = OMEGA_LAMBDA
        self.dark_matter_density = 0.2589
        self.baryonic_matter_density = 0.0486
        self.fundamental_constants = {
            'G': G,
            'c': C,
            'h': H,
            'alpha': ALPHA
        }
        self.dimensions = 3
        self.active_sectors = set()
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve universe over time"""
        super().evolve(time_delta)
        self.age += time_delta
        
        # Universe expansion affects distances between galaxies
        self._apply_cosmic_expansion(time_delta)
        
        # Only evolve active sectors to save computational resources
        self._evolve_active_sectors(time_delta)
        
        # Occasionally spawn anomalies
        if random.random() < 0.01 * time_delta:
            self._spawn_anomaly()
            
    def _apply_cosmic_expansion(self, time_delta: float):
        """Apply cosmic expansion to galaxy positions"""
        # Simplified model: expansion increases with distance from origin
        expansion_factor = self.expansion_rate * time_delta / 1000
        
        # Apply to galaxies
        for galaxy_id in self.galaxies:
            galaxy = DRM.get_entity(galaxy_id)
            if galaxy:
                # Scale positions outward from origin
                old_pos = galaxy.position
                new_pos = tuple(p * (1 + expansion_factor * abs(p)) for p in old_pos)
                galaxy.position = new_pos
                
                # Update sectors if needed
                old_sector = tuple(int(p // 100) for p in old_pos)
                new_sector = tuple(int(p // 100) for p in new_pos)
                
                if old_sector != new_sector:
                    DRM.update_entity_sectors(galaxy.entity_id, old_sector, new_sector)
    
    def _evolve_active_sectors(self, time_delta: float):
        """Evolve only the active sectors in the universe"""
        for sector in self.active_sectors:
            # Evolve galaxies in this sector
            galaxies = DRM.query_entities(EntityType.GALAXY.value, sector)
            for galaxy in galaxies:
                galaxy.evolve(time_delta)
                
            # Evolve anomalies in this sector
            anomalies = DRM.query_entities(EntityType.ANOMALY.value, sector)
            for anomaly in anomalies:
                anomaly.evolve(time_delta)
                
    def _spawn_anomaly(self):
        """Randomly spawn a new anomaly in the universe"""
        # Select a random active sector
        if not self.active_sectors:
            return
            
        sector = random.choice(list(self.active_sectors))
        
        # Create a new anomaly
        anomaly = Anomaly()
        anomaly.anomaly_type = random.choice(list(AnomalyType))
        anomaly.intensity = random.uniform(0.3, 1.0)
        anomaly.stability = random.uniform(0.5, 1.0)
        anomaly.radius = random.uniform(0.1, 10.0)  # Light years
        
        # Set danger level based on type and intensity
        if anomaly.anomaly_type in [AnomalyType.BLACK_HOLE, AnomalyType.STRANGE_MATTER]:
            anomaly.danger_level = 0.7 + (0.3 * anomaly.intensity)
        elif anomaly.anomaly_type in [AnomalyType.WORMHOLE, AnomalyType.DIMENSIONAL_RIFT]:
            anomaly.danger_level = 0.5 + (0.4 * anomaly.intensity)
        else:
            anomaly.danger_level = 0.2 + (0.3 * anomaly.intensity)
            
        # Determine if traversable (for wormholes and rifts)
        if anomaly.anomaly_type in [AnomalyType.WORMHOLE, AnomalyType.DIMENSIONAL_RIFT]:
            anomaly.is_traversable = random.random() < 0.3
            
            if anomaly.is_traversable:
                # Connect to another sector
                connected_sector = self._find_connection_target(sector)
                anomaly.connects_to = connected_sector
                
        # Add to reality
        anomaly.sectors.add(sector)
        anomaly.add_to_reality([sector])
        
    def _find_connection_target(self, source_sector: Tuple) -> Tuple:
        """Find a valid target sector for wormhole/rift connections"""
        # Usually connect to a distant part of the same universe
        x, y, z = source_sector
        
        # Random offset at least 10 sectors away
        min_distance = 10
        while True:
            dx = random.randint(-100, 100)
            dy = random.randint(-100, 100)
            dz = random.randint(-100, 100)
            
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            if distance >= min_distance:
                break
                
        return (x + dx, y + dy, z + dz)
    
    def add_galaxy(self, galaxy: Galaxy):
        """Add a galaxy to this universe"""
        galaxy.universe_id = self.entity_id
        self.galaxies.append(galaxy.entity_id)


class GalaxyCluster(CosmicEntity):
    """A collection of gravitationally bound galaxies"""
    
    def __init__(self, cluster_id: str = None):
        super().__init__(cluster_id)
        self.entity_type = EntityType.GALAXY_CLUSTER
        self.galaxies = []
        self.size = 0  # Megaparsecs
        self.mass = 0  # Solar masses
        self.dark_matter_ratio = 0.85
        self.center_position = (0, 0, 0)
        self.universe_id = None
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve galaxy cluster over time"""
        super().evolve(time_delta)
        
        # Gravitational effects on member galaxies
        self._update_galaxy_positions(time_delta)
        
        # Merger chance
        self._check_galaxy_mergers()
        
    def _update_galaxy_positions(self, time_delta: float):
        """Update positions of galaxies based on gravity"""
        # Simplified model: galaxies orbit around cluster center
        # and sometimes move closer or further based on interactions
        
        for galaxy_id in self.galaxies:
            galaxy = DRM.get_entity(galaxy_id)
            if not galaxy:
                continue
                
            # Calculate vector from center to galaxy
            galaxy_pos = galaxy.position
            center = self.center_position
            
            # Vector from center to galaxy
            offset = tuple(g - c for g, c in zip(galaxy_pos, center))
            distance = math.sqrt(sum(o**2 for o in offset))
            
            if distance == 0:
                continue  # At center, no movement
                
            # Normalized direction vector
            direction = tuple(o / distance for o in offset)
            
            # Orbital velocity (decreases with distance)
            orbit_speed = math.sqrt(G * self.mass / distance) * time_delta / 1000
            
            # Perpendicular vector for orbit (simplified to 2D orbit in xy plane)
            orbit_vector = (-direction[1], direction[0], 0)
            
            # Random perturbation
            perturbation = tuple(random.uniform(-0.1, 0.1) for _ in range(3))
            
            # Update position: apply orbital motion and small random shift
            new_pos = tuple(
                g + (orbit_vector[i] * orbit_speed) + (perturbation[i] * orbit_speed / 10)
                for i, g in enumerate(galaxy_pos)
            )
            
            # Update galaxy position
            galaxy.position = new_pos
            
            # Update sectors if needed
            old_sector = tuple(int(p // 100) for p in galaxy_pos)
            new_sector = tuple(int(p // 100) for p in new_pos)
            
            if old_sector != new_sector:
                DRM.update_entity_sectors(galaxy.entity_id, old_sector, new_sector)
    
    def _check_galaxy_mergers(self):
        """Check for and handle galaxy mergers"""
        # This would be a complex calculation in reality
        # Simplified version: check for galaxies that are very close to each other
        
        galaxy_objects = []
        for galaxy_id in self.galaxies:
            galaxy = DRM.get_entity(galaxy_id)
            if galaxy:
                galaxy_objects.append(galaxy)
                
        # Check all pairs of galaxies
        for i, galaxy1 in enumerate(galaxy_objects):
            for galaxy2 in galaxy_objects[i+1:]:
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(galaxy1.position, galaxy2.position)))
                
                # If galaxies are very close, consider a merger
                merger_threshold = (galaxy1.size + galaxy2.size) / 2
                if distance < merger_threshold * 0.8:
                    self._merge_galaxies(galaxy1, galaxy2)
                    break  # Only handle one merger at a time
                    
    def _merge_galaxies(self, galaxy1: Galaxy, galaxy2: Galaxy):
        """Merge two galaxies together"""
        # Create a new merged galaxy
        merged_galaxy = Galaxy()
        
        # Combine properties
        merged_galaxy.galaxy_type = GalaxyType.PECULIAR  # Mergers produce peculiar galaxies
        merged_galaxy.size = galaxy1.size + galaxy2.size
        merged_galaxy.metallicity = (galaxy1.metallicity + galaxy2.metallicity) / 2
        merged_galaxy.black_hole_mass = galaxy1.black_hole_mass + galaxy2.black_hole_mass
        
        # Position at center of mass
        total_mass = (galaxy1.size + galaxy1.black_hole_mass) + (galaxy2.size + galaxy2.black_hole_mass)
        mass1 = galaxy1.size + galaxy1.black_hole_mass
        mass2 = galaxy2.size + galaxy2.black_hole_mass
        
        merged_galaxy.position = tuple(
            (p1 * mass1 + p2 * mass2) / total_mass
            for p1, p2 in zip(galaxy1.position, galaxy2.position)
        )
        
        # Inherit stars from both galaxies
        merged_galaxy.stars = galaxy1.stars + galaxy2.stars
        
        # Update star's galaxy_id
        for star_id in merged_galaxy.stars:
            star = DRM.get_entity(star_id)
            if star:
                star.galaxy_id = merged_galaxy.entity_id
                
        # Add merger motif
        merged_galaxy.motifs.append('galaxy_merger')
        
        # Register with DRM
        sector = tuple(int(p // 100) for p in merged_galaxy.position)
        merged_galaxy.sectors.add(sector)
        merged_galaxy.add_to_reality([sector])
        
        # Add to cluster
        self.galaxies.append(merged_galaxy.entity_id)
        
        # Remove old galaxies from cluster
        self.galaxies.remove(galaxy1.entity_id)
        self.galaxies.remove(galaxy2.entity_id)
        
        # Mark old galaxies for removal
        galaxy1.motifs.append('marked_for_removal')
        galaxy2.motifs.append('marked_for_removal')


class Moon(CosmicEntity):
    """Natural satellite orbiting a planet"""
    
    def __init__(self, moon_id: str = None):
        super().__init__(moon_id)
        self.entity_type = EntityType.MOON
        self.planet_id = None
        self.radius = 0
        self.mass = 0
        self.orbital_distance = 0
        self.orbital_period = 0
        self.rotation_period = 0
        self.surface = {}
        self.habitability_index = 0
        self.has_atmosphere = False
        self.atmosphere = {}
        self.temperature = 0
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve moon over time"""
        super().evolve(time_delta)
        
        # Get parent planet
        planet = DRM.get_entity(self.planet_id)
        if not planet:
            return
            
        # Geological processes
        if random.random() < 0.005 * time_delta:
            self._evolve_geology()
            
        # Tidal effects
        self._apply_tidal_effects(planet, time_delta)
        
    def _evolve_geology(self):
        """Evolve geological features"""
        # Chance for geological activity
        if random.random() < 0.2:
            # Add or modify surface features
            feature = random.choice(['craters', 'plains', 'mountains', 'canyons', 'volcanic'])
            self.surface[feature] = self.surface.get(feature, 0) + random.uniform(0.05, 0.1)
            
            # Normalize percentages
            total = sum(self.surface.values())
            if total > 1:
                for key in self.surface:
                    self.surface[key] /= total
                    
    def _apply_tidal_effects(self, planet: Planet, time_delta: float):
        """Apply tidal effects between moon and planet"""
        # Tidal locking - rotation period gradually matches orbital period
        if self.rotation_period != self.orbital_period:
            adjustment = (self.orbital_period - self.rotation_period) * 0.001 * time_delta
            self.rotation_period += adjustment
            
        # Tidal forces can cause heating in moon's core
        if self.orbital_distance < planet.radius * 5:
            # Close orbits cause stronger tidal heating
            heating = 0.1 * time_delta * (planet.radius * 5 / self.orbital_distance)
            
            # Add volcanic activity if heating is significant
            if heating > 0.05 and random.random() < heating:
                self.surface['volcanic'] = self.surface.get('volcanic', 0) + 0.05
                
                # Normalize percentages
                total = sum(self.surface.values())
                if total > 1:
                    for key in self.surface:
                        self.surface[key] /= total


class Asteroid(CosmicEntity):
    """Small rocky body in space"""
    
    def __init__(self, asteroid_id: str = None):
        super().__init__(asteroid_id)
        self.entity_type = EntityType.ASTEROID
        self.size = 0
        self.mass = 0
        self.composition = {}  # material -> percentage
        self.orbit = None  # Star ID or planet ID
        self.orbital_distance = 0
        self.orbital_period = 0
        self.is_hazardous = False
        self.trajectory = []  # List of positions
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve asteroid over time"""
        super().evolve(time_delta)
        
        # Update position based on orbital parameters
        self._update_position(time_delta)
        
        # Check for collisions
        self._check_collisions()
        
    def _update_position(self, time_delta: float):
        """Update asteroid position based on orbit"""
        # Get the object being orbited
        orbit_center = DRM.get_entity(self.orbit)
        if not orbit_center:
            return
            
        # Calculate new position
        old_pos = self.position
        
        # Simple circular orbit for now
        orbit_angle = (2 * math.pi / self.orbital_period) * time_delta
        
        # Rotation matrix for orbit_angle around z-axis
        cos_theta = math.cos(orbit_angle)
        sin_theta = math.sin(orbit_angle)
        
        # Vector from center to asteroid
        offset = tuple(a - b for a, b in zip(old_pos, orbit_center.position))
        
        # Apply rotation
        new_offset = (
            offset[0] * cos_theta - offset[1] * sin_theta,
            offset[0] * sin_theta + offset[1] * cos_theta,
            offset[2]
        )
        
        # New position
        new_pos = tuple(c + o for c, o in zip(orbit_center.position, new_offset))
        self.position = new_pos
        
        # Record trajectory point
        self.trajectory.append(new_pos)
        if len(self.trajectory) > 10:
            self.trajectory.pop(0)
            
        # Update sectors if needed
        old_sector = tuple(int(p // 100) for p in old_pos)
        new_sector = tuple(int(p // 100) for p in new_pos)
        
        if old_sector != new_sector:
            DRM.update_entity_sectors(self.entity_id, old_sector, new_sector)
    
    def _check_collisions(self):
        """Check for collisions with planets"""
        if not self.is_hazardous:
            return
            
        # Get planets in current sector
        sector = tuple(int(p // 100) for p in self.position)
        planets = DRM.query_entities(EntityType.PLANET.value, sector)
        
        for planet in planets:
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, planet.position)))
            
            # If asteroid is closer than planet radius, collision occurs
            if distance < planet.radius:
                self._handle_collision(planet)
                break
                
    def _handle_collision(self, planet: Planet):
        """Handle asteroid collision with a planet"""
        # Impact effects depend on asteroid size relative to planet
        size_ratio = self.size / planet.radius
        
        if size_ratio < 0.001:  # Tiny asteroid
            # Minimal impact, maybe a small crater
            if 'craters' in planet.surface:
                planet.surface['craters'] += 0.001
            else:
                planet.surface['craters'] = 0.001
            planet.motifs.append('minor_impact')
                
        elif size_ratio < 0.01:  # Small asteroid
            # Notable impact, might affect climate briefly
            if 'craters' in planet.surface:
                planet.surface['craters'] += 0.01
            else:
                planet.surface['craters'] = 0.01
            planet.motifs.append('moderate_impact')
            # Slight climate change
            planet.temperature += random.uniform(-1, 1)
            
        elif size_ratio < 0.1:  # Medium asteroid
            # Significant impact, climate effects
            planet.surface['craters'] = planet.surface.get('craters', 0) + 0.05
            planet.motifs.append('major_impact')
            planet.temperature += random.uniform(-5, 5)
            # Possible mass extinction
            if planet.biosphere_complexity > 0.5:
                planet.biosphere_complexity *= 0.8
                planet.motifs.append('mass_extinction')
                
        else:  # Large asteroid
            # Catastrophic impact
            planet.surface['craters'] = planet.surface.get('craters', 0) + 0.2
            planet.motifs.append('cataclysmic_impact')
            planet.temperature += random.uniform(-10, 10)
            # Drastic reduction in biosphere and civilizations
            planet.biosphere_complexity *= 0.2
            for civ_id in planet.civilizations:
                civ = DRM.get_entity(civ_id)
                if civ:
                    civ.population = max(0, int(civ.population * 0.1))
                    civ.development_level = max(0, civ.development_level - 0.2)
        
        # Mark asteroid for removal
        self.motifs.append('marked_for_removal')


# -------------------------------------------------------------------------
# MotifSeeder System
# -------------------------------------------------------------------------
class MotifCategory(Enum):
    """Categories of symbolic motifs that can be applied to entities"""
    PRIMORDIAL = "primordial"  # Ancient/original patterns
    ELEMENTAL = "elemental"    # Related to fundamental forces/elements
    HARMONIC = "harmonic"      # Pattern relationships and resonance
    CHAOTIC = "chaotic"        # Disorder and unpredictability
    LUMINOUS = "luminous"      # Light, radiation, visibility
    SHADOW = "shadow"          # Darkness, obscurity, mystery
    RECURSIVE = "recursive"    # Self-referential patterns
    ASCENDANT = "ascendant"    # Evolution, growth, transcendence
    DIMENSIONAL = "dimensional"  # Spatial and dimensional properties
    TEMPORAL = "temporal"      # Time-related patterns
    VITAL = "vital"            # Life and consciousness
    ENTROPIC = "entropic"      # Decay, dissolution, heat death
    CRYSTALLINE = "crystalline"  # Order, structure, lattice
    ABYSSAL = "abyssal"        # Depth, void, emptiness


class MotifSeeder:
    """
    Symbolic initialization system that assigns thematic motifs to cosmic entities.
    Creates meaningful patterns and relationships between entities.
    """
    
    # Motif pools for different entity types
    _motif_pools = {
        EntityType.UNIVERSE: {
            MotifCategory.PRIMORDIAL: ["first_breath", "genesis_point", "all_potential", "undivided_unity", "original_void"],
            MotifCategory.DIMENSIONAL: ["hyperbolic_manifold", "dimensional_flux", "space_weave", "boundary_condition"],
            MotifCategory.RECURSIVE: ["fractal_seed", "nested_reality", "self_similar_pattern", "recursive_boundary"],
            MotifCategory.TEMPORAL: ["timestream_source", "eternal_now", "temporal_ocean", "omega_point"],
        },
        EntityType.GALAXY: {
            MotifCategory.LUMINOUS: ["starfire_spiral", "radiant_halo", "stellar_tapestry", "cosmic_lighthouse"],
            MotifCategory.CHAOTIC: ["stellar_tempest", "void_turbulence", "dark_flow", "random_scatter"],
            MotifCategory.CRYSTALLINE: ["stellar_lattice", "harmonic_arrangement", "symmetric_pattern"],
            MotifCategory.ENTROPIC: ["heat_death_advance", "dispersal_pattern", "expansion_drift"],
        },
        EntityType.STAR: {
            MotifCategory.ELEMENTAL: ["plasma_heart", "fusion_crucible", "energy_fountain", "elemental_forge"],
            MotifCategory.LUMINOUS: ["light_bearer", "radiation_pulse", "constant_dawn", "warmth_giver"],
            MotifCategory.VITAL: ["life_catalyst", "habitable_anchor", "biosphere_enabler"],
            MotifCategory.ENTROPIC: ["stellar_decay", "supernova_potential", "fuel_consumer"],
        },
        EntityType.PLANET: {
            MotifCategory.ELEMENTAL: ["earth_dreamer", "water_memory", "air_whisper", "fire_core"],
            MotifCategory.CRYSTALLINE: ["tectonic_dreams", "mineral_consciousness", "geometric_growth"],
            MotifCategory.VITAL: ["life_cradle", "sentience_potential", "evolutionary_canvas", "gaia_mind"],
            MotifCategory.RECURSIVE: ["ecosystem_cycle", "weather_patterns", "geological_layers"],
        },
        EntityType.CIVILIZATION: {
            MotifCategory.VITAL: ["collective_consciousness", "dream_weaver", "thought_network", "memory_keeper"],
            MotifCategory.ASCENDANT: ["transcendence_path", "knowledge_seeker", "pattern_recognizer"],
            MotifCategory.RECURSIVE: ["cultural_iteration", "technological_ladder", "historical_cycle"],
            MotifCategory.TEMPORAL: ["future_dreamer", "past_keeper", "now_experiencer", "legacy_builder"],
        },
        EntityType.ANOMALY: {
            MotifCategory.CHAOTIC: ["pattern_breaker", "uncertainty_spike", "probability_storm", "anomalous_field"],
            MotifCategory.DIMENSIONAL: ["space_fold", "reality_tear", "boundary_transgression"],
            MotifCategory.SHADOW: ["dark_secret", "unknown_variable", "void_whisper", "cosmos_blindspot"],
            MotifCategory.ABYSSAL: ["bottomless_depth", "infinite_recursion", "meaning_void", "null_state"],
        },
    }
    
    # Counter-motifs create opposing forces and dramatic tension
    _counter_motifs = {
        "radiant_halo": "void_shadow",
        "fusion_crucible": "cold_silence",
        "light_bearer": "darkness_bringer",
        "life_cradle": "death_harbor",
        "knowledge_seeker": "mystery_keeper",
        "pattern_recognizer": "chaos_embracer",
        "self_similar_pattern": "fractal_disruption",
        "timestream_source": "entropy_sink",
        "dimensional_flux": "spatial_stasis",
    }
    
    # Resonant motifs that reinforce and amplify each other
    _resonant_motifs = {
        "first_breath": ["genesis_point", "all_potential"],
        "stellar_tempest": ["void_turbulence", "dark_flow"],
        "plasma_heart": ["fusion_crucible", "energy_fountain"],
        "tectonic_dreams": ["mineral_consciousness", "geometric_growth"],
        "transcendence_path": ["knowledge_seeker", "pattern_recognizer"],
    }
    
    @classmethod
    def seed_motifs(cls, entity: CosmicEntity, seed_value: int = None):
        """
        Initialize an entity with symbolic motifs based on its type.
        
        Args:
            entity: The cosmic entity to initialize
            seed_value: Optional seed for deterministic motif generation
        
        Returns:
            List of applied motifs
        """
        if not hasattr(entity, 'entity_type'):
            return []
            
        # Use provided seed or entity's hash
        if seed_value is None:
            seed_value = hash(entity.entity_id)
        
        # Seed random generator for deterministic results
        random.seed(seed_value)
        
        # Get motif pool for this entity type
        entity_pools = cls._motif_pools.get(entity.entity_type, {})
        if not entity_pools:
            # Use generic pool if no specific one exists
            entity_pools = {
                MotifCategory.ELEMENTAL: ["generic_pattern", "basic_form", "standard_structure"],
                MotifCategory.CHAOTIC: ["random_element", "unpredictable_aspect", "complex_behavior"],
            }
        
        # Select 2-5 categories based on seed
        available_categories = list(entity_pools.keys())
        num_categories = min(len(available_categories), random.randint(2, 5))
        selected_categories = random.sample(available_categories, num_categories)
        
        # Select 1-3 motifs from each category
        selected_motifs = []
        for category in selected_categories:
            motifs_in_category = entity_pools[category]
            num_motifs = min(len(motifs_in_category), random.randint(1, 3))
            selected_motifs.extend(random.sample(motifs_in_category, num_motifs))
        
        # Add motifs to entity
        entity.motifs = selected_motifs.copy()
        
        # Add counter-motifs with some probability
        cls._add_counter_motifs(entity, seed_value)
        
        # Reset random seed to avoid affecting other random processes
        random.seed()
        
        return entity.motifs
    
    @classmethod
    def _add_counter_motifs(cls, entity: CosmicEntity, seed_value: int):
        """Add opposing motifs for dramatic tension"""
        random.seed(seed_value + 1)  # Different seed from main motifs
        
        for motif in entity.motifs.copy():
            if motif in cls._counter_motifs and random.random() < 0.3:
                counter = cls._counter_motifs[motif]
                entity.motifs.append(counter)
                
    @classmethod
    def find_resonance(cls, entity1: CosmicEntity, entity2: CosmicEntity) -> float:
        """
        Calculate motif resonance between two entities.
        Returns a value between 0 (no resonance) and 1 (perfect resonance).
        """
        if not hasattr(entity1, 'motifs') or not hasattr(entity2, 'motifs'):
            return 0.0
            
        # Direct motif matches
        common_motifs = set(entity1.motifs).intersection(set(entity2.motifs))
        direct_score = len(common_motifs) / max(len(entity1.motifs) + len(entity2.motifs), 1)
        
        # Resonant motif groups
        resonance_score = 0
        for motif1 in entity1.motifs:
            if motif1 in cls._resonant_motifs:
                for resonant in cls._resonant_motifs[motif1]:
                    if resonant in entity2.motifs:
                        resonance_score += 0.5  # Half point for resonant matches
        
        resonance_score = min(1.0, resonance_score / max(len(entity1.motifs) + len(entity2.motifs), 1))
        
        # Counter-motif tension (reduces resonance)
        tension_score = 0
        for motif1 in entity1.motifs:
            if motif1 in cls._counter_motifs and cls._counter_motifs[motif1] in entity2.motifs:
                tension_score += 1
                
        for motif2 in entity2.motifs:
            if motif2 in cls._counter_motifs and cls._counter_motifs[motif2] in entity1.motifs:
                tension_score += 1
                
        tension_score = min(1.0, tension_score / max(len(entity1.motifs) + len(entity2.motifs), 1))
        
        # Combine scores (direct matches + resonance - tension)
        return max(0, min(1.0, direct_score + resonance_score - tension_score))


# -------------------------------------------------------------------------
# ScrollMemory System
# -------------------------------------------------------------------------
class ScrollMemoryEvent:
    """Represents a significant event in an entity's timeline"""
    
    def __init__(self, timestamp: float, event_type: str, description: str, importance: float = 0.5,
                entities_involved: List[str] = None, motifs_added: List[str] = None, location=None):
        self.timestamp = timestamp
        self.event_type = event_type
        self.description = description
        self.importance = importance  # 0.0 to 1.0
        self.entities_involved = entities_involved or []
        self.motifs_added = motifs_added or []
        self.location = location  # Coordinates or sector
        
    def to_dict(self) -> Dict:
        """Convert event to dictionary for storage"""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'description': self.description,
            'importance': self.importance,
            'entities_involved': self.entities_involved,
            'motifs_added': self.motifs_added,
            'location': self.location
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ScrollMemoryEvent':
        """Create event from dictionary"""
        return cls(
            timestamp=data.get('timestamp', 0),
            event_type=data.get('event_type', 'unknown'),
            description=data.get('description', ''),
            importance=data.get('importance', 0.5),
            entities_involved=data.get('entities_involved', []),
            motifs_added=data.get('motifs_added', []),
            location=data.get('location')
        )
    
    def __str__(self) -> str:
        """String representation of event"""
        return f"[T:{self.timestamp:.2f}] {self.event_type}: {self.description}"


class ScrollMemory:
    """
    Memory system that records significant events in an entity's history.
    Provides continuity and emergence of meaningful narratives.
    """
    
    def __init__(self, owner_id: str, capacity: int = 100):
        self.owner_id = owner_id
        self.events = []
        self.capacity = capacity
        self.last_consolidation = 0
        self.thematic_summary = {}  # Event type -> frequency count
        
    def record_event(self, event: ScrollMemoryEvent):
        """
        Add a new event to the memory scroll
        
        Args:
            event: The event to record
        """
        self.events.append(event)
        
        # Update thematic summary
        self.thematic_summary[event.event_type] = self.thematic_summary.get(event.event_type, 0) + 1
        
        # Ensure we don't exceed capacity by consolidating or pruning
        if len(self.events) > self.capacity:
            self._consolidate_memory()
    
    def get_events_by_type(self, event_type: str) -> List[ScrollMemoryEvent]:
        """Retrieve all events of a specific type"""
        return [e for e in self.events if e.event_type == event_type]
    
    def get_events_by_timeframe(self, start_time: float, end_time: float) -> List[ScrollMemoryEvent]:
        """Retrieve events within a specific timeframe"""
        return [e for e in self.events if start_time <= e.timestamp <= end_time]
    
    def get_events_involving_entity(self, entity_id: str) -> List[ScrollMemoryEvent]:
        """Retrieve events involving a specific entity"""
        return [e for e in self.events if entity_id in e.entities_involved]
    
    def get_most_important_events(self, count: int = 10) -> List[ScrollMemoryEvent]:
        """Retrieve the most important events"""
        sorted_events = sorted(self.events, key=lambda e: e.importance, reverse=True)
        return sorted_events[:count]
    
    def get_narrative_arc(self) -> Dict[str, List[ScrollMemoryEvent]]:
        """
        Identify narrative arcs in the entity's history
        Returns a dict mapping arc type to sequence of events
        """
        arcs = {}
        
        # Group events by type
        event_types = set(e.event_type for e in self.events)
        for etype in event_types:
            type_events = sorted(self.get_events_by_type(etype), key=lambda e: e.timestamp)
            if len(type_events) >= 3:
                arcs[etype] = type_events
                
        return arcs
    
    def _consolidate_memory(self):
        """
        Consolidate memory by merging or removing less important events
        This is called when the scroll exceeds capacity
        """
        if len(self.events) <= self.capacity:
            return
            
        # Sort events by importance
        sorted_events = sorted(self.events, key=lambda e: e.importance)
        
        # Identify candidates for consolidation (lowest importance events)
        consolidation_candidates = sorted_events[:len(sorted_events) // 3]
        
        # Group candidates by type and timeframe
        by_type_and_time = {}
        for event in consolidation_candidates:
            time_bucket = int(event.timestamp / 10) * 10  # Group by 10 time units
            key = (event.event_type, time_bucket)
            if key not in by_type_and_time:
                by_type_and_time[key] = []
            by_type_and_time[key].append(event)
        
        # Merge groups that have multiple events
        events_to_remove = []
        for (event_type, _), group in by_type_and_time.items():
            if len(group) > 1:
                # Create a consolidated event
                earliest = min(e.timestamp for e in group)
                importance = max(e.importance for e in group)
                entities = set()
                motifs = set()
                descriptions = []
                
                for e in group:
                    entities.update(e.entities_involved)
                    motifs.update(e.motifs_added)
                    descriptions.append(e.description)
                    events_to_remove.append(e)
                
                consolidated = ScrollMemoryEvent(
                    timestamp=earliest,
                    event_type=event_type,
                    description=f"Multiple events: {'; '.join(descriptions[:3])}",
                    importance=importance,
                    entities_involved=list(entities),
                    motifs_added=list(motifs)
                )
                
                self.events.append(consolidated)
        
        # Remove consolidated events
        for event in events_to_remove:
            self.events.remove(event)
            
        # If still over capacity, remove least important events
        if len(self.events) > self.capacity:
            self.events = sorted(self.events, key=lambda e: e.importance, reverse=True)[:self.capacity]
        
        # Update last consolidation time
        self.last_consolidation = max(e.timestamp for e in self.events) if self.events else 0
        
        # Rebuild thematic summary
        self._rebuild_thematic_summary()
    
    def _rebuild_thematic_summary(self):
        """Rebuild the thematic summary after consolidation"""
        self.thematic_summary = {}
        for event in self.events:
            self.thematic_summary[event.event_type] = self.thematic_summary.get(event.event_type, 0) + 1
    
    def get_memory_keywords(self) -> List[str]:
        """Extract keywords that define this entity's identity based on memory"""
        keywords = []
        
        # Add most frequent event types
        for event_type, count in sorted(self.thematic_summary.items(), key=lambda x: x[1], reverse=True)[:3]:
            keywords.append(event_type)
            
        # Add most common motifs
        motif_count = {}
        for event in self.events:
            for motif in event.motifs_added:
                motif_count[motif] = motif_count.get(motif, 0) + 1
                
        for motif, _ in sorted(motif_count.items(), key=lambda x: x[1], reverse=True)[:3]:
            keywords.append(motif)
            
        return keywords
    
    def generate_timeline_summary(self, max_events: int = 5) -> str:
        """Generate a textual summary of the entity's history"""
        if not self.events:
            return "No recorded history."
        
        # Sort by timestamp
        sorted_events = sorted(self.events, key=lambda e: e.timestamp)
        
        # Select key events based on importance and distribution across time
        events_count = len(sorted_events)
        if events_count <= max_events:
            key_events = sorted_events
        else:
            # Choose some top importance events plus some distributed across time
            by_importance = sorted(sorted_events, key=lambda e: e.importance, reverse=True)
            top_half = max_events // 2
            
            # Take top events by importance
            key_events = by_importance[:top_half]
            
            # Take some distributed across timeline
            time_range = sorted_events[-1].timestamp - sorted_events[0].timestamp
            if time_range > 0:
                indices = [int(i * (events_count - 1) / (max_events - top_half)) for i in range(max_events - top_half)]
                for idx in indices:
                    if sorted_events[idx] not in key_events:
                        key_events.append(sorted_events[idx])
            
            # Sort by timestamp
            key_events = sorted(key_events, key=lambda e: e.timestamp)
        
        # Generate summary text
        summary = []
        for event in key_events:
            summary.append(f"T{event.timestamp:.1f}: {event.description}")
        
        return "\n".join(summary)


# Augment the CosmicEntity class with ScrollMemory
def add_scroll_memory_to_entity(entity: CosmicEntity):
    """Add a ScrollMemory to a cosmic entity if it doesn't already have one"""
    if not hasattr(entity, 'scroll_memory'):
        entity.scroll_memory = ScrollMemory(entity.entity_id)
        return entity.scroll_memory
    return getattr(entity, 'scroll_memory')


# Add scroll_memory attribute and methods to CosmicEntity
def augment_entities_with_scroll_memory():
    """Augment all existing entities with ScrollMemory"""
    for entity_id, entity in DRM.entities.items():
        if isinstance(entity, CosmicEntity) and not hasattr(entity, 'scroll_memory'):
            add_scroll_memory_to_entity(entity)


# Add record_event method to CosmicEntity
def record_scroll_event(self, event_type: str, description: str, importance: float = 0.5,
                      entities_involved: List[str] = None, motifs_added: List[str] = None,
                      location=None, timestamp: float = None):
    """
    Record an event in this entity's memory scroll
    
    Args:
        event_type: Category of event
        description: Text description
        importance: How important this event is (0.0-1.0)
        entities_involved: List of entity IDs involved
        motifs_added: List of motifs related to this event
        location: Where the event occurred
        timestamp: When the event occurred (defaults to current time)
    """
    if not hasattr(self, 'scroll_memory'):
        add_scroll_memory_to_entity(self)
        
    if timestamp is None:
        timestamp = self.last_update_time
        
    # Create the event
    event = ScrollMemoryEvent(
        timestamp=timestamp,
        event_type=event_type,
        description=description,
        importance=importance,
        entities_involved=entities_involved or [],
        motifs_added=motifs_added or [],
        location=location
    )
    
    # Record in scroll memory
    self.scroll_memory.record_event(event)
    
    # If motifs were added, update entity's motifs
    if motifs_added:
        for motif in motifs_added:
            if motif not in self.motifs:
                self.motifs.append(motif)
    
    # For important events, ensure they're shared with related entities
    if importance >= 0.7 and entities_involved:
        self._propagate_event_to_related_entities(event)
            
    return event


def _propagate_event_to_related_entities(self, event: ScrollMemoryEvent):
    """Share important events with related entities"""
    for entity_id in event.entities_involved:
        if entity_id != self.entity_id:
            entity = DRM.get_entity(entity_id)
            if isinstance(entity, CosmicEntity):
                # Create a copy of the event with reduced importance
                related_event = ScrollMemoryEvent(
                    timestamp=event.timestamp,
                    event_type=event.event_type,
                    description=f"Related: {event.description}",
                    importance=event.importance * 0.8,  # Less important for related entities
                    entities_involved=event.entities_involved,
                    motifs_added=event.motifs_added,
                    location=event.location
                )
                
                # Add to other entity's scroll
                if hasattr(entity, 'scroll_memory'):
                    entity.scroll_memory.record_event(related_event)
                else:
                    add_scroll_memory_to_entity(entity).record_event(related_event)


# Bind new methods to CosmicEntity
CosmicEntity.record_scroll_event = record_scroll_event
CosmicEntity._propagate_event_to_related_entities = _propagate_event_to_related_entities


# -------------------------------------------------------------------------
# CultureEngine System
# -------------------------------------------------------------------------
class BeliefType(Enum):
    """Types of beliefs that shape cultural identity"""
    COSMOLOGY = "cosmology"  # Origin/structure of universe
    MORALITY = "morality"    # Right and wrong
    ONTOLOGY = "ontology"    # Nature of being/reality
    EPISTEMOLOGY = "epistemology"  # Knowledge acquisition
    ESCHATOLOGY = "eschatology"  # Ultimate destiny
    AXIOLOGY = "axiology"   # Values and value judgments
    THEOLOGY = "theology"   # Divine/transcendent entities
    TELEOLOGY = "teleology" # Purpose and design


class SocialStructure(Enum):
    """Organizational patterns for civilizations"""
    TRIBAL = "tribal"          # Kinship-based groups
    FEUDAL = "feudal"          # Hierarchical land ownership
    IMPERIAL = "imperial"      # Centralized empire
    REPUBLIC = "republic"      # Representative governance
    DIRECT_DEMOCRACY = "direct_democracy"  # Citizen voting
    TECHNOCRACY = "technocracy"  # Rule by technical experts
    OLIGARCHY = "oligarchy"    # Rule by small group
    HIVE_MIND = "hive_mind"    # Collective consciousness
    DISTRIBUTED = "distributed"  # Decentralized networks
    QUANTUM_CONSENSUS = "quantum_consensus"  # Superposition of choices


class SymbolicArchetype(Enum):
    """Cultural archetypes that appear across civilizations"""
    CREATOR = "creator"        # Creative/generative force
    DESTROYER = "destroyer"    # Destructive/transformative force
    TRICKSTER = "trickster"    # Chaos agent, boundary-crosser
    SAGE = "sage"              # Wisdom figure
    HERO = "hero"              # Challenger of adversity
    RULER = "ruler"            # Authority figure
    CAREGIVER = "caregiver"    # Nurturing force
    EXPLORER = "explorer"      # Seeker of knowledge/territory
    INNOCENT = "innocent"      # Pure, untainted perspective
    SHADOW = "shadow"          # Hidden/suppressed aspects


class CultureEngine:
    """
    Simulates the emergence and evolution of cultures within civilizations,
    including belief systems, archetypes, naming conventions, and social structures.
    """
    
    def __init__(self, civilization: Civilization):
        self.civilization = civilization
        self.belief_systems = {}  # BeliefType -> value (0.0-1.0 scale)
        self.social_structure = None
        self.archetypes = []  # List of active archetypes
        self.naming_patterns = {}  # Category -> pattern
        self.languages = []
        self.values = {}  # Value name -> importance (0.0-1.0)
        self.taboos = []
        self.rituals = []
        self.cultural_motifs = []
        self.cultural_age = 0
        self.adaptations = []
        self.cultural_coherence = 0.8  # How unified the culture is (0.0-1.0)
        self.divergent_subcultures = []  # List of emerging subcultures
        
        # Name components for procedural naming
        self.name_components = {
            'prefixes': [],
            'roots': [],
            'suffixes': []
        }
        
        # Dynamic variables that track cultural changes over time
        self.cultural_shift_momentum = {}  # Aspect -> direction and strength
        self.external_influences = []  # Other cultures influencing this one
        
        # Initialize culture based on civilization traits and planet
        self._initialize_culture()
    
    def _initialize_culture(self):
        """Initialize culture based on civilization traits and environment"""
        # Get planet data for environmental influences
        planet = DRM.get_entity(self.civilization.planet_id) if self.civilization.planet_id else None
        
        # Initialize belief systems with random tendencies
        for belief_type in BeliefType:
            self.belief_systems[belief_type] = random.random()
        
        # Environmental influences on beliefs and values
        if planet:
            # Planet with extreme climate affects cosmology
            if hasattr(planet, 'temperature'):
                if planet.temperature < -30:
                    # Cold world - more structured, ordered cosmology
                    self.belief_systems[BeliefType.COSMOLOGY] = max(0.7, self.belief_systems[BeliefType.COSMOLOGY])
                    self.values['endurance'] = 0.9
                    self.values['community'] = 0.8
                elif planet.temperature > 40:
                    # Hot world - more chaotic cosmology
                    self.belief_systems[BeliefType.COSMOLOGY] = min(0.3, self.belief_systems[BeliefType.COSMOLOGY])
                    self.values['adaptation'] = 0.9
                    self.values['resourcefulness'] = 0.8
            
            # Geological activity affects ontology (nature of being)
            if 'tectonic_dreams' in planet.motifs:
                self.belief_systems[BeliefType.ONTOLOGY] = max(0.6, self.belief_systems[BeliefType.ONTOLOGY])
                self.archetypes.append(SymbolicArchetype.DESTROYER)
            
            # Water worlds influence values
            if hasattr(planet, 'surface') and planet.surface.get('water', 0) > 0.7:
                self.values['flow'] = 0.8
                self.values['depth'] = 0.7
                self.archetypes.append(SymbolicArchetype.EXPLORER)
        
        # Choose initial social structure based on civ development
        self._select_social_structure()
        
        # Select initial archetypes (2-4 primary ones)
        self._initialize_archetypes()
        
        # Generate naming patterns
        self._generate_naming_conventions()
        
        # Initialize cultural motifs (separate from entity motifs)
        self._initialize_cultural_motifs()
    
    def _select_social_structure(self):
        """Select appropriate social structure based on civilization traits"""
        if not self.civilization:
            self.social_structure = SocialStructure.TRIBAL
            return
            
        # Choose based on development level and tech focus
        if self.civilization.development_level < 0.2:
            # Early civilization
            self.social_structure = SocialStructure.TRIBAL
        elif self.civilization.development_level < 0.4:
            # Emerging civilization
            self.social_structure = random.choice([
                SocialStructure.TRIBAL, 
                SocialStructure.FEUDAL
            ])
        elif self.civilization.development_level < 0.6:
            # Established civilization
            if hasattr(self.civilization, 'tech_focus'):
                if self.civilization.tech_focus == DevelopmentArea.SOCIAL_ORGANIZATION:
                    self.social_structure = random.choice([
                        SocialStructure.REPUBLIC,
                        SocialStructure.DIRECT_DEMOCRACY
                    ])
                else:
                    self.social_structure = random.choice([
                        SocialStructure.FEUDAL,
                        SocialStructure.IMPERIAL,
                        SocialStructure.OLIGARCHY
                    ])
        elif self.civilization.development_level < 0.8:
            # Advanced civilization
            if hasattr(self.civilization, 'tech_focus'):
                if self.civilization.tech_focus == DevelopmentArea.COMPUTATION:
                    self.social_structure = SocialStructure.TECHNOCRACY
                elif self.civilization.quantum_understanding > 0.7:
                    self.social_structure = SocialStructure.QUANTUM_CONSENSUS
                else:
                    self.social_structure = random.choice([
                        SocialStructure.REPUBLIC,
                        SocialStructure.DIRECT_DEMOCRACY,
                        SocialStructure.TECHNOCRACY
                    ])
        else:
            # Highly advanced civilization
            if self.civilization.quantum_understanding > 0.9:
                if random.random() < 0.3:
                    self.social_structure = SocialStructure.HIVE_MIND
                else:
                    self.social_structure = SocialStructure.QUANTUM_CONSENSUS
            else:
                self.social_structure = random.choice([
                    SocialStructure.TECHNOCRACY,
                    SocialStructure.DISTRIBUTED
                ])
    
    def _initialize_archetypes(self):
        """Select initial cultural archetypes"""
        # Always include CREATOR archetype
        self.archetypes.append(SymbolicArchetype.CREATOR)
        
        # Select 1-3 additional archetypes
        available_archetypes = [a for a in SymbolicArchetype if a != SymbolicArchetype.CREATOR]
        num_additional = random.randint(1, 3)
        self.archetypes.extend(random.sample(available_archetypes, num_additional))
        
        # Make sure archetypes list contains Enum values, not plain strings
        self.archetypes = [a if isinstance(a, SymbolicArchetype) else a for a in self.archetypes]
    
    def _generate_naming_conventions(self):
        """Generate naming patterns for this culture"""
        # Initialize name components based on culture characteristics
        consonants = 'bcdfghjklmnpqrstvwxyz'
        vowels = 'aeiou'
        
        # Adjust sound palette based on environment and beliefs
        if self.belief_systems.get(BeliefType.COSMOLOGY, 0) > 0.7:
            # Ordered cosmos - more structured names
            consonants = 'kptbdgmnsr'
            vowels = 'aeiou'
        elif self.belief_systems.get(BeliefType.COSMOLOGY, 0) < 0.3:
            # Chaotic cosmos - more varied sounds
            consonants = 'bcdfghjklmnpqrstvwxz'
            vowels = 'aeiouy'
        
        # Generate specific components
        num_prefixes = random.randint(5, 12)
        num_roots = random.randint(10, 20)
        num_suffixes = random.randint(5, 12)
        
        for _ in range(num_prefixes):
            length = random.randint(1, 3)
            prefix = ''
            for i in range(length):
                if i % 2 == 0:
                    prefix += random.choice(consonants)
                else:
                    prefix += random.choice(vowels)
            self.name_components['prefixes'].append(prefix)
            
        for _ in range(num_roots):
            length = random.randint(2, 4)
            root = ''
            start_with = random.choice([0, 1])  # 0: consonant, 1: vowel
            for i in range(length):
                if (i + start_with) % 2 == 0:
                    root += random.choice(consonants)
                else:
                    root += random.choice(vowels)
            self.name_components['roots'].append(root)
            
        for _ in range(num_suffixes):
            length = random.randint(1, 3)
            suffix = ''
            for i in range(length):
                if i % 2 == 0:
                    suffix += random.choice(vowels)
                else:
                    suffix += random.choice(consonants)
            self.name_components['suffixes'].append(suffix)
        
        # Define naming patterns for different entity types
        self.naming_patterns = {
            'person': lambda: self._generate_name('person'),
            'place': lambda: self._generate_name('place'),
            'concept': lambda: self._generate_name('concept'),
            'deity': lambda: self._generate_name('deity')
        }
    
    def _generate_name(self, entity_type: str) -> str:
        """Generate a name based on cultural patterns"""
        result = ''
        
        if entity_type == 'person':
            # Person names can have prefix + root or root + suffix
            if random.random() < 0.5 and self.name_components['prefixes']:
                result += random.choice(self.name_components['prefixes'])
            
            result += random.choice(self.name_components['roots'])
            
            if random.random() < 0.5 and self.name_components['suffixes']:
                result += random.choice(self.name_components['suffixes'])
                
        elif entity_type == 'place':
            # Places often have compound structure: root + root or root + suffix
            result += random.choice(self.name_components['roots'])
            
            if random.random() < 0.7:
                if random.random() < 0.5 and self.name_components['roots']:
                    result += random.choice(self.name_components['roots'])
                elif self.name_components['suffixes']:
                    result += random.choice(self.name_components['suffixes'])
                    
        elif entity_type == 'concept':
            # Concepts can be more abstract: prefix + root or root + suffix
            if random.random() < 0.6 and self.name_components['prefixes']:
                result += random.choice(self.name_components['prefixes'])
                
            result += random.choice(self.name_components['roots'])
            
            if random.random() < 0.6 and self.name_components['suffixes']:
                result += random.choice(self.name_components['suffixes'])
                
        elif entity_type == 'deity':
            # Deities often have grander names: prefix + root + suffix
            if self.name_components['prefixes']:
                result += random.choice(self.name_components['prefixes'])
                
            result += random.choice(self.name_components['roots'])
            
            if self.name_components['suffixes']:
                result += random.choice(self.name_components['suffixes'])
        
        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:]
            
        return result
    
    def _initialize_cultural_motifs(self):
        """Initialize cultural motifs based on environment and beliefs"""
        # Carry over relevant motifs from civilization
        if hasattr(self.civilization, 'motifs'):
            for motif in self.civilization.motifs:
                if any(term in motif for term in ['consciousness', 'mind', 'dream', 'thought', 'memory']):
                    self.cultural_motifs.append(motif)
        
        # Add basic cultural motifs based on social structure
        if self.social_structure == SocialStructure.TRIBAL:
            self.cultural_motifs.extend(['kinship_bonds', 'ancestral_wisdom', 'seasonal_cycles'])
        elif self.social_structure == SocialStructure.FEUDAL:
            self.cultural_motifs.extend(['hierarchical_order', 'loyalty_chains', 'land_connection'])
        elif self.social_structure == SocialStructure.IMPERIAL:
            self.cultural_motifs.extend(['centralized_power', 'expansionist_destiny', 'glory_cult'])
        elif self.social_structure == SocialStructure.REPUBLIC:
            self.cultural_motifs.extend(['collective_wisdom', 'balanced_powers', 'civic_duty'])
        elif self.social_structure == SocialStructure.DIRECT_DEMOCRACY:
            self.cultural_motifs.extend(['voice_of_all', 'consensus_seeking', 'public_discourse'])
        elif self.social_structure == SocialStructure.TECHNOCRACY:
            self.cultural_motifs.extend(['optimization_ideal', 'expertise_value', 'system_thinking'])
        elif self.social_structure == SocialStructure.QUANTUM_CONSENSUS:
            self.cultural_motifs.extend(['probability_thinking', 'superposition_identity', 'entangled_fate'])
        
        # Add motifs based on archetypes
        for archetype in self.archetypes:
            if archetype == SymbolicArchetype.CREATOR:
                self.cultural_motifs.append('genesis_narrative')
            elif archetype == SymbolicArchetype.DESTROYER:
                self.cultural_motifs.append('renewal_through_ending')
            elif archetype == SymbolicArchetype.TRICKSTER:
                self.cultural_motifs.append('wisdom_through_disruption')
            elif archetype == SymbolicArchetype.HERO:
                self.cultural_motifs.append('individual_transcendence')
    
    def evolve_culture(self, time_delta: float):
        """
        Evolve culture over time, responding to internal and external pressures
        
        Args:
            time_delta: Time increment for evolution
        """
        self.cultural_age += time_delta
        
        # Evolve belief systems
        self._evolve_beliefs(time_delta)
        
        # Check for social structure transitions
        self._check_social_structure_transition(time_delta)
        
        # Evolve cultural motifs
        self._evolve_cultural_motifs(time_delta)
        
        # Handle subculture emergence and divergence
        self._handle_cultural_divergence(time_delta)
        
        # Update naming conventions periodically
        if random.random() < 0.05 * time_delta:
            self._evolve_naming_conventions()
        
        # Record significant cultural events
        self._generate_cultural_events(time_delta)
    
    def _evolve_beliefs(self, time_delta: float):
        """Evolve belief systems over time"""
        # Get external influences
        external_pressure = self._calculate_external_pressure()
        
        # Update each belief system
        for belief_type in self.belief_systems:
            # Natural drift - beliefs gradually change
            drift = (random.random() - 0.5) * 0.05 * time_delta
            
            # External influence pushes belief systems
            ext_influence = 0
            if external_pressure.get(belief_type):
                target, strength = external_pressure[belief_type]
                ext_influence = (target - self.belief_systems[belief_type]) * strength * time_delta
            
            # Internal consistency pressure
            internal_pressure = 0
            if belief_type == BeliefType.COSMOLOGY and BeliefType.ONTOLOGY in self.belief_systems:
                # Cosmology and ontology tend to align
                ontology_val = self.belief_systems[BeliefType.ONTOLOGY]
                internal_pressure = (ontology_val - self.belief_systems[belief_type]) * 0.02 * time_delta
            
            # Development level can push certain beliefs
            dev_pressure = 0
            if hasattr(self.civilization, 'development_level'):
                if belief_type == BeliefType.EPISTEMOLOGY:
                    # Higher development pushes toward rational epistemology
                    dev_target = min(0.8, self.civilization.development_level)
                    dev_pressure = (dev_target - self.belief_systems[belief_type]) * 0.03 * time_delta
                    
                elif belief_type == BeliefType.THEOLOGY:
                    # Higher technical development can reduce theological focus
                    if hasattr(self.civilization, 'tech_levels') and 'computation' in self.civilization.tech_levels:
                        comp_level = self.civilization.tech_levels['computation']
                        if comp_level > 0.7:
                            dev_target = max(0.2, 1.0 - comp_level)
                            dev_pressure = (dev_target - self.belief_systems[belief_type]) * 0.02 * time_delta
            
            # Combine all influences
            total_change = drift + ext_influence + internal_pressure + dev_pressure
            
            # Apply change with limits
            self.belief_systems[belief_type] = max(0.01, min(0.99, self.belief_systems[belief_type] + total_change))
    
    def _calculate_external_pressure(self) -> Dict:
        """Calculate external cultural influences"""
        pressure = {}
        
        # Consider influences from known civilizations
        if hasattr(self.civilization, 'known_civilizations'):
            for civ_id in self.civilization.known_civilizations:
                other_civ = DRM.get_entity(civ_id)
                if not other_civ or not hasattr(other_civ, 'culture_engine'):
                    continue
                    
                # Calculate influence strength based on development difference
                dev_diff = other_civ.development_level - self.civilization.development_level
                influence_strength = 0.1  # Base influence
                
                if dev_diff > 0.2:
                    # More advanced civilizations have stronger influence
                    influence_strength += dev_diff * 0.3
                    
                # Add influence for each belief system
                for belief_type, value in other_civ.culture_engine.belief_systems.items():
                    if belief_type not in pressure:
                        pressure[belief_type] = (value, influence_strength)
                    else:
                        # Average with existing influences
                        current_target, current_strength = pressure[belief_type]
                        new_strength = current_strength + influence_strength
                        new_target = (current_target * current_strength + value * influence_strength) / new_strength
                        pressure[belief_type] = (new_target, new_strength)
        
        return pressure
    
    def _check_social_structure_transition(self, time_delta: float):
        """Check if social structure should evolve"""
        if not hasattr(self.civilization, 'development_level'):
            return
            
        current = self.social_structure
        dev_level = self.civilization.development_level
        
        # Development thresholds that might trigger transitions
        if current == SocialStructure.TRIBAL and dev_level > 0.3:
            if random.random() < 0.1 * time_delta:
                self.social_structure = SocialStructure.FEUDAL
                self._record_social_transition(current, self.social_structure)
                
        elif current == SocialStructure.FEUDAL and dev_level > 0.5:
            if random.random() < 0.1 * time_delta:
                choices = [SocialStructure.IMPERIAL, SocialStructure.REPUBLIC]
                self.social_structure = random.choice(choices)
                self._record_social_transition(current, self.social_structure)
                
        elif current == SocialStructure.IMPERIAL and dev_level > 0.7:
            if random.random() < 0.1 * time_delta:
                if hasattr(self.civilization, 'tech_focus') and self.civilization.tech_focus == DevelopmentArea.COMPUTATION:
                    self.social_structure = SocialStructure.TECHNOCRACY
                else:
                    self.social_structure = SocialStructure.REPUBLIC
                self._record_social_transition(current, self.social_structure)
                
        elif current == SocialStructure.REPUBLIC and dev_level > 0.8:
            if random.random() < 0.1 * time_delta:
                choices = [SocialStructure.DIRECT_DEMOCRACY, SocialStructure.TECHNOCRACY]
                self.social_structure = random.choice(choices)
                self._record_social_transition(current, self.social_structure)
                
        elif dev_level > 0.9 and hasattr(self.civilization, 'quantum_understanding'):
            if self.civilization.quantum_understanding > 0.8 and random.random() < 0.1 * time_delta:
                self.social_structure = SocialStructure.QUANTUM_CONSENSUS
                self._record_social_transition(current, self.social_structure)
    
    def _record_social_transition(self, old_structure, new_structure):
        """Record a social structure transition event"""
        if hasattr(self.civilization, 'record_scroll_event'):
            self.civilization.record_scroll_event(
                event_type="social_evolution",
                description=f"Society transformed from {old_structure.value} to {new_structure.value}",
                importance=0.7,
                motifs_added=[f"social_{new_structure.value}"]
            )
    
    def _evolve_cultural_motifs(self, time_delta: float):
        """Evolve cultural motifs over time"""
        # Chance to add new motifs
        if random.random() < 0.1 * time_delta:
            # Potential new motifs based on current state
            potential_motifs = []
            
            # Add motifs based on belief strengths
            for belief_type, value in self.belief_systems.items():
                if value > 0.7:
                    if belief_type == BeliefType.COSMOLOGY:
                        potential_motifs.extend(['cosmic_order', 'celestial_harmony'])
                    elif belief_type == BeliefType.ONTOLOGY:
                        potential_motifs.extend(['being_awareness', 'essence_focus'])
                    elif belief_type == BeliefType.EPISTEMOLOGY:
                        potential_motifs.extend(['truth_seeking', 'knowledge_path'])
                    elif belief_type == BeliefType.THEOLOGY:
                        potential_motifs.extend(['divine_connection', 'transcendent_aspiration'])
            
            # Add motifs based on current social structure
            if self.social_structure == SocialStructure.QUANTUM_CONSENSUS:
                potential_motifs.extend(['quantum_identity', 'probability_ethics'])
            elif self.social_structure == SocialStructure.TECHNOCRACY:
                potential_motifs.extend(['optimization_mandate', 'efficiency_pursuit'])
                
            # Select a new motif if any are available
            if potential_motifs and random.random() < 0.3:
                new_motif = random.choice(potential_motifs)
                if new_motif not in self.cultural_motifs:
                    self.cultural_motifs.append(new_motif)
                    
                    # Record this cultural development
                    if hasattr(self.civilization, 'record_scroll_event'):
                        self.civilization.record_scroll_event(
                            event_type="cultural_development",
                            description=f"Culture developed new motif: {new_motif}",
                            importance=0.5,
                            motifs_added=[new_motif]
                        )
        
        # Chance to lose motifs that no longer fit
        if self.cultural_motifs and random.random() < 0.05 * time_delta:
            # Find motifs that might be lost based on belief shifts
            for motif in self.cultural_motifs[:]:  # Copy to avoid modification during iteration
                # Examples of checking for obsolete motifs
                if motif == 'divine_connection' and self.belief_systems.get(BeliefType.THEOLOGY, 0.5) < 0.3:
                    self.cultural_motifs.remove(motif)
                elif motif == 'cosmic_order' and self.belief_systems.get(BeliefType.COSMOLOGY, 0.5) < 0.3:
                    self.cultural_motifs.remove(motif)
    
    def _handle_cultural_divergence(self, time_delta: float):
        """Handle emergence of subcultures and cultural divergence"""
        # Factors that increase chance of subculture formation:
        # 1. Population size
        # 2. Multiple colonized planets
        # 3. Low cultural coherence
        # 4. High development level
        
        divergence_chance = 0.01 * time_delta  # Base chance
        
        if hasattr(self.civilization, 'population') and self.civilization.population > 1000000:
            divergence_chance += 0.05
            
        if hasattr(self.civilization, 'colonized_planets') and len(self.civilization.colonized_planets) > 1:
            divergence_chance += 0.1 * len(self.civilization.colonized_planets)
            
        divergence_chance *= (2.0 - self.cultural_coherence)  # Lower coherence increases chance
        
        if hasattr(self.civilization, 'development_level'):
            divergence_chance *= (1.0 + self.civilization.development_level)  # Higher development increases chance
            
        # Check for new subculture formation
        if random.random() < divergence_chance:
            self._form_new_subculture()
            
        # Evolve existing subcultures
        for subculture in self.divergent_subcultures:
            subculture['divergence'] = min(1.0, subculture['divergence'] + 0.05 * time_delta)
            
            # Subculture might influence main culture
            if random.random() < 0.1 * time_delta:
                influence_strength = subculture['divergence'] * 0.1
                for belief_type, value in subculture['beliefs'].items():
                    if belief_type in self.belief_systems:
                        self.belief_systems[belief_type] += (value - self.belief_systems[belief_type]) * influence_strength
                
                # Subculture might contribute motifs to main culture
                if random.random() < subculture['divergence'] * 0.2:
                    potential_motifs = [m for m in subculture['motifs'] if m not in self.cultural_motifs]
                    if potential_motifs:
                        self.cultural_motifs.append(random.choice(potential_motifs))
    
    def _form_new_subculture(self):
        """Create a new divergent subculture"""
        # Base new subculture on main culture, but with variations
        new_subculture = {
            'name': self._generate_name('concept'),
            'formation_time': self.cultural_age,
            'divergence': 0.1,  # Starts only slightly different
            'beliefs': {},
            'motifs': [],
            'social_structure': self.social_structure,
            'location': None  # Will be set if specific location exists
        }
        
        # Copy beliefs with variations
        for belief_type, value in self.belief_systems.items():
            variation = (random.random() - 0.5) * 0.4  # -0.2 to 0.2 variation
            new_subculture['beliefs'][belief_type] = max(0.1, min(0.9, value + variation))
            
        # Copy some motifs, add some new ones
        common_motifs = random.sample(self.cultural_motifs, min(len(self.cultural_motifs), 3))
        new_subculture['motifs'].extend(common_motifs)
        
        # Add some unique motifs
        unique_motifs = [
            'identity_seeking',
            'tradition_breaking',
            'path_divergence',
            'alternative_vision',
            'cultural_rebellion'
        ]
        new_subculture['motifs'].append(random.choice(unique_motifs))
        
        # Set location if on multiple planets
        if hasattr(self.civilization, 'colonized_planets') and self.civilization.colonized_planets:
            new_subculture['location'] = random.choice(self.civilization.colonized_planets)
            
        # Add to subcultures list
        self.divergent_subcultures.append(new_subculture)
        
        # Reduce overall cultural coherence
        self.cultural_coherence = max(0.1, self.cultural_coherence - 0.05)
        
        # Record this cultural event
        if hasattr(self.civilization, 'record_scroll_event'):
            self.civilization.record_scroll_event(
                event_type="cultural_divergence",
                description=f"New subculture formed: {new_subculture['name']}",
                importance=0.6,
                motifs_added=['cultural_divergence']
            )
            
        return new_subculture
    
    def _evolve_naming_conventions(self):
        """Evolve naming conventions over time"""
        # Chance to add new components
        component_types = ['prefixes', 'roots', 'suffixes']
        component_type = random.choice(component_types)
        
        # Create a new component
        consonants = 'bcdfghjklmnpqrstvwxyz'
        vowels = 'aeiou'
        
        length = random.randint(1, 3)
        new_component = ''
        for i in range(length):
            if i % 2 == 0:
                new_component += random.choice(consonants)
            else:
                new_component += random.choice(vowels)
                
        self.name_components[component_type].append(new_component)
        
        # Chance to remove old components (but always keep at least 3 of each)
        if len(self.name_components[component_type]) > 5 and random.random() < 0.3:
            removed = random.choice(self.name_components[component_type])
            self.name_components[component_type].remove(removed)
    
    def _generate_cultural_events(self, time_delta: float):
        """Generate cultural events based on current state"""
        # Chance for significant cultural events
        if random.random() < 0.05 * time_delta:
            # Possible event types
            event_types = [
                "artistic_revolution",
                "philosophical_breakthrough",
                "spiritual_awakening",
                "linguistic_evolution",
                "value_shift",
                "ritual_creation",
                "archetype_transformation"
            ]
            
            event_type = random.choice(event_types)
            event_description = ""
            event_importance = 0.5
            event_motifs = []
            
            if event_type == "artistic_revolution":
                art_form = random.choice(["visual", "musical", "literary", "performative", "immersive"])
                movement_name = self._generate_name('concept')
                event_description = f"New artistic movement '{movement_name}' revolutionized {art_form} expression"
                event_motifs = ["artistic_innovation"]
                
            elif event_type == "philosophical_breakthrough":
                philosopher = self._generate_name('person')
                concept = self._generate_name('concept')
                event_description = f"Philosopher {philosopher} introduced concept of '{concept}'"
                event_motifs = ["philosophical_advance"]
                event_importance = 0.7
                
            elif event_type == "spiritual_awakening":
                leader = self._generate_name('person')
                teaching = self._generate_name('concept')
                event_description = f"Spiritual leader {leader} revealed teachings of '{teaching}'"
                event_motifs = ["spiritual_revelation"]
                event_importance = 0.8
                
            elif event_type == "linguistic_evolution":
                language_name = self._generate_name('concept')
                event_description = f"Language evolved into new dialect: {language_name}"
                event_motifs = ["linguistic_drift"]
                
            elif event_type == "value_shift":
                old_value = random.choice(list(self.values.keys())) if self.values else "tradition"
                new_value = self._generate_name('concept')
                event_description = f"Cultural values shifted from '{old_value}' toward '{new_value}'"
                event_motifs = ["value_evolution"]
                # Update values dictionary
                self.values[new_value] = 0.8
                if old_value in self.values:
                    self.values[old_value] = max(0.1, self.values[old_value] - 0.3)
                
            elif event_type == "ritual_creation":
                ritual_name = self._generate_name('concept')
                purpose = random.choice(["harmony", "cleansing", "remembrance", "transition", "union"])
                event_description = f"New ritual '{ritual_name}' established for {purpose}"
                event_motifs = ["ritual_creation"]
                self.rituals.append(ritual_name)
                
            elif event_type == "archetype_transformation":
                old_archetype = random.choice(self.archetypes) if self.archetypes else SymbolicArchetype.CREATOR
                available_archetypes = [a for a in SymbolicArchetype if a not in self.archetypes]
                if available_archetypes:
                    new_archetype = random.choice(available_archetypes)
                    event_description = f"Cultural archetype shifted from {old_archetype.value} to {new_archetype.value}"
                    event_motifs = ["archetype_shift"]
                    # Update archetypes
                    if old_archetype in self.archetypes:
                        self.archetypes.remove(old_archetype)
                    self.archetypes.append(new_archetype)
            
            # Record event in civilization's scroll memory
            if hasattr(self.civilization, 'record_scroll_event') and event_description:
                self.civilization.record_scroll_event(
                    event_type=event_type,
                    description=event_description,
                    importance=event_importance,
                    motifs_added=event_motifs
                )

# Add culture_engine to Civilization class
def _initialize_culture_engine(self):
    """Initialize culture engine for this civilization"""
    if not hasattr(self, 'culture_engine'):
        self.culture_engine = CultureEngine(self)
        
    # Record culture initialization in scroll memory
    if hasattr(self, 'record_scroll_event'):
        self.record_scroll_event(
            event_type="culture_genesis",
            description=f"Culture established with {self.culture_engine.social_structure.value} structure",
            importance=0.8,
            motifs_added=self.culture_engine.cultural_motifs[:3]
        )
    
    return self.culture_engine

# Update evolve method to use culture engine
def _civilization_evolve_with_culture(self, time_delta: float):
    """Evolve civilization with cultural component"""
    # Original evolve method (store reference to original)
    if hasattr(self, '_original_evolve'):
        self._original_evolve(time_delta)
    else:
        super(Civilization, self).evolve(time_delta)
    
    # Ensure culture engine exists
    if not hasattr(self, 'culture_engine'):
        self._initialize_culture_engine()
        
    # Evolve culture
    self.culture_engine.evolve_culture(time_delta)

# Bind culture methods to Civilization
Civilization._initialize_culture_engine = _initialize_culture_engine
Civilization._original_evolve = Civilization.evolve  # Store original method
Civilization.evolve = _civilization_evolve_with_culture  # Replace with new method


# -------------------------------------------------------------------------
# Civilization Interaction System
# -------------------------------------------------------------------------
class InteractionType(Enum):
    """Types of relationships between civilizations"""
    CONFLICT = "conflict"            # Active hostility
    COOPERATION = "cooperation"      # Mutual aid and exchange
    COMPETITION = "competition"      # Non-violent rivalry
    CULTURAL_EXCHANGE = "cultural_exchange"  # Ideas and beliefs flow
    TRADE = "trade"                  # Economic relations
    SUBJUGATION = "subjugation"      # Dominance of one over another
    ISOLATION = "isolation"          # Deliberate separation
    OBSERVATION = "observation"      # One studying the other covertly
    HYBRID = "hybrid"                # Complex mix of relationship types


class CivilizationInteraction:
    """
    Manages relationships and interactions between civilizations,
    including conflict, cooperation, and hybrid states.
    """
    
    def __init__(self, civ1_id: str, civ2_id: str):
        self.civ1_id = civ1_id
        self.civ2_id = civ2_id
        self.relation_id = f"relation_{civ1_id}_{civ2_id}"
        self.interaction_type = InteractionType.OBSERVATION
        self.motif_resonance = 0.0
        self.technological_parity = 0.0
        self.cultural_compatibility = 0.0
        self.tension = 0.0
        self.shared_history = []  # List of historical events
        self.diplomatic_status = "neutral"
        self.treaties = []
        self.war_status = False
        self.trade_volume = 0
        self.last_update_time = 0
        
        # Initialize the relationship
        self._initialize_relationship()
    
    def _initialize_relationship(self):
        """Set initial relationship parameters based on civilizations"""
        civ1 = DRM.get_entity(self.civ1_id)
        civ2 = DRM.get_entity(self.civ2_id)
        
        if not civ1 or not civ2:
            return
            
        # Calculate technological parity (0 = disparate, 1 = equal)
        tech_diff = {}
        for area in DevelopmentArea:
            if hasattr(civ1.tech_levels, area) and hasattr(civ2.tech_levels, area):
                tech_diff[area] = abs(civ1.tech_levels[area] - civ2.tech_levels[area])
                
        if tech_diff:
            avg_tech_diff = sum(tech_diff.values()) / len(tech_diff)
            self.technological_parity = 1.0 - avg_tech_diff
        
        # Calculate motif resonance using MotifSeeder
        self.motif_resonance = MotifSeeder.find_resonance(civ1, civ2)
        
        # Initial cultural compatibility based on belief systems
        if hasattr(civ1, 'culture_engine') and hasattr(civ2, 'culture_engine'):
            belief_compatibility = 0.0
            belief_count = 0
            
            for belief_type in BeliefType:
                if (belief_type in civ1.culture_engine.belief_systems and 
                    belief_type in civ2.culture_engine.belief_systems):
                    diff = abs(civ1.culture_engine.belief_systems[belief_type] - 
                               civ2.culture_engine.belief_systems[belief_type])
                    belief_compatibility += (1.0 - diff)
                    belief_count += 1
                    
            if belief_count > 0:
                self.cultural_compatibility = belief_compatibility / belief_count
        
        # Initialize tension based on inverted compatibility
        self.tension = 0.3 + (1.0 - self.cultural_compatibility) * 0.4 + (1.0 - self.motif_resonance) * 0.3
        
        # Select initial interaction type
        self._update_interaction_type()
    
    def _update_interaction_type(self):
        """Determine interaction type based on current metrics"""
        if self.war_status:
            self.interaction_type = InteractionType.CONFLICT
            return
            
        # Get the civilizations
        civ1 = DRM.get_entity(self.civ1_id)
        civ2 = DRM.get_entity(self.civ2_id)
        
        if not civ1 or not civ2:
            return
            
        # Development gap can lead to subjugation
        dev_gap = abs(civ1.development_level - civ2.development_level)
        if dev_gap > 0.4 and self.tension > 0.6:
            # Significant development gap and high tension
            higher_civ = civ1 if civ1.development_level > civ2.development_level else civ2
            if higher_civ.entity_id == self.civ1_id:
                self.interaction_type = InteractionType.SUBJUGATION
                return
                
        # High tension leads to conflict or competition
        if self.tension > 0.7:
            if self.technological_parity > 0.8:
                # Similar tech levels mean direct conflict is more likely
                self.interaction_type = InteractionType.CONFLICT
            else:
                # Different tech levels lead to competition
                self.interaction_type = InteractionType.COMPETITION
            return
            
        # High cultural compatibility encourages cooperation or exchange
        if self.cultural_compatibility > 0.7:
            if self.technological_parity > 0.6:
                # Similar tech and culture leads to full cooperation
                self.interaction_type = InteractionType.COOPERATION
            else:
                # Cultural similarity but tech difference leads to cultural exchange
                self.interaction_type = InteractionType.CULTURAL_EXCHANGE
            return
            
        # High tech parity and moderate cultural compatibility can lead to trade
        if self.technological_parity > 0.6 and self.cultural_compatibility > 0.4:
            self.interaction_type = InteractionType.TRADE
            return
            
        # Low resonance and low tension can lead to isolation
        if self.motif_resonance < 0.3 and self.tension < 0.4:
            self.interaction_type = InteractionType.ISOLATION
            return
            
        # Complex mix of factors
        if (0.3 < self.motif_resonance < 0.7 and 
            0.3 < self.technological_parity < 0.7 and
            0.3 < self.cultural_compatibility < 0.7):
            self.interaction_type = InteractionType.HYBRID
            return
            
        # Default to observation
        self.interaction_type = InteractionType.OBSERVATION
    
    def update_relationship(self, time_delta: float):
        """Update the relationship between civilizations over time"""
        self.last_update_time += time_delta
        
        # Get the civilizations
        civ1 = DRM.get_entity(self.civ1_id)
        civ2 = DRM.get_entity(self.civ2_id)
        
        if not civ1 or not civ2:
            return
            
        # Update tech parity
        tech_diff = {}
        for area in DevelopmentArea:
            if hasattr(civ1.tech_levels, area) and hasattr(civ2.tech_levels, area):
                tech_diff[area] = abs(civ1.tech_levels[area] - civ2.tech_levels[area])
                
        if tech_diff:
            avg_tech_diff = sum(tech_diff.values()) / len(tech_diff)
            self.technological_parity = 1.0 - avg_tech_diff
            
        # Cultural compatibility evolves based on current interaction
        if hasattr(civ1, 'culture_engine') and hasattr(civ2, 'culture_engine'):
            if self.interaction_type == InteractionType.CULTURAL_EXCHANGE:
                # Cultural exchange increases compatibility
                self.cultural_compatibility = min(1.0, self.cultural_compatibility + 0.05 * time_delta)
            elif self.interaction_type == InteractionType.CONFLICT:
                # Conflict decreases compatibility
                self.cultural_compatibility = max(0.0, self.cultural_compatibility - 0.05 * time_delta)
                
        # Tension evolves based on interaction type and random factors
        tension_change = 0.0
        
        if self.interaction_type == InteractionType.CONFLICT:
            tension_change = 0.05  # Conflicts increase tension
        elif self.interaction_type == InteractionType.COOPERATION:
            tension_change = -0.05  # Cooperation decreases tension
        elif self.interaction_type == InteractionType.TRADE:
            tension_change = -0.02  # Trade slightly decreases tension
        
        # Random factors
        tension_change += (random.random() - 0.5) * 0.04 * time_delta
        
        # Apply tension change
        self.tension = max(0.1, min(0.9, self.tension + tension_change))
        
        # Check for significant events
        if random.random() < 0.1 * time_delta:
            self._generate_interaction_event()
            
        # Re-evaluate interaction type
        self._update_interaction_type()
        
        # Update each civilization based on interaction
        self._apply_interaction_effects(time_delta)
    
    def _generate_interaction_event(self):
        """Generate a significant event in the relationship"""
        civ1 = DRM.get_entity(self.civ1_id)
        civ2 = DRM.get_entity(self.civ2_id)
        
        if not civ1 or not civ2:
            return
            
        event_desc = ""
        event_importance = 0.5
        event_motifs = []
        
        # Different events based on interaction type
        if self.interaction_type == InteractionType.CONFLICT:
            if random.random() < 0.3:
                # Major conflict
                event_desc = f"War broke out between {civ1.entity_id} and {civ2.entity_id}"
                event_importance = 0.9
                event_motifs = ["interspecies_war"]
                self.war_status = True
            else:
                # Minor conflict
                event_desc = f"Border skirmish between {civ1.entity_id} and {civ2.entity_id}"
                event_importance = 0.7
                event_motifs = ["territorial_dispute"]
                
        elif self.interaction_type == InteractionType.COOPERATION:
            if random.random() < 0.3:
                # Major alliance
                event_desc = f"Alliance formed between {civ1.entity_id} and {civ2.entity_id}"
                event_importance = 0.8
                event_motifs = ["interspecies_alliance"]
                self.treaties.append(("Alliance", self.last_update_time))
            else:
                # Joint project
                event_desc = f"Joint technological project between {civ1.entity_id} and {civ2.entity_id}"
                event_importance = 0.6
                event_motifs = ["technological_cooperation"]
                
        elif self.interaction_type == InteractionType.TRADE:
            event_desc = f"Trade agreement established between {civ1.entity_id} and {civ2.entity_id}"
            event_importance = 0.5
            event_motifs = ["interspecies_commerce"]
            self.trade_volume += 0.2
            self.treaties.append(("Trade Agreement", self.last_update_time))
            
        elif self.interaction_type == InteractionType.CULTURAL_EXCHANGE:
            event_desc = f"Cultural exchange program between {civ1.entity_id} and {civ2.entity_id}"
            event_importance = 0.6
            event_motifs = ["cultural_transmission"]
            
        elif self.interaction_type == InteractionType.SUBJUGATION:
            # Determine which civilization is dominant
            dominant = civ1 if civ1.development_level > civ2.development_level else civ2
            subjugated = civ2 if dominant == civ1 else civ1
            
            event_desc = f"{dominant.entity_id} established dominance over {subjugated.entity_id}"
            event_importance = 0.8
            event_motifs = ["power_hierarchy", "empire_building"]
            
        # Record the event in both civilizations' scroll memories
        if event_desc:
            if hasattr(civ1, 'record_scroll_event'):
                civ1.record_scroll_event(
                    event_type="diplomacy",
                    description=event_desc,
                    importance=event_importance,
                    entities_involved=[civ1.entity_id, civ2.entity_id],
                    motifs_added=event_motifs
                )
                
            if hasattr(civ2, 'record_scroll_event'):
                civ2.record_scroll_event(
                    event_type="diplomacy",
                    description=event_desc,
                    importance=event_importance,
                    entities_involved=[civ1.entity_id, civ2.entity_id],
                    motifs_added=event_motifs
                )
                
            # Add to shared history
            self.shared_history.append({
                'time': self.last_update_time,
                'description': event_desc,
                'type': self.interaction_type.value,
                'importance': event_importance
            })
    
    def _apply_interaction_effects(self, time_delta: float):
        """Apply effects of the relationship to both civilizations"""
        civ1 = DRM.get_entity(self.civ1_id)
        civ2 = DRM.get_entity(self.civ2_id)
        
        if not civ1 or not civ2:
            return
            
        # Different effects based on interaction type
        if self.interaction_type == InteractionType.COOPERATION:
            # Boost tech in areas where the other civ is stronger
            for area in DevelopmentArea:
                if (hasattr(civ1.tech_levels, area) and 
                    hasattr(civ2.tech_levels, area)):
                    # Civ 1 learns from Civ 2 in areas where Civ 2 is stronger
                    if civ2.tech_levels[area] > civ1.tech_levels[area]:
                        boost = (civ2.tech_levels[area] - civ1.tech_levels[area]) * 0.1 * time_delta
                        civ1.tech_levels[area] = min(1.0, civ1.tech_levels[area] + boost)
                    
                    # Civ 2 learns from Civ 1
                    if civ1.tech_levels[area] > civ2.tech_levels[area]:
                        boost = (civ1.tech_levels[area] - civ2.tech_levels[area]) * 0.1 * time_delta
                        civ2.tech_levels[area] = min(1.0, civ2.tech_levels[area] + boost)
            
        elif self.interaction_type == InteractionType.CONFLICT:
            # War pushes military technology but drains resources
            if hasattr(civ1.tech_levels, DevelopmentArea.WEAPONRY):
                civ1.tech_levels[DevelopmentArea.WEAPONRY] = min(1.0, civ1.tech_levels[DevelopmentArea.WEAPONRY] + 0.05 * time_delta)
                
            if hasattr(civ2.tech_levels, DevelopmentArea.WEAPONRY):
                civ2.tech_levels[DevelopmentArea.WEAPONRY] = min(1.0, civ2.tech_levels[DevelopmentArea.WEAPONRY] + 0.05 * time_delta)
                
            # Population and development penalties
            civ1.population = max(1000, int(civ1.population * (1 - 0.02 * time_delta)))
            civ2.population = max(1000, int(civ2.population * (1 - 0.02 * time_delta)))
            
            # Check for war resolution
            if random.random() < 0.05 * time_delta:
                self._resolve_conflict()
                
        elif self.interaction_type == InteractionType.CULTURAL_EXCHANGE:
            # Exchange cultural motifs
            if hasattr(civ1, 'culture_engine') and hasattr(civ2, 'culture_engine'):
                if random.random() < 0.1 * time_delta:
                    # Civ 1 learns from Civ 2
                    if civ2.culture_engine.cultural_motifs:
                        motif = random.choice(civ2.culture_engine.cultural_motifs)
                        if motif not in civ1.culture_engine.cultural_motifs:
                            civ1.culture_engine.cultural_motifs.append(motif)
                    
                    # Civ 2 learns from Civ 1
                    if civ1.culture_engine.cultural_motifs:
                        motif = random.choice(civ1.culture_engine.cultural_motifs)
                        if motif not in civ2.culture_engine.cultural_motifs:
                            civ2.culture_engine.cultural_motifs.append(motif)
                            
                # Belief systems influence each other
                for belief_type in BeliefType:
                    if (belief_type in civ1.culture_engine.belief_systems and 
                        belief_type in civ2.culture_engine.belief_systems):
                        # Both belief systems drift toward each other
                        civ1_belief = civ1.culture_engine.belief_systems[belief_type]
                        civ2_belief = civ2.culture_engine.belief_systems[belief_type]
                        
                        drift = (civ2_belief - civ1_belief) * 0.05 * time_delta
                        civ1.culture_engine.belief_systems[belief_type] += drift
                        civ2.culture_engine.belief_systems[belief_type] -= drift
        
        elif self.interaction_type == InteractionType.TRADE:
            # Trade boosts economies and certain technologies
            boost = 0.02 * time_delta * self.trade_volume
            
            # Population growth boost
            civ1.population = int(civ1.population * (1 + 0.01 * boost))
            civ2.population = int(civ2.population * (1 + 0.01 * boost))
            
            # Technology boost in energy, materials, computation
            for area in [DevelopmentArea.ENERGY, DevelopmentArea.MATERIALS, DevelopmentArea.COMPUTATION]:
                if hasattr(civ1.tech_levels, area):
                    civ1.tech_levels[area] = min(1.0, civ1.tech_levels[area] + 0.01 * boost)
                if hasattr(civ2.tech_levels, area):
                    civ2.tech_levels[area] = min(1.0, civ2.tech_levels[area] + 0.01 * boost)
                    
        elif self.interaction_type == InteractionType.SUBJUGATION:
            # Determine dominant civilization
            dominant = civ1 if civ1.development_level > civ2.development_level else civ2
            subjugated = civ2 if dominant == civ1 else civ1
            
            # Resource flow from subjugated to dominant
            # Dominant grows faster, subjugated grows slower
            dominant.population = int(dominant.population * (1 + 0.02 * time_delta))
            subjugated.population = max(1000, int(subjugated.population * (1 - 0.01 * time_delta)))
            
            # Technology transfer (one-way)
            for area in DevelopmentArea:
                if (hasattr(dominant.tech_levels, area) and 
                    hasattr(subjugated.tech_levels, area)):
                    if dominant.tech_levels[area] > subjugated.tech_levels[area]:
                        # Slow tech transfer to subjugated
                        boost = (dominant.tech_levels[area] - subjugated.tech_levels[area]) * 0.05 * time_delta
                        subjugated.tech_levels[area] = min(dominant.tech_levels[area] * 0.8, 
                                                         subjugated.tech_levels[area] + boost)
    
    def _resolve_conflict(self):
        """Resolve an ongoing conflict between civilizations"""
        civ1 = DRM.get_entity(self.civ1_id)
        civ2 = DRM.get_entity(self.civ2_id)
        
        if not civ1 or not civ2:
            return
            
        # Determine victor based on multiple factors
        # - Military technology
        # - Population
        # - Overall development
        # - Random chance
        
        # Calculate military strength
        mil_tech1 = civ1.tech_levels.get(DevelopmentArea.WEAPONRY, 0.1)
        mil_tech2 = civ2.tech_levels.get(DevelopmentArea.WEAPONRY, 0.1)
        
        # Population factor (logarithmic to avoid pure numbers determining outcome)
        pop_factor1 = math.log10(max(1000, civ1.population)) / 10
        pop_factor2 = math.log10(max(1000, civ2.population)) / 10
        
        # Overall development
        dev1 = civ1.development_level
        dev2 = civ2.development_level
        
        # Calculate total strength
        strength1 = mil_tech1 * 0.5 + pop_factor1 * 0.3 + dev1 * 0.2
        strength2 = mil_tech2 * 0.5 + pop_factor2 * 0.3 + dev2 * 0.2
        
        # Add random factor (fog of war)
        strength1 *= random.uniform(0.8, 1.2)
        strength2 *= random.uniform(0.8, 1.2)
        
        # Determine victor
        victor = civ1 if strength1 > strength2 else civ2
        defeated = civ2 if victor == civ1 else civ1
        
        # War resolution effects
        self.war_status = False
        
        # Victor gains, defeated loses
        if victor.development_level > 0.7 and defeated.development_level > 0.7:
            # Advanced civilizations sign peace treaty
            event_desc = f"Peace treaty signed between {civ1.entity_id} and {civ2.entity_id}"
            self.treaties.append(("Peace Treaty", self.last_update_time))
            
            # Reset tension
            self.tension = 0.4
            
            # Both lose some population
            victor.population = int(victor.population * 0.95)
            defeated.population = int(defeated.population * 0.9)
            
        elif victor.development_level - defeated.development_level > 0.3:
            # Significant tech gap leads to subjugation
            event_desc = f"{victor.entity_id} defeated and subjugated {defeated.entity_id}"
            
            # Set interaction to subjugation
            self.interaction_type = InteractionType.SUBJUGATION
            
            # Population effects
            victor.population = int(victor.population * 0.97)
            defeated.population = int(defeated.population * 0.7)
            
        else:
            # Standard victory
            event_desc = f"{victor.entity_id} defeated {defeated.entity_id} in war"
            
            # Population effects
            victor.population = int(victor.population * 0.95)
            defeated.population = int(defeated.population * 0.8)
            
            # Reset tension
            self.tension = 0.5
            
        # Record event in both civilizations' scroll memories
        if hasattr(civ1, 'record_scroll_event'):
            civ1.record_scroll_event(
                event_type="war_resolution",
                description=event_desc,
                importance=0.8,
                entities_involved=[civ1.entity_id, civ2.entity_id],
                motifs_added=["war_conclusion"]
            )
            
        if hasattr(civ2, 'record_scroll_event'):
            civ2.record_scroll_event(
                event_type="war_resolution",
                description=event_desc,
                importance=0.8,
                entities_involved=[civ1.entity_id, civ2.entity_id],
                motifs_added=["war_conclusion"]
            )
            
        # Add to shared history
        self.shared_history.append({
            'time': self.last_update_time,
            'description': event_desc,
            'type': 'war_resolution',
            'importance': 0.8
        })


# Central registry for civilization interactions
class DiplomaticRegistry:
    """Manages all civilization interactions in the simulation"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DiplomaticRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the diplomatic registry"""
        self.relationships = {}  # (civ1_id, civ2_id) -> CivilizationInteraction
        self.alliances = {}  # alliance_id -> [civ_ids]
        self.wars = {}  # war_id -> [civ_ids, start_time, cause]
        self.trade_networks = {}  # network_id -> [civ_ids, trade_volume]
    
    def get_relationship(self, civ1_id: str, civ2_id: str) -> CivilizationInteraction:
        """Get or create relationship between two civilizations"""
        # Sort IDs to ensure consistent key
        if civ1_id > civ2_id:
            civ1_id, civ2_id = civ2_id, civ1_id
            
        key = (civ1_id, civ2_id)
        
        if key not in self.relationships:
            self.relationships[key] = CivilizationInteraction(civ1_id, civ2_id)
            
        return self.relationships[key]
    
    def update_all_relationships(self, time_delta: float):
        """Update all diplomatic relationships"""
        for relationship in self.relationships.values():
            relationship.update_relationship(time_delta)
            
        # Update meta-structures like alliances and trade networks
        self._update_diplomatic_structures()
    
    def _update_diplomatic_structures(self):
        """Update alliance networks, trade blocs, etc."""
        # Reset structures
        self.alliances = {}
        self.trade_networks = {}
        
        # Rebuild based on current relationships
        processed = set()
        
        for (civ1_id, civ2_id), relationship in self.relationships.items():
            if (civ1_id, civ2_id) in processed:
                continue
                
            processed.add((civ1_id, civ2_id))
            
            # Check for alliances
            if relationship.interaction_type == InteractionType.COOPERATION:
                # Find existing alliance to add to
                alliance_id = None
                for aid, members in self.alliances.items():
                    if civ1_id in members or civ2_id in members:
                        alliance_id = aid
                        break
                        
                if alliance_id:
                    # Add to existing alliance
                    if civ1_id not in self.alliances[alliance_id]:
                        self.alliances[alliance_id].append(civ1_id)
                    if civ2_id not in self.alliances[alliance_id]:
                        self.alliances[alliance_id].append(civ2_id)
                else:
                    # Create new alliance
                    alliance_id = f"alliance_{civ1_id}_{civ2_id}"
                    self.alliances[alliance_id] = [civ1_id, civ2_id]
                    
            # Check for trade networks
            elif relationship.interaction_type == InteractionType.TRADE:
                # Find existing trade network
                network_id = None
                for nid, data in self.trade_networks.items():
                    members = data['members']
                    if civ1_id in members or civ2_id in members:
                        network_id = nid
                        break
                        
                if network_id:
                    # Add to existing network
                    if civ1_id not in self.trade_networks[network_id]['members']:
                        self.trade_networks[network_id]['members'].append(civ1_id)
                    if civ2_id not in self.trade_networks[network_id]['members']:
                        self.trade_networks[network_id]['members'].append(civ2_id)
                    # Update volume
                    self.trade_networks[network_id]['volume'] += relationship.trade_volume
                else:
                    # Create new trade network
                    network_id = f"trade_{civ1_id}_{civ2_id}"
                    self.trade_networks[network_id] = {
                        'members': [civ1_id, civ2_id],
                        'volume': relationship.trade_volume
                    }
                    
        # Update wars
        active_wars = {}
        for (civ1_id, civ2_id), relationship in self.relationships.items():
            if relationship.war_status:
                # Find existing war
                war_id = None
                for wid, data in self.wars.items():
                    if civ1_id in data['members'] and civ2_id in data['members']:
                        war_id = wid
                        break
                        
                if not war_id:
                    # New war
                    war_id = f"war_{civ1_id}_{civ2_id}"
                    history = relationship.shared_history
                    cause = "Unknown"
                    
                    # Find war cause in history
                    for event in reversed(history):
                        if event['type'] == InteractionType.CONFLICT.value:
                            cause = event['description']
                            break
                            
                    active_wars[war_id] = {
                        'members': [civ1_id, civ2_id],
                        'start_time': relationship.last_update_time,
                        'cause': cause
                    }
                else:
                    # Existing war
                    active_wars[war_id] = self.wars[war_id]
                    
        self.wars = active_wars
    
    def get_civilization_conflicts(self, civ_id: str) -> List[Dict]:
        """Get all conflicts a civilization is involved in"""
        conflicts = []
        
        for war_id, data in self.wars.items():
            if civ_id in data['members']:
                conflicts.append({
                    'war_id': war_id,
                    'opponents': [m for m in data['members'] if m != civ_id],
                    'start_time': data['start_time'],
                    'cause': data['cause']
                })
                
        return conflicts
    
    def get_civilization_alliances(self, civ_id: str) -> List[Dict]:
        """Get all alliances a civilization is part of"""
        alliances = []
        
        for alliance_id, members in self.alliances.items():
            if civ_id in members:
                alliances.append({
                    'alliance_id': alliance_id,
                    'members': [m for m in members if m != civ_id]
                })
                
        return alliances
    
    def get_civilization_trade_partners(self, civ_id: str) -> List[Dict]:
        """Get all trade partners of a civilization"""
        partners = []
        
        for network_id, data in self.trade_networks.items():
            if civ_id in data['members']:
                partners.append({
                    'network_id': network_id,
                    'partners': [m for m in data['members'] if m != civ_id],
                    'volume': data['volume']
                })
                
        return partners


# Initialize the diplomatic registry singleton
DIPLOMATIC_REGISTRY = DiplomaticRegistry()
```
# === Integrated Environmental Scroll Modules ===
# Source: scroll_modules.py
# This module contains the integrated environmental scroll modules for the Cosmic Scroll system.
# === World & Environmental Systems ===

```
# -------------------------------------------------------------------------
# Life & Biology Systems
# -------------------------------------------------------------------------
import numpy as np
import uuid
import math
import random
from enum import Enum, auto
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable, Deque
from collections import defaultdict, deque
import logging

class MutationType(Enum):
    """Types of mutations that can occur in biological and symbolic entities"""
    POINT = "point"              # Small change to a single trait or gene
    DUPLICATION = "duplication"  # Copying of a trait or gene
    DELETION = "deletion"        # Removal of a trait or gene
    INVERSION = "inversion"      # Reversal of a trait or gene sequence
    INSERTION = "insertion"      # Addition of new trait or gene
    SYMBOLIC = "symbolic"        # Change to symbolic/motif elements
    RECURSIVE = "recursive"      # Creates self-referential patterns
    MOTIF = "motif"              # Alters motif expression or attunement
    NARRATIVE = "narrative"      # Changes story pattern of organism
    QUANTUM = "quantum"          # Probabilistic mutation affecting multiple states

logger = logging.getLogger(__name__)

class MetabolicProcess(Enum):
    """Types of metabolic processes that can occur in living entities"""
    PHOTOSYNTHESIS = "photosynthesis"       # Light to energy conversion
    RESPIRATION = "respiration"             # Energy extraction from molecules
    CHEMOSYNTHESIS = "chemosynthesis"       # Chemical to energy conversion
    RADIOSYNTHESIS = "radiosynthesis"       # Radiation to energy conversion
    QUANTUM_ENTANGLEMENT = "entanglement"   # Energy from quantum processes
    SYMBOLIC_ABSORPTION = "symbolic"        # Energy from meaning patterns
    MOTIF_CYCLING = "motif_cycling"         # Energy from motif transformations
    THERMAL_CYCLING = "thermal_cycling"     # Energy from temperature differences
    ETHERIC_EXTRACTION = "etheric"          # Energy from etheric fields
    HARMONIC_RESONANCE = "harmonic"         # Energy from harmonic fields


class MetabolicResource(Enum):
    """Resources that can be metabolized by entities within the cosmic simulation"""
    PHYSICAL_MATTER = "physical_matter"          # Standard physical material
    SYMBOLIC_ESSENCE = "symbolic_essence"        # Raw symbolic potential
    TEMPORAL_FLUX = "temporal_flux"              # Time-based energy
    NARRATIVE_THREAD = "narrative_thread"        # Story-based resource
    EMOTIONAL_RESIDUE = "emotional_residue"      # Crystallized emotion
    BELIEF_CURRENT = "belief_current"            # Faith and conviction energy
    VOID_EXTRACT = "void_extract"                # Absence and emptiness distilled
    MEMORY_FRAGMENT = "memory_fragment"          # Processed memory energy
    MOTIF_CONCENTRATE = "motif_concentrate"      # Purified pattern energy
    QUANTUM_POTENTIAL = "quantum_potential"      # Possibility-state energy


class MetabolicPathway(Enum):
    """Fundamental pathways through which entities process resources"""
    TRANSMUTATION = "transmutation"          # Converting one resource to another
    RESONANCE = "resonance"                  # Using harmonic matching to extract energy
    ABSORPTION = "absorption"                # Direct incorporation of resources
    CATALYSIS = "catalysis"                  # Facilitating reactions without consumption
    FUSION = "fusion"                        # Combining resources into new forms
    FILTRATION = "filtration"                # Separating and purifying resources
    CRYSTALLIZATION = "crystallization"      # Solidifying abstract resources
    RECURSION = "recursion"                  # Processing through self-similar cycles
    ENTROPIC = "entropic"                    # Extracting energy from disorder
    SYMBOLIC = "symbolic"                    # Processing through meaning transformation


class RecursiveMetabolism:
    """
    Manages the complex resource processing systems within cosmic entities.
    
    The RecursiveMetabolism system handles how entities consume, process, transform, 
    and generate resources within the symbolic ecosystem. It functions as the entity's
    internal economy, defining how it interacts with available resources and converts
    them into usable energy and byproducts.
    
    Through recursive processing loops that can reference their own outputs as inputs,
    entities develop complex metabolic networks that evolve over time based on resource
    availability, environmental conditions, and the entity's changing needs.
    
    Key features:
    - Multi-stage resource processing chains
    - Symbolic transmutation of resource types
    - Recursive processing loops with variable depth
    - Adaptive pathway strengthening based on usage
    - Metabolic byproduct generation and excretion
    - Resource storage and usage optimization
    """
    
    def __init__(self, owner_entity: Any, complexity: float = 0.5, 
                 primary_pathways: List[MetabolicPathway] = None,
                 preferred_resources: List[MetabolicResource] = None,
                 max_recursion_depth: int = 3):
        """
        Initialize a new RecursiveMetabolism system.
        
        Args:
            owner_entity: The entity this metabolism belongs to
            complexity: How complex the metabolic network is (0.0-1.0)
            primary_pathways: List of primary metabolic pathways
            preferred_resources: List of preferred resource types
            max_recursion_depth: Maximum depth of recursive processing
        """
        self.owner = owner_entity
        self.complexity = complexity
        self.primary_pathways = primary_pathways or self._initialize_primary_pathways()
        self.preferred_resources = preferred_resources or self._initialize_preferred_resources()
        self.max_recursion_depth = max_recursion_depth
        
        # Core metabolism components
        self.pathways = self._initialize_pathways()
        self.catalysts = {}
        self.inhibitors = {}
        self.storage = {}
        self.current_processes = []
        
        # Performance metrics
        self.efficiency = {}  # resource_type -> efficiency_value
        self.processing_rate = {}  # pathway -> rate_value
        self.waste_production = {}  # resource_type -> waste_amount
        self.energy_output = 0.0
        
        # Adaptive components
        self.adaptation_history = []
        self.resource_affinities = {}  # resource_type -> affinity_value
        self.pathway_strengths = {p: 0.5 for p in self.primary_pathways}  # Initialize pathway strengths
        
        # Temporal tracking
        self.process_cycles = 0
        self.last_update_time = 0.0
        self.current_recursion_depth = 0
        
        # Symbolic interaction
        self.motif_resonance = {}  # motif -> resonance_value
        self.symbolic_byproducts = []
        self.anomaly_threshold = 0.8
        self.integration_with_environment = 0.5
        
        # Initialize system
        self._initialize_efficiency()
        self._initialize_resource_storage()
        self._initialize_motif_resonance()
        """
        Initialize a recursive metabolism system.
        
        Args:
            owner_entity: The entity this metabolism belongs to
            base_efficiency: Base efficiency of metabolic processes (0.0-1.0)
            primary_process: Primary metabolic process type
            secondary_processes: Secondary metabolic processes
            recursion_depth: How many recursive levels the metabolism operates on
            symbolic_affinity: Dictionary mapping symbolic motifs to affinity values
        """
        self.owner = owner_entity
        self.base_efficiency = max(0.1, min(0.95, base_efficiency))
        self.primary_process = primary_process or self._select_default_process()
        self.secondary_processes = secondary_processes or [self._select_default_process()]
        self.recursion_depth = max(1, min(5, recursion_depth))
        self.symbolic_affinity = symbolic_affinity or self._generate_symbolic_affinity()
        
        # Subprocesses at different scales (molecular, cellular, organ, organism, ecosystem)
        self.subprocesses = self._initialize_subprocesses()
        
        # Byproducts and wastes generated by metabolism
        self.byproducts = {}
        for process in [self.primary_process] + self.secondary_processes:
            self.byproducts.update(self._generate_subprocess_byproducts(process))
            
        # Energy storage in different forms
        self.energy_reserves = {
            "physical": 100.0,
            "symbolic": 50.0,
            "quantum": 20.0,
            "harmonic": 30.0
        }
        
        # Input requirements for different processes
        self.input_requirements = self._calculate_input_requirements()
        
        # Metabolic motifs that influence the process
        self.metabolic_motifs = self._initialize_metabolic_motifs()
        
        # Adaptation history
        self.adaptation_history = []
        
        # Current state tracking
        self.current_efficiency = self.base_efficiency
        self.cycle_count = 0
        self.resource_stress = 0.0
        self.current_input_cache = {}
        self.output_history = deque(maxlen=10)  # Keep track of recent outputs
    
    def _select_default_process(self) -> MetabolicProcess:
        """Select an appropriate default metabolic process based on entity traits."""
        if not self.owner:
            return MetabolicProcess.RESPIRATION
            
        # Check for owner's environment
        environment_type = "unknown"
        if hasattr(self.owner, "planet_id"):
            planet_id = self.owner.planet_id
            planet = DRM.get_entity(planet_id) if planet_id else None
            if planet:
                # Check environment conditions
                if hasattr(planet, "surface"):
                    water_percent = planet.surface.get("water", 0)
                    if water_percent > 0.7:
                        environment_type = "aquatic"
                    elif water_percent > 0.3:
                        environment_type = "mixed"
                    else:
                        environment_type = "terrestrial"
                        
                # Check for radiation levels
                if hasattr(planet, "star_id"):
                    star_id = planet.star_id
                    star = DRM.get_entity(star_id) if star_id else None
                    if star and hasattr(star, "luminosity") and star.luminosity > 1.5:
                        return MetabolicProcess.RADIOSYNTHESIS
                        
        # Check for entity motifs to determine best process
        if hasattr(self.owner, "motifs"):
            motifs = self.owner.motifs
            
            if "light_bearer" in motifs or "energy_fountain" in motifs:
                return MetabolicProcess.PHOTOSYNTHESIS
                
            if "quantum_fluctuation" in motifs or "probability_storm" in motifs:
                return MetabolicProcess.QUANTUM_ENTANGLEMENT
                
            if "harmonic_arrangement" in motifs or "resonance_pattern" in motifs:
                return MetabolicProcess.HARMONIC_RESONANCE
                
            if "pattern_recognizer" in motifs or "meaning_weaver" in motifs:
                return MetabolicProcess.SYMBOLIC_ABSORPTION
        
        # Default based on environment
        if environment_type == "aquatic":
            return MetabolicProcess.CHEMOSYNTHESIS
        elif environment_type == "terrestrial":
            return random.choice([MetabolicProcess.PHOTOSYNTHESIS, MetabolicProcess.RESPIRATION])
        else:
            return random.choice([p for p in MetabolicProcess])
    
    def _generate_symbolic_affinity(self) -> Dict[str, float]:
        """Generate symbolic affinities based on entity traits."""
        affinities = {}
        
        # Base affinities
        base_motifs = [
            "energy_flow", "recursive_cycle", "transformation_pattern",
            "harmonic_resonance", "symbolic_digestion", "quantum_conversion"
        ]
        
        # Add base affinities
        for motif in base_motifs:
            affinities[motif] = random.uniform(0.3, 0.7)
            
        # Add affinities based on owner motifs
        if hasattr(self.owner, "motifs"):
            for motif in self.owner.motifs:
                if "flow" in motif or "energy" in motif or "cycle" in motif:
                    affinities[motif] = random.uniform(0.6, 0.9)
        
        # Add process-specific affinities
        if self.primary_process == MetabolicProcess.PHOTOSYNTHESIS:
            affinities["light_conversion"] = random.uniform(0.7, 0.95)
            affinities["solar_resonance"] = random.uniform(0.6, 0.85)
            
        elif self.primary_process == MetabolicProcess.QUANTUM_ENTANGLEMENT:
            affinities["quantum_uncertainty"] = random.uniform(0.7, 0.95)
            affinities["probability_weaving"] = random.uniform(0.6, 0.85)
            
        elif self.primary_process == MetabolicProcess.SYMBOLIC_ABSORPTION:
            affinities["meaning_extraction"] = random.uniform(0.7, 0.95)
            affinities["narrative_digestion"] = random.uniform(0.6, 0.85)
        
        return affinities
    
    def _initialize_subprocesses(self) -> Dict[str, Dict]:
        """Initialize metabolic subprocesses at different scales."""
        processes = {}
        
        # Scales from microscopic to macroscopic
        scales = ["quantum", "molecular", "cellular", "organ", "organism", "ecosystem"]
        
        # Limit to recursion depth
        active_scales = scales[:min(self.recursion_depth + 1, len(scales))]
        
        for scale in active_scales:
            # Process characteristics vary by scale
            if scale == "quantum":
                efficiency = self.base_efficiency * random.uniform(0.8, 1.2)
                process_types = [MetabolicProcess.QUANTUM_ENTANGLEMENT, MetabolicProcess.HARMONIC_RESONANCE]
                
            elif scale == "molecular":
                efficiency = self.base_efficiency * random.uniform(0.9, 1.1)
                process_types = [MetabolicProcess.RESPIRATION, MetabolicProcess.CHEMOSYNTHESIS]
                
            elif scale == "cellular":
                efficiency = self.base_efficiency * random.uniform(0.9, 1.05)
                process_types = [self.primary_process]
                
            elif scale == "organ":
                efficiency = self.base_efficiency * random.uniform(0.85, 1.0)
                process_types = self.secondary_processes
                
            elif scale == "organism":
                efficiency = self.base_efficiency * random.uniform(0.8, 0.95)
                process_types = [self.primary_process]
                
            else:  # ecosystem
                efficiency = self.base_efficiency * random.uniform(0.7, 0.9)
                process_types = [random.choice([p for p in MetabolicProcess])]
            
            # Create process for this scale
            processes[scale] = {
                "efficiency": efficiency,
                "process_types": process_types,
                "current_activity": 0.0,
                "byproducts": {},
                "inputs": {},
                "outputs": {},
                "feedback_loops": []
            }
            
            # Add scale-specific feedback loops between adjacent scales
            if active_scales.index(scale) > 0:
                previous_scale = active_scales[active_scales.index(scale) - 1]
                processes[scale]["feedback_loops"].append({
                    "from_scale": previous_scale,
                    "to_scale": scale,
                    "intensity": random.uniform(0.2, 0.8),
                    "transformation": random.choice(["amplifying", "dampening", "oscillating"]),
                    "resource_type": random.choice(["energy", "information", "structure", "entropy"])
                })
        
        return processes
    
    def _generate_subprocess_byproducts(self, process_type: MetabolicProcess) -> Dict[str, float]:
        """Generate byproducts for a specific metabolic process."""
        byproducts = {}
        
        # Common byproducts for all processes
        byproducts["heat"] = random.uniform(0.1, 0.3)
        byproducts["entropy"] = random.uniform(0.05, 0.15)
        
        # Process-specific byproducts
        if process_type == MetabolicProcess.PHOTOSYNTHESIS:
            byproducts["oxygen"] = random.uniform(0.3, 0.5)
            byproducts["glucose"] = random.uniform(0.4, 0.6)
            byproducts["growth_potential"] = random.uniform(0.2, 0.4)
            
        elif process_type == MetabolicProcess.RESPIRATION:
            byproducts["carbon_dioxide"] = random.uniform(0.3, 0.5)
            byproducts["water"] = random.uniform(0.2, 0.4)
            byproducts["atp"] = random.uniform(0.5, 0.7)
            
        elif process_type == MetabolicProcess.CHEMOSYNTHESIS:
            byproducts["sulfur_compounds"] = random.uniform(0.2, 0.4)
            byproducts["mineral_structures"] = random.uniform(0.3, 0.5)
            byproducts["chemical_energy"] = random.uniform(0.4, 0.6)
            
        elif process_type == MetabolicProcess.RADIOSYNTHESIS:
            byproducts["radiation_resistance"] = random.uniform(0.4, 0.6)
            byproducts["isotopes"] = random.uniform(0.2, 0.4)
            byproducts["mutation_potential"] = random.uniform(0.3, 0.5)
            
        elif process_type == MetabolicProcess.QUANTUM_ENTANGLEMENT:
            byproducts["quantum_information"] = random.uniform(0.4, 0.6)
            byproducts["probability_waves"] = random.uniform(0.3, 0.5)
            byproducts["entanglement_network"] = random.uniform(0.2, 0.4)
            
        elif process_type == MetabolicProcess.SYMBOLIC_ABSORPTION:
            byproducts["meaning_fragments"] = random.uniform(0.4, 0.6)
            byproducts["narrative_threads"] = random.uniform(0.3, 0.5)
            byproducts["symbolic_structures"] = random.uniform(0.2, 0.4)
            
        elif process_type == MetabolicProcess.MOTIF_CYCLING:
            byproducts["motif_resonance"] = random.uniform(0.4, 0.6)
            byproducts["pattern_seeds"] = random.uniform(0.3, 0.5)
            byproducts["thematic_energy"] = random.uniform(0.2, 0.4)
            
        elif process_type == MetabolicProcess.HARMONIC_RESONANCE:
            byproducts["harmonic_waves"] = random.uniform(0.4, 0.6)
            byproducts["resonance_nodes"] = random.uniform(0.3, 0.5)
            byproducts["attunement_field"] = random.uniform(0.2, 0.4)
            
        return byproducts
    
    def _initialize_metabolic_motifs(self) -> List[str]:
        """Initialize metabolic motifs based on processes and entity traits."""
        motifs = []
        
        # Base motifs common to all metabolisms
        base_motifs = ["energy_flow", "resource_cycle", "entropy_management"]
        motifs.extend(base_motifs)
        
        # Process-specific motifs
        process_motifs = {
            MetabolicProcess.PHOTOSYNTHESIS: ["light_harvesting", "solar_alchemy", "growth_cycle"],
            MetabolicProcess.RESPIRATION: ["oxidation_rhythm", "energy_extraction", "cellular_breath"],
            MetabolicProcess.CHEMOSYNTHESIS: ["mineral_transmutation", "chemical_cascade", "elemental_binding"],
            MetabolicProcess.RADIOSYNTHESIS: ["radiation_harvest", "particle_weaving", "decay_reversal"],
            MetabolicProcess.QUANTUM_ENTANGLEMENT: ["probability_harvest", "quantum_threading", "uncertainty_mapping"],
            MetabolicProcess.SYMBOLIC_ABSORPTION: ["meaning_d


class RecursiveMetabolism:
    """
    Models the symbiotic, fractal energy exchange system in biological entities.
    
    RecursiveMetabolism implements a complex bioenergetic system that functions across
    multiple scales simultaneously, from molecular to ecosystem levels. It manages the
    transformation of various energy forms between symbolic and physical states,
    creating recursive cycles that allow for emergent properties at higher levels.
    
    The system features:
    - Multi-scale energy cycling across different levels of organization
    - Symbolic/physical energy transduction mechanisms
    - Recursive feedback loops between metabolic processes
    - Adaptive pathways that evolve based on environmental conditions
    - Motif integration for symbolic meaning in biological processes
    """
    
    def __init__(self, 
                 owner_entity: Any,
                 base_efficiency: float = 0.7, 
                 primary_process: MetabolicProcess = None, 
                 secondary_processes: List[MetabolicProcess] = None,
                 recursion_depth: int = 3,
                 symbolic_affinity: Dict[str, float] = None):
        """
        Initialize a recursive metabolism system.
        
        Args:
            owner_entity: The entity (planet, flora, fauna) containing this metabolism
            base_efficiency: Base efficiency of energy conversion (0.0-1.0)
            primary_process: Main metabolic process
            secondary_processes: Additional metabolic processes
            recursion_depth: How many levels of recursive processing to maintain
            symbolic_affinity: Affinity for different symbolic energies
        """
        self.owner = owner_entity
        self.base_efficiency = base_efficiency
        self.primary_process = primary_process or self._select_default_process()
        self.secondary_processes = secondary_processes or []
        self.recursion_depth = recursion_depth
        self.symbolic_affinity = symbolic_affinity or self._generate_symbolic_affinity()
        
        # Energy storage at different recursion levels
        self.energy_pools = {
            level: {
                "physical": 50.0 * (0.7 ** level),  # Physical energy decreases with level
                "symbolic": 30.0 * (1.3 ** level),  # Symbolic energy increases with level
                "capacity": 100.0 * (1.1 ** level)  # Capacity increases with level
            } for level in range(recursion_depth + 1)
        }
        
        # Process efficiencies vary by level
        self.process_efficiencies = {
            process: {
                level: base_efficiency * (0.9 + 0.2 * random.random()) * (1.0 + (0.1 * level if process == MetabolicProcess.SYMBOLIC_ABSORPTION else 0)) 
                for level in range(recursion_depth + 1)
            } for process in [self.primary_process] + self.secondary_processes
        }
        
        # Byproducts from metabolism
        self.byproducts = {}
        
        # Energy transfer rates between levels (upward and downward)
        self.upward_transfer_rate = 0.15  # % of energy moving to higher level
        self.downward_transfer_rate = 0.25  # % of energy moving to lower level
        
        # Adaptation metrics
        self.adaptation_pressure = 0.0
        self.adaptation_threshold = 0.7
        self.adaptation_history = deque(maxlen=10)
        
        # Symbiotic relationships
        self.symbiotic_links = {}
        
        # Create recursive subprocesses with their own cycles
        self.subprocesses = self._initialize_subprocesses()
        
        # Integration with motif system
        self.metabolic_motifs = self._initialize_metabolic_motifs()
        
        # Cycle parameters
        self.cycle_count = 0
        self.cycle_length = 12  # Steps per complete metabolic cycle
        self.cycle_phase = 0.0  # Position in current cycle (0.0-1.0)
        
        # Energy needs based on primary process
        self.input_requirements = self._calculate_input_requirements()
        
        # Stats tracking
        self.total_energy_produced = 0
        self.total_energy_consumed = 0
        
    def _select_default_process(self) -> MetabolicProcess:
        """Select an appropriate default metabolic process based on owner entity"""
        if hasattr(self.owner, 'entity_type'):
            if hasattr(EntityType, 'PLANET') and self.owner.entity_type == EntityType.PLANET:
                return MetabolicProcess.RADIOSYNTHESIS
            # For flora/fauna determined by inherited characteristics
            elif hasattr(self.owner, 'is_photosynthetic') and self.owner.is_photosynthetic:
                return MetabolicProcess.PHOTOSYNTHESIS
            elif hasattr(self.owner, 'is_chemosynthetic') and self.owner.is_chemosynthetic:
                return MetabolicProcess.CHEMOSYNTHESIS
            elif hasattr(self.owner, 'is_symbolic_processor') and self.owner.is_symbolic_processor:
                return MetabolicProcess.SYMBOLIC_ABSORPTION
        
        # Default based on planet conditions if available
        if hasattr(self.owner, 'planet') and self.owner.planet:
            if hasattr(self.owner.planet, 'get_trait') and self.owner.planet.get_trait('solar_radiation', 0.5) > 0.6:
                return MetabolicProcess.PHOTOSYNTHESIS
            elif hasattr(self.owner.planet, 'get_trait') and self.owner.planet.get_trait('chemical_energy', 0.3) > 0.7:
                return MetabolicProcess.CHEMOSYNTHESIS
        
        # Fallback
        return random.choice([
            MetabolicProcess.PHOTOSYNTHESIS,
            MetabolicProcess.RESPIRATION,
            MetabolicProcess.CHEMOSYNTHESIS
        ])
    
    def _generate_symbolic_affinity(self) -> Dict[str, float]:
        """Generate affinities for different symbolic energy types"""
        # Base symbolic energy types
        affinities = {
            "light": random.uniform(0.1, 0.9),
            "darkness": random.uniform(0.1, 0.9),
            "order": random.uniform(0.1, 0.9),
            "chaos": random.uniform(0.1, 0.9),
            "creation": random.uniform(0.1, 0.9),
            "destruction": random.uniform(0.1, 0.9),
            "transcendence": random.uniform(0.1, 0.9),
            "immanence": random.uniform(0.1, 0.9),
            "complexity": random.uniform(0.1, 0.9),
            "simplicity": random.uniform(0.1, 0.9)
        }
        
        # If owner has motifs, incorporate those into affinities
        if hasattr(self.owner, 'motifs'):
            for motif in self.owner.motifs:
                # Extract key concepts from motif name
                for symbol in affinities.keys():
                    if symbol in motif:
                        affinities[symbol] += 0.2
        
        # Normalize to ensure sum is reasonable
        total = sum(affinities.values())
        if total > 0:
            affinities = {k: v / total * len(affinities) for k, v in affinities.items()}
        
        return affinities
    
    def _initialize_subprocesses(self) -> Dict[str, Dict]:
        """Initialize recursive metabolic subprocesses"""
        subprocesses = {}
        
        # Create subprocess for each level except the highest
        for level in range(self.recursion_depth):
            process_count = 2 + level  # More subprocesses at deeper levels
            
            level_processes = {}
            for i in range(process_count):
                process_type = random.choice([p for p in MetabolicProcess])
                
                # Each subprocess has its own cycle and efficiency
                level_processes[f"subprocess_{level}_{i}"] = {
                    "type": process_type,
                    "efficiency": random.uniform(0.5, 0.9),
                    "cycle_offset": random.uniform(0, 1.0),
                    "cycle_length": 6 + random.randint(0, 12),
                    "input_ratio": random.uniform(0.1, 0.5),
                    "output_ratio": random.uniform(0.1, 0.5),
                    "symbolic_ratio": random.uniform(0.2, 0.8),
                    "byproducts": self._generate_subprocess_byproducts(process_type)
                }
            
            subprocesses.update(level_processes)
                
        return subprocesses
    
    def _generate_subprocess_byproducts(self, process_type: MetabolicProcess) -> Dict[str, float]:
        """Generate appropriate byproducts for a given metabolic process"""
        byproducts = {}
        
        # Different processes produce different byproducts
        if process_type == MetabolicProcess.PHOTOSYNTHESIS:
            byproducts["oxygen"] = random.uniform(0.1, 0.3)
            byproducts["glucose"] = random.uniform(0.05, 0.15)
            byproducts["symbolic_light"] = random.uniform(0.01, 0.1) 
            
        elif process_type == MetabolicProcess.RESPIRATION:
            byproducts["carbon_dioxide"] = random.uniform(0.1, 0.3)
            byproducts["water"] = random.uniform(0.05, 0.15)
            byproducts["heat"] = random.uniform(0.1, 0.2)
            
        elif process_type == MetabolicProcess.CHEMOSYNTHESIS:
            byproducts["sulfur_compounds"] = random.uniform(0.05, 0.2)
            byproducts["mineral_deposits"] = random.uniform(0.01, 0.1)
            
        elif process_type == MetabolicProcess.RADIOSYNTHESIS:
            byproducts["radiation_particles"] = random.uniform(0.01, 0.05)
            byproducts["heavy_elements"] = random.uniform(0.005, 0.02)
            
        elif process_type == MetabolicProcess.SYMBOLIC_ABSORPTION:
            byproducts["meaning_fragments"] = random.uniform(0.1, 0.3)
            byproducts["pattern_echoes"] = random.uniform(0.05, 0.15)
            
        elif process_type == MetabolicProcess.MOTIF_CYCLING:
            byproducts["motif_residue"] = random.uniform(0.1, 0.2)
            byproducts["thematic_essence"] = random.uniform(0.05, 0.15)
            
        # Add some random byproducts for all processes
        if random.random() < 0.3:
            byproducts["quantum_fluctuations"] = random.uniform(0.01, 0.05)
        if random.random() < 0.2:
            byproducts["emergent_patterns"] = random.uniform(0.01, 0.08)
            
        return byproducts
    
    def _initialize_metabolic_motifs(self) -> List[str]:
        """Initialize motifs associated with this metabolic system"""
        motifs = []
        
        # Add motifs based on primary process
        if self.primary_process == MetabolicProcess.PHOTOSYNTHESIS:
            motifs.append("light_transmutation")
            motifs.append("solar_dependency")
            
        elif self.primary_process == MetabolicProcess.RESPIRATION:
            motifs.append("oxygen_cycle")
            motifs.append("energy_extraction")
            
        elif self.primary_process == MetabolicProcess.CHEMOSYNTHESIS:
            motifs.append("chemical_alchemy")
            motifs.append("mineral_digestion")
            
        elif self.primary_process == MetabolicProcess.RADIOSYNTHESIS:
            motifs.append("radiation_harvester")
            motifs.append("particle_absorption")
            
        elif self.primary_process == MetabolicProcess.SYMBOLIC_ABSORPTION:
            motifs.append("meaning_processor") 
            motifs.append("pattern_consumer")
            
        elif self.primary_process == MetabolicProcess.QUANTUM_ENTANGLEMENT:
            motifs.append("quantum_harvester")
            motifs.append("probability_feeder")
            
        # Add motifs related to recursion
        if self.recursion_depth > 3:
            motifs.append("deep_recursive_metabolism")
        
        # Add motifs based on symbolic affinity
        strongest_affinity = max(self.symbolic_affinity.items(), key=lambda x: x[1])
        motifs.append(f"{strongest_affinity[0]}_affinity")
        
        return motifs
    
    def _calculate_input_requirements(self) -> Dict[str, float]:
        """Calculate resource requirements based on metabolic process"""
        requirements = {}
        
        # Base requirements by process type
        if self.primary_process == MetabolicProcess.PHOTOSYNTHESIS:
            requirements["light"] = 1.0
            requirements["water"] = 0.6
            requirements["carbon_dioxide"] = 0.4
            
        elif self.primary_process == MetabolicProcess.RESPIRATION:
            requirements["oxygen"] = 0.8
            requirements["glucose"] = 0.7
            
        elif self.primary_process == MetabolicProcess.CHEMOSYNTHESIS:
            requirements["hydrogen_sulfide"] = 0.6
            requirements["carbon_dioxide"] = 0.4
            requirements["minerals"] = 0.5
            
        elif self.primary_process == MetabolicProcess.RADIOSYNTHESIS:
            requirements["radiation"] = 0.8
            requirements["heavy_elements"] = 0.3
            
        elif self.primary_process == MetabolicProcess.SYMBOLIC_ABSORPTION:
            requirements["meaning_density"] = 0.7
            requirements["pattern_complexity"] = 0.5
            
        elif self.primary_process == MetabolicProcess.MOTIF_CYCLING:
            requirements["motif_presence"] = 0.9
            requirements["narrative_flow"] = 0.4
        
        # Add symbolic requirements
        strongest_symbols = sorted(self.symbolic_affinity.items(), key=lambda x: x[1], reverse=True)[:3]
        for symbol, affinity in strongest_symbols:
            requirements[f"symbolic_{symbol}"] = affinity * 0.5
            
        return requirements


```python
class RecursiveMetabolism:
    """
    Models the symbiotic, fractal energy exchange system in biological entities.
    
    RecursiveMetabolism implements a complex bioenergetic system that functions across
    multiple scales simultaneously, from molecular to ecosystem levels. It manages the
    transformation of various energy forms between symbolic and physical states,
    creating recursive cycles that allow for emergent properties at higher levels.
    
    The system features:
    - Multi-scale energy cycling across different levels of organization
    - Symbolic/physical energy transduction mechanisms
    - Recursive feedback loops between metabolic processes
    - Adaptive pathways that evolve based on environmental conditions
    - Motif integration for symbolic meaning in biological processes
    """
    
    def __init__(self, 
                 owner_entity: Any,
                 base_efficiency: float = 0.7, 
                 primary_process: MetabolicProcess = None, 
                 secondary_processes: List[MetabolicProcess] = None,
                 recursion_depth: int = 3,
                 symbolic_affinity: Dict[str, float] = None):
        """
        Initialize a recursive metabolism system.
        
        Args:
            owner_entity: The entity (planet, flora, fauna) containing this metabolism
            base_efficiency: Base efficiency of energy conversion (0.0-1.0)
            primary_process: Main metabolic process
            secondary_processes: Additional metabolic processes
            recursion_depth: How many levels of recursive processing to maintain
            symbolic_affinity: Affinity for different symbolic energies
        """
        self.owner = owner_entity
        self.base_efficiency = base_efficiency
        self.primary_process = primary_process or self._select_default_process()
        self.secondary_processes = secondary_processes or []
        self.recursion_depth = recursion_depth
        self.symbolic_affinity = symbolic_affinity or self._generate_symbolic_affinity()
        
        # Energy storage at different recursion levels
        self.energy_pools = {
            level: {
                "physical": 50.0 * (0.7 ** level),  # Physical energy decreases with level
                "symbolic": 30.0 * (1.3 ** level),  # Symbolic energy increases with level
                "capacity": 100.0 * (1.1 ** level)  # Capacity increases with level
            } for level in range(recursion_depth + 1)
        }
        
        # Process efficiencies vary by level
        self.process_efficiencies = {
            process: {
                level: base_efficiency * (0.9 + 0.2 * random.random()) * (1.0 + (0.1 * level if process == MetabolicProcess.SYMBOLIC_ABSORPTION else 0)) 
                for level in range(recursion_depth + 1)
            } for process in [self.primary_process] + self.secondary_processes
        }
        
        # Byproducts from metabolism
        self.byproducts = {}
        
        # Energy transfer rates between levels (upward and downward)
        self.upward_transfer_rate = 0.15  # % of energy moving to higher level
        self.downward_transfer_rate = 0.25  # % of energy moving to lower level
        
        # Adaptation metrics
        self.adaptation_pressure = 0.0
        self.adaptation_threshold = 0.7
        self.adaptation_history = deque(maxlen=10)
        
        # Symbiotic relationships
        self.symbiotic_links = {}
        
        # Create recursive subprocesses with their own cycles
        self.subprocesses = self._initialize_subprocesses()
        
        # Integration with motif system
        self.metabolic_motifs = self._initialize_metabolic_motifs()
        
        # Cycle parameters
        self.cycle_count = 0
        self.cycle_length = 12  # Steps per complete metabolic cycle
        self.cycle_phase = 0.0  # Position in current cycle (0.0-1.0)
        
        # Energy needs based on primary process
        self.input_requirements = self._calculate_input_requirements()
        
        # Stats tracking
        self.total_energy_produced = 0
        self.total_energy_consumed = 0
        
    def _select_default_process(self) -> MetabolicProcess:
        """Select an appropriate default metabolic process based on owner entity"""
        if hasattr(self.owner, 'entity_type'):
            if self.owner.entity_type == EntityType.PLANET:
                return MetabolicProcess.RADIOSYNTHESIS
            # For flora/fauna determined by inherited characteristics
            elif hasattr(self.owner, 'is_photosynthetic') and self.owner.is_photosynthetic:
                return MetabolicProcess.PHOTOSYNTHESIS
            elif hasattr(self.owner, 'is_chemosynthetic') and self.owner.is_chemosynthetic:
                return MetabolicProcess.CHEMOSYNTHESIS
            elif hasattr(self.owner, 'is_symbolic_processor') and self.owner.is_symbolic_processor:
                return MetabolicProcess.SYMBOLIC_ABSORPTION
        
        # Default based on planet conditions if available
        if hasattr(self.owner, 'planet') and self.owner.planet:
            if self.owner.planet.get_trait('solar_radiation', 0.5) > 0.6:
                return MetabolicProcess.PHOTOSYNTHESIS
            elif self.owner.planet.get_trait('chemical_energy', 0.3) > 0.7:
                return MetabolicProcess.CHEMOSYNTHESIS
        
        # Fallback
        return random.choice([
            MetabolicProcess.PHOTOSYNTHESIS,
            MetabolicProcess.RESPIRATION,
            MetabolicProcess.CHEMOSYNTHESIS
        ])
    
    def _generate_symbolic_affinity(self) -> Dict[str, float]:
        """Generate affinities for different symbolic energy types"""
        # Base symbolic energy types
        affinities = {
            "light": random.uniform(0.1, 0.9),
            "darkness": random.uniform(0.1, 0.9),
            "order": random.uniform(0.1, 0.9),
            "chaos": random.uniform(0.1, 0.9),
            "creation": random.uniform(0.1, 0.9),
            "destruction": random.uniform(0.1, 0.9),
            "transcendence": random.uniform(0.1, 0.9),
            "immanence": random.uniform(0.1, 0.9),
            "complexity": random.uniform(0.1, 0.9),
            "simplicity": random.uniform(0.1, 0.9)
        }
        
        # If owner has motifs, incorporate those into affinities
        if hasattr(self.owner, 'motifs'):
            for motif in self.owner.motifs:
                # Extract key concepts from motif name
                for symbol in affinities.keys():
                    if symbol in motif:
                        affinities[symbol] += 0.2
        
        # Normalize to ensure sum is reasonable
        total = sum(affinities.values())
        if total > 0:
            affinities = {k: v / total * len(affinities) for k, v in affinities.items()}
        
        return affinities
    
    def _initialize_subprocesses(self) -> Dict[str, Dict]:
        """Initialize recursive metabolic subprocesses"""
        subprocesses = {}
        
        # Create subprocess for each level except the highest
        for level in range(self.recursion_depth):
            process_count = 2 + level  # More subprocesses at deeper levels
            
            level_processes = {}
            for i in range(process_count):
                process_type = random.choice([p for p in MetabolicProcess])
                
                # Each subprocess has its own cycle and efficiency
                level_processes[f"subprocess_{level}_{i}"] = {
                    "type": process_type,
                    "efficiency": random.uniform(0.5, 0.9),
                    "cycle_offset": random.uniform(0, 1.0),
                    "cycle_length": 6 + random.randint(0, 12),
                    "input_ratio": random.uniform(0.1, 0.5),
                    "output_ratio": random.uniform(0.1, 0.5),
                    "symbolic_ratio": random.uniform(0.2, 0.8),
                    "byproducts": self._generate_subprocess_byproducts(process_type)
                }
            
            subprocesses.update(level_processes)
                
        return subprocesses
    
    def _generate_subprocess_byproducts(self, process_type: MetabolicProcess) -> Dict[str, float]:
        """Generate appropriate byproducts for a given metabolic process"""
        byproducts = {}
        
        # Different processes produce different byproducts
        if process_type == MetabolicProcess.PHOTOSYNTHESIS:
            byproducts["oxygen"] = random.uniform(0.1, 0.3)
            byproducts["glucose"] = random.uniform(0.05, 0.15)
            byproducts["symbolic_light"] = random.uniform(0.01, 0.1) 
            
        elif process_type == MetabolicProcess.RESPIRATION:
            byproducts["carbon_dioxide"] = random.uniform(0.1, 0.3)
            byproducts["water"] = random.uniform(0.05, 0.15)
            byproducts["heat"] = random.uniform(0.1, 0.2)
            
        elif process_type == MetabolicProcess.CHEMOSYNTHESIS:
            byproducts["sulfur_compounds"] = random.uniform(0.05, 0.2)
            byproducts["mineral_deposits"] = random.uniform(0.01, 0.1)
            
        elif process_type == MetabolicProcess.RADIOSYNTHESIS:
            byproducts["radiation_particles"] = random.uniform(0.01, 0.05)
            byproducts["heavy_elements"] = random.uniform(0.005, 0.02)
            
        elif process_type == MetabolicProcess.SYMBOLIC_ABSORPTION:
            byproducts["meaning_fragments"] = random.uniform(0.1, 0.3)
            byproducts["pattern_echoes"] = random.uniform(0.05, 0.15)
            
        elif process_type == MetabolicProcess.MOTIF_CYCLING:
            byproducts["motif_residue"] = random.uniform(0.1, 0.2)
            byproducts["thematic_essence"] = random.uniform(0.05, 0.15)
            
        # Add some random byproducts for all processes
        if random.random() < 0.3:
            byproducts["quantum_fluctuations"] = random.uniform(0.01, 0.05)
        if random.random() < 0.2:
            byproducts["emergent_patterns"] = random.uniform(0.01, 0.08)
            
        return byproducts
    
    def _initialize_metabolic_motifs(self) -> List[str]:
        """Initialize motifs associated with this metabolic system"""
        motifs = []
        
        # Add motifs based on primary process
        if self.primary_process == MetabolicProcess.PHOTOSYNTHESIS:
            motifs.append("light_transmutation")
            motifs.append("solar_dependency")
            
        elif self.primary_process == MetabolicProcess.RESPIRATION:
            motifs.append("oxygen_cycle")
            motifs.append("energy_extraction")
            
        elif self.primary_process == MetabolicProcess.CHEMOSYNTHESIS:
            motifs.append("chemical_alchemy")
            motifs.append("mineral_digestion")
            
        elif self.primary_process == MetabolicProcess.RADIOSYNTHESIS:
            motifs.append("radiation_harvester")
            motifs.append("particle_absorption")
            
        elif self.primary_process == MetabolicProcess.SYMBOLIC_ABSORPTION:
            motifs.append("meaning_processor") 
            motifs.append("pattern_consumer")
            
        elif self.primary_process == MetabolicProcess.QUANTUM_ENTANGLEMENT:
            motifs.append("quantum_harvester")
            motifs.append("probability_feeder")
            
        # Add motifs related to recursion
        if self.recursion_depth > 3:
            motifs.append("deep_recursive_metabolism")
        
        # Add motifs based on symbolic affinity
        strongest_affinity = max(self.symbolic_affinity.items(), key=lambda x: x[1])
        motifs.append(f"{strongest_affinity[0]}_affinity")
        
        return motifs
    
    def _calculate_input_requirements(self) -> Dict[str, float]:
        """Calculate resource requirements based on metabolic process"""
        requirements = {}
        
        # Base requirements by process type
        if self.primary_process == MetabolicProcess.PHOTOSYNTHESIS:
            requirements["light"] = 1.0
            requirements["water"] = 0.6
            requirements["carbon_dioxide"] = 0.4
            
        elif self.primary_process == MetabolicProcess.RESPIRATION:
            requirements["oxygen"] = 0.8
            requirements["glucose"] = 0.7
            
        elif self.primary_process == MetabolicProcess.CHEMOSYNTHESIS:
            requirements["hydrogen_sulfide"] = 0.6
            requirements["carbon_dioxide"] = 0.4
            requirements["minerals"] = 0.5
            
        elif self.primary_process == MetabolicProcess.RADIOSYNTHESIS:
            requirements["radiation"] = 0.8
            requirements["heavy_elements"] = 0.3
            
        elif self.primary_process == MetabolicProcess.SYMBOLIC_ABSORPTION:
            requirements["meaning_density"] = 0.7
            requirements["pattern_complexity"] = 0.5
            
        elif self.primary_process == MetabolicProcess.MOTIF_CYCLING:
            requirements["motif_presence"] = 0.9
            requirements["narrative_flow"] = 0.4
        
        # Add symbolic requirements
        strongest_symbols = sorted(self.symbolic_affinity.items(), key=lambda x: x[1], reverse=True)[:3]
        for symbol, affinity in strongest_symbols:
            requirements[f"symbolic_{symbol}"] = affinity * 0.5
            
        return requirements
    
    def process_cycle(self, environmental_inputs: Dict[str, float], time_delta: float = 1.0) -> Dict[str, Any]:
        """
        Process a full metabolic cycle with environmental inputs.
        
        Args:
            environmental_inputs: Dictionary of available resources
            time_delta: Time progression factor
            
        Returns:
            Dict containing metabolic results, byproducts, and events
        """
        # Advance cycle phase
        phase_increment = time_delta / self.cycle_length
        self.cycle_phase = (self.cycle_phase + phase_increment) % 1.0
        
        # Complete cycles counter
        if self.cycle_phase < phase_increment:
            self.cycle_count += 1
        
        # Process each level starting from the lowest (most physical)
        results = {
            "energy_produced": 0,
            "symbolic_energy_produced": 0,
            "byproducts": {},
            "adaptation_events": [],
            "efficiency_change": 0,
            "energy_state": {}
        }
        
        # Run primary and secondary processes at each level
        for level in range(self.recursion_depth + 1):
            level_results = self._process_level(level, environmental_inputs, time_delta)
            
            # Accumulate results
            results["energy_produced"] += level_results["physical_energy"]
            results["symbolic_energy_produced"] += level_results["symbolic_energy"]
            
            # Merge byproducts
            for byproduct, amount in level_results["byproducts"].items():
                if byproduct in results["byproducts"]:
                    results["byproducts"][byproduct] += amount
                else:
                    results["byproducts"][byproduct] = amount
            
            # Add any adaptation events
            results["adaptation_events"].extend(level_results["adaptation_events"])
        
        # Process energy transfer between levels
        self._transfer_energy_between_levels()
        
        # Check for adaptation pressure and possibly evolve
        adaptation_result = self._check_adaptation(environmental_inputs, time_delta)
        results["adaptation_events"].extend(adaptation_result["events"])
        results["efficiency_change"] = adaptation_result["efficiency_change"]
        
        # Run symbiotic processes if any exist
        if self.symbiotic_links:
            symbiotic_result = self._process_symbiotic_relationships(time_delta)
            results["symbiotic_energy"] = symbiotic_result["energy"]
            results["adaptation_events"].extend(symbiotic_result["events"])
        
        # Update total stats
        self.total_energy_produced += results["energy_produced"]
        
        # Record current energy state
        results["energy_state"] = {level: pool.copy() for level, pool in self.energy_pools.items()}
        
        return results
    
    def _process_level(self, level: int, environmental_inputs: Dict[str, float], time_delta: float) -> Dict[str, Any]:
        """Process metabolism at a specific recursive level"""
        level_results = {
            "physical_energy": 0,
            "symbolic_energy": 0,
            "byproducts": {},
            "adaptation_events": []
        }
        
        # Adjust inputs based on level (higher levels use more symbolic resources)
        adjusted_inputs = self._adjust_inputs_for_level(environmental_inputs, level)
        
        # Calculate current efficiency based on cycle phase
        phase_efficiency = self._calculate_phase_efficiency(level)
        
        # Process primary metabolism
        primary_result = self._process_single_metabolism(
            self.primary_process, 
            level,
            adjusted_inputs,
            phase_efficiency,
            time_delta
        )
        
        # Add to level results
        level_results["physical_energy"] += primary_result["physical_energy"]
        level_results["symbolic_energy"] += primary_result["symbolic_energy"]
        
        # Process byproducts
        for byproduct, amount in primary_result["byproducts"].items():
            if byproduct in level_results["byproducts"]:
                level_results["byproducts"][byproduct] += amount
            else:
                level_results["byproducts"][byproduct] = amount
        
        # Process secondary metabolisms
        """
        Initialize a recursive metabolism system.
        
        Args:
            owner_entity: The entity (planet, flora, fauna) containing this metabolism
            base_efficiency: Base efficiency of energy conversion (0.0-1.0)
            primary_process: Main metabolic process
            secondary_processes: Additional metabolic processes
            recursion_depth: How many levels of recursive processing to maintain
            symbolic_affinity: Affinity for different symbolic energies
        """
        self.owner = owner_entity
        self.base_efficiency = base_efficiency
        self.primary_process = primary_process or self._select_default_process()
        self.secondary_processes = secondary_processes or []
        self.recursion_depth = recursion_depth
        self.symbolic_affinity = symbolic_affinity or self._generate_symbolic_affinity()
        
        # Energy storage at different recursion levels
        self.energy_pools = {
            level: {
                "physical": 50.0 * (0.7 ** level),  # Physical energy decreases with level
                "symbolic": 30.0 * (1.3 ** level),  # Symbolic energy increases with level
                "capacity": 100.0 * (1.1 ** level)  # Capacity increases with level
            } for level in range(recursion_depth + 1)
        }
        
        # Process efficiencies vary by level
        self.process_efficiencies = {
            process: {
                level: base_efficiency * (0.9 + 0.2 * random.random()) * (1.0 + (0.1 * level if process == MetabolicProcess.SYMBOLIC_ABSORPTION else 0)) 
                for level in range(recursion_depth + 1)
            } for process in [self.primary_process] + self.secondary_processes
        }
        
        # Byproducts from metabolism
        self.byproducts = {}
        
        # Energy transfer rates between levels (upward and downward)
        self.upward_transfer_rate = 0.15  # % of energy moving to higher level
        self.downward_transfer_rate = 0.25  # % of energy moving to lower level
        
        # Adaptation metrics
        self.adaptation_pressure = 0.0
        self.adaptation_threshold = 0.7
        self.adaptation_history = deque(maxlen=10)
        
        # Symbiotic relationships
        self.symbiotic_links = {}
        
        # Create recursive subprocesses with their own cycles
        self.subprocesses = self._initialize_subprocesses()
        
        # Integration with motif system
        self.metabolic_motifs = self._initialize_metabolic_motifs()
        
        # Cycle parameters
        self.cycle_count = 0
        self.cycle_length = 12  # Steps per complete metabolic cycle
        self.cycle_phase = 0.0  # Position in current cycle (0.0-1.0)
        
        # Energy needs based on primary process
        self.input_requirements = self._calculate_input_requirements()
        
        # Stats tracking
        self.total_energy_produced = 0
        self.total_energy_consumed = 0
        
    def _select_default_process(self) -> MetabolicProcess:
        """Select an appropriate default metabolic process based on owner entity"""
        if hasattr(self.owner, 'entity_type'):
            if self.owner.entity_type == EntityType.PLANET:
                return MetabolicProcess.RADIOSYNTHESIS
            # For flora/fauna determined by inherited characteristics
            elif hasattr(self.owner, 'is_photosynthetic') and self.owner.is_photosynthetic:
                return MetabolicProcess.PHOTOSYNTHESIS
            elif hasattr(self.owner, 'is_chemosynthetic') and self.owner.is_chemosynthetic:
                return MetabolicProcess.CHEMOSYNTHESIS
            elif hasattr(self.owner, 'is_symbolic_processor') and self.owner.is_symbolic_processor:
                return MetabolicProcess.SYMBOLIC_ABSORPTION
        
        # Default based on planet conditions if available
        if hasattr(self.owner, 'planet') and self.owner.planet:
            if self.owner.planet.get_trait('solar_radiation', 0.5) > 0.6:
                return MetabolicProcess.PHOTOSYNTHESIS
            elif self.owner.planet.get_trait('chemical_energy', 0.3) > 0.7:
                return MetabolicProcess.CHEMOSYNTHESIS
        
        # Fallback
        return random.choice([
            MetabolicProcess.PHOTOSYNTHESIS,
            MetabolicProcess.RESPIRATION,
            MetabolicProcess.CHEMOSYNTHESIS
        ])
    
    def _generate_symbolic_affinity(self) -> Dict[str, float]:
        """Generate affinities for different symbolic energy types"""
        # Base symbolic energy types
        affinities = {
            "light": random.uniform(0.1, 0.9),
            "darkness": random.uniform(0.1, 0.9),
            "order": random.uniform(0.1, 0.9),
            "chaos": random.uniform(0.1, 0.9),
            "creation": random.uniform(0.1, 0.9),
            "destruction": random.uniform(0.1, 0.9),
            "transcendence": random.uniform(0.1, 0.9),
            "immanence": random.uniform(0.1, 0.9),
            "complexity": random.uniform(0.1, 0.9),
            "simplicity": random.uniform(0.1, 0.9)
        }
        
        # If owner has motifs, incorporate those into affinities
        if hasattr(self.owner, 'motifs'):
            for motif in self.owner.motifs:
                # Extract key concepts from motif name
                for symbol in affinities.keys():
                    if symbol in motif:
                        affinities[symbol] += 0.2
        
        # Normalize to ensure sum is reasonable
        total = sum(affinities.values())
        if total > 0:
            affinities = {k: v / total * len(affinities) for k, v in affinities.items()}
        
        return affinities
    
    def _initialize_subprocesses(self) -> Dict[str, Dict]:
        """Initialize recursive metabolic subprocesses"""
        subprocesses = {}
        
        # Create subprocess for each level except the highest
        for level in range(self.recursion_depth):
            process_count = 2 + level  # More subprocesses at deeper levels
            
            level_processes = {}
            for i in range(process_count):
                process_type = random.choice([p for p in MetabolicProcess])
                
                # Each subprocess has its own cycle and efficiency
                level_processes[f"subprocess_{level}_{i}"] = {
                    "type": process_type,
                    "efficiency": random.uniform(0.5, 0.9),
                    "cycle_offset": random.uniform(0, 1.0),
                    "cycle_length": 6 + random.randint(0, 12),
                    "input_ratio": random.uniform(0.1, 0.5),
                    "output_ratio": random.uniform(0.1, 0.5),
                    "symbolic_ratio": random.uniform(0.2, 0.8),
                    "byproducts": self._generate_subprocess_byproducts(process_type)
                }
                
        return subprocesses
    
    def _generate_subprocess_byproducts(self, process_type: MetabolicProcess) -> Dict[str, float]:
        """Generate appropriate byproducts for a given metabolic process"""
        byproducts = {}
        
        # Different processes produce different byproducts
        if process_type == MetabolicProcess.PHOTOSYNTHESIS:
            byproducts["oxygen"] = random.uniform(0.1, 0.3)
            byproducts["glucose"] = random.uniform(0.05, 0.15)
            byproducts["symbolic_light"] = random.uniform(0.01, 0.1) 
            
        elif process_type == MetabolicProcess.RESPIRATION:
            byproducts["carbon_dioxide"] = random.uniform(0.1, 0.3)
            byproducts["water"] = random.uniform(0.05, 0.15)
            byproducts["heat"] = random.uniform(0.1, 0.2)
            
        elif process_type == MetabolicProcess.CHEMOSYNTHESIS:
            byproducts["sulfur_compounds"] = random.uniform(0.05, 0.2)
            byproducts["mineral_deposits"] = random.uniform(0.01, 0.1)
            
        elif process_type == MetabolicProcess.RADIOSYNTHESIS:
            byproducts["radiation_particles"] = random.uniform(0.01, 0.05)
            byproducts["heavy_elements"] = random.uniform(0.005, 0.02)
            
        elif process_type == MetabolicProcess.SYMBOLIC_ABSORPTION:
            byproducts["meaning_fragments"] = random.uniform(0.1, 0.3)
            byproducts["pattern_echoes"] = random.uniform(0.05, 0.15)
            
        elif process_type == MetabolicProcess.MOTIF_CYCLING:
            byproducts["motif_residue"] = random.uniform(0.1, 0.2)
            byproducts["thematic_essence"] = random.uniform(0.05, 0.15)
            
        # Add some random byproducts for all processes
        if random.random() < 0.3:
            byproducts["quantum_fluctuations"] = random.uniform(0.01, 0.05)
        if random.random() < 0.2:
            byproducts["emergent_patterns"] = random.uniform(0.01, 0.08)
            
        return byproducts
    
    def _initialize_metabolic_motifs(self) -> List[str]:
        """Initialize motifs associated with this metabolic system"""
        motifs = []
        
        # Add motifs based on primary process
        if self.primary_process == MetabolicProcess.PHOTOSYNTHESIS:
            motifs.append("light_transmutation")
            motifs.append("solar_dependency")
            
        elif self.primary_process == MetabolicProcess.RESPIRATION:
            motifs.append("oxygen_cycle")
            motifs.append("energy_extraction")
            
        elif self.primary_process == MetabolicProcess.CHEMOSYNTHESIS:
            motifs.append("chemical_alchemy")
            motifs.append("mineral_digestion")
            
        elif self.primary_process == MetabolicProcess.RADIOSYNTHESIS:
            motifs.append("radiation_harvester")
            motifs.append("particle_absorption")
            
        elif self.primary_process == MetabolicProcess.SYMBOLIC_ABSORPTION:
            motifs.append("meaning_processor") 
            motifs.append("pattern_consumer")
            
        elif self.primary_process == MetabolicProcess.QUANTUM_ENTANGLEMENT:
            motifs.append("quantum_harvester")
            motifs.append("probability_feeder")
            
        # Add motifs related to recursion
        if self.recursion_depth > 3:
            motifs.append("deep_recursive_metabolism")
        
        # Add motifs based on symbolic affinity
        strongest_affinity = max(self.symbolic_affinity.items(), key=lambda x: x[1])
        motifs.append(f"{strongest_affinity[0]}_affinity")
        
        return motifs
    
    def _calculate_input_requirements(self) -> Dict[str, float]:
        """Calculate resource requirements based on metabolic process"""
        requirements = {}
        
        # Base requirements by process type
        if self.primary_process == MetabolicProcess.PHOTOSYNTHESIS:
            requirements["light"] = 1.0
            requirements["water"] = 0.6
            requirements["carbon_dioxide"] = 0.4
            
        elif self.primary_process == MetabolicProcess.RESPIRATION:
            requirements["oxygen"] = 0.8
            requirements["glucose"] = 0.7
            
        elif self.primary_process == MetabolicProcess.CHEMOSYNTHESIS:
            requirements["hydrogen_sulfide"] = 0.6
            requirements["carbon_dioxide"] = 0.4
            requirements["minerals"] = 0.5
            
        elif self.primary_process == MetabolicProcess.RADIOSYNTHESIS:
            requirements["radiation"] = 0.8
            requirements["heavy_elements"] = 0.3
            
        elif self.primary_process == MetabolicProcess.SYMBOLIC_ABSORPTION:
            requirements["meaning_density"] = 0.7
            requirements["pattern_complexity"] = 0.5
            
        elif self.primary_process == MetabolicProcess.MOTIF_CYCLING:
            requirements["motif_presence"] = 0.9
            requirements["narrative_flow"] = 0.4
        
        # Add symbolic requirements
        strongest_symbols = sorted(self.symbolic_affinity.items(), key=lambda x: x[1], reverse=True)[:3]
        for symbol, affinity in strongest_symbols:
            requirements[f"symbolic_{symbol}"] = affinity * 0.5
            
        return requirements
    
    def process_cycle(self, environmental_inputs: Dict[str, float], time_delta: float = 1.0) -> Dict[str, Any]:
        """
        Process a full metabolic cycle with environmental inputs.
        
        Args:
            environmental_inputs: Dictionary of available resources
            time_delta: Time progression factor
            
        Returns:
            Dict containing metabolic results, byproducts, and events
        """
        # Advance cycle phase
        phase_increment = time_delta / self.cycle_length
        self.cycle_phase = (self.cycle_phase + phase_increment) % 1.0
        
        # Complete cycles counter
        if self.cycle_phase < phase_increment:
            self.cycle_count += 1
        
        # Process each level starting from the lowest (most physical)
        results = {
            "energy_produced": 0,
            "symbolic_energy_produced": 0,
            "byproducts": {},
            "adaptation_events": [],
            "efficiency_change": 0,
            "energy_state": {}
        }
        
        # Run primary and secondary processes at each level
        for level in range(self.recursion_depth + 1):
            level_results = self._process_level(level, environmental_inputs, time_delta)
            
            # Accumulate results
            results["energy_produced"] += level_results["physical_energy"]
            results["symbolic_energy_produced"] += level_results["symbolic_energy"]
            
            # Merge byproducts
            for byproduct, amount in level_results["byproducts"].items():
                if byproduct in results["byproducts"]:
                    results["byproducts"][byproduct] += amount
                else:
                    results["byproducts"][byproduct] = amount
            
            # Add any adaptation events
            results["adaptation_events"].extend(level_results["adaptation_events"])
        
        # Process energy transfer between levels
        self._transfer_energy_between_levels()
        
        # Check for adaptation pressure and possibly evolve
        adaptation_result = self._check_adaptation(environmental_inputs, time_delta)
        results["adaptation_events"].extend(adaptation_result["events"])
        results["efficiency_change"] = adaptation_result["efficiency_change"]
        
        # Run symbiotic processes if any exist
        if self.symbiotic_links:
            symbiotic_result = self._process_symbiotic_relationships(time_delta)
            results["symbiotic_energy"] = symbiotic_result["energy"]
            results["adaptation_events"].extend(symbiotic_result["events"])
        
        # Update total stats
        self.total_energy_produced += results["energy_produced"]
        
        # Record current energy state
        results["energy_state"] = {level: pool.copy() for level, pool in self.energy_pools.items()}
        
        return results
    
    def _process_level(self, level: int, environmental_inputs: Dict[str, float], time_delta: float) -> Dict[str, Any]:
        """Process metabolism at a specific recursive level"""
        level_results = {
            "physical_energy": 0,
            "symbolic_energy": 0,
            "byproducts": {},
            "adaptation_events": []
        }
        
        # Adjust inputs based on level (higher levels use more symbolic resources)
        adjusted_inputs = self._adjust_inputs_for_level(environmental_inputs, level)
        
        # Calculate current efficiency based on cycle phase
        phase_efficiency = self._calculate_phase_efficiency(level)
        
        # Process primary metabolism
        primary_result = self._process_single_metabolism(

class Storm:
    """
    Represents a weather system that carries symbolic content and affects entities
    and landscapes within its influence.
    """
    
    def __init__(self, center, storm_type, radius, intensity, symbolic_content):
        self.center = center
        self.storm_type = storm_type
        self.radius = radius
        self.intensity = intensity
        self.symbolic_content = symbolic_content
        self.age = 0
        self.trajectory = []
        self.affected_regions = set()
        self.emotional_signature = self._derive_emotional_signature()
        self.dissolution_rate = 0.05
        
    def update(self, world_state):
        """Update the storm's position, intensity, and effects."""
        self.age += 1
        
        # Natural dissipation over time
        self.intensity *= (1 - self.dissolution_rate)
        
        # Movement based on prevailing currents
        self._move(world_state.wind_patterns)
        
        # Record the current position in trajectory
        self.trajectory.append(self.center)
        
        # Apply effects to regions within storm radius
        self._apply_effects(world_state)
        
        # Return True if the storm should persist, False if it should dissipate
        return self.intensity > 0.1
    
    def _move(self, wind_patterns):
        """Move the storm based on prevailing winds."""
        x, y = self.center
        
        # Get wind vector at current position
        wind_x, wind_y = wind_patterns.get_vector(x, y)
        
        # Add some randomness to movement
        wind_x += (random.random() - 0.5) * 0.5
        wind_y += (random.random() - 0.5) * 0.5
        
        # Scale movement by intensity (stronger storms move faster)
        movement_scale = 0.5 + (self.intensity * 0.1)
        
        # Calculate new position
        new_x = x + (wind_x * movement_scale)
        new_y = y + (wind_y * movement_scale)
        
        # Ensure within world bounds
        new_x = max(0, min(new_x, WORLD_SIZE - 1))
        new_y = max(0, min(new_y, WORLD_SIZE - 1))
        
        self.center = (new_x, new_y)
    
    def _apply_effects(self, world_state):
        """Apply the storm's effects to regions within its influence."""
        x, y = self.center
        
        for i in range(max(0, int(x - self.radius)), min(WORLD_SIZE, int(x + self.radius + 1))):
            for j in range(max(0, int(y - self.radius)), min(WORLD_SIZE, int(y + self.radius + 1))):
                # Calculate distance from storm center
                distance = math.sqrt((i - x)**2 + (j - y)**2)
                
                if distance <= self.radius:
                    # Calculate intensity at this point (decreases with distance)
                    local_intensity = self.intensity * (1 - (distance / self.radius))
                    
                    # Record affected region
                    self.affected_regions.add((i, j))
                    
                    # Apply symbolic effects
                    for symbol, potency in self.symbolic_content.items():
                        effect_strength = local_intensity * potency
                        world_state.apply_symbolic_effect((i, j), symbol, effect_strength)
    
    def _derive_emotional_signature(self):
        """Derive the emotional signature based on storm type and content."""
        base_emotions = {
            "hurricane": {"fear": 0.8, "awe": 0.7, "transformation": 0.9},
            "thunderstorm": {"shock": 0.6, "clarity": 0.5, "tension": 0.7},
            "fog": {"confusion": 0.8, "mystery": 0.9, "introspection": 0.6},
            "rain": {"melancholy": 0.4, "renewal": 0.7, "calm": 0.5}
        }
        
        # Start with base emotions for this storm type
        emotional_signature = base_emotions.get(self.storm_type, {})
        
        # Modify based on symbolic content
        for symbol in self.symbolic_content:
            if "fear" in symbol.lower():
                emotional_signature["fear"] = emotional_signature.get("fear", 0) + 0.2
            if "renewal" in symbol.lower():
                emotional_signature["hope"] = emotional_signature.get("hope", 0) + 0.3
            if "chaos" in symbol.lower():
                emotional_signature["confusion"] = emotional_signature.get("confusion", 0) + 0.4
                
        return emotional_signature


class SeasonalCycle:
    """
    Controls the progression of seasonal patterns that affect weather, terrain,
    and entity behavior throughout the year.
    """
    
    def __init__(self, starting_season="spring", cycle_length=120):
        """
        Initialize the seasonal cycle.
        
        Args:
            starting_season (str): The season to start with
            cycle_length (int): Total time steps in a complete year
        """
        self.seasons = ["winter", "spring", "summer", "autumn"]
        self.season_durations = {
            "winter": cycle_length // 4,
            "spring": cycle_length // 4,
            "summer": cycle_length // 4,
            "autumn": cycle_length // 4
        }
        self.current_season = starting_season
        self.time_in_season = 0
        self.cycle_length = cycle_length
        self.year_count = 0
        self.seasonal_effects = self._initialize_seasonal_effects()
        self.transition_thresholds = self._calculate_transition_thresholds()
        self.seasonal_events = []
    
    def advance(self, accelerate=False):
        """
        Advances the seasonal cycle by one time step.
        
        Args:
            accelerate (bool): Whether to speed up seasonal progression
        
        Returns:
            bool: True if the season changed, False otherwise
        """
        self.time_in_season += 1 + (1 if accelerate else 0)
        season_changed = False
        
        # Check if we need to transition to the next season
        if self.time_in_season >= self.season_durations[self.current_season]:
            current_idx = self.seasons.index(self.current_season)
            next_idx = (current_idx + 1) % len(self.seasons)
            self.current_season = self.seasons[next_idx]
            self.time_in_season = 0
            season_changed = True
            
            # Check if we've completed a full year
            if next_idx == 0:
                self.year_count += 1
        
        # Generate seasonal events
        if random.random() < 0.1 or season_changed:
            self._generate_seasonal_event()
            
        return season_changed
    
    def get_seasonal_modifier(self, aspect):
        """
        Gets the modifier for a specific aspect based on current season.
        
        Args:
            aspect (str): The aspect to get a modifier for (e.g., 'temperature')
            
        Returns:
            float: The modifier value
        """
        base_modifier = self.seasonal_effects[self.current_season].get(aspect, 1.0)
        
        # Calculate where we are in the season (0.0 to 1.0)
        progress = self.time_in_season / self.season_durations[self.current_season]
        
        # Get next season for interpolation
        current_idx = self.seasons.index(self.current_season)
        next_idx = (current_idx + 1) % len(self.seasons)
        next_season = self.seasons[next_idx]
        
        # Get modifier for next season
        next_modifier = self.seasonal_effects[next_season].get(aspect, 1.0)
        
        # Smooth transition near season boundaries
        if progress > 0.8:
            transition_weight = (progress - 0.8) / 0.2
            return base_modifier * (1 - transition_weight) + next_modifier * transition_weight
        else:
            return base_modifier
    
    def get_diffusion_modifier(self):
        """Returns the seasonal modifier for motif diffusion."""
        return self.get_seasonal_modifier("diffusion_rate")
    
    def get_current_season_info(self):
        """
        Returns detailed information about the current season state.
        
        Returns:
            dict: Information about the current seasonal state
        """
        progress = self.time_in_season / self.season_durations[self.current_season]
        
        return {
            "season": self.current_season,
            "progress": progress,
            "year": self.year_count,
            "temperature": self.get_seasonal_modifier("temperature"),
            "precipitation": self.get_seasonal_modifier("precipitation"),
            "day_length": self.get_seasonal_modifier("day_length"),
            "growth_rate": self.get_seasonal_modifier("growth_rate"),
            "active_events": self.seasonal_events
        }
    
    def _initialize_seasonal_effects(self):
        """Initialize the effects of each season on different aspects."""
        return {
            "winter": {
                "temperature": 0.2,
                "precipitation": 0.7,
                "day_length": 0.3,
                "growth_rate": 0.1,
                "emotional_intensity": 0.5,
                "introspection": 0.9,
                "memory_clarity": 0.7,
                "diffusion_rate": 0.5
            },
            "spring": {
                "temperature": 0.6,
                "precipitation": 0.8,
                "day_length": 0.6,
                "growth_rate": 1.0,
                "emotional_intensity": 0.7,
                "creativity": 0.9,
                "renewal": 1.0,
                "diffusion_rate": 0.8
            },
            "summer": {
                "temperature": 1.0,
                "precipitation": 0.4,
                "day_length": 1.0,
                "growth_rate": 0.7,
                "emotional_intensity": 0.8,
                "energy": 1.0,
                "activity": 0.9,
                "diffusion_rate": 1.0
            },
            "autumn": {
                "temperature": 0.5,
                "precipitation": 0.6,
                "day_length": 0.5,
                "growth_rate": 0.3,
                "emotional_intensity": 0.9,
                "reflection": 0.8,
                "transition": 0.9,
                "diffusion_rate": 0.7
            }
        }
    
    def _calculate_transition_thresholds(self):
        """Calculate the time step thresholds for season transitions."""
        thresholds = {}
        cumulative = 0
        
        for season in self.seasons:
            thresholds[season] = cumulative
            cumulative += self.season_durations[season]
            
        return thresholds
    
    def _generate_seasonal_event(self):
        """Generate a random seasonal event appropriate to the current season."""
        season_events = {
            "winter": ["frost_ritual", "hibernation_trance", "snow_pattern_emergence", "solstice_inversion"],
            "spring": ["bloom_cascade", "renewal_surge", "seedling_awakening", "equinox_balance"],
            "summer": ["solar_intensity_peak", "growth_acceleration", "energy_abundance", "solstice_alignment"],
            "autumn": ["harvest_culmination", "leaf_transition_wave", "preparation_cycle", "equinox_reflection"]
        }
        
        # Select a random event for this season
        potential_events = season_events[self.current_season]
        event = random.choice(potential_events)
        
        # Give the event a random duration and intensity
        duration = random.randint(3, 10)
        intensity = 0.3 + random.random() * 0.7
        
        seasonal_event = {
            "name": event,
            "remaining_duration": duration,
            "intensity": intensity,
            "effects": self._generate_event_effects(event, intensity)
        }
        
        # Add to active events
        self.seasonal_events.append(seasonal_event)
        
        # Clean up expired events
        self.seasonal_events = [e for e in self.seasonal_events if e["remaining_duration"] > 0]
        
        # Decrease duration for all events
        for e in self.seasonal_events:
            e["remaining_duration"] -= 1
    
    def _generate_event_effects(self, event_name, intensity):
        """Generate the effects for a seasonal event."""
        # Base effects by event category
        if "frost" in event_name or "snow" in event_name:
            return {
                "temperature": -0.2 * intensity,
                "movement_rate": -0.3 * intensity,
                "crystallization": 0.5 * intensity
            }
        elif "bloom" in event_name or "growth" in event_name:
            return {
                "growth_rate": 0.3 * intensity,
                "diversity": 0.2 * intensity,
                "vitality": 0.4 * intensity
            }
        elif "solstice" in event_name or "equinox" in event_name:
            return {
                "symbolic_resonance": 0.5 * intensity,
                "temporal_fluidity": 0.3 * intensity,
                "pattern_clarity": 0.4 * intensity
            }
        else:
            return {
                "emotional_intensity": 0.2 * intensity,
                "symbolic_flow": 0.3 * intensity,
                "memory_access": 0.3 * intensity
            }


class CurrentLayer:
    """
    Represents a single depth layer in the oceanic current system.
    """
    
    def __init__(self, width, height, depth, velocity_factor=1.0):
        """
        Initialize a current layer.
        
        Args:
            width (int): Width of the world
            height (int): Height of the world
            depth (float): Depth level of this layer
            velocity_factor (float): Base speed multiplier for this layer
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.velocity_factor = velocity_factor
        self.current_vectors = np.zeros((width, height, 3))  # (x, y, depth) vectors
        self.memory_concentration = np.zeros((width, height))
        self.symbolic_content = {}
        
        # Initialize with some basic current patterns
        self._initialize_current_patterns()
    
    def get_current_vector(self, x, y):
        """Get the current vector at the specified coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.current_vectors[int(x), int(y)]
        return [0, 0, 0]
    
    def update_currents(self, temperature_gradients, salinity_map):
        """
        Update current vectors based on temperature and salinity.
        
        Args:
            temperature_gradients (ndarray): Temperature distribution
            salinity_map (ndarray): Salinity distribution
            
        Returns:
            significant_changes (list): Locations with major current changes
        """
        significant_changes = []
        
        # Calculate pressure gradients from temperature and salinity
        pressure = temperature_gradients * 0.7 + salinity_map * 0.3
        
        # Calculate gradient of pressure field
        dx, dy = np.gradient(pressure)
        
        # Scale gradients by velocity factor
        dx *= self.velocity_factor
        dy *= self.velocity_factor
        
        # Add some continuity from previous state (currents don't change instantly)
        prev_dx = self.current_vectors[:, :, 0]
        prev_dy = self.current_vectors[:, :, 1]
        
        new_dx = prev_dx * 0.7 + dx * 0.3
        new_dy = prev_dy * 0.7 + dy * 0.3
        
        # Find significant changes
        change_magnitude = np.sqrt((new_dx - prev_dx)**2 + (new_dy - prev_dy)**2)
        significant_points = np.where(change_magnitude > 0.2)
        for i in range(len(significant_points[0])):
            x, y = significant_points[0][i], significant_points[1][i]
            significant_changes.append((x, y, change_magnitude[x, y]))
        
        # Update current vectors
        self.current_vectors[:, :, 0] = new_dx
        self.current_vectors[:, :, 1] = new_dy
        
        # Vertical component is smaller (slower vertical mixing)
        vert_component = (np.random.random((self.width, self.height)) - 0.5) * 0.1
        self.current_vectors[:, :, 2] = vert_component
        
        return significant_changes
    
    def deposit_memory(self, x, y, memory_content, intensity):
        """
        Deposit memory content into currents at specified location.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            memory_content (dict): Memory information to deposit
            intensity (float): Strength of the memory imprint
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.memory_concentration[int(x), int(y)] += intensity
            
            # Create unique identifier for this memory deposit
            memory_id = str(uuid.uuid4())
            
            # Store the memory content
            self.symbolic_content[memory_id] = {
                "content": memory_content,
                "location": (x, y),
                "intensity": intensity,
                "deposit_time": 0,  # Will be incremented as time passes
                "decay_rate": 0.01
            }
    
    def retrieve_memories(self, x, y, radius=3):
        """
        Retrieve memories from the specified location within radius.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            radius (int): Search radius
            
        Returns:
            memories (list): Memories found in the area
        """
        memories = []
        
        for memory_id, memory in self.symbolic_content.items():
            mx, my = memory["location"]
            distance = math.sqrt((x - mx)**2 + (y - my)**2)
            
            if distance <= radius:
                # Stronger memories are more likely to be retrieved
                retrieval_chance = memory["intensity"] * (1 - (distance / radius))
                
                if random.random() < retrieval_chance:
                    memories.append(memory["content"])
        
        return memories
    
    def update_memories(self):
        """
        Update stored memories, applying decay and diffusion effects.
        
        Returns:
            forgotten_memories (list): Memories that have faded away
        """
        forgotten_memories = []
        memories_to_remove = []
        
        # Process each memory
        for memory_id, memory in self.symbolic_content.items():
            # Increment age
            memory["deposit_time"] += 1
            
            # Apply decay
            memory["intensity"] *= (1 - memory["decay_rate"])
            
            # If intensity drops too low, mark for removal
            if memory["intensity"] < 0.1:
                memories_to_remove.append(memory_id)
                forgotten_memories.append(memory["content"])
                
            # Apply diffusion (memories slowly drift with currents)
            x, y = memory["location"]
            vector = self.get_current_vector(x, y)
            
            new_x = x + vector[0] * 0.1
            new_y = y + vector[1] * 0.1
            
            # Keep within bounds
            new_x = max(0, min(new_x, self.width - 1))
            new_y = max(0, min(new_y, self.height - 1))
            
            memory["location"] = (new_x, new_y)
        
        # Remove forgotten memories
        for memory_id in memories_to_remove:
            del self.symbolic_content[memory_id]
            
        return forgotten_memories
    
    def _initialize_current_patterns(self):
        """Initialize basic current patterns."""
        # Create several circular current patterns (gyres)
        for _ in range(3):
            center_x = random.randint(0, self.width - 1)
            center_y = random.randint(0, self.height - 1)
            radius = random.randint(30, 100)
            clockwise = random.choice([True, False])
            
            self._create_circular_current(center_x, center_y, radius, clockwise)
            
        # Add some linear currents
        for _ in range(2):
            start_y = random.randint(0, self.height - 1)
            width = random.randint(20, 50)
            direction = 1 if random.random() > 0.5 else -1
            
            self._create_linear_current(start_y, width, direction)
    
    def _create_circular_current(self, center_x, center_y, radius, clockwise):
        """Create a circular current pattern (gyre)."""
        direction = 1 if clockwise else -1
        
        for x in range(max(0, center_x - radius), min(self.width, center_x + radius + 1)):
            for y in range(max(0, center_y - radius), min(self.height, center_y + radius + 1)):
                dx = x - center_x
                dy = y - center_y
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance <= radius:
                    # Create circular flow
                    strength = 1 - (distance / radius)  # Stronger near center
                    
                    # Calculate tangential vector components
                    angle = math.atan2(dy, dx) + (math.pi / 2 * direction)
                    flow_x = math.cos(angle) * strength * self.velocity_factor
                    flow_y = math.sin(angle) * strength * self.velocity_factor
                    
                    self.current_vectors[x, y, 0] = flow_x
                    self.current_vectors[x, y, 1] = flow_y
    
    def _create_linear_current(self, start_y, width, direction):
        """Create a linear current flowing across the map."""
        for y in range(max(0, start_y - width // 2), min(self.height, start_y + width // 2 + 1)):
            for x in range(self.width):
                # Distance from center of the current
                dist_from_center = abs(y - start_y)
                
                if dist_from_center <= width // 2:
                    # Strength based on distance from center (strongest at center)
                    strength = 1 - (dist_from_center / (width / 2))
                    
                    # Set horizontal flow
                    self.current_vectors[x, y, 0] = direction * strength * self.velocity_factor


class ThermohalineCirculation:
    """
    Simulates the global circulation of ocean currents driven by temperature
    and salinity differences.
    """
    
    def __init__(self, world_geometry):
        """
        Initialize the thermohaline circulation system.
        
        Args:
            world_geometry (WorldGeometry): Physical shape of the world
        """
        self.world_geometry = world_geometry
        self.conveyor_belt = self._initialize_conveyor_belt()
        self.upwelling_zones = self._initialize_upwelling_zones()
        self.downwelling_zones = self._initialize_downwelling_zones()
        self.circulation_strength = 1.0
        self.memory_transport_rate = 0.3
        self.symbolic_payload = {}
    
    def update(self, temperature_gradients, salinity_map):
        """
        Update the thermohaline circulation based on temperature and salinity.
        
        Args:
            temperature_gradients (ndarray): Temperature distribution
            salinity_map (ndarray): Salinity distribution
            
        Returns:
            circulation_events (list): Significant events in the circulation
        """
        circulation_events = []
        
        # Calculate driving forces for circulation
        driving_force = self._calculate_driving_force(temperature_gradients, salinity_map)
        
        # Update circulation strength based on driving force
        old_strength = self.circulation_strength
        self.circulation_strength = old_strength * 0.9 + driving_force * 0.1
        
        # Detect significant changes in circulation
        if abs(self.circulation_strength - old_strength) > 0.2:
            circulation_events.append({
                "type": "circulation_shift",
                "magnitude": abs(self.circulation_strength - old_strength),
                "direction": "strengthening" if self.circulation_strength > old_strength else "weakening"
            })
        
        # Update conveyor belt positions based on new strength
        self._update_conveyor_belt()
        
        # Process memory transport along the conveyor
        memory_events = self._process_memory_transport()
        circulation_events.extend(memory_events)
        
        # Update upwelling and downwelling zones
        upwelling_events = self._update_upwelling_zones()
        circulation_events.extend(upwelling_events)
        
        downwelling_events = self._update_downwelling_zones()
        circulation_events.extend(downwelling_events)
        
        return circulation_events
    
    def add_memory_payload(self, location, memory_content, intensity):
        """
        Add memory content to be transported by the circulation.
        
        Args:
            location (tuple): (x, y) coordinates where memory enters circulation
            memory_content (dict): The memory data to transport
            intensity (float): Strength of the memory imprint
            
        Returns:
            bool: True if successfully added, False otherwise
        """
        # Find nearest point on conveyor belt
        nearest_point, nearest_idx = self._find_nearest_conveyor_point(location)
        
        if nearest_point is not None:
            # Calculate distance to conveyor
            distance = math.sqrt((location[0] - nearest_point[0])**2 + 
                               (location[1] - nearest_point[1])**2)
            
            # Only add if close enough to conveyor
            if distance < 20:
                memory_id = str(uuid.uuid4())
                
                self.symbolic_payload[memory_id] = {
                    "content": memory_content,
                    "intensity": intensity * (1 - distance/20),  # Weaken with distance
                    "position": nearest_idx,  # Position along conveyor belt
                    "depth": 0,  # Surface level initially
                    "age": 0
                }
                return True
                
        return False
    
    def get_surface_currents(self):
        """
        Get the surface current vectors for the entire world.
        
        Returns:
            surface_currents (ndarray): Array of (x, y) current vectors
        """
        surface_currents = np.zeros((self.world_geometry.width, self.world_geometry.height, 2))
        
        # Add conveyor belt influence
        for i in range(len(self.conveyor_belt) - 1):
            p1 = self.conveyor_belt[i]
            p2 = self.conveyor_belt[(i + 1) % len(self.conveyor_belt)]
            
            # Direction vector
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx**2 + dy**2)
            
            if length > 0:
                dx /= length
                dy /= length
                
                # Influence nearby points
                for radius in range(1, 20):
                    influence = max(0, (20 - radius) / 20) * self.circulation_strength
                    
                    for angle in range(0, 360, 10):
                        rad = math.radians(angle)
                        cx = int(p1[0] + radius * math.cos(rad))
                        cy = int(p1[1] + radius * math.sin(rad))
                        
                        if 0 <= cx < self.world_geometry.width and 0 <= cy < self.world_geometry.height:
                            surface_currents[cx, cy, 0] += dx * influence
                            surface_currents[cx, cy, 1] += dy * influence
        
        return surface_currents
    
    def _calculate_driving_force(self, temperature_gradients, salinity_map):
        """Calculate the driving force for thermohaline circulation."""
        # Sample points in upwelling and downwelling zones
        upwelling_temps = []
        upwelling_salinity = []
        downwelling_temps = []
        downwelling_salinity = []
        
        for point in self.upwelling_zones:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < self.world_geometry.width and 0 <= y < self.world_geometry.height:
                upwelling_temps.append(temperature_gradients[x, y])
                upwelling_salinity.append(salinity_map[x, y])
        
        for point in self.downwelling_zones:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < self.world_geometry.width and 0 <= y < self.world_geometry.height:
                downwelling_temps.append(temperature_gradients[x, y])
                downwelling_salinity.append(salinity_map[x, y])
        
        # Calculate average temperature and salinity differences
        if upwelling_temps and downwelling_temps:
            avg_upwelling_temp = sum(upwelling_temps) / len(upwelling_temps)
            avg_upwelling_salinity = sum(upwelling_salinity) / len(upwelling_salinity)
            
            avg_downwelling_temp = sum(downwelling_temps) / len(downwelling_temps)
            avg_downwelling_salinity = sum(downwelling_salinity) / len(downwelling_salinity)
            
            # Temperature and salinity differences drive circulation
            temp_diff = abs(avg_downwelling_temp - avg_upwelling_temp)
            salinity_diff = abs(avg_downwelling_salinity - avg_upwelling_salinity)
            
            # Combine factors (both temperature and salinity differences are important)
            driving_force = 0.5 + ((temp_diff * 0.7 + salinity_diff * 0.3) / 2)
            
            return driving_force
        
        return 1.0  # Default
    
    def _initialize_conveyor_belt(self):
        """Initialize the path of the global conveyor belt."""
        width, height = self.world_geometry.width, self.world_geometry.height
        
        # Create a simple loop around the world
        conveyor_belt = []
        
        # Top edge
        for x in range(0, width, 10):
            conveyor_belt.append((x,
            # Continue from where ThermohalineCirculation left off

        # Look for a suitable point along the conveyor belt
        min_distance = float('inf')
        nearest_point = None
        nearest_idx = None
        
        for i, point in enumerate(self.conveyor_belt):
            distance = math.sqrt((location[0] - point[0])**2 + (location[1] - point[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_point = point
                nearest_idx = i
        
        return nearest_point, nearest_idx
    
    def _process_memory_transport(self):
        """Process the transport of memories along the conveyor belt."""
        memory_events = []
        memories_to_remove = []
        
        # Process each memory in the payload
        for memory_id, memory in self.symbolic_payload.items():
            # Age the memory
            memory["age"] += 1
            
            # Move along conveyor belt
            belt_length = len(self.conveyor_belt)
            movement_rate = self.memory_transport_rate * self.circulation_strength
            new_position = (memory["position"] + movement_rate) % belt_length
            memory["position"] = new_position
            
            # Integer position for accessing conveyor belt
            idx = int(memory["position"])
            
            # Check if memory passes through upwelling or downwelling zone
            current_location = self.conveyor_belt[idx % belt_length]
            
            for upwell in self.upwelling_zones:
                if self._point_distance(current_location, upwell) < 10:
                    # Memory moves upward
                    memory["depth"] = max(0, memory["depth"] - 0.2)
                    
                    # If reaches surface, create event
                    if memory["depth"] < 0.1 and memory["depth"] > 0:
                        memory_events.append({
                            "type": "memory_resurface",
                            "location": current_location,
                            "content": memory["content"],
                            "intensity": memory["intensity"]
                        })
                        memory["depth"] = 0
            
            for downwell in self.downwelling_zones:
                if self._point_distance(current_location, downwell) < 10:
                    # Memory moves downward
                    memory["depth"] += 0.2
                    
                    # If goes deep enough, create event
                    if memory["depth"] > 0.9 and memory["depth"] < 1.1:
                        memory_events.append({
                            "type": "memory_descent",
                            "location": current_location,
                            "content": memory["content"],
                            "intensity": memory["intensity"]
                        })
            
            # Decay intensity over time
            memory["intensity"] *= (0.997)  # Very slow decay
            
            # Remove if too weak
            if memory["intensity"] < 0.1:
                memories_to_remove.append(memory_id)
                memory_events.append({
                    "type": "memory_dissolution",
                    "location": self.conveyor_belt[idx % belt_length],
                    "content": memory["content"]
                })
        
        # Remove decayed memories
        for memory_id in memories_to_remove:
            del self.symbolic_payload[memory_id]
            
        return memory_events
    
    def _update_conveyor_belt(self):
        """Update the conveyor belt path based on circulation strength."""
        # Adjust the path slightly based on circulation strength
        for i in range(len(self.conveyor_belt)):
            # Add some random drift
            drift_x = (random.random() - 0.5) * 0.5
            drift_y = (random.random() - 0.5) * 0.5
            
            # Stronger circulation means less drift
            drift_factor = max(0.1, 1 - self.circulation_strength)
            
            x, y = self.conveyor_belt[i]
            x += drift_x * drift_factor
            y += drift_y * drift_factor
            
            # Keep within world bounds
            x = max(0, min(x, self.world_geometry.width - 1))
            y = max(0, min(y, self.world_geometry.height - 1))
            
            self.conveyor_belt[i] = (x, y)
    
    def _initialize_upwelling_zones(self):
        """Initialize the upwelling zones where deep water rises."""
        width, height = self.world_geometry.width, self.world_geometry.height
        upwelling_zones = []
        
        # Create several upwelling zones
        for _ in range(5):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            upwelling_zones.append((x, y))
        
        return upwelling_zones
    
    def _initialize_downwelling_zones(self):
        """Initialize the downwelling zones where surface water sinks."""
        width, height = self.world_geometry.width, self.world_geometry.height
        downwelling_zones = []
        
        # Create several downwelling zones
        for _ in range(5):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            downwelling_zones.append((x, y))
        
        return downwelling_zones
    
    def _update_upwelling_zones(self):
        """Update the upwelling zones and process their effects."""
        events = []
        
        for i, point in enumerate(self.upwelling_zones):
            # Random drift
            x, y = point
            x += (random.random() - 0.5) * 0.5
            y += (random.random() - 0.5) * 0.5
            
            # Keep within bounds
            x = max(0, min(x, self.world_geometry.width - 1))
            y = max(0, min(y, self.world_geometry.height - 1))
            
            self.upwelling_zones[i] = (x, y)
            
            # Occasionally generate symbolic emergence events
            if random.random() < 0.05 * self.circulation_strength:
                # Find memories in the vicinity that might surface
                nearby_memories = []
                
                for memory_id, memory in self.symbolic_payload.items():
                    idx = int(memory["position"])
                    memory_location = self.conveyor_belt[idx % len(self.conveyor_belt)]
                    distance = self._point_distance(memory_location, (x, y))
                    
                    if distance < 20 and memory["depth"] > 0.5:
                        nearby_memories.append(memory)
                
                if nearby_memories:
                    # Select one memory to surface
                    selected_memory = random.choice(nearby_memories)
                    
                    events.append({
                        "type": "memory_emergence",
                        "location": (x, y),
                        "content": selected_memory["content"],
                        "intensity": selected_memory["intensity"] * self.circulation_strength
                    })
        
        return events
    
    def _update_downwelling_zones(self):
        """Update the downwelling zones and process their effects."""
        events = []
        
        for i, point in enumerate(self.downwelling_zones):
            # Random drift
            x, y = point
            x += (random.random() - 0.5) * 0.5
            y += (random.random() - 0.5) * 0.5
            
            # Keep within bounds
            x = max(0, min(x, self.world_geometry.width - 1))
            y = max(0, min(y, self.world_geometry.height - 1))
            
            self.downwelling_zones[i] = (x, y)
            
            # Occasionally generate memory archiving events
            if random.random() < 0.03 * self.circulation_strength:
                events.append({
                    "type": "memory_archiving",
                    "location": (x, y),
                    "depth_increase": 0.3 * self.circulation_strength
                })
        
        return events
    
    def _point_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class WindPattern:
    """
    Represents atmospheric wind patterns that affect weather, entity movement,
    and the diffusion of motifs and thematic elements.
    """
    
    def __init__(self, width, height):
        """
        Initialize the wind pattern system.
        
        Args:
            width (int): Width of the world
            height (int): Height of the world
        """
        self.width = width
        self.height = height
        self.vector_field = np.zeros((width, height, 2))  # x, y components
        self.pressure_systems = []
        self.prevailing_direction = (1, 0)  # Default east
        self.turbulence = 0.3
        self.seasonal_modifier = 1.0
        
        # Initialize with basic patterns
        self._initialize_basic_patterns()
    
    def update(self, temperature_map, terrain_height_map, seasonal_modifier):
        """
        Update wind patterns based on temperature, terrain, and season.
        
        Args:
            temperature_map (ndarray): Temperature distribution
            terrain_height_map (ndarray): Terrain elevation
            seasonal_modifier (float): Seasonal influence factor
            
        Returns:
            wind_events (list): Significant wind events
        """
        wind_events = []
        self.seasonal_modifier = seasonal_modifier
        
        # Update pressure systems
        self._update_pressure_systems(temperature_map)
        
        # Calculate new wind vectors
        new_vector_field = np.zeros_like(self.vector_field)
        
        # Add influence from prevailing winds
        self._add_prevailing_wind(new_vector_field)
        
        # Add influence from pressure systems
        self._add_pressure_system_influence(new_vector_field)
        
        # Add terrain effects
        self._add_terrain_effects(new_vector_field, terrain_height_map)
        
        # Add turbulence
        self._add_turbulence(new_vector_field)
        
        # Smooth transitions from previous state
        self._smooth_transition(new_vector_field)
        
        # Detect significant changes
        wind_events = self._detect_wind_events()
        
        return wind_events
    
    def get_vector(self, x, y):
        """Get wind vector at the specified location."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.vector_field[int(x), int(y)]
        return np.array([0, 0])
    
    def add_pressure_system(self, center, radius, is_high_pressure, intensity):
        """
        Add a new pressure system to the wind patterns.
        
        Args:
            center (tuple): (x, y) center of the pressure system
            radius (float): Radius of influence
            is_high_pressure (bool): True for high pressure, False for low pressure
            intensity (float): Strength of the pressure system
        """
        self.pressure_systems.append({
            "center": center,
            "radius": radius,
            "is_high_pressure": is_high_pressure,
            "intensity": intensity,
            "age": 0,
            "max_age": random.randint(20, 50)
        })
    
    def _initialize_basic_patterns(self):
        """Initialize basic wind patterns."""
        # Set prevailing wind direction
        angle = random.uniform(0, 2 * math.pi)
        self.prevailing_direction = (math.cos(angle), math.sin(angle))
        
        # Add basic prevailing wind
        self._add_prevailing_wind(self.vector_field)
        
        # Add some initial pressure systems
        for _ in range(3):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            radius = random.randint(30, 100)
            is_high = random.choice([True, False])
            intensity = random.uniform(0.3, 0.8)
            
            self.add_pressure_system((x, y), radius, is_high, intensity)
            
        # Add influence from initial pressure systems
        self._add_pressure_system_influence(self.vector_field)
    
    def _add_prevailing_wind(self, vector_field):
        """Add prevailing wind component to vector field."""
        dx, dy = self.prevailing_direction
        strength = 0.5 * self.seasonal_modifier
        
        for x in range(self.width):
            for y in range(self.height):
                vector_field[x, y, 0] += dx * strength
                vector_field[x, y, 1] += dy * strength
    
    def _add_pressure_system_influence(self, vector_field):
        """Add influence from pressure systems."""
        for system in self.pressure_systems:
            center_x, center_y = system["center"]
            radius = system["radius"]
            is_high = system["is_high_pressure"]
            intensity = system["intensity"]
            
            # Determine influence range
            min_x = max(0, int(center_x - radius))
            max_x = min(self.width, int(center_x + radius + 1))
            min_y = max(0, int(center_y - radius))
            max_y = min(self.height, int(center_y + radius + 1))
            
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    dx = x - center_x
                    dy = y - center_y
                    distance = math.sqrt(dx**2 + dy**2)
                    
                    if distance <= radius:
                        # Calculate influence strength
                        strength = intensity * (1 - (distance / radius))
                        
                        # Direction depends on pressure type
                        if is_high:
                            # High pressure: outward
                            if distance > 0:  # Avoid division by zero
                                norm = distance
                                vector_field[x, y, 0] += (dx / norm) * strength
                                vector_field[x, y, 1] += (dy / norm) * strength
                        else:
                            # Low pressure: inward
                            if distance > 0:  # Avoid division by zero
                                norm = distance
                                vector_field[x, y, 0] -= (dx / norm) * strength
                                vector_field[x, y, 1] -= (dy / norm) * strength
    
    def _add_terrain_effects(self, vector_field, terrain_height_map):
        """Add terrain influences on wind patterns."""
        # Calculate terrain gradient
        gradient_x, gradient_y = np.gradient(terrain_height_map)
        
        for x in range(self.width):
            for y in range(self.height):
                # Stronger winds get diverted by terrain more
                terrain_influence = min(1.0, terrain_height_map[x, y])
                
                # Add terrain channeling effect
                channel_strength = 0.3 * terrain_influence
                vector_field[x, y, 0] -= gradient_x[x, y] * channel_strength
                vector_field[x, y, 1] -= gradient_y[x, y] * channel_strength
    
    def _add_turbulence(self, vector_field):
        """Add random turbulence to wind patterns."""
        turbulence_strength = self.turbulence * self.seasonal_modifier
        
        for x in range(self.width):
            for y in range(self.height):
                vector_field[x, y, 0] += (random.random() - 0.5) * turbulence_strength
                vector_field[x, y, 1] += (random.random() - 0.5) * turbulence_strength
    
    def _smooth_transition(self, new_vector_field):
        """Smooth transition between previous and new wind vectors."""
        # Blend old and new
        self.vector_field = self.vector_field * 0.7 + new_vector_field * 0.3
    
    def _update_pressure_systems(self, temperature_map):
        """Update pressure systems based on temperature and age."""
        systems_to_remove = []
        
        for i, system in enumerate(self.pressure_systems):
            # Age the system
            system["age"] += 1
            
            # Update intensity based on age
            life_fraction = system["age"] / system["max_age"]
            if life_fraction < 0.2:
                # Intensifying
                system["intensity"] = min(1.0, system["intensity"] * 1.05)
            elif life_fraction > 0.8:
                # Weakening
                system["intensity"] *= 0.95
                
            # Move the system
            cx, cy = system["center"]
            
            # Movement depends on pressure type and prevailing winds
            dx, dy = self.prevailing_direction
            
            # High pressure systems tend to move with prevailing winds
            # Low pressure systems have more complex movement
            if system["is_high_pressure"]:
                move_x = dx * 0.3 + (random.random() - 0.5) * 0.2
                move_y = dy * 0.3 + (random.random() - 0.5) * 0.2
            else:
                move_x = dx * 0.2 + (random.random() - 0.5) * 0.4
                move_y = dy * 0.2 + (random.random() - 0.5) * 0.4
            
            cx += move_x
            cy += move_y
            
            # Keep within bounds
            cx = max(0, min(cx, self.width - 1))
            cy = max(0, min(cy, self.height - 1))
            
            system["center"] = (cx, cy)
            
            # Remove if too old
            if system["age"] >= system["max_age"]:
                systems_to_remove.append(i)
        
        # Remove old systems
        for idx in sorted(systems_to_remove, reverse=True):
            self.pressure_systems.pop(idx)
            
        # Randomly generate new systems
        if random.random() < 0.05 and len(self.pressure_systems) < 8:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            radius = random.randint(30, 100)
            is_high = random.choice([True, False])
            
            # Temperature influences pressure system type
            if 0 <= x < self.width and 0 <= y < self.height:
                local_temp = temperature_map[x, y]
                # Higher chance of high pressure in cold areas
                if local_temp < 0.5 and random.random() < 0.7:
                    is_high = True
                # Higher chance of low pressure in warm areas
                elif local_temp > 0.5 and random.random() < 0.7:
                    is_high = False
            
            intensity = random.uniform(0.3, 0.8)
            self.add_pressure_system((x, y), radius, is_high, intensity)
    
    def _detect_wind_events(self):
        """Detect significant wind events based on patterns."""
        wind_events = []
        
        # Look for circular patterns (potential cyclones)
        for system in self.pressure_systems:
            if not system["is_high_pressure"] and system["intensity"] > 0.7:
                wind_events.append({
                    "type": "cyclone",
                    "location": system["center"],
                    "intensity": system["intensity"],
                    "radius": system["radius"]
                })
            elif system["is_high_pressure"] and system["intensity"] > 0.8:
                wind_events.append({
                    "type": "anticyclone",
                    "location": system["center"],
                    "intensity": system["intensity"],
                    "radius": system["radius"]
                })
        
        # Look for strong directional flows (jet streams)
        max_flow = 0
        max_flow_location = None
        max_flow_direction = None
        
        sample_points = 100
        for _ in range(sample_points):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            
            vx, vy = self.vector_field[x, y]
            magnitude = math.sqrt(vx**2 + vy**2)
            
            if magnitude > max_flow:
                max_flow = magnitude
                max_flow_location = (x, y)
                max_flow_direction = (vx, vy)
        
        if max_flow > 1.5:
            wind_events.append({
                "type": "jet_stream",
                "location": max_flow_location,
                "intensity": max_flow,
                "direction": max_flow_direction
            })
        
        return wind_events


class ClimateZone:
    """
    Represents a distinct climatic region with characteristic weather patterns,
    seasonal variations, and symbolic associations.
    """
    
    def __init__(self, center, radius, climate_type, intensity=1.0):
        """
        Initialize a climate zone.
        
        Args:
            center (tuple): (x, y) center coordinates
            radius (float): Radius of influence
            climate_type (str): Type of climate (desert, tundra, etc.)
            intensity (float): Strength of climate influence
        """
        self.center = center
        self.radius = radius
        self.climate_type = climate_type
        self.intensity = intensity
        self.seasonal_variations = self._initialize_seasonal_variations()
        self.symbolic_associations = self._initialize_symbolic_associations()
        self.weather_tendencies = self._initialize_weather_tendencies()
        self.boundary_blending = 20  # Width of transition zone
    
    def get_influence(self, position):
        """
        Calculate the influence of this climate zone at the given position.
        
        Args:
            position (tuple): (x, y) coordinates to check
            
        Returns:
            influence (float): Strength of influence (0-1)
            climate_data (dict): Climate effects at this position
        """
        x, y = position
        cx, cy = self.center
        
        distance = math.sqrt((x - cx)**2 + (y - cy)**2)
        
        # No influence beyond radius + blending zone
        if distance > self.radius + self.boundary_blending:
            return 0, {}
        
        # Full influence within core radius
        if distance <= self.radius:
            influence = self.intensity
        else:
            # Linear falloff in blending zone
            blend_distance = distance - self.radius
            influence = self.intensity * (1 - (blend_distance / self.boundary_blending))
        
        # Get climate data modified by distance from center
        climate_data = {
            "type": self.climate_type,
            "temperature_modifier": self._get_temperature_modifier() * influence,
            "precipitation_modifier": self._get_precipitation_modifier() * influence,
            "symbolic_elements": self._get_scaled_symbolic_elements(influence),
            "weather_bias": self._get_scaled_weather_bias(influence)
        }
        
        return influence, climate_data
    
    def update_seasonal_state(self, current_season, progress):
        """
        Update the climate zone's state based on seasonal changes.
        
        Args:
            current_season (str): The current season
            progress (float): Progress through the current season (0-1)
        """
        # Adjust intensity based on seasonal alignment
        seasonal_alignment = self.seasonal_variations.get(current_season, {}).get("intensity_modifier", 1.0)
        self.intensity = min(1.0, self.intensity * 0.9 + seasonal_alignment * 0.1)
        
        # Some climate zones may expand or contract seasonally
        radius_modifier = self.seasonal_variations.get(current_season, {}).get("radius_modifier", 1.0)
        self.radius *= 0.95  # Slight contraction by default
        self.radius = min(100, max(30, self.radius * radius_modifier))
    
    def _get_temperature_modifier(self):
        """Get the temperature modifier for this climate."""
        base_modifiers = {
            "desert": 1.5,
            "tundra": -1.0,
            "temperate": 0.0,
            "tropical": 1.0,
            "arctic": -1.5,
            "oceanic": -0.3
        }
        
        return base_modifiers.get(self.climate_type, 0.0)
    
    def _get_precipitation_modifier(self):
        """Get the precipitation modifier for this climate."""
        base_modifiers = {
            "desert": -0.8,
            "tundra": -0.2,
            "temperate": 0.2,
            "tropical": 0.8,
            "arctic": -0.5,
            "oceanic": 0.5
        }
        
        return base_modifiers.get(self.climate_type, 0.0)
    
    def _get_scaled_symbolic_elements(self, influence):
        """Get symbolic elements scaled by influence strength."""
        scaled_elements = {}
        
        for symbol, value in self.symbolic_associations.items():
            scaled_elements[symbol] = value * influence
            
        return scaled_elements
    
    def _get_scaled_weather_bias(self, influence):
        """Get weather tendencies scaled by influence strength."""
        scaled_bias = {}
        
        for weather, bias in self.weather_tendencies.items():
            scaled_bias[weather] = bias * influence
            
        return scaled_bias
    
    def _initialize_seasonal_variations(self):
        """Initialize seasonal variations for this climate type."""
        if self.climate_type == "desert":
            return {
                "winter": {"intensity_modifier": 0.9, "radius_modifier": 0.95},
                "spring": {"intensity_modifier": 1.0, "radius_modifier": 1.05},
                "summer": {"intensity_modifier": 1.2, "radius_modifier": 1.1},
                "autumn": {"intensity_modifier": 1.0, "radius_modifier": 1.0}
            }
        elif self.climate_type == "tundra":
            return {
                "winter": {"intensity_modifier": 1.2, "radius_modifier": 1.1},
                "spring": {"intensity_modifier": 0.9, "radius_modifier": 0.9},
                "summer": {"intensity_modifier": 0.7, "radius_modifier": 0.8},
                "autumn": {"intensity_modifier": 1.0, "radius_modifier": 1.0}
            }
        elif self.climate_type == "temperate":
            return {
                "winter": {"intensity_modifier": 0.8, "radius_modifier": 0.9},
                "spring": {"intensity_modifier": 1.1, "radius_modifier": 1.0},
                "summer": {"intensity_modifier": 1.0, "radius_modifier": 1.0},
                "autumn": {"intensity_modifier": 1.1, "radius_modifier": 1.0}
            }
        elif self.climate_type == "tropical":
            return {
                "winter": {"intensity_modifier": 0.9, "radius_modifier": 0.95},
                "spring": {"intensity_modifier": 1.0, "radius_modifier": 1.0},
                "summer": {"intensity_modifier": 1.1, "radius_modifier": 1.05},
                "autumn": {"intensity_modifier": 1.0, "radius_modifier": 1.0}
            }
        elif self.climate_type == "arctic":
            return {
                "winter": {"intensity_modifier": 1.3, "radius_modifier": 1.2},
                "spring": {"intensity_modifier": 1.0, "radius_modifier": 0.9},
                "summer": {"intensity_modifier": 0.6, "radius_modifier": 0.7},
                "autumn": {"intensity_modifier": 0.9, "radius_modifier": 0.95}
            }
        elif self.climate_type == "oceanic":
            return {
                "winter": {"intensity_modifier": 1.1, "radius_modifier": 1.05},
                "spring": {"intensity_modifier": 1.0, "radius_modifier": 1.0},
                "summer": {"intensity_modifier": 0.9, "radius_modifier": 0.95},
                "autumn": {"intensity_modifier": 1.0, "radius_modifier": 1.0}
            }
        else:
            return {
                "winter": {"intensity_modifier": 1.0, "radius_modifier": 1.0},
                "spring": {"intensity_modifier": 1.0, "radius_modifier": 1.0},
                "summer": {"intensity_modifier": 1.0, "radius_modifier": 1.0},
                "autumn": {"intensity_modifier": 1.0, "radius_modifier": 1.0}
            }
    
    def _initialize_symbolic_associations(self):
        """Initialize symbolic associations for this climate type."""
        if self.climate_type == "desert":
            return {
                "isolation": 0.8,
                "endurance": 0.9,
                "clarity": 0.7,
                "timelessness": 0.8,
                "transformation": 0.6
            }
        elif self.climate_type == "tundra":
            return {
                "solitude": 0.8,
                "stillness": 0.9,
                "preservation": 0.7,
                "patience": 0.8,
                "reflection": 0.7
            }
        elif self.climate_type == "temperate":
            return {
                "balance": 0.9,
                "renewal": 0.7,
                "cycles": 0.8,
                "harmony": 0.7,
                "adaptation": 0.8
            }
        elif self.climate_type == "tropical":
            return {
                "abundance": 0.9,
                "vitality": 0.8,
                "diversity": 0.9,
                "growth": 0.8,
                "interconnection": 0.7
            }
        elif self.climate_type == "arctic":
            return {
                "endurance": 0.9,
                "isolation": 0.8,
                "purity": 0.7,
                "dormancy": 0.8,
                "preservation": 0.9
            }
        elif self.climate_type == "oceanic":
            return {
                "fluidity": 0.9,
                "connection": 0.8,
                "depth": 0.7,
                "mystery": 0.8,
                "transformation": 0.6
            }
        else:
            return {
                "neutral": 0.5,
                "adaptation": 0.5,
                "balance": 0.5,
                "change": 0.5,
                "transformation": 0.5
            }
    def _initialize_weather_tendencies(self):
        """Initialize weather tendencies for this climate type."""
        if self.climate_type == "desert":
            return {
                "dry": 0.9,
                "hot": 0.8,
                "windy": 0.5,
                "clear": 0.7
            }
        elif self.climate_type == "tundra":
            return {
                "cold": 0.9,
                "dry": 0.7,
                "windy": 0.6,
                "cloudy": 0.5
            }
        elif self.climate_type == "temperate":
            return {
                "moderate": 0.8,
                "rainy": 0.6,
                "cloudy": 0.7,
                "windy": 0.5
            }
        elif self.climate_type == "tropical":
            return {
                "humid": 0.9,
                "rainy": 0.8,
                "calm": 0.6,
                "sunny": 0.7
            }
        elif self.climate_type == "arctic":
            return {
                "cold": 1.0,
                "dry": 0.8,
                "windy": 0.7,
                "cloudy": 0.6
            }
        elif self.climate_type == "oceanic":
            return {
                "mild": 0.8,
                "humid": 0.7,
                "windy": 0.6,
                "rainy": 0.5
            }
        else:
            return {
                "neutral": 0.5,
                "variable": 0.5,
                "calm": 0.5,
                "unpredictable": 0.5
            }

    def __repr__(self):
        """String representation of the climate zone."""
        return f"ClimateZone(center={self.center}, radius={self.radius}, type={self.climate_type})"


class WorldState:
    """
    Maintains the global state of the environmental systems and their interaction
    with cosmic entities in the simulation.
    
    Serves as a bridge between the environmental modules and the cosmic entities,
    facilitating the flow of symbolic content and environmental effects.
    """
    
    def __init__(self, width=WORLD_SIZE, height=WORLD_SIZE):
        """Initialize the world state with environmental systems."""
        self.width = width
        self.height = height
        self.time = 0
        self.seasonal_cycle = SeasonalCycle()
        self.wind_patterns = WindPattern(width, height)
        self.climate_zones = []
        self.active_storms = []
        self.temperature_map = np.zeros((width, height))
        self.precipitation_map = np.zeros((width, height))
        self.humidity_map = np.zeros((width, height))
        self.terrain_height_map = np.zeros((width, height))
        self.symbolic_influence_map = {}  # symbol -> influence array
        self.event_history = []
        
        # Dynamic systems
        self.ocean_currents = None  # Will be initialized if oceans are present
        
        # Initialize terrain and climate
        self._initialize_terrain()
        self._initialize_climate_zones()
    
    def update(self, time_delta):
        """
        Update all environmental systems and their interactions.
        
        Args:
            time_delta: Time increment
            
        Returns:
            events: List of significant environmental events
        """
        self.time += time_delta
        events = []
        
        # Update season
        season_changed = self.seasonal_cycle.advance()
        if season_changed:
            events.append({
                "type": "season_change",
                "season": self.seasonal_cycle.current_season,
                "year": self.seasonal_cycle.year_count
            })
        
        # Get seasonal info
        seasonal_info = self.seasonal_cycle.get_current_season_info()
        
        # Update climate zones based on season
        for zone in self.climate_zones:
            zone.update_seasonal_state(seasonal_info["season"], seasonal_info["progress"])
        
        # Generate temperature and precipitation maps
        self._update_climate_maps(seasonal_info)
        
        # Update wind patterns
        wind_events = self.wind_patterns.update(
            self.temperature_map, 
            self.terrain_height_map,
            seasonal_info["temperature"]  # Use temperature as seasonal modifier
        )
        events.extend(wind_events)
        
        # Update ocean currents if present
        if self.ocean_currents:
            current_events = self.ocean_currents.update(
                self.temperature_map,
                self.humidity_map  # Using humidity as a proxy for salinity
            )
            events.extend(current_events)
        
        # Update existing storms
        self._update_storms(time_delta)
        
        # Generate new storms
        storm_chance = 0.05 * time_delta * seasonal_info["precipitation"]
        if random.random() < storm_chance:
            storm = self._generate_storm(seasonal_info)
            if storm:
                self.active_storms.append(storm)
                events.append({
                    "type": "storm_formation",
                    "storm_type": storm.storm_type,
                    "location": storm.center,
                    "intensity": storm.intensity
                })
        
        # Record significant events
        for event in events:
            if event not in self.event_history:  # Avoid duplicates
                self.event_history.append(event)
                
        # Trim event history if it gets too long
        if len(self.event_history) > 100:
            self.event_history = self.event_history[-100:]
            
        return events
    
    def apply_symbolic_effect(self, position, symbol, strength):
        """
        Apply a symbolic effect to the world at the given position.
        
        Args:
            position: (x,y) coordinates
            symbol: The symbolic concept
            strength: Effect strength
        """
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            if symbol not in self.symbolic_influence_map:
                self.symbolic_influence_map[symbol] = np.zeros((self.width, self.height))
            
            self.symbolic_influence_map[symbol][int(x), int(y)] += strength
    
    def get_local_climate(self, position):
        """
        Get the climate conditions at the specified position.
        
        Args:
            position: (x,y) coordinates
            
        Returns:
            climate_data: Climate information at this position
        """
        x, y = position
        climate_data = {
            "temperature": 0,
            "precipitation": 0,
            "humidity": 0,
            "wind": [0, 0],
            "symbolic_elements": {},
            "active_effects": []
        }
        
        # Get base values from maps
        if 0 <= x < self.width and 0 <= y < self.height:
            climate_data["temperature"] = self.temperature_map[int(x), int(y)]
            climate_data["precipitation"] = self.precipitation_map[int(x), int(y)]
            climate_data["humidity"] = self.humidity_map[int(x), int(y)]
            climate_data["wind"] = self.wind_patterns.get_vector(x, y)
        
        # Get influences from climate zones
        zone_influences = []
        for zone in self.climate_zones:
            influence, zone_data = zone.get_influence(position)
            if influence > 0:
                zone_influences.append((influence, zone_data))
        
        # Combine zone influences
        if zone_influences:
            total_influence = sum(infl for infl, _ in zone_influences)
            
            for influence, zone_data in zone_influences:
                weight = influence / total_influence
                
                # Weighted contribution to temperature and precipitation
                climate_data["temperature"] += zone_data["temperature_modifier"] * weight
                climate_data["precipitation"] += zone_data["precipitation_modifier"] * weight
                
                # Combine symbolic elements
                for symbol, value in zone_data["symbolic_elements"].items():
                    if symbol in climate_data["symbolic_elements"]:
                        climate_data["symbolic_elements"][symbol] += value * weight
                    else:
                        climate_data["symbolic_elements"][symbol] = value * weight
        
        # Check for active storms at this location
        for storm in self.active_storms:
            storm_x, storm_y = storm.center
            distance = math.sqrt((x - storm_x)**2 + (y - storm_y)**2)
            
            if distance <= storm.radius:
                # Calculate how much this point is affected by the storm
                influence = storm.intensity * (1 - (distance / storm.radius))
                
                # Add storm to active effects
                climate_data["active_effects"].append({
                    "type": "storm",
                    "storm_type": storm.storm_type,
                    "intensity": influence
                })
                
                # Storms modify local conditions
                if storm.storm_type == "hurricane":
                    climate_data["wind"][0] *= (1 + influence)
                    climate_data["wind"][1] *= (1 + influence)
                    climate_data["precipitation"] *= (1 + influence)
                elif storm.storm_type == "thunderstorm":
                    climate_data["precipitation"] *= (1 + influence * 0.5)
                    climate_data["temperature"] -= influence * 0.1
                elif storm.storm_type == "fog":
                    climate_data["humidity"] *= (1 + influence * 0.3)
                    climate_data["precipitation"] *= (1 + influence * 0.1)
                
                # Add storm's symbolic content
                for symbol, potency in storm.symbolic_content.items():
                    value = potency * influence
                    if symbol in climate_data["symbolic_elements"]:
                        climate_data["symbolic_elements"][symbol] += value
                    else:
                        climate_data["symbolic_elements"][symbol] = value
        
        # Get symbolic influences from map
        for symbol, influence_map in self.symbolic_influence_map.items():
            if 0 <= x < self.width and 0 <= y < self.height:
                value = influence_map[int(x), int(y)]
                if value > 0.1:  # Only include significant influences
                    if symbol in climate_data["symbolic_elements"]:
                        climate_data["symbolic_elements"][symbol] += value
                    else:
                        climate_data["symbolic_elements"][symbol] = value
        
        return climate_data
    
    def get_dominant_symbols(self, position, count=3):
        """
        Get the dominant symbolic elements at the specified position.
        
        Args:
            position: (x,y) coordinates
            count: Maximum number of symbols to return
            
        Returns:
            symbols: List of (symbol, strength) tuples
        """
        climate_data = self.get_local_climate(position)
        symbols = climate_data["symbolic_elements"]
        
        # Sort by strength
        sorted_symbols = sorted(symbols.items(), key=lambda x: x[1], reverse=True)
        
        # Return top symbols
        return sorted_symbols[:count]
    
    def deposit_memory(self, position, memory_content, intensity=1.0):
        """
        Deposit memory content into the environment.
        
        Args:
            position: (x,y) coordinates
            memory_content: The memory data
            intensity: Strength of memory imprint
            
        Returns:
            success: Boolean indicating successful deposit
        """
        # If ocean currents exist, attempt to deposit in currents
        if self.ocean_currents:
            return self.ocean_currents.add_memory_payload(position, memory_content, intensity)
        
        # Otherwise, apply as a symbolic influence
        for key, value in memory_content.items():
            if isinstance(value, (int, float)) and value > 0:
                self.apply_symbolic_effect(position, key, value * intensity)
        
        return True
    
    def _initialize_terrain(self):
        """Initialize terrain height map."""
        # Simple fractal terrain generation
        scale = 8
        terrain = np.zeros((self.width, self.height))
        
        while scale > 0:
            for x in range(0, self.width, scale):
                for y in range(0, self.height, scale):
                    terrain[x:x+scale, y:y+scale] += random.random() * scale
            scale //= 2
            
        # Normalize to 0-1 range
        if np.max(terrain) > 0:
            terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))
            
        self.terrain_height_map = terrain
    
    def _initialize_climate_zones(self):
        """Initialize climate zones based on terrain."""
        # Create several climate zones of different types
        climate_types = ["desert", "tundra", "temperate", "tropical", "arctic", "oceanic"]
        
        for _ in range(6):
            climate_type = random.choice(climate_types)
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            radius = random.randint(50, 150)
            
            self.climate_zones.append(ClimateZone((x, y), radius, climate_type))
    
    def _update_climate_maps(self, seasonal_info):
        """Update temperature, precipitation, and humidity maps."""
        # Initialize with base values influenced by terrain and season
        for x in range(self.width):
            for y in range(self.height):
                # Terrain influence on temperature (higher = cooler)
                height_influence = -0.5 * self.terrain_height_map[x, y]
                
                # Seasonal base temperature (normalized to -1 to 1 range)
                seasonal_temp = seasonal_info["temperature"] * 2 - 1
                
                # Combine factors
                self.temperature_map[x, y] = 0.5 + (seasonal_temp * 0.3 + height_influence * 0.2)
                
                # Terrain influence on precipitation (higher = more rain, up to a point)
                precipitation_factor = self.terrain_height_map[x, y] * (1 - self.terrain_height_map[x, y] / 2)
                
                # Seasonal precipitation
                seasonal_precip = seasonal_info["precipitation"]
                
                self.precipitation_map[x, y] = seasonal_precip * 0.7 + precipitation_factor * 0.3
                
                # Humidity correlates with precipitation but is more affected by temperature
                self.humidity_map[x, y] = self.precipitation_map[x, y] * 0.7 - (max(0, self.temperature_map[x, y] - 0.5) * 0.3)
        
        # Apply climate zone influences
        for x in range(self.width):
            for y in range(self.height):
                position = (x, y)
                climate_data = self.get_local_climate(position)
                
                # Update maps with climate zone influences
                self.temperature_map[x, y] = climate_data["temperature"]
                self.precipitation_map[x, y] = climate_data["precipitation"]
                self.humidity_map[x, y] = climate_data["humidity"]
        
        # Smooth the maps
        self.temperature_map = self._smooth_map(self.temperature_map)
        self.precipitation_map = self._smooth_map(self.precipitation_map)
        self.humidity_map = self._smooth_map(self.humidity_map)
        
        # Ensure values are in valid ranges
        self.temperature_map = np.clip(self.temperature_map, 0, 1)
        self.precipitation_map = np.clip(self.precipitation_map, 0, 1)
        self.humidity_map = np.clip(self.humidity_map, 0, 1)
    
    def _smooth_map(self, map_data, kernel_size=3):
        """Apply smoothing to a map."""
        from scipy.ndimage import uniform_filter
        return uniform_filter(map_data, size=kernel_size, mode='reflect')
    
    def _update_storms(self, time_delta):
        """Update all active storms."""
        storms_to_remove = []
        
        for storm in self.active_storms:
            # Update the storm
            still_active = storm.update(self)
            
            if not still_active:
                storms_to_remove.append(storm)
        
        # Remove dissipated storms
        for storm in storms_to_remove:
            self.active_storms.remove(storm)
    
    def _generate_storm(self, seasonal_info):
        """Generate a new storm based on current conditions."""
        # Choose a random position
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        
        # Storm chance depends on local conditions
        local_precip = self.precipitation_map[x, y]
        local_temp = self.temperature_map[x, y]
        
        # Only generate storm if conditions are right
        if local_precip < 0.4:
            return None  # Too dry
            
        # Determine storm type based on conditions
        storm_type = None
        if local_temp > 0.7 and local_precip > 0.7:
            storm_type = "hurricane"
            radius = random.randint(30, 60)
            intensity = 0.7 + random.random() * 0.3
        elif local_temp > 0.5 and local_precip > 0.6:
            storm_type = "thunderstorm"
            radius = random.randint(10, 30)
            intensity = 0.5 + random.random() * 0.5
        elif local_temp < 0.4 and local_precip > 0.5:
            storm_type = "fog"
            radius = random.randint(20, 40)
            intensity = 0.4 + random.random() * 0.4
        else:
            storm_type = "rain"
            radius = random.randint(15, 35)
            intensity = 0.3 + random.random() * 0.4
        
        if not storm_type:
            return None
            
        # Generate symbolic content for the storm
        symbolic_content = self._generate_storm_symbolism(storm_type, seasonal_info)
        
        # Create the storm
        return Storm((x, y), storm_type, radius, intensity, symbolic_content)
    
    def _generate_storm_symbolism(self, storm_type, seasonal_info):
        """Generate symbolic content for a storm."""
        symbolic_content = {}
        
        # Base symbolism by storm type
        if storm_type == "hurricane":
            base_symbols = ["chaos", "transformation", "power", "destruction", "renewal"]
            symbol_count = random.randint(2, 4)
        elif storm_type == "thunderstorm":
            base_symbols = ["revelation", "energy", "conflict", "clarity", "change"]
            symbol_count = random.randint(2, 3)
        elif storm_type == "fog":
            base_symbols = ["mystery", "concealment", "confusion", "illusion", "transition"]
            symbol_count = random.randint(2, 3)
        elif storm_type == "rain":
            base_symbols = ["cleansing", "renewal", "life", "sadness", "contemplation"]
            symbol_count = random.randint(1, 3)
        else:
            base_symbols = ["change", "cycle", "nature", "wildness"]
            symbol_count = random.randint(1, 2)
        
        # Select random symbols
        selected_symbols = random.sample(base_symbols, symbol_count)
        
        # Seasonal influences
        season = seasonal_info["season"]
        if season == "winter":
            if random.random() < 0.5:
                selected_symbols.append("stillness")
        elif season == "spring":
            if random.random() < 0.5:
                selected_symbols.append("growth")
        elif season == "summer":
            if random.random() < 0.5:
                selected_symbols.append("vigor")
        elif season == "autumn":
            if random.random() < 0.5:
                selected_symbols.append("transition")
        
        # Assign potency to each symbol
        for symbol in selected_symbols:
            symbolic_content[symbol] = 0.5 + random.random() * 0.5
        
        return symbolic_content


# -------------------------------------------------------------------------
# Emotional Resonance System
# -------------------------------------------------------------------------

class FloralGrowthPattern(Enum):
    """Defines the fundamental growth patterns for symbolic flora"""
    BRANCHING = "branching"      # Forked, divergent growth patterns
    SPIRAL = "spiral"            # Fibonacci-sequence based spiral patterns
    LAYERED = "layered"          # Concentric growth from center outward
    FRACTAL = "fractal"          # Self-similar patterns at different scales
    RADIAL = "radial"            # Growth outward from central point
    LATTICE = "lattice"          # Grid-like structured growth
    CHAOTIC = "chaotic"          # Unpredictable, emergent growth patterns
    HARMONIC = "harmonic"        # Wavelike undulating growth
    MIRRORED = "mirrored"        # Symmetrical balanced growth
    ADAPTIVE = "adaptive"        # Context-responsive variable growth


class NutrientType(Enum):
    """Types of nutrients that symbolic flora can consume"""
    PHYSICAL = "physical"          # Material resources
    SYMBOLIC = "symbolic"          # Meaning and pattern resources
    EMOTIONAL = "emotional"        # Feeling-based resources
    TEMPORAL = "temporal"          # Time-based resources
    ENTROPIC = "entropic"          # Disorder-based resources
    HARMONIC = "harmonic"          # Resonance-based resources
    VOID = "void"                  # Emptiness and negative space
    NARRATIVE = "narrative"        # Story and sequence resources
    QUANTUM = "quantum"            # Probability and uncertainty resources
    METAPHORIC = "metaphoric"      # Symbolic transformation resources


class FloraEvolutionStage(Enum):
    """Evolutionary stages for motif flora"""
    SEED = "seed"                    # Initial pattern state
    EMERGENT = "emergent"            # First growth and establishment
    MATURING = "maturing"            # Development of core structures
    FLOWERING = "flowering"          # Peak expression and reproduction
    SEEDING = "seeding"              # Distribution of pattern copies
    WITHERING = "withering"          # Decline and entropy increase
    COMPOSTING = "composting"        # Breaking down into base patterns
    DORMANT = "dormant"              # Suspended animation state
    RESURGENT = "resurgent"          # Renewal after dormancy
    TRANSCENDENT = "transcendent"    # Evolution beyond original pattern


class MotifFloraSystem:
    """
    Models the growth, propagation, and evolution of symbolic plant life.
    
    MotifFloraSystem represents the botanical dimension of the symbolic ecosystem,
    where meaning-patterns grow, interact with their environment, and evolve based
    on available nutrients and conditions. These flora form the foundation of many
    symbolic food chains and serve as processors of raw symbolic matter into more
    structured forms.
    
    The system features:
    - Pattern-based growth with multiple archetypal structures
    - Symbolic nutrient metabolism and motif expression
    - Environmental response and adaptation mechanisms
    - Seasonal and cyclical behavior with growth phases
    - Symbolic pollination and cross-fertilization of ideas
    - Emergent properties through flora communities and forests
    """
    
    def __init__(self, 
                 owner_entity: Any,
                 base_pattern: str = None,
                 growth_style: FloralGrowthPattern = None,
                 evolution_stage: FloraEvolutionStage = FloraEvolutionStage.SEED,
                 maturation_rate: float = 0.05,
                 symbolic_metabolism: Dict[NutrientType, float] = None,
                 primary_motifs: List[str] = None):
        """
        Initialize a new MotifFloraSystem with the specified parameters.
        
        Args:
            owner_entity: The entity this flora system belongs to
            base_pattern: The foundational pattern for this flora (auto-generated if None)
            growth_style: The fundamental growth pattern (randomly chosen if None)
            evolution_stage: Current evolutionary stage of the flora
            maturation_rate: Base rate at which the flora evolves (0.0-1.0)
            symbolic_metabolism: Dictionary mapping nutrient types to processing efficiencies
            primary_motifs: List of primary motifs expressed by this flora
        """
        self.owner = owner_entity
        self.base_pattern = base_pattern or self._generate_base_pattern()
        self.growth_style = growth_style or random.choice(list(FloralGrowthPattern))
        self.evolution_stage = evolution_stage
        self.maturation_rate = maturation_rate
        self.symbolic_metabolism = symbolic_metabolism or self._initialize_metabolism()
        self.primary_motifs = primary_motifs or self._initialize_motifs()
        
        # Growth metrics
        self.growth_factor = 0.0  # 0.0 (seed) to 1.0 (fully grown)
        self.health = 1.0  # 0.0 (dead) to 1.0 (perfect health)
        self.pattern_density = 0.1  # Pattern richness/complexity
        self.root_depth = 0.0  # Symbolic grounding/stability
        self.canopy_spread = 0.0  # Area of influence
        
        # Environmental response
        self.seasonal_state = {}  # Current response to seasons
        self.adaptation_history = []  # Record of adaptations
        self.environmental_responses = self._initialize_environmental_responses()
        
        # Reproduction and propagation
        self.seed_bank = []  # Generated seed patterns
        self.pollination_vectors = set()  # Entities that can spread pollen
        self.cross_pollination_record = {}  # History of genetic exchanges
        
        # Community metrics
        self.symbiotic_relationships = {}  # Flora-fauna relationships
        self.competitive_relationships = {}  # Resource competition
        self.community_role = {}  # Function within larger ecosystem
        
        # Temporal tracking
        self.age = 0.0
        self.growth_cycles_completed = 0
        self.last_update_time = 0.0
        self.seasonal_cycle_position = 0.0  # 0.0-1.0 position in seasonal cycle
        
        # Manifestation and effects
        self.sensory_properties = self._initialize_sensory_properties()
        self.symbolic_effects = self._initialize_symbolic_effects()
        self.produced_resources = {}  # Resources generated for other entities
        
        # Evolution potential
        self.mutation_potential = 0.2  # Likelihood of mutation during reproduction
        self.adaptation_pressure = 0.0  # Current evolutionary pressure
        self.evolutionary_direction = {}  # Current trends in adaptation
    
    # Additional methods for MotifFloraSystem would go here...

# Add flora_system attribute to CosmicEntity
def add_flora_system_to_entity(entity: Any, growth_style: FloralGrowthPattern = None, primary_motifs: List[str] = None):
    """Add a MotifFloraSystem to an entity if it doesn't already have one"""
    if not hasattr(entity, 'flora_system'):
        entity.flora_system = MotifFloraSystem(entity, growth_style=growth_style, primary_motifs=primary_motifs)
    return entity.flora_system

class EmotionalState(Enum):
    """Fundamental emotional states that entities can experience and project"""
    JOY = "joy"                 # Expansive, light, uplifting
    SORROW = "sorrow"           # Contracting, heavy, descending
    FEAR = "fear"               # Tense, scattered, retreating
    ANGER = "anger"             # Sharp, hot, advancing
    WONDER = "wonder"           # Opening, receptive, crystalline
    SERENITY = "serenity"       # Stable, flowing, balanced
    DETERMINATION = "determination"  # Focused, directed, persistent
    CONFUSION = "confusion"     # Dispersed, foggy, seeking
    LONGING = "longing"         # Reaching, resonant, yearning
    TRANSCENDENCE = "transcendence"  # Dissolving, unifying, infinite


class EmotionType(Enum):
    """Fundamental emotion types that can be experienced within the cosmic simulation"""
    JOY = "joy"                      # Positive, expansive emotional state
    SORROW = "sorrow"                # Grief, loss, negative but processing state
    FEAR = "fear"                    # Anticipatory negative state, protective
    ANGER = "anger"                  # Energetic, boundary-setting state
    CURIOSITY = "curiosity"          # Exploratory, learning-focused state
    LOVE = "love"                    # Connection-seeking, bonding state
    AWE = "awe"                      # Transcendent wonder, perspective-shifting
    CONTENTMENT = "contentment"      # Peaceful, satisfied state
    DISGUST = "disgust"              # Rejection-oriented, protective state
    SURPRISE = "surprise"            # Pattern-interruption, alertness state
    DESPAIR = "despair"              # Hope-loss, energy-draining state
    ECSTASY = "ecstasy"              # Peak positive state, boundary-dissolving
    MALICE = "malice"                # Intentional harm-seeking state
    COMPASSION = "compassion"        # Suffering-recognition with desire to help
    AMBIVALENCE = "ambivalence"      # Mixed, contradictory emotional state
    APATHY = "apathy"                # Depleted, disconnected emotional state
    NOSTALGIA = "nostalgia"          # Past-oriented, memory-tied state


class EmotionalResonanceBody:
    """
    Manages the emotional state, processing, and influence of entities within the cosmic simulation.
    
    The EmotionalResonanceBody functions as the feeling center of an entity, allowing it to experience,
    process, express, and transmit emotions throughout the symbolic ecosystem. Emotions act as a form
    of energy that can flow between entities, leave resonant traces in environments, and influence
    the development and behavior of entities over time.
    
    Through emotional resonance mechanics, entities can form deep connections, imprint their emotional
    states onto objects, locations, or other entities, and participate in collective emotional fields
    that emerge from group dynamics.
    
    Key features:
    - Multi-layered emotional state tracking
    - Emotional imprinting and environmental influence
    - Resonance detection and response
    - Emotional weather generation and sensitivity
    - Emotional memory and trauma processing
    - Harmonic and dissonant emotional field interactions
    """
    
    def __init__(self, owner_entity: Any, base_sensitivity: float = 0.5,
                 dominant_emotions: List[EmotionType] = None,
                 emotional_capacity: float = 1.0,
                 memory_persistence: float = 0.7):
        self.owner = owner_entity
        self.base_resonance = base_resonance or self._initialize_base_resonance()
        self.current_state = self._calculate_current_state()
        self.projection_radius = projection_radius
        self.receptivity = receptivity
        self.resonance_signature = {}  # Develops over time
        self.emotional_memory = []
        self.active_harmonics = []
        self.resonance_connections = {}  # entity_id -> connection_strength
        self.emotional_weather = {}
        self.state_history = deque(maxlen=100)
        self.state_transitions = defaultdict(int)  # Tracks emotional state changes
        self.harmonic_nodes = self._initialize_harmonic_nodes()
        self.dissonance_threshold = 0.7
        self.resonance_evolution_rate = 0.05
        
        # Keep track of the last update time
        self.last_update_time = getattr(self.owner, 'last_update_time', 0)
        
        # Initialize state history with current state
        self.state_history.append((self.current_state, self.last_update_time))
    
    def _initialize_base_resonance(self) -> Dict[EmotionalState, float]:
        """Initialize base emotional resonance values"""
        base = {}
        
        # Start with low values for all emotions
        for state in EmotionalState:
            base[state] = 0.1 + 0.1 * random.random()
        
        # Select 1-3 dominant emotions
        dominant_count = random.randint(1, 3)
        dominant_emotions = random.sample(list(EmotionalState), dominant_count)
        
        for emotion in dominant_emotions:
            base[emotion] = 0.4 + 0.4 * random.random()
        
        # If entity has motifs, use them to influence base resonance
        if hasattr(self.owner, 'motifs'):
            for motif in self.owner.motifs:
                self._apply_motif_influence(base, motif)
        
        # Normalize to ensure sum is around 1.0
        total = sum(base.values())
        if total > 0:
            for state in base:
                base[state] /= total
        
        return base
    
    def _apply_motif_influence(self, resonance: Dict[EmotionalState, float], motif: str):
        """Apply influence from a motif to the emotional resonance"""
        # Map common motifs to emotional states they enhance
        motif_influences = {
            "fire": [EmotionalState.ANGER, EmotionalState.DETERMINATION],
            "water": [EmotionalState.SERENITY, EmotionalState.SORROW],
            "air": [EmotionalState.JOY, EmotionalState.WONDER],
            "earth": [EmotionalState.DETERMINATION, EmotionalState.SERENITY],
            
            "light": [EmotionalState.JOY, EmotionalState.TRANSCENDENCE],
            "dark": [EmotionalState.FEAR, EmotionalState.CONFUSION],
            
            "growth": [EmotionalState.JOY, EmotionalState.DETERMINATION],
            "decay": [EmotionalState.SORROW, EmotionalState.TRANSCENDENCE],
            
            "cycle": [EmotionalState.SERENITY, EmotionalState.WONDER],
            "chaos": [EmotionalState.CONFUSION, EmotionalState.FEAR],
            "order": [EmotionalState.SERENITY, EmotionalState.DETERMINATION],
            
            "knowledge": [EmotionalState.WONDER, EmotionalState.DETERMINATION],
            "mystery": [EmotionalState.WONDER, EmotionalState.CONFUSION],
            "wisdom": [EmotionalState.SERENITY, EmotionalState.TRANSCENDENCE],
            
            "journey": [EmotionalState.DETERMINATION, EmotionalState.LONGING],
            "return": [EmotionalState.JOY, EmotionalState.SERENITY],
            
            "transformation": [EmotionalState.WONDER, EmotionalState.TRANSCENDENCE],
            "stasis": [EmotionalState.SERENITY, EmotionalState.FEAR],
            
            "connection": [EmotionalState.JOY, EmotionalState.LONGING],
            "isolation": [EmotionalState.SORROW, EmotionalState.LONGING],
        }
        
        # Check for partial matches in the motif
        matched_influences = []
        for key, influences in motif_influences.items():
            if key in motif.lower():
                matched_influences.extend(influences)
        
        # Apply influence to matched emotions
        for emotion in matched_influences:
            resonance[emotion] = min(1.0, resonance[emotion] + 0.15)
    
    def _initialize_harmonic_nodes(self) -> List[Dict]:
        """Initialize harmonic nodes for emotional resonance"""
        nodes = []
        
        # Create 3-7 harmonic nodes
        num_nodes = random.randint(3, 7)
        
        for i in range(num_nodes):
            # Select a primary and secondary emotion for this node
            primary = random.choice(list(EmotionalState))
            secondary_options = [e for e in EmotionalState if e != primary]
            secondary = random.choice(secondary_options)
            
            # Create the node
            node = {
                "id": f"node_{i}",
                "primary_emotion": primary,
                "secondary_emotion": secondary,
                "amplitude": 0.2 + 0.6 * random.random(),
                "frequency": 0.5 + 1.5 * random.random(),
                "phase": random.random() * 2 * math.pi,
                "connections": []
            }
            
            nodes.append(node)
        
        # Create connections between nodes (not fully connected)
        for i, node in enumerate(nodes):
            # Connect to 1-3 other nodes
            connection_count = min(len(nodes) - 1, random.randint(1, 3))
            connection_targets = random.sample([j for j in range(len(nodes)) if j != i], connection_count)
            
            for target in connection_targets:
                node["connections"].append({
                    "target": target,
                    "strength": 0.2 + 0.6 * random.random(),
                    "delay": 0.1 + 0.4 * random.random()
                })
        
        return nodes
    
    def _calculate_current_state(self) -> EmotionalState:
        """Calculate the current dominant emotional state"""
        if not self.base_resonance:
            return EmotionalState.SERENITY
        
        # Find the emotion with highest resonance
        dominant_emotion = max(self.base_resonance.items(), key=lambda x: x[1])[0]
        return dominant_emotion

# Add emotional_resonance attribute to CosmicEntity
def add_emotional_resonance_to_entity(entity: Any, base_resonance: Dict[EmotionalState, float] = None):
    """Add an EmotionalResonanceBody to an entity if it doesn't already have one"""
    if not hasattr(entity, 'emotional_resonance'):
        entity.emotional_resonance = EmotionalResonanceBody(entity, base_resonance)
    return entity.emotional_resonance

# -------------------------------------------------------------------------
# Environment Integration System
# -------------------------------------------------------------------------

def integrate_environment_with_planet(planet: Planet, world_state: WorldState = None):
    """
    Integrate environmental systems with a planet entity.
    
    Args:
        planet: The planet to integrate with
        world_state: Optional existing WorldState (creates new one if None)
    
    Returns:
        world_state: The created or updated WorldState
    """
    if not world_state:
        world_state = WorldState()
    
    # Link planet's climate data with environmental system
    planet.env_state = world_state
    
    # Use planet's surface features to initialize terrain
    if hasattr(planet, 'surface') and planet.surface:
        # Adjust terrain height based on surface features
        for x in range(world_state.width):
            for y in range(world_state.height):
                height = world_state.terrain_height_map[x, y]
                
                # Increase height for mountainous regions
                if 'mountains' in planet.surface:
                    if random.random() < planet.surface['mountains']:
                        height += random.random() * 0.5
                
                # Lower height for water regions
                if 'water' in planet.surface:
                    if random.random() < planet.surface['water']:
                        height -= random.random() * 0.3
                
                # Create crater depressions
                if 'craters' in planet.surface:
                    if random.random() < planet.surface['craters'] * 0.3:
                        crater_radius = random.randint(5, 20)
                        crater_x = random.randint(0, world_state.width - 1)
                        crater_y = random.randint(0, world_state.height - 1)
                        
                        dist = math.sqrt((x - crater_x)**2 + (y - crater_y)**2)
                        if dist < crater_radius:
                            depth = (1 - (dist / crater_radius)) * 0.3
                            height -= depth
                
                # Update height with constraints
                world_state.terrain_height_map[x, y] = max(0, min(1, height))
    
    # Set up climate zones based on planet's climate data
    if hasattr(planet, 'climate') and planet.climate:
        world_state.climate_zones = []
        
        for climate_type, coverage in planet.climate.items():
            # Create multiple zones for each climate type based on coverage
            zone_count = max(1, int(coverage * 10))
            for _ in range(zone_count):
                x = random.randint(0, world_state.width - 1)
                y = random.randint(0, world_state.height - 1)
                
                # Zone radius based on coverage
                radius = int(30 + coverage * 100)
                
                # Create climate zone
                zone = ClimateZone((x, y), radius, climate_type)
                world_state.climate_zones.append(zone)
    
    # Initialize ocean currents if planet has significant water
    if hasattr(planet, 'surface') and planet.surface.get('water', 0) > 0.3:
        # Create ocean current system
        ocean_depth = 3
        current_layers = []
        
        for depth in range(ocean_depth):
            layer = CurrentLayer(
                width=world_state.width,
                height=world_state.height,
                depth=depth,
                velocity_factor=1.0 - (depth * 0.3)  # Surface moves faster
            )
            current_layers.append(layer)
        
        # Initialize global circulation if planet has sufficient water
        if planet.surface.get('water', 0) > 0.6:
            # Create placeholder for world geometry
            class WorldGeometry:
                def __init__(self, width, height):
                    self.width = width
                    self.height = height
            
            geometry = WorldGeometry(world_state.width, world_state.height)
            circulation = ThermohalineCirculation(geometry)
            
            # Add circulation system to world state
            world_state.ocean_circulation = circulation
    
    # Add symbolic influence from planet's motifs
    if hasattr(planet, 'motifs'):
        for motif in planet.motifs:
            # Choose random locations to apply influence
            for _ in range(10):
                x = random.randint(0, world_state.width - 1)
                y = random.randint(0, world_state.height - 1)
                world_state.apply_symbolic_effect((x, y), motif, 0.7)
    
    # Add method to planet to update environment
    def update_environment(self, time_delta: float):
        """Update the planet's environmental systems."""
        if hasattr(self, 'env_state'):
            # Update environment
            events = self.env_state.update(time_delta)
            
            # Process significant environmental events
            for event in events:
                # Record major events in scroll memory if available
                if event.get('type') in ['storm_formation', 'season_change'] and hasattr(self, 'record_scroll_event'):
                    importance = 0.5
                    if event['type'] == 'storm_formation' and event.get('storm_type') == 'hurricane':
                        importance = 0.7
                    
                    self.record_scroll_event(
                        event_type=f"environmental_{event['type']}",
                        description=f"{event['type'].replace('_', ' ').title()}: {event.get('storm_type', event.get('season', ''))}",
                        importance=importance,
                        motifs_added=[event['type']]
                    )
                
                # Storms can affect surface features
                if event['type'] == 'storm_formation' and hasattr(self, 'surface'):
                    storm_type = event.get('storm_type')
                    intensity = event.get('intensity', 0.5)
                    
                    if storm_type == 'hurricane' and intensity > 0.7:
                        # Hurricanes erode coastlines
                        if 'water' in self.surface and self.surface['water'] > 0.3:
                            coastal_change = 0.01 * intensity
                            self.surface['water'] = min(1.0, self.surface['water'] + coastal_change)
                    
                    elif storm_type in ['rain', 'thunderstorm'] and 'volcanic' in self.surface:
                        # Rain can erode volcanic terrain
                        erosion = 0.005 * intensity
                        self.surface['volcanic'] = max(0, self.surface['volcanic'] - erosion)
                        self.surface['sedimentary'] = self.surface.get('sedimentary', 0) + erosion
    
    # Bind method to planet
    planet.update_environment = types.MethodType(update_environment, planet)
    
    # Extend planet's evolve method to include environment
    original_evolve = planet.evolve
    
    def evolve_with_environment(self, time_delta: float):
        """Evolve planet including environmental systems."""
        # Call original evolve method
        original_evolve(time_delta)
        
        # Update environment
        self.update_environment(time_delta)
    
    # Replace evolve method
    planet.evolve = types.MethodType(evolve_with_environment, planet)
    
    return world_state


def apply_environmental_effects_to_civilization(civilization: Civilization, world_state: WorldState):
    """
    Apply environmental effects to a civilization.
    
    Args:
        civilization: The civilization to affect
        world_state: The world state containing environmental data
    """
    if not world_state:
        return
    
    # Get planet
    planet = DRM.get_entity(civilization.planet_id) if civilization.planet_id else None
    if not planet or not hasattr(planet, 'env_state'):
        return
    
    # Choose a representative location for the civilization
    if hasattr(civilization, 'home_sector'):
        # Convert sector coordinates to world coordinates
        sector = civilization.home_sector
        if sector:
            x = (sector[0] % world_state.width + world_state.width // 4) % world_state.width
            y = (sector[1] % world_state.height + world_state.height // 3) % world_state.height
            position = (x, y)
            
            # Get local climate
            climate_data = world_state.get_local_climate(position)
            
            # Apply effects to civilization
            
            # 1. Environmental symbolism affects culture
            if hasattr(civilization, 'culture_engine'):
                # Get dominant symbols
                dominant_symbols = world_state.get_dominant_symbols(position)
                
                for symbol, strength in dominant_symbols:
                    # Add to cultural motifs if strong enough
                    if strength > 0.6 and random.random() < strength * 0.3:
                        if symbol not in civilization.culture_engine.cultural_motifs:
                            civilization.culture_engine.cultural_motifs.append(symbol)
                            
                            # Record symbolic integration in scroll memory
                            if hasattr(civilization, 'record_scroll_event'):
                                civilization.record_scroll_event(
                                    event_type="cultural_adaptation",
                                    description=f"Environment influenced culture through '{symbol}' symbolism",
                                    importance=0.4,
                                    motifs_added=[f"environmental_{symbol}"]
                                )
            
            # 2. Extreme climate affects population
            if climate_data.get('temperature') < 0.2 or climate_data.get('temperature') > 0.8:
                # Extreme temperatures can slow population growth
                civilization.population = int(civilization.population * 0.995)
                
            # 3. Storm effects
            for effect in climate_data.get('active_effects', []):
                if effect.get('type') == 'storm' and effect.get('intensity', 0) > 0.7:
                    # Strong storms temporarily disrupt civilization
                    if random.random() < effect.get('intensity', 0) * 0.2:
                        # Slow development slightly
                        civilization.development_level = max(0, civilization.development_level - 0.01)
                        
                        # Record major storm impact
                        if hasattr(civilization, 'record_scroll_event'):
                            civilization.record_scroll_event(
                                event_type="natural_disaster",
                                description=f"A powerful {effect.get('storm_type')} disrupted society",
                                importance=0.6,
                                motifs_added=["weather_vulnerability"]
                            )
            
            # 4. Precipitation affects technology focus
            precipitation = climate_data.get('precipitation', 0.5)
            if precipitation > 0.7 and hasattr(civilization, 'tech_focus'):
                # Wet environments might encourage water management technology
                if random.random() < 0.05 and civilization.tech_focus != DevelopmentArea.MATERIALS:
                    old_focus = civilization.tech_focus
                    civilization.tech_focus = DevelopmentArea.MATERIALS
                    
                    # Record tech focus shift
                    if hasattr(civilization, 'record_scroll_event'):
                        civilization.record_scroll_event(
                            event_type="technological_shift",
                            description=f"Abundant water resources shifted focus from {old_focus.value} to {DevelopmentArea.MATERIALS.value}",
                            importance=0.5,
                            motifs_added=["environmental_adaptation"]
                        )


# Initialize imports needed for environment integration
import types
import math
import random
