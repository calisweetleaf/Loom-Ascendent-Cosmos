# ================================================================
#  LOOM ASCENDANT COSMOS — RECURSIVE SYSTEM MODULE
#  Author: Morpheus (Creator), Somnus Development Collective
#  License: Proprietary Software License Agreement (Somnus Development Collective)
#  Integrity Hash (SHA-256): d3ab9688a5a20b8065990cd9b91805e3d892d6e72472f69dd9afe719250c5e37
# ================================================================

import random
import logging
import math
import uuid
import time
import os
import json
from enum import Enum, auto
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Tuple, Callable, Set, Deque
import numpy as np # For numerical operations, especially in environmental models
import types # For dynamic method binding if needed later

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='genesis_cosmos.log')
logger = logging.getLogger(__name__)

# ===== Global Constants =====
WORLD_SIZE = 100 # Default world size, can be overridden by specific system configs

# ===== Engine Module Imports (External) =====
from aether_engine import AetherEngine #, AetherPattern - AetherPattern not directly used here
# from quantum_physics import QuantumPhysics # Not directly used by classes in this consolidated file
# from quantum_bridge import QuantumBridge # Not directly used
# from quantum_and_physics import QuantumAndPhysics # Appears to be a typo, likely quantum_physics
# from perception_module import PerceptionModule # Not directly used
# from paradox_engine import ParadoxEngine # Not directly used
# from harmonic_engine import HarmonicEngine # Not directly used
# from main import CoreDispatcher # Optional, and 'main' is too generic, likely needs specific path if used

# ===== ENUMERATIONS (Consolidated) =====

class MutationType(Enum):
    """Types of mutations that can occur in biological and symbolic entities"""
    POINT = "point"
    DUPLICATION = "duplication"
    DELETION = "deletion"
    INVERSION = "inversion"
    INSERTION = "insertion"
    SYMBOLIC = "symbolic"
    RECURSIVE = "recursive"
    MOTIF = "motif"
    NARRATIVE = "narrative"
    QUANTUM = "quantum"

class BreathPhase(Enum):
    """Enumeration of possible breathing cycle phases.
       Also used for cosmic breath cycle in ScrollMemoryEvent context."""
    INHALE = "inhale" # auto() was used, but string values are more descriptive
    HOLD_IN = "hold_in" # auto()
    EXHALE = "exhale" # auto()
    HOLD_OUT = "hold_out" # auto()

class EntityType(Enum):
    """Types of entities in the simulation.
       Consolidated from multiple definitions."""
    PHYSICAL = "physical" # auto()
    CONCEPTUAL = "conceptual" # auto()
    HYBRID = "hybrid" # auto()
    CONSCIOUS = "conscious" # auto()
    COLLECTIVE = "collective" # auto()
    # From later definitions (Planetary Framework context)
    UNIVERSE = "universe"
    GALAXY_CLUSTER = "galaxy_cluster"
    GALAXY = "galaxy"
    STAR = "star"
    PLANET = "planet"
    MOON = "moon"
    ASTEROID = "asteroid"
    CIVILIZATION = "civilization"
    ANOMALY = "anomaly"

class EventType(Enum):
    """Types of events that can occur in the simulation.
       Consolidated from multiple definitions."""
    CREATION = "creation" # auto()
    TRANSFORMATION = "transformation" # auto()
    INTERACTION = "interaction" # auto()
    DISSOLUTION = "dissolution" # auto()
    AWAKENING = "awakening" # auto()
    CONVERGENCE = "convergence" # auto()
    # From later definitions
    DESTRUCTION = "destruction"
    DISCOVERY = "discovery"
    DIVERGENCE = "divergence"
    DORMANCY = "dormancy"
    EMERGENCE = "emergence"

class MotifCategory(Enum):
    """Categories of motifs in the symbolic system.
       Consolidated from multiple definitions."""
    ELEMENTAL = "elemental" # auto()
    STRUCTURAL = "structural" # auto()
    NARRATIVE = "narrative" # auto()
    ARCHETYPAL = "archetypal" # auto()
    HARMONIC = "harmonic" # auto()
    # From later, more detailed definitions
    LUMINOUS = "luminous"
    ABYSSAL = "abyssal"
    VITAL = "vital"
    ENTROPIC = "entropic"
    CRYSTALLINE = "crystalline"
    CHAOTIC = "chaotic"
    RECURSIVE = "recursive"
    TEMPORAL = "temporal"
    DIMENSIONAL = "dimensional"
    CONNECTIVE = "connective"
    SHADOW = "shadow"
    ASCENDANT = "ascendant"
    PRIMORDIAL = "primordial" # From MotifSeeder

class EntityState(Enum):
    """States of cosmic entities in the simulation"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DORMANT = "dormant"
    DESTROYED = "destroyed"
    TRANSFORMED = "transformed"
    EMERGING = "emerging"
    EVOLVING = "evolving"
    ASCENDING = "ascending"
    DESCENDING = "descending"
    CONVERGING = "converging"
    DIVERGING = "diverging"
    AWAKENING = "awakening" # Duplicates one from EventType context, but likely distinct here
    SLEEPING = "sleeping"
    FADING = "fading"
    BIRTH = "birth"
    DEATH = "death"
    REBIRTH = "rebirth"
    AWAKENED = "awakened" # Differs from AWAKENING

class GalaxyType(Enum):
    SPIRAL = "spiral"
    ELLIPTICAL = "elliptical"
    IRREGULAR = "irregular"
    PECULIAR = "peculiar"
    DWARF = "dwarf"

class StarType(Enum):
    O = "O"
    B = "B"
    A = "A"
    F = "F"
    G = "G"
    K = "K"
    M = "M"
    L = "L"
    T = "T"
    Y = "Y"
    NEUTRON = "neutron"
    WHITE_DWARF = "white_dwarf"
    BLACK_HOLE = "black_hole"

class PlanetType(Enum):
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

class CivilizationType(Enum):
    TYPE_0 = "pre_industrial"
    TYPE_1 = "planetary"
    TYPE_2 = "stellar"
    TYPE_3 = "galactic"
    TYPE_4 = "cosmic"

class DevelopmentArea(Enum):
    ENERGY = "energy"
    COMPUTATION = "computation"
    MATERIALS = "materials"
    BIOLOGY = "biology"
    SPACE_TRAVEL = "space_travel"
    WEAPONRY = "weaponry"
    COMMUNICATION = "communication"
    SOCIAL_ORGANIZATION = "social_organization"

class AnomalyType(Enum):
    WORMHOLE = "wormhole"
    BLACK_HOLE = "black_hole" # Repeated from StarType, but contextually an anomaly
    NEUTRON_STAR = "neutron_star" # Repeated from StarType
    DARK_MATTER_CLOUD = "dark_matter_cloud"
    QUANTUM_FLUCTUATION = "quantum_fluctuation"
    TIME_DILATION = "time_dilation"
    DIMENSIONAL_RIFT = "dimensional_rift"
    COSMIC_STRING = "cosmic_string"
    STRANGE_MATTER = "strange_matter"
    REALITY_BUBBLE = "reality_bubble"

class BeliefType(Enum):
    COSMOLOGY = "cosmology"
    MORALITY = "morality"
    ONTOLOGY = "ontology"
    EPISTEMOLOGY = "epistemology"
    ESCHATOLOGY = "eschatology"
    AXIOLOGY = "axiology"
    THEOLOGY = "theology"
    TELEOLOGY = "teleology"

class SocialStructure(Enum):
    TRIBAL = "tribal"
    FEUDAL = "feudal"
    IMPERIAL = "imperial"
    REPUBLIC = "republic"
    DIRECT_DEMOCRACY = "direct_democracy"
    TECHNOCRACY = "technocracy"
    OLIGARCHY = "oligarchy"
    HIVE_MIND = "hive_mind"
    DISTRIBUTED = "distributed"
    QUANTUM_CONSENSUS = "quantum_consensus"

class SymbolicArchetype(Enum):
    CREATOR = "creator"
    DESTROYER = "destroyer"
    TRICKSTER = "trickster"
    SAGE = "sage"
    HERO = "hero"
    RULER = "ruler"
    CAREGIVER = "caregiver"
    EXPLORER = "explorer"
    INNOCENT = "innocent"
    SHADOW = "shadow" # Also a MotifCategory, ensure distinct usage

class InteractionType(Enum):
    CONFLICT = "conflict"
    COOPERATION = "cooperation"
    COMPETITION = "competition"
    CULTURAL_EXCHANGE = "cultural_exchange"
    TRADE = "trade"
    SUBJUGATION = "subjugation"
    ISOLATION = "isolation"
    OBSERVATION = "observation"
    HYBRID = "hybrid"

class MetabolicProcess(Enum):
    """Consolidated and most comprehensive definition of MetabolicProcess."""
    PHOTOSYNTHESIS = "photosynthesis"
    RESPIRATION = "respiration"
    CHEMOSYNTHESIS = "chemosynthesis"
    RADIOSYNTHESIS = "radiosynthesis"
    QUANTUM_ENTANGLEMENT = "quantum_entanglement"
    SYMBOLIC_ABSORPTION = "symbolic_absorption"
    MOTIF_CYCLING = "motif_cycling"
    HARMONIC_RESONANCE = "harmonic_resonance"
    # From earlier, simpler definitions, potentially different context or to be merged:
    # ANABOLISM = auto()
    # CATABOLISM = auto()
    # THERMAL_CYCLING = "thermal_cycling"     # Energy from temperature differences
    # ETHERIC_EXTRACTION = "etheric"          # Energy from etheric fields

class MetabolicResource(Enum):
    PHYSICAL_MATTER = "physical_matter"
    SYMBOLIC_ESSENCE = "symbolic_essence"
    TEMPORAL_FLUX = "temporal_flux"
    NARRATIVE_THREAD = "narrative_thread"
    EMOTIONAL_RESIDUE = "emotional_residue"
    BELIEF_CURRENT = "belief_current"
    VOID_EXTRACT = "void_extract"
    MEMORY_FRAGMENT = "memory_fragment"
    MOTIF_CONCENTRATE = "motif_concentrate"
    QUANTUM_POTENTIAL = "quantum_potential"

class MetabolicPathway(Enum):
    TRANSMUTATION = "transmutation"
    RESONANCE = "resonance"
    ABSORPTION = "absorption"
    CATALYSIS = "catalysis"
    FUSION = "fusion"
    FILTRATION = "filtration"
    CRYSTALLIZATION = "crystallization"
    RECURSION = "recursion"
    ENTROPIC = "entropic"
    SYMBOLIC = "symbolic"

class FloralGrowthPattern(Enum):
    BRANCHING = "branching"
    SPIRAL = "spiral"
    LAYERED = "layered"
    FRACTAL = "fractal"
    RADIAL = "radial"
    LATTICE = "lattice"
    CHAOTIC = "chaotic"
    HARMONIC = "harmonic"
    MIRRORED = "mirrored"
    ADAPTIVE = "adaptive"

class NutrientType(Enum):
    PHYSICAL = "physical"
    SYMBOLIC = "symbolic"
    EMOTIONAL = "emotional"
    TEMPORAL = "temporal"
    ENTROPIC = "entropic"
    HARMONIC = "harmonic"
    VOID = "void"
    NARRATIVE = "narrative"
    QUANTUM = "quantum"
    METAPHORIC = "metaphoric"

class FloraEvolutionStage(Enum):
    SEED = "seed"
    EMERGENT = "emergent"
    MATURING = "maturing"
    FLOWERING = "flowering"
    SEEDING = "seeding"
    WITHERING = "withering"
    COMPOSTING = "composting"
    DORMANT = "dormant"
    RESURGENT = "resurgent"
    TRANSCENDENT = "transcendent"

class EmotionalState(Enum):
    JOY = "joy"
    SORROW = "sorrow"
    FEAR = "fear"
    ANGER = "anger"
    WONDER = "wonder"
    SERENITY = "serenity"
    DETERMINATION = "determination"
    CONFUSION = "confusion"
    LONGING = "longing"
    TRANSCENDENCE = "transcendence"

class EmotionType(Enum): # Note: Similar to EmotionalState, but kept separate as original file did.
    JOY = "joy"
    SORROW = "sorrow"
    FEAR = "fear"
    ANGER = "anger"
    CURIOSITY = "curiosity"
    LOVE = "love"
    AWE = "awe"
    CONTENTMENT = "contentment"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    DESPAIR = "despair"
    ECSTASY = "ecstasy"
    MALICE = "malice"
    COMPASSION = "compassion"
    AMBIVALENCE = "ambivalence"
    APATHY = "apathy"
    NOSTALGIA = "nostalgia"


# ===== DATA CLASSES & SIMPLE CLASSES (Consolidated & Ordered) =====

# Physical constants (used by CosmicEntity subclasses)
G = 6.67430e-11
C = 299792458
H = 6.62607015e-34
ALPHA = 7.2973525693e-3
OMEGA_LAMBDA = 0.6889


@dataclass
class Motif:
    """A symbolic pattern that can be applied to entities and events."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    category: MotifCategory
    attributes: Dict[str, float]
    resonance: float = 0.0
    creation_tick: int = 0
    last_updated: int = 0

    def calculate_resonance(self, current_tick: int) -> float:
        time_factor = math.exp(-0.01 * (current_tick - self.last_updated))
        self.resonance = 0.5 + (random.random() * 0.5 * time_factor)
        self.last_updated = current_tick
        return self.resonance

@dataclass
class Entity: # Basic Entity definition
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    entity_type: EntityType # Using the consolidated EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    motifs: Set[str] = field(default_factory=set) # Set of Motif IDs
    creation_tick: int = 0
    last_updated: int = 0
    # For CosmicEntity compatibility, these can be added or handled by subclasses
    traits: Dict[str, Any] = field(default_factory=dict)
    sectors: Set[Tuple] = field(default_factory=set) # For DRM

    def add_motif(self, motif_id: str):
        self.motifs.add(motif_id)

    def remove_motif(self, motif_id: str):
        if motif_id in self.motifs:
            self.motifs.remove(motif_id)

    # Methods from CosmicEntity base class
    def evolve(self, time_delta: float):
        self.last_updated_time = getattr(self, 'last_updated_time', 0) + time_delta

    def add_to_reality(self, sectors: List[Tuple] = None):
        if sectors:
            self.sectors.update(sectors)
        DRM.store_entity(self.id, self, list(self.sectors))

    def get_trait(self, trait_name: str, default=None):
        return self.traits.get(trait_name, default)

    def set_trait(self, trait_name: str, value):
        self.traits[trait_name] = value

    def has_motif(self, motif_name: str) -> bool: # motif_name here is likely an ID or a name
        # Assuming self.motifs stores motif IDs or names.
        # If self.motifs stores Motif objects, this needs adjustment.
        return motif_name in self.motifs


@dataclass
class Event: # Basic Event definition
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType # Using the consolidated EventType
    description: str
    entities: List[str] # List of entity IDs
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    tick: int = 0
    motifs: Set[str] = field(default_factory=set) # Set of Motif IDs

    def add_motif(self, motif_id: str):
        self.motifs.add(motif_id)

@dataclass
class MetabolicProcessInfo: # Renamed from MetabolicProcess to avoid conflict with Enum
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    process_type: MetabolicProcess # Using the Enum
    entities: List[str] # List of entity IDs affected
    rate: float = 1.0
    active: bool = True
    created_tick: int = 0

    def process(self, delta_time: float) -> Dict[str, Any]:
        if not self.active:
            return {"status": "inactive"}
        result = {
            "status": "active",
            "energy_consumed": self.rate * delta_time,
            "transformation_progress": min(1.0, random.random() * self.rate * delta_time)
        }
        return result

@dataclass
class ScrollMemoryEvent: # Assuming this is the canonical one
    timestamp: float
    event_type: str # Or EventType enum
    description: str
    importance: float = 0.5
    entities_involved: List[str] = field(default_factory=list)
    motifs_added: List[str] = field(default_factory=list)
    location: Optional[Any] = None # Coordinates or sector

    def to_dict(self) -> Dict:
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
        return f"[T:{self.timestamp:.2f}] {self.event_type}: {self.description}"


# ===== DIMENSIONAL REALITY MANAGER =====
class DimensionalRealityManager:
    """Singleton class for managing dimensional reality."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DimensionalRealityManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.entities: Dict[str, Any] = {}  # id -> entity object
        self.entity_sector_map: Dict[str, Set[Tuple]] = defaultdict(set)
        self.sector_entity_map: Dict[Tuple, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.query_cache: Dict[Tuple, List[Any]] = {}
        self.time_dilation_factor: float = 1.0
        self.reality_coherence: float = 1.0
        self.active_observers: Set[str] = set()

    def store_entity(self, entity_id: str, entity_obj: Any, sectors: Optional[List[Tuple]] = None):
        self.entities[entity_id] = entity_obj
        entity_type_val = EntityType.ANOMALY.value # Default
        if hasattr(entity_obj, 'entity_type') and isinstance(entity_obj.entity_type, EntityType):
            entity_type_val = entity_obj.entity_type.value

        if sectors:
            for sector in sectors:
                self.entity_sector_map[entity_id].add(sector)
                self.sector_entity_map[sector][entity_type_val].add(entity_id)

    def query_entities(self, entity_type_str: str, sector: Optional[Tuple] = None) -> List[Any]:
        cache_key = (entity_type_str, sector)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        results: List[Any]
        if sector is None:
            results = [e for e in self.entities.values()
                       if hasattr(e, 'entity_type') and e.entity_type.value == entity_type_str]
        else:
            entity_ids = self.sector_entity_map[sector].get(entity_type_str, set())
            results = [self.entities[eid] for eid in entity_ids if eid in self.entities]

        self.query_cache[cache_key] = results
        return results

    def invalidate_cache(self, sector: Optional[Tuple] = None):
        if sector is None:
            self.query_cache.clear()
        else:
            keys_to_remove = [k for k in self.query_cache if k[1] == sector]
            for key in keys_to_remove:
                del self.query_cache[key]

    def get_entity(self, entity_id: str) -> Optional[Any]:
        return self.entities.get(entity_id)

    def update_entity_sectors(self, entity_id: str, old_sector: Tuple, new_sector: Tuple) -> bool:
        if entity_id not in self.entities:
            return False
        entity_obj = self.entities[entity_id]
        entity_type_val = EntityType.ANOMALY.value
        if hasattr(entity_obj, 'entity_type') and isinstance(entity_obj.entity_type, EntityType):
            entity_type_val = entity_obj.entity_type.value

        if old_sector in self.sector_entity_map and entity_type_val in self.sector_entity_map[old_sector]:
            self.sector_entity_map[old_sector][entity_type_val].discard(entity_id)
            self.entity_sector_map[entity_id].discard(old_sector)

        self.sector_entity_map[new_sector][entity_type_val].add(entity_id)
        self.entity_sector_map[entity_id].add(new_sector)

        self.invalidate_cache(old_sector)
        self.invalidate_cache(new_sector)
        return True

    def register_observer(self, observer_id: str, position: Tuple):
        self.active_observers.add(observer_id)
        self._adjust_reality_coherence(position)

    def _adjust_reality_coherence(self, focal_point: Tuple, radius: int = 3):
        pass # Placeholder

    def _get_sectors_in_radius(self, center: Tuple, radius: int) -> List[Tuple]:
        sectors = []
        # Assuming center is (x,y,z)-like
        if len(center) < 3: return sectors # Basic check
        x, y, z = center[0], center[1], center[2]
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx*dx + dy*dy + dz*dz <= radius*radius:
                        sectors.append((x + dx, y + dy, z + dz))
        return sectors

DRM = DimensionalRealityManager() # Singleton instance


# ===== CORE SIMULATION CLASSES (Consolidated & Ordered) =====

class CosmicScroll:
    """ Core symbolic pattern repository. Using the more feature-rich definition. """
    def __init__(self):
        self.entities: Dict[str, Entity] = {} # Using consolidated Entity
        self.world_state: Optional[WorldState] = None # Defined later
        self.motif_library: Dict[str, Motif] = {} # Using consolidated Motif
        self.motif_feedback_queue: Deque[Dict] = deque(maxlen=50)
        self.entity_motifs: Dict[str, Set[str]] = defaultdict(set) # entity_id -> set of motif_ids
        self.entity_types: Dict[EntityType, Set[str]] = defaultdict(set) # entity_type -> set of entity_ids
        self.event_history: List[Event] = [] # Using consolidated Event
        self.tick_count: int = 0
        self.time_scale: float = 1.0 # From an earlier CosmicScroll definition
        self.breath_phase: BreathPhase = BreathPhase.INHALE
        self.breath_progress: float = 0.0
        self.history: Dict[str, Any] = {
            "creation_time": datetime.now(),
            "tick_history": [],
            "significant_events": []
        }
        # From another CosmicScroll definition
        self.patterns: Dict[str, Any] = {} # pattern_id -> pattern_data
        self.active_threads: List[str] = []
        self.dormant_threads: List[str] = []
        self.symbolic_density: float = 0.0
        self.processes: Dict[str, MetabolicProcessInfo] = {} # Using renamed MetabolicProcessInfo

    def tick(self, delta_time: float = 1.0): # Merged tick logic
        adjusted_delta = delta_time * self.time_scale
        self.tick_count += 1

        # Update lifecycle for entities with 'civilization' in their name
        if self.world_state: # world_state might not be set
            current_sim_time = getattr(self.world_state, 'current_time', self.tick_count * adjusted_delta)
            for entity_obj in self.entities.values():
                if "civilization" in entity_obj.name.lower() and hasattr(entity_obj, 'birth_time'):
                    entity_obj.last_update_time = max(
                        getattr(entity_obj, 'last_update_time', 0.0),
                        current_sim_time
                    )
                    entity_obj.age = max(
                        getattr(entity_obj, 'age', 0.0),
                        entity_obj.last_update_time - entity_obj.birth_time
                    )
                    if hasattr(entity_obj, 'growth_cycle_duration') and entity_obj.growth_cycle_duration > 0:
                        entity_obj.growth_cycles_completed = max(
                            getattr(entity_obj, 'growth_cycles_completed', 0),
                            entity_obj.age // entity_obj.growth_cycle_duration
                        )
                        entity_obj.growth_factor = min(1.0, entity_obj.age / entity_obj.growth_cycle_duration)
                    if hasattr(entity_obj, 'lifespan') and entity_obj.lifespan > 0:
                        entity_obj.health = max(0.0, 1.0 - (entity_obj.age / entity_obj.lifespan))
                        maturation = 1.0 - (entity_obj.age / entity_obj.lifespan)
                        entity_obj.maturation_rate = min(1.0, max(0.0, maturation))
        return {"tick": self.tick_count, "delta_time": adjusted_delta}


    def add_pattern(self, pattern_id: str, pattern_data: Dict[str, Any]):
        self.patterns[pattern_id] = pattern_data

    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        return self.patterns.get(pattern_id)

    def activate_thread(self, thread_id: str) -> bool:
        if thread_id in self.dormant_threads:
            self.dormant_threads.remove(thread_id)
            self.active_threads.append(thread_id)
            return True
        return False

    def calculate_symbolic_density(self) -> float:
        pattern_count = len(self.patterns)
        thread_count = len(self.active_threads) + len(self.dormant_threads)
        if pattern_count == 0: return 0.0
        return (0.7 * pattern_count + 0.3 * thread_count) / 100.0

    def get_motif_feedback(self, max_items: int = 10) -> List[Dict]:
        recent_motifs = list(self.motif_feedback_queue)[-max_items:]
        feedback = {
            "tick_count": self.tick_count,
            "breath_phase": self.breath_phase.value, # Use .value for Enum
            "breath_progress": self.breath_progress,
            "motifs": recent_motifs,
            "motif_count": len(self.motif_library),
            "entity_count": len(self.entities),
            "dominant_categories": self._get_dominant_motif_categories()
        }
        return feedback

    def _get_dominant_motif_categories(self) -> Dict[str, float]:
        category_strengths: Dict[str, float] = defaultdict(float)
        for motif_id in self.motif_library: # Iterate over keys
            motif = self.motif_library[motif_id]
            # Assuming motif objects have 'category', 'strength', 'resonance'
            if hasattr(motif, 'category') and isinstance(motif.category, MotifCategory) and \
               hasattr(motif, 'attributes') and isinstance(motif.attributes, dict) and \
               hasattr(motif, 'resonance'):
                strength_attr = motif.attributes.get("strength", 0.5) # Default strength if not in attributes
                category_strengths[motif.category.value] += strength_attr * motif.resonance
        total_strength = sum(category_strengths.values()) or 1.0
        return {category: strength / total_strength for category, strength in category_strengths.items()}

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]: # Return type Entity
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: Union[str, EntityType]) -> List[Entity]: # Return type List[Entity]
        type_val = entity_type.value if isinstance(entity_type, EntityType) else entity_type
        entity_ids = set() # Ensure entity_ids is a set for type consistency
        # Check if self.entity_types is Dict[EntityType, Set[str]]
        if isinstance(self.entity_types, dict) and all(isinstance(k, EntityType) for k in self.entity_types.keys()):
             entity_ids = self.entity_types.get(EntityType(type_val), set()) if EntityType(type_val) in self.entity_types else set()

        return [self.entities[eid] for eid in entity_ids if eid in self.entities]


    def get_entity_motifs(self, entity_id: str) -> List[Motif]: # Return List[Motif]
        motif_ids = self.entity_motifs.get(entity_id, set())
        return [self.motif_library[mid] for mid in motif_ids if mid in self.motif_library]

    def get_events_by_entity(self, entity_id: str, max_events: int = 10) -> List[Event]: # Return List[Event]
        events_for_entity = [
            event_obj for event_obj in self.event_history
            if entity_id in event_obj.entities # entities is List[str] of IDs
        ]
        return sorted(events_for_entity, key=lambda e: e.timestamp, reverse=True)[:max_events]


    def get_simulation_stats(self) -> Dict:
        entity_type_counts = {etype.value: len(eids) for etype, eids in self.entity_types.items()}
        return {
            "tick_count": self.tick_count,
            "entity_count": len(self.entities),
            "entity_types": entity_type_counts,
            "event_count": len(self.event_history),
            "motif_count": len(self.motif_library),
            "breath_phase": self.breath_phase.value,
            "creation_time": self.history["creation_time"].isoformat(),
            "runtime": (datetime.now() - self.history["creation_time"]).total_seconds(),
            "significant_events": len(self.history["significant_events"])
        }

    def _generate_motif_name(self, category: MotifCategory) -> str:
        # This method was defined floating, now part of CosmicScroll
        # (name_components definition was also floating, moved here for encapsulation or could be global)
        name_components_local = {
            MotifCategory.LUMINOUS: {"prefixes": ["radiant", "glowing"], "roots": ["light", "sun"], "suffixes": ["beam", "ray"]},
            MotifCategory.ABYSSAL: {"prefixes": ["deep", "dark"], "roots": ["abyss", "void"], "suffixes": ["pit", "chasm"]},
            MotifCategory.VITAL: {"prefixes": ["living", "growing"], "roots": ["life", "bloom"], "suffixes": ["seed", "heart"]},
            MotifCategory.ENTROPIC: {"prefixes": ["decaying", "fading"], "roots": ["entropy", "dust"], "suffixes": ["dissolution", "end"]},
            MotifCategory.CRYSTALLINE: {"prefixes": ["ordered", "structured"], "roots": ["crystal", "form"], "suffixes": ["lattice", "grid"]},
            MotifCategory.CHAOTIC: {"prefixes": ["wild", "turbulent"], "roots": ["chaos", "storm"], "suffixes": ["vortex", "frenzy"]},
            MotifCategory.ELEMENTAL: {"prefixes": ["primal", "raw"], "roots": ["element", "earth"], "suffixes": ["essence", "force"]},
            MotifCategory.HARMONIC: {"prefixes": ["resonant", "balanced"], "roots": ["harmony", "chord"], "suffixes": ["wave", "pulse"]},
            MotifCategory.RECURSIVE: {"prefixes": ["nested", "iterative"], "roots": ["recursion", "loop"], "suffixes": ["iteration", "echo"]},
            MotifCategory.TEMPORAL: {"prefixes": ["flowing", "passing"], "roots": ["time", "moment"], "suffixes": ["flow", "cycle"]},
            MotifCategory.DIMENSIONAL: {"prefixes": ["spatial", "volumetric"], "roots": ["space", "realm"], "suffixes": ["expanse", "horizon"]},
            MotifCategory.CONNECTIVE: {"prefixes": ["linking", "binding"], "roots": ["connection", "web"], "suffixes": ["thread", "nexus"]},
            MotifCategory.SHADOW: {"prefixes": ["hidden", "veiled"], "roots": ["shadow", "mask"], "suffixes": ["cloak", "shroud"]},
            MotifCategory.ASCENDANT: {"prefixes": ["rising", "elevating"], "roots": ["ascension", "peak"], "suffixes": ["flight", "journey"]},
            MotifCategory.PRIMORDIAL: {"prefixes": ["ancient", "first"], "roots": ["origin", "source"], "suffixes": ["seed", "spark"]},
        }
        components = name_components_local.get(category, {
            "prefixes": ["mysterious"], "roots": ["pattern"], "suffixes": ["manifestation"]
        })
        name_structure = random.choice(["{prefix}_{root}", "{root}_{suffix}", "{prefix}_{root}_{suffix}"])
        name_parts = {
            "prefix": random.choice(components["prefixes"]),
            "root": random.choice(components["roots"]),
            "suffix": random.choice(components["suffixes"])
        }
        return name_structure.format(**name_parts)


class CosmicScrollManager: # Using the most feature-complete definition
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CosmicScrollManager, cls).__new__(cls)
            # Ensure _initialize is called only once
            cls._instance._initialized = False
        return cls._instance

    def _initialize(self):
        if self._initialized:
            return
        self.entities: Dict[str, Entity] = {}
        self.entity_types: Dict[EntityType, Set[str]] = defaultdict(set)
        self.motif_library: Dict[str, Motif] = {}
        self.entity_motifs: Dict[str, Set[str]] = defaultdict(set)
        self.event_history: List[Event] = []
        self.recent_events: Deque[Event] = deque(maxlen=100)
        self.tick_count: int = 0
        self.time_scale: float = 1.0
        self.breath_cycle_length: int = 12
        self.breath_phase: BreathPhase = BreathPhase.INHALE
        self.breath_progress: float = 0.0
        self.inhale_ratio: float = 0.3
        self.hold_in_ratio: float = 0.2
        self.exhale_ratio: float = 0.3
        self.hold_out_ratio: float = 0.2
        self.history: Dict[str, Any] = {
            "creation_time": datetime.now(),
            "tick_history": [],
            "significant_events": []
        }
        self.motif_feedback_queue: Deque[Dict] = deque(maxlen=50)
        self.cosmic_scroll: CosmicScroll = CosmicScroll() # Instantiate the scroll
        self.processes: Dict[str, MetabolicProcessInfo] = {}
        self._initialized = True
        logger.info("CosmicScrollManager initialized")


    def tick(self, delta_time: float = 1.0) -> Dict:
        adjusted_delta = delta_time * self.time_scale
        self.tick_count += 1
        self._update_breath_cycle() # Removed adjusted_delta, as original didn't use it.
                                   # If it should, it needs to be passed.

        evolved_entities_ids = self._evolve_entities(adjusted_delta) # Expects list of IDs
        generated_events_data = self._generate_events() # Expects list of event data dicts

        for event_data_dict in generated_events_data:
            self.log_event(event_data_dict) # log_event will handle conversion to Event object

        tick_info = {
            "tick_id": self.tick_count,
            "timestamp": datetime.now().isoformat(), # Using isoformat for consistency
            "delta_time": adjusted_delta,
            "breath_phase": self.breath_phase.value,
            "breath_progress": self.breath_progress,
            "entities_evolved_count": len(evolved_entities_ids),
            "events_generated_count": len(generated_events_data),
            "symbolic_density": self.cosmic_scroll.calculate_symbolic_density()
        }
        self.history["tick_history"].append(tick_info)
        if len(self.history["tick_history"]) > 100:
            self.history["tick_history"] = self.history["tick_history"][-100:]
        logger.debug(f"Tick {self.tick_count} completed")
        return tick_info


    def _update_breath_cycle(self): # Removed delta_time, assuming it's based on ticks
        total_progress = (self.tick_count % self.breath_cycle_length) / self.breath_cycle_length
        if total_progress < self.inhale_ratio:
            self.breath_phase = BreathPhase.INHALE
            self.breath_progress = total_progress / self.inhale_ratio if self.inhale_ratio > 0 else 0
        elif total_progress < (self.inhale_ratio + self.hold_in_ratio):
            self.breath_phase = BreathPhase.HOLD_IN
            self.breath_progress = (total_progress - self.inhale_ratio) / self.hold_in_ratio if self.hold_in_ratio > 0 else 0
        elif total_progress < (self.inhale_ratio + self.hold_in_ratio + self.exhale_ratio):
            self.breath_phase = BreathPhase.EXHALE
            self.breath_progress = (total_progress - self.inhale_ratio - self.hold_in_ratio) / self.exhale_ratio if self.exhale_ratio > 0 else 0
        else:
            self.breath_phase = BreathPhase.HOLD_OUT
            self.breath_progress = (total_progress - self.inhale_ratio - self.hold_in_ratio - self.exhale_ratio) / self.hold_out_ratio if self.hold_out_ratio > 0 else 0


    def _evolve_entities(self, delta_time: float) -> List[str]:
        evolved_entity_ids = []
        for entity_id, entity_obj in self.entities.items():
            if hasattr(entity_obj, 'evolve') and callable(getattr(entity_obj, 'evolve')):
                try:
                    entity_obj.evolve(delta_time) # Assuming evolve method is part of Entity
                    evolved_entity_ids.append(entity_id)
                except Exception as e:
                    logger.error(f"Error evolving entity {entity_id}: {e}")
        return evolved_entity_ids

    def _generate_events(self) -> List[Dict]: # Returns list of event data dicts
        events_data = []
        if random.random() < 0.1 and len(self.entities) >= 2:
            involved_entity_ids = random.sample(list(self.entities.keys()), 2)
            event_type_enum = random.choice(list(EventType))
            event_data = {
                "type": event_type_enum.value, # Store enum value
                "timestamp_val": self.tick_count, # Changed key to avoid conflict
                "entities_involved": involved_entity_ids,
                "description": f"Random interaction between {involved_entity_ids[0]} and {involved_entity_ids[1]}",
                "importance": random.uniform(0.1, 1.0)
            }
            events_data.append(event_data)
        return events_data

    def register_entity(self, entity_obj: Entity) -> str: # Takes Entity object
        if not entity_obj.id: # Assuming id is already part of Entity
            entity_obj.id = f"{entity_obj.name.lower()}_{uuid.uuid4().hex}"

        self.entities[entity_obj.id] = entity_obj
        if isinstance(entity_obj.entity_type, EntityType):
             self.entity_types[entity_obj.entity_type].add(entity_obj.id)

        creation_event_data = {
            "type": EventType.CREATION.value,
            "timestamp_val": self.tick_count,
            "entities_involved": [entity_obj.id],
            "description": f"Creation of {entity_obj.name}",
            "importance": 0.7
        }
        self.log_event(creation_event_data)
        logger.info(f"Entity registered: {entity_obj.id}")
        return entity_obj.id

    def log_event(self, event_data: Dict) -> str: # Takes event data dict
        event_id = event_data.get("id", f"event_{uuid.uuid4().hex}")
        event_data["id"] = event_id
        event_data["timestamp"] = event_data.get("timestamp_val", self.tick_count) # Use consistent key
        # Convert event_data to Event object
        # Ensure 'event_type' is present and is an Enum member or valid string for EventType
        evt_type_val = event_data.pop('type', EventType.INTERACTION.value)
        evt_description = event_data.pop('description', "Event description missing")
        evt_entities = event_data.pop('entities_involved', [])

        event_obj = Event(id=event_id,
                          event_type=EventType(evt_type_val),
                          description=evt_description,
                          entities=evt_entities,
                          properties=event_data, # Store rest as properties
                          tick=event_data["timestamp"])

        self.event_history.append(event_obj)
        self.recent_events.append(event_obj)

        if event_obj.properties.get("importance", 0) > 0.7:
            self.history["significant_events"].append(event_obj.to_dict()) # Store as dict

        if event_obj.properties.get("importance", 0) > 0.3:
            self.generate_motif(event_obj) # Pass Event object

        logger.debug(f"Event logged: {event_id}")
        return event_id

    def generate_motif(self, event_obj: Event) -> Optional[str]: # Takes Event object
        if not event_obj.entities:
            return None
        event_type_val = event_obj.event_type.value

        category_mapping = {
            EventType.CREATION.value: [MotifCategory.LUMINOUS, MotifCategory.VITAL],
            # ... (rest of mappings)
        }
        potential_categories = category_mapping.get(event_type_val, list(MotifCategory))
        chosen_category = random.choice(potential_categories)
        motif_name = self.cosmic_scroll._generate_motif_name(chosen_category) # Call from cosmic_scroll instance

        motif_obj = Motif(name=motif_name, category=chosen_category,
                          attributes={"strength": event_obj.properties.get("importance", 0.5),
                                      "source_event_id": event_obj.id,
                                      "creation_tick": self.tick_count},
                          resonance=random.uniform(0.3, 0.9))
        self.motif_library[motif_obj.id] = motif_obj

        for entity_id_str in event_obj.entities:
            if entity_id_str in self.entities:
                self.entity_motifs[entity_id_str].add(motif_obj.id)
                entity_instance = self.entities[entity_id_str]
                if hasattr(entity_instance, 'motifs') and isinstance(entity_instance.motifs, set):
                     entity_instance.motifs.add(motif_obj.id) # Store ID

        self.motif_feedback_queue.append(motif_obj.id) # Store ID
        logger.debug(f"Motif generated: {motif_obj.name} ({motif_obj.id})")
        return motif_obj.id


    def _process_metabolic_processes(self, delta_time: float) -> List[Dict[str, Any]]:
        results = []
        for process_id, process_obj in self.processes.items():
            if process_obj.active:
                result = process_obj.process(delta_time)
                result["process_id"] = process_id
                results.append(result)
        return results

    def _update_motifs(self) -> Dict[str, float]:
        motif_data = {}
        for motif_id, motif_obj in self.motif_library.items():
            resonance = motif_obj.calculate_resonance(self.tick_count)
            motif_data[motif_id] = resonance
        return motif_data

    def create_entity(self, name: str, entity_type: EntityType, properties: Dict[str, Any] = None) -> str:
        entity_obj = Entity(name=name, entity_type=entity_type, properties=properties or {})
        entity_obj.creation_tick = self.tick_count
        return self.register_entity(entity_obj)


    def create_motif(self, name: str, category: MotifCategory, attributes: Dict[str, float]) -> str:
        motif_obj = Motif(name=name, category=category, attributes=attributes)
        motif_obj.creation_tick = self.tick_count
        self.motif_library[motif_obj.id] = motif_obj
        logger.info(f"Created motif: {name} (ID: {motif_obj.id}, Category: {category.name})")
        return motif_obj.id

    def create_metabolic_process(self, name: str, process_type: MetabolicProcess, entities: List[str], rate: float = 1.0) -> str:
        # Changed to accept MetabolicProcess enum
        process_obj = MetabolicProcessInfo(name=name, process_type=process_type, entities=entities, rate=rate)
        process_obj.created_tick = self.tick_count
        self.processes[process_obj.id] = process_obj
        logger.info(f"Created metabolic process: {name} (ID: {process_obj.id}, Entities: {len(entities)})")
        return process_obj.id

    def associate_motif_with_entity(self, entity_id: str, motif_id: str) -> bool:
        if entity_id not in self.entities or motif_id not in self.motif_library:
            return False
        self.entities[entity_id].add_motif(motif_id) # Assuming Entity has add_motif
        self.entity_motifs[entity_id].add(motif_id)
        return True

    def get_entity_motifs(self, entity_id: str) -> List[Dict[str, Any]]: # Returns list of motif data
        if entity_id not in self.entities:
            return []
        result = []
        for motif_id_str in self.entity_motifs.get(entity_id, set()):
            motif_obj = self.motif_library.get(motif_id_str)
            if motif_obj:
                result.append({
                    "id": motif_obj.id, "name": motif_obj.name,
                    "category": motif_obj.category.value, "resonance": motif_obj.resonance
                })
        return result

    def get_simulation_state(self) -> Dict[str, Any]:
        return {
            "tick_count": self.tick_count,
            "entity_count": len(self.entities),
            "motif_count": len(self.motif_library),
            "event_count": len(self.event_history),
            "breath_phase": self.breath_phase.value,
            "breath_progress": self.breath_progress,
            "symbolic_density": self.cosmic_scroll.calculate_symbolic_density(),
            "active_processes": sum(1 for p in self.processes.values() if p.active)
        }

    def get_motif_feedback(self, max_items: int = 10) -> List[Dict]: # From later CosmicScrollManager
        recent_motifs_data = []
        # Iterate over a copy of the deque for safety if it's modified elsewhere
        for motif_id in list(self.motif_feedback_queue)[-max_items:]:
            motif_obj = self.motif_library.get(motif_id) # Assuming feedback queue stores IDs
            if motif_obj:
                 recent_motifs_data.append({"id": motif_obj.id, "name": motif_obj.name, "category": motif_obj.category.value, "resonance": motif_obj.resonance})

        feedback = {
            "tick_count": self.tick_count,
            "breath_phase": self.breath_phase.value,
            "breath_progress": self.breath_progress,
            "motifs": recent_motifs_data,
            "motif_count": len(self.motif_library),
            "entity_count": len(self.entities),
            "dominant_categories": self.cosmic_scroll._get_dominant_motif_categories() # Call from instance
        }
        return feedback

cosmic_scroll_manager = CosmicScrollManager() # Singleton instance

# ===== COSMIC ENTITY DEFINITIONS (Galaxy, Star, Planet, etc.) =====
# These classes will inherit from the consolidated `Entity` class.
# I'll use the more detailed definitions found later in the original file.

class CosmicEntity(Entity): # Inherits from consolidated Entity
    def __init__(self, entity_id: Optional[str] = None, name: str = "Unnamed Cosmic Entity", entity_type: EntityType = EntityType.ANOMALY, properties: Optional[Dict[str, Any]] = None):
        actual_id = entity_id or f"{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        super().__init__(id=actual_id, name=name, entity_type=entity_type, properties=properties or {})
        # last_update_time is part of base Entity now via evolve()
        # self.scroll_memory = ScrollMemory(self.id) # ScrollMemory defined later

    @property
    def scroll_id(self): # To maintain compatibility if used
        return self.id

    # record_scroll_event and _propagate_event_to_related_entities will be added after ScrollMemory definition.


class Galaxy(CosmicEntity):
    def __init__(self, galaxy_id: Optional[str] = None, name: str = "Unnamed Galaxy"):
        super().__init__(entity_id=galaxy_id, name=name, entity_type=EntityType.GALAXY)
        self.galaxy_type: Optional[GalaxyType] = None
        self.stars: List[str] = [] # List of Star IDs
        self.age: float = 0.0
        self.size: float = 0.0 # e.g. in light years
        self.active_regions: Set[Tuple] = set() # Sectors
        self.metallicity: float = 0.0
        self.black_hole_mass: float = 0.0
        self.star_formation_rate: float = 0.0

    def evolve(self, time_delta: float):
        super().evolve(time_delta)
        self.age += time_delta
        if self.active_regions:
            for region in self.active_regions:
                stars_in_region = DRM.query_entities(EntityType.STAR.value, region)
                for star_obj in stars_in_region: # Iterate over objects
                    if hasattr(star_obj, 'galaxy_id') and star_obj.galaxy_id == self.id:
                        star_obj.evolve(time_delta)

    def add_star(self, star_obj: 'Star'): # Takes Star object
        star_obj.galaxy_id = self.id # Assuming Star has galaxy_id attribute
        self.stars.append(star_obj.id)


class Star(CosmicEntity):
    def __init__(self, star_id: Optional[str] = None, name: str = "Unnamed Star"):
        super().__init__(entity_id=star_id, name=name, entity_type=EntityType.STAR)
        self.star_type: Optional[StarType] = None
        self.planets: List[str] = [] # List of Planet IDs
        self.luminosity: float = 0.0
        self.mass: float = 0.0
        self.radius: float = 0.0
        self.temperature: float = 0.0
        self.age: float = 0.0
        self.life_expectancy: float = 10e9 # Default 10 billion years
        self.color: Tuple[int, int, int] = (255, 255, 255) # RGB
        self.galaxy_id: Optional[str] = None
        self.position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.habitable_zone: Tuple[float, float] = (0.0, 0.0)

    def evolve(self, time_delta: float):
        super().evolve(time_delta)
        self.age += time_delta
        self._update_stellar_properties()
        for planet_id_str in self.planets:
            planet_obj = DRM.get_entity(planet_id_str)
            if planet_obj:
                planet_obj.evolve(time_delta)

    def _update_stellar_properties(self):
        if self.life_expectancy > 0 and self.age > self.life_expectancy :
            self._initiate_stellar_death()
        else:
            if self.life_expectancy > 0 :
                age_fraction = self.age / self.life_expectancy
                luminosity_factor = 1 + (0.1 * age_fraction)
                self.luminosity *= luminosity_factor
            self._update_habitable_zone()

    def _update_habitable_zone(self):
        if self.luminosity > 0 :
            luminosity_factor = math.sqrt(self.luminosity)
            self.habitable_zone = (0.95 * luminosity_factor, 1.37 * luminosity_factor)

    def _initiate_stellar_death(self):
        # Simplified, from original
        if self.mass < 0.5: self.star_type = StarType.WHITE_DWARF
        elif self.mass < 8: self.star_type = StarType.WHITE_DWARF
        elif self.mass < 20: self.star_type = StarType.NEUTRON
        else: self.star_type = StarType.BLACK_HOLE
        # Add motifs for flavor
        if random.random() < 0.3: self.motifs.add('stellar_remnant')


class Planet(CosmicEntity): # Simplified Planet definition from later in file
    def __init__(self, planet_id: Optional[str] = None, name: str = "Unnamed Planet"):
        super().__init__(entity_id=planet_id, name=name, entity_type=EntityType.PLANET)
        self.planet_type: Optional[PlanetType] = None
        self.star_id: Optional[str] = None
        self.moons: List[str] = [] # Moon IDs
        self.mass: float = 0.0
        self.radius: float = 0.0
        self.density: float = 0.0
        self.gravity: float = 0.0
        self.orbital_period: float = 0.0
        self.rotation_period: float = 0.0
        self.axial_tilt: float = 0.0
        self.orbital_eccentricity: float = 0.0
        self.albedo: float = 0.0
        self.atmosphere: Dict[str, float] = {} # Component -> percentage
        self.surface: Dict[str, float] = {}  # Feature -> percentage
        self.climate: Dict[str, float] = {} # Zone -> percentage
        self.has_magnetosphere: bool = False
        self.temperature: float = 0.0 # Average surface temperature in Celsius
        self.civilizations: List[str] = [] # Civilization IDs
        self.habitability_index: float = 0.0
        self.biosphere_complexity: float = 0.0
        self.position: Tuple[float, float, float] = (0.0,0.0,0.0) # Added for _evolve_climate

    def evolve(self, time_delta: float):
        super().evolve(time_delta)
        self._evolve_geology(time_delta)
        self._evolve_climate(time_delta)
        if self.biosphere_complexity > 0:
            self._evolve_biosphere(time_delta)
        for civ_id_str in self.civilizations:
            civ_obj = DRM.get_entity(civ_id_str)
            if civ_obj:
                civ_obj.evolve(time_delta)

    def _evolve_geology(self, time_delta: float):
        if self.has_motif('tectonic_dreams') and random.random() < 0.01 * time_delta:
            event_type = random.choice(['volcanic', 'earthquake', 'erosion'])
            if event_type == 'volcanic':
                self.surface['volcanic'] = self.surface.get('volcanic', 0) + 0.01
                self.atmosphere['sulfur'] = self.atmosphere.get('sulfur', 0) + 0.005
            elif event_type == 'earthquake' and 'mountains' in self.surface :
                self.surface['mountains'] += 0.005
            elif event_type == 'erosion' and self.surface.get('mountains',0) > 0.01:
                 self.surface['mountains'] -= 0.005
                 self.surface['sedimentary'] = self.surface.get('sedimentary', 0) + 0.005

    def _evolve_climate(self, time_delta: float):
        star_obj = DRM.get_entity(self.star_id) if self.star_id else None
        if not star_obj or not hasattr(star_obj, 'luminosity') or not hasattr(star_obj, 'position'): return

        star_distance_sq = sum((a - b) ** 2 for a, b in zip(self.position, star_obj.position))
        if star_distance_sq == 0: star_distance_sq = 1.0 # Avoid division by zero
        luminosity_factor = star_obj.luminosity / star_distance_sq

        greenhouse_factor = 1.0
        greenhouse_factor += self.atmosphere.get('carbon_dioxide', 0.0) * 10
        greenhouse_factor += self.atmosphere.get('methane', 0.0) * 30

        base_temp_kelvin = -270 + (340 * luminosity_factor)
        self.temperature = base_temp_kelvin * greenhouse_factor - 273.15
        self._update_climate_zones()


    def _update_climate_zones(self): # Simplified from original
        self.climate = {}
        if self.temperature < -50: self.climate['polar'] = 0.8; self.climate['tundra'] = 0.2
        elif self.temperature < 15: self.climate['temperate'] = 0.6
        else: self.climate['desert'] = 0.6; self.climate['tropical'] = 0.3
        water_percentage = self.surface.get('water', 0)
        if water_percentage > 0.3 and 'desert' in self.climate:
             self.climate['desert'] = max(0, self.climate['desert'] - water_percentage * 0.3)
             self.climate['oceanic'] = self.climate.get('oceanic',0) + water_percentage * 0.3


    def _evolve_biosphere(self, time_delta: float):
        if self.habitability_index > 0:
            growth_rate = 0.001 * time_delta * self.habitability_index
            self.biosphere_complexity = min(1.0, self.biosphere_complexity + growth_rate)
            if (self.biosphere_complexity > 0.8 and
                self.has_motif('exogenesis') and # Check if motif is present
                not self.civilizations and
                random.random() < 0.001 * time_delta):
                self._spawn_civilization()

    def _spawn_civilization(self):
        civ_obj = Civilization(name=f"Civ-{self.name[:5]}-{random.randint(100,999)}") # Create with name
        civ_obj.planet_id = self.id
        civ_obj.home_sector = list(self.sectors)[0] if self.sectors else None
        civ_obj.traits['adaptations'] = []
        if self.temperature < -10: civ_obj.traits['adaptations'].append('cold_resistance')
        if self.temperature > 30: civ_obj.traits['adaptations'].append('heat_resistance')
        if self.gravity > 1.3: civ_obj.traits['adaptations'].append('high_gravity')
        civ_obj.development_level = 0.1
        civ_obj.add_to_reality(list(self.sectors))
        self.civilizations.append(civ_obj.id)


class Civilization(CosmicEntity):
    def __init__(self, civ_id: Optional[str] = None, name: str = "Unnamed Civilization"):
        super().__init__(entity_id=civ_id, name=name, entity_type=EntityType.CIVILIZATION)
        self.planet_id: Optional[str] = None
        self.civ_type: CivilizationType = CivilizationType.TYPE_0
        self.development_level: float = 0.0
        self.population: int = 0
        self.tech_focus: Optional[DevelopmentArea] = None
        self.tech_levels: Dict[DevelopmentArea, float] = {area: 0.0 for area in DevelopmentArea}
        self.colonized_planets: List[str] = []
        self.home_sector: Optional[Tuple] = None
        self.communication_range: float = 0.0
        self.ftl_capability: bool = False
        self.quantum_understanding: float = 0.0
        self.known_civilizations: List[str] = []
        # self.culture_engine: Optional[CultureEngine] = None # CultureEngine defined later

    def evolve(self, time_delta: float):
        super().evolve(time_delta)
        dev_rate = 0.01 * time_delta * (1 - self.development_level)
        self.development_level = min(1.0, self.development_level + dev_rate)
        self._check_advancement()
        self._evolve_population(time_delta)
        self._evolve_technology(time_delta)
        self._check_expansion()
        if random.random() < 0.01 * time_delta: self._check_contact()
        if hasattr(self, 'culture_engine') and self.culture_engine: # Check if culture_engine exists
            self.culture_engine.evolve_culture(time_delta)


    def _check_advancement(self):
        if self.development_level >= 0.99:
            current_type_val = self.civ_type.value
            if current_type_val == CivilizationType.TYPE_0.value: self.civ_type = CivilizationType.TYPE_1; self.motifs.add('industrial_revolution')
            elif current_type_val == CivilizationType.TYPE_1.value: self.civ_type = CivilizationType.TYPE_2; self.motifs.add('stellar_expansion'); self.ftl_capability = True
            elif current_type_val == CivilizationType.TYPE_2.value: self.civ_type = CivilizationType.TYPE_3; self.motifs.add('galactic_network')
            if self.civ_type.value != current_type_val: self.development_level = 0.1


    def _evolve_population(self, time_delta: float):
        if self.population == 0: self.population = 1000
        else:
            growth_rate = 0.02 * time_delta
            carrying_capacity = 10**10 * (1 + len(self.colonized_planets))
            if carrying_capacity > 0 :
                growth = growth_rate * self.population * (1 - self.population / carrying_capacity)
                self.population = max(1000, int(self.population + growth))

    def _evolve_technology(self, time_delta: float):
        if not self.tech_focus: self.tech_focus = random.choice(list(DevelopmentArea))
        for area in DevelopmentArea:
            advance_rate = 0.005 * time_delta
            if area == self.tech_focus: advance_rate *= 2
            # .value.count('_') was problematic, assuming progression by enum order
            type_order = list(CivilizationType)
            civ_type_idx = type_order.index(self.civ_type) if self.civ_type in type_order else 0
            advance_rate *= (1 + 0.5 * civ_type_idx)
            self.tech_levels[area] = min(1.0, self.tech_levels[area] + advance_rate)
        if random.random() < 0.01 * time_delta: self.tech_focus = random.choice(list(DevelopmentArea))
        if self.tech_levels[DevelopmentArea.SPACE_TRAVEL] > 0.7 and self.tech_levels[DevelopmentArea.ENERGY] > 0.8: self.ftl_capability = True
        self.communication_range = 10 * self.tech_levels[DevelopmentArea.COMMUNICATION]
        if self.ftl_capability: self.communication_range *= 100
        self.quantum_understanding = (self.tech_levels[DevelopmentArea.COMPUTATION] * 0.5 +
                                     self.tech_levels[DevelopmentArea.ENERGY] * 0.3 +
                                     self.tech_levels[DevelopmentArea.MATERIALS] * 0.2)

    def _check_expansion(self): # Simplified, assumes DRM is globally accessible
        if self.civ_type != CivilizationType.TYPE_0 and self.tech_levels[DevelopmentArea.SPACE_TRAVEL] > 0.5 and \
           random.random() < 0.05 * self.tech_levels[DevelopmentArea.SPACE_TRAVEL]:
            home_planet_obj = DRM.get_entity(self.planet_id) if self.planet_id else None
            if not home_planet_obj: return
            star_obj = DRM.get_entity(home_planet_obj.star_id) if hasattr(home_planet_obj, 'star_id') and home_planet_obj.star_id else None
            if not star_obj: return

            # Find candidates (simplified)
            # In a full system, this would query DRM for planets in star.planets or nearby_stars
            # For now, we'll assume it finds a hypothetical target_planet object.
            # This part is highly dependent on how entities are stored and queried.
            pass # Placeholder for expansion logic

    def _find_nearby_stars(self, reference_star: Star, max_distance: float) -> List[Star]: # Takes Star object
        nearby_stars_found = []
        ref_pos = reference_star.position
        galaxy_obj = DRM.get_entity(reference_star.galaxy_id) if reference_star.galaxy_id else None
        if not galaxy_obj or not hasattr(galaxy_obj, 'stars'): return nearby_stars_found

        all_star_ids_in_galaxy = galaxy_obj.stars # Assuming this stores star IDs
        for star_id_str in all_star_ids_in_galaxy:
            star_obj = DRM.get_entity(star_id_str)
            if star_obj and star_obj.id != reference_star.id:
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(ref_pos, star_obj.position)))
                if distance <= max_distance:
                    nearby_stars_found.append(star_obj)
        return nearby_stars_found


    def _check_contact(self): # Simplified
        all_civ_ids = []
        # Assuming DRM.query_entities returns list of actual entity objects
        for civ_obj in DRM.query_entities(EntityType.CIVILIZATION.value):
             all_civ_ids.append(civ_obj.id)

        for other_civ_id in all_civ_ids:
            if other_civ_id == self.id or other_civ_id in self.known_civilizations:
                continue
            # Further logic would involve checking distance and communication range
            # This is a placeholder for contact logic
            pass

    def _initialize_culture_engine(self): # From later in file
        if not hasattr(self, 'culture_engine'):
            self.culture_engine = CultureEngine(self)
        if hasattr(self, 'record_scroll_event') and callable(self.record_scroll_event):
            self.record_scroll_event(
                event_type="culture_genesis",
                description=f"Culture established with {self.culture_engine.social_structure.value} structure",
                importance=0.8,
                motifs_added=self.culture_engine.cultural_motifs[:3]
            )
        return self.culture_engine


# Other CosmicEntity subclasses (Anomaly, Universe, GalaxyCluster, Moon, Asteroid)
# would follow a similar pattern, inheriting from CosmicEntity and using consolidated Enums.
# For brevity in this step, I'll stub them but ensure they use the correct base.

class Anomaly(CosmicEntity):
    def __init__(self, anomaly_id: Optional[str] = None, name: str = "Unnamed Anomaly"):
        super().__init__(entity_id=anomaly_id, name=name, entity_type=EntityType.ANOMALY)
        self.anomaly_type: Optional[AnomalyType] = None
        # ... other attributes from the most complete Anomaly definition
        self.intensity = 0.0
        self.stability = 0.0
        self.radius = 0.0
        self.effects = []
        self.danger_level = 0.0
        self.is_traversable = False
        self.connects_to = None
        self.discovery_chance = 0.1

class Universe(CosmicEntity):
    def __init__(self, universe_id: Optional[str] = None, name: str = "The Universe"):
        super().__init__(entity_id=universe_id, name=name, entity_type=EntityType.UNIVERSE)
        self.galaxies: List[str] = [] # Galaxy IDs
        self.age: float = 0.0
        self.expansion_rate: float = 67.8
        # ... other attributes
        self.dark_energy_density = OMEGA_LAMBDA
        self.dark_matter_density = 0.2589
        self.baryonic_matter_density = 0.0486
        self.fundamental_constants = {'G': G, 'c': C, 'h': H, 'alpha': ALPHA}
        self.dimensions = 3 # Spatial dimensions, or total if including time elsewhere
        self.active_sectors: Set[Tuple] = set()


class GalaxyCluster(CosmicEntity):
    def __init__(self, cluster_id: Optional[str] = None, name: str = "Unnamed Galaxy Cluster"):
        super().__init__(entity_id=cluster_id, name=name, entity_type=EntityType.GALAXY_CLUSTER)
        self.galaxies: List[str] = [] # Galaxy IDs
        # ... other attributes
        self.size = 0.0
        self.mass = 0.0
        self.dark_matter_ratio = 0.85
        self.center_position = (0.0,0.0,0.0)
        self.universe_id: Optional[str] = None


class Moon(CosmicEntity):
    def __init__(self, moon_id: Optional[str] = None, name: str = "Unnamed Moon"):
        super().__init__(entity_id=moon_id, name=name, entity_type=EntityType.MOON)
        self.planet_id: Optional[str] = None
        # ... other attributes
        self.radius = 0.0
        self.mass = 0.0
        self.orbital_distance = 0.0
        self.orbital_period = 0.0
        self.rotation_period = 0.0
        self.surface = {}
        self.habitability_index = 0.0
        self.has_atmosphere = False
        self.atmosphere = {}
        self.temperature = 0.0


class Asteroid(CosmicEntity):
    def __init__(self, asteroid_id: Optional[str] = None, name: str = "Unnamed Asteroid"):
        super().__init__(entity_id=asteroid_id, name=name, entity_type=EntityType.ASTEROID)
        self.size: float = 0.0
        # ... other attributes
        self.mass = 0.0
        self.composition = {}
        self.orbit: Optional[str] = None # Star ID or Planet ID
        self.orbital_distance = 0.0
        self.orbital_period = 0.0
        self.is_hazardous = False
        self.trajectory = []
        self.position = (0.0,0.0,0.0) # Added for _update_position

    def _update_position(self, time_delta: float): # Example of fixing a method that needs position
        orbit_center_obj = DRM.get_entity(self.orbit) if self.orbit else None
        if not orbit_center_obj or not hasattr(orbit_center_obj, 'position'): return
        if self.orbital_period == 0: return # Avoid division by zero

        old_pos = self.position
        orbit_angle = (2 * math.pi / self.orbital_period) * time_delta
        cos_theta = math.cos(orbit_angle)
        sin_theta = math.sin(orbit_angle)
        offset = tuple(a - b for a, b in zip(old_pos, orbit_center_obj.position))
        new_offset = (
            offset[0] * cos_theta - offset[1] * sin_theta,
            offset[0] * sin_theta + offset[1] * cos_theta,
            offset[2] # Assuming 2D orbit in XY plane for simplicity
        )
        self.position = tuple(c + o for c, o in zip(orbit_center_obj.position, new_offset))
        # ... rest of trajectory and sector update logic

    def _check_collisions(self): pass # Placeholder
    def _handle_collision(self, planet: Planet): pass # Placeholder


# ===== ScrollMemory System (Consolidated) =====
class ScrollMemory: # Using the more complete definition
    def __init__(self, owner_id: str, capacity: int = 100):
        self.owner_id = owner_id
        self.events: List[ScrollMemoryEvent] = [] # Uses consolidated ScrollMemoryEvent
        self.capacity = capacity
        self.last_consolidation: float = 0.0 # Assuming it's a timestamp
        self.thematic_summary: Dict[str, int] = {}

    def record_event(self, event: ScrollMemoryEvent):
        self.events.append(event)
        self.thematic_summary[event.event_type] = self.thematic_summary.get(event.event_type, 0) + 1
        if len(self.events) > self.capacity:
            self._consolidate_memory()

    def get_events_by_type(self, event_type_str: str) -> List[ScrollMemoryEvent]:
        return [e for e in self.events if e.event_type == event_type_str]

    def get_events_by_timeframe(self, start_time: float, end_time: float) -> List[ScrollMemoryEvent]:
        return [e for e in self.events if start_time <= e.timestamp <= end_time]

    def get_events_involving_entity(self, entity_id: str) -> List[ScrollMemoryEvent]:
        return [e for e in self.events if entity_id in e.entities_involved]

    def get_most_important_events(self, count: int = 10) -> List[ScrollMemoryEvent]:
        return sorted(self.events, key=lambda e: e.importance, reverse=True)[:count]

    def _consolidate_memory(self): # Simplified logic
        if len(self.events) <= self.capacity: return
        self.events = sorted(self.events, key=lambda e: e.importance, reverse=True)[:self.capacity]
        self.last_consolidation = self.events[-1].timestamp if self.events else 0
        self._rebuild_thematic_summary()

    def _rebuild_thematic_summary(self):
        self.thematic_summary.clear()
        for event in self.events:
            self.thematic_summary[event.event_type] = self.thematic_summary.get(event.event_type, 0) + 1
    # ... other ScrollMemory methods like get_narrative_arc, get_memory_keywords, generate_timeline_summary

# Functions to augment CosmicEntity with ScrollMemory
def add_scroll_memory_to_entity(entity: CosmicEntity):
    if not hasattr(entity, 'scroll_memory'):
        entity.scroll_memory = ScrollMemory(entity.id)
    return entity.scroll_memory

def record_scroll_event(entity_instance: CosmicEntity, event_type: str, description: str, importance: float = 0.5,
                      entities_involved: List[str] = None, motifs_added: List[str] = None,
                      location=None, timestamp: float = None):
    if not hasattr(entity_instance, 'scroll_memory'):
        add_scroll_memory_to_entity(entity_instance)

    ts = timestamp if timestamp is not None else getattr(entity_instance, 'last_update_time', time.time())

    event = ScrollMemoryEvent(
        timestamp=ts, event_type=event_type, description=description, importance=importance,
        entities_involved=entities_involved or [], motifs_added=motifs_added or [], location=location
    )
    entity_instance.scroll_memory.record_event(event)
    if motifs_added: # Ensure motifs attribute is a set
        if not hasattr(entity_instance, 'motifs') or not isinstance(entity_instance.motifs, set):
            entity_instance.motifs = set()
        for motif in motifs_added:
            entity_instance.motifs.add(motif)
    # _propagate_event_to_related_entities needs to be defined or called carefully
    return event

# Bind to CosmicEntity class after its definition
CosmicEntity.record_scroll_event = record_scroll_event


# ===== CultureEngine System & Dependencies =====
class CultureEngine:
    def __init__(self, civilization: Civilization): # Depends on Civilization
        self.civilization = civilization
        self.belief_systems: Dict[BeliefType, float] = {bt: random.random() for bt in BeliefType}
        self.social_structure: Optional[SocialStructure] = None
        self.archetypes: List[SymbolicArchetype] = []
        self.naming_patterns: Dict[str, Callable[[], str]] = {}
        self.languages: List[str] = []
        self.values: Dict[str, float] = {}
        self.taboos: List[str] = []
        self.rituals: List[str] = []
        self.cultural_motifs: List[str] = []
        self.cultural_age: float = 0.0
        self.adaptations: List[str] = []
        self.cultural_coherence: float = 0.8
        self.divergent_subcultures: List[Dict] = []
        self.name_components: Dict[str, List[str]] = {'prefixes': [], 'roots': [], 'suffixes': []}
        self.cultural_shift_momentum: Dict[str, Any] = {}
        self.external_influences: List[Any] = []
        self._initialize_culture()

    def _initialize_culture(self): # Simplified, assumes DRM is accessible
        planet = DRM.get_entity(self.civilization.planet_id) if self.civilization.planet_id else None
        if planet and hasattr(planet, 'temperature') and hasattr(planet, 'motifs') and hasattr(planet, 'surface'):
            if planet.temperature < -30: self.belief_systems[BeliefType.COSMOLOGY] = max(0.7, self.belief_systems[BeliefType.COSMOLOGY])
            # ... other environmental influences
        self._select_social_structure()
        self._initialize_archetypes()
        self._generate_naming_conventions() # This method needs to be defined within CultureEngine
        self._initialize_cultural_motifs()

    def _generate_naming_conventions(self): # Moved into class
        # (Logic from the original _generate_naming_conventions)
        consonants = 'bcdfghjklmnpqrstvwxyz'
        vowels = 'aeiou'
        # ... (rest of the generation logic)
        self.name_components['prefixes'] = [random.choice(consonants)+random.choice(vowels) for _ in range(10)]
        self.name_components['roots'] = [random.choice(vowels)+random.choice(consonants)+random.choice(vowels) for _ in range(20)]
        self.name_components['suffixes'] = [random.choice(consonants)+random.choice(vowels) for _ in range(10)]

        self.naming_patterns = {
            'person': lambda: self._generate_name('person'),
            'place': lambda: self._generate_name('place'),
            'concept': lambda: self._generate_name('concept'),
            'deity': lambda: self._generate_name('deity')
        }


    def _generate_name(self, entity_type: str) -> str: # Moved into class
        # (Logic from the original _generate_name)
        if not self.name_components['roots']: return "DefaultName" # Ensure components exist
        result = random.choice(self.name_components['roots'])
        if random.random() < 0.5 and self.name_components['prefixes']:
            result = random.choice(self.name_components['prefixes']) + result
        if random.random() < 0.5 and self.name_components['suffixes']:
            result += random.choice(self.name_components['suffixes'])
        return result.capitalize()


    def _select_social_structure(self): pass # Placeholder
    def _initialize_archetypes(self): pass # Placeholder
    def _initialize_cultural_motifs(self): pass # Placeholder
    def evolve_culture(self, time_delta: float): pass # Placeholder
    def _evolve_beliefs(self, time_delta: float): pass # Placeholder
    def _calculate_external_pressure(self) -> Dict: return {} # Placeholder
    def _check_social_structure_transition(self, time_delta: float): pass # Placeholder
    def _record_social_transition(self, old_structure, new_structure): pass # Placeholder
    def _evolve_cultural_motifs(self, time_delta: float): pass # Placeholder
    def _handle_cultural_divergence(self, time_delta: float): pass # Placeholder
    def _form_new_subculture(self): pass # Placeholder
    def _evolve_naming_conventions(self): pass # Placeholder
    def _generate_cultural_events(self, time_delta: float): pass # Placeholder


# Bind CultureEngine to Civilization (Done after Civilization is fully defined)
Civilization.culture_engine = property(lambda self: self._initialize_culture_engine())


# ===== CivilizationInteraction System =====
class CivilizationInteraction:
    def __init__(self, civ1_id: str, civ2_id: str): # Depends on Civilization, DevelopmentArea, DRM, MotifSeeder
        self.civ1_id = civ1_id
        self.civ2_id = civ2_id
        # ... (attributes from original)
        self.interaction_type = InteractionType.OBSERVATION
        self.motif_resonance = 0.0
        self.technological_parity = 0.0
        self.cultural_compatibility = 0.0
        self.tension = 0.0
        self.shared_history: List[Dict] = []
        self.diplomatic_status = "neutral"
        self.treaties: List[Tuple[str, float]] = [] # (Treaty Name, Timestamp)
        self.war_status = False
        self.trade_volume = 0.0
        self.last_update_time = 0.0 # Assuming float timestamp
        self._initialize_relationship()

    def _initialize_relationship(self): # Simplified, assumes DRM is accessible
        civ1 = DRM.get_entity(self.civ1_id)
        civ2 = DRM.get_entity(self.civ2_id)
        if not civ1 or not civ2 or not hasattr(civ1, 'tech_levels') or not hasattr(civ2, 'tech_levels'): return
        # ... (rest of init logic)

    def _update_interaction_type(self): pass # Placeholder
    def update_relationship(self, time_delta: float): pass # Placeholder
    def _generate_interaction_event(self): pass # Placeholder
    def _apply_interaction_effects(self, time_delta: float): pass # Placeholder
    def _resolve_conflict(self): pass # Placeholder


class DiplomaticRegistry: # Depends on CivilizationInteraction
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DiplomaticRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _initialize(self):
        if self._initialized: return
        self.relationships: Dict[Tuple[str, str], CivilizationInteraction] = {}
        # ... (other attributes)
        self.alliances = {}
        self.wars = {}
        self.trade_networks = {}
        self._initialized = True

    def get_relationship(self, civ1_id: str, civ2_id: str) -> CivilizationInteraction:
        key = tuple(sorted((civ1_id, civ2_id)))
        if key not in self.relationships:
            self.relationships[key] = CivilizationInteraction(civ1_id, civ2_id)
        return self.relationships[key]
    # ... (other DiplomaticRegistry methods)

DIPLOMATIC_REGISTRY = DiplomaticRegistry()


# ===== Life & Biology Systems (Metabolism, Flora, Emotional Resonance) =====
# Using the more detailed/later definitions for these.

process_motifs = { # Defined after MetabolicProcess enum
    MetabolicProcess.PHOTOSYNTHESIS: ["light_harvesting", "solar_alchemy", "growth_cycle"],
    MetabolicProcess.RESPIRATION: ["oxidation_rhythm", "energy_extraction", "cellular_breath"],
    MetabolicProcess.CHEMOSYNTHESIS: ["mineral_transmutation", "chemical_cascade", "elemental_binding"],
    MetabolicProcess.RADIOSYNTHESIS: ["radiation_harvest", "particle_weaving", "decay_reversal"],
    MetabolicProcess.QUANTUM_ENTANGLEMENT: ["probability_harvest", "quantum_threading", "uncertainty_mapping"],
    MetabolicProcess.SYMBOLIC_ABSORPTION: ["meaning_derivation", "semantic_integration", "symbolic_conversion"],
    MetabolicProcess.MOTIF_CYCLING: ["pattern_recognition", "motif_amplification", "thematic_resonance"],
    MetabolicProcess.HARMONIC_RESONANCE: ["harmonic_alignment", "frequency_attunement", "wave_synchronization"]
}


class RecursiveMetabolism: # Using the first, more detailed definition
    def __init__(self, owner_entity: Any, complexity: float = 0.5,
                 primary_pathways: Optional[List[MetabolicPathway]] = None,
                 preferred_resources: Optional[List[MetabolicResource]] = None,
                 max_recursion_depth: int = 3):
        self.owner = owner_entity
        self.complexity = complexity
        # ... (rest of attributes from the first definition)
        self.primary_pathways = primary_pathways or self._initialize_primary_pathways()
        self.preferred_resources = preferred_resources or self._initialize_preferred_resources()
        self.max_recursion_depth = max_recursion_depth
        self.pathways = self._initialize_pathways()
        self.catalysts: Dict[str, Any] = {}
        self.inhibitors: Dict[str, Any] = {}
        self.storage: Dict[MetabolicResource, float] = {}
        self.current_processes: List[Any] = [] # Store active process instances or IDs
        self.efficiency: Dict[MetabolicResource, float] = {}
        self.processing_rate: Dict[MetabolicPathway, float] = {}
        self.waste_production: Dict[MetabolicResource, float] = {}
        self.energy_output: float = 0.0
        self.adaptation_history: List[Any] = []
        self.resource_affinities: Dict[MetabolicResource, float] = {}
        self.pathway_strengths: Dict[MetabolicPathway, float] = {p: 0.5 for p in self.primary_pathways}
        self.process_cycles: int = 0
        self.last_update_time: float = 0.0
        self.current_recursion_depth: int = 0
        self.motif_resonance: Dict[str, float] = {} # motif_id -> resonance_value
        self.symbolic_byproducts: List[Any] = []
        self.anomaly_threshold: float = 0.8
        self.integration_with_environment: float = 0.5
        self._initialize_efficiency()
        self._initialize_resource_storage()
        self._initialize_motif_resonance()

    def _initialize_primary_pathways(self) -> List[MetabolicPathway]: return [random.choice(list(MetabolicPathway))] # Placeholder
    def _initialize_preferred_resources(self) -> List[MetabolicResource]: return [random.choice(list(MetabolicResource))] # Placeholder
    def _initialize_pathways(self) -> Dict[MetabolicPathway, Any]: return {p: {} for p in self.primary_pathways} # Placeholder
    def _initialize_efficiency(self): pass # Placeholder
    def _initialize_resource_storage(self): pass # Placeholder
    def _initialize_motif_resonance(self): pass # Placeholder
    # ... (other methods from the first RecursiveMetabolism definition)


class MotifFloraSystem: # Depends on FloralGrowthPattern, FloraEvolutionStage, NutrientType
    def __init__(self, owner_entity: Any, base_pattern: Optional[str] = None,
                 growth_style: Optional[FloralGrowthPattern] = None,
                 evolution_stage: FloraEvolutionStage = FloraEvolutionStage.SEED,
                 maturation_rate: float = 0.05,
                 symbolic_metabolism: Optional[Dict[NutrientType, float]] = None,
                 primary_motifs: Optional[List[str]] = None):
        self.owner = owner_entity
        self.base_pattern = base_pattern or self._generate_base_pattern()
        # ... (rest of attributes and methods)
        self.growth_style = growth_style or random.choice(list(FloralGrowthPattern))
        self.evolution_stage = evolution_stage
        self.maturation_rate = maturation_rate
        self.symbolic_metabolism = symbolic_metabolism or self._initialize_metabolism()
        self.primary_motifs = primary_motifs or self._initialize_motifs()
        self.growth_factor = 0.0
        self.health = 1.0
        self.pattern_density = 0.1
        self.root_depth = 0.0
        self.canopy_spread = 0.0
        self.seasonal_state = {}
        self.adaptation_history = []
        self.environmental_responses = self._initialize_environmental_responses()
        self.seed_bank = []
        self.pollination_vectors = set()
        self.cross_pollination_record = {}
        self.symbiotic_relationships = {}
        self.competitive_relationships = {}
        self.community_role = {}
        self.age = 0.0
        self.growth_cycles_completed = 0
        self.last_update_time = 0.0
        self.seasonal_cycle_position = 0.0
        self.sensory_properties = self._initialize_sensory_properties()
        self.symbolic_effects = self._initialize_symbolic_effects()
        self.produced_resources = {}
        self.mutation_potential = 0.2
        self.adaptation_pressure = 0.0
        self.evolutionary_direction = {}

    def _generate_base_pattern(self) -> str: return "default_flora_pattern" # Placeholder
    def _initialize_metabolism(self) -> Dict[NutrientType, float]: return {nt: random.random() for nt in NutrientType} # Placeholder
    def _initialize_motifs(self) -> List[str]: return ["growth", "connection"] # Placeholder
    def _initialize_environmental_responses(self): return {} # Placeholder
    def _initialize_sensory_properties(self): return {} # Placeholder
    def _initialize_symbolic_effects(self): return {} # Placeholder


class EmotionalResonanceBody: # Depends on EmotionalState, EmotionType
    def __init__(self, owner_entity: Any,
                 base_resonance: Optional[Dict[EmotionalState, float]] = None, # Changed from base_sensitivity
                 projection_radius: float = 10.0, # Added default
                 receptivity: float = 0.5, # Added default
                 # dominant_emotions: Optional[List[EmotionType]] = None, # Not used in chosen __init__
                 # emotional_capacity: float = 1.0, # Not used
                 # memory_persistence: float = 0.7 # Not used
                 ):
        self.owner = owner_entity
        # Corrected: ensure self.base_resonance is initialized before _calculate_current_state
        self._initialized_base_resonance_val = base_resonance if base_resonance is not None else self._initialize_base_resonance()
        self.base_resonance = self._initialized_base_resonance_val # Store the actual dict
        self.current_state: EmotionalState = self._calculate_current_state()
        self.projection_radius = projection_radius
        self.receptivity = receptivity
        self.resonance_signature: Dict[Any, Any] = {}
        self.emotional_memory: List[Any] = []
        self.active_harmonics: List[Any] = []
        self.resonance_connections: Dict[str, float] = {}
        self.emotional_weather: Dict[Any, Any] = {}
        self.state_history: Deque[Tuple[EmotionalState, float]] = deque(maxlen=100)
        self.state_transitions: Dict[Tuple[EmotionalState, EmotionalState], int] = defaultdict(int)
        self.harmonic_nodes: List[Dict] = self._initialize_harmonic_nodes()
        self.dissonance_threshold: float = 0.7
        self.resonance_evolution_rate: float = 0.05
        self.last_update_time = getattr(self.owner, 'last_update_time', time.time()) # Use time.time() as fallback
        self.state_history.append((self.current_state, self.last_update_time))

    def _initialize_base_resonance(self) -> Dict[EmotionalState, float]:
        base = {state: 0.1 + 0.1 * random.random() for state in EmotionalState}
        dominant_emotions = random.sample(list(EmotionalState), random.randint(1,3))
        for emotion in dominant_emotions: base[emotion] = 0.4 + 0.4 * random.random()
        # ... (motif influence logic from original)
        total = sum(base.values())
        return {k: v / total for k, v in base.items()} if total > 0 else base

    def _calculate_current_state(self) -> EmotionalState:
        if not self.base_resonance: return EmotionalState.SERENITY # Default
        return max(self.base_resonance.items(), key=lambda item: item[1])[0]

    def _initialize_harmonic_nodes(self) -> List[Dict]: return [] # Placeholder
    def _apply_motif_influence(self, resonance: Dict[EmotionalState, float], motif: str): pass # Placeholder


# ===== ENVIRONMENTAL SYSTEMS (Storm, SeasonalCycle, etc.) =====

class Storm: # Depends on WORLD_SIZE
    def __init__(self, center: Tuple[float, float], storm_type: str, radius: float, intensity: float, symbolic_content: Dict[str, float]):
        self.center = center
        self.storm_type = storm_type
        # ... (attributes and methods from original)
        self.radius = radius
        self.intensity = intensity
        self.symbolic_content = symbolic_content
        self.age = 0
        self.trajectory: List[Tuple[float,float]] = []
        self.affected_regions: Set[Tuple[int,int]] = set()
        self.emotional_signature = self._derive_emotional_signature()
        self.dissolution_rate = 0.05

    def update(self, world_state_obj: 'WorldState'): # Takes WorldState object
        self.age += 1
        self.intensity *= (1 - self.dissolution_rate)
        self._move(world_state_obj.wind_patterns) # Assuming wind_patterns is attribute of WorldState
        self.trajectory.append(self.center)
        self._apply_effects(world_state_obj)
        return self.intensity > 0.1

    def _move(self, wind_patterns_obj: 'WindPattern'): # Takes WindPattern object
        x, y = self.center
        # Assuming get_vector returns (dx, dy) tuple or similar
        wind_dx, wind_dy = wind_patterns_obj.get_vector(x,y)[:2] # Take first 2 for 2D movement
        # ... (rest of move logic)
        new_x = x + (wind_dx * (0.5 + self.intensity * 0.1))
        new_y = y + (wind_dy * (0.5 + self.intensity * 0.1))
        self.center = (max(0, min(new_x, WORLD_SIZE -1)), max(0, min(new_y, WORLD_SIZE -1)))


    def _apply_effects(self, world_state_obj: 'WorldState'): # Takes WorldState object
        # ... (logic to apply effects to world_state_obj)
        pass
    def _derive_emotional_signature(self) -> Dict[str, float]: return {} # Placeholder


class SeasonalCycle:
    def __init__(self, starting_season: str = "spring", cycle_length: int = 120):
        self.seasons = ["winter", "spring", "summer", "autumn"]
        # ... (attributes and methods from original)
        self.season_durations = {s: cycle_length // 4 for s in self.seasons}
        self.current_season = starting_season
        self.time_in_season = 0
        self.cycle_length = cycle_length
        self.year_count = 0
        self.seasonal_effects = self._initialize_seasonal_effects()
        self.transition_thresholds = self._calculate_transition_thresholds()
        self.seasonal_events: List[Dict] = []


    def advance(self, accelerate: bool = False) -> bool: # Returns bool
        # ... (logic from original)
        return False # Placeholder
    def get_seasonal_modifier(self, aspect: str) -> float: return 1.0 # Placeholder
    def get_diffusion_modifier(self) -> float: return self.get_seasonal_modifier("diffusion_rate") # Placeholder
    def get_current_season_info(self) -> Dict: return {} # Placeholder
    def _initialize_seasonal_effects(self) -> Dict : return {} # Placeholder
    def _calculate_transition_thresholds(self) -> Dict : return {} # Placeholder
    def _generate_seasonal_event(self): pass # Placeholder
    def _generate_event_effects(self, event_name: str, intensity: float) -> Dict : return {} # Placeholder


class CurrentLayer: # Depends on uuid
    def __init__(self, width: int, height: int, depth: float, velocity_factor: float = 1.0):
        self.width = width
        self.height = height
        # ... (attributes and methods from original)
        self.depth = depth
        self.velocity_factor = velocity_factor
        self.current_vectors = np.zeros((width, height, 3))
        self.memory_concentration = np.zeros((width, height))
        self.symbolic_content: Dict[str, Dict] = {} # memory_id -> memory_data
        self._initialize_current_patterns()

    def get_current_vector(self, x: float, y: float) -> List[float]: # x,y are floats
         if 0 <= x < self.width and 0 <= y < self.height:
            return self.current_vectors[int(x), int(y)].tolist() # Return list
         return [0.0, 0.0, 0.0]

    def update_currents(self, temperature_gradients: np.ndarray, salinity_map: np.ndarray) -> List[Tuple[int,int,float]]:
        # ... (logic from original)
        return [] # Placeholder
    def deposit_memory(self, x: int, y: int, memory_content: Dict, intensity: float): pass # Placeholder
    def retrieve_memories(self, x: int, y: int, radius: int = 3) -> List[Dict]: return [] # Placeholder
    def update_memories(self) -> List[Dict]: return [] # Placeholder
    def _initialize_current_patterns(self): pass # Placeholder
    def _create_circular_current(self, center_x: int, center_y: int, radius: int, clockwise: bool): pass # Placeholder
    def _create_linear_current(self, start_y: int, width: int, direction: int): pass # Placeholder


class ThermohalineCirculation: # Depends on math, uuid, CurrentLayer (WorldGeometry is simple)
    def __init__(self, world_geometry_obj: Any): # Takes any object with width/height
        self.world_geometry = world_geometry_obj
        self.conveyor_belt: List[Tuple[float,float]] = self._initialize_conveyor_belt()
        # ... (attributes and methods)
        self.upwelling_zones = self._initialize_upwelling_zones()
        self.downwelling_zones = self._initialize_downwelling_zones()
        self.circulation_strength = 1.0
        self.memory_transport_rate = 0.3
        self.symbolic_payload: Dict[str, Dict] = {} # memory_id -> memory_data

    def _initialize_conveyor_belt(self) -> List[Tuple[float,float]]: # Moved into class
        # (Logic from original _initialize_conveyor_belt)
        # Simplified example:
        belt = []
        for x_coord in range(0, self.world_geometry.width, 20): belt.append((float(x_coord), 10.0))
        for y_coord in range(0, self.world_geometry.height, 20): belt.append((float(self.world_geometry.width - 10), float(y_coord)))
        # ... (complete the loop)
        return belt

    def _find_nearest_conveyor_point(self, location: Tuple[float,float]) -> Tuple[Optional[Tuple[float,float]], Optional[int]]: # Moved into class
        min_dist = float('inf')
        nearest_pt = None
        nearest_idx_val = None
        for i, pt in enumerate(self.conveyor_belt):
            dist = math.sqrt((location[0] - pt[0])**2 + (location[1] - pt[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_pt = pt
                nearest_idx_val = i
        return nearest_pt, nearest_idx_val


    def update(self, temperature_gradients: np.ndarray, salinity_map: np.ndarray) -> List[Dict]: return [] # Placeholder
    def add_memory_payload(self, location: Tuple[float,float], memory_content: Dict, intensity: float) -> bool: return False # Placeholder
    def get_surface_currents(self) -> np.ndarray : return np.zeros((self.world_geometry.width, self.world_geometry.height, 2)) # Placeholder
    def _calculate_driving_force(self, temperature_gradients: np.ndarray, salinity_map: np.ndarray) -> float : return 1.0 # Placeholder

    def _process_memory_transport(self) -> List[Dict]: return [] # Placeholder
    def _update_conveyor_belt(self): pass # Placeholder
    def _initialize_upwelling_zones(self) -> List[Tuple[float,float]] : return [] # Placeholder
    def _initialize_downwelling_zones(self) -> List[Tuple[float,float]] : return [] # Placeholder
    def _update_upwelling_zones(self) -> List[Dict] : return [] # Placeholder
    def _update_downwelling_zones(self) -> List[Dict] : return [] # Placeholder
    def _point_distance(self, p1: Tuple[float,float], p2: Tuple[float,float]) -> float : return 0.0 # Placeholder


class WindPattern: # Depends on math, random, numpy
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # ... (attributes and methods)
        self.vector_field = np.zeros((width, height, 2))
        self.pressure_systems: List[Dict] = []
        self.prevailing_direction = (1.0,0.0)
        self.turbulence = 0.3
        self.seasonal_modifier = 1.0
        self._initialize_basic_patterns()

    def update(self, temperature_map: np.ndarray, terrain_height_map: np.ndarray, seasonal_modifier_val: float) -> List[Dict]: # Renamed seasonal_modifier
        # ... (logic from original)
        return [] # Placeholder
    def get_vector(self, x: float, y: float) -> np.ndarray : # x,y are float
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.vector_field[int(x), int(y)]
        return np.array([0.0,0.0])
    def add_pressure_system(self, center: Tuple[float,float], radius: float, is_high_pressure: bool, intensity: float): pass # Placeholder
    def _initialize_basic_patterns(self): pass # Placeholder
    def _add_prevailing_wind(self, vector_field: np.ndarray): pass # Placeholder
    def _add_pressure_system_influence(self, vector_field: np.ndarray): pass # Placeholder
    def _add_terrain_effects(self, vector_field: np.ndarray, terrain_height_map: np.ndarray): pass # Placeholder
    def _add_turbulence(self, vector_field: np.ndarray): pass # Placeholder
    def _smooth_transition(self, new_vector_field: np.ndarray): pass # Placeholder
    def _update_pressure_systems(self, temperature_map: np.ndarray): pass # Placeholder
    def _detect_wind_events(self) -> List[Dict]: return [] # Placeholder


class ClimateZone: # Depends on math
    def __init__(self, center: Tuple[float,float], radius: float, climate_type: str, intensity: float = 1.0):
        self.center = center
        self.radius = radius
        # ... (attributes and methods)
        self.climate_type = climate_type
        self.intensity = intensity
        self.seasonal_variations = self._initialize_seasonal_variations()
        self.symbolic_associations = self._initialize_symbolic_associations()
        self.weather_tendencies = self._initialize_weather_tendencies()
        self.boundary_blending = 20


    def get_influence(self, position: Tuple[float,float]) -> Tuple[float, Dict]: return (0.0, {}) # Placeholder
    def update_seasonal_state(self, current_season: str, progress: float): pass # Placeholder
    def _get_temperature_modifier(self) -> float : return 0.0 # Placeholder
    def _get_precipitation_modifier(self) -> float : return 0.0 # Placeholder
    def _get_scaled_symbolic_elements(self, influence: float) -> Dict : return {} # Placeholder
    def _get_scaled_weather_bias(self, influence: float) -> Dict : return {} # Placeholder
    def _initialize_seasonal_variations(self) -> Dict : return {} # Placeholder
    def _initialize_symbolic_associations(self) -> Dict : return {} # Placeholder
    def _initialize_weather_tendencies(self) -> Dict : return {} # Placeholder


class WorldState: # Depends on SeasonalCycle, WindPattern, ClimateZone, Storm, CurrentLayer, ThermohalineCirculation, numpy
    def __init__(self, width: int = WORLD_SIZE, height: int = WORLD_SIZE):
        self.width = width
        self.height = height
        # ... (attributes and methods)
        self.time = 0.0
        self.seasonal_cycle = SeasonalCycle()
        self.wind_patterns = WindPattern(width, height)
        self.climate_zones: List[ClimateZone] = []
        self.active_storms: List[Storm] = []
        self.temperature_map = np.zeros((width,height))
        self.precipitation_map = np.zeros((width,height))
        self.humidity_map = np.zeros((width,height))
        self.terrain_height_map = np.zeros((width,height))
        self.symbolic_influence_map: Dict[str, np.ndarray] = {}
        self.event_history: List[Dict] = []
        self.ocean_currents: Optional[ThermohalineCirculation] = None # Use Thermohaline for this
        self._initialize_terrain()
        self._initialize_climate_zones()
        if self.terrain_height_map.mean() < 0.3: # Heuristic for having oceans
            class WorldGeom: # Simple geometry for Thermohaline
                def __init__(self, w,h): self.width=w; self.height=h
            self.ocean_currents = ThermohalineCirculation(WorldGeom(width,height))


    def update(self, time_delta: float) -> List[Dict]: return [] # Placeholder
    def apply_symbolic_effect(self, position: Tuple[int,int], symbol: str, strength: float): pass # Placeholder
    def get_local_climate(self, position: Tuple[int,int]) -> Dict: return {} # Placeholder
    def get_dominant_symbols(self, position: Tuple[int,int], count: int =3) -> List[Tuple[str,float]]: return [] # Placeholder
    def deposit_memory(self, position: Tuple[int,int], memory_content: Dict, intensity: float=1.0) -> bool: return False # Placeholder
    def _initialize_terrain(self): pass # Placeholder
    def _initialize_climate_zones(self): pass # Placeholder
    def _update_climate_maps(self, seasonal_info: Dict): pass # Placeholder
    def _smooth_map(self, map_data: np.ndarray, kernel_size: int =3) -> np.ndarray: # Placeholder
        from scipy.ndimage import uniform_filter # Import locally if heavy
        return uniform_filter(map_data, size=kernel_size, mode='reflect')
    def _update_storms(self, time_delta: float): pass # Placeholder
    def _generate_storm(self, seasonal_info: Dict) -> Optional[Storm]: return None # Placeholder
    def _generate_storm_symbolism(self, storm_type: str, seasonal_info: Dict) -> Dict[str, float]: return {} # Placeholder


# ===== INTEGRATION FUNCTIONS (Example stubs) =====

def integrate_environment_with_planet(planet: Planet, world_state: Optional[WorldState] = None) -> WorldState:
    # This function would modify planet and world_state based on each other
    # For now, just ensures planet has an env_state
    if not world_state: world_state = WorldState()
    if not hasattr(planet, 'env_state'): planet.env_state = world_state # Simplified link
    return world_state

def apply_environmental_effects_to_civilization(civilization: Civilization, world_state: WorldState):
    # This function would apply effects from world_state to civilization
    pass

# Final check for singleton instance, ensure it's at the end after all class defs if it depends on them
# cosmic_scroll_manager = CosmicScrollManager() # Already defined earlier, this would be a redefinition.
# Ensure it's instantiated after all its dependent classes are defined if there are such strict dependencies.
# Given the current structure, its placement after the core data classes like Entity, Motif, Event is fine.
# The `cosmic_scroll` attribute of `CosmicScrollManager` instantiates `CosmicScroll`.

# Ensure all methods intended to be part of classes are correctly indented.
# The original file had several floating method definitions.
# Example: _generate_motif_name was moved into CosmicScroll.
# _find_nearest_conveyor_point was moved into ThermohalineCirculation.

# Clean up redundant logger configurations if any (kept one main config at top).
# Removed redundant imports of standard libraries like random, math, etc., throughout the file.
# Corrected `AuthorNo` to `Author` in comments.

logger.info("cosmic_scroll.py refactored and loaded.")
