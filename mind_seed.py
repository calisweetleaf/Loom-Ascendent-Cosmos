# === COSMIC SCROLL MODULE ===
# This module defines core data structures, enums, and systems for the Cosmic Scroll simulation.

import random
import logging
import math
import uuid
import time
from enum import Enum, auto
from collections import defaultdict, deque
from datetime import datetime, timedelta 
from typing import List, Dict, Any, Union, Optional, Set, Tuple, Callable 
from dataclasses import dataclass, field, asdict, fields
import numpy as np
# from scipy.ndimage import uniform_filter # For WorldState map smoothing, if WorldState is moved here

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                    filename='cosmic_scroll_simulation.log',
                    filemode='w') 
logger = logging.getLogger(__name__)

# ===== Constants & Global Configuration =====
# WORLD_SIZE = 100 # Kept in cosmic_scroll.py for now
G_CONST = 6.67430e-11
C_LIGHT = 299792458   
H_PLANCK = 6.62607015e-34 
SIGMA_SB = 5.670374419e-8

# ===== Forward Declarations (useful if classes reference each other within this file) =====
class CosmicScrollManager: pass
class CosmicEntity: pass 
class Civilization: pass 
class Motif: pass
class Event: pass
class MetabolicProcess: pass # The class
class RecursiveMetabolism: pass
class ScrollMemory: pass
class CultureEngine: pass
class DiplomaticRelationData: pass # Renamed from DiplomaticRelation to avoid conflict if a class is named that
class DiplomaticRegistry: pass
class Storm: pass
class SeasonalCycle: pass
class CurrentLayer: pass
class ThermohalineCirculation: pass
class WindPattern: pass
class ClimateZone: pass
class WorldState: pass
class Planet: pass 
class Star: pass
class Galaxy: pass
class Universe: pass

# ===== Core Enums (Consolidated and Corrected) =====

class MotifCategoryEnum(Enum): # Renamed from MotifCategory
    PRIMORDIAL = "primordial"; ELEMENTAL = "elemental"; STRUCTURAL = "structural"
    NARRATIVE = "narrative"; ARCHETYPAL = "archetypal"; HARMONIC = "harmonic"
    CHAOTIC = "chaotic"; LUMINOUS = "luminous"; SHADOW = "shadow"
    RECURSIVE = "recursive"; ASCENDANT = "ascendant"; DIMENSIONAL = "dimensional"
    TEMPORAL = "temporal"; VITAL = "vital"; ENTROPIC = "entropic"
    CRYSTALLINE = "crystalline"; ABYSSAL = "abyssal"; CONNECTIVE = "connective"
    MUTATIVE = "mutative"; TECHNOLOGICAL = "technological"; PHILOSOPHICAL = "philosophical"
    EMOTIONAL_ARCHETYPE = "emotional_archetype" 

class EntityTypeEnum(Enum): # Renamed from EntityType
    COSMIC_STRUCTURE = "cosmic_structure"; BIOLOGICAL = "biological"; SYMBOLIC = "symbolic"
    ENERGETIC = "energetic"; ANOMALY = "anomaly"; CONSTRUCT = "construct" 
    NARRATIVE_AGENT = "narrative_agent"; PLANETARY_SYSTEM = "planetary_system"
    CIVILIZATION_UNIT = "civilization_unit"; SCROLL_ARTIFACT = "scroll_artifact"
    DIMENSIONAL_BEING = "dimensional_being"; ENVIRONMENTAL_PHENOMENON = "environmental_phenomenon"
    UNIVERSE = "universe"; GALAXY_CLUSTER = "galaxy_cluster"; GALAXY = "galaxy" 
    STAR = "star"; PLANET = "planet"; MOON = "moon"; ASTEROID = "asteroid"


class EventTypeEnum(Enum): # Renamed from EventType
    CREATION = "creation"; DESTRUCTION = "destruction"; TRANSFORMATION = "transformation"
    INTERACTION = "interaction"; DISCOVERY = "discovery"; CONVERGENCE = "convergence"
    DIVERGENCE = "divergence"; AWAKENING = "awakening"; DORMANCY = "dormancy"
    EMERGENCE = "emergence"; MANIFESTATION = "manifestation"; RESONANCE = "resonance"
    DISSONANCE = "dissonance"; MUTATION_EVENT = "mutation_event"; PROPAGATION = "propagation"
    STABILIZATION = "stabilization"; DESTABILIZATION = "destabilization"
    CULTURAL_SHIFT = "cultural_shift"; DIPLOMATIC_EVENT = "diplomatic_event"
    METABOLIC_EVENT = "metabolic_event"; ENVIRONMENTAL_EVENT = "environmental_event"
    ANOMALY_FORMATION = "anomaly_formation"; ANOMALY_DECAY = "anomaly_decay"
    SCROLL_MEMORY_EVENT = "scroll_memory_event" 

class MetabolicProcessTypeEnum(Enum): # Renamed from MetabolicProcessType (which was MetabolicProcess in cosmic_scroll)
    PHOTOSYNTHESIS = "photosynthesis"; CHEMOSYNTHESIS = "chemosynthesis"; RADIOSYNTHESIS = "radiosynthesis"
    RESPIRATION = "respiration"; SYMBOLIC_ABSORPTION = "symbolic_absorption"
    NARRATIVE_CONSUMPTION = "narrative_consumption"; MOTIF_CYCLING = "motif_cycling"
    ENTROPIC_HARVESTING = "entropic_harvesting"; HARMONIC_CONVERSION = "harmonic_conversion"
    QUANTUM_METABOLISM = "quantum_metabolism"; VOID_ASSIMILATION = "void_assimilation"
    TEMPORAL_FEEDING = "temporal_feeding"

class BreathPhaseEnum(Enum): # Renamed from BreathPhase
    INHALE = "inhale"; HOLD_IN = "hold_in"; EXHALE = "exhale"; HOLD_OUT = "hold_out"

class MutationTypeEnum(Enum): # Renamed from MutationType
    POINT = "point"; DUPLICATION = "duplication"; DELETION = "deletion"; INVERSION = "inversion"
    INSERTION = "insertion"; SYMBOLIC_MUTATION = "symbolic_mutation"; RECURSIVE_MUTATION = "recursive_mutation"
    MOTIF_EXPRESSION = "motif_expression"; NARRATIVE_SHIFT = "narrative_shift"
    QUANTUM_TUNNEL = "quantum_tunnel"; EPIGENETIC = "epigenetic"; ARCHETYPAL_MERGE = "archetypal_merge"

class MetabolicResourceEnum(Enum): # Renamed from MetabolicResource
    RAW_MATTER = "raw_matter"; ENERGY_PHOTONIC = "energy_photonic"; ENERGY_CHEMICAL = "energy_chemical"
    ENERGY_THERMAL = "energy_thermal"; ENERGY_KINETIC = "energy_kinetic"; ENERGY_QUANTUM = "energy_quantum"
    ENERGY_SYMBOLIC = "energy_symbolic"; NUTRIENT_ORGANIC = "nutrient_organic"
    NUTRIENT_INORGANIC = "nutrient_inorganic"; INFORMATION_PATTERN = "information_pattern"
    COMPLEXITY = "complexity"; ENTROPY = "entropy"; MOTIF_ESSENCE = "motif_essence"
    NARRATIVE_FUEL = "narrative_fuel"; VOID_SUBSTANCE = "void_substance"; EMOTIONAL_ENERGY = "emotional_energy"

class GalaxyTypeEnum(Enum): 
    SPIRAL = "spiral"; ELLIPTICAL = "elliptical"; LENTICULAR = "lenticular"; IRREGULAR = "irregular"; PECULIAR = "peculiar"; DWARF_SPIRAL = "dwarf_spiral"; DWARF_ELLIPTICAL = "dwarf_elliptical"; DWARF_IRREGULAR = "dwarf_irregular"; ACTIVE_NUCLEUS = "active_nucleus"

class StarTypeEnum(Enum): 
    O = "O"; B = "B"; A = "A"; F = "F"; G = "G"; K = "K"; M = "M"; L = "L"; T = "T"; Y = "Y"; WHITE_DWARF = "white_dwarf"; NEUTRON_STAR = "neutron_star"; BLACK_HOLE_STELLAR = "black_hole_stellar"; PROTOSTAR = "protostar"; RED_GIANT = "red_giant"; RED_SUPERGIANT = "red_supergiant"; BLUE_GIANT = "blue_giant"; WOLF_RAYET = "wolf_rayet"; CARBON_STAR = "carbon_star"; VARIABLE_STAR = "variable_star"

class PlanetTypeEnum(Enum): 
    TERRESTRIAL = "terrestrial"; GAS_GIANT = "gas_giant"; ICE_GIANT = "ice_giant"; DWARF_PLANET = "dwarf_planet"; SUPER_EARTH = "super_earth"; MINI_NEPTUNE = "mini_neptune"; HOT_JUPITER = "hot_jupiter"; OCEAN_WORLD = "ocean_world"; DESERT_WORLD = "desert_world"; ICE_WORLD = "ice_world"; LAVA_WORLD = "lava_world"; CARBON_PLANET = "carbon_planet"; IRON_PLANET = "iron_planet"; CORELESS_PLANET = "coreless_planet"; PUFFY_PLANET = "puffy_planet"; CHTHONIAN_PLANET = "chthonian_planet"

class CivilizationTypeEnum(Enum): 
    TYPE_0 = "pre_planetary"; TYPE_I = "planetary"; TYPE_II = "stellar"; TYPE_III = "galactic"; TYPE_IV = "universal"; TYPE_V = "multiversal"; SYMBOLIC_TRANSCENDENT = "symbolic_transcendent"; VOID_DWELLING = "void_dwelling"

class DevelopmentAreaEnum(Enum): # Renamed from DevelopmentArea
    ENERGY_SYSTEMS = "energy_systems"; COMPUTATIONAL_SCIENCE = "computational_science"; MATERIALS_SCIENCE = "materials_science"; BIOTECHNOLOGY_GENETICS = "biotechnology_genetics"; PROPULSION_TRANSPORT = "propulsion_transport"; WEAPONRY_DEFENSE = "weaponry_defense"; COMMUNICATION_NETWORKS = "communication_networks"; SOCIAL_ORGANIZATION_GOVERNANCE = "social_organization_governance"; ENVIRONMENTAL_ENGINEERING = "environmental_engineering"; SYMBOLIC_ENGINEERING = "symbolic_engineering"; QUANTUM_ENGINEERING = "quantum_engineering"; COSMIC_AWARENESS = "cosmic_awareness"

class AnomalyTypeEnum(Enum): 
    WORMHOLE = "wormhole"; SPATIAL_RIFT = "spatial_rift"; TEMPORAL_DISTORTION = "temporal_distortion"; EXOTIC_MATTER_REGION = "exotic_matter_region"; QUANTUM_ENTANGLEMENT_FIELD = "quantum_entanglement_field"; NARRATIVE_VORTEX = "narrative_vortex"; DIMENSIONAL_BLEED = "dimensional_bleed"; COSMIC_STRING_FRAGMENT = "cosmic_string_fragment"; REALITY_BUBBLE_UNSTABLE = "reality_bubble_unstable"; VOID_NULL_REGION = "void_null_region"; HARMONIC_CASCADE_EVENT = "harmonic_cascade_event"; MEMORY_ECHO_CLUSTER = "memory_echo_cluster"

class BeliefTypeEnum(Enum): # Renamed from BeliefType
    COSMOLOGY = "cosmology"; ONTOLOGY = "ontology"; EPISTEMOLOGY = "epistemology"; AXIOLOGY = "axiology"; THEOLOGY = "theology"; ESCHATOLOGY = "eschatology"; SOCIOLOGY = "sociology_belief"; ANTHROPOLOGY = "anthropology_belief"; XENOLOGY = "xenology_belief"

class SocialStructureTypeEnum(Enum): # Renamed from SocialStructure (or SocialStructureType)
    NOMADIC_TRIBES = "nomadic_tribes"; SETTLED_COMMUNITIES = "settled_communities"; CHIEFDOMS_PRINCIPALITIES = "chiefdoms_principalities"; CITY_STATES_FEDERATIONS = "city_states_federations"; KINGDOMS_EMPIRES = "kingdoms_empires"; REPUBLIC_DEMOCRACY = "republic_democracy"; CORPORATOCRACY = "corporatocracy"; TECHNOCRACY = "technocracy"; THEOCRACY = "theocracy"; COLLECTIVE_CONSCIOUSNESS = "collective_consciousness"; DECENTRALIZED_NETWORK = "decentralized_network"; QUANTUM_GOVERNANCE = "quantum_governance"; ANARCHO_SYNDICALISM = "anarcho_syndicalism"; GALACTIC_FEDERATION = "galactic_federation"

class SymbolicArchetypeEnum(Enum): # Renamed from SymbolicArchetype
    THE_CREATOR = "the_creator"; THE_RULER = "the_ruler"; THE_SAGE = "the_sage"; THE_HERO = "the_hero"; THE_OUTLAW = "the_outlaw"; THE_EXPLORER = "the_explorer"; THE_LOVER = "the_lover"; THE_JESTER = "the_jester"; THE_CAREGIVER = "the_caregiver"; THE_MAGICIAN = "the_magician"; THE_INNOCENT = "the_innocent"; THE_ORPHAN = "the_orphan"; THE_DESTROYER = "the_destroyer"; THE_SHADOW_SELF = "the_shadow_self"

class InteractionTypeEnum(Enum): # Renamed from InteractionType
    FIRST_CONTACT = "first_contact"; WAR = "war"; PEACE_TREATY = "peace_treaty"; ALLIANCE = "alliance"; TRADE_AGREEMENT = "trade_agreement"; CULTURAL_EXCHANGE = "cultural_exchange"; TECHNOLOGICAL_COOPERATION = "technological_cooperation"; SUBJUGATION_VASSALAGE = "subjugation_vassalage"; COLD_WAR = "cold_war"; ESPIONAGE = "espionage"; FEDERATION_MEMBERSHIP = "federation_membership"; ISOLATIONISM_NON_INTERFERENCE = "isolationism_non_interference"; SCIENTIFIC_OBSERVATION = "scientific_observation"; IDEOLOGICAL_CONFLICT = "ideological_conflict"; PROTECTORATE = "protectorate"; MIGRATION_PACT = "migration_pact"; REQUEST_FOR_AID = "request_for_aid"; DISPUTE_MEDIATION = "dispute_mediation"

class FloralGrowthPatternEnum(Enum): # Renamed from FloralGrowthPattern
    BRANCHING = "branching"; SPIRAL = "spiral"; LAYERED = "layered"; FRACTAL = "fractal"; RADIAL = "radial"; LATTICE = "lattice"; CHAOTIC = "chaotic"; HARMONIC = "harmonic"; MIRRORED = "mirrored"; ADAPTIVE = "adaptive"

class NutrientTypeEnum(Enum): # Renamed from NutrientType
    PHYSICAL = "physical"; SYMBOLIC = "symbolic"; EMOTIONAL = "emotional"; TEMPORAL = "temporal"; ENTROPIC = "entropic"; HARMONIC = "harmonic"; VOID = "void"; NARRATIVE = "narrative"; QUANTUM = "quantum"; METAPHORIC = "metaphoric"

class FloraEvolutionStageEnum(Enum): # Renamed from FloraEvolutionStage
    SEED = "seed"; EMERGENT = "emergent"; MATURING = "maturing"; FLOWERING = "flowering"; SEEDING = "seeding"; WITHERING = "withering"; COMPOSTING = "composting"; DORMANT = "dormant"; RESURGENT = "resurgent"; TRANSCENDENT = "transcendent"

class EmotionalStateEnum(Enum): # Renamed from EmotionalState
    JOY = "joy"; SORROW = "sorrow"; FEAR = "fear"; ANGER = "anger"; WONDER = "wonder"; SERENITY = "serenity"; DETERMINATION = "determination"; CONFUSION = "confusion"; LONGING = "longing"; TRANSCENDENCE = "transcendence"

# ===== Core Data Structures =====

@dataclass
class CulturalTrait: 
    name: str
    description: str
    value: float 
    category: str 
    effects: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class DiplomaticRelationData: # Renamed from DiplomaticRelation
    civ_id_1: str
    civ_id_2: str
    current_status: InteractionTypeEnum = InteractionTypeEnum.ISOLATIONISM_NON_INTERFERENCE
    trust_level: float = 0.0
    cooperation_level: float = 0.0
    tension_level: float = 0.1
    last_interaction_tick: int = 0
    shared_history_event_ids: List[str] = field(default_factory=list)
    active_treaties: Dict[str, Any] = field(default_factory=dict)
    communication_established: bool = False
    
    @property
    def relation_id(self) -> str: 
        return "_".join(sorted(["relation", self.civ_id_1, self.civ_id_2]))
    
    def update_relation_metrics(self, trust_change: float=0, tension_change: float=0, coop_change: float=0):
        self.trust_level = max(-1.0, min(1.0, self.trust_level + trust_change))
        self.tension_level = max(0.0, min(1.0, self.tension_level + tension_change))
        self.cooperation_level = max(0.0, min(1.0, self.cooperation_level + coop_change))
    
    def __repr__(self) -> str: 
        return (f"DiplomaticRelationData({self.civ_id_1} <> {self.civ_id_2}: {self.current_status.name}, "
                f"T:{self.trust_level:.2f}, X:{self.tension_level:.2f})")

@dataclass
class MetabolicPathwayData: # Renamed from MetabolicPathway
    name: str
    input_resources: Dict[MetabolicResourceEnum, float]
    output_resources: Dict[MetabolicResourceEnum, float]
    byproducts: Dict[MetabolicResourceEnum, float] = field(default_factory=dict)
    efficiency: float = 0.7
    activation_threshold: float = 0.1
    processing_time: float = 1.0
    symbolic_catalysts: List[str] = field(default_factory=list)
    current_activity_level: float = 0.1 
    
    def can_activate(self, available_inputs: Dict[MetabolicResourceEnum, float]) -> bool:
        if not self.input_resources: return True
        for res, req_amt in self.input_resources.items():
            if available_inputs.get(res, 0) < req_amt * self.activation_threshold * self.current_activity_level: 
                return False
        return True
    
    def process(self, available_inputs: Dict[MetabolicResourceEnum, float], delta_time: float) -> Tuple[Dict[MetabolicResourceEnum, float], Dict[MetabolicResourceEnum, float], Dict[MetabolicResourceEnum, float]]:
        consumed = defaultdict(float)
        produced = defaultdict(float)
        byproducts_generated = defaultdict(float) # Renamed to avoid conflict with self.byproducts

        if self.current_activity_level <= 1e-6 or not self.can_activate(available_inputs):
            return consumed, produced, byproducts_generated

        max_cycles_by_res = float('inf')
        if self.input_resources:
            for res, req in self.input_resources.items():
                if req > 1e-6:
                    max_cycles_by_res = min(max_cycles_by_res, available_inputs.get(res, 0) / req)
        
        proc_rate_per_tick = (1.0 / self.processing_time if self.processing_time > 1e-6 else float('inf'))
        possible_cycles_this_tick = proc_rate_per_tick * self.current_activity_level * delta_time
        actual_cycles = min(max_cycles_by_res, possible_cycles_this_tick)

        if actual_cycles <= 1e-6:
            return consumed, produced, byproducts_generated
        
        for res, req_amt in self.input_resources.items(): 
            actual_consumed_for_resource = min(req_amt * actual_cycles, available_inputs.get(res,0))
            consumed[res] += actual_consumed_for_resource
        
        limiting_input_ratio = 1.0
        for res, req_amt in self.input_resources.items():
            if req_amt * actual_cycles > 1e-9: 
                 ratio = consumed.get(res,0) / (req_amt * actual_cycles)
                 limiting_input_ratio = min(limiting_input_ratio, ratio)
        actual_cycles_processed = actual_cycles * limiting_input_ratio

        for res, prod_amt in self.output_resources.items():
            produced[res] += prod_amt * actual_cycles_processed * self.efficiency
        for res, byproduct_amt in self.byproducts.items(): # Accessing the field of the dataclass instance
            byproducts_generated[res] += byproduct_amt * actual_cycles_processed
        
        return consumed, produced, byproducts_generated


    triggered_motif_ids: List[str] = field(default_factory=list) # Storing IDs for now
    location_context: Optional[Any] = None # Could be sector, coordinates, etc.
    emotional_valence: Optional[float] = None # e.g., -1 (negative) to 1 (positive)
    causal_event_id: Optional[str] = None # Link to a preceding event

    def to_dict(self) -> Dict[str, Any]:
        # Enum fields could be stored as their .value if needed for pure JSON
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrollMemoryEventData':
        # This ensures only fields defined in the dataclass are passed to the constructor
        # which is good practice if the data source might have extra fields.
        known_field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_field_names}
        # Potential future enhancement: Convert event_type string back to EventTypeEnum if needed.
        return cls(**filtered_data)

    def __str__(self) -> str:
        return (f"[T:{self.timestamp:.1f},Type:{self.event_type},Imp:{self.importance:.2f}] "
                f"{self.description[:60]}{'...' if len(self.description) > 60 else ''}")


class ScrollMemory:
    """
    Memory system that records significant events in an entity's history.
    Provides continuity and emergence of meaningful narratives.
    """
    def __init__(self, owner_id: str, capacity: int = 100, manager_ref: Optional['CosmicScrollManager'] = None):
        self.owner_id: str = owner_id
        self.capacity: int = capacity
        self.events: deque[ScrollMemoryEventData] = deque(maxlen=capacity)
        self.last_consolidation_tick: int = 0
        self.thematic_summary: Dict[str, int] = defaultdict(int) # event_type_str -> count
        self.manager_ref: Optional['CosmicScrollManager'] = manager_ref # For context if needed

    def record_memory(self, event_data: ScrollMemoryEventData, current_tick: int):
        """
        Add a new event to the memory scroll.
        The deque will automatically handle capacity by discarding the oldest events.
        """
        self.events.append(event_data)
        self.thematic_summary[event_data.event_type] += 1 # event_type is already string

        # Basic consolidation relies on deque's maxlen.
        # More advanced consolidation could be triggered periodically.
        if current_tick - self.last_consolidation_tick > self.capacity * 0.5: # Example: consolidate every half-capacity worth of ticks
            self._consolidate_memory(current_tick)


    def _consolidate_memory(self, current_tick: int):
        """
        Placeholder for advanced memory consolidation strategies.
        Currently, deque's maxlen handles basic capacity. This method could, in the future,
        summarize old events, merge similar ones, or identify long-term patterns.
        """
        # For now, just update the consolidation tick.
        # If events were actually removed/merged in a more complex way,
        # self.thematic_summary might need rebuilding.
        logger.debug(f"ScrollMemory for {self.owner_id}: _consolidate_memory called at tick {current_tick}. Currently relies on deque maxlen.")
        self.last_consolidation_tick = current_tick
        # Example future logic:
        # if len(self.events) > self.capacity * 0.8: # Only run if significantly full
            # Prune by importance beyond maxlen, or merge related low-importance events.
            # For example, remove 10% of the least important events if over capacity by a certain threshold
            # num_to_prune = len(self.events) - int(self.capacity * 0.9)
            # if num_to_prune > 0:
            #     sorted_by_importance = sorted(list(self.events), key=lambda e: e.importance)
            #     removed_count = 0
            #     new_events_list = []
            #     # ... logic to decide which to keep vs remove/summarize
            #     # This part is complex and deferred.
            #     pass


    def get_events_by_type(self, event_type_str: str) -> List[ScrollMemoryEventData]:
        """Retrieve all events of a specific type string."""
        return [e for e in self.events if e.event_type == event_type_str]

    def get_events_by_timeframe(self, start_timestamp: float, end_timestamp: float) -> List[ScrollMemoryEventData]:
        """Retrieve events within a specific timeframe."""
        return [e for e in self.events if start_timestamp <= e.timestamp <= end_timestamp]

    def get_events_involving_entity(self, entity_id: str) -> List[ScrollMemoryEventData]:
        """Retrieve events involving a specific entity."""
        return [e for e in self.events if entity_id in e.involved_entity_ids]

    def get_most_important_events(self, count: int = 10) -> List[ScrollMemoryEventData]:
        """Retrieve the most important events."""
        return sorted(list(self.events), key=lambda e: e.importance, reverse=True)[:count]

    def get_memory_keywords(self, count: int = 5) -> List[str]:
        """Extract keywords that define this entity's identity based on memory."""
        keywords: List[str] = []
        
        # Add most frequent event types
        sorted_event_types = sorted(self.thematic_summary.items(), key=lambda x: x[1], reverse=True)
        for event_type_str, _ in sorted_event_types[:count]:
            keywords.append(event_type_str)
            
        # Future: Could add common motifs from triggered_motif_ids or terms from descriptions
        return keywords

    def generate_timeline_summary(self, max_events: int = 5) -> str:
        """Generate a textual summary of the entity's history."""
        if not self.events:
            return "No recorded history."
        
        # Get a mix of recent and important events for summary
        recent_events = list(self.events)[-max_events:]
        important_events = self.get_most_important_events(max_events)
        
        # Combine and unique
        summary_events_dict: Dict[float, ScrollMemoryEventData] = {e.timestamp: e for e in recent_events}
        for e in important_events:
            if e.timestamp not in summary_events_dict: # Prioritize recent if timestamp collision
                 summary_events_dict[e.timestamp] = e
        
        # Sort chronologically for the summary
        sorted_summary_events = sorted(list(summary_events_dict.values()), key=lambda e: e.timestamp, reverse=True)[:max_events]

        summary_lines = [f"Key memories for {self.owner_id}:"]
        for event_data in sorted_summary_events:
            summary_lines.append(f"  - {str(event_data)}") # Uses ScrollMemoryEventData.__str__
        
        return "\n".join(summary_lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ScrollMemory state."""
        return {
            "owner_id": self.owner_id,
            "capacity": self.capacity,
            "events": [event.to_dict() for event in self.events], # Serialize each event
            "last_consolidation_tick": self.last_consolidation_tick,
            "thematic_summary": dict(self.thematic_summary) # Convert defaultdict to dict
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], manager_ref: Optional['CosmicScrollManager'] = None) -> 'ScrollMemory':
        """Deserialize ScrollMemory state."""
        memory = cls(
            owner_id=data["owner_id"],
            capacity=data["capacity"],
            manager_ref=manager_ref
        )
        # Events are stored with maxlen, so appending will maintain capacity.
        # If events were stored beyond maxlen in dict, this would truncate to capacity.
        raw_events = data.get("events", [])
        memory.events.extend([ScrollMemoryEventData.from_dict(e_data) for e_data in raw_events])
        
        memory.last_consolidation_tick = data.get("last_consolidation_tick", 0)
        memory.thematic_summary = defaultdict(int, data.get("thematic_summary", {}))
        return memory

class Motif:
    def __init__(self, name: str, category: MotifCategoryEnum, 
                 attributes: Dict[str, Any], description: str = "", 
                 intensity: float = 0.5, base_resonance: float = 0.5,
                 creation_tick: int = 0): 
        self.id: str = str(uuid.uuid4())
        self.name: str = name
        self.category: MotifCategoryEnum = category 
        self.attributes: Dict[str, Any] = attributes
        self.description: str = description
        self.intensity: float = intensity
        self.current_resonance: float = base_resonance 
        self.creation_tick: int = creation_tick 
        self.last_activation_tick: int = creation_tick 
        self.associated_entities: Set[str] = set() 
    
    def update_resonance(self, global_field_resonance: float, activity_bonus: float = 0.0) -> None:
        decay_rate = 0.01 
        self.current_resonance += (global_field_resonance - self.current_resonance) * decay_rate + activity_bonus
        self.current_resonance = max(0.0, min(1.0, self.current_resonance))
    
    def __repr__(self) -> str: 
        return (f"Motif(ID: {self.id[-6:]}, Name: {self.name}, Cat: {self.category.name}, "
                f"Int:{self.intensity:.2f}, Res:{self.current_resonance:.2f})")

class Entity: 
    def __init__(self, name: str, entity_type: EntityTypeEnum, 
                 properties: Optional[Dict[str, Any]] = None, 
                 initial_motifs: Optional[List[Motif]] = None,
                 initial_energy: float = 100.0,
                 creation_tick: int = 0):
        self.id: str = str(uuid.uuid4())
        self.name: str = name
        self.entity_type: EntityTypeEnum = entity_type 
        self.properties: Dict[str, Any] = properties or {}
        self.active_motifs: Dict[str, Motif] = {}
        if initial_motifs: 
            for m in initial_motifs: self.add_motif(m)
        
        self.creation_tick: int = creation_tick
        self.last_update_tick: int = creation_tick
        self.age: float = 0.0
        self.energy_level: float = initial_energy
        self.current_state: str = "active" 
        self.relationships: Dict[str, str] = {} 
        self.spatial_sectors: Set[Any] = set() 
        self.manager_ref: Optional['CosmicScrollManager'] = None # Forward reference

    def add_motif(self, motif: Motif, source_event_id: Optional[str] = None) -> None:
        if motif.id not in self.active_motifs:
            self.active_motifs[motif.id] = motif
            motif.associated_entities.add(self.id)
            
    def remove_motif(self, motif_id: str, source_event_id: Optional[str] = None) -> bool:
        m = self.active_motifs.pop(motif_id, None)
        if m:
            m.associated_entities.discard(self.id)
            return True
        return False
    
    def has_motif(self, name_or_id: str) -> bool:
        if name_or_id in self.active_motifs: return True
        return any(m.name == name_or_id for m in self.active_motifs.values())
    
    def get_property(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)
    
    def set_property(self, key: str, value: Any) -> None:
        self.properties[key] = value
    
    def update_energy(self, amount: float, source: Optional[str] = None) -> None:
        self.energy_level = max(0, self.energy_level + amount)
        if self.energy_level == 0 and self.current_state not in ["depleted", "destroyed"]:
            self.current_state = "depleted"
            
    def evolve(self, tick_data: Dict[str, Any]) -> List['Event']: # Forward reference for Event
        delta_time = tick_data.get("delta_time", 1.0)
        self.last_update_tick = tick_data.get("tick_number", self.last_update_tick + 1)
        self.age += delta_time
        
        upkeep_cost = (0.01 + 0.005 * len(self.active_motifs) + 0.002 * len(self.properties)) * delta_time
        self.update_energy(-upkeep_cost, "base_upkeep")
        
        if self.energy_level == 0 and self.current_state not in ["depleted", "destroyed", "dormant_depleted"]:
            self.current_state = "dormant_depleted"
            
        return [] 
    
    def register_in_drm(self, sectors: Optional[List[Any]] = None) -> None:
        logger.debug(f"Entity {self.id} registration in DRM called (DRM integration needed).")

    def update_spatial_sectors(self, new_sectors: List[Any]) -> None:
        logger.debug(f"Entity {self.id} sector update called (DRM integration needed).")

    def __repr__(self) -> str:
        return (f"Entity(ID: {self.id[-6:]}, Name: {self.name}, EType: {self.entity_type.name}, "
                f"State: {self.current_state}, Energy: {self.energy_level:.1f})")

class Event: 
    def __init__(self, event_type: EventTypeEnum, description: str, 
                 involved_entity_ids: List[str], 
                 properties: Optional[Dict[str, Any]] = None, 
                 triggered_motifs: Optional[List[Motif]] = None,
                 importance: float = 0.5, simulation_tick: int = 0): 
        self.id: str = str(uuid.uuid4())
        self.event_type: EventTypeEnum = event_type 
        self.description: str = description
        self.involved_entity_ids: List[str] = involved_entity_ids
        self.properties: Dict[str, Any] = properties or {}
        self.timestamp: datetime = datetime.now() 
        self.simulation_tick: int = simulation_tick 
        self.triggered_motifs: Dict[str, Motif] = {m.id: m for m in triggered_motifs} if triggered_motifs else {}
        self.importance: float = importance
    
    def add_involved_entity(self, entity_id: str) -> None: 
        if entity_id not in self.involved_entity_ids:
            self.involved_entity_ids.append(entity_id)
    
    def add_triggered_motif(self, motif: Motif) -> None:
        if motif.id not in self.triggered_motifs:
            self.triggered_motifs[motif.id] = motif
            motif.last_activation_tick = self.simulation_tick 
    
    def __repr__(self) -> str:
        motif_names = [m.name for m in self.triggered_motifs.values()][:2]
        return (f"Event(ID: {self.id[-6:]}, Tick: {self.simulation_tick}, Type: {self.event_type.name}, "
                f"Imp: {self.importance:.2f}, Desc: '{self.description[:30]}...', Motifs: {motif_names})")


class MetabolicProcess: 
    def __init__(self, name: str, process_type: MetabolicProcessTypeEnum, 
                 target_entity_ids: List[str], 
                 rate: float = 1.0, efficiency: float = 0.7, duration: Optional[float] = None,
                 start_tick: int = 0): 
        self.id: str = str(uuid.uuid4())
        self.name: str = name
        self.process_type: MetabolicProcessTypeEnum = process_type 
        self.target_entity_ids: List[str] = target_entity_ids
        self.rate: float = rate
        self.efficiency: float = efficiency
        self.is_active: bool = True
        self.start_tick: int = start_tick 
        self.duration: Optional[float] = duration 
        self.elapsed_ticks: float = 0.0
        self.byproducts: Dict[MetabolicResourceEnum, float] = {} 
    
    def execute_step(self, delta_time: float, entities_accessor: Callable[[str], Optional[Entity]]) -> Dict[str, Any]:
        if not self.is_active:
            return {"status": "inactive", "process_id": self.id}

        self.elapsed_ticks += delta_time
        
        if self.duration is not None and self.elapsed_ticks >= self.duration:
            self.is_active = False
            logger.info(f"MetabolicProcess {self.name} ({self.id}) completed.")
            return {"status": "completed", "process_id": self.id, "final_output": {}}

        energy_consumed_total = 0.0
        outputs_generated = defaultdict(float)
        
        for entity_id in self.target_entity_ids:
            target_entity = entities_accessor(entity_id)
            if not target_entity or target_entity.current_state == "depleted":
                continue
            
            cost_to_entity = (self.rate * delta_time) / self.efficiency if self.efficiency > 1e-6 else float('inf')
            actual_consumed_from_entity = min(cost_to_entity, target_entity.energy_level)
            target_entity.update_energy(-actual_consumed_from_entity, source=f"metabolic_process_{self.name}")
            energy_consumed_total += actual_consumed_from_entity
            
            processed_amount_effective = actual_consumed_from_entity * self.efficiency 
            
            outputs_generated[MetabolicResourceEnum.COMPLEXITY] += processed_amount_effective * 0.1 
            for byproduct_res, amount_per_rate_unit in self.byproducts.items():
                outputs_generated[byproduct_res] += amount_per_rate_unit * processed_amount_effective
        
        return {
            "status": "active", 
            "process_id": self.id, 
            "energy_consumed": energy_consumed_total, 
            "outputs": dict(outputs_generated), 
            "progress": min(1.0, (self.elapsed_ticks / self.duration) if self.duration else 0.1)
        }

    def set_byproducts(self, byproducts: Dict[MetabolicResourceEnum, float]): 
        self.byproducts = byproducts
    
    def __repr__(self) -> str:
        return (f"MetabolicProcess(ID: {self.id[-6:]}, Name: {self.name}, "
                f"Type: {self.process_type.name}, Active: {self.is_active})")


class CosmicEntity(Entity): 
    def __init__(self, name: str, entity_type: EntityTypeEnum, 
                 initial_properties: Optional[Dict[str, Any]] = None,
                 initial_motifs: Optional[List[Motif]] = None,
                 creation_tick: int = 0,
                 initial_energy: float = 100.0):
        super().__init__(name, entity_type, initial_properties, initial_motifs, creation_tick=creation_tick, initial_energy=initial_energy)
        self._scroll_memory_instance_: Optional[ScrollMemory] = None
        self._culture_engine_instance_: Optional[CultureEngine] = None
        self._recursive_metabolism_instance_: Optional[RecursiveMetabolism] = None
        self._emotional_resonance_body_instance_: Optional[Any] = None 
        self.planet_id: Optional[str] = None 
        self.star_id: Optional[str] = None 
        self.galaxy_id: Optional[str] = None 
        self.universe_id: Optional[str] = None 
        self.home_sector: Optional[Any] = None 
        self.position: Tuple[float, float, float] = (0.0, 0.0, 0.0) 

    def record_memory(self, event_type_str: str, description: str, 
                      importance: float, manager: 'CosmicScrollManager', 
                      involved_entity_ids: Optional[List[str]] = None, 
                      triggered_motif_ids: Optional[List[str]] = None,
                      location_context: Optional[Any] = None,
                      emotional_valence: Optional[float] = None,
                      causal_event_id: Optional[str] = None) -> None:
        
        memory_system = self.get_or_create_scroll_memory(manager)
        
        actual_involved_ids = involved_entity_ids if involved_entity_ids is not None else []
        if self.id not in actual_involved_ids: 
            actual_involved_ids.append(self.id)

        event_data = ScrollMemoryEventData(
            timestamp=manager.current_time, 
            event_type=event_type_str, 
            description=description,
            importance=importance,
            involved_entity_ids=actual_involved_ids,
            triggered_motif_ids=triggered_motif_ids or [],
            location_context=location_context,
            emotional_valence=emotional_valence,
            causal_event_id=causal_event_id
        )
        memory_system.record_memory(event_data, manager.tick_count)

    def get_or_create_scroll_memory(self, manager: Optional['CosmicScrollManager'] = None) -> ScrollMemory:
        if self._scroll_memory_instance_ is None:
            # Use passed manager, or self.manager_ref if available, or None
            effective_manager = manager if manager else self.manager_ref
            self._scroll_memory_instance_ = ScrollMemory(owner_id=self.id, manager_ref=effective_manager)
            logger.info(f"ScrollMemory initialized for Entity {self.id} ({self.name}).")
        return self._scroll_memory_instance_

    def get_scroll_memory(self, manager: Optional['CosmicScrollManager'] = None) -> ScrollMemory:
        """Public accessor for an entity's scroll memory, ensures initialization."""
        return self.get_or_create_scroll_memory(manager)


    # Placeholder for other system getters like run_metabolic_cycle
    # These would typically be initialized by their respective managers or in subclass constructors.

# Placeholder for DimensionalRealityManager (DRM)
# This would be a more complex class managing spatial indexing of entities.
class DimensionalRealityManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DimensionalRealityManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.entities: Dict[str, CosmicEntity] = {}
        self.entity_sector_map: Dict[str, Set[Tuple[int, ...]]] = defaultdict(set) # Sectors are tuples of ints
        self.sector_entity_map: Dict[Tuple[int, ...], Dict[EntityTypeEnum, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        self.query_cache: Dict[Tuple, List[str]] = {} # (entity_type, sector_tuple_or_None) -> List[entity_id]
        self.active_observers: Dict[str, Dict[str, Any]] = {} # observer_id -> {"position": tuple, "radius": float, "last_seen_tick": int}
        
        # For _get_sectors_in_radius. Defines the size of a grid cell for spatial indexing.
        self.sector_size: float = 100.0 
        
        # Conceptual map for reality coherence. Actual implementation of its effects would be elsewhere.
        # If used, it would need dimensions, e.g., based on WORLD_SIZE if that constant were accessible here.
        # For now, operations on it will be conceptual (logging).
        self.reality_coherence_map: Optional[np.ndarray] = None 
        # Example initialization if WORLD_SIZE were available:
        # self.reality_coherence_map = np.ones((WORLD_SIZE // self.sector_size, WORLD_SIZE // self.sector_size))

        logger.info("DimensionalRealityManager initialized with cache, observers, and sector_size.")

    def _get_sector_from_position(self, position: Tuple[float, ...]) -> Tuple[int, ...]:
        """Converts a world position to a sector tuple."""
        if not self.sector_size or self.sector_size <= 0:
            logger.error("Sector size is not configured correctly in DRM.")
            return tuple(int(p) for p in position) # Fallback, though likely incorrect
        return tuple(int(p / self.sector_size) for p in position)

    def _get_sectors_in_radius(self, center_pos_world: Tuple[float, ...], radius_world: float) -> List[Tuple[int, ...]]:
        """
        Calculates all grid sectors that fall within a given world radius from a world center.
        Assumes sectors are defined by integer coordinates based on self.sector_size.
        """
        if radius_world < 0 or self.sector_size <= 0:
            return []

        center_sector = self._get_sector_from_position(center_pos_world)
        # Determine the search range in terms of sectors
        # Add 1 to radius in sector units to be conservative due to integer conversion of sectors
        radius_in_sectors = math.ceil(radius_world / self.sector_size) 
        
        affected_sectors: Set[Tuple[int, ...]] = set()
        
        # Simple N-dimensional bounding box iteration (example for 3D)
        # This can be optimized (e.g. check sphere-sector intersection more precisely)
        # For now, iterate a bounding box of sectors and check distance from world_center to sector_center
        
        if not center_sector: # Should not happen if center_pos_world is valid
            return []

        num_dimensions = len(center_sector)
        
        # Create iterators for each dimension of the bounding box
        # Example: for 3D, center_sector = (csx, csy, csz)
        # iter_ranges = [range(cs_i - radius_in_sectors, cs_i + radius_in_sectors + 1) for cs_i in center_sector]
        
        # This part is tricky to make truly N-dimensional without recursion or more complex loops.
        # For a common 2D or 3D case:
        if num_dimensions == 2:
            for dx in range(-radius_in_sectors, radius_in_sectors + 1):
                for dy in range(-radius_in_sectors, radius_in_sectors + 1):
                    test_sector = (center_sector[0] + dx, center_sector[1] + dy)
                    # Check if this sector (its center or any part) is within the radius
                    # A simple check: distance from center_pos_world to the center of test_sector
                    test_sector_center_world = tuple((ts_i + 0.5) * self.sector_size for ts_i in test_sector)
                    dist_sq = sum((c_w - ts_w)**2 for c_w, ts_w in zip(center_pos_world, test_sector_center_world))
                    if dist_sq <= radius_world**2:
                        affected_sectors.add(test_sector)
        elif num_dimensions == 3:
            for dx in range(-radius_in_sectors, radius_in_sectors + 1):
                for dy in range(-radius_in_sectors, radius_in_sectors + 1):
                    for dz in range(-radius_in_sectors, radius_in_sectors + 1):
                        test_sector = (center_sector[0] + dx, center_sector[1] + dy, center_sector[2] + dz)
                        test_sector_center_world = tuple((ts_i + 0.5) * self.sector_size for ts_i in test_sector)
                        dist_sq = sum((c_w - ts_w)**2 for c_w, ts_w in zip(center_pos_world, test_sector_center_world))
                        # A more accurate check would involve sphere-AABB intersection.
                        # For simplicity, if sector center is within radius + sector diagonal/2
                        # This is an approximation.
                        if dist_sq <= (radius_world + (self.sector_size * math.sqrt(num_dimensions) / 2))**2:
                             # Check if any corner of the sector is within the radius for better accuracy
                            corners = []
                            for i in range(1 << num_dimensions): # Iterate over 2^N corners
                                corner_offset = [( (i >> j) & 1 ) for j in range(num_dimensions)]
                                corner_world_pos = tuple((ts_coord + offset_val) * self.sector_size for ts_coord, offset_val in zip(test_sector, corner_offset))
                                corners.append(corner_world_pos)
                            
                            # Check if sphere center is within minkowski sum of sector AABB and sphere (radius -radius_world)
                            # Or, simpler: if any corner is in radius, or if sphere center is in sector, or if sphere intersects sector boundaries
                            # For now, simplified check:
                            if dist_sq <= radius_world**2: # if center of sector is in radius
                                affected_sectors.add(test_sector)
                            else: # Check if any corner is within radius
                                for corner in corners:
                                    dist_sq_corner = sum((c_w - cr_w)**2 for c_w, cr_w in zip(center_pos_world, corner))
                                    if dist_sq_corner <= radius_world**2:
                                        affected_sectors.add(test_sector)
                                        break 
        else:
            logger.warning(f"DRM._get_sectors_in_radius currently only supports 2D/3D. Found {num_dimensions} dimensions.")
            affected_sectors.add(center_sector) # Fallback for other dimensions

        return list(affected_sectors)

    def query_entities(self, entity_type: EntityTypeEnum, sector: Optional[Tuple[int, ...]] = None, use_cache: bool = True) -> List[CosmicEntity]:
        cache_key = (entity_type, sector)
        if use_cache and cache_key in self.query_cache:
            entity_ids = self.query_cache[cache_key]
            return [self.entities[eid] for eid in entity_ids if eid in self.entities]

        found_entities_ids: Set[str] = set()
        if sector is None: # Query across all sectors
            for entity_obj in self.entities.values():
                if entity_obj.entity_type == entity_type:
                    found_entities_ids.add(entity_obj.id)
        else: # Query specific sector
            if sector in self.sector_entity_map and entity_type in self.sector_entity_map[sector]:
                found_entities_ids.update(self.sector_entity_map[sector][entity_type])
        
        # Store IDs in cache
        self.query_cache[cache_key] = list(found_entities_ids)
        
        # Return actual entity objects
        return [self.entities[eid] for eid in found_entities_ids if eid in self.entities]

    def invalidate_cache(self, sector: Optional[Tuple[int, ...]] = None, entity_type: Optional[EntityTypeEnum] = None):
        if sector is None and entity_type is None:
            self.query_cache.clear()
            logger.debug("DRM cache fully invalidated.")
            return

        keys_to_remove = []
        for cache_key_type, cache_key_sector in self.query_cache.keys():
            match = True
            if entity_type is not None and cache_key_type != entity_type:
                match = False
            if sector is not None and cache_key_sector != sector:
                match = False
            
            if match:
                keys_to_remove.append((cache_key_type, cache_key_sector))
        
        for key in keys_to_remove:
            if key in self.query_cache: # Check if still exists, could be removed by another match
                 del self.query_cache[key]
        logger.debug(f"DRM cache invalidated for sector: {sector}, type: {entity_type}. Removed {len(keys_to_remove)} entries.")


    def get_entity(self, entity_id: str) -> Optional[CosmicEntity]:
        return self.entities.get(entity_id)

    def register_entity(self, entity: CosmicEntity, sectors: Optional[List[Tuple[int, ...]]] = None):
        self.entities[entity.id] = entity
        
        # Determine sectors from entity position if not provided
        entity_sectors_tuples: List[Tuple[int, ...]]
        if sectors:
            entity_sectors_tuples = sectors
        elif hasattr(entity, 'position') and entity.position is not None:
            entity_sectors_tuples = [self._get_sector_from_position(entity.position)]
        else:
            entity_sectors_tuples = [] # Cannot determine sector
            logger.warning(f"Entity {entity.id} registered without sector information or position.")

        for sector_tuple in entity_sectors_tuples:
            self.entity_sector_map[entity.id].add(sector_tuple)
            self.sector_entity_map[sector_tuple][entity.entity_type].add(entity.id)
            self.invalidate_cache(sector=sector_tuple, entity_type=entity.entity_type) # Invalidate cache for this specific sector/type
        
        entity.spatial_sectors.update(entity_sectors_tuples) # Ensure entity itself knows its sectors
        logger.info(f"Entity {entity.id} ({entity.name}) registered in DRM in sectors: {entity_sectors_tuples}")


    def update_entity_sectors(self, entity_id: str, old_sectors: List[Tuple[int, ...]], new_sectors: List[Tuple[int, ...]]):
        entity = self.entities.get(entity_id)
        if not entity: 
            logger.warning(f"Attempted to update sectors for non-existent entity ID: {entity_id}")
            return

        current_entity_type = entity.entity_type

        for sector_tuple in old_sectors:
            if sector_tuple in self.sector_entity_map and current_entity_type in self.sector_entity_map[sector_tuple]:
                self.sector_entity_map[sector_tuple][current_entity_type].discard(entity_id)
            self.entity_sector_map[entity_id].discard(sector_tuple)
            self.invalidate_cache(sector=sector_tuple, entity_type=current_entity_type)

        for sector_tuple in new_sectors:
            self.sector_entity_map[sector_tuple][current_entity_type].add(entity_id)
            self.entity_sector_map[entity_id].add(sector_tuple)
            self.invalidate_cache(sector=sector_tuple, entity_type=current_entity_type)
        
        if hasattr(entity, 'spatial_sectors'): 
            entity.spatial_sectors = set(new_sectors)
        logger.debug(f"Entity {entity_id} sectors updated. Old: {old_sectors}, New: {new_sectors}")

    def register_observer(self, observer_id: str, position: Tuple[float, ...], observation_radius: float = 100.0, current_tick: int = 0):
        self.active_observers[observer_id] = {
            "position": position,
            "radius": observation_radius,
            "last_seen_tick": current_tick # Requires current_tick to be passed or DRM needs manager_ref
        }
        logger.info(f"Observer {observer_id} registered at {position} with radius {observation_radius}.")
        self._adjust_reality_coherence(position, observation_radius)

    def _adjust_reality_coherence(self, focal_point: Tuple[float, ...], radius: float, coherence_boost: float = 0.1):
        affected_sectors = self._get_sectors_in_radius(focal_point, radius)
        logger.info(f"Adjusting reality coherence (conceptually) for {len(affected_sectors)} sectors around {focal_point} within radius {radius}.")
        # Conceptual: If self.reality_coherence_map was initialized as a NumPy array:
        # for sector_coords in affected_sectors:
        #     # This assumes sector_coords can directly index the map, might need adjustment based on map structure
        #     try:
        #         if self.reality_coherence_map is not None: # Check if it's initialized
        #             # Ensure indices are within bounds
        #             # This part is highly dependent on the map's dimensions and indexing scheme
        #             # For a simple 2D example:
        #             if len(sector_coords) == 2 and \
        #                0 <= sector_coords[0] < self.reality_coherence_map.shape[0] and \
        #                0 <= sector_coords[1] < self.reality_coherence_map.shape[1]:
        #                 self.reality_coherence_map[sector_coords] = min(1.0, self.reality_coherence_map[sector_coords] + coherence_boost)
        #     except IndexError:
        #         logger.warning(f"Sector coordinates {sector_coords} out of bounds for reality_coherence_map.")
        #     except TypeError: # If map is None or coords are not valid indices
        #          logger.warning(f"Could not apply coherence boost for sector {sector_coords} due to map/coordinate issue.")
        # For now, this remains a conceptual operation logged above.
        # The actual effect would be entities in these sectors checking a "coherence level"
        # or the simulation manager prioritizing updates for entities in high-coherence sectors.
        if affected_sectors: # Log which sectors are affected
            logger.debug(f"Sectors affected by coherence adjustment: {affected_sectors[:5]}...")


DRM = DimensionalRealityManager() # Instantiate the singleton

logger.info("mind_seed.py core definitions established.")
