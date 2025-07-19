# === Integrated Environmental Scroll Modules ===
# Source: scroll_modules.py
# This module contains the integrated environmental scroll modules for the Cosmic Scroll system.
# === World & Environmental Systems ===

import random
import logging
import math
import uuid
import time
import os
import json
from enum import Enum, auto # Retain for local enums if any, or if used by remaining classes directly
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Tuple, Set, Callable # Ensure all used typings are here

# Import from mind_seed.py
from mind_seed import (
    MotifCategoryEnum, EntityTypeEnum, EventTypeEnum, MetabolicProcessTypeEnum, 
    BreathPhaseEnum, MutationTypeEnum, MetabolicResourceEnum, GalaxyTypeEnum, 
    StarTypeEnum, PlanetTypeEnum, CivilizationTypeEnum, DevelopmentAreaEnum, 
    AnomalyTypeEnum, BeliefTypeEnum, SocialStructureTypeEnum, SymbolicArchetypeEnum,
    InteractionTypeEnum, FloralGrowthPatternEnum, NutrientTypeEnum, 
    FloraEvolutionStageEnum, EmotionalStateEnum,
    CulturalTrait, DiplomaticRelationData, MetabolicPathwayData, ScrollMemoryEventData,
    Motif, Entity, Event, MetabolicProcess, # Note: MetabolicProcess is the class here
    CosmicEntity,
    DimensionalRealityManager, # DRM is a singleton, usually instantiated in mind_seed
    DRM # Import the instance
)
# Assuming QuantumPhysics, QuantumBridge, ParadoxEngine, HarmonicEngine, PerceptionModule, AetherEngine
# are defined elsewhere or are specific and not part of the core consolidation.
# If they are also core, they should be in mind_seed. For now, I'll assume they are separate.

# Configure logging
# Ensure this is consistent with how logging is handled in mind_seed.py
# If mind_seed.py configures a root logger, this might not be necessary or could be simplified.
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler() # Default to console output if not configured
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Default level

WORLD_SIZE = 100 

# ===== Placeholder Classes (if any were *not* moved to mind_seed and are still needed) =====
# Example:
# class QuantumPhysics: pass 
# class QuantumBridge: pass
# class ParadoxEngine: pass
# class HarmonicEngine: pass
# class PerceptionModule: pass
# class AetherEngine: pass


# ===== Cosmos Scroll Components (Classes that are specific to cosmic_scroll.py or orchestrate mind_seed types) =====

class CosmicScroll: # This is likely the repository/data store for patterns
    """
    The core symbolic pattern repository of the simulation.
    Stores and manages the fundamental patterns that give rise to reality.
    """
    def __init__(self):
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.active_threads: List[str] = []
        self.dormant_threads: List[str] = []
        self.symbolic_density: float = 0.0
        # Removed entities, world_state, motif_library, motif_feedback_queue, 
        # entity_motifs, entity_types, event_history, tick_count, 
        # breath_phase, breath_progress, simulation_history (or self.history)
        
    def add_pattern(self, pattern_id: str, pattern_data: Dict[str, Any]):
        """Add a new pattern to the scroll"""
        self.patterns[pattern_id] = pattern_data
        logger.info(f"Pattern '{pattern_id}' added to CosmicScroll.")
        
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a pattern by ID"""
        return self.patterns.get(pattern_id)
        
    def activate_thread(self, thread_id: str) -> bool:
        """Activate a dormant narrative thread"""
        if thread_id in self.dormant_threads:
            self.dormant_threads.remove(thread_id)
            self.active_threads.append(thread_id)
            logger.info(f"Narrative thread activated: {thread_id}")
            return True
        logger.warning(f"Attempted to activate non-dormant or unknown thread: {thread_id}")
        return False
        
    def calculate_symbolic_density(self) -> float:
        """Calculate the current symbolic density of the scroll"""
        pattern_count = len(self.patterns)
        thread_count = len(self.active_threads) + len(self.dormant_threads)
        
        if pattern_count == 0 and thread_count == 0: # Avoid division by zero if empty
            self.symbolic_density = 0.0
            return 0.0
            
        # Robust calculation for symbolic_density
        total_elements = pattern_count + thread_count
        if total_elements == 0:
            self.symbolic_density = 0.0
        else:
            # Weighted average, ensuring weights sum to 1 if both are present,
            # or full weight to one if the other is zero.
            w_pattern = 0.7 if pattern_count > 0 else 0.0
            w_thread = 0.3 if thread_count > 0 else 0.0
            if pattern_count > 0 and thread_count > 0:
                norm_factor = w_pattern + w_thread # Should be 1.0
            elif pattern_count > 0:
                norm_factor = w_pattern
            elif thread_count > 0:
                norm_factor = w_thread
            else: # Should not happen if total_elements > 0
                norm_factor = 1.0 

            if norm_factor > 0: # Avoid division by zero if somehow weights are zero
                self.symbolic_density = (w_pattern * pattern_count + w_thread * thread_count) / ( (pattern_count + thread_count) * norm_factor)
            else:
                self.symbolic_density = 0.0
        return self.symbolic_density

    def get_motif_feedback(self, max_items: int = 10) -> List[Dict[str, Any]]:
        recent_motifs_data: List[Dict[str, Any]] = []
        # Iterate over a copy for safety if queue could be modified elsewhere (though unlikely here)
        for motif_obj in list(self.motif_feedback_queue)[-max_items:]: 
            if isinstance(motif_obj, Motif):
                 recent_motifs_data.append({
                     "id": motif_obj.id, 
                     "name": motif_obj.name, 
                     "category": motif_obj.category.value, 
                     "resonance": motif_obj.current_resonance
                 })
            else:
                logger.warning(f"Non-Motif object found in motif_feedback_queue: {type(motif_obj)}")
        
        feedback = {
            "tick_count": self.tick_count,
            "breath_phase": self.breath_phase.value, 
            "breath_progress": self.breath_progress,
            "motifs": recent_motifs_data,
            "motif_count": len(self.motif_library),
            "entity_count": len(self.entities),
            "dominant_categories": self._get_dominant_motif_categories() # Call to updated method
        }
        return feedback
    
    def _get_dominant_motif_categories(self) -> Dict[str, float]:
        category_strengths: Dict[MotifCategoryEnum, float] = defaultdict(float)
        for motif_obj in self.motif_library.values(): # Iterate directly over Motif objects
            if isinstance(motif_obj, Motif): # Redundant if library is well-maintained, but safe
                 category_strengths[motif_obj.category] += motif_obj.current_resonance
            else:
                 logger.warning(f"Invalid item in motif_library: {motif_obj}")

        total_strength = sum(category_strengths.values())
        if total_strength == 0:
            # Return all categories with 0 strength if no motifs or all have 0 resonance
            return {cat.value: 0.0 for cat in MotifCategoryEnum} 

        return {category.value: strength / total_strength for category, strength in category_strengths.items()}

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: EntityTypeEnum) -> List[Entity]:
        entity_ids = self.entity_types.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

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
            # Initialization moved to _initialize to ensure it happens only once
        return cls._instance
    
    def __init__(self): # Ensure __init__ can be called multiple times without re-initializing if new returns existing
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialize()
        self._initialized = True

    def _initialize(self):
        """Initialize the Cosmic Scroll Manager"""
        self.entities: Dict[str, Entity] = {}  # entity_id -> entity object
        self.entity_types: Dict[EntityTypeEnum, Set[str]] = defaultdict(set) # Correctly typed
        
        self.motif_library: Dict[str, Motif] = {} # Correctly typed
        self.entity_motifs: Dict[str, Set[str]] = defaultdict(set) 
        
        self.event_history: List[Event] = [] # Correctly typed
        self.recent_events: deque[Event] = deque(maxlen=100) # Correctly typed
        
        self.tick_count: int = 0
        self.current_time: float = 0.0 # Added per task
        self.time_scale: float = 1.0
        self.breath_cycle_length: int = 12 
        self.breath_phase: BreathPhaseEnum = BreathPhaseEnum.INHALE # Correctly typed
        self.breath_progress: float = 0.0
        
        self.inhale_ratio: float = 0.3
        self.hold_in_ratio: float = 0.2
        self.exhale_ratio: float = 0.3
        self.hold_out_ratio: float = 0.2
        
        self.simulation_history: Dict[str, Any] = { # Renamed and sub-keys renamed
            "creation_time": datetime.now(),
            "tick_history_log": [], 
            "significant_events_log": [] 
        }
        
        self.motif_feedback_queue: deque[Motif] = deque(maxlen=50) # Typed to hold Motif objects
        
        self.cosmic_scroll_repository: CosmicScroll = CosmicScroll() # Correctly typed
        self.active_processes: Dict[str, MetabolicProcess] = {} # Correctly typed
        
        logger.info("CosmicScrollManager initialized")
    
    def tick(self, delta_time: float = 1.0) -> Dict[str, Any]:
        """
        Advance the simulation forward one step.
        """
        actual_delta_time = delta_time * self.time_scale
        self.tick_count += 1
        self.current_time += actual_delta_time
        
        collected_events: List[Event] = []

        # Update breath cycle and collect potential event
        breath_event = self._update_breath_cycle(actual_delta_time)
        if breath_event:
            collected_events.append(breath_event)
        
        # Process metabolic processes
        process_results = self._process_metabolic_processes(actual_delta_time)
        collected_events.extend(process_results.get("events", []))
        
        # Generate spontaneous events
        spontaneous_events = self._generate_spontaneous_events()
        collected_events.extend(spontaneous_events)
        
        # Update motifs (currently doesn't generate events directly, but could)
        self._update_motifs()

        # Evolve all entities
        for entity_id in list(self.entities.keys()): 
            entity = self.entities.get(entity_id)
            if entity:
                try:
                    entity_events = entity.evolve({
                        "delta_time": actual_delta_time, 
                        "tick_number": self.tick_count, 
                        "manager_ref": self
                    })
                    if entity_events: # Ensure it's not None
                        collected_events.extend(entity_events)
                except Exception as e:
                    logger.error(f"Error evolving entity {entity.id} ({entity.name}): {e}", exc_info=True)


        # Log all collected events from this tick
        for event_obj in collected_events:
            self.log_event_object(event_obj)

        tick_data_summary = {
            "tick_number": self.tick_count,
            "time_elapsed_this_tick": actual_delta_time,
            "current_simulation_time": self.current_time,
            "breath_phase": self.breath_phase.name, # .name for Enum member string
            "breath_progress": self.breath_progress,
            "events_this_tick_count": len(collected_events),
            "active_processes_count": len(self.active_processes),
            "symbolic_density": self.cosmic_scroll_repository.calculate_symbolic_density()
        }
        
        self.simulation_history["tick_history_log"].append(tick_data_summary)
        if len(self.simulation_history["tick_history_log"]) > 1000:
             self.simulation_history["tick_history_log"].pop(0)
        
        return tick_data_summary
    
    def _update_breath_cycle(self, delta_time: float) -> Optional[Event]:
        """Updates the breath cycle and returns an Event if the phase changed."""
        if self.breath_cycle_length <= 0: return None # Avoid division by zero
        
        cycle_progress_increment = delta_time / self.breath_cycle_length
        self.breath_progress += cycle_progress_increment
        
        previous_phase = self.breath_phase
        phase_changed = False

        if self.breath_phase == BreathPhaseEnum.INHALE and self.breath_progress >= self.inhale_ratio:
            self.breath_phase = BreathPhaseEnum.HOLD_IN
            self.breath_progress -= self.inhale_ratio # Carry over excess progress
            phase_changed = True
        elif self.breath_phase == BreathPhaseEnum.HOLD_IN and self.breath_progress >= self.hold_in_ratio:
            self.breath_phase = BreathPhaseEnum.EXHALE
            self.breath_progress -= self.hold_in_ratio
            phase_changed = True
        elif self.breath_phase == BreathPhaseEnum.EXHALE and self.breath_progress >= self.exhale_ratio:
            self.breath_phase = BreathPhaseEnum.HOLD_OUT
            self.breath_progress -= self.exhale_ratio
            phase_changed = True
        elif self.breath_phase == BreathPhaseEnum.HOLD_OUT and self.breath_progress >= self.hold_out_ratio:
            self.breath_phase = BreathPhaseEnum.INHALE
            self.breath_progress -= self.hold_out_ratio
            phase_changed = True
        
        if phase_changed:
            logger.info(f"Breath phase changed from {previous_phase.name} to {self.breath_phase.name} at tick {self.tick_count}")
            return Event(
                event_type=EventTypeEnum.ENVIRONMENTAL_EVENT, 
                description=f"Breath phase transitioned from {previous_phase.name} to {self.breath_phase.name}.",
                involved_entity_ids=[], # Or perhaps a global "world" entity ID
                simulation_tick=self.tick_count,
                importance=0.2 # Low importance, routine event
            )
        return None

    def _process_metabolic_processes(self, delta_time: float) -> Dict[str, Any]:
        """Processes active metabolic processes and returns a summary including generated events."""
        results_summary: Dict[str, Any] = {"total_energy_consumed": 0.0, "total_outputs": defaultdict(float), "events": []}
        
        for process_id in list(self.active_processes.keys()):
            process = self.active_processes.get(process_id)
            if process and process.is_active:
                step_result = process.execute_step(delta_time, lambda eid: self.entities.get(eid))
                
                results_summary["total_energy_consumed"] += step_result.get("energy_consumed", 0.0)
                for res, amount in step_result.get("outputs", {}).items():
                    results_summary["total_outputs"][res] += amount # Ensure res is appropriate type if dict is typed
                
                status = step_result.get("status")
                if status == "completed" or status == "inactive":
                    if process_id in self.active_processes: # Check if not already removed
                        del self.active_processes[process_id]
                    event_desc = f"Metabolic process {process.name} ({process_id}) {status}."
                    event = Event(
                        event_type=EventTypeEnum.METABOLIC_EVENT, 
                        description=event_desc, 
                        involved_entity_ids=process.target_entity_ids, 
                        simulation_tick=self.tick_count,
                        properties={"process_id": process.id, "final_status": status}
                    )
                    results_summary["events"].append(event)
            elif not process:
                 logger.warning(f"Process {process_id} not found during metabolic processing (potentially removed mid-tick).")
        return results_summary

    def _generate_spontaneous_events(self) -> List[Event]:
        events: List[Event] = []
        if random.random() < 0.05: # Chance for a spontaneous event
            if not self.entities: 
                return events

            event_type_enum = random.choice(list(EventTypeEnum))
            num_involved = random.randint(1, min(2, len(self.entities))) # 1 or 2 entities
            involved_entity_ids = random.sample(list(self.entities.keys()), num_involved)
            
            event = Event(
                event_type=event_type_enum,
                description=f"A spontaneous {event_type_enum.name.lower()} occurrence was observed.",
                involved_entity_ids=involved_entity_ids,
                simulation_tick=self.tick_count,
                importance=random.uniform(0.1, 0.3) # Generally low importance
            )
            events.append(event)
        return events
    
    def _update_motifs(self) -> Dict[str, float]:
        motif_resonances: Dict[str, float] = {}
        # Example: global resonance field influenced by breath cycle progress
        global_field_resonance = 0.3 + (self.breath_progress * 0.4) 
        if self.breath_phase in [BreathPhaseEnum.INHALE, BreathPhaseEnum.HOLD_IN]:
            global_field_resonance += 0.15
        else:
            global_field_resonance -= 0.1
        global_field_resonance = max(0.1, min(0.9, global_field_resonance)) # Clamp

        for motif_id, motif_obj in self.motif_library.items():
            if isinstance(motif_obj, Motif): # Ensure it's a Motif object
                # activity_bonus could be based on recent events involving this motif
                activity_bonus = 0.0
                if self.recent_events:
                    for recent_event in self.recent_events:
                        if motif_id in recent_event.triggered_motifs:
                            activity_bonus += recent_event.importance * 0.05 
                activity_bonus = min(0.2, activity_bonus) # Cap bonus

                motif_obj.update_resonance(global_field_resonance, activity_bonus)
                motif_resonances[motif_id] = motif_obj.current_resonance
            else:
                logger.warning(f"Item with ID {motif_id} in motif_library is not a Motif instance: {type(motif_obj)}")
        return motif_resonances

    def create_entity(self, name: str, entity_type: EntityTypeEnum, 
                      properties: Optional[Dict[str, Any]] = None, 
                      initial_motifs_data: Optional[List[Dict[str, Any]]] = None, # Clearer type hint
                      initial_energy: float = 100.0) -> Entity: # Return type is Entity from mind_seed
        
        actual_initial_motifs: List[Motif] = []
        if initial_motifs_data:
            for m_data in initial_motifs_data:
                category_str = m_data.get("category", "PRIMORDIAL").upper()
                try:
                    cat_enum = MotifCategoryEnum[category_str]
                except KeyError:
                    logger.warning(f"Invalid motif category '{category_str}' for entity '{name}'. Defaulting to PRIMORDIAL.")
                    cat_enum = MotifCategoryEnum.PRIMORDIAL
                
                motif = Motif(
                    name=m_data.get("name", "Unnamed Motif"), 
                    category=cat_enum, 
                    attributes=m_data.get("attributes", {}),
                    description=m_data.get("description", ""),
                    intensity=float(m_data.get("intensity", 0.5)), # Ensure float
                    base_resonance=float(m_data.get("base_resonance", 0.5)), # Ensure float
                    creation_tick=self.tick_count
                )
                if motif.id not in self.motif_library: # Add if not already present (e.g. pre-defined motifs)
                    self.motif_library[motif.id] = motif
                actual_initial_motifs.append(motif)

        entity = CosmicEntity( # Using CosmicEntity from mind_seed now
            name=name, 
            entity_type=entity_type, 
            initial_properties=properties, 
            initial_motifs=actual_initial_motifs, 
            creation_tick=self.tick_count,
            initial_energy=initial_energy
        )
        entity.manager_ref = self 
        
        self.entities[entity.id] = entity
        self.entity_types[entity_type].add(entity.id)
        
        DRM.register_entity(entity) # Use imported DRM instance

        logger.info(f"Entity '{name}' (ID: {entity.id}, Type: {entity_type.name}) created.")
        
        creation_event = Event(
            event_type=EventTypeEnum.CREATION,
            description=f"Entity '{entity.name}' ({entity.id}) of type {entity_type.name} manifested in reality.",
            involved_entity_ids=[entity.id],
            simulation_tick=self.tick_count,
            importance=0.8 
        )
        self.log_event_object(creation_event)
        return entity

    def create_motif(self, name: str, category: MotifCategoryEnum, 
                     attributes: Dict[str, Any], description: str = "",
                     intensity: float = 0.5, base_resonance: float = 0.5) -> Motif: # Added intensity, base_resonance
        motif = Motif(
            name=name, category=category, attributes=attributes, description=description,
            intensity=intensity, base_resonance=base_resonance, creation_tick=self.tick_count
        )
        if motif.id in self.motif_library:
            logger.warning(f"Motif with ID {motif.id} (Name: {name}) already exists. Returning existing.")
            return self.motif_library[motif.id]
            
        self.motif_library[motif.id] = motif
        logger.info(f"Motif '{name}' (ID: {motif.id}, Category: {category.name}) created.")
        
        motif_event = Event(
            event_type=EventTypeEnum.MANIFESTATION,
            description=f"Symbolic Motif '{name}' ({category.name}) gained distinct form.",
            involved_entity_ids=[], 
            triggered_motifs=[motif], 
            simulation_tick=self.tick_count,
            importance=0.5 # Motifs manifesting is moderately important
        )
        self.log_event_object(motif_event)
        return motif

    def create_metabolic_process(self, name: str, process_type: MetabolicProcessTypeEnum, 
                                 target_entity_ids: List[str], 
                                 rate: float = 1.0, efficiency: float = 0.7, 
                                 duration: Optional[float] = None, 
                                 byproducts: Optional[Dict[MetabolicResourceEnum, float]] = None) -> MetabolicProcess:
        
        # Validate target_entity_ids
        valid_target_ids = [eid for eid in target_entity_ids if eid in self.entities]
        if not valid_target_ids:
            logger.error(f"Cannot create metabolic process '{name}': No valid target entities provided or found.")
            # Consider raising an error or returning a specific indicator
            raise ValueError("No valid target entities for metabolic process.")


        process = MetabolicProcess(
            name=name, process_type=process_type, target_entity_ids=valid_target_ids, 
            rate=rate, efficiency=efficiency, duration=duration, start_tick=self.tick_count
        )
        if byproducts:
            process.set_byproducts(byproducts)
        
        self.active_processes[process.id] = process
        logger.info(f"Metabolic process '{name}' (ID: {process.id}) initiated for entities: {valid_target_ids}.")
        
        proc_init_event = Event(
            event_type=EventTypeEnum.METABOLIC_EVENT, 
            description=f"Metabolic process '{name}' (type: {process_type.name}) has begun for {len(valid_target_ids)} entities.",
            involved_entity_ids=valid_target_ids,
            simulation_tick=self.tick_count,
            properties={"process_id": process.id, "process_name": name, "rate": rate, "efficiency": efficiency},
            importance=0.6 # Process initiation is noteworthy
        )
        self.log_event_object(proc_init_event)
        return process

    def log_event_object(self, event: Event):
        """Logs an already created Event object."""
        if not isinstance(event, Event): # Basic type check
            logger.error(f"Invalid object passed to log_event_object: {type(event)}. Expected Event instance.")
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

# === Integrated Environmental Scroll Modules ===
# Source: scroll_modules.py
# This module contains the integrated environmental scroll modules for the Cosmic Scroll system.
# === World & Environmental Systems ===

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
             class RecursiveMetabolism:
                 def __init__(self, base_efficiency: float = 0.7):
           self.base_efficiency = max(0.1, min(0.95, base_efficiency))
        primary_process = primary_process if primary_process is not None else self._select_default_process()
                 # Ensure this is inside a method or constructor of the class
            self.primary_process = primary_process
efficiency = max(0.1, min(0.95, base_efficiency))
primary_process = primary_process if primary_process is not None else self._select_default_process()
self.# Ensure this line is inside a method or constructor of the class
self.# Ensure this line is inside a method or constructor of the class
self.primary_process = primary_process
secondary_processes = secondary_processes or [self._select_default_process()]
self.recursion_depth = max(1, min(5, recursion_depth))
self.symbolic_affinity = symbolic_affinity or self._generate_symbolic_affinity()

# Subprocesses at different scales (molecular, cellular, organ, organism, ecosystem)
self.subprocesses = self._initialize_subprocesses()

# Byproducts and wastes generated by metabolism
self.byproducts = {}
for process in [self.primary_process] + secondary_processes:
    self.byproducts.update(self._generate_subprocess_byproducts(process))

# Energy storage in different forms
self.def __init__(self, base_efficiency: float = 0.7):
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
             def __init__(self, base_efficiency: float = 0.7):
            self.current_efficiency = base_efficiency
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
            MetabolicProcess.SYMBOLIC_ABSORPTION: ["meaning_digestion", "narrative_weaving", "symbolic_transformation"],
            MetabolicProcess.MOTIF_CYCLING: ["pattern_recycling", "thematic_evolution", "motif_harmonization"],
            MetabolicProcess.HARMONIC_RESONANCE: ["resonance_harvest", "harmonic_synthesis", "attunement_cycle"]
        }
        if self.primary_process in process_motifs:
            motifs.extend(process_motifs[self.primary_process])
            


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
            adjusted_inputs, time_delta
        )
        level_results["physical_energy"] += primary_result["energy"]
        level_results["symbolic_energy"] += primary_result["symbolic_energy"]
        level_results["byproducts"].update(primary_result["byproducts"])
        level_results["adaptation_events"].extend(primary_result["adaptation_events"])

        return level_results

class Storm:
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
            "tick_count": self.tick_count,
            "current_time": self.current_time,
            "entity_count": len(self.entities),
            "motif_count": len(self.motif_library),
            "event_count": len(self.event_history),
            "breath_phase": self.breath_phase.name,
            "breath_progress": self.breath_progress,
            "symbolic_density": self.cosmic_scroll_repository.calculate_symbolic_density(),
            "active_processes_count": len(self.active_processes)
        }

# Make CosmicScrollManager a singleton accessible via an instance
cosmic_scroll_manager_instance = CosmicScrollManager()

# Any other classes that were in cosmic_scroll.py and were NOT moved to mind_seed.py
# would be defined here. For example, WorldState and its related environmental classes,
# if they are not considered "core data structures" but rather part of this specific module's simulation logic.
# The problem description implies these environmental classes might remain or be part of cosmic_scroll.py's domain.

# Example: If WorldState and its components are here:
# from mind_seed import WORLD_SIZE # If it were moved
# class WorldState: ... uses WORLD_SIZE ...
# class Storm: ...
# class SeasonalCycle: ...
# etc.

# For now, I'll assume these environmental classes are either defined below,
# or this file primarily orchestrates the core types from mind_seed.
# The initial analysis showed WorldState and its components in cosmic_scroll.py, so they would remain here,
# using the Enums and base classes from mind_seed where appropriate.

# Placeholder for Environmental Systems if they remain in this file
# These would now use Enums from mind_seed.py
# class Storm:
#     def __init__(self, center: Tuple[float, float], storm_type: str, radius: float, intensity: float, symbolic_content: Dict[str, float]):
#         self.center = center
#         self.storm_type = storm_type # Could be an Enum if defined, e.g. StormTypeEnum
#         # ...
# class SeasonalCycle:
#      def __init__(self, starting_season: str = "spring", cycle_length: int = 120):
#          self.current_season = starting_season # could use a SeasonEnum
#         # ...

# All other classes like DimensionalRealityManager, MotifSeeder, ScrollMemory, CultureEngine, DiplomaticRegistry
# if they were defined in cosmic_scroll.py and are not just orchestrators of CosmicEntity,
# they would also be here, refactored to use mind_seed types.
# However, the prompt implies mind_seed.py is for "core data structure definitions".
# So, classes like CosmicScrollManager are more of "manager/orchestrator" classes.
# DRM was moved to mind_seed.py as it's a core data structure manager.

logger.info("cosmic_scroll.py refactored to use definitions from mind_seed.py.")
