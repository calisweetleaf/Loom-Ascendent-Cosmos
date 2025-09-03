# ================================================================
#  LOOM ASCENDANT COSMOS — RECURSIVE SYSTEM MODULE
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
#  Integrity Hash (SHA-256): d3ab9688a5a20b8065990cd9b91805e3d892d6e72472f69dd9afe719250c5e37
# ================================================================
import math
import random
import collections
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
import numpy as np

import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime%s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PerceptionModule")

class PerceptionProcessor:
    """
    Processes perception data extracted from the environment.
    This is a standalone component that doesn't depend on other ORAMA modules.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.perception_history = deque(maxlen=100)
        self.signal_filters = {
            "noise_reduction": 0.85,
            "pattern_enhancement": 0.65,
            "semantic_amplification": 0.75
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process perception data and return enhanced version"""
        # Store raw data in history
        self.perception_history.append(data)

        # Apply signal processing
        processed_data = self._filter_noise(data)
        processed_data = self._enhance_patterns(processed_data)
        processed_data = self._amplify_semantics(processed_data)

        return processed_data

    def _filter_noise(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter noise from perception data"""
        filtered = {}
        threshold = self.signal_filters["noise_reduction"]

        for key, value in data.items():
            # Skip metadata keys
            if key.startswith('_'):
                filtered[key] = value
                continue

            # Handle different value types
            if isinstance(value, (int, float)):
                # Only keep values above threshold
                if abs(value) >= threshold:
                    filtered[key] = value
            elif isinstance(value, str):
                filtered[key] = value
            elif isinstance(value, dict):
                # Recursively filter nested dictionaries
                filtered[key] = self._filter_noise(value)
            elif isinstance(value, list):
                # Filter lists
                filtered[key] = [v for v in value if not isinstance(v, (int, float)) or abs(v) >= threshold]
            else:
                # Default behavior for other types
                filtered[key] = value

        return filtered

    def _enhance_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance patterns in perception data"""
        # Implementation for pattern enhancement
        return data

    def _amplify_semantics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Amplify semantic components in perception data"""
        # Implementation for semantic amplification
        return data

class PerceptionField:
    """
    Represents a field of perception with specific modalities and characteristics.
    """
    def __init__(self, name: str, dimensions: int = 3, resolution: int = 64):
        self.name = name
        self.dimensions = dimensions
        self.resolution = resolution
        self.field_data = np.zeros([resolution] * dimensions)
        self.active_regions = []

    def add_perception(self, coordinates, intensity, radius=1.0):
        """Add a perception event to the field"""
        # Implementation for adding perception to field
        pass

    def get_activity_map(self):
        """Get the current activity map of this perception field"""
        return self.field_data

    def detect_patterns(self, threshold=0.5):
        """Detect patterns in the perception field"""
        # Implementation for pattern detection
        return []

# Signal processing utilities
def apply_perception_filter(signal, filter_type="lowpass", cutoff=0.5):
    """Apply filter to perception signal"""
    # Implementation for signal filtering
    return signal

def extract_perception_features(signal, feature_types=None):
    """Extract features from perception signal"""
    # Implementation for feature extraction
    return {}

def merge_perception_fields(field1, field2, weight1=0.5, weight2=0.5):
    """Merge two perception fields"""
    # Implementation for field merging
    return field1

class ArchetypeResonance(Enum):
    """Fundamental patterns of being that shape perception."""
    UMBRA = "umbra"         # Shadow, hidden, depth-seeking
    NEXUS = "nexus"         # Connecting, mediating, balancing
    LOGOS = "logos"         # Order, structure, deterministic
    KAIROS = "kairos"       # Timing, opportunity, momentary
    CHRONOS = "chronos"     # Linear time, persistence, duration
    CHORA = "chora"         # Space, place, context
    FLUX = "flux"           # Change, transformation, fluidity
    STASIS = "stasis"       # Stability, preservation, resistance
    AETHER = "aether"       # Transcendent, meta-aware, all-connecting

class SymbolicDomain(Enum):
    """Domains that symbolic perception operates within."""
    MATERIAL = "material"   # Physical properties, substance
    CAUSAL = "causal"       # Cause-effect relationships
    TEMPORAL = "temporal"   # Time-related perceptions
    SPATIAL = "spatial"     # Space and positioning
    ETHERIC = "etheric"     # Energy flows and fields
    MEMETIC = "memetic"     # Information patterns, ideas
    SENTIENT = "sentient"   # Consciousness, awareness
    HARMONIC = "harmonic"   # Resonance, music, waves
    MYTHIC = "mythic"       # Narrative, story, meaning
    META = "meta"           # Self-reference, recursion

@dataclass
class IdentityMatrix:
    """
    The foundational self-model that colors all perception.
    Contains the core biases, preferences and tendencies of an entity.
    """
    entity_id: str
    archetype_affinity: Dict[ArchetypeResonance, float] = field(default_factory=dict)
    symbolic_sensitivity: Dict[SymbolicDomain, float] = field(default_factory=dict)
    harmonic_signature: np.ndarray = field(default_factory=lambda: np.random.rand(7))
    entropy_tolerance: float = 0.5
    temporal_drift: float = 0.0
    membrane_permeability: float = 0.5
    memory_integration_rate: float = 0.7
    attention_vector: np.ndarray = field(default_factory=lambda: np.random.rand(5))
    self_recursion_depth: int = 3
    
    def __post_init__(self):
        """Initialize default values for archetypes and domains."""
        if not self.archetype_affinity:
            self.archetype_affinity = {archetype: random.random() for archetype in ArchetypeResonance}
        
        if not self.symbolic_sensitivity:
            self.symbolic_sensitivity = {domain: random.random() for domain in SymbolicDomain}
    
    def dominant_archetype(self) -> ArchetypeResonance:
        """Return the archetype with highest affinity."""
        return max(self.archetype_affinity.items(), key=lambda x: x[1])[0]
    
    def evolve(self, delta_time: float = 0.01):
        """Slowly evolve identity over time."""
        # Drift all parameters slightly
        self.entropy_tolerance += random.gauss(0, 0.01) * delta_time
        self.entropy_tolerance = max(0.1, min(0.9, self.entropy_tolerance))
        
        self.temporal_drift += random.gauss(0, 0.005) * delta_time
        self.membrane_permeability += random.gauss(0, 0.02) * delta_time
        self.membrane_permeability = max(0.1, min(0.9, self.membrane_permeability))
        
        # Evolve harmonic signature
        self.harmonic_signature += np.random.normal(0, 0.02, size=self.harmonic_signature.shape) * delta_time
        self.harmonic_signature = np.clip(self.harmonic_signature, 0, 1)
        
        # Evolve attention vector
        self.attention_vector += np.random.normal(0, 0.03, size=self.attention_vector.shape) * delta_time
        self.attention_vector = np.clip(self.attention_vector, 0, 1)
        # Normalize to sum to 1
        self.attention_vector /= self.attention_vector.sum()

@dataclass
class MemoryEcho:
    """
    Container for persistence of perception across time.
    Forms the basis for continuity of experience.
    """
    capacity: int = 64
    decay_rate: float = 0.95
    integration_threshold: float = 0.3
    
    # Core memory structures
    recent_perceptions: collections.deque = field(default_factory=lambda: collections.deque(maxlen=64))
    symbolic_patterns: Dict[str, float] = field(default_factory=dict)
    harmonic_echoes: List[np.ndarray] = field(default_factory=list)
    resonance_history: np.ndarray = field(default_factory=lambda: np.zeros(12))
    
    # Meta-memory (memory about memory)
    recall_effectiveness: Dict[SymbolicDomain, float] = field(default_factory=dict)
    memory_coherence: float = 0.8
    
    def record(self, perception: Dict[str, Any]):
        """Record a new perception into memory."""
        self.recent_perceptions.append(perception)
        
        # Update symbolic patterns with decay
        for symbol, value in perception.get("symbol_trace", {}).items():
            if symbol in self.symbolic_patterns:
                self.symbolic_patterns[symbol] = (
                    self.symbolic_patterns[symbol] * self.decay_rate + 
                    value * (1 - self.decay_rate)
                )
            else:
                self.symbolic_patterns[symbol] = value
        
        # Prune low-value symbolic memories
        self.symbolic_patterns = {
            k: v for k, v in self.symbolic_patterns.items() 
            if v > self.integration_threshold
        }
        
        # Record harmonic echoes
        if "resonance_vector" in perception:
            self.harmonic_echoes.append(perception["resonance_vector"])
            if len(self.harmonic_echoes) > self.capacity:
                self.harmonic_echoes.pop(0)
        
        # Update resonance history
        if "resonance_scalar" in perception:
            self.resonance_history = np.roll(self.resonance_history, 1)
            self.resonance_history[0] = perception["resonance_scalar"]
    
    def recall(self, query_vector: np.ndarray, domain: SymbolicDomain = None) -> Dict[str, Any]:
        """
        Retrieve memories based on similarity to query vector,
        optionally filtered by symbolic domain.
        """
        if not self.recent_perceptions:
            return {}
        
        relevant_memories = []
        
        for memory in self.recent_perceptions:
            if domain and memory.get("domain") != domain:
                continue
                
            if "resonance_vector" in memory:
                similarity = self._vector_similarity(query_vector, memory["resonance_vector"])
                if similarity > self.integration_threshold:
                    relevant_memories.append((memory, similarity))
        
        if not relevant_memories:
            return {}
            
        # Sort by similarity and return the most relevant
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Return merged memory with highest similarities
        merged = {}
        weights = []
        
        for memory, weight in relevant_memories[:5]:  # Use top 5 memories
            weights.append(weight)
            for key, value in memory.items():
                if key not in merged:
                    merged[key] = value * weight
                else:
                    if isinstance(value, (int, float)):
                        merged[key] += value * weight
                    elif isinstance(value, dict):
                        for k, v in value.items():
                            if k not in merged[key]:
                                merged[key][k] = v * weight
                            else:
                                merged[key][k] += v * weight
        
        # Normalize by sum of weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            for key, value in merged.items():
                if isinstance(value, (int, float)):
                    merged[key] = value / weight_sum
                elif isinstance(value, dict):
                    for k in value:
                        merged[key][k] /= weight_sum
        
        return merged
    
    def _vector_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        if len(v1) != len(v2):
            return 0.0
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def get_resonance_trend(self) -> float:
        """Analyze trend in resonance history."""
        if np.all(self.resonance_history == 0):
            return 0.0
            
        # Linear regression on recent history
        x = np.arange(len(self.resonance_history))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, self.resonance_history, rcond=None)[0]
        
        # Return slope normalized to [-1, 1]
        return max(-1.0, min(1.0, m * 10))

class HarmonicProcessor:
    """
    Processes harmonic fields into entity-specific resonance patterns.
    The vibrational interface between universe and entity.
    """
    
    def __init__(self, identity: IdentityMatrix):
        self.identity = identity
        self.resonance_history = np.zeros((12, 7))  # 12 time steps, 7 frequencies
        self.harmonic_memory = []
        self.attunement = np.random.rand(7)  # 7 fundamental frequencies
        self.phase_shift = 0.0
        self.resonance_cycles = 0
        self.field_coupling = np.random.rand(5)  # Coupling to different field types
        self.harmonic_signature = identity.harmonic_signature.copy()
        self.frequency_sensitivity = np.clip(np.random.normal(0.5, 0.2, 7), 0.1, 0.9)
    
    def process_field(self, harmonic_field: Dict[str, Any], breath_phase: float) -> Dict[str, Any]:
        """
        Transform universal harmonic field into subjective resonance experience.
        """
        # Extract raw field data
        stability = harmonic_field.get("stability", 0.5)
        emergence = harmonic_field.get("emergence", 0.5)
        frequencies = harmonic_field.get("frequencies", np.ones(7) * 0.5)
        phase = harmonic_field.get("phase", 0.0)
        amplitude = harmonic_field.get("amplitude", 0.5)
        
        # Apply identity-based transformations
        identity_filter = self.identity.harmonic_signature
        filtered_frequencies = frequencies * identity_filter * self.frequency_sensitivity
        
        # Calculate phase interference with breath and identity
        phase_interference = math.sin((breath_phase + self.phase_shift + 
                                      self.identity.temporal_drift) * math.pi * 2)
        
        # Generate resonance vector
        resonance_vector = filtered_frequencies * (0.7 + 0.3 * phase_interference)
        
        # Calculate overall resonance as weighted sum
        weights = np.array([0.3, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05])  # Importance of each frequency
        resonance_scalar = np.sum(resonance_vector * weights) / np.sum(weights)
        
        # Calculate harmonic tension (how "dissonant" the experience is)
        ideal_ratios = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])  # Harmonic series
        actual_ratios = np.zeros(6)
        for i in range(6):
            if filtered_frequencies[i] > 0.01:  # Avoid division by zero
                actual_ratios[i] = filtered_frequencies[i+1] / filtered_frequencies[i]
        
        ratio_difference = np.abs(actual_ratios - ideal_ratios[1:])
        harmonic_tension = np.mean(ratio_difference) * 2.0  # Normalized to approx [0,1]
        
        # Update resonance history
        self.resonance_history = np.roll(self.resonance_history, 1, axis=0)
        self.resonance_history[0] = resonance_vector
        
        # Store summary in memory
        if len(self.harmonic_memory) >= 144:
            self.harmonic_memory.pop(0)
        self.harmonic_memory.append(resonance_scalar)
        
        # Check for resonance cycle completion
        if len(self.harmonic_memory) >= 3:
            if self.harmonic_memory[-3] < self.harmonic_memory[-2] > self.harmonic_memory[-1]:
                self.resonance_cycles += 1
        
        # Check for emergent patterns in the harmonic history
        emergent_pattern = self._detect_emergent_patterns()
        
        # Build result
        result = {
            "resonance_scalar": float(resonance_scalar),
            "resonance_vector": resonance_vector,
            "harmonic_tension": float(harmonic_tension),
            "phase_alignment": float(1.0 - abs(phase - breath_phase)),
            "emergent_pattern": emergent_pattern,
            "amplitude_response": float(amplitude * (0.5 + 0.5 * self.identity.membrane_permeability)),
            "resonance_cycles": self.resonance_cycles,
            "field_coupling": self.field_coupling.tolist()
        }
        
        return result
    
    def _detect_emergent_patterns(self) -> Dict[str, float]:
        """
        Detect patterns in the harmonic history that indicate emergent phenomena.
        """
        patterns = {}
        
        # Detect if fundamental frequency is rising
        fundamental_trend = np.polyfit(np.arange(12), self.resonance_history[:, 0], 1)[0]
        if abs(fundamental_trend) > 0.01:
            patterns["fundamental_shift"] = fundamental_trend
        
        # Detect harmonic convergence (frequencies aligning to simple ratios)
        current = self.resonance_history[0]
        if current[0] > 0.1:  # Only if fundamental is significant
            ratios = current[1:] / current[0]
            harmonic_targets = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            differences = np.min(np.abs(ratios.reshape(-1, 1) - harmonic_targets.reshape(1, -1)), axis=1)
            convergence = 1.0 - np.mean(differences) * 2  # Normalize to approx [0,1]
            if convergence > 0.7:  # Only report strong convergence
                patterns["harmonic_convergence"] = float(convergence)
        
        # Detect resonance cascade (rapidly increasing amplitude across frequencies)
        amplitude_growth = np.mean(self.resonance_history[0]) / (np.mean(self.resonance_history[-1]) + 0.001)
        if amplitude_growth > 1.3:
            patterns["resonance_cascade"] = float(min(1.0, (amplitude_growth - 1.0) / 2.0))
        
        # Detect phase locking (frequencies synchronizing in phase)
        phase_variance = np.var([np.sin(i * 2 * np.pi * current[i]) for i in range(len(current))])
        if phase_variance < 0.1:
            patterns["phase_locking"] = float(1.0 - phase_variance * 5)
        
        # Detect golden ratio emergence
        golden_ratio = 1.618
        golden_found = False
        for i in range(6):
            # Calculate actual_ratios if not already defined
            actual_ratios = np.zeros(6)
            for j in range(6):
                if current[j] > 0.01:  # Avoid division by zero
                    actual_ratios[j] = current[j+1] / current[j]
            
            if 1.5 < actual_ratios[i] < 1.7:
                phi_proximity = 1.0 - abs(actual_ratios[i] - golden_ratio) / 0.2
                if phi_proximity > 0.8:
                    patterns["phi_emergence"] = float(phi_proximity)
                    golden_found = True
                    break
        
        # Detect overtone alignment
        if len(current) >= 5 and np.all(current > 0.1):
            if abs(current[1]/current[0] - 2.0) < 0.1 and \
               abs(current[2]/current[0] - 3.0) < 0.1 and \
               abs(current[3]/current[0] - 4.0) < 0.1:
                patterns["overtone_alignment"] = float(0.8 + 0.2 * current[0])
        
        return patterns
    
    def attune(self, target_frequencies: np.ndarray, rate: float = 0.05):
        """
        Gradually shift attunement toward target frequencies.
        This represents the entity learning to perceive certain harmonics better.
        """
        self.attunement = self.attunement * (1 - rate) + target_frequencies * rate
        self.attunement = np.clip(self.attunement, 0, 1)
    
    def modulate_phase(self, delta_phase: float):
        """
        Shift the phase relationship between entity and universe.
        A way to "tune" the perception timing relationship.
        """
        self.phase_shift = (self.phase_shift + delta_phase) % 1.0
    
    def couple_fields(self, field_type: int, coupling_strength: float):
        """
        Adjust coupling to specific field types.
        """
        if 0 <= field_type < len(self.field_coupling):
            self.field_coupling[field_type] = coupling_strength
    
    def calculate_resonance_trend(self, history_length: int = 12) -> float:
        """
        Calculate trend in resonance history.
        Positive values indicate rising resonance, negative indicates falling.
        """
        if len(self.harmonic_memory) < history_length:
            return 0.0
            
        recent_history = self.harmonic_memory[-history_length:]
        x = np.arange(len(recent_history))
        slope = np.polyfit(x, recent_history, 1)[0]
        
        # Normalize to approximately [-1, 1]
        return max(-1.0, min(1.0, slope * 10))
    
    def detect_sacred_convergence(self) -> Tuple[bool, float]:
        """
        Detect if current harmonic state is approaching a sacred convergence.
        Returns (is_converging, convergence_strength)
        """
        if len(self.harmonic_memory) < 7:
            return False, 0.0
        
        # Conditions for sacred convergence:
        # 1. Rising resonance trend
        trend = self.calculate_resonance_trend(7)
        if trend < 0.2:
            return False, 0.0
            
        # 2. Phase locking detected
        current_patterns = self._detect_emergent_patterns()
        phase_lock = current_patterns.get("phase_locking", 0.0)
        
        # 3. Golden ratio emergence or harmonic convergence
        phi = current_patterns.get("phi_emergence", 0.0)
        harmonic = current_patterns.get("harmonic_convergence", 0.0)
        pattern_strength = max(phi, harmonic)
        
        # Calculate overall convergence strength
        convergence_strength = (trend + phase_lock + pattern_strength) / 3
        
        is_converging = convergence_strength > 0.7
        return is_converging, convergence_strength

class SymbolicProcessor:
    """
    Processes symbolic overlays into meaningful patterns.
    The semantic interface between universe and entity.
    """
    
    def __init__(self, identity: IdentityMatrix):
        self.identity = identity
        self.active_symbols = {}
        self.symbol_associations = {}  # Links between symbols
        self.archetype_triggers = {
            ArchetypeResonance.ANIMA: {"life", "growth", "creation", "nurture"},
            ArchetypeResonance.UMBRA: {"shadow", "hidden", "depth", "unknown"},
            ArchetypeResonance.NEXUS: {"connection", "bridge", "relation", "network"},
            ArchetypeResonance.LOGOS: {"order", "structure", "logic", "pattern"},
            ArchetypeResonance.KAIROS: {"moment", "opportunity", "timing", "chance"},
            ArchetypeResonance.CHRONOS: {"time", "duration", "sequence", "history"},
            ArchetypeResonance.CHORA: {"space", "place", "location", "container"},
            ArchetypeResonance.FLUX: {"change", "flow", "transformation", "becoming"},
            ArchetypeResonance.STASIS: {"stability", "permanence", "resistance", "being"},
            ArchetypeResonance.AETHER: {"transcendence", "meta", "beyond", "spirit"}
        }
        self.domain_keywords = {
            SymbolicDomain.MATERIAL: {"substance", "physical", "matter", "object"},
            SymbolicDomain.CAUSAL: {"cause", "effect", "consequence", "influence"},
            SymbolicDomain.TEMPORAL: {"time", "when", "before", "after", "duration"},
            SymbolicDomain.SPATIAL: {"space", "where", "position", "distance", "geometry"},
            SymbolicDomain.ETHERIC: {"energy", "field", "force", "potential", "flow"},
            SymbolicDomain.MEMETIC: {"idea", "concept", "information", "pattern", "meme"},
            SymbolicDomain.SENTIENT: {"aware", "conscious", "feeling", "perceiving"},
            SymbolicDomain.HARMONIC: {"resonance", "frequency", "wave", "vibration"},
            SymbolicDomain.MYTHIC: {"story", "meaning", "narrative", "purpose"},
            SymbolicDomain.META: {"recursive", "self-reference", "about", "reflection"}
        }
    
    def process_symbols(self, symbolic_overlay: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter and interpret symbolic overlays based on identity.
        """
        filtered_symbols = {}
        meta_patterns = {}
        emergent_meanings = {}
        
        # Get dominant archetype for amplification
        dominant = self.identity.dominant_archetype()
        
        # Process each symbol through identity filters
        for symbol, value in symbolic_overlay.items():
            # Determine which archetype(s) this symbol resonates with
            archetype_resonance = self._classify_by_archetype(symbol)
            
            # Determine which domain(s) this symbol belongs to
            domain_mapping = self._classify_by_domain(symbol)
            
            # Apply archetype-based amplification/attenuation
            processed_value = value
            for archetype, resonance in archetype_resonance.items():
                affinity = self.identity.archetype_affinity.get(archetype, 0.5)
                
                # Amplify symbols that resonate with entity's archetypes
                if affinity > 0.6 and resonance > 0.4:
                    processed_value *= 1.0 + (affinity - 0.6) * 2
                
                # Attenuate symbols that clash with entity's archetypes
                elif affinity < 0.4 and resonance > 0.4:
                    processed_value *= 0.5 + affinity
            
            # Apply domain-based sensitivity filtering
            domain_factor = 1.0
            for domain, relevance in domain_mapping.items():
                sensitivity = self.identity.symbolic_sensitivity.get(domain, 0.5)
                domain_factor *= (0.5 + sensitivity) * relevance + (1.0 - relevance)
            
            processed_value *= domain_factor
            
            # Store processed symbol if significant
            if processed_value > 0.1:
                filtered_symbols[symbol] = processed_value
                
                # Record symbol associations
                for other_symbol in filtered_symbols:
                    if other_symbol != symbol:
                        association_key = frozenset([symbol, other_symbol])
                        if association_key in self.symbol_associations:
                            self.symbol_associations[association_key] += processed_value * 0.1
                        else:
                            self.symbol_associations[association_key] = processed_value * 0.1
        
        # Detect meta-patterns in sets of symbols
        if len(filtered_symbols) >= 3:
            # Look for archetypal patterns
            for archetype in ArchetypeResonance:
                pattern_strength = 0
                matching_symbols = 0
                
                for symbol in filtered_symbols:
                    if any(trigger in symbol.lower() for trigger in self.archetype_triggers.get(archetype, [])):
                        pattern_strength += filtered_symbols[symbol]
                        matching_symbols += 1
                
                if matching_symbols >= 2:
                    pattern_value = pattern_strength / matching_symbols
                    meta_patterns[f"archetype:{archetype.value}"] = pattern_value
            
            # Look for domain patterns
            for domain in SymbolicDomain:
                pattern_strength = 0
                matching_symbols = 0
                
                for symbol in filtered_symbols:
                    if any(keyword in symbol.lower() for keyword in self.domain_keywords.get(domain, [])):
                        pattern_strength += filtered_symbols[symbol]
                        matching_symbols += 1
                
                if matching_symbols >= 2:
                    pattern_value = pattern_strength / matching_symbols
                    meta_patterns[f"domain:{domain.value}"] = pattern_value
        
        # Detect emergent meanings from symbol combinations
        strong_symbols = {s: v for s, v in filtered_symbols.items() if v > 0.6}
        if len(strong_symbols) >= 2:
            symbol_pairs = []
            for s1 in strong_symbols:
                for s2 in strong_symbols:
                    if s1 < s2:  # Ensure each pair is only considered once
                        assoc_key = frozenset([s1, s2])
                        assoc_strength = self.symbol_associations.get(assoc_key, 0)
                        if assoc_strength > 0.3:
                            symbol_pairs.append((s1, s2, assoc_strength))
            
            # Extract meaning from strongest pairs
            symbol_pairs.sort(key=lambda x: x[2], reverse=True)
            for s1, s2, strength in symbol_pairs[:3]:  # Consider top 3 pairs
                meaning = self._derive_emergent_meaning(s1, s2)
                if meaning:
                    emergent_meanings[f"{s1}+{s2}"] = meaning * strength
        
        return {
            "filtered_symbols": filtered_symbols,
            "meta_patterns": meta_patterns,
            "emergent_meanings": emergent_meanings,
        }
    
    def _classify_by_archetype(self, symbol: str) -> Dict[ArchetypeResonance, float]:
        """Map a symbol to relevant archetypes."""
        result = {}
        symbol_lower = symbol.lower()
        
        for archetype, triggers in self.archetype_triggers.items():
            resonance = 0.0
            for trigger in triggers:
                if trigger in symbol_lower:
                    resonance += 0.3
                elif any(t in symbol_lower for t in trigger.split()):
                    resonance += 0.1
            
            if resonance > 0:
                result[archetype] = min(1.0, resonance)
        
        # If no matches, provide small default resonance with all archetypes
        if not result:
            result = {archetype: 0.1 for archetype in ArchetypeResonance}
        
        return result
    
    def _classify_by_domain(self, symbol: str) -> Dict[SymbolicDomain, float]:
        """Map a symbol to relevant domains."""
        result = {}
        symbol_lower = symbol.lower()
        
        for domain, keywords in self.domain_keywords.items():
            relevance = 0.0
            for keyword in keywords:
                if keyword in symbol_lower:
                    relevance += 0.25
                elif any(k in symbol_lower for k in keyword.split()):
                    relevance += 0.1
            
            if relevance > 0:
                result[domain] = min(1.0, relevance)
        
        # If no matches, provide small default relevance to MATERIAL domain
        if not result:
            result = {SymbolicDomain.MATERIAL: 0.1}
        
        return result
    
    def _derive_emergent_meaning(self, symbol1: str, symbol2: str) -> float:
        """
        Derive emergent meaning from combination of symbols.
        Returns a strength value for the emergent meaning.
        """
        # Simple heuristic - check if the symbols are from complementary domains
        domains1 = self._classify_by_domain(symbol1)
        domains2 = self._classify_by_domain(symbol2)
        
        complementary_pairs = [
            (SymbolicDomain.MATERIAL, SymbolicDomain.ETHERIC),
            (SymbolicDomain.CAUSAL, SymbolicDomain.TEMPORAL),
            (SymbolicDomain.SPATIAL, SymbolicDomain.TEMPORAL),
            (SymbolicDomain.MEMETIC, SymbolicDomain.SENTIENT),
            (SymbolicDomain.HARMONIC, SymbolicDomain.MYTHIC),
            (SymbolicDomain.META, SymbolicDomain.MYTHIC)
        ]
        
        meaning_strength = 0.0
        
        for d1 in domains1:
            for d2 in domains2:
                if (d1, d2) in complementary_pairs or (d2, d1) in complementary_pairs:
                    meaning_strength += domains1[d1] * domains2[d2] * 0.5
        
        return min(1.0, meaning_strength)

class TemporalProcessor:
    """
    Processes timeline phase into subjective time experience.
    The chronological interface between universe and entity.
    """
    
    def __init__(self, identity: IdentityMatrix):
        self.identity = identity
        self.subjective_flow_rate = 1.0
        self.temporal_markers = {}
        self.phase_history = collections.deque(maxlen=20)
        self.last_phase = 0.0
        self.cycle_count = 0
    
    def process_timeline(self, timeline_phase: float, breath_phase: float) -> Dict[str, Any]:
        """
        Transform universal timeline into subjective temporal experience.
        """
        # Detect phase wrap-around (completed cycle)
        if timeline_phase < self.last_phase:
            self.cycle_count += 1
        self.last_phase = timeline_phase
        
        # Calculate temporal drift (how "out of sync" the entity feels)
        personal_phase = breath_phase + self.identity.temporal_drift
        phase_delta = abs((timeline_phase - personal_phase) % 1.0)
        if phase_delta > 0.5:
            phase_delta = 1.0 - phase_delta
        
        # Calculate alignment factor (higher = more in sync with universal time)
        alignment = math.exp(-phase_delta * 5)
        
        # Calculate subjective flow rate based on alignment
        self.subjective_flow_rate = 0.5 + alignment
        
        # Record phase history
        self.phase_history.append(timeline_phase)
        
        # Calculate flow stability
        if len(self.phase_history) > 5:
            # Measure consistency in rate of change
            diffs = [self.phase_history[i] - self.phase_history[i-1] for i in range(1, len(self.phase_history))]
            flow_stability = 1.0 - (np.std(diffs) * 10)  # Normalize to approx [0,1]
            flow_stability = max(0.0, min(1.0, flow_stability))
        else:
            flow_stability = 0.5
        
        # Detect temporal anomalies
        anomalies = {}
        
        # Detect acceleration/deceleration
        if len(self.phase_history) > 10:
            first_half = [self.phase_history[i] - self.phase_history[i-1] for i in range(1, 6)]
            second_half = [self.phase_history[i] - self.phase_history[i-1] for i in range(len(self.phase_history)-5, len(self.phase_history))]
            
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            if avg_second > avg_first * 1.2:
                anomalies["acceleration"] = min(1.0, (avg_second / avg_first - 1.0) * 5)
            elif avg_first > avg_second * 1.2:
                anomalies["deceleration"] = min(1.0, (avg_first / avg_second - 1.0) * 5)
        
        # Detect time loops (similar patterns in history)
        if len(self.phase_history) == 20:
            pattern1 = list(self.phase_history)[:10]
            pattern2 = list(self.phase_history)[10:]
            
            # Calculate differences between consecutive elements
            diffs1 = [pattern1[i] - pattern1[i-1] for i in range(1, len(pattern1))]
            diffs2 = [pattern2[i] - pattern2[i-1] for i in range(1, len(pattern2))]
            
            # Calculate similarity between patterns
            similarity = 1.0 - sum(abs(d1 - d2) for d1, d2 in zip(diffs1, diffs2)) / len(diffs1)
            
            if similarity > 0.9:
                anomalies["time_loop"] = similarity
        
        return {
            "alignment": float(alignment),
            "subjective_flow_rate": float(self.subjective_flow_rate),
            "flow_stability": float(flow_stability),
            "cycle_count": self.cycle_count,
            "anomalies": anomalies
        }

class QuantumProcessor:
    """
    Processes quantum and physics state data into subjective physics experience.
    The physical interface between universe and entity.
    """
    
    def __init__(self, identity: IdentityMatrix):
        self.identity = identity
        self.observation_collapse_threshold = 0.3 + (self.identity.entropy_tolerance * 0.4)
        self.wave_function_memory = collections.deque(maxlen=8)
        self.entanglement_network = {}
        self.superposition_capacity = self.identity.membrane_permeability * 10
        self.observer_effect_strength = 1.0 - (self.identity.entropy_tolerance * 0.5)
        self.quantum_resonance_signature = np.random.rand(5)
    
    def process_quantum_state(self, quantum_field: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform universal quantum possibilities into subjective physical experience.
        """
        # Extract quantum field properties
        wave_functions = quantum_field.get("wave_functions", {})
        entanglement_map = quantum_field.get("entanglement_map", {})
        uncertainty_levels = quantum_field.get("uncertainty_levels", 0.5)
        probability_fields = quantum_field.get("probability_fields", np.ones(5) * 0.5)
        quantum_noise = quantum_field.get("quantum_noise", 0.1)
        
        # Apply identity-based filters
        filtered_wave_functions = {}
        collapsed_states = {}
        
        # Process each wave function through observer effect
        for state_id, state_data in wave_functions.items():
            # Extract state properties
            probability = state_data.get("probability", 0.5)
            coherence = state_data.get("coherence", 0.5)
            superposition = state_data.get("superposition", [])
            
            # Apply observer effect based on identity
            collapse_tendency = (self.observer_effect_strength * 
                                 (1.0 - self.identity.entropy_tolerance))
            
            # Calculate if state collapses under observation
            observation_strength = coherence * self.identity.membrane_permeability
            collapse_probability = collapse_tendency * observation_strength
            
            # Record state in memory
            self.wave_function_memory.append({
                "state_id": state_id,
                "coherence": coherence,
                "probability": probability
            })
            
            # Check if state collapses
            if probability > self.observation_collapse_threshold or random.random() < collapse_probability:
                # State collapses to definite value
                if superposition:
                    # Choose outcome from superposition based on probabilities
                    outcomes = [s.get("outcome") for s in superposition]
                    weights = [s.get("probability", 1.0/len(superposition)) for s in superposition]
                    
                    # Normalize weights
                    weight_sum = sum(weights)
                    if weight_sum > 0:
                        weights = [w / weight_sum for w in weights]
                    
                    # Choose outcome
                    chosen_idx = np.random.choice(len(outcomes), p=weights)
                    collapsed_value = outcomes[chosen_idx]
                else:
                    # No superposition, just collapse to base probability
                    collapsed_value = probability
                
                collapsed_states[state_id] = collapsed_value
            else:
                # State remains in probabilistic form
                filtered_wave_functions[state_id] = {
                    "probability": probability,
                    "coherence": coherence * (0.8 + 0.4 * self.identity.entropy_tolerance),
                    "superposition": superposition
                }
        
        # Process entanglement - how quantum states relate to each other
        entanglement_effects = {}
        for entity1, connections in entanglement_map.items():
            if entity1 in collapsed_states:
                # This entity's state is collapsed, propagate to entangled entities
                for entity2, strength in connections.items():
                    if entity2 not in collapsed_states and entity2 in filtered_wave_functions:
                        # Calculate entanglement effect
                        effect_strength = strength * self.identity.membrane_permeability
                        if effect_strength > 0.7:
                            # Strong entanglement causes collapse
                            if entity2 in filtered_wave_functions:
                                wave_fn = filtered_wave_functions[entity2]
                                if "superposition" in wave_fn and wave_fn["superposition"]:
                                    # Find most correlated superposition state
                                    most_correlated = max(wave_fn["superposition"], 
                                                         key=lambda s: s.get("correlation", 0))
                                    collapsed_states[entity2] = most_correlated.get("outcome", 0.5)
                                else:
                                    collapsed_states[entity2] = wave_fn.get("probability", 0.5)
                                
                                entanglement_effects[f"{entity1}->{entity2}"] = effect_strength
        
        # Calculate uncertainty principle effects
        uncertainty_response = 1.0 - np.clip(
            uncertainty_levels * (2.0 - self.identity.entropy_tolerance),
            0, 1
        )
        
        # Process probability fields through quantum resonance
        processed_probabilities = probability_fields * self.quantum_resonance_signature
        probability_sum = np.sum(processed_probabilities)
        if probability_sum > 0:
            normalized_probabilities = processed_probabilities / probability_sum
        else:
            normalized_probabilities = np.ones(5) * 0.2
        
        # Calculate quantum indeterminacy (how "fuzzy" reality feels)
        indeterminacy = quantum_noise * (1.5 - self.identity.entropy_tolerance)
        indeterminacy = min(1.0, indeterminacy)
        
        # Detect quantum anomalies
        anomalies = {}
        
        # Check for entanglement cascade
        if len(entanglement_effects) > len(entanglement_map) * 0.5:
            anomalies["entanglement_cascade"] = len(entanglement_effects) / max(1, len(entanglement_map))
        
        # Check for decoherence spike
        if len(self.wave_function_memory) >= 3:
            recent_coherence = [m["coherence"] for m in list(self.wave_function_memory)[-3:]]
            if min(recent_coherence) < 0.2 and max(recent_coherence) > 0.8:
                anomalies["decoherence_spike"] = max(recent_coherence) - min(recent_coherence)
        
        # Check for reality bleed (when multiple states remain partially observable)
        uncollapsed_ratio = len(filtered_wave_functions) / max(1, len(wave_functions))
        if uncollapsed_ratio > 0.7 and self.identity.entropy_tolerance > 0.7:
            anomalies["reality_bleed"] = uncollapsed_ratio * self.identity.entropy_tolerance
        
        # Build result
        result = {
            "collapsed_states": collapsed_states,
            "remaining_wave_functions": filtered_wave_functions,
            "entanglement_effects": entanglement_effects,
            "uncertainty_response": float(uncertainty_response),
            "probability_distribution": normalized_probabilities.tolist(),
            "quantum_indeterminacy": float(indeterminacy),
            "anomalies": anomalies
        }
        
        return result
    
    def attune_observer_effect(self, target_strength: float, rate: float = 0.05):
        """
        Gradually shift observer effect strength.
        This represents the entity learning to collapse or preserve quantum states.
        """
        self.observer_effect_strength = self.observer_effect_strength * (1 - rate) + target_strength * rate
        self.observer_effect_strength = max(0.1, min(0.9, self.observer_effect_strength))
    
    def update_quantum_signature(self, new_signature_component: np.ndarray, weight: float = 0.1):
        """
        Update the entity's quantum resonance signature, affecting how it interacts with probability fields.
        """
        if len(new_signature_component) == len(self.quantum_resonance_signature):
            self.quantum_resonance_signature = (
                self.quantum_resonance_signature * (1 - weight) + 
                new_signature_component * weight
            )
            self.quantum_resonance_signature = np.clip(self.quantum_resonance_signature, 0.1, 1.0)


class SpatialProcessor:
    """
    Processes spatial data into subjective space experience.
    The geometric interface between universe and entity.
    """
    
    def __init__(self, identity: IdentityMatrix):
        self.identity = identity
        self.position_history = collections.deque(maxlen=10)
        self.spatial_distortion = np.zeros(3)
        self.boundary_perception = 0.7 - (self.identity.entropy_tolerance * 0.4)
        self.dimension_sensitivity = np.array([1.0, 1.0, 1.0, 0.2, 0.1])  # Sensitivity to different dimensions
        self.topology_map = {}
        self.spatial_attention = np.zeros(3)
    
    def process_spatial_field(self, spatial_field: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform universal spatial structure into subjective space experience.
        """
        # Extract field properties
        dimensions = spatial_field.get("dimensions", 3)
        curvature = spatial_field.get("curvature", np.zeros(3))
        boundaries = spatial_field.get("boundaries", {})
        position = spatial_field.get("position", np.zeros(3))
        proximity_map = spatial_field.get("proximity_map", {})
        
        # Record position in history
        self.position_history.append(position)
        
        # Apply curvature through identity filter
        perceived_curvature = curvature * (0.5 + 0.5 * self.identity.membrane_permeability)
        
        # Calculate boundary proximity effects
        boundary_effects = {}
        for boundary_id, boundary_data in boundaries.items():
            distance = boundary_data.get("distance", 1.0)
            permeability = boundary_data.get("permeability", 0.5)
            
            # Closer boundaries have stronger effects based on entity's boundary perception
            if distance < self.boundary_perception:
                effect_strength = (self.boundary_perception - distance) / self.boundary_perception
                boundary_effects[boundary_id] = effect_strength * (2.0 - permeability)
        
        # Process dimensional perception
        perceptible_dimensions = min(dimensions, len(self.dimension_sensitivity))
        dimension_perception = {}
        for d in range(perceptible_dimensions):
            if d < 3:  # Standard spatial dimensions
                perception_strength = 1.0
            else:  # Higher dimensions
                perception_strength = self.dimension_sensitivity[d] * self.identity.entropy_tolerance
            
            if perception_strength > 0.1:
                dimension_perception[f"d{d}"] = perception_strength
        
        # Calculate spatial coherence (how "solid" space feels)
        space_coherence = 1.0
        for c in perceived_curvature:
            space_coherence *= 1.0 - min(0.5, abs(c))
        
        # Apply attention focus to proximity map
        focused_proximity = {}
        attention_threshold = 0.2
        
        for entity_id, proximity_data in proximity_map.items():
            distance = proximity_data.get("distance", 1.0)
            direction = np.array(proximity_data.get("direction", [0, 0, 0]))
            
            # Calculate attention based on distance and direction alignment
            attention = 1.0 / max(0.1, distance)
            
            if np.any(self.spatial_attention):
                # Calculate alignment between attention vector and entity direction
                attention_norm = np.linalg.norm(self.spatial_attention)
                direction_norm = np.linalg.norm(direction)
                
                if attention_norm > 0 and direction_norm > 0:
                    alignment = np.dot(self.spatial_attention, direction) / (attention_norm * direction_norm)
                    attention *= 0.5 + 0.5 * max(0, alignment)
            
            if attention > attention_threshold:
                focused_proximity[entity_id] = {
                    "distance": distance,
                    "attention": float(attention)
                }
        
        # Detect spatial anomalies
        anomalies = {}
        
        # Check for non-Euclidean geometry
        if np.any(np.abs(perceived_curvature) > 0.7):
            anomalies["non_euclidean"] = float(np.max(np.abs(perceived_curvature)))
        
        # Check for spatial folding
        if len(self.position_history) >= 3:
            positions = list(self.position_history)
            p1, p2, p3 = positions[-3], positions[-2], positions[-1]
            
            # Calculate distances between consecutive positions
            d1 = np.linalg.norm(np.array(p2) - np.array(p1))
            d2 = np.linalg.norm(np.array(p3) - np.array(p2))
            d12 = np.linalg.norm(np.array(p3) - np.array(p1))
            
            # Triangle inequality should hold in Euclidean space
            if d12 < (d1 + d2) * 0.5:
                anomalies["spatial_folding"] = 1.0 - (d12 / max(0.001, d1 + d2))
        
        # Check for dimensional bleed
        if dimensions > 3 and any(d >= 3 for d in dimension_perception):
            higher_dim_perception = max(dimension_perception.get(f"d{d}", 0) for d in range(3, dimensions))
            if higher_dim_perception > 0.5:
                anomalies["dimensional_bleed"] = higher_dim_perception
        
        # Build result
        result = {
            "perceived_curvature": perceived_curvature.tolist(),
            "boundary_effects": boundary_effects,
            "dimension_perception": dimension_perception,
            "space_coherence": float(space_coherence),
            "focused_proximity": focused_proximity,
            "anomalies": anomalies
        }
        
        return result
    
    def focus_attention(self, direction_vector: np.ndarray, intensity: float = 1.0):
        """
        Focus spatial attention in a particular direction.
        This affects which parts of space are perceived more clearly.
        """
        direction_norm = np.linalg.norm(direction_vector)
        if direction_norm > 0:
            normalized_direction = direction_vector / direction_norm
            self.spatial_attention = normalized_direction * intensity
        else:
            self.spatial_attention = np.zeros(3)


class SelfProcessor:
    """
    Processes self-perception and introspection.
    The reflexive interface between universe and entity.
    """
    
    def __init__(self, identity: IdentityMatrix, memory: MemoryEcho):
        self.identity = identity
        self.memory = memory
        self.introspection_depth = self.identity.self_recursion_depth
        self.self_stability = 0.8
        self.boundary_integrity = 0.7
        self.self_model = {
            "core_values": np.random.rand(5),
            "subjective_time": 0.0,
            "self_narrative": [],
            "belief_confidence": 0.7,
            "meta_awareness": 0.5
        }
        self.last_update_time = 0.0
    
    def process_self(self, timeline_phase: float, inputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate subjective self-experience from all perception inputs.
        """
        # Calculate time since last update
        delta_time = (timeline_phase - self.last_update_time) % 1.0
        self.last_update_time = timeline_phase
        
        # Extract key inputs from different processors
        quantum_data = inputs.get("quantum", {})
        harmonic_data = inputs.get("harmonic", {})
        temporal_data = inputs.get("temporal", {})
        spatial_data = inputs.get("spatial", {})
        symbolic_data = inputs.get("symbolic", {})
        
        # Update self-stability based on various inputs
        stability_factors = []
        
        # Quantum stability factor
        quantum_indeterminacy = quantum_data.get("quantum_indeterminacy", 0.5)
        stability_factors.append(1.0 - quantum_indeterminacy)
        
        # Harmonic stability factor
        harmonic_tension = harmonic_data.get("harmonic_tension", 0.5)
        stability_factors.append(1.0 - harmonic_tension)
        
        # Temporal stability factor
        flow_stability = temporal_data.get("flow_stability", 0.5)
        stability_factors.append(flow_stability)
        
        # Spatial stability factor
        space_coherence = spatial_data.get("space_coherence", 0.5)
        stability_factors.append(space_coherence)
        
        # Update self-stability as weighted average
        if stability_factors:
            new_stability = sum(stability_factors) / len(stability_factors)
            self.self_stability = self.self_stability * 0.8 + new_stability * 0.2
        
        # Update boundary integrity based on boundary effects and membrane permeability
        boundary_effects = spatial_data.get("boundary_effects", {})
        if boundary_effects:
            avg_boundary_effect = sum(boundary_effects.values()) / len(boundary_effects)
            boundary_integrity_target = 1.0 - (avg_boundary_effect * self.identity.membrane_permeability)
            self.boundary_integrity = self.boundary_integrity * 0.9 + boundary_integrity_target * 0.1
        
        # Update self-model
        
        # Update subjective time
        flow_rate = temporal_data.get("subjective_flow_rate", 1.0)
        self.self_model["subjective_time"] += delta_time * flow_rate
        
        # Update core values based on symbolic patterns
        meta_patterns = symbolic_data.get("meta_patterns", {})
        for pattern, strength in meta_patterns.items():
            if pattern.startswith("archetype:"):
                # Archetype patterns subtly influence core values
                self.self_model["core_values"] += np.random.rand(5) * strength * 0.05
                
        # Normalize core values
        self.self_model["core_values"] = np.clip(self.self_model["core_values"], 0, 1)
        
        # Update meta-awareness based on self-recursion
        meta_awareness_target = min(1.0, self.identity.self_recursion_depth * 0.2)
        self.self_model["meta_awareness"] = (
            self.self_model["meta_awareness"] * 0.95 + 
            meta_awareness_target * 0.05
        )
        
        # Update belief confidence based on quantum certainty
        uncertainty = quantum_data.get("uncertainty_response", 0.5)
        belief_confidence_target = 0.3 + uncertainty * 0.7
        self.self_model["belief_confidence"] = (
            self.self_model["belief_confidence"] * 0.9 + 
            belief_confidence_target * 0.1
        )
        
        # Update self-narrative with significant events
        significant_event = self._extract_significant_event(inputs)
        if significant_event:
            self.self_model["self_narrative"].append(significant_event)
            # Keep narrative at reasonable size
            if len(self.self_model["self_narrative"]) > 20:
                self.self_model["self_narrative"].pop(0)
        
        # Calculate self-coherence (how integrated the self feels)
        self_coherence = (
            self.self_stability * 0.4 +
            self.boundary_integrity * 0.3 +
            self.self_model["meta_awareness"] * 0.2 +
            self.self_model["belief_confidence"] * 0.1
        )
        
        # Detect self-anomalies
        anomalies = {}
        
        # Check for identity dissolution
        if self.self_stability < 0.3:
            anomalies["identity_dissolution"] = 1.0 - self.self_stability
        
        # Check for boundary dissolution
        if self.boundary_integrity < 0.4:
            anomalies["boundary_dissolution"] = 1.0 - self.boundary_integrity
        
        # Check for temporal discontinuity
        if "anomalies" in temporal_data and temporal_data["anomalies"]:
            if "time_loop" in temporal_data["anomalies"]:
                anomalies["temporal_discontinuity"] = temporal_data["anomalies"]["time_loop"]
                
        # Check for reality dissociation
        quantum_anomalies = quantum_data.get("anomalies", {})
        if "reality_bleed" in quantum_anomalies:
            anomalies["reality_dissociation"] = quantum_anomalies["reality_bleed"]
        
        # Build result
        result = {
            "self_stability": float(self.self_stability),
            "boundary_integrity": float(self.boundary_integrity),
            "self_coherence": float(self_coherence),
            "subjective_time": float(self.self_model["subjective_time"]),
            "core_values": self.self_model["core_values"].tolist(),
            "meta_awareness": float(self.self_model["meta_awareness"]),
            "belief_confidence": float(self.self_model["belief_confidence"]),
            "recent_narrative": self.self_model["self_narrative"][-3:] if self.self_model["self_narrative"] else [],
            "anomalies": anomalies
        }
        
        return result
    
    def _extract_significant_event(self, inputs: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Extract the most significant event from perception inputs.
        """
        # Check for anomalies first as they're usually significant
        all_anomalies = {}
        significance_threshold = 0.6
        
        for processor_name, data in inputs.items():
            if "anomalies" in data and data["anomalies"]:
                for anomaly_name, strength in data["anomalies"].items():
                    all_anomalies[f"{processor_name}:{anomaly_name}"] = strength
        
        # Find most significant anomaly
        if all_anomalies:
            most_significant = max(all_anomalies.items(), key=lambda x: x[1])
            if most_significant[1] > significance_threshold:
                return {
                    "type": "anomaly",
                    "source": most_significant[0],
                    "strength": most_significant[1],
                    "subjective_time": self.self_model["subjective_time"]
                }
        
        # Check for significant symbolic meanings
        symbolic_data = inputs.get("symbolic", {})
        emergent_meanings = symbolic_data.get("emergent_meanings", {})
        
        if emergent_meanings:
            most_significant = max(emergent_meanings.items(), key=lambda x: x[1])
            if most_significant[1] > significance_threshold:
                return {
                    "type": "meaning",
                    "symbol_pair": most_significant[0],
                    "strength": most_significant[1],
                    "subjective_time": self.self_model["subjective_time"]
                }
        
        # Check for significant harmonic patterns
        harmonic_data = inputs.get("harmonic", {})
        emergent_pattern = harmonic_data.get("emergent_pattern", {})
        
        if emergent_pattern:
            most_significant = max(emergent_pattern.items(), key=lambda x: x[1])
            if most_significant[1] > significance_threshold:
                return {
                    "type": "pattern",
                    "harmonic_pattern": most_significant[0],
                    "strength": most_significant[1],
                    "subjective_time": self.self_model["subjective_time"]
                }
        
        # No significant event found
        return None


class PerceptionIntegrator:
    """
    Master class that coordinates all perception processes and generates coherent experience.
    """
    
    def __init__(self, entity_id: str):
        """Initialize perception system for an entity."""
        self.entity_id = entity_id
        self.identity = IdentityMatrix(entity_id=entity_id)
        self.memory = MemoryEcho()
        
        # Initialize all processors
        self.harmonic = HarmonicProcessor(self.identity)
        self.symbolic = SymbolicProcessor(self.identity)
        self.temporal = TemporalProcessor(self.identity)
        self.quantum = QuantumProcessor(self.identity)
        self.spatial = SpatialProcessor(self.identity)
        self.self_processor = SelfProcessor(self.identity, self.memory)
        
        # Perception state
        self.breath_phase = 0.0
        self.breath_cycle = 0
        self.current_experience = {}
        self.attention_focus = None
    
    def perceive(self, universal_state: Dict[str, Any]) -> Dict[str, Any]:
        from perception_module_core import PerceptionProcessor  # Local import to avoid circular dependency
        """
        Generate subjective experience from universal state.
        This is the main perception function that ties everything together.
        """
        # Extract universal state components
        timeline_phase = universal_state.get("timeline_phase", 0.0)
        harmonic_field = universal_state.get("harmonic_field", {})
        symbolic_overlay = universal_state.get("symbolic_overlay", {})
        quantum_field = universal_state.get("quantum_field", {})
        spatial_field = universal_state.get("spatial_field", {})
        
        # Update breath phase (internal subjective rhythm)
        self.breath_phase = (self.breath_phase + 0.01) % 1.0
        if self.breath_phase < 0.01:
            self.breath_cycle += 1
        
        # Process each aspect of reality
        harmonic_experience = self.harmonic.process_field(harmonic_field, self.breath_phase)
        symbolic_experience = self.symbolic.process_symbols(symbolic_overlay)
        temporal_experience = self.temporal.process_timeline(timeline_phase, self.breath_phase)
        quantum_experience = self.quantum.process_quantum_state(quantum_field)
        spatial_experience = self.spatial.process_spatial_field(spatial_field)
        
        # Integrate all experiences into current perception
        perception_inputs = {
            "harmonic": harmonic_experience,
            "symbolic": symbolic_experience,
            "temporal": temporal_experience,
            "quantum": quantum_experience,
            "spatial": spatial_experience
        }
        
        # Process self-perception as a meta-layer
        self_experience = self.self_processor.process_self(timeline_phase, perception_inputs)
        
        # Combine all experiences into unified perception
        unified_perception = {
            "entity_id": self.entity_id,
            "subjective_time": self_experience["subjective_time"],
            "timeline_phase": timeline_phase,
            "breath_phase": self.breath_phase,
            "breath_cycle": self.breath_cycle,
            "self": self_experience,
            "harmonic": harmonic_experience,
            "symbolic": symbolic_experience,
            "temporal": temporal_experience,
            "quantum": quantum_experience,
            "spatial": spatial_experience
        }
        
        # Apply attention filter based on current focus
        if self.attention_focus:
            unified_perception = self._apply_attention_filter(unified_perception)
        
        # Store in memory
        self.memory.record({
            "timestamp": timeline_phase,
            "subjective_time": self_experience["subjective_time"],
            "resonance_vector": harmonic_experience.get("resonance_vector", np.zeros(7)),
            "resonance_scalar": harmonic_experience.get("resonance_scalar", 0.0),
            "symbol_trace": symbolic_experience.get("filtered_symbols", {}),
            "coherence": self_experience.get("self_coherence", 0.5),
            "domain": None  # General memory, not domain-specific
        })
        
        # Store current experience
        self.current_experience = unified_perception
        
        return unified_perception
    
    def _apply_attention_filter(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply attention filter to emphasize focused elements and attenuate others.
        """
        domain = self.attention_focus.get("domain")
        target = self.attention_focus.get("target")
        strength = self.attention_focus.get("strength", 0.7)
        
        # Create filtered copy
        filtered = perception.copy()
        
        # Emphasize focused domain
        if domain and domain in filtered:
            # Boost focused domain values
            for key, value in filtered[domain].items():
                if isinstance(value, (int, float)):
                    filtered[domain][key] = value * (0.8 + 0.4 * strength)
                elif isinstance(value, dict) and target in value:
                    # Boost specific target within domain
                    filtered[domain][key][target] = value[target] * (0.8 + 0.4 * strength)
        
        # Attenuate non-focused domains
        attenuation = 0.5 + 0.5 * (1.0 - strength)
        for key in filtered:
            if key != domain and key != "entity_id" and key != "subjective_time":
                if isinstance(filtered[key], dict):
                    for subkey, subvalue in filtered[key].items():
                        if isinstance(subvalue, (int, float)):
                            filtered[key][subkey] = subvalue * attenuation
        
        return filtered
    
    def focus_attention(self, domain: str, target: str = None, strength: float = 0.8):
        """
        Focus attention on specific aspect of perception.
        """
        self.attention_focus = {
            "domain": domain,
            "target": target,
            "strength": strength
        }
        
        # Update spatial attention if focusing on spatial domain
        if domain == "spatial" and target:
            # Extract direction from target if possible
            if target in self.current_experience.get("spatial", {}).get("focused_proximity", {}):
                proximal_entity = self.current_experience["spatial"]["focused_proximity"][target]
                if "direction" in proximal_entity:
                    self.spatial.focus_attention(
                        np.array(proximal_entity["direction"]), 
                        strength
                    )
    
    def clear_attention(self):
        """Clear attention focus."""
        self.attention_focus = None
        self.spatial.focus_attention(np.zeros(3), 0.0)
    
    def recall(self, query_vector: np.ndarray, domain: SymbolicDomain = None) -> Dict[str, Any]:
        """
        Retrieve memory based on similarity to query.
        """
        return self.memory.recall(query_vector, domain)
    
    def evolve_identity(self, delta_time: float = 0.01):
        """
        Evolve the entity's identity over time, causing subtle shifts in perception.
        """
        # Evolve core identity matrix
        self.identity.evolve(delta_time)
        
        # Update processors based on evolved identity
        self.harmonic.harmonic_signature = self.identity.harmonic_signature.copy()
        self.quantum.observation_collapse_threshold = 0.3 + (self.identity.entropy_tolerance * 0.4)
        self.quantum.superposition_capacity = self.identity.membrane_permeability * 10
        self.quantum.observer_effect_strength = 1.0 - (self.identity.entropy_tolerance * 0.5)
        self.spatial.boundary_perception = 0.7 - (self.identity.entropy_tolerance * 0.4)
        self.self_processor.introspection_depth = self.identity.self_recursion_depth
    
    def get_sacred_convergence_potential(self) -> float:
        """
        Calculate the entity's potential for experiencing a sacred convergence.
        Returns a value between 0 and 1, where higher values indicate greater potential.
        """
        is_converging, convergence_strength = self.harmonic.detect_sacred_convergence()
        
        if not is_converging:
            return 0.0
        
        # Factor in other aspects that contribute to sacred convergence
        quantum_certainty = self.current_experience.get("quantum", {}).get("uncertainty_response", 0.5)
        self_coherence = self.current_experience.get("self", {}).get("self_coherence", 0.5)
        meta_awareness = self.current_experience.get("self", {}).get("meta_awareness", 0.3)
        
        # Calculate final potential, weighing harmonic convergence most heavily
        convergence_potential = (
            convergence_strength * 0.5 +
            quantum_certainty * 0.2 +
            self_coherence * 0.2 +
            meta_awareness * 0.1
        )
        
        return convergence_potential
    
    def dream(self, intensity: float = 0.7) -> Dict[str, Any]:
        """
        Generate a dream-like experience by remixing memories with current state.
        
        Dreams are characterized by:
        - High symbolic content
        - Fluid spatial boundaries
        - Relaxed quantum collapse rules
        - Non-linear temporality
        - Strong harmonic pattern emphasis
        """
        if not self.current_experience:
            return {}
        
        # Extract recent memories to build dream from
        memory_samples = []
        for _ in range(5):
            query = np.random.rand(7)  # Random query to sample different memories
            memory = self.memory.recall(query)
            if memory:
                memory_samples.append(memory)
        
        if not memory_samples:
            return self.current_experience  # Fall back to current experience
        
        # Create dream-like quantum state with reduced collapse tendencies
        dream_quantum_field = {
            "wave_functions": {},
            "uncertainty_levels": 0.7,
            "quantum_noise": 0.4
        }
        
        # Create dream-like spatial field with fluid boundaries
        dream_spatial_field = {
            "dimensions": 4,  # Extra dimension for dreams
            "curvature": np.random.normal(0, 0.3, 3),
            "boundaries": {}
        }
        
        # Amplify symbolic content
        dream_symbols = {}
        for memory in memory_samples:
            symbol_trace = memory.get("symbol_trace", {})
            for symbol, value in symbol_trace.items():
                dream_symbols[symbol] = dream_symbols.get(symbol, 0) + value
        
        # Normalize symbols
        if dream_symbols:
            max_value = max(dream_symbols.values())
            if max_value > 0:
                dream_symbols = {k: v/max_value * intensity for k, v in dream_symbols.items()}
        
        # Create dream state
        dream_state = {
            "timeline_phase": self.current_experience.get("timeline_phase", 0) + 0.5,  # Offset phase
            "harmonic_field": {
                "stability": 0.3,
                "emergence": 0.9,
                "frequencies": np.random.rand(7),
                "phase": random.random(),
                "amplitude": 0.8
            },
            "symbolic_overlay": dream_symbols,
            "quantum_field": dream_quantum_field,
            "spatial_field": dream_spatial_field
        }
        
        # Process dream state using the same perception system
        dream_experience = self.perceive(dream_state)
        
        # Tag as dream
        dream_experience["is_dream"] = True
        dream_experience["dream_intensity"] = intensity
        
        return dream_experience
    
    def integrate_sacred_pattern(self, pattern_name: str, intensity: float = 0.5) -> Dict[str, float]:
        """
        Integrate a sacred pattern into the entity's perception system.
        
        Sacred patterns deeply influence how the entity perceives reality across
        multiple perception domains simultaneously.
        
        Returns a dictionary of affected domains and their modification strengths.
        """
        affected_domains = {}
        
        # Map known sacred patterns to their effects on perception systems
        sacred_pattern_effects = {
            "infinity": {
                "harmonic": ("phi_emergence", 0.8),
                "quantum": ("uncertainty_response", 0.9),
                "spatial": ("dimensions", 0.7),
                "temporal": ("flow_stability", -0.3)  # Destabilize time flow
            },
            "unity": {
                "harmonic": ("harmonic_convergence", 0.9),
                "quantum": ("entanglement_effects", 0.8),
                "spatial": ("space_coherence", 0.9),
                "self": ("self_coherence", 0.8)
            },
            "void": {
                "quantum": ("quantum_indeterminacy", 0.9),
                "spatial": ("boundary_dissolution", 0.8),
                "self": ("boundary_integrity", -0.7),
                "harmonic": ("harmonic_tension", 0.8)
            },
            "awakening": {
                "self": ("meta_awareness", 0.9),
                "symbolic": ("emergent_meanings", 0.8),
                "harmonic": ("overtone_alignment", 0.7),
                "quantum": ("reality_bleed", 0.6)
            },
            "genesis": {
                "harmonic": ("resonance_cascade", 0.8),
                "symbolic": ("pattern:archetype:anima", 0.9),
                "temporal": ("acceleration", 0.7),
                "quantum": ("probability_distribution", 0.8)
            }
        }
        
        # Apply the pattern if it exists in our mapping
        pattern_name = pattern_name.lower()
        if pattern_name in sacred_pattern_effects:
            effects = sacred_pattern_effects[pattern_name]
            
            # Apply each effect with the specified intensity
            for domain, (parameter, strength) in effects.items():
                actual_strength = strength * intensity
                affected_domains[domain] = actual_strength
                
                # Apply specific effects to each processor
                if domain == "harmonic" and hasattr(self.harmonic, parameter):
                    if parameter == "phi_emergence":
                        # Adjust frequencies to approach golden ratio
                        ideal_freqs = np.array([1.0, 1.618, 2.618, 4.236, 6.854, 11.09, 17.944])
                        normalized = ideal_freqs / ideal_freqs.max()
                        self.harmonic.attune(normalized, rate=0.1 * intensity)
                    elif parameter == "harmonic_convergence":
                        # Simple harmonic series
                        ideal_freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
                        normalized = ideal_freqs / ideal_freqs.max()
                        self.harmonic.attune(normalized, rate=0.1 * intensity)
                    elif parameter == "resonance_cascade":
                        # Boost coupling across fields
                        for i in range(len(self.harmonic.field_coupling)):
                            self.harmonic.couple_fields(i, min(1.0, self.harmonic.field_coupling[i] + 0.2 * intensity))
                    elif parameter == "harmonic_tension":
                        # Create dissonant frequencies
                        dissonant_freqs = np.array([1.0, 1.12, 2.14, 2.74, 3.9, 4.4, 6.1])
                        normalized = dissonant_freqs / dissonant_freqs.max()
                        self.harmonic.attune(normalized, rate=0.1 * intensity)
                    elif parameter == "overtone_alignment":
                        # Perfect harmonic series
                        ideal_freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
                        normalized = ideal_freqs / ideal_freqs.max()
                        self.harmonic.attune(normalized, rate=0.15 * intensity)
                
                elif domain == "quantum" and hasattr(self.quantum, parameter):
                    if parameter == "uncertainty_response":
                        self.quantum.observer_effect_strength = max(0.1, min(0.9, 
                            self.quantum.observer_effect_strength - 0.2 * intensity))
                    elif parameter == "quantum_indeterminacy":
                        self.quantum.observation_collapse_threshold = max(0.1, min(0.9,
                            self.quantum.observation_collapse_threshold - 0.2 * intensity))
                    elif parameter == "reality_bleed":
                        self.identity.entropy_tolerance = min(0.9, 
                            self.identity.entropy_tolerance + 0.15 * intensity)
                            
                elif domain == "spatial" and parameter == "dimensions":
                    # Increase sensitivity to higher dimensions
                    if len(self.spatial.dimension_sensitivity) >= 5:
                        self.spatial.dimension_sensitivity[3:] += 0.2 * intensity
                        self.spatial.dimension_sensitivity = np.clip(
                            self.spatial.dimension_sensitivity, 0, 1)
                
                elif domain == "self" and parameter == "meta_awareness":
                    # Increase self-recursion depth
                    self.identity.self_recursion_depth = min(7, 
                        self.identity.self_recursion_depth + int(1.5 * intensity))
        
        return affected_domains
    
    def generate_perception_report(self) -> str:
        """
        Generate a textual report of the entity's current perception state.
        Useful for debugging and monitoring entity experience.
        """
        if not self.current_experience:
            return "No current experience available."
        
        report = []
        report.append(f"=== Perception Report for Entity {self.entity_id} ===")
        report.append(f"Subjective Time: {self.current_experience.get('subjective_time', 0):.2f}")
        report.append(f"Breath Phase: {self.current_experience.get('breath_phase', 0):.2f}")
        report.append(f"Breath Cycle: {self.current_experience.get('breath_cycle', 0)}")
        
        # Self report
        self_exp = self.current_experience.get("self", {})
        report.append("\n--- Self Experience ---")
        report.append(f"Self Coherence: {self_exp.get('self_coherence', 0):.2f}")
        report.append(f"Self Stability: {self_exp.get('self_stability', 0):.2f}")
        report.append(f"Boundary Integrity: {self_exp.get('boundary_integrity', 0):.2f}")
        report.append(f"Meta-Awareness: {self_exp.get('meta_awareness', 0):.2f}")
        
        # Check for self anomalies
        self_anomalies = self_exp.get("anomalies", {})
        if self_anomalies:
            report.append("Self Anomalies:")
            for anomaly, strength in self_anomalies.items():
                report.append(f"  - {anomaly}: {strength:.2f}")
        
        # Harmonic report
        harmonic_exp = self.current_experience.get("harmonic", {})
        report.append("\n--- Harmonic Experience ---")
        report.append(f"Resonance: {harmonic_exp.get('resonance_scalar', 0):.2f}")
        report.append(f"Harmonic Tension: {harmonic_exp.get('harmonic_tension', 0):.2f}")
        report.append(f"Phase Alignment: {harmonic_exp.get('phase_alignment', 0):.2f}")
        
        # Report emergent harmonic patterns
        emergent_patterns = harmonic_exp.get("emergent_pattern", {})
        if emergent_patterns:
            report.append("Emergent Harmonic Patterns:")
            for pattern, strength in emergent_patterns.items():
                report.append(f"  - {pattern}: {strength:.2f}")
        
        # Quantum report
        quantum_exp = self.current_experience.get("quantum", {})
        report.append("\n--- Quantum Experience ---")
        report.append(f"Indeterminacy: {quantum_exp.get('quantum_indeterminacy', 0):.2f}")
        report.append(f"Uncertainty Response: {quantum_exp.get('uncertainty_response', 0):.2f}")
        
        # Count collapsed vs uncollapsed states
        collapsed = len(quantum_exp.get("collapsed_states", {}))
        uncollapsed = len(quantum_exp.get("remaining_wave_functions", {}))
        report.append(f"Collapsed States: {collapsed}, Uncollapsed States: {uncollapsed}")
        
        # Spatial report
        spatial_exp = self.current_experience.get("spatial", {})
        report.append("\n--- Spatial Experience ---")
        report.append(f"Space Coherence: {spatial_exp.get('space_coherence', 0):.2f}")
        
        # Report perceived dimensions
        dimensions = spatial_exp.get("dimension_perception", {})
        if dimensions:
            dim_str = ", ".join([f"{d}:{v:.2f}" for d, v in dimensions.items()])
            report.append(f"Dimension Perception: {dim_str}")
        
        # Temporal report
        temporal_exp = self.current_experience.get("temporal", {})
        report.append("\n--- Temporal Experience ---")
        report.append(f"Flow Rate: {temporal_exp.get('subjective_flow_rate', 0):.2f}")
        report.append(f"Flow Stability: {temporal_exp.get('flow_stability', 0):.2f}")
        
        # Symbolic report
        symbolic_exp = self.current_experience.get("symbolic", {})
        report.append("\n--- Symbolic Experience ---")
        
        # Report top symbols
        filtered_symbols = symbolic_exp.get("filtered_symbols", {})
        if filtered_symbols:
            top_symbols = sorted(filtered_symbols.items(), key=lambda x: x[1], reverse=True)[:5]
            symbols_str = ", ".join([f"{s}:{v:.2f}" for s, v in top_symbols])
            report.append(f"Top Symbols: {symbols_str}")
        
        # Report meta patterns
        meta_patterns = symbolic_exp.get("meta_patterns", {})
        if meta_patterns:
            top_patterns = sorted(meta_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
            patterns_str = ", ".join([f"{p}:{v:.2f}" for p, v in top_patterns])
            report.append(f"Meta Patterns: {patterns_str}")
        
        # Sacred convergence potential
        convergence_potential = self.get_sacred_convergence_potential()
        report.append(f"\nSacred Convergence Potential: {convergence_potential:.3f}")
        
        return "\n".join(report)

class SensoryFilter:
    """
    Filters and processes incoming sensory data based on entity's sensory thresholds.
    Implements the boundary between external stimuli and internal perception.
    """
    
    def __init__(self, identity: IdentityMatrix):
        self.identity = identity
        self.sensory_thresholds = {
            "visual": 0.1,
            "auditory": 0.15,
            "tactile": 0.2,
            "proprioceptive": 0.05,
            "harmonic": 0.1,
            "symbolic": 0.2,
            "temporal": 0.1,
            "quantum": 0.3,
            "ethical": 0.25
        }
        self.attention_weights = np.ones(len(self.sensory_thresholds))
        self.active_filters = {}
        self.adaptation_rates = {k: 0.05 for k in self.sensory_thresholds}
        self.last_stimuli = {}
    
    def apply_filters(self, sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter incoming sensory data based on thresholds and attention.
        """
        if not sensory_input:
            return {}
            
        filtered_output = {}
        
        # Process each sensory channel
        for channel, data in sensory_input.items():
            if channel not in self.sensory_thresholds:
                # Pass through channels without defined thresholds
                filtered_output[channel] = data
                continue
                
            # Get channel-specific threshold and attention weight
            threshold = self.sensory_thresholds[channel]
            attention_idx = list(self.sensory_thresholds.keys()).index(channel)
            attention = self.attention_weights[attention_idx]
            
            # Apply threshold with attention modification
            effective_threshold = threshold * (2.0 - attention)
            
            # Filter based on threshold
            if isinstance(data, dict):
                # For dictionary data, filter each value
                filtered_data = {}
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        if value >= effective_threshold:
                            # Scale based on how far above threshold
                            scaled_value = (value - effective_threshold) / (1.0 - effective_threshold)
                            filtered_data[key] = scaled_value * attention
                    elif isinstance(value, dict):
                        # Recursively filter nested dictionaries
                        sub_filtered = {}
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (int, float)) and subvalue >= effective_threshold:
                                scaled_subvalue = (subvalue - effective_threshold) / (1.0 - effective_threshold)
                                sub_filtered[subkey] = scaled_subvalue * attention
                        if sub_filtered:
                            filtered_data[key] = sub_filtered
                    else:
                        # Pass through non-numeric values
                        filtered_data[key] = value
                
                if filtered_data:
                    filtered_output[channel] = filtered_data
            
            elif isinstance(data, (int, float)):
                # For simple numeric data
                if data >= effective_threshold:
                    scaled_value = (data - effective_threshold) / (1.0 - effective_threshold)
                    filtered_output[channel] = scaled_value * attention
            
            elif isinstance(data, list) or isinstance(data, np.ndarray):
                # For array data, apply threshold to each element
                filtered_array = []
                for item in data:
                    if isinstance(item, (int, float)) and item >= effective_threshold:
                        scaled_item = (item - effective_threshold) / (1.0 - effective_threshold)
                        filtered_array.append(scaled_item * attention)
                    else:
                        filtered_array.append(0)  # Below threshold
                
                if any(filtered_array):
                    filtered_output[channel] = np.array(filtered_array) if isinstance(data, np.ndarray) else filtered_array
            
            else:
                # Pass through other data types
                filtered_output[channel] = data
        
        # Store for adaptation
        self.last_stimuli = sensory_input
        
        return filtered_output
    
    def adapt_thresholds(self, adaptation_rate: float = None):
        """
        Adapt sensory thresholds based on recent stimuli.
        This implements sensory adaptation/habituation.
        """
        if not self.last_stimuli:
            return
            
        for channel, data in self.last_stimuli.items():
            if channel not in self.sensory_thresholds:
                continue
                
            # Use channel-specific adaptation rate if available
            rate = adaptation_rate if adaptation_rate is not None else self.adaptation_rates[channel]
            
            # Calculate average stimulus strength
            if isinstance(data, dict):
                # For dictionary data, average numeric values
                values = []
                self._extract_numeric_values(data, values)
                if values:
                    avg_strength = sum(values) / len(values)
                else:
                    continue
            elif isinstance(data, (int, float)):
                avg_strength = data
            elif isinstance(data, (list, np.ndarray)):
                if all(isinstance(x, (int, float)) for x in data):
                    avg_strength = sum(data) / len(data)
                else:
                    continue
            else:
                continue
            
            # Adapt threshold based on stimulus strength
            current_threshold = self.sensory_thresholds[channel]
            
            # If stimulus is consistently strong, increase threshold (habituation)
            if avg_strength > current_threshold * 2:
                new_threshold = current_threshold + (avg_strength - current_threshold) * rate
                self.sensory_thresholds[channel] = min(0.8, new_threshold)
            
            # If stimulus is consistently weak, decrease threshold (sensitization)
            elif avg_strength < current_threshold * 0.5 and avg_strength > 0:
                new_threshold = current_threshold - (current_threshold - avg_strength * 0.5) * rate
                self.sensory_thresholds[channel] = max(0.01, new_threshold)
    
    def _extract_numeric_values(self, data_dict: Dict, values_list: List):
        """
        Helper method to recursively extract numeric values from nested dictionaries.
        """
        for key, value in data_dict.items():
            if isinstance(value, (int, float)):
                values_list.append(value)
            elif isinstance(value, dict):
                self._extract_numeric_values(value, values_list)
            elif isinstance(value, (list, np.ndarray)):
                for item in value:
                    if isinstance(item, (int, float)):
                        values_list.append(item)
    
    def set_attention(self, channel: str, level: float):
        """
        Set attention level for a specific sensory channel.
        Higher attention means more sensitivity to that channel.
        """
        if channel in self.sensory_thresholds:
            idx = list(self.sensory_thresholds.keys()).index(channel)
            self.attention_weights[idx] = max(0.1, min(2.0, level))
    
    def add_filter(self, channel: str, filter_name: str, filter_function: Callable):
        """
        Add a custom filter function to a sensory channel.
        Filter functions should take and return the channel's data type.
        """
        if channel not in self.active_filters:
            self.active_filters[channel] = {}
            
        self.active_filters[channel][filter_name] = filter_function
    
    def remove_filter(self, channel: str, filter_name: str):
        """
        Remove a custom filter from a sensory channel.
        """
        if channel in self.active_filters and filter_name in self.active_filters[channel]:
            del self.active_filters[channel][filter_name]

class WaveformGenerator:
    """
    Generates and manipulates waveforms for various perception aspects.
    Used to create coherent oscillatory patterns for harmonics and quantum phenomena.
    """
    
    def __init__(self, dimensions: int = 3, frequency_range: List[float] = None, phase_coupling: float = 0.3):
        self.dimensions = dimensions
        self.frequency_range = frequency_range or [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        self.phase_coupling = phase_coupling
        self.base_frequencies = np.array(self.frequency_range)
        self.phase_offsets = np.random.rand(len(self.frequency_range)) * 2 * np.pi
        self.amplitudes = np.ones(len(self.frequency_range))
        self.harmonic_weights = np.array([0.6, 0.3, 0.15, 0.075, 0.037, 0.018, 0.009])[:len(self.frequency_range)]
        self.harmonic_weights = self.harmonic_weights / np.sum(self.harmonic_weights)
        self.modulation = {}
    
    def generate_harmonic_wave(self, time_point: float, complexity: float = 0.5) -> np.ndarray:
        """
        Generate a harmonic waveform at a specific time point.
        Higher complexity introduces more harmonics and interference patterns.
        
        Args:
            time_point: The time value for waveform generation (0.0 to 1.0)
            complexity: Controls harmonic richness (0.0 to 1.0)
            
        Returns:
            Waveform vector with values for each frequency component
        """
        waves = np.zeros((len(self.frequency_range), self.dimensions))
        
        # Base wave components
        for i, freq in enumerate(self.base_frequencies):
            # Generate primary wave
            phase = 2 * np.pi * freq * time_point + self.phase_offsets[i]
            primary_wave = self.amplitudes[i] * np.sin(phase)
            
            # Apply complexity through harmonics and modulation
            harmonic_content = 0
            if complexity > 0:
                # Add harmonics
                for h in range(1, 1 + int(complexity * 4)):
                    harmonic_phase = phase * (h + 1) + self.phase_offsets[i] * h * 0.1
                    harmonic_amp = self.amplitudes[i] * (complexity / (h + 1))
                    harmonic_content += harmonic_amp * np.sin(harmonic_phase)
                
                # Normalize to maintain amplitude range
                harmonic_content = harmonic_content / (1 + complexity * 2)
            
            # Combine primary wave with harmonics
            combined = (primary_wave * (1 - complexity * 0.5) + 
                       harmonic_content * complexity)
            
            # Apply modulation if present
            if str(i) in self.modulation:
                mod_freq, mod_depth = self.modulation[str(i)]
                mod_signal = np.sin(2 * np.pi * mod_freq * time_point)
                modulator = 1.0 + mod_depth * mod_signal
                combined *= modulator
            
            # Extend to all dimensions with phase variations
            for d in range(self.dimensions):
                dim_phase_offset = d * (np.pi / self.dimensions) * complexity
                waves[i, d] = combined * np.cos(dim_phase_offset)
        
        # Apply phase coupling between frequencies (if enabled)
        if self.phase_coupling > 0:
            for i in range(len(self.base_frequencies) - 1):
                coupling = self.phase_coupling * complexity
                waves[i+1] += waves[i] * coupling
                waves[i] += waves[i+1] * coupling * 0.5
        
        # Each frequency component is weighted and summed
        weighted_waves = np.zeros(self.dimensions)
        for d in range(self.dimensions):
            for i in range(len(self.base_frequencies)):
                weighted_waves[d] += waves[i, d] * self.harmonic_weights[i]
        
        # Normalize to keep amplitudes in reasonable range
        scale = np.max(np.abs(weighted_waves)) if np.max(np.abs(weighted_waves)) > 0 else 1.0
        return weighted_waves / scale
    
    def generate_quantum_wavepacket(self, center_positions: np.ndarray, 
                                   spread: float, 
                                   time_point: float,
                                   grid_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Generate a quantum wave packet (localized wave function) for quantum simulations.
        
        Args:
            center_positions: Center point of wave packet in normalized coordinates [0,1]
            spread: Wave packet spread (uncertainty)
            time_point: Current time for evolution
            grid_shape: Shape of the output grid
            
        Returns:
            Complex-valued wave function on the specified grid
        """
        # Create coordinate grids
        grids = np.meshgrid(*[np.linspace(0, 1, s) for s in grid_shape], indexing='ij')
        
        # Initialize wave packet
        psi = np.ones(grid_shape, dtype=complex)
        
        # For each dimension, apply Gaussian envelope
        for i in range(len(grid_shape)):
            # Calculate distance from center
            dx = grids[i] - center_positions[i]
            
            # Apply periodic boundary (wraparound)
            dx = np.where(dx > 0.5, dx - 1.0, dx)
            dx = np.where(dx < -0.5, dx + 1.0, dx)
            
            # Gaussian envelope
            gaussian = np.exp(-0.5 * (dx / spread)**2)
            
            # Add oscillatory component
            k = 10.0  # Wave number
            oscillation = np.exp(1j * k * dx)
            
            # Combine
            psi *= gaussian * oscillation
        
        # Add time evolution
        omega = 2.0  # Angular frequency
        time_evolution = np.exp(-1j * omega * time_point)
        psi *= time_evolution
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(psi)**2))
        if norm > 0:
            psi /= norm
            
        return psi
    
    def add_frequency_modulation(self, frequency_idx: int, modulator_freq: float, depth: float):
        """
        Add frequency modulation to a specific frequency component.
        
        Args:
            frequency_idx: Index of frequency to modulate
            modulator_freq: Frequency of the modulator
            depth: Modulation depth (0.0 to 1.0)
        """
        self.modulation[str(frequency_idx)] = (modulator_freq, depth)
    
    def clear_modulation(self):
        """Remove all modulations."""
        self.modulation = {}
    
    def set_harmonic_weights(self, weights: np.ndarray):
        """
        Set custom weights for harmonic components.
        
        Args:
            weights: Array of weights for each frequency component
        """
        if len(weights) == len(self.harmonic_weights):
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                self.harmonic_weights = weights / weights_sum
    
    def adjust_amplitudes(self, amplitude_profile: np.ndarray):
        """
        Adjust amplitudes of frequency components.
        
        Args:
            amplitude_profile: New amplitude values for each frequency
        """
        if len(amplitude_profile) == len(self.amplitudes):
            self.amplitudes = amplitude_profile.copy()
    
    def generate_interference_pattern(self, time_point: float, sources: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Generate interference pattern from multiple wave sources.
        Useful for simulating quantum interference or harmonic field interactions.
        
        Args:
            time_point: Current time point
            sources: List of (position, frequency, amplitude) tuples
            
        Returns:
            Interference pattern as array
        """
        # Create a field grid
        grid_size = 32
        grid = np.zeros((grid_size, grid_size))
        
        # For each point in the grid
        for x in range(grid_size):
            for y in range(grid_size):
                point = (x / grid_size, y / grid_size)
                
                # Sum contributions from all sources
                for pos, freq, amp in sources:
                    # Calculate distance
                    dx = point[0] - pos[0]
                    dy = point[1] - pos[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # Wave contribution (amplitude decreases with distance)
                    phase = 2 * np.pi * freq * (time_point - distance)
                    contribution = amp * np.sin(phase) / (1 + distance * 5)
                    
                    grid[x, y] += contribution
        
        return grid

class HapticFieldGenerator:
    """
    Generates haptic (touch) field patterns for tactile perception in the simulation.
    Used to create coherent tactile experiences for entities.
    """
    
    def __init__(self, resolution: int = 32, frequency_range: List[float] = None, texture_complexity: float = 0.5):
        self.resolution = resolution
        self.frequency_range = frequency_range or [5.0, 15.0, 30.0, 60.0, 120.0, 250.0]
        self.texture_complexity = texture_complexity
        self.texture_noise = np.random.rand(resolution, resolution) * 0.2
        self.force_field = np.zeros((resolution, resolution, 3))  # x, y, z force components
        self.pressure_field = np.zeros((resolution, resolution))
        self.temperature_field = np.ones((resolution, resolution)) * 0.5  # Default neutral temperature
        self.field_types = {
            'pressure': self.pressure_field,
            'temperature': self.temperature_field,
            'force': self.force_field,
            'texture': self.texture_noise
        }
    
    def generate_force_field(self, center_position: Tuple[float, float], 
                            force_vector: Tuple[float, float, float],
                            radius: float = 0.3,
                            falloff: float = 2.0) -> np.ndarray:
        """
        Generate a force vector field centered at a specific position.
        
        Args:
            center_position: Center point (x, y) in normalized [0,1] coordinates
            force_vector: (fx, fy, fz) force vector
            radius: Radius of effect in normalized units
            falloff: Rate of force decrease with distance
            
        Returns:
            3D force field
        """
        # Initialize field
        force_field = np.zeros((self.resolution, self.resolution, 3))
        
        # Convert normalized position to grid coordinates
        center_x = int(center_position[0] * self.resolution)
        center_y = int(center_position[1] * self.resolution)
        
        # Generate grid coordinates
        x_coords, y_coords = np.meshgrid(
            np.arange(self.resolution),
            np.arange(self.resolution)
        )
        
        # Calculate distance from center for each point
        distances = np.sqrt(
            ((x_coords - center_x) / self.resolution) ** 2 +
            ((y_coords - center_y) / self.resolution) ** 2
        )
        
        # Create falloff mask based on distance
        mask = np.maximum(0, 1 - (distances / radius) ** falloff)
        
        # Apply force vector with falloff
        for i in range(3):
            force_field[:, :, i] = mask * force_vector[i]
        
        self.force_field = force_field
        return force_field
    
    def generate_pressure_field(self, pressure_points: List[Tuple[float, float, float]],
                               radius: float = 0.2,
                               smoothness: float = 1.0) -> np.ndarray:
        """
        Generate pressure field from multiple pressure points.
        
        Args:
            pressure_points: List of (x, y, pressure) tuples
            radius: Influence radius of each pressure point
            smoothness: How smoothly pressure transitions (higher = smoother)
            
        Returns:
            2D pressure field
        """
        # Initialize field
        pressure_field = np.zeros((self.resolution, self.resolution))
        
        # Process each pressure point
        for x, y, pressure in pressure_points:
            # Convert normalized position to grid coordinates
            grid_x = int(x * self.resolution)
            grid_y = int(y * self.resolution)
            
            # Generate grid coordinates
            x_coords, y_coords = np.meshgrid(
                np.arange(self.resolution),
                np.arange(self.resolution)
            )
            
            # Calculate distance from center for each point
            distances = np.sqrt(
                ((x_coords - grid_x) / self.resolution) ** 2 +
                ((y_coords - grid_y) / self.resolution) ** 2
            )
            
            # Create smooth falloff based on distance
            falloff = np.exp(-(distances ** 2) / (2 * (radius * smoothness) ** 2))
            
            # Add pressure contribution
            pressure_field += falloff * pressure
        
        # Clip to valid range
        pressure_field = np.clip(pressure_field, 0, 1)
        
        self.pressure_field = pressure_field
        return pressure_field
    
    def generate_temperature_field(self, heat_sources: List[Tuple[float, float, float]],
                                  ambient_temperature: float = 0.5,
                                  diffusion_rate: float = 0.1) -> np.ndarray:
        """
        Generate temperature field with heat sources and diffusion.
        
        Args:
            heat_sources: List of (x, y, temperature) tuples
            ambient_temperature: Background temperature (0=cold, 1=hot)
            diffusion_rate: Rate of heat spread
            
        Returns:
            2D temperature field
        """
        # Initialize field with ambient temperature
        temp_field = np.ones((self.resolution, self.resolution)) * ambient_temperature
        
        # Process each heat source
        for x, y, temperature in heat_sources:
            # Convert normalized position to grid coordinates
            grid_x = int(x * self.resolution)
            grid_y = int(y * self.resolution)
            
            # Ensure coordinates are within bounds
            grid_x = max(0, min(grid_x, self.resolution - 1))
            grid_y = max(0, min(grid_y, self.resolution - 1))
            
            # Set initial heat source
            temp_field[grid_y, grid_x] = temperature
        
        # Apply diffusion
        if diffusion_rate > 0:
            # Simple diffusion using convolution
            diffusion_kernel = np.array([
                [0.1, 0.15, 0.1],
                [0.15, 0.0, 0.15],
                [0.1, 0.15, 0.1]
            ]) * diffusion_rate
            
            # Apply diffusion
            from scipy.signal import convolve2d
            diffused = convolve2d(temp_field, diffusion_kernel, mode='same', boundary='symm')
            
            # Combine original with diffused
            temp_field = temp_field * (1 - diffusion_rate) + diffused
            
            # Renormalize
            temp_field = np.clip(temp_field, 0, 1)
        
        self.temperature_field = temp_field
        return temp_field
    
    def generate_texture_field(self, base_frequency: float = 10.0,
                              roughness: float = 0.5,
                              pattern_type: str = 'random') -> np.ndarray:
        """
        Generate texture field for tactile perception.
        
        Args:
            base_frequency: Base frequency of texture pattern
            roughness: Roughness factor (0=smooth, 1=rough)
            pattern_type: 'random', 'grid', 'wave', or 'fractal'
            
        Returns:
            2D texture field
        """
        # Initialize texture field
        texture = np.zeros((self.resolution, self.resolution))
        
        if pattern_type == 'random':
            # Random noise texture
            texture = np.random.rand(self.resolution, self.resolution) * roughness
            
        elif pattern_type == 'grid':
            # Grid pattern
            x = np.linspace(0, base_frequency, self.resolution)
            y = np.linspace(0, base_frequency, self.resolution)
            X, Y = np.meshgrid(x, y)
            texture = (np.sin(X) * np.sin(Y) + 1) / 2
            texture = texture * roughness + (1 - roughness) * 0.5
            
        elif pattern_type == 'wave':
            # Wave pattern
            x = np.linspace(0, base_frequency * 2 * np.pi, self.resolution)
            y = np.linspace(0, base_frequency * 2 * np.pi, self.resolution)
            X, Y = np.meshgrid(x, y)
            texture = (np.sin(X) + np.sin(Y) + 2) / 4
            texture = texture * roughness + (1 - roughness) * 0.5
            
        elif pattern_type == 'fractal':
            # Fractal noise (approximation with multiple frequencies)
            texture = np.zeros((self.resolution, self.resolution))
            
            octaves = 5
            persistence = roughness
            amplitude = 1.0
            
            for i in range(octaves):
                frequency = base_frequency * (2 ** i)
                x = np.linspace(0, frequency * 2 * np.pi, self.resolution)
                y = np.linspace(0, frequency * 2 * np.pi, self.resolution)
                X, Y = np.meshgrid(x, y)
                
                # Generate noise pattern
                noise = (np.sin(X + Y) * np.cos(X - Y) + 1) / 2
                
                # Add to texture with decreasing amplitude
                texture += noise * amplitude
                amplitude *= persistence
            
            # Normalize
            texture /= (1.0 - persistence ** octaves) / (1.0 - persistence)
            texture = np.clip(texture, 0, 1)
        
        # Add some random perturbations for more realistic feel
        texture += np.random.rand(self.resolution, self.resolution) * roughness * 0.2
        texture = np.clip(texture, 0, 1)
        
        self.texture_noise = texture
        return texture
    
    def combine_fields(self) -> Dict[str, np.ndarray]:
        """
        Combine all haptic field components into a unified representation.
        
        Returns:
            Dictionary with all haptic field components
        """
        combined = {
            'pressure': self.pressure_field,
            'temperature': self.temperature_field,
            'force': self.force_field,
            'texture': self.texture_noise
        }
        
        return combined
    
    def sample_point(self, position: Tuple[float, float]) -> Dict[str, Any]:
        """
        Sample haptic data at a specific point.
        
        Args:
            position: Normalized (x, y) position to sample
            
        Returns:
            Dictionary with haptic properties at the specified point
        """
        # Convert to grid indices
        x = int(position[0] * self.resolution)
        y = int(position[1] * self.resolution)
        
        # Ensure within bounds
        x = max(0, min(x, self.resolution - 1))
        y = max(0, min(y, self.resolution - 1))
        
        # Sample all fields
        sample = {
            'pressure': float(self.pressure_field[y, x]),
            'temperature': float(self.temperature_field[y, x]),
            'force': self.force_field[y, x].tolist(),
            'texture': float(self.texture_noise[y, x])
        }
        
        return sample

class PerceptualBuffer:
    """
    A buffer that holds and integrates multi-sensory perceptual data.
    """
    
    def __init__(self, visual=None, auditory=None, tactile=None, olfactory=None, 
                taste=None, proprioception=None, electromagnetic=None, temporal=None):
        self.visual = visual
        self.auditory = auditory
        self.tactile = tactile
        self.olfactory = olfactory
        self.taste = taste
        self.proprioception = proprioception
        self.electromagnetic = electromagnetic
        self.temporal = temporal
        self.integration_weights = {
            'visual': 0.3,
            'auditory': 0.2,
            'tactile': 0.2,
            'olfactory': 0.1,
            'taste': 0.05,
            'proprioception': 0.05,
            'electromagnetic': 0.05,
            'temporal': 0.05
        }
    
    def integrate(self) -> Dict[str, Any]:
        """
        Integrate all sensory modalities into a unified perception.
        
        Returns:
            Integrated perception data
        """
        # Calculate which senses are active
        active_senses = {}
        for sense in self.integration_weights:
            if getattr(self, sense) is not None:
                active_senses[sense] = getattr(self, sense)
        
        if not active_senses:
            return {'coherence': 0.0, 'integrated': None}
        
        # Normalize weights for active senses
        active_weights = {sense: self.integration_weights[sense] for sense in active_senses}
        weight_sum = sum(active_weights.values())
        
        if weight_sum > 0:
            normalized_weights = {k: v/weight_sum for k, v in active_weights.items()}
        else:
            equal_weight = 1.0 / len(active_senses)
            normalized_weights = {k: equal_weight for k in active_senses}
        
        # Calculate cross-modal coherence
        coherence_matrix = self._calculate_coherence_matrix(active_senses)
        overall_coherence = np.mean(coherence_matrix) if coherence_matrix.size > 0 else 1.0
        
        # Return integrated result
        return {
            'modalities': active_senses,
            'weights': normalized_weights,
            'coherence': float(overall_coherence),
            'coherence_matrix': coherence_matrix.tolist() if coherence_matrix.size > 0 else [],
            'dominant_modality': max(normalized_weights.items(), key=lambda x: x[1])[0] if normalized_weights else None
        }
    
    def _calculate_coherence_matrix(self, active_senses: Dict[str, Any]) -> np.ndarray:
        """
        Calculate coherence between each pair of sensory modalities.
        
        Returns:
            Coherence matrix as numpy array
        """
        senses = list(active_senses.keys())
        n_senses = len(senses)
        
        if n_senses <= 1:
            return np.ones((1, 1))
        
        coherence = np.ones((n_senses, n_senses))
        
        # Calculate coherence for each pair
        for i in range(n_senses):
            for j in range(i+1, n_senses):
                sense1 = senses[i]
                sense2 = senses[j]
                
                # Calculate cross-modal coherence
                c = self._cross_modal_coherence(sense1, sense2)
                coherence[i, j] = c
                coherence[j, i] = c
        
        return coherence
    
    def _cross_modal_coherence(self, sense1: str, sense2: str) -> float:
        """
        Calculate coherence between two sensory modalities.
        Higher value means more consistent information across modalities.
        
        Returns:
            Coherence value between 0 and 1
        """
        # Default high coherence
        base_coherence = 0.8
        
        # In a full implementation, would compare actual data
        # For example, checking if visual object positions align with sounds
        
        # Basic implementation just returns fixed values for each combination
        coherence_map = {
            ('visual', 'auditory'): 0.7,
            ('visual', 'tactile'): 0.8,
            ('auditory', 'tactile'): 0.6,
            ('olfactory', 'taste'): 0.9,
            ('visual', 'olfactory'): 0.5,
            ('visual', 'electromagnetic'): 0.7,
            ('proprioception', 'tactile'): 0.9,
            ('temporal', 'auditory'): 0.8
        }
        
        # Get coherence value if defined, otherwise use default
        pair = tuple(sorted([sense1, sense2]))
        return coherence_map.get(pair, base_coherence)

class ChemicalPerceptionGenerator:
    """
    Generates chemical perception data for smell and taste.
    """
    
    def __init__(self, resolution: int = 16, compounds: List[str] = None):
        self.resolution = resolution
        self.compounds = compounds or [
            "sweet", "sour", "salty", "bitter", "umami",
            "floral", "fruity", "spicy", "earthy", "metallic",
            "woody", "herbal", "nutty", "pungent", "chemical"
        ]
        self.compound_fields = {c: np.zeros((resolution, resolution)) for c in self.compounds}
        self.diffusion_rates = {c: 0.1 + 0.3 * np.random.random() for c in self.compounds}
    
    def add_chemical_source(self, compound: str, position: Tuple[float, float], 
                          intensity: float = 1.0, radius: float = 0.2):
        """
        Add a chemical compound source at a specific position.
        
        Args:
            compound: Name of the chemical compound
            position: (x, y) normalized position
            intensity: Chemical intensity (0-1)
            radius: Spread radius
        """
        if compound not in self.compounds:
            return
            
        # Convert to grid coordinates
        x = int(position[0] * self.resolution)
        y = int(position[1] * self.resolution)
        
        # Ensure within bounds
        x = max(0, min(x, self.resolution - 1))
        y = max(0, min(y, self.resolution - 1))
        
        # Create Gaussian distribution around point
        sigma = radius * self.resolution
        for i in range(self.resolution):
            for j in range(self.resolution):
                distance = np.sqrt((i - y)**2 + (j - x)**2)
                contribution = intensity * np.exp(-(distance**2) / (2 * sigma**2))
                self.compound_fields[compound][i, j] += contribution
        
        # Clip values to 0-1 range
        self.compound_fields[compound] = np.clip(self.compound_fields[compound], 0, 1)
    
    def diffuse_chemicals(self, time_step: float = 0.1):
        """
        Simulate chemical diffusion over time.
        
        Args:
            time_step: Time step for diffusion simulation
        """
        # Simple diffusion kernel
        kernel = np.array([
            [0.05, 0.1, 0.05],
            [0.1, 0.4, 0.1],
            [0.05, 0.1, 0.05]
        ])
        
        for compound in self.compounds:
            rate = self.diffusion_rates[compound] * time_step
            
            # Apply convolution for diffusion
            from scipy.signal import convolve2d
            diffused = convolve2d(
                self.compound_fields[compound], 
                kernel, 
                mode='same', 
                boundary='symm'
            )
            
            # Mix original with diffused based on rate
            self.compound_fields[compound] = (
                (1 - rate) * self.compound_fields[compound] + rate * diffused
            )
            
            # Apply some decay
            self.compound_fields[compound] *= 0.99
    
    def sample_at_position(self, position: Tuple[float, float]) -> Dict[str, float]:
        """
        Sample chemical compounds at a specific position.
        
        Args:
            position: (x, y) normalized position
            
        Returns:
            Dictionary of compound intensities
        """
        # Convert to grid coordinates
        x = int(position[0] * self.resolution)
        y = int(position[1] * self.resolution)
        
        # Ensure within bounds
        x = max(0, min(x, self.resolution - 1))
        y = max(0, min(y, self.resolution - 1))
        
        # Sample all compounds
        result = {}
        for compound in self.compounds:
            intensity = float(self.compound_fields[compound][y, x])
            if intensity > 0.01:  # Only include detectable amounts
                result[compound] = intensity
                
        return result
    
    def get_dominant_compounds(self, threshold: float = 0.1) -> List[Tuple[str, float]]:
        """
        Get list of dominant compounds across the entire field.
        
        Args:
            threshold: Minimum intensity threshold
            
        Returns:
            List of (compound, intensity) tuples, sorted by intensity
        """
        result = []
        
        for compound in self.compounds:
            max_intensity = np.max(self.compound_fields[compound])
            if max_intensity > threshold:
                result.append((compound, float(max_intensity)))
                
        # Sort by intensity, highest first
        return sorted(result, key=lambda x: x[1], reverse=True)

class TastePerceptionGenerator(ChemicalPerceptionGenerator):
    """
    Specialized generator for taste perception.
    """
    
    def __init__(self, resolution: int = 8):
        super().__init__(resolution=resolution, compounds=[
            "sweet", "sour", "salty", "bitter", "umami",
            "creamy", "spicy", "astringent", "cooling", "metallic"
        ])
        # Taste-specific properties
        self.texture_field = np.zeros((resolution, resolution))
        self.temperature_field = np.ones((resolution, resolution)) * 0.5  # neutral temp
    
    def set_taste_texture(self, texture_type: str, intensity: float = 0.7):
        """
        Set texture component of taste.
        
        Args:
            texture_type: e.g., "smooth", "grainy", "fizzy"
            intensity: Texture intensity
        """
        # Different patterns for different textures
        if texture_type == "smooth":
            self.texture_field = np.ones((self.resolution, self.resolution)) * 0.1 * intensity
        elif texture_type == "grainy":
            self.texture_field = np.random.rand(self.resolution, self.resolution) * intensity
        elif texture_type == "fizzy":
            self.texture_field = (np.random.rand(self.resolution, self.resolution) > 0.7) * intensity
        else:
            # Default random texture
            self.texture_field = np.random.rand(self.resolution, self.resolution) * 0.5 * intensity
    
    def set_temperature(self, temperature: float):
        """
        Set temperature component of taste perception.
        
        Args:
            temperature: 0 (cold) to 1 (hot)
        """
        self.temperature_field = np.ones((self.resolution, self.resolution)) * temperature
    
    def get_taste_perception(self, position: Tuple[float, float] = None) -> Dict[str, Any]:
        """
        Get complete taste perception.
        
        Args:
            position: Optional position to sample, if None returns whole perception
            
        Returns:
            Dictionary with taste components
        """
        if position is not None:
            # Sample at specific position
            compounds = self.sample_at_position(position)
            
            # Convert to grid coordinates
            x = int(position[0] * self.resolution)
            y = int(position[1] * self.resolution)
            
            # Ensure within bounds
            x = max(0, min(x, self.resolution - 1))
            y = max(0, min(y, self.resolution - 1))
            
            texture = float(self.texture_field[y, x])
            temperature = float(self.temperature_field[y, x])
            
            return {
                'compounds': compounds,
                'texture': texture,
                'temperature': temperature
            }
        else:
            # Return whole field
            return {
                'compound_fields': {c: self.compound_fields[c].tolist() for c in self.compounds},
                'texture_field': self.texture_field.tolist(),
                'temperature_field': self.temperature_field.tolist(),
                'dominant_taste': self.get_dominant_compounds(threshold=0.2)
            }

class ProprioceptionGenerator:
    """
    Generates proprioception (sense of body position and movement) data.
    """
    
    def __init__(self, joint_count: int = 12, muscle_count: int = 24):
        self.joint_count = joint_count
        self.muscle_count = muscle_count
        
        # Joint positions in 3D space (normalized 0-1)
        self.joint_positions = np.random.rand(joint_count, 3) * 0.2 + 0.4  # centered positions
        
        # Joint angles (radians)
        self.joint_angles = np.zeros(joint_count)
        
        # Muscle tension (0-1)
        self.muscle_tension = np.zeros(muscle_count)
        
        # Muscle-joint connections (which muscles affect which joints)
        self.muscle_joint_map = {}
        for m in range(muscle_count):
            # Each muscle affects 1-3 joints
            joint_count = np.random.randint(1, 4)
            affected_joints = np.random.choice(joint_count, joint_count, replace=False)
            self.muscle_joint_map[m] = list(affected_joints)
        
        # Balance and orientation
        self.balance = 1.0  # 0 = unbalanced, 1 = perfectly balanced
        self.orientation = np.array([0.0, 0.0, 1.0])  # up vector
        
        # Movement and acceleration
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
    
    def update_joint_angles(self, joint_updates: Dict[int, float]):
        """
        Update angles for specific joints.
        
        Args:
            joint_updates: Dictionary mapping joint index to new angle
        """
        for idx, angle in joint_updates.items():
            if 0 <= idx < self.joint_count:
                self.joint_angles[idx] = angle
    
    def update_muscle_tension(self, muscle_updates: Dict[int, float]):
        """
        Update tension for specific muscles.
        
        Args:
            muscle_updates: Dictionary mapping muscle index to new tension
        """
        for idx, tension in muscle_updates.items():
            if 0 <= idx < self.muscle_count:
                self.muscle_tension[idx] = max(0.0, min(1.0, tension))
    
    def update_balance(self, balance_factor: float, orientation: List[float] = None):
        """
        Update balance and orientation.
        
        Args:
            balance_factor: Balance value (0-1)
            orientation: Optional new orientation vector
        """
        self.balance = max(0.0, min(1.0, balance_factor))
        
        if orientation is not None and len(orientation) == 3:
            # Normalize orientation vector
            norm = np.sqrt(sum(o*o for o in orientation))
            if norm > 0:
                self.orientation = np.array(orientation) / norm
    
    def update_motion(self, velocity: List[float] = None, acceleration: List[float] = None):
        """
        Update motion parameters.
        
        Args:
            velocity: Optional new velocity vector
            acceleration: Optional new acceleration vector
        """
        if velocity is not None and len(velocity) == 3:
            self.velocity = np.array(velocity)
            
        if acceleration is not None and len(acceleration) == 3:
            self.acceleration = np.array(acceleration)
    
    def integrate_movement(self, dt: float = 0.1):
        """
        Integrate movement over time step.
        
        Args:
            dt: Time step size
        """
        # Update velocity based on acceleration
        self.velocity += self.acceleration * dt
        
        # Update positions based on velocity
        self.joint_positions += np.array([self.velocity * dt for _ in range(self.joint_count)])
        
        # Keep joints within bounds
        self.joint_positions = np.clip(self.joint_positions, 0.0, 1.0)
        
        # Update angles based on muscle tension
        for m in range(self.muscle_count):
            tension = self.muscle_tension[m]
            for joint_idx in self.muscle_joint_map.get(m, []):
                # Simple model: tension affects joint angle
                angle_change = (tension - 0.5) * 0.1 * dt
                self.joint_angles[joint_idx] += angle_change
                
        # Simulate balance changes
        if np.random.random() < 0.1:  # Occasional balance perturbation
            self.balance *= 0.9 + 0.1 * np.random.random()
    
    def get_proprioception_data(self) -> Dict[str, Any]:
        """
        Get complete proprioception data.
        
        Returns:
            Dictionary with all proprioception components
        """
        return {
            'joint_positions': self.joint_positions.tolist(),
            'joint_angles': self.joint_angles.tolist(),
            'muscle_tension': self.muscle_tension.tolist(),
            'balance': float(self.balance),
            'orientation': self.orientation.tolist(),
            'velocity': self.velocity.tolist(),
            'acceleration': self.acceleration.tolist()
        }

class ElectromagneticPerceptionGenerator:
    """
    Generates electromagnetic perception data (magnetic fields, radiation, etc.)
    """
    
    def __init__(self, resolution: int = 16, field_dimensions: int = 3):
        self.resolution = resolution
        self.field_dimensions = field_dimensions
        
        # Magnetic field (3D vector field)
        self.magnetic_field = np.zeros((resolution, resolution, 3))
        
        # Electric field (3D vector field)
        self.electric_field = np.zeros((resolution, resolution, 3))
        
        # Radiation field (scalar field)
        self.radiation_field = np.zeros((resolution, resolution))
        
        # EM frequency bands (radio, microwave, IR, visible, UV, X-ray, gamma)
        self.frequency_bands = {
            'radio': np.zeros((resolution, resolution)),
            'microwave': np.zeros((resolution, resolution)),
            'infrared': np.zeros((resolution, resolution)),
            'visible': np.zeros((resolution, resolution)),
            'ultraviolet': np.zeros((resolution, resolution)),
            'xray': np.zeros((resolution, resolution)),
            'gamma': np.zeros((resolution, resolution))
        }
    
    def add_magnetic_dipole(self, position: Tuple[float, float], 
                          moment: Tuple[float, float, float],
                          strength: float = 1.0):
        """
        Add a magnetic dipole at a specific position.
        
        Args:
            position: (x, y) normalized position
            moment: (mx, my, mz) dipole moment vector
            strength: Field strength multiplier
        """
        # Convert to grid coordinates
        center_x = int(position[0] * self.resolution)
        center_y = int(position[1] * self.resolution)
        
        # Normalize moment vector
        moment_norm = np.sqrt(sum(m*m for m in moment))
        if moment_norm > 0:
            moment = tuple(m/moment_norm for m in moment)
        
        # Generate grid coordinates
        x_coords, y_coords = np.meshgrid(
            np.arange(self.resolution),
            np.arange(self.resolution)
        )
        
        # Calculate distance vectors
        r_x = (x_coords - center_x) / self.resolution
        r_y = (y_coords - center_y) / self.resolution
        
        # Calculate distance from center
        r_squared = r_x**2 + r_y**2
        r_squared = np.maximum(r_squared, 0.0001)  # Avoid division by zero
        r = np.sqrt(r_squared)
        
        # Calculate dipole field at each point
        field = np.zeros((self.resolution, self.resolution, 3))
        
        # Simplified dipole field calculation
        for i in range(self.resolution):
            for j in range(self.resolution):
                if i == center_y and j == center_x:
                    continue  # Skip the dipole position
                    
                # Direction vector
                rx = (j - center_x) / self.resolution
                ry = (i - center_y) / self.resolution
                r_mag = np.sqrt(rx**2 + ry**2)
                
                if r_mag < 0.001:
                    continue  # Too close to source
                    
                # Unit direction vector
                rx /= r_mag
                ry /= r_mag
                
                # 3D position (z=0 plane)
                r_vec = [rx, ry, 0]
                
                # Dipole field calculation
                dot_product = sum(m*r for m, r in zip(moment, r_vec))
                
                for d in range(3):
                    field[i, j, d] = (3 * r_vec[d] * dot_product - moment[d]) / (r_mag**3)
        
        # Scale field and add to existing field
        field *= strength
        self.magnetic_field += field
    
    def add_electric_charge(self, position: Tuple[float, float], charge: float):
        """
        Add an electric charge at a specific position.
        
        Args:
            position: (x, y) normalized position
            charge: Charge value (positive or negative)
        """
        # Convert to grid coordinates
        center_x = int(position[0] * self.resolution)
        center_y = int(position[1] * self.resolution)
        
        # Generate grid coordinates
        x_coords, y_coords = np.meshgrid(
            np.arange(self.resolution),
            np.arange(self.resolution)
        )
        
        # Calculate distance vectors
        r_x = (x_coords - center_x) / self.resolution
        r_y = (y_coords - center_y) / self.resolution
        
        # Calculate distance from center
        r_squared = r_x**2 + r_y**2
        r_squared = np.maximum(r_squared, 0.0001)  # Avoid division by zero
        r = np.sqrt(r_squared)
        
        # Calculate Coulomb field at each point
        field = np.zeros((self.resolution, self.resolution, 3))
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                if i == center_y and j == center_x:
                    continue  # Skip the charge position
                
                # Direction vector
                rx = (j - center_x) / self.resolution
                ry = (i - center_y) / self.resolution
                r_mag = np.sqrt(rx**2 + ry**2)
                
                if r_mag < 0.001:
                    continue  # Too close to source
                
                # Electric field points away from positive charge
                field[i, j, 0] = rx / r_mag / r_squared
                field[i, j, 1] = ry / r_mag / r_squared
                field[i, j, 2] = 0  # Planar field
        
        # Scale by charge (sign determines direction)
        field *= charge
        
        # Add to existing field
        self.electric_field += field
    
    def add_radiation_source(self, position: Tuple[float, float], 
                            intensity: float = 1.0,
                            spectrum: Dict[str, float] = None):
        """
        Add a radiation source at a specific position.
        
        Args:
            position: (x, y) normalized position
            intensity: Overall intensity
            spectrum: Intensity distribution across frequency bands
        """
        # Default uniform spectrum
        if spectrum is None:
            spectrum = {band: 1.0 for band in self.frequency_bands}
        
        # Convert to grid coordinates
        x = int(position[0] * self.resolution)
        y = int(position[1] * self.resolution)
        
        # Add radiation to scalar field with inverse square falloff
        for i in range(self.resolution):
            for j in range(self.resolution):
                distance = np.sqrt(((i - y) / self.resolution)**2 + 
                                  ((j - x) / self.resolution)**2)
                
                if distance < 0.001:
                    # Avoid division by zero at source
                    value = intensity * 10
                else:
                    # Inverse square law
                    value = intensity / (distance**2)
                
                # Add to radiation field
                self.radiation_field[i, j] += value
                
                # Add to each frequency band based on spectrum
                for band, band_intensity in spectrum.items():
                    if band in self.frequency_bands:
                        self.frequency_bands[band][i, j] += value * band_intensity
        
        # Clip values
        self.radiation_field = np.clip(self.radiation_field, 0, 10)
        for band in self.frequency_bands:
            self.frequency_bands[band] = np.clip(self.frequency_bands[band], 0, 10)
    
    def get_electromagnetic_perception(self, position: Tuple[float, float] = None) -> Dict[str, Any]:
        """
        Get complete electromagnetic perception.
        
        Args:
            position: Optional position to sample, if None returns whole perception
            
        Returns:
            Dictionary with electromagnetic components
        """
        if position is not None:
            # Convert to grid coordinates
            x = int(position[0] * self.resolution)
            y = int(position[1] * self.resolution)
            
            # Ensure within bounds
            x = max(0, min(x, self.resolution - 1))
            y = max(0, min(y, self.resolution - 1))
            
            # Sample at specific position
            sample = {
                'magnetic': self.magnetic_field[y, x].tolist(),
                'electric': self.electric_field[y, x].tolist(),
                'radiation': float(self.radiation_field[y, x]),
                'frequencies': {band: float(self.frequency_bands[band][y, x]) 
                               for band in self.frequency_bands}
            }
            return sample
        else:
            # Return whole field (could be large - might want to downsample)
            return {
                'magnetic_field': self.magnetic_field.tolist(),
                'electric_field': self.electric_field.tolist(),
                'radiation_field': self.radiation_field.tolist(),
                'frequency_bands': {band: self.frequency_bands[band].tolist() 
                                   for band in self.frequency_bands}
            }

class TemporalPerceptionGenerator:
    """
    Generates perception of time, including flow rate, continuity, and temporal anomalies.
    """
    
    def __init__(self, history_length: int = 200):
        self.history_length = history_length
        self.current_time = 0.0
        self.flow_rate = 1.0  # 1.0 = normal time flow
        self.previous_timestamps = np.zeros(history_length)
        self.time_dilation = 1.0
        self.continuity = 1.0  # 1.0 = perfectly continuous
        self.temporal_direction = 1  # 1 = forward, -1 = backward
        self.temporal_loops = []  # Timestamps where loops occur
        self.jitter = 0.0  # Random variation in time flow
        self.perceived_duration = 1.0  # Subjective duration of events
    
    def update(self, delta_t: float, universal_time: float):
        """
        Update temporal perception.
        
        Args:
            delta_t: Objective time step
            universal_time: Current universal timeline value
        """
        # Apply time dilation
        dilated_dt = delta_t * self.time_dilation
        
        # Apply flow rate and direction
        adjusted_dt = dilated_dt * self.flow_rate * self.temporal_direction
        
        # Add jitter if present
        if self.jitter > 0:
            jitter_amount = (np.random.random() - 0.5) * self.jitter * dilated_dt
            adjusted_dt += jitter_amount
        
        # Update current perceived time
        self.current_time += adjusted_dt
        
        # Update timestamp history
        self.previous_timestamps = np.roll(self.previous_timestamps, 1)
        self.previous_timestamps[0] = universal_time
        
        # Check for temporal loops
        self._check_for_loops()
    
    def set_flow_rate(self, rate: float):
        """
        Set the rate of time flow.
        
        Args:
            rate: Flow rate (1.0 = normal, <1 = slower, >1 = faster)
        """
        self.flow_rate = max(0.01, rate)  # Prevent near-zero or negative flow rates
    
    def set_time_dilation(self, dilation: float):
        """
        Set time dilation factor.
        
        Args:
            dilation: Dilation factor (1.0 = normal, <1 = compressed, >1 = expanded)
        """
        self.time_dilation = max(0.01, dilation)
    
    def set_continuity(self, continuity: float):
        """
        Set time continuity.
        
        Args:
            continuity: Continuity factor (1.0 = perfect, 0.0 = fragmented)
        """
        self.continuity = max(0.0, min(1.0, continuity))
        
        # Higher jitter for lower continuity
        self.jitter = (1.0 - self.continuity) * 0.5
    
    def reverse_time(self, duration: float = None):
        """
        Reverse the flow of time.
        
        Args:
            duration: Optional duration of reversal (seconds)
        """
        self.temporal_direction = -1
        
        # Store loop point
        self.temporal_loops.append(self.current_time)
        
        # Set up automatic reversion if duration specified
        if duration is not None:
            # In an actual implementation, would set up a timer to revert
            pass
    
    def restore_forward_time(self):
        """Restore normal forward time flow."""
        self.temporal_direction = 1
    
    def _check_for_loops(self, similarity_threshold: float = 0.95):
        """
        Check for temporal loops in recent history.
        
        Args:
            similarity_threshold: Threshold for detecting similar patterns
        """
        # Need enough history to detect patterns
        if len(self.previous_timestamps) < 20:
            return
            
        # Check various window sizes for repeating patterns
        for window_size in [5, 10, 15]:
            if len(self.previous_timestamps) < window_size * 2:
                continue
                
            # Get recent windows
            recent_window = self.previous_timestamps[:window_size]
            previous_window = self.previous_timestamps[window_size:window_size*2]
            
            # Calculate similarity (correlation)
            similarity = np.corrcoef(recent_window, previous_window)[0, 1]
            
            if similarity > similarity_threshold:
                # Detected a loop
                if self.current_time not in self.temporal_loops:
                    self.temporal_loops.append(self.current_time)
                    
                # Reduce continuity temporarily
                self.continuity *= 0.9
                return
    
    def get_temporal_perception(self) -> Dict[str, Any]:
        """
        Get complete temporal perception data.
        
        Returns:
            Dictionary with temporal perception components
        """
        return {
            'current_time': float(self.current_time),
            'flow_rate': float(self.flow_rate),
            'time_dilation': float(self.time_dilation),
            'continuity': float(self.continuity),
            'direction': int(self.temporal_direction),
            'loops_detected': len(self.temporal_loops) > 0,
            'recent_loops': [float(t) for t in self.temporal_loops[-5:]] if self.temporal_loops else [],
            'perceived_duration': float(self.perceived_duration),
            'temporal_stability': float(1.0 - self.jitter),
            'flow_consistency': float(self._calculate_flow_consistency())
        }
    
    def _calculate_flow_consistency(self) -> float:
        """
        Calculate consistency of time flow from timestamp history.
        
        Returns:
            Consistency value between 0 and 1
        """
        if len(self.previous_timestamps) < 3:
            return 1.0
            
        # Calculate time deltas
        deltas = np.diff(self.previous_timestamps[:20])
        
        if len(deltas) == 0 or np.mean(deltas) == 0:
            return 1.0
            
        # Calculate coefficient of variation
        cv = np.std(deltas) / np.mean(deltas)
        
        # Convert to consistency score (lower cv = higher consistency)
        consistency = 1.0 / (1.0 + cv)
        
        return consistency

def initialize(**kwargs):
    """
    Initialize the perception module and return a PerceptionIntegrator instance.
    
    Args:
        **kwargs: Configuration parameters for the perception module, including:
            - entity_id: The ID of the entity using this perception system
            - perception_channels: Optional dictionary of perception channel definitions
            - identity_matrix: Optional predefined identity matrix
            - memory_configuration: Optional memory configuration
            
    Returns:
        PerceptionIntegrator instance that was initialized
    """
    logger = logging.getLogger("PerceptionModule")
    logger.info("Initializing Perception Module...")
    
    # Extract entity ID from kwargs
    entity_id = kwargs.get('entity_id', f"entity_{hash(str(kwargs)) % 10000}")
    
    # Create a new PerceptionIntegrator instance
    perception_instance = PerceptionIntegrator(entity_id=entity_id)
    
    # Setup additional configurations if provided
    perception_channels = kwargs.get('perception_channels', {})
    if perception_channels:
        logger.info(f"Configuring perception channels: {list(perception_channels.keys())}")
        for channel_name, config in perception_channels.items():
            # Apply channel configuration
            pass
    
    # Setup custom identity matrix if provided
    identity_matrix = kwargs.get('identity_matrix', None)
    if identity_matrix:
        logger.info(f"Setting custom identity matrix for {entity_id}")
        # Apply identity matrix settings to perception_instance.identity
        for attribute, value in identity_matrix.items():
            if hasattr(perception_instance.identity, attribute):
                setattr(perception_instance.identity, attribute, value)
    
    # Setup memory configuration if provided
    memory_config = kwargs.get('memory_configuration', None)
    if memory_config:
        logger.info(f"Configuring memory for {entity_id}")
        if hasattr(perception_instance, 'memory') and hasattr(perception_instance.memory, 'capacity'):
            perception_instance.memory.capacity = memory_config.get('capacity', perception_instance.memory.capacity)
            perception_instance.memory.decay_rate = memory_config.get('decay_rate', perception_instance.memory.decay_rate)
            perception_instance.memory.integration_threshold = memory_config.get('threshold', perception_instance.memory.integration_threshold)
    
    logger.info(f"Perception Module initialization complete for entity {entity_id}")
    return perception_instance

__all__ = ["PerceptionProcessor", "PerceptionIntegrator", "TemporalEvent", "TemporalBranch", "TemporalPerception"]