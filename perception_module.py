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

class ArchetypeResonance(Enum):
    """Fundamental patterns of being that shape perception."""
    ANIMA = "anima"         # Life-bringing, nurturing, creative
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