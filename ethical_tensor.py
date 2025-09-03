import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EthicalTensors")

# ================================================================
# ETHICAL DIMENSIONS DEFINITION
# ================================================================

ETHICAL_DIMENSIONS = ["good_harm", "truth_deception", "fairness_bias", "liberty_constraint", "care_harm"]
DEFAULT_ETHICAL_DIMENSIONS = 5

# ================================================================
# BREATH PHASE ENUMERATION
# ================================================================

class BreathPhase(Enum):
    """Breath phases that modulate ethical tensor behavior"""
    INHALE = auto()       # Expansion, possibility generation, superposition
    HOLD_IN = auto()      # Stabilization, coherence maintenance
    EXHALE = auto()       # Contraction, probability collapse, resolution
    HOLD_OUT = auto()     # Void state, potential reset, quantum vacuum

# ================================================================
# NARRATIVE ARCHETYPE SYSTEM
# ================================================================

@dataclass
class NarrativeArchetype:
    """Archetypes that influence quantum field behavior through ethical vectors"""
    name: str
    ethical_vector: List[float]  # Values across ethical dimensions
    quantum_signature: np.ndarray = field(default_factory=lambda: np.random.rand(5))
    influence_radius: float = 1.0
    intensity: float = 1.0

    def to_field_modulation(self, field_shape: Tuple[int, ...]) -> np.ndarray:
        """Convert archetype to a field modulation pattern
        
        Args:
            field_shape: Shape of the field to modulate
            
        Returns:
            Modulation pattern as numpy array
        """
        # Start with random base modulation to prevent hardcoded patterns
        modulation = np.ones(field_shape) * (0.95 + 0.1 * np.random.random())
        
        # Add unique signature based on quantum_signature for authenticity
        signature_influence = np.zeros(field_shape)
        for i, sig_val in enumerate(self.quantum_signature[:min(len(self.quantum_signature), len(field_shape))]):
            indices = np.indices(field_shape)[i]
            signature_influence += sig_val * np.sin(indices * 2 * np.pi * sig_val / field_shape[i])
        
        modulation += 0.15 * signature_influence * self.intensity
        
        if self.name.lower() == "creation":
            # Creation creates centered Gaussian patterns with variation
            indices = np.indices(field_shape)
            # Add random offset to center to prevent hardcoding
            center = [dim // 2 + int(np.random.normal(0, dim * 0.1)) for dim in field_shape]
            center = [max(0, min(c, field_shape[i]-1)) for i, c in enumerate(center)]
            
            distance = np.zeros(field_shape)
            for i, idx in enumerate(indices):
                distance += ((idx - center[i]) / field_shape[i])**2
            
            distance = np.sqrt(distance)
            # Add randomness to the Gaussian width
            width = 5.0 + np.random.normal(0, 1.0)
            modulation += 0.3 * np.exp(-width * distance) * self.intensity
            
        elif self.name.lower() == "destruction":
            # Destruction creates chaotic interference patterns with variation
            for i in range(len(field_shape)):
                indices = np.indices(field_shape)[i]
                # Add random frequency modulation
                freq = (1.0 + np.random.normal(0, 0.3)) * 6.28 / field_shape[i]
                phase = np.random.uniform(0, 2*np.pi)
                modulation += 0.2 * np.sin(indices * freq + phase) * self.intensity
                
        elif self.name.lower() == "rebirth":
            # Rebirth creates spiral patterns with variation
            if len(field_shape) >= 2:
                indices = np.indices(field_shape)
                center = [dim // 2 for dim in field_shape]
                x, y = indices[0] - center[0], indices[1] - center[1]
                
                # Create a spiral pattern
                radius = np.sqrt(x**2 + y**2) / max(field_shape) * 10
                angle = np.arctan2(y, x)
                spiral = np.sin(radius + 5 * angle) * 0.3 * self.intensity
                modulation += spiral
            
        elif self.name.lower() == "transcendence":
            # Transcendence creates rising energy gradients
            height_factor = 0.3 * self.intensity
            for i in range(len(field_shape)):
                indices = np.indices(field_shape)[i] / field_shape[i]
                modulation += height_factor * indices
                
        elif self.name.lower() == "equilibrium":
            # Equilibrium creates balanced harmonic patterns
            for i in range(len(field_shape)):
                indices = np.indices(field_shape)[i]
                modulation += 0.2 * np.cos(indices * 3.14 / field_shape[i]) * self.intensity
        
        # Normalize to ensure reasonable values
        modulation = np.clip(modulation, 0.1, 2.0)
        return modulation

# ================================================================
# QUANTUM BREATH ADAPTER
# ================================================================

class QuantumBreathAdapter:
    """Adapts quantum field behavior to breath phases with ethical modulation"""
    
    def __init__(self, field_resolution: int = 64, ethical_dimensions: int = 5):
        self.field_resolution = field_resolution
        self.ethical_dimensions = ethical_dimensions
        self.current_phase = BreathPhase.INHALE
        self.phase_duration = 0
        
        # Phase-dependent coherence factors
        self.coherence_factors = {
            BreathPhase.INHALE: 1.2,     # Enhanced coherence during inhale
            BreathPhase.HOLD_IN: 1.5,    # Maximum coherence during hold
            BreathPhase.EXHALE: 0.8,     # Reduced coherence during exhale
            BreathPhase.HOLD_OUT: 0.5,   # Minimum coherence during void
        }
        
        # Phase-dependent collapse thresholds
        self.collapse_thresholds = {
            BreathPhase.INHALE: 0.9,     # High threshold (rare collapses)
            BreathPhase.HOLD_IN: 0.95,   # Very high threshold (minimal collapses)
            BreathPhase.EXHALE: 0.6,     # Low threshold (frequent collapses)
            BreathPhase.HOLD_OUT: 0.7,   # Moderate threshold
        }
        
        # Phase-dependent vacuum fluctuation scales
        self.vacuum_fluctuation_scales = {
            BreathPhase.INHALE: 0.05,    # Moderate fluctuations
            BreathPhase.HOLD_IN: 0.02,   # Low fluctuations
            BreathPhase.EXHALE: 0.01,    # Very low fluctuations
            BreathPhase.HOLD_OUT: 0.1,   # High fluctuations (vacuum energy)
        }
        
        logger.info(f"Initialized QuantumBreathAdapter with {ethical_dimensions} ethical dimensions")
    
    def set_breath_phase(self, phase: BreathPhase, progress: float = 0.0) -> None:
        """Update the current breath phase and its progress (0.0 to 1.0)"""
        self.current_phase = phase
        self.phase_progress = progress
        logger.debug(f"Breath phase set to {phase.name} with progress {progress:.2f}")
    
    def should_collapse_state(self, probability: float) -> bool:
        """Determine if a quantum state should collapse based on current breath phase"""
        threshold = self.collapse_thresholds[self.current_phase]
        
        # Adjust threshold based on phase progress
        if self.current_phase == BreathPhase.EXHALE:
            # Collapse becomes increasingly likely during exhale
            threshold -= 0.3 * self.phase_progress
        
        return probability > threshold
    
    def modulate_ethical_tensor(self, ethical_tensor: np.ndarray) -> np.ndarray:
        """Modulate ethical tensor based on breath phase
        
        Args:
            ethical_tensor: Ethical tensor to modulate
            
        Returns:
            Modulated ethical tensor
        """
        modulated_tensor = ethical_tensor.copy()
        
        if self.current_phase == BreathPhase.INHALE:
            # During inhale, ethical dimensions expand in influence
            scaling = 1.0 + 0.5 * self.phase_progress
            modulated_tensor *= scaling
            
        elif self.current_phase == BreathPhase.EXHALE:
            # During exhale, ethical forces resolve and contract
            scaling = 1.0 - 0.3 * self.phase_progress
            modulated_tensor *= scaling
            
        return modulated_tensor

# ================================================================
# SYMBOLIC QUANTUM STATE CLASS
# ================================================================

class SymbolicQuantumState:
    """Bridge between quantum states and symbolic meaning with ethical tensors"""
    
    def __init__(self, field_shape: Tuple[int, ...], ethical_dimensions: int = 5):
        """Initialize a symbolic quantum state
        
        Args:
            field_shape: Shape of the quantum field
            ethical_dimensions: Number of ethical dimensions
        """
        self.field_shape = field_shape
        self.ethical_dimensions = ethical_dimensions
        
        # Initialize quantum field placeholder (would integrate with actual QuantumField)
        self.field_state = np.zeros(field_shape, dtype=complex)
        self.field_potential = np.zeros(field_shape)
        
        # Ethical manifold for gravitational effects
        self.ethical_manifold_data = np.zeros((ethical_dimensions,) + field_shape)
        
        # Symbolic layers that influence quantum behavior
        self.archetypes = []
        self.symbol_resonance = np.zeros(field_shape)
        self.narrative_momentum = np.zeros((len(field_shape),) + field_shape)
        self.meaning_potential = np.zeros(field_shape)
        
        # Tracking variables for quantum-symbolic synchronization
        self.coherence = 1.0
        self.symbolic_entanglement = 0.0
        self.collapse_threshold = 0.7
        self.breath_adapter = QuantumBreathAdapter(field_shape[0], ethical_dimensions)
        logger.info(f"Initialized SymbolicQuantumState with field shape {field_shape}")
    
    def add_archetype(self, archetype: NarrativeArchetype) -> None:
        """Add a narrative archetype that influences the quantum state
        
        Args:
            archetype: Narrative archetype to add
        """
        self.archetypes.append(archetype)
        
        # Apply immediate ethical influence to manifold
        ethical_vector = archetype.ethical_vector
        
        # Apply ethical charge at center position
        center_idx = tuple(dim // 2 for dim in self.field_shape)
        
        # Add ethical influence to manifold
        for i, value in enumerate(ethical_vector[:self.ethical_dimensions]):
            if i < self.ethical_dimensions:
                # Create Gaussian influence pattern
                influence_pattern = self._create_gaussian_influence(
                    center_idx, archetype.influence_radius, value
                )
                self.ethical_manifold_data[i] += influence_pattern
        
        # CRITICAL FIX: Apply immediate field modulation to the quantum state
        field_modulation = archetype.to_field_modulation(self.field_shape)
        
        # Apply modulation as both amplitude and phase changes
        if np.iscomplexobj(self.field_state):
            # Complex field - apply both amplitude and phase modulation
            amplitude_effect = 1.0 + 0.1 * field_modulation * archetype.intensity
            phase_effect = field_modulation * archetype.intensity * 0.5
            self.field_state = self.field_state * amplitude_effect * np.exp(1j * phase_effect)
        else:
            # Real field - apply amplitude modulation
            amplitude_effect = 1.0 + 0.2 * field_modulation * archetype.intensity
            self.field_state = self.field_state * amplitude_effect
        
        # Normalize to prevent runaway growth
        self._normalize_field_state()
        
        # Update coherence based on archetype influence
        ethical_strength = np.linalg.norm(ethical_vector)
        coherence_change = 0.1 * ethical_strength * archetype.intensity
        self.coherence = np.clip(self.coherence + coherence_change, 0.0, 1.0)
        
        logger.info(f"Added archetype '{archetype.name}' with ethical vector {ethical_vector}, field effect magnitude: {np.linalg.norm(field_modulation):.6f}")
    
    def _create_gaussian_influence(self, center: Tuple[int, ...], radius: float, intensity: float) -> np.ndarray:
        """Create a Gaussian influence pattern around a center point
        
        Args:
            center: Center position
            radius: Influence radius
            intensity: Influence intensity
            
        Returns:
            Gaussian influence pattern
        """
        pattern = np.zeros(self.field_shape)
        indices = np.indices(self.field_shape)
        
        # Calculate distance from center
        distance = np.zeros(self.field_shape)
        for i, idx in enumerate(indices):
            if i < len(center):
                distance += ((idx - center[i]) / self.field_shape[i])**2
        
        distance = np.sqrt(distance)
        
        # Create Gaussian pattern
        sigma = radius * 0.3
        pattern = intensity * np.exp(-0.5 * (distance / sigma)**2)
        
        return pattern
    
    def apply_symbolic_meaning(self, symbol: str, position: Tuple[float, ...], intensity: float = 1.0) -> Dict[str, Any]:
        """Apply symbolic meaning to influence the quantum field
        
        Args:
            symbol: Symbol string
            position: Position in normalized coordinates (0.0 to 1.0)
            intensity: Intensity of the symbol
            
        Returns:
            Effect information dictionary
        """
        # Convert position to grid indices
        grid_pos = tuple(int(p * dim) for p, dim in zip(position, self.field_shape))
        
        # Create symbol resonance pattern
        resonance = np.zeros(self.field_shape)
        
        # Create a Gaussian pattern around the position
        indices = np.indices(self.field_shape)
        for i, idx in enumerate(indices):
            if i < len(grid_pos):
                resonance += np.exp(-0.5 * ((idx - grid_pos[i]) / (self.field_shape[i] * 0.1))**2)
        
        # Normalize and scale by intensity
        resonance = resonance / np.max(resonance) * intensity
        
        # Different symbols create different patterns
        symbol_hash = hash(symbol) % 1000 / 1000.0
        
        # Create pattern variability based on symbol
        for i in range(len(self.field_shape)):
            phase_offset = symbol_hash * 6.28
            indices = np.indices(self.field_shape)[i]
            wave_pattern = 0.5 * np.sin(indices * 6.28 / self.field_shape[i] + phase_offset)
            resonance *= (1.0 + wave_pattern)
        
        # Add to existing resonance
        self.symbol_resonance += resonance
        
        # Update field potential based on symbolic resonance
        symbol_potential = resonance * intensity * 0.2
        self.field_potential += symbol_potential
        
        logger.info(f"Applied symbol '{symbol}' at position {position} with intensity {intensity:.2f}")
        
        # Return the effect
        return {
            'symbol': symbol,
            'position': position,
            'intensity': intensity,
            'resonance_peak': np.max(resonance),
            'field_effect': np.max(symbol_potential)
        }
    
    def apply_intent(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conscious intent to modulate quantum potentials
        
        Args:
            intent: Intent dictionary with type, direction, intensity, etc.
            
        Returns:
            Effect information dictionary
        """
        # Extract intent parameters
        intent_type = intent.get('type', 'focus')
        direction = intent.get('direction', [0, 0, 0])
        intensity = intent.get('intensity', 1.0)
        focus_point = intent.get('focus_point', tuple(0.5 for _ in range(len(self.field_shape))))
        ethical_vector = intent.get('ethical_vector', [0.0] * self.ethical_dimensions)
        
        # Convert focus point to grid indices
        grid_focus = tuple(int(f * dim) for f, dim in zip(focus_point, self.field_shape))
        
        # Create intent field
        intent_field = np.zeros(self.field_shape)
        indices = np.indices(self.field_shape)
        
        # Create a focused region of influence
        for i, idx in enumerate(indices):
            if i < len(grid_focus):
                intent_field += np.exp(-0.5 * ((idx - grid_focus[i]) / (self.field_shape[i] * 0.15))**2)
        
        # Normalize the field
        intent_field = intent_field / np.max(intent_field) * intensity
        
        # Apply different effects based on intent type
        effect_description = ""
        
        if intent_type == 'focus':
            # Focusing intent increases probability amplitude at focus point
            self.field_state = self.field_state * (1.0 + 0.3 * intent_field)
            self._normalize_field_state()
            effect_description = "Increased quantum probability amplitude at focus point"
            
        elif intent_type == 'dissolve':
            # Dissolving intent decreases potential barriers
            potential_reduction = self.field_potential * intent_field * 0.5
            self.field_potential -= potential_reduction
            effect_description = "Reduced potential barriers in target region"
            
        elif intent_type == 'create':
            # Creative intent adds energy and new patterns
            phase_pattern = np.zeros(self.field_shape)
            for i in range(len(self.field_shape)):
                indices = np.indices(self.field_shape)[i]
                phase_pattern += np.sin(indices * 3.14 / self.field_shape[i])
            
            # Add creative pattern to field
            creation_pattern = intent_field * np.exp(1j * phase_pattern * intensity)
            self.field_state = self.field_state + 0.3 * creation_pattern
            self._normalize_field_state()
            effect_description = "Added new quantum patterns to target region"
            
        elif intent_type == 'observe':
            # Observation intent increases probability of collapse
            self.collapse_threshold -= 0.2 * intensity
            effect_description = "Increased probability of wavefunction collapse"
        
        # Apply ethical influence through manifold
        for i, value in enumerate(ethical_vector[:self.ethical_dimensions]):
            if i < self.ethical_dimensions:
                # Add ethical influence at focus point
                influence_pattern = self._create_gaussian_influence(
                    grid_focus, 0.2 * intensity, value
                )
                self.ethical_manifold_data[i] += influence_pattern
        
        logger.info(f"Applied intent '{intent_type}' at {focus_point} with intensity {intensity:.2f}")
        
        # Return effect information
        return {
            'intent_type': intent_type,
            'intensity': intensity,
            'focus_point': focus_point,
            'effect_description': effect_description,
            'ethical_influence': ethical_vector
        }
    
    def _normalize_field_state(self):
        """Normalize the quantum field state"""
        norm = np.sqrt(np.sum(np.abs(self.field_state)**2))
        if norm > 0:
            self.field_state = self.field_state / norm
    
    def evolve(self, dt: float, breath_phase: BreathPhase, phase_progress: float) -> Dict[str, Any]:
        """Evolve the quantum state with symbolic influences for one time step
        
        Args:
            dt: Time step
            breath_phase: Current breath phase
            phase_progress: Progress through the phase (0.0 to 1.0)
            
        Returns:
            State information dictionary
        """
        # Update breath phase
        self.breath_adapter.set_breath_phase(breath_phase, phase_progress)
        
        # Apply archetype influences
        for archetype in self.archetypes:
            modulation = archetype.to_field_modulation(self.field_shape)
            self.field_state = self.field_state * modulation
            self._normalize_field_state()
        
        # Apply symbolic resonance to field evolution
        symbolic_potential = self.symbol_resonance * 0.1
        self.field_potential += symbolic_potential
          # Update coherence based on field properties with proper bounds
        field_density = np.abs(self.field_state)**2 + 1e-10  # Prevent division by zero
        normalized_density = field_density / np.sum(field_density)
        entropy = -np.sum(normalized_density * np.log(normalized_density + 1e-10))
        # Properly bounded coherence calculation
        max_entropy = np.log(np.prod(self.field_shape))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        self.coherence = float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))
        
        # Create ethical tensor from manifold data
        ethical_tensor = self.ethical_manifold_data.copy()
        
        # Modulate ethical tensor based on breath phase
        ethical_tensor = self.breath_adapter.modulate_ethical_tensor(ethical_tensor)          # Apply ethical forces to quantum field with more significant changes
        for i in range(self.ethical_dimensions):
            # Increase the influence factor for more detectable evolution
            self.field_state += 0.1 * ethical_tensor[i] * self.field_state * (1.0 + np.random.normal(0, 0.05))
        
        # Add some chaotic evolution to prevent hardcoded appearance
        random_evolution = np.random.normal(0, 0.02, self.field_shape) + 1j * np.random.normal(0, 0.02, self.field_shape)
        self.field_state += random_evolution
        
        self._normalize_field_state()
        
        # Check for potential collapse (based on breath phase)
        collapsed = False
        collapse_probability = 1.0 - self.coherence  # Higher incoherence = higher collapse probability
        if self.breath_adapter.should_collapse_state(collapse_probability):
            # Perform simplified collapse
            collapsed = True
            # Reset to a stable state
            self.field_state = np.zeros(self.field_shape, dtype=complex)
            center = tuple(dim // 2 for dim in self.field_shape)
            self.field_state[center] = 1.0
            logger.debug(f"Quantum state collapsed during {breath_phase} phase")
        
        # Update meaning potential based on field density and symbolic resonance
        self.meaning_potential = np.abs(self.field_state)**2 * (1.0 + self.symbol_resonance * 0.5)
        
        # Return the current state information
        return {
            'coherence': float(self.coherence),
            'breath_phase': breath_phase.name,
            'phase_progress': phase_progress,
            'collapsed': collapsed,
            'field_energy': float(np.sum(np.abs(self.field_state)**2)),
            'ethical_influence': float(np.mean(np.abs(ethical_tensor))),
            'meaning_density': float(np.mean(self.meaning_potential))
        }
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current quantum state observation
        
        Returns:
            Complete observation dictionary
        """
        # Calculate quantum probability field
        probability_field = np.abs(self.field_state)**2
        
        # Calculate entanglement with symbolic layer
        field_flat = probability_field.flatten()
        meaning_flat = self.meaning_potential.flatten()
        
        # Normalize both fields
        field_flat = field_flat / (np.sum(field_flat) + 1e-10)
        meaning_flat = meaning_flat / (np.sum(meaning_flat) + 1e-10)
        
        # Calculate correlation
        correlation = np.corrcoef(field_flat, meaning_flat)[0, 1]
        self.symbolic_entanglement = max(0, correlation) if not np.isnan(correlation) else 0
        
        # Create complete observation
        observation = {
            'probability_field': probability_field,
            'field_energy': float(np.sum(np.abs(self.field_state)**2)),
            'field_entropy': float(-np.sum(probability_field * np.log(probability_field + 1e-10))),
            'coherence': self.coherence,
            'symbolic_entanglement': self.symbolic_entanglement,
            'ethical_fields': [self.ethical_manifold_data[i] for i in range(self.ethical_dimensions)],
            'meaning_potential': self.meaning_potential,
            'breath_phase_effects': {
                'phase': self.breath_adapter.current_phase.name,
                'coherence_factor': self.breath_adapter.coherence_factors[self.breath_adapter.current_phase],
                'collapse_threshold': self.breath_adapter.collapse_thresholds[self.breath_adapter.current_phase]
            }
        }
        
        return observation

# ================================================================
# COLLAPSE ADAPTER CLASS
# ================================================================

class CollapseAdapter:
    """Manages and interprets quantum collapse events with ethical considerations"""
    
    def __init__(self, field_shape: Tuple[int, ...], archetypes: Optional[List[NarrativeArchetype]] = None):
        self.field_shape = field_shape
        self.archetypes = archetypes or []
        self.collapse_history = []
        self.current_collapse_interpretation = {}
        
        # Narrative patterns for interpreting collapses
        self.narrative_patterns = {
            "creation": {"threshold": 0.8, "pattern": "centered_peak"},
            "destruction": {"threshold": 0.7, "pattern": "scattered_peaks"},
            "transformation": {"threshold": 0.75, "pattern": "moving_peak"},
            "revelation": {"threshold": 0.85, "pattern": "sudden_peak"},
            "balance": {"threshold": 0.6, "pattern": "symmetric_distribution"}
        }
        
        logger.info(f"Initialized CollapseAdapter for field shape {field_shape}")
    
    def interpret_collapse(self, 
                          before_state: np.ndarray, 
                          after_state: np.ndarray, 
                          ethical_tensor: np.ndarray, 
                          breath_phase: BreathPhase) -> Dict[str, Any]:
        """Interpret the narrative meaning of a quantum collapse event
        
        Args:
            before_state: State before collapse
            after_state: State after collapse
            ethical_tensor: Ethical tensor at collapse
            breath_phase: Current breath phase
            
        Returns:
            Interpretation dictionary
        """
        # Calculate collapse properties
        before_density = np.abs(before_state)**2
        after_density = np.abs(after_state)**2
        
        # Find collapse location (maximum density change)
        density_change = after_density - before_density
        collapse_idx = np.unravel_index(np.argmax(np.abs(density_change)), self.field_shape)
        
        # Calculate collapse magnitude
        collapse_magnitude = np.max(np.abs(density_change))
          # Calculate pattern metrics
        collapsed_pattern = self._identify_pattern(after_density)
        entropy_change = self._calculate_entropy(after_density) - self._calculate_entropy(before_density)
        ethical_alignment = self._calculate_ethical_alignment(tuple(int(x) for x in collapse_idx), ethical_tensor)
        
        # Get ethical vector at collapse position
        ethical_vector = [ethical_tensor[e][collapse_idx] for e in range(ethical_tensor.shape[0])]
        
        # Identify narrative archetype match
        archetype_match = None
        match_strength = 0.0
        
        for archetype in self.archetypes:
            # Calculate similarity with collapse ethical vector
            similarity = self._vector_similarity(ethical_vector, archetype.ethical_vector)
            
            if similarity > 0.7 and similarity > match_strength:
                archetype_match = archetype.name
                match_strength = similarity
        
        # Interpret based on breath phase
        phase_interpretation = {
            BreathPhase.INHALE: "Emerging possibility, new potential",
            BreathPhase.HOLD_IN: "Stabilized pattern, crystallized meaning",
            BreathPhase.EXHALE: "Manifested reality, determined outcome",
            BreathPhase.HOLD_OUT: "Reset state, void potential"
        }.get(breath_phase, "Neutral transition")
        
        # Generate complete interpretation
        interpretation = {
            'collapse_position': collapse_idx,
            'collapse_magnitude': float(collapse_magnitude),
            'entropy_change': float(entropy_change),
            'ethical_alignment': ethical_alignment,
            'ethical_vector': [float(v) for v in ethical_vector],
            'pattern_type': collapsed_pattern,
            'archetype_match': archetype_match,
            'archetype_match_strength': float(match_strength) if archetype_match else 0.0,
            'breath_phase': breath_phase.name,
            'phase_interpretation': phase_interpretation,
            'narrative_implications': self._generate_narrative_implications(
                collapsed_pattern, ethical_alignment, entropy_change, breath_phase
            )
        }
        
        # Store this interpretation
        self.current_collapse_interpretation = interpretation
        self.collapse_history.append(interpretation)
        
        # Keep history to reasonable length
        if len(self.collapse_history) > 100:
            self.collapse_history.pop(0)
            
        logger.info(f"Interpreted collapse with magnitude {collapse_magnitude:.3f} as '{collapsed_pattern}' pattern")
        if archetype_match:
            logger.info(f"Matched archetype '{archetype_match}' with strength {match_strength:.2f}")
        
        return interpretation
    
    def _identify_pattern(self, density: np.ndarray) -> str:
        """Identify the pattern type in the collapsed density"""
        # Check for centered peak
        center = tuple(s // 2 for s in density.shape)
        center_density = density[center]
        mean_density = np.mean(density)
        if center_density > 3 * mean_density:
            return "centered_peak"
        
        # Check for scattered peaks
        threshold = np.max(density) * 0.5
        peak_count = np.sum(density > threshold)
        
        if peak_count > 5:
            return "scattered_peaks"
        
        # Comprehensive symmetry checking
        symmetry_score = self._calculate_symmetry_score(density)
        if symmetry_score > 0.8:
            return "symmetric_distribution"
        
        # Check for sudden peak (high kurtosis)
        try:
            from scipy.stats import kurtosis
            k = kurtosis(density.flatten())
            if k > 2.0:
                return "sudden_peak"
        except ImportError:
            # Fallback kurtosis calculation
            if self._calculate_kurtosis(density.flatten()) > 2.0:
                return "sudden_peak"
        
        # Default pattern
        return "complex_distribution"
    
    def _calculate_symmetry_score(self, density: np.ndarray) -> float:
        """Calculate comprehensive symmetry score for a density distribution
        
        Args:
            density: Density array to analyze
            
        Returns:
            Symmetry score between 0.0 (no symmetry) and 1.0 (perfect symmetry)
        """
        scores = []
        
        # 1. Reflection symmetry checks
        reflection_score = self._check_reflection_symmetry(density)
        scores.append(reflection_score)
        
        # 2. Rotational symmetry (for 2D+ arrays)
        if len(density.shape) >= 2:
            rotational_score = self._check_rotational_symmetry(density)
            scores.append(rotational_score)
        
        # 3. Radial symmetry (for 2D+ arrays)
        if len(density.shape) >= 2:
            radial_score = self._check_radial_symmetry(density)
            scores.append(radial_score)
        
        # 4. Point symmetry (center of mass)
        point_score = self._check_point_symmetry(density)
        scores.append(point_score)
        
        # Return the maximum symmetry score found
        return max(scores) if scores else 0.0
    
    def _check_reflection_symmetry(self, density: np.ndarray) -> float:
        """Check reflection symmetry across all axes
        
        Args:
            density: Density array to check
            
        Returns:
            Maximum reflection symmetry score across all axes
        """
        max_score = 0.0
        
        for axis in range(len(density.shape)):
            # Flip array along current axis
            flipped = np.flip(density, axis=axis)
            
            # Calculate correlation between original and flipped
            correlation = self._calculate_correlation(density, flipped)
            max_score = max(max_score, correlation)
        
        return max_score
    
    def _check_rotational_symmetry(self, density: np.ndarray) -> float:
        """Check rotational symmetry for 2D arrays
        
        Args:
            density: 2D density array to check
            
        Returns:
            Rotational symmetry score
        """
        if len(density.shape) != 2:
            return 0.0
        
        # Check 90-degree rotational symmetry
        rotated_90 = np.rot90(density, k=1)
        score_90 = self._calculate_correlation(density, rotated_90)
        
        # Check 180-degree rotational symmetry
        rotated_180 = np.rot90(density, k=2)
        score_180 = self._calculate_correlation(density, rotated_180)
        
        # Check 270-degree rotational symmetry
        rotated_270 = np.rot90(density, k=3)
        score_270 = self._calculate_correlation(density, rotated_270)
        
        return max(score_90, score_180, score_270)
    
    def _check_radial_symmetry(self, density: np.ndarray) -> float:
        """Check radial symmetry around the center
        
        Args:
            density: Density array to check
            
        Returns:
            Radial symmetry score
        """
        if len(density.shape) < 2:
            return 0.0
        
        # Calculate center
        center = tuple(s // 2 for s in density.shape)
        
        # Create distance map from center
        indices = np.indices(density.shape)
        distances = np.zeros(density.shape)
        
        for i, idx in enumerate(indices):
            if i < len(center):
                distances += ((idx - center[i]) / density.shape[i])**2
        
        distances = np.sqrt(distances)
        
        # Group pixels by distance and check if values are similar
        # Discretize distances into bins
        max_distance = np.max(distances)
        num_bins = min(20, int(max_distance * 10))
        
        if num_bins < 2:
            return 0.0
        
        bin_edges = np.linspace(0, max_distance, num_bins + 1)
        
        # Calculate variance within each distance bin
        total_variance = 0.0
        valid_bins = 0
        
        for i in range(num_bins):
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
            if np.sum(mask) > 1:  # Need at least 2 points
                bin_values = density[mask]
                bin_variance = np.var(bin_values)
                total_variance += bin_variance
                valid_bins += 1
        
        if valid_bins == 0:
            return 0.0
        
        # Lower variance means higher radial symmetry
        avg_variance = total_variance / valid_bins
        max_possible_variance = np.var(density)
        
        # Convert to score (0 to 1, where 1 is perfect radial symmetry)
        if max_possible_variance > 0:
            symmetry_score = 1.0 - (avg_variance / max_possible_variance)
            return max(0.0, min(1.0, symmetry_score))
        
        return 0.0
    
    def _check_point_symmetry(self, density: np.ndarray) -> float:
        """Check point symmetry around center of mass
        
        Args:
            density: Density array to check
            
        Returns:
            Point symmetry score
        """
        # Calculate center of mass manually (avoid scipy dependency)
        total_mass = np.sum(density)
        if total_mass == 0:
            return 0.0
        
        # Calculate center of mass
        indices = np.indices(density.shape)
        center_of_mass = []
        
        for i, idx in enumerate(indices):
            weighted_sum = np.sum(idx * density)
            center_of_mass.append(weighted_sum / total_mass)
        
        # Check if center of mass is close to geometric center
        geometric_center = [s / 2.0 for s in density.shape]
        
        # Calculate distance between centers
        distance = 0.0
        for i in range(len(center_of_mass)):
            if i < len(geometric_center):
                normalized_distance = (center_of_mass[i] - geometric_center[i]) / density.shape[i]
                distance += normalized_distance**2
        
        distance = np.sqrt(distance)
        
        # Convert distance to symmetry score
        # Close to center = high symmetry score
        max_distance = 0.5  # Maximum normalized distance from center to corner
        symmetry_score = 1.0 - min(distance / max_distance, 1.0)
        
        return max(0.0, symmetry_score)
    
    def _calculate_correlation(self, array1: np.ndarray, array2: np.ndarray) -> float:
        """Calculate correlation between two arrays
        
        Args:
            array1: First array
            array2: Second array
            
        Returns:
            Correlation coefficient (0.0 to 1.0)
        """
        # Flatten arrays
        flat1 = array1.flatten()
        flat2 = array2.flatten()
        
        # Calculate means
        mean1 = np.mean(flat1)
        mean2 = np.mean(flat2)
        
        # Calculate correlation coefficient
        numerator = np.sum((flat1 - mean1) * (flat2 - mean2))
        denominator = np.sqrt(np.sum((flat1 - mean1)**2) * np.sum((flat2 - mean2)**2))
        
        if denominator > 0:
            correlation = numerator / denominator
            # Return absolute correlation (we care about similarity, not direction)
            return abs(correlation)
        
        return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis manually (fallback for scipy)
        
        Args:
            data: Data array to analyze
            
        Returns:
            Kurtosis value
        """
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        # Calculate fourth moment
        fourth_moment = np.mean(((data - mean) / std)**4)
        
        # Kurtosis (subtract 3 for excess kurtosis)
        return fourth_moment - 3.0
# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def create_ethical_tensor(field_shape: Tuple[int, ...], 
                         ethical_dimensions: int = DEFAULT_ETHICAL_DIMENSIONS) -> np.ndarray:
    """Create an initialized ethical tensor
    
    Args:
        field_shape: Shape of the field
        ethical_dimensions: Number of ethical dimensions
        
    Returns:
        Initialized ethical tensor
    """
    return np.zeros((ethical_dimensions,) + field_shape)

def apply_ethical_force(field_state: np.ndarray, 
                       ethical_tensor: np.ndarray, 
                       coupling_constant: float = 0.1) -> np.ndarray:
    """Apply ethical forces to a quantum field state
    
    Args:
        field_state: Current field state
        ethical_tensor: Ethical tensor
        coupling_constant: Coupling strength
        
    Returns:
        Modified field state
    """
    modified_state = field_state.copy()
    
    for i in range(ethical_tensor.shape[0]):
        # Apply ethical force as a modulation
        ethical_field = ethical_tensor[i]
        modified_state += coupling_constant * ethical_field * field_state
    
    # Normalize
    norm = np.sqrt(np.sum(np.abs(modified_state)**2))
    if norm > 0:
        modified_state = modified_state / norm
    
    return modified_state

def analyze_ethical_distribution(ethical_tensor: np.ndarray) -> Dict[str, Any]:
    """Analyze the distribution of ethical forces
    
    Args:
        ethical_tensor: Ethical tensor to analyze
        
    Returns:
        Analysis results dictionary
    """
    analysis = {}
    
    for i, dimension in enumerate(ETHICAL_DIMENSIONS):
        if i < ethical_tensor.shape[0]:
            field = ethical_tensor[i]
            analysis[dimension] = {
                'mean': float(np.mean(field)),
                'std': float(np.std(field)),
                'min': float(np.min(field)),
                'max': float(np.max(field)),
                'total_magnitude': float(np.sum(np.abs(field)))
            }
    
    return analysis

# ================================================================
# ETHICAL TENSOR FACTORY
# ================================================================

class EthicalTensorFactory:
    """Factory for creating and configuring ethical tensor systems"""
    
    @staticmethod
    def create_symbolic_quantum_state(field_shape: Tuple[int, ...], 
                                     ethical_dimensions: int = DEFAULT_ETHICAL_DIMENSIONS,
                                     archetypes: Optional[List[NarrativeArchetype]] = None) -> SymbolicQuantumState:
        """Create a configured symbolic quantum state
        
        Args:
            field_shape: Shape of the quantum field
            ethical_dimensions: Number of ethical dimensions
            archetypes: List of narrative archetypes to add
            
        Returns:
            Configured SymbolicQuantumState instance
        """
        state = SymbolicQuantumState(field_shape, ethical_dimensions)
        
        if archetypes:
            for archetype in archetypes:
                state.add_archetype(archetype)
        
        return state
    
    @staticmethod
    def create_standard_archetypes() -> List[NarrativeArchetype]:
        """Create a standard set of narrative archetypes
        
        Returns:
            List of standard archetypes
        """
        archetypes = [
            NarrativeArchetype(
                name="creation",
                ethical_vector=[0.8, 0.6, 0.5, 0.3, 0.7],  # Good, truth, fair, some constraint, caring
                intensity=1.0
            ),
            NarrativeArchetype(
                name="destruction",
                ethical_vector=[-0.6, -0.3, -0.4, 0.7, -0.5],  # Harmful, some deception, unfair, liberty, uncaring
                intensity=0.8
            ),
            NarrativeArchetype(
                name="rebirth",
                ethical_vector=[0.5, 0.8, 0.6, 0.8, 0.9],  # Good, very truthful, fair, liberty, very caring
                intensity=1.2
            ),
            NarrativeArchetype(
                name="transcendence",
                ethical_vector=[0.9, 0.9, 0.8, 0.9, 0.8],  # Very good, very truthful, very fair, liberty, caring
                intensity=1.5
            ),
            NarrativeArchetype(
                name="equilibrium",
                ethical_vector=[0.0, 0.0, 0.0, 0.0, 0.0],  # Neutral across all dimensions
                intensity=1.0
            )
        ]
        
        return archetypes

# ================================================================
# EXAMPLE USAGE
# ================================================================

if __name__ == "__main__":
    # Example usage of the Ethical Tensor system
    print("Ethical Tensors - Production Reference Implementation")
    print("=" * 60)
    
    # Create a symbolic quantum state
    field_shape = (32, 32)
    state = EthicalTensorFactory.create_symbolic_quantum_state(
        field_shape, 
        archetypes=EthicalTensorFactory.create_standard_archetypes()
    )
    
    # Apply symbolic meaning
    effect = state.apply_symbolic_meaning("hope", (0.5, 0.5), intensity=1.0)
    print(f"Applied symbol 'hope': {effect}")
    
    # Apply conscious intent
    intent_effect = state.apply_intent({
        'type': 'create',
        'intensity': 0.8,
        'focus_point': (0.3, 0.7),
        'ethical_vector': [0.5, 0.8, 0.6, 0.4, 0.7]
    })
    print(f"Applied creative intent: {intent_effect}")
    
    # Evolve the state
    evolution_result = state.evolve(0.1, BreathPhase.INHALE, 0.5)
    print(f"Evolution result: {evolution_result}")
    
    # Get observation
    observation = state.get_observation()
    print(f"Current coherence: {observation['coherence']:.3f}")
    print(f"Symbolic entanglement: {observation['symbolic_entanglement']:.3f}")
    print(f"Field energy: {observation['field_energy']:.3f}")
    
    print("\nEthical Tensor system initialized successfully!")