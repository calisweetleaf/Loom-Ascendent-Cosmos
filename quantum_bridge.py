# ================================================================
#  LOOM ASCENDANT COSMOS — RECURSIVE SYSTEM MODULE
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
#  Integrity Hash (SHA-256): d3ab9688a5a20b8065990cd9b91805e3d892d6e72472f69dd9afe719250c5e37
# ================================================================
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum, auto

from quantum_physics import QuantumField, QuantumStateVector, EthicalGravityManifold, PhysicsConstants, SymbolicOperators

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantumBridge")

class BreathPhase(Enum):
    """Breath phases from Timeline Engine that will affect quantum behavior"""
    INHALE = auto()       # Expansion, possibility generation, superposition
    HOLD_IN = auto()      # Stabilization, coherence maintenance
    EXHALE = auto()       # Contraction, probability collapse, resolution
    HOLD_OUT = auto()     # Void state, potential reset, quantum vacuum

@dataclass
class NarrativeArchetype:
    """Archetypes that influence quantum field behavior"""
    name: str
    ethical_vector: List[float]  # Values across ethical dimensions
    quantum_signature: np.ndarray = field(default_factory=lambda: np.random.rand(5))
    influence_radius: float = 1.0
    intensity: float = 1.0
    
    def to_field_modulation(self, field_shape: Tuple[int, ...]) -> np.ndarray:
        """Convert archetype to a field modulation pattern"""
        modulation = np.ones(field_shape) * 0.05  # Base modulation
        
        # Create pattern based on archetype
        if self.name.lower() == "creation":
            # Creation increases energy in center
            indices = np.indices(field_shape)
            center = [dim // 2 for dim in field_shape]
            distance = np.zeros(field_shape)
            
            for i, idx in enumerate(indices):
                distance += ((idx - center[i]) / field_shape[i])**2
            
            distance = np.sqrt(distance)
            modulation += 0.3 * np.exp(-5 * distance) * self.intensity
            
        elif self.name.lower() == "destruction":
            # Destruction creates chaotic interference patterns
            for i in range(len(field_shape)):
                indices = np.indices(field_shape)[i]
                modulation += 0.2 * np.sin(indices * 6.28 / field_shape[i]) * self.intensity
                
        elif self.name.lower() == "rebirth":
            # Rebirth creates spiral patterns
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

class QuantumBreathAdapter:
    """Adapts quantum field behavior to the breath phase from timeline engine"""
    
    def __init__(self, field_resolution: int = 64, ethical_dimensions: int = 5):
        self.field_resolution = field_resolution
        self.ethical_dimensions = ethical_dimensions
        self.current_phase = BreathPhase.INHALE
        self.phase_duration = 0
        self.constants = PhysicsConstants()
        
        # Parameters that control quantum behavior in different breath phases
        self.coherence_factors = {
            BreathPhase.INHALE: 1.2,     # Enhanced coherence during inhale
            BreathPhase.HOLD_IN: 1.5,    # Maximum coherence during hold
            BreathPhase.EXHALE: 0.8,     # Reduced coherence during exhale
            BreathPhase.HOLD_OUT: 0.5,   # Minimum coherence during void
        }
        
        self.collapse_thresholds = {
            BreathPhase.INHALE: 0.9,     # High threshold (rare collapses)
            BreathPhase.HOLD_IN: 0.95,   # Very high threshold (minimal collapses)
            BreathPhase.EXHALE: 0.6,     # Low threshold (frequent collapses)
            BreathPhase.HOLD_OUT: 0.7,   # Moderate threshold
        }
        
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
    
    def apply_breath_to_field(self, field: QuantumField) -> QuantumField:
        """Modify quantum field behavior based on current breath phase"""
        # Apply coherence factor
        coherence_factor = self.coherence_factors[self.current_phase]
        
        # Add vacuum fluctuations appropriate to the phase
        fluctuation_scale = self.vacuum_fluctuation_scales[self.current_phase]
        field.vacuum_fluctuations(scale=fluctuation_scale)
        
        # Modify field evolution based on breath phase
        if self.current_phase == BreathPhase.INHALE:
            # Enhance superposition - reduce potential barriers
            field.potential *= (1.0 - 0.3 * self.phase_progress)
            
        elif self.current_phase == BreathPhase.HOLD_IN:
            # Stabilize field - reinforce existing patterns
            field_density = np.abs(field.psi)**2
            field.potential += 0.1 * field_density * self.phase_progress
            
        elif self.current_phase == BreathPhase.EXHALE:
            # Increase probability of collapse - enhance potential barriers
            field.potential *= (1.0 + 0.4 * self.phase_progress)
            
        elif self.current_phase == BreathPhase.HOLD_OUT:
            # Reset to near-vacuum state as progress increases
            if self.phase_progress > 0.8:
                # Near the end of HOLD_OUT, prepare for new cycle
                field.vacuum_fluctuations(scale=0.2)
                field.potential = np.zeros_like(field.potential)
        
        return field
    
    def should_collapse_state(self, probability: float) -> bool:
        """Determine if a quantum state should collapse based on current breath phase"""
        threshold = self.collapse_thresholds[self.current_phase]
        
        # Adjust threshold based on phase progress
        if self.current_phase == BreathPhase.EXHALE:
            # Collapse becomes increasingly likely during exhale
            threshold -= 0.3 * self.phase_progress
        
        return probability > threshold
    
    def modulate_ethical_tensor(self, ethical_tensor: np.ndarray) -> np.ndarray:
        """Modulate ethical tensor based on breath phase"""
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

class SymbolicQuantumState:
    """Bridge between quantum states and symbolic meaning"""
    
    def __init__(self, field_shape: Tuple[int, ...], ethical_dimensions: int = 5):
        """Initialize a symbolic quantum state"""
        self.field_shape = field_shape
        self.ethical_dimensions = ethical_dimensions
        
        # Quantum field components
        self.field = QuantumField({"grid_size": field_shape[0], "dimensions": len(field_shape) + 1})
        self.ethical_manifold = EthicalGravityManifold(dimensions=3+ethical_dimensions, resolution=field_shape[0])
        
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
        """Add a narrative archetype that influences the quantum state"""
        self.archetypes.append(archetype)
        
        # Apply immediate ethical influence
        ethical_vector = archetype.ethical_vector
        
        # Calculate center position
        center_pos = [0.5] * (len(self.field_shape) + 1)  # +1 for time dimension
        
        # Apply ethical charge to manifold
        self.ethical_manifold.apply_ethical_charge(
            position=center_pos,
            ethical_vector=ethical_vector,
            radius=archetype.influence_radius
        )
        
        logger.info(f"Added archetype '{archetype.name}' with ethical vector {ethical_vector}")
    
    def apply_symbolic_meaning(self, symbol: str, position: Tuple[float, ...], intensity: float = 1.0) -> None:
        """Apply symbolic meaning to influence the quantum field"""
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
        self.field.potential += symbol_potential
        
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
        """Apply conscious intent to modulate quantum potentials"""
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
            self.field.psi = self.field.psi * (1.0 + 0.3 * intent_field)
            self.field.normalize()
            effect_description = "Increased quantum probability amplitude at focus point"
            
        elif intent_type == 'dissolve':
            # Dissolving intent decreases potential barriers
            potential_reduction = self.field.potential * intent_field * 0.5
            self.field.potential -= potential_reduction
            effect_description = "Reduced potential barriers in target region"
            
        elif intent_type == 'create':
            # Creative intent adds energy and new patterns
            phase_pattern = np.zeros(self.field_shape)
            for i in range(len(self.field_shape)):
                indices = np.indices(self.field_shape)[i]
                phase_pattern += np.sin(indices * 3.14 / self.field_shape[i])
            
            # Add creative pattern to field
            creation_pattern = intent_field * np.exp(1j * phase_pattern * intensity)
            self.field.psi = self.field.psi + 0.3 * creation_pattern
            self.field.normalize()
            effect_description = "Added new quantum patterns to target region"
            
        elif intent_type == 'observe':
            # Observation intent increases probability of collapse
            # This is handled through the collapse threshold
            self.collapse_threshold -= 0.2 * intensity
            effect_description = "Increased probability of wavefunction collapse"
        
        # Apply ethical influence through manifold
        center_pos = list(focus_point) + [0.5]  # Add time dimension
        self.ethical_manifold.apply_ethical_charge(
            position=center_pos,
            ethical_vector=ethical_vector,
            radius=0.2 * intensity
        )
        
        logger.info(f"Applied intent '{intent_type}' at {focus_point} with intensity {intensity:.2f}")
        
        # Return effect information
        return {
            'intent_type': intent_type,
            'intensity': intensity,
            'focus_point': focus_point,
            'effect_description': effect_description,
            'ethical_influence': ethical_vector
        }
    
    def evolve(self, dt: float, breath_phase: BreathPhase, phase_progress: float) -> Dict[str, Any]:
        """Evolve the quantum state with symbolic influences for one time step"""
        # Update breath phase
        self.breath_adapter.set_breath_phase(breath_phase, phase_progress)
        
        # Apply breath phase effects to quantum field
        self.field = self.breath_adapter.apply_breath_to_field(self.field)
        
        # Apply archetype influences
        for archetype in self.archetypes:
            modulation = archetype.to_field_modulation(self.field_shape)
            self.field.psi = self.field.psi * modulation
            self.field.normalize()
        
        # Apply symbolic resonance to field evolution
        symbolic_potential = self.symbol_resonance * 0.1
        self.field.potential += symbolic_potential
        
        # Evolve quantum field
        self.field.evolve_field(dt)
        
        # Update coherence based on field properties
        field_density = np.abs(self.field.psi)**2
        entropy = -np.sum(field_density * np.log(field_density + 1e-10))
        self.coherence = 1.0 / (1.0 + entropy * 0.1)
        
        # Apply ethical tensor to field (ethical gravity)
        ethical_tensor = np.zeros((self.ethical_dimensions,) + self.field_shape)
        for dim in range(min(self.ethical_dimensions, self.ethical_manifold.ethical_dimensions)):
            ethical_field = self.ethical_manifold.get_ethical_field(dim)
            if ethical_field is not None:
                # Reshape ethical field if needed to match quantum field
                if ethical_field.shape != self.field_shape:
                    from scipy.ndimage import zoom
                    zoom_factors = tuple(fs / ef for fs, ef in zip(self.field_shape, ethical_field.shape))
                    ethical_field = zoom(ethical_field, zoom_factors)
                
                ethical_tensor[dim] = ethical_field
        
        # Modulate ethical tensor based on breath phase
        ethical_tensor = self.breath_adapter.modulate_ethical_tensor(ethical_tensor)
        
        # Apply ethical forces to quantum field
        self.field.psi = SymbolicOperators.ethical_force_application(
            self.field.psi, 
            ethical_tensor, 
            coupling_constant=0.1
        )
        
        # Check for potential collapse (based on breath phase)
        collapsed = False
        if self.breath_adapter.should_collapse_state(self.coherence):
            # Perform collapse using symbolic operators
            positions = np.indices(self.field_shape)
            self.field.psi, collapse_position = SymbolicOperators.quantum_collapse(
                self.field.psi, 
                positions, 
                method="ethical_weight" if breath_phase == BreathPhase.EXHALE else "random"
            )
            collapsed = True
            logger.debug(f"Quantum state collapsed during {breath_phase} phase")
        
        # Update meaning potential based on field density and symbolic resonance
        self.meaning_potential = np.abs(self.field.psi)**2 * (1.0 + self.symbol_resonance * 0.5)
        
        # Return the current state information
        return {
            'coherence': float(self.coherence),
            'breath_phase': breath_phase.name,
            'phase_progress': phase_progress,
            'collapsed': collapsed,
            'field_energy': float(self.field._calculate_action()),
            'ethical_influence': float(np.mean(np.abs(ethical_tensor))),
            'meaning_density': float(np.mean(self.meaning_potential))
        }
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current quantum state observation for the universe engine"""
        # Calculate key quantum observables
        field_observables = self.field.measure_observables()
        
        # Extract quantum probability field
        probability_field = np.abs(self.field.psi)**2
        
        # Extract ethical field information
        ethical_fields = []
        for dim in range(min(self.ethical_dimensions, self.ethical_manifold.ethical_dimensions)):
            ethical_field = self.ethical_manifold.get_ethical_field(dim)
            if ethical_field is not None:
                ethical_fields.append(ethical_field)
        
        # Calculate entanglement with symbolic layer (simplified measure)
        # Cross-correlation between quantum probability and meaning potential
        field_flat = probability_field.flatten()
        meaning_flat = self.meaning_potential.flatten()
        
        # Normalize both fields
        field_flat = field_flat / np.sum(field_flat)
        meaning_flat = meaning_flat / np.sum(meaning_flat)
        
        # Calculate correlation
        correlation = np.corrcoef(field_flat, meaning_flat)[0, 1]
        self.symbolic_entanglement = max(0, correlation)
        
        # Create complete observation
        observation = {
            'probability_field': probability_field,
            'field_energy': field_observables['total_energy'],
            'field_entropy': field_observables['entropy'],
            'coherence': self.coherence,
            'symbolic_entanglement': self.symbolic_entanglement,
            'ethical_fields': ethical_fields,
            'meaning_potential': self.meaning_potential,
            'breath_phase_effects': {
                'phase': self.breath_adapter.current_phase.name,
                'coherence_factor': self.breath_adapter.coherence_factors[self.breath_adapter.current_phase],
                'collapse_threshold': self.breath_adapter.collapse_thresholds[self.breath_adapter.current_phase]
            }
        }
        
        return observation

class CollapseAdapter:
    """Manages and interprets quantum collapse events for the universe engine"""
    
    def __init__(self, field_shape: Tuple[int, ...], archetypes: List[NarrativeArchetype] = None):
        self.field_shape = field_shape
        self.archetypes = archetypes or []
        self.collapse_history = []
        self.current_collapse_interpretation = {}
        
        # Narrative patterns for interpreting collapses
        self.narrative_patterns = {
            "creation": {
                "threshold": 0.8,
                "pattern": "centered_peak"
            },
            "destruction": {
                "threshold": 0.7,
                "pattern": "scattered_peaks"
            },
            "transformation": {
                "threshold": 0.75,
                "pattern": "moving_peak"
            },
            "revelation": {
                "threshold": 0.85,
                "pattern": "sudden_peak"
            },
            "balance": {
                "threshold": 0.6,
                "pattern": "symmetric_distribution"
            }
        }
        
        logger.info(f"Initialized CollapseAdapter for field shape {field_shape}")
    
    def interpret_collapse(self, 
                          before_state: np.ndarray, 
                          after_state: np.ndarray, 
                          ethical_tensor: np.ndarray, 
                          breath_phase: BreathPhase) -> Dict[str, Any]:
        """Interpret the narrative meaning of a quantum collapse event"""
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
        ethical_alignment = self._calculate_ethical_alignment(collapse_idx, ethical_tensor)
        
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
        # Calculate common pattern metrics
        
        # Check for centered peak
        center = tuple(s // 2 for s in density.shape)
        center_density = density[center]
        mean_density = np.mean(density)
        if center_density > 3 * mean_density:
            return "centered_peak"
        
        # Check for scattered peaks
        peaks = []
        threshold = np.max(density) * 0.5
        for idx in np.ndindex(density.shape):
            if density[idx] > threshold:
                peaks.append(idx)
        
        if len(peaks) > 5:
            return "scattered_peaks"
        
        # Check for symmetry
        from scipy.ndimage import center_of_mass
        com = center_of_mass(density)
        
        # Measure symmetry around center of mass
        flipped = np.zeros_like(density)
        for idx in np.ndindex(density.shape):
            flipped_idx = tuple(int(2 * com[i] - idx[i]) % density.shape[i] for i in range(len(idx)))
            flipped[idx] = density[flipped_idx]
        
        symmetry = 1.0 - np.mean(np.abs(density - flipped)) / np.max(density)
        if symmetry > 0.8:
            return "symmetric_distribution"
        
        # Check for sudden peak (high kurtosis)
        from scipy.stats import kurtosis
        k = kurtosis(density.flatten())
        if k > 2.0:
            return "sudden_peak"
        
        # Default pattern
        return "complex_distribution"
    
    def _calculate_entropy(self, density: np.ndarray) -> float:
        """Calculate the entropy of a probability distribution"""
        # Normalize to ensure it's a proper probability distribution
        pdf = density / np.sum(density)
        # Shannon entropy
        return -np.sum(pdf * np.log(pdf + 1e-10))
    
    def _calculate_ethical_alignment(self, position: Tuple[int, ...], ethical_tensor: np.ndarray) -> Dict[str, float]:
        """Calculate ethical alignment at a position"""
        ethical_vector = [ethical_tensor[e][position] for e in range(ethical_tensor.shape[0])]
        
        # Simplified ethical dimensions for interpretation
        dimension_names = ["good_harm", "truth_deception", "fairness_bias", "liberty_constraint", "care_harm"]
        
        alignment = {}
        for i, name in enumerate(dimension_names):
            if i < len(ethical_vector):
                alignment[name] = float(ethical_vector[i])
        
        return alignment
    
    def _vector_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            # Pad shorter vector
            if len(vec1) < len(vec2):
                vec1 = vec1 + [0.0] * (len(vec2) - len(vec1))
            else:
                vec2 = vec2 + [0.0] * (len(vec1) - len(vec2))
        
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Cosine similarity
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    def _generate_narrative_implications(self, 
                                        pattern: str, 
                                        ethical_alignment: Dict[str, float], 
                                        entropy_change: float, 
                                        breath_phase: BreathPhase) -> List[str]:
        """Generate narrative implications based on collapse parameters"""
        implications = []
        
        # Pattern-based implications
        pattern_implications = {
            "centered_peak": "Focus and clarity emerging at the center of attention",
            "scattered_peaks": "Fragmentation and multiplicity of perspectives",
            "symmetric_distribution": "Balance and harmony establishing across polarities",
            "sudden_peak": "Revelation or breakthrough manifesting suddenly",
            "complex_distribution": "Intricate pattern of possibilities taking shape"
        }
        
        if pattern in pattern_implications:
            implications.append(pattern_implications[pattern])
        
        # Ethical alignment implications
        for dimension, value in ethical_alignment.items():
            if abs(value) > 0.5:
                if dimension == "good_harm" and value > 0:
                    implications.append("Benevolent influences strengthening")
                elif dimension == "good_harm" and value < 0:
                    implications.append("Harmful elements gaining influence")
                elif dimension == "truth_deception" and value > 0:
                    implications.append("Truth becoming more apparent")
                elif dimension == "truth_deception" and value < 0:
                    implications.append("Deception or illusion spreading")
        
        # Entropy implications
        if entropy_change > 0.5:
            implications.append("Increasing complexity and possibility")
        elif entropy_change < -0.5:
            implications.append("Simplifying toward clarity and resolution")
        
        # Breath phase implications
        phase_implications = {
            BreathPhase.INHALE: "New potentials opening and expanding",
            BreathPhase.HOLD_IN: "Stabilizing and integrating the pattern",
            BreathPhase.EXHALE: "Manifesting and concretizing into reality",
            BreathPhase.HOLD_OUT: "Releasing and creating space for renewal"
        }
        
        if breath_phase in phase_implications:
            implications.append(phase_implications[breath_phase])
        
        return implications
    
    def get_meta_narrative(self, timespan: int = 10) -> Dict[str, Any]:
        """Generate a meta-narrative from recent collapse history"""
        # Need at least a few collapse events
        if len(self.collapse_history) < 3:
            return {"narrative": "Insufficient events to form a meta-narrative"}
        
        # Use recent history
        recent_history = self.collapse_history[-min(timespan, len(self.collapse_history)):]
        
        # Analyze patterns in collapse history
        pattern_counts = {}
        archetype_counts = {}
        ethical_vectors = []
        
        for event in recent_history:
            # Count patterns
            pattern = event['pattern_type']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Count archetypes
            if event['archetype_match']:
                archetype = event['archetype_match']
                archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
            
            # Collect ethical vectors
            if 'ethical_vector' in event:
                ethical_vectors.append(event['ethical_vector'])
        
        # Find dominant pattern and archetype
        dominant_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else None
        dominant_archetype = max(archetype_counts.items(), key=lambda x: x[1])[0] if archetype_counts else None
        
        # Calculate average ethical vector
        if ethical_vectors:
            # Pad vectors to same length
            max_len = max(len(v) for v in ethical_vectors)
            padded_vectors = [v + [0.0] * (max_len - len(v)) for v in ethical_vectors]
            average_ethical = [sum(v[i] for v in padded_vectors) / len(padded_vectors) 
                             for i in range(max_len)]
        else:
            average_ethical = []
        
        # Generate narratives for different patterns
        narrative_templates = {
            "centered_peak": "A focal point of intense significance is developing, drawing all attention inward.",
            "scattered_peaks": "Multiple threads of possibility are emerging simultaneously, creating a tapestry of divergent potentials.",
            "symmetric_distribution": "Balance is establishing itself across polarities, creating harmony and equilibrium.",
            "sudden_peak": "A sudden breakthrough or revelation has occurred, dramatically altering the landscape of possibility.",
            "complex_distribution": "A complex pattern of interrelations is forming, suggesting nuanced and intricate developments."
        }
        
        # Generate narrative for dominant pattern
        base_narrative = narrative_templates.get(
            dominant_pattern, 
            "The unfolding pattern suggests multiple interpretations and possibilities."
        )
        
        # Add archetype influence if present
        archetype_narrative = ""
        if dominant_archetype:
            archetype_narrative = f" The influence of the {dominant_archetype} archetype is strongly present."
        
        # Add ethical dimension if significant
        ethical_narrative = ""
        if average_ethical and any(abs(v) > 0.3 for v in average_ethical):
            # Find most significant dimension
            max_dim = np.argmax(np.abs(average_ethical))
            max_val = average_ethical[max_dim]
            
            # Ethical dimension names
            dim_names = ["good/harm", "truth/deception", "fairness/bias", "liberty/constraint", "care/harm"]
            
            if max_dim < len(dim_names):
                dim_name = dim_names[max_dim]
                ethical_narrative = f" The ethical dimension of {dim_name} is significantly"
                ethical_narrative += " positive." if max_val > 0 else " negative."
        
        # Combine narratives
        full_narrative = base_narrative + archetype_narrative + ethical_narrative
        
        return {
            "narrative": full_narrative,
            "dominant_pattern": dominant_pattern,
            "dominant_archetype": dominant_archetype,
            "average_ethical_vector": average_ethical,
            "analysis_timespan": len(recent_history)
        }