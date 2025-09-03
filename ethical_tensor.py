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
        self.phase_progress = 0.0  # added explicit initialization
        
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
            # removed redundant if check
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
        pattern = np.zeros(self.field_shape)
        indices = np.indices(self.field_shape)
        distance = np.zeros(self.field_shape)
        for i, idx in enumerate(indices[:len(center)]):
            distance += ((idx - center[i]) / self.field_shape[i])**2
        distance = np.sqrt(distance)
        sigma = max(1e-9, radius * 0.3)
        return intensity * np.exp(-0.5 * (distance / sigma)**2)

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
        for i, idx in enumerate(indices[:len(grid_pos)]):
            resonance += np.exp(-0.5 * ((idx - grid_pos[i]) / (self.field_shape[i] * 0.1))**2)
        peak = np.max(resonance)
        if peak > 0:
            resonance = resonance / peak * intensity
        
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
        for i, idx in enumerate(indices[:len(grid_focus)]):
            intent_field += np.exp(-0.5 * ((idx - grid_focus[i]) / (self.field_shape[i] * 0.15))**2)
        peak = np.max(intent_field)
        if peak > 0:
            intent_field = intent_field / peak * intensity
        
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

    def reset_quantum_state(self, maintain_archetypes: bool = True) -> Dict[str, Any]:
        """Reset the quantum state to initial conditions
        
        Args:
            maintain_archetypes: Whether to keep existing archetypes
            
        Returns:
            Reset information dictionary
        """
        # Store previous state information for comparison
        previous_energy = float(np.sum(np.abs(self.field_state)**2))
        previous_coherence = self.coherence
        previous_archetype_count = len(self.archetypes)
        
        # Reset core quantum field components
        self.field_state = np.zeros(self.field_shape, dtype=complex)
        self.field_potential = np.zeros(self.field_shape)
        self.symbol_resonance = np.zeros(self.field_shape)
        self.meaning_potential = np.zeros(self.field_shape)
        
        # Reset tracking variables
        self.coherence = 1.0
        self.symbolic_entanglement = 0.0
        self.collapse_threshold = 0.7
        
        # Initialize with quantum vacuum fluctuations
        vacuum_amplitude = 0.01
        vacuum_real = np.random.normal(0, vacuum_amplitude, self.field_shape)
        vacuum_imag = np.random.normal(0, vacuum_amplitude, self.field_shape)
        self.field_state = vacuum_real + 1j * vacuum_imag
        self._normalize_field_state()
        
        # Conditionally reset ethical manifold and archetypes
        if not maintain_archetypes:
            self.ethical_manifold_data = np.zeros((self.ethical_dimensions,) + self.field_shape)
            self.archetypes.clear()
        else:
            # Re-apply archetype influences to the reset field
            for archetype in self.archetypes:
                field_modulation = archetype.to_field_modulation(self.field_shape)
                amplitude_effect = 1.0 + 0.05 * field_modulation * archetype.intensity
                self.field_state = self.field_state * amplitude_effect
                self._normalize_field_state()
        
        # Reset breath adapter to initial state
        self.breath_adapter = QuantumBreathAdapter(self.field_shape[0], self.ethical_dimensions)
        
        reset_info = {
            'previous_energy': previous_energy,
            'previous_coherence': previous_coherence,
            'archetypes_maintained': maintain_archetypes,
            'archetype_count': previous_archetype_count if maintain_archetypes else 0,
            'new_energy': float(np.sum(np.abs(self.field_state)**2)),
            'vacuum_fluctuation_amplitude': vacuum_amplitude,
            'reset_timestamp': np.random.randint(0, 1000000)  # Simulation timestamp
        }
        
        logger.info(f"Quantum state reset: energy {previous_energy:.6f} -> {reset_info['new_energy']:.6f}")
        return reset_info

    def measure_quantum_property(self, property_type: str, region: Optional[Tuple[slice, ...]] = None) -> Dict[str, Any]:
        """Measure specific quantum properties of the field
        
        Args:
            property_type: Type of property to measure ('energy', 'coherence', 'entanglement', 'phase', 'momentum')
            region: Optional region to measure (tuple of slices)
            
        Returns:
            Measurement results dictionary
        """
        # Define measurement region
        if region is None:
            measurement_field = self.field_state
            measurement_region = "full_field"
        else:
            measurement_field = self.field_state[region]
            measurement_region = f"region_{region}"
        
        measurement_results = {
            'property_type': property_type,
            'measurement_region': measurement_region,
            'field_shape': measurement_field.shape,
            'measurement_uncertainty': np.random.normal(0, 0.001)  # Quantum measurement uncertainty
        }
        
        if property_type == 'energy':
            # Calculate various energy components
            kinetic_energy = np.sum(np.abs(measurement_field)**2)
            potential_energy = np.sum(self.field_potential[region] if region else self.field_potential)
            
            # Calculate energy density distribution
            energy_density = np.abs(measurement_field)**2
            energy_variance = np.var(energy_density)
            energy_skewness = self._calculate_skewness(energy_density.flatten())
            
            measurement_results.update({
                'total_energy': float(kinetic_energy + potential_energy),
                'kinetic_energy': float(kinetic_energy),
                'potential_energy': float(potential_energy),
                'energy_density_mean': float(np.mean(energy_density)),
                'energy_density_variance': float(energy_variance),
                'energy_density_skewness': float(energy_skewness),
                'peak_energy_location': tuple(np.unravel_index(np.argmax(energy_density), energy_density.shape))
            })
            
        elif property_type == 'coherence':
            # Multi-faceted coherence measurement
            spatial_coherence = self._calculate_spatial_coherence(measurement_field)
            temporal_coherence = self.coherence  # From evolution tracking
            
            # Phase coherence analysis
            phase_field = np.angle(measurement_field)
            phase_variance = np.var(phase_field)
            phase_coherence = np.exp(-phase_variance)
            
            measurement_results.update({
                'spatial_coherence': float(spatial_coherence),
                'temporal_coherence': float(temporal_coherence),
                'phase_coherence': float(phase_coherence),
                'overall_coherence': float((spatial_coherence + temporal_coherence + phase_coherence) / 3),
                'phase_variance': float(phase_variance),
                'coherence_length': self._estimate_coherence_length(measurement_field)
            })
            
        elif property_type == 'entanglement':
            # Quantum entanglement measurement with symbolic layer
            field_entropy = self._calculate_field_entropy(measurement_field)
            symbolic_correlation = self._calculate_symbolic_field_correlation(measurement_field, region)
            
            # Bipartite entanglement estimation
            if len(measurement_field.shape) >= 2:
                bipartite_entanglement = self._calculate_bipartite_entanglement(measurement_field)
            else:
                bipartite_entanglement = 0.0
            
            measurement_results.update({
                'field_entropy': float(field_entropy),
                'symbolic_correlation': float(symbolic_correlation),
                'bipartite_entanglement': float(bipartite_entanglement),
                'entanglement_measure': float(self.symbolic_entanglement),
                'von_neumann_entropy': self._calculate_von_neumann_entropy(measurement_field)
            })
            
        elif property_type == 'phase':
            # Comprehensive phase analysis
            phase_field = np.angle(measurement_field)
            phase_gradient = self._calculate_phase_gradient(phase_field)
            phase_circulation = self._calculate_phase_circulation(phase_field)
            
            # Topological phase properties
            winding_number = self._calculate_winding_number(phase_field)
            phase_singularities = self._detect_phase_singularities(phase_field)
            
            measurement_results.update({
                'mean_phase': float(np.mean(phase_field)),
                'phase_variance': float(np.var(phase_field)),
                'phase_gradient_magnitude': float(np.mean(np.abs(phase_gradient))),
                'phase_circulation': float(phase_circulation),
                'winding_number': int(winding_number),
                'singularity_count': len(phase_singularities),
                'singularity_locations': phase_singularities
            })
            
        elif property_type == 'momentum':
            # Quantum momentum measurement via phase gradients
            momentum_field = self._calculate_momentum_field(measurement_field)
            momentum_magnitude = np.abs(momentum_field)
            
            # Momentum distribution statistics
            momentum_mean = np.mean(momentum_magnitude)
            momentum_variance = np.var(momentum_magnitude)
            momentum_peak_location = np.unravel_index(np.argmax(momentum_magnitude), momentum_magnitude.shape)
            
            measurement_results.update({
                'momentum_field_shape': momentum_field.shape,
                'mean_momentum_magnitude': float(momentum_mean),
                'momentum_variance': float(momentum_variance),
                'peak_momentum_location': tuple(momentum_peak_location),
                'momentum_distribution_entropy': float(self._calculate_entropy(momentum_magnitude)),
                'uncertainty_product': self._calculate_position_momentum_uncertainty(measurement_field)
            })
            
        else:
            # Unknown property type
            measurement_results.update({
                'error': f"Unknown property type: {property_type}",
                'available_properties': ['energy', 'coherence', 'entanglement', 'phase', 'momentum']
            })
        
        logger.debug(f"Measured quantum property '{property_type}' in region '{measurement_region}'")
        return measurement_results

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data distribution"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        # Calculate third moment (skewness)
        third_moment = np.mean(((data - mean) / std)**3)
        return float(third_moment)

    def _calculate_spatial_coherence(self, field: np.ndarray) -> float:
        """Calculate spatial coherence of the quantum field"""
        # Use correlation-based coherence measure
        field_flat = field.flatten()
        
        # Calculate autocorrelation at various spatial separations
        coherence_sum = 0.0
        sample_points = min(100, len(field_flat) // 2)
        
        for separation in range(1, sample_points):
            if separation < len(field_flat):
                correlation = np.corrcoef(field_flat[:-separation], field_flat[separation:])[0, 1]
                if not np.isnan(correlation):
                    coherence_sum += abs(correlation) * np.exp(-separation / sample_points)
        
        return coherence_sum / sample_points if sample_points > 0 else 0.0

    def _estimate_coherence_length(self, field: np.ndarray) -> float:
        """Estimate the coherence length of the quantum field"""
        # Simple coherence length estimation based on field correlation decay
        if len(field.shape) < 2:
            return 1.0
        
        center = tuple(s // 2 for s in field.shape)
        reference_value = field[center]
        
        # Calculate correlation as function of distance from center
        distances = []
        correlations = []
        
        for offset in range(1, min(field.shape) // 2):
            if offset < field.shape[0] and offset < field.shape[1]:
                test_points = [
                    (center[0] + offset, center[1]),
                    (center[0] - offset, center[1]),
                    (center[0], center[1] + offset),
                    (center[0], center[1] - offset)
                ]
                
                valid_correlations = []
                for point in test_points:
                    if all(0 <= point[i] < field.shape[i] for i in range(len(point))):
                        correlation = abs(np.conj(reference_value) * field[point])
                        valid_correlations.append(correlation)
                
                if valid_correlations:
                    distances.append(offset)
                    correlations.append(np.mean(valid_correlations))
        
        # Find where correlation drops to 1/e
        coherence_threshold = np.abs(reference_value) / np.e
        coherence_length = 1.0
        
        for i, corr in enumerate(correlations):
            if corr < coherence_threshold:
                coherence_length = distances[i] if i < len(distances) else 1.0
                break
        
        return float(coherence_length)

    def _calculate_symbolic_field_correlation(self, field: np.ndarray, region: Optional[Tuple[slice, ...]] = None) -> float:
        """Calculate correlation between quantum field and symbolic meaning potential"""
        field_density = np.abs(field)**2
        
        if region is not None:
            meaning_region = self.meaning_potential[region]
        else:
            meaning_region = self.meaning_potential
        
        # Ensure compatible shapes
        if field_density.shape != meaning_region.shape:
            # Resize or crop to match
            min_shape = tuple(min(field_density.shape[i], meaning_region.shape[i]) for i in range(len(field_density.shape)))
            field_density = field_density[:min_shape[0], :min_shape[1] if len(min_shape) > 1 else ...]
            meaning_region = meaning_region[:min_shape[0], :min_shape[1] if len(min_shape) > 1 else ...]
        
        # Calculate correlation
        field_flat = field_density.flatten()
        meaning_flat = meaning_region.flatten()
        
        if len(field_flat) == 0 or len(meaning_flat) == 0:
            return 0.0
        
        correlation_matrix = np.corrcoef(field_flat, meaning_flat)
        correlation = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
        
        return float(abs(correlation))

    def _calculate_bipartite_entanglement(self, field: np.ndarray) -> float:
        """Calculate bipartite entanglement between field regions"""
        if len(field.shape) < 2:
            return 0.0
        
        # Split field into two regions
        mid_point = field.shape[0] // 2
        region_a = field[:mid_point, :]
        region_b = field[mid_point:, :]
        
        # Calculate density matrices for each region
        rho_a = np.abs(region_a)**2
        rho_b = np.abs(region_b)**2
        
        # Normalize
        rho_a = rho_a / (np.sum(rho_a) + 1e-10)
        rho_b = rho_b / (np.sum(rho_b) + 1e-10)
        
        # Calculate von Neumann entropy for each region
        entropy_a = -np.sum(rho_a * np.log(rho_a + 1e-10))
        entropy_b = -np.sum(rho_b * np.log(rho_b + 1e-10))
        
        # Combined system entropy
        rho_combined = np.abs(field)**2
        rho_combined = rho_combined / (np.sum(rho_combined) + 1e-10)
        entropy_combined = -np.sum(rho_combined * np.log(rho_combined + 1e-10))
        
        # Entanglement measure (simplified)
        entanglement = entropy_a + entropy_b - entropy_combined
        return max(0.0, float(entanglement))

    def _calculate_field_entropy(self, field: np.ndarray) -> float:
        """Calculate quantum field entropy"""
        probability_density = np.abs(field)**2
        probability_density = probability_density / (np.sum(probability_density) + 1e-10)
        
        # Shannon entropy
        entropy = -np.sum(probability_density * np.log(probability_density + 1e-10))
        return float(entropy)

    def _calculate_von_neumann_entropy(self, field: np.ndarray) -> float:
        """Calculate von Neumann entropy (quantum generalization of Shannon entropy)"""
        # For a pure state, von Neumann entropy is zero
        # For mixed states, we approximate using field density
        density_matrix = np.outer(field.flatten(), np.conj(field.flatten()))
        eigenvalues = np.real(np.linalg.eigvals(density_matrix))
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero eigenvalues
        
        if len(eigenvalues) == 0:
            return 0.0
        
        von_neumann_entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        return float(von_neumann_entropy)

    def _calculate_phase_gradient(self, phase_field: np.ndarray) -> np.ndarray:
        """Calculate gradient of phase field"""
        gradients = []
        
        for axis in range(len(phase_field.shape)):
            gradient = np.gradient(phase_field, axis=axis)
            gradients.append(gradient)
        
        # Calculate magnitude of gradient vector
        gradient_magnitude = np.zeros_like(phase_field)
        for grad in gradients:
            gradient_magnitude += grad**2
        
        return np.sqrt(gradient_magnitude)

    def _calculate_phase_circulation(self, phase_field: np.ndarray) -> float:
        """Calculate phase circulation around field boundary"""
        if len(phase_field.shape) < 2:
            return 0.0
        
        # Calculate circulation around field boundary
        height, width = phase_field.shape[:2]
        
        # Top edge (left to right)
        top_circulation = np.sum(np.diff(phase_field[0, :]))
        
        # Right edge (top to bottom)
        right_circulation = np.sum(np.diff(phase_field[:, -1]))
        
        # Bottom edge (right to left)
        bottom_circulation = -np.sum(np.diff(phase_field[-1, :]))
        
        # Left edge (bottom to top)
        left_circulation = -np.sum(np.diff(phase_field[:, 0]))
        
        total_circulation = top_circulation + right_circulation + bottom_circulation + left_circulation
        return float(total_circulation)

    def _calculate_winding_number(self, phase_field: np.ndarray) -> int:
        """Calculate topological winding number of phase field"""
        if len(phase_field.shape) < 2:
            return 0
        
        # Calculate winding number by integrating phase around closed loops
        circulation = self._calculate_phase_circulation(phase_field)
        winding_number = int(np.round(circulation / (2 * np.pi)))
        
        return winding_number

    def _detect_phase_singularities(self, phase_field: np.ndarray) -> List[Tuple[int, ...]]:
        """Detect phase singularities (vortices) in the field"""
        if len(phase_field.shape) < 2:
            return []
        
        singularities = []
        height, width = phase_field.shape[:2]
        
        # Look for rapid phase changes that indicate singularities
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # Calculate phase differences around a small loop
                phases = [
                    phase_field[i-1, j],
                    phase_field[i, j-1],
                    phase_field[i+1, j],
                    phase_field[i, j+1]
                ]
                
                # Calculate total phase change around loop
                total_change = 0
                for k in range(len(phases)):
                    phase_diff = phases[(k+1) % len(phases)] - phases[k]
                    
                    # Handle phase wrapping
                    while phase_diff > np.pi:
                        phase_diff -= 2 * np.pi
                    while phase_diff < -np.pi:
                        phase_diff += 2 * np.pi
                    
                    total_change += phase_diff
                
                # If total change is close to ±2π, we have a singularity
                if abs(abs(total_change) - 2 * np.pi) < 0.5:
                    singularities.append((i, j))
        
        return singularities

    def _calculate_momentum_field(self, field: np.ndarray) -> np.ndarray:
        """Calculate momentum field from quantum field via phase gradients"""
        phase_field = np.angle(field)
        
        # Momentum is proportional to phase gradient (ℏ∇φ)
        momentum_components = []
        
        for axis in range(len(field.shape)):
            phase_gradient = np.gradient(phase_field, axis=axis)
            momentum_components.append(phase_gradient)
        
        # Calculate momentum magnitude
        momentum_magnitude = np.zeros_like(phase_field)
        for component in momentum_components:
            momentum_magnitude += component**2
        
        return np.sqrt(momentum_magnitude) * 1.054571817e-34  # ℏ (reduced Planck constant)

    def _calculate_position_momentum_uncertainty(self, field: np.ndarray) -> float:
        """Calculate position-momentum uncertainty product (Heisenberg uncertainty)"""
        # Position uncertainty
        probability_density = np.abs(field)**2
        probability_density = probability_density / (np.sum(probability_density) + 1e-10)
        
        # Calculate position expectation values and variances
        indices = np.indices(field.shape)
        position_variances = []
        
        for axis in range(len(field.shape)):
            pos_expectation = np.sum(indices[axis] * probability_density)
            pos_squared_expectation = np.sum(indices[axis]**2 * probability_density)
            position_variance = pos_squared_expectation - pos_expectation**2
            position_variances.append(max(position_variance, 1e-10))
        
        # Momentum uncertainty (from phase gradients)
        momentum_field = self._calculate_momentum_field(field)
        momentum_variance = np.var(momentum_field)
        
        # Calculate uncertainty product for first dimension
        position_uncertainty = np.sqrt(position_variances[0])
        momentum_uncertainty = np.sqrt(momentum_variance)
        
        uncertainty_product = position_uncertainty * momentum_uncertainty
        heisenberg_limit = 1.054571817e-34 / 2  # ℏ/2
        
        return float(uncertainty_product / heisenberg_limit)  # Normalized to Heisenberg limit

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
    
    def reset_history(self) -> None:
        """Reset only narrative collapse history."""
        self.collapse_history.clear()
        self.current_collapse_interpretation = {}

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
        symmetry_score = max(scores) if scores else 0.0
        return max(0.0, min(1.0, float(symmetry_score)))

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
        
        for i, idx in enumerate(indices[:len(center)]):
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
        total = np.sum(density)
        if total == 0:
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
    
    def _calculate_entropy(self, density: np.ndarray) -> float:
        """Calculate entropy of a probability density distribution
        
        Args:
            density: Probability density array
            
        Returns:
            Shannon entropy value
        """
        # Normalize density to ensure it sums to 1
        normalized_density = density / (np.sum(density) + 1e-10)
        
        # Add small epsilon to prevent log(0)
        normalized_density = normalized_density + 1e-10
        
        # Calculate Shannon entropy
        entropy = -np.sum(normalized_density * np.log(normalized_density))
        
        return float(entropy)
    
    def _calculate_ethical_alignment(self, position: Tuple[int, ...], ethical_tensor: np.ndarray) -> Dict[str, float]:
        alignment = {}
        
        for i, dimension in enumerate(ETHICAL_DIMENSIONS):
            if i < ethical_tensor.shape[0]:
                # Get ethical value at position
                ethical_value = ethical_tensor[i][position]
                
                # Calculate alignment strength (absolute value normalized)
                max_value = np.max(np.abs(ethical_tensor[i]))
                if max_value > 0:
                    alignment[dimension] = float(np.abs(ethical_value) / max_value)
                else:
                    alignment[dimension] = 0.0
        
        return alignment
    
    def _vector_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate similarity between two vectors using cosine similarity
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Cosine similarity value (0.0 to 1.0)
        """
        # Convert to numpy arrays and ensure same length
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        min_length = min(len(v1), len(v2))
        v1 = v1[:min_length]
        v2 = v2[:min_length]
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            # Return absolute similarity (0.0 to 1.0)
            return abs(float(similarity))
        
        return 0.0
    
    def _generate_narrative_implications(self, 
                                       pattern_type: str, 
                                       ethical_alignment: Dict[str, float], 
                                       entropy_change: float, 
                                       breath_phase: BreathPhase) -> List[str]:
        """Generate narrative implications based on collapse characteristics
        
        Args:
            pattern_type: Type of collapse pattern
            ethical_alignment: Ethical alignment values
            entropy_change: Change in entropy
            breath_phase: Current breath phase
            
        Returns:
            List of narrative implication strings
        """
        implications = []
        
        # Pattern-based implications
        if pattern_type == "centered_peak":
            implications.append("Focus of attention crystallizes a singular outcome")
            if entropy_change < -0.5:
                implications.append("Reality consolidates around a point of certainty")
        
        elif pattern_type == "scattered_peaks":
            implications.append("Multiple possibilities manifest simultaneously")
            if entropy_change > 0.5:
                implications.append("Complexity emerges from quantum superposition")
        
        elif pattern_type == "symmetric_distribution":
            implications.append("Balance and harmony emerge in the collapse")
            implications.append("Universal principles assert their influence")
        
        elif pattern_type == "sudden_peak":
            implications.append("Unexpected revelation breaks through uncertainty")
            implications.append("A hidden truth suddenly becomes manifest")
        
        # Ethical-based implications
        dominant_ethics = [k for k, v in ethical_alignment.items() if v > 0.7]
        
        if "good_harm" in dominant_ethics:
            implications.append("Ethical imperative toward beneficial outcomes")
        
        if "truth_deception" in dominant_ethics:
            implications.append("Authenticity and transparency guide manifestation")
        
        if "fairness_bias" in dominant_ethics:
            implications.append("Justice and equity influence the outcome")
        
        if "liberty_constraint" in dominant_ethics:
            implications.append("Freedom and autonomy shape the result")
        
        if "care_harm" in dominant_ethics:
            implications.append("Compassion and empathy direct the collapse")
        
        # Entropy-based implications
        if entropy_change < -1.0:
            implications.append("Order emerges from chaos")
        elif entropy_change > 1.0:
            implications.append("New complexity unfolds from simplicity")
        
        # Breath phase implications
        if breath_phase == BreathPhase.INHALE:
            implications.append("New potential breathes into existence")
        elif breath_phase == BreathPhase.HOLD_IN:
            implications.append("Stabilized energy holds the pattern in place")
        elif breath_phase == BreathPhase.EXHALE:
            implications.append("Manifested reality releases into form")
        elif breath_phase == BreathPhase.HOLD_OUT:
            implications.append("Void state prepares for new possibilities")
        
        # Ensure we always return at least one implication
        if not implications:
            implications.append("Quantum potential collapses into classical reality")
        
        return implications

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