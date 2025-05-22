# ================================================================
#  LOOM ASCENDANT COSMOS — RECURSIVE SYSTEM MODULE
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
#  Integrity Hash (SHA-256): d3ab9688a5a20b8065990cd9b91805e3d892d6e72472f69dd9afe719250c5e37
# ================================================================
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, Set, TypeVar
from dataclasses import dataclass, field
import logging
import uuid
import time
from enum import Enum, auto
from collections import defaultdict
import json
import concurrent.futures
from functools import lru_cache

# Configure logging with more detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AetherEngine")

# Type variables for generics
T = TypeVar('T')
P = TypeVar('P', bound='AetherPattern')

class EncodingType(Enum):
    """Standardized encoding types with semantic meaning"""
    BINARY = auto()    # Boolean state representation
    SYMBOLIC = auto()  # Abstract character-based encoding
    VOXEL = auto()     # 3D volumetric encoding
    GLYPH = auto()     # Visual symbolic encoding
    QUANTUM = auto()   # Probabilistic quantum state encoding
    FRACTAL = auto()   # Self-similar recursive encoding
    WAVE = auto()      # Frequency/amplitude based encoding

class InteractionProtocol(Enum):
    """Standard interaction protocols between patterns"""
    COMBINE = auto()   # Simple fusion
    ENTANGLE = auto()  # Quantum-like entanglement
    TRANSFORM = auto() # State transformation
    CASCADE = auto()   # Triggering chain reactions
    RESONATE = auto()  # Harmonic interactions
    ANNIHILATE = auto() # Mutual destruction/cancellation
    CATALYZE = auto()  # Facilitate other interactions

@dataclass(frozen=True)
class AetherPattern:
    """Immutable matter/energy encoding per Genesis spec"""
    core: bytes                         # SHA3-512 hashed essential properties
    mutations: Tuple[bytes, ...]        # Allowable transformation paths (immutable)
    interactions: Dict[str, str]        # Interaction protocol signatures
    encoding_type: EncodingType         # Standardized encoding type
    recursion_level: int = 0            # Current modification depth
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional properties
    creation_timestamp: float = field(default_factory=time.time)  # Creation time
    
    def __post_init__(self):
        """Validate the pattern after initialization"""
        if isinstance(self.encoding_type, str):
            try:
                object.__setattr__(self, 'encoding_type', EncodingType[self.encoding_type.upper()])
            except KeyError:
                raise ValueError(f"Invalid encoding type: {self.encoding_type}")
        elif not isinstance(self.encoding_type, EncodingType):
            raise TypeError(f"Encoding type must be an instance of EncodingType, got {type(self.encoding_type)}")

    @property
    def pattern_id(self) -> str:
        """Globally unique pattern identifier"""
        return hashlib.blake2b(
            self.core + b''.join(self.mutations),
            digest_size=64,
            key=b'genesis_aether'
        ).hexdigest()
    
    @property
    def age(self) -> float:
        """Age of pattern in seconds"""
        return time.time() - self.creation_timestamp
    
    @property
    def complexity(self) -> float:
        """Calculate pattern complexity metric"""
        return (len(self.core) * (self.recursion_level + 1)) / 1024.0
    
    def serialize(self) -> Dict[str, Any]:
        """Convert pattern to serializable dictionary"""
        return {
            'pattern_id': self.pattern_id,
            'core_hash': hashlib.sha256(self.core).hexdigest(),
            'encoding_type': self.encoding_type.name,
            'recursion_level': self.recursion_level,
            'mutation_count': len(self.mutations),
            'interactions': list(self.interactions.keys()),
            'metadata': self.metadata,
            'creation_timestamp': self.creation_timestamp
        }
    
    def __hash__(self):
        """Make pattern hashable for set operations"""
        return hash(self.pattern_id)

class PhysicsConstraints:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def update_constraint(self, key: str, value: Any) -> None:
        """Dynamically update a physics constraint."""
        setattr(self, key, value)
        logger.info(f"Physics constraint updated: {key} = {value}")

    def get_all_constraints(self) -> Dict[str, Any]:
        """Return all current physics constraints."""
        return self.__dict__

    # Example physical constants (can be expanded significantly)
    DEFAULT_GRAVITATIONAL_CONSTANT: float = 6.674e-11  # m^3 kg^-1 s^-2
    DEFAULT_SPEED_OF_LIGHT: float = 299792458  # m/s
    DEFAULT_PLANCK_CONSTANT: float = 6.626e-34  # J s
    DEFAULT_BOLTZMANN_CONSTANT: float = 1.380e-23  # J K^-1
    DEFAULT_ELEMENTARY_CHARGE: float = 1.602e-19  # C

    def __post_init__(self):
        """Initialize default values if not provided."""
        defaults = {
            "gravitational_constant": self.DEFAULT_GRAVITATIONAL_CONSTANT,
            "speed_of_light": self.DEFAULT_SPEED_OF_LIGHT,
            "planck_constant": self.DEFAULT_PLANCK_CONSTANT,
            "boltzmann_constant": self.DEFAULT_BOLTZMANN_CONSTANT,
            "elementary_charge": self.DEFAULT_ELEMENTARY_CHARGE,
            "max_recursion_depth": 3,
            "min_pattern_size": 64,
            "default_energy_level": 1.0,
            "allow_exotic_matter": False,
        }
        for key, value in defaults.items():
            if not hasattr(self, key):
                setattr(self, key, value)

class AetherSpace:
    """Multi-dimensional manifold containing patterns"""
    def __init__(self, dimensions: int = 3, engine: Optional['AetherEngine'] = None):
        self.dimensions = dimensions
        self.engine = engine # Reference to the main engine for accessing physics, etc.
        self.patterns: Dict[AetherPattern, Tuple[float, ...]] = {} # Store pattern and its position
        self.spatial_index = None # Placeholder for k-d tree or similar
        self._indexed_patterns: List[AetherPattern] = [] # For mapping KDTree indices to patterns
        self.interaction_history: List[Dict[str, Any]] = []
        self._last_cleanup = time.time()
        self._build_spatial_index() # Initialize spatial index

    def _build_spatial_index(self) -> None:
        """Build or rebuild the spatial index (e.g., k-d tree)."""
        if not self.patterns:
            self.spatial_index = None
            self._indexed_patterns = []
            logger.info("Pattern dictionary is empty. Spatial index reset.")
            return
        
        try:
            from scipy.spatial import KDTree
            
            # Ensure patterns and their positions are consistently ordered
            self._indexed_patterns = list(self.patterns.keys())
            points = [self.patterns[p] for p in self._indexed_patterns if p in self.patterns]

            if not points: # No points to index
                self.spatial_index = None
                self._indexed_patterns = []
                logger.info("No valid points to build spatial index.")
                return

            self.spatial_index = KDTree(points)
            logger.info(f"Built KDTree spatial index with {len(points)} patterns.")
        except ImportError:
            logger.warning("SciPy not found. Spatial indexing will be less efficient. Consider installing SciPy.")
            self.spatial_index = None 
            self._indexed_patterns = [] # Clear pattern list if KDTree cannot be built
        except Exception as e:
            logger.error(f"Error building spatial index: {e}")
            self.spatial_index = None
            self._indexed_patterns = []
    
    def add_pattern(self, pattern: AetherPattern, position: Tuple[float, ...]) -> None:
        """Add pattern to space at specific position and rebuild index."""
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} dimensions, got {len(position)}")
            
        self.patterns[pattern] = position
        self._build_spatial_index() # Rebuild index after adding
        logger.debug(f"Added pattern {pattern.pattern_id} at {position}")
    
    def remove_pattern(self, pattern: AetherPattern) -> None:
        """Remove pattern from space and rebuild index."""
        if pattern in self.patterns:
            del self.patterns[pattern]
            self._build_spatial_index() # Rebuild index after removing
            logger.debug(f"Removed pattern {pattern.pattern_id}")
        else:
            logger.warning(f"Attempted to remove non-existent pattern {pattern.pattern_id}")
    
    def get_nearby_patterns(self, position: Tuple[float, ...], radius: float) -> List[AetherPattern]:
        """Find patterns within radius of position using the spatial index."""
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} dimensions, got {len(position)}")

        nearby_patterns: List[AetherPattern] = []
        if self.spatial_index is None or not hasattr(self, '_indexed_patterns') or not self._indexed_patterns:
            # Fallback to brute-force search if KDTree is not available or not built
            logger.warning("KDTree not available or not properly initialized, performing brute-force nearby search.")
            for p, pos in self.patterns.items():
                # Calculate squared Euclidean distance for efficiency
                dist_sq = sum((px - qx)**2 for px, qx in zip(pos, position))
                if dist_sq <= radius**2:
                    nearby_patterns.append(p)
            return nearby_patterns

        try:
            # Query the KDTree for neighbors within the given radius
            indices = self.spatial_index.query_ball_point(position, r=radius)
            
            if indices:
                for i in indices:
                    if i < len(self._indexed_patterns):
                        nearby_patterns.append(self._indexed_patterns[i])
                    else:
                        logger.error(f"KDTree index {i} out of bounds for _indexed_patterns list (len: {len(self._indexed_patterns)}). Possible de-sync.")
        except Exception as e:
            logger.error(f"Error querying spatial index: {e}. Falling back to brute-force search.")
            # Fallback to brute-force on error
            for p, pos in self.patterns.items():
                dist_sq = sum((px - qx)**2 for px, qx in zip(pos, position))
                if dist_sq <= radius**2:
                    nearby_patterns.append(p)
        
        logger.debug(f"Found {len(nearby_patterns)} patterns near {position} within radius {radius}")
        return nearby_patterns
    
    def _generate_neighbor_offsets(self, radius: int) -> List[Tuple[int, ...]]:
        """Generate offsets for neighboring grid cells (used by fallback or simpler index)."""
        # This method might be deprecated or adapted if KDTree is primary.
        # For now, keep for potential fallback or other uses.
        offsets = []
        r = int(radius) # Ensure integer radius for range
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if self.dimensions == 2:
                    offsets.append((i, j))
                elif self.dimensions == 3:
                    for k in range(-r, r + 1):
                        offsets.append((i, j, k))
                # Add more dimensions if needed
        return offsets

    def trigger_interactions(self, pattern: AetherPattern, interaction_type: InteractionProtocol) -> None:
        """
        Trigger interactions for a pattern with its neighbors.
        This method uses the AetherEngine's interaction handlers.
        """
        if not self.engine:
            logger.error("AetherSpace cannot trigger interactions without a reference to AetherEngine.")
            return

        position = self.patterns.get(pattern)
        if not position:
            logger.warning(f"Pattern {pattern.pattern_id} not found in AetherSpace for interaction.")
            return

        # Define an interaction radius based on physics or pattern properties
        # Ensure self.engine.physics is not None and get() is a valid method.
        interaction_radius = 10.0 # Default fixed radius
        if self.engine.physics:
            interaction_radius = self.engine.physics.get("default_interaction_radius", 10.0)
        
        nearby_patterns = self.get_nearby_patterns(position, interaction_radius)
        
        # Filter out the source pattern itself and ensure patterns are valid
        target_patterns = [p for p in nearby_patterns if p != pattern and p in self.patterns]

        if not target_patterns:
            logger.info(f"No valid nearby patterns for {pattern.pattern_id} to interact with via {interaction_type.name}.")
            return

        # Delegate to AetherEngine's interaction handlers
        handler = self.engine.interaction_handlers.get(interaction_type)
        if handler:
            try:
                all_involved_patterns = [pattern] + target_patterns
                # The handler might modify the space (add/remove patterns) or return results
                result = handler(all_involved_patterns) 
                
                # Log the interaction
                log_entry = {
                    "type": interaction_type.name,
                    "initiator": pattern.pattern_id,
                    "targets": [p.pattern_id for p in target_patterns],
                    "timestamp": time.time(),
                    "result_summary": self._summarize_interaction_result(result, interaction_type, all_involved_patterns)
                }
                self.interaction_history.append(log_entry)
                
                if self.engine.metrics: # Ensure metrics dict is available
                    self.engine.metrics['interactions_processed'] = self.engine.metrics.get('interactions_processed', 0) + 1

                # Process results: e.g., add new patterns, remove old ones based on handler's logic
                self._process_interaction_result(result, interaction_type, all_involved_patterns, position)
                
            except Exception as e:
                logger.error(f"Error during interaction {interaction_type.name} for pattern {pattern.pattern_id}: {e}", exc_info=True)
        else:
            logger.warning(f"No handler found for interaction type {interaction_type.name}")

    def _summarize_interaction_result(self, result: Any, interaction_type: InteractionProtocol, involved_patterns: List[AetherPattern]) -> str:
        """Helper to create a summary string for logging interaction results."""
        if isinstance(result, AetherPattern):
            return f"New pattern created: {result.pattern_id}"
        elif isinstance(result, list) and all(isinstance(p, AetherPattern) for p in result):
            return f"New patterns created: {[p.pattern_id for p in result]}"
        elif result is None and interaction_type == InteractionProtocol.ANNIHILATE:
            return f"Annihilation of patterns: {[p.pattern_id for p in involved_patterns]}"
        elif isinstance(result, tuple) and len(result) == 2 and interaction_type == InteractionProtocol.ENTANGLE: # Example for entangle
             return f"Entangled patterns with key: {result[1]}" # Assuming (patterns, key)
        return "Interaction completed."


    def _process_interaction_result(self, result: Any, interaction_type: InteractionProtocol, 
                                    involved_patterns: List[AetherPattern], initiator_position: Tuple[float,...]):
        """Process the results of an interaction, updating AetherSpace."""
        if isinstance(result, AetherPattern) and result not in self.patterns:
            # New pattern created, add it to the space (e.g., at initiator's position or other)
            self.add_pattern(result, initiator_position)
            logger.info(f"Interaction {interaction_type.name} resulted in new pattern {result.pattern_id} added at {initiator_position}")

        elif isinstance(result, list) and all(isinstance(p, AetherPattern) for p in result):
            # Multiple new patterns created
            for new_p in result:
                if new_p not in self.patterns:
                    self.add_pattern(new_p, initiator_position) # Or determine individual positions
                    logger.info(f"Interaction {interaction_type.name} resulted in new pattern {new_p.pattern_id} added at {initiator_position}")
        
        elif result is None and interaction_type == InteractionProtocol.ANNIHILATE:
            # Annihilation: remove involved patterns from the space
            for p_annihilated in involved_patterns:
                if p_annihilated in self.patterns:
                    self.remove_pattern(p_annihilated)
            logger.info(f"Interaction {interaction_type.name} resulted in annihilation of patterns: {[p.pattern_id for p in involved_patterns]}.")

        # TRANSFORM: The handler returns the *new* pattern. The old one might need to be removed
        # if the transformation is in-place destructive.
        # This depends on the specific handler's logic.
        # For now, let's assume _handle_transform returns the new state, and we might need to remove the original.
        # This requires careful design of interaction handlers.
        # If a transform replaces a pattern, the old one should be removed.
        if interaction_type == InteractionProtocol.TRANSFORM and len(involved_patterns) == 1:
             original_pattern = involved_patterns[0]
             if isinstance(result, AetherPattern) and result != original_pattern:
                 if original_pattern in self.patterns:
                     self.remove_pattern(original_pattern)
                     logger.info(f"Pattern {original_pattern.pattern_id} removed after transformation into {result.pattern_id}.")

        # Other interaction types might have specific ways to handle results.
        # For example, ENTANGLE might modify metadata of existing patterns, not create/remove.
        # The AetherPattern dataclass is frozen, so entangle handler would return NEW patterns with updated metadata.
        # Then those new patterns replace old ones.
        if interaction_type == InteractionProtocol.ENTANGLE:
            if isinstance(result, list) and all(isinstance(p, AetherPattern) for p in result):
                original_positions = {p: self.patterns.get(p) for p in involved_patterns if p in self.patterns}
                for old_p in involved_patterns: 
                    if old_p in self.patterns:
                        self.remove_pattern(old_p)
                for i, new_entangled_p in enumerate(result):
                    # Try to map new pattern to an old one if lists are corresponding
                    # This assumes result[i] is the new state of involved_patterns[i]
                    pos = original_positions.get(involved_patterns[i], initiator_position) if i < len(involved_patterns) else initiator_position
                    if new_entangled_p not in self.patterns: # Check if it's genuinely new or already added
                        self.add_pattern(new_entangled_p, pos)
                        logger.info(f"Pattern {new_entangled_p.pattern_id} (entangled state) added at {pos}.")
                    else: # Pattern might be re-added if already exists, ensure position is correct
                        self.patterns[new_entangled_p] = pos # Update position if necessary

        elif interaction_type == InteractionProtocol.CATALYZE:
            # Handler returns list of new (catalyzed) reactant patterns.
            # Original catalyst (involved_patterns[0]) remains.
            # Original reactants (involved_patterns[1:]) are replaced.
            if isinstance(result, list) and all(isinstance(p, AetherPattern) for p in result):
                original_reactant_patterns = involved_patterns[1:]
                original_positions = {p: self.patterns.get(p) for p in original_reactant_patterns if p in self.patterns}

                for old_reactant in original_reactant_patterns:
                    if old_reactant in self.patterns:
                        self.remove_pattern(old_reactant)
                        logger.info(f"Original reactant pattern {old_reactant.pattern_id} removed post-catalysis.")
                
                for i, new_catalyzed_p in enumerate(result):
                    # Assume result[i] is the new state of original_reactant_patterns[i]
                    # Determine position for the new catalyzed pattern
                    # Default to initiator's position if mapping is difficult or reactant was transient
                    pos = initiator_position 
                    if i < len(original_reactant_patterns):
                        # Try to get position of the original reactant it replaces
                        original_reactant = original_reactant_patterns[i]
                        pos = original_positions.get(original_reactant, initiator_position)
                    
                    if new_catalyzed_p not in self.patterns:
                        self.add_pattern(new_catalyzed_p, pos)
                        logger.info(f"Pattern {new_catalyzed_p.pattern_id} (catalyzed state) added at {pos}.")
                    else: # If somehow it's already there, ensure correct position
                        self.patterns[new_catalyzed_p] = pos


    def cleanup_inactive(self, max_age: float = 3600) -> int:
        """Remove patterns older than max_age seconds"""
        current_time = time.time()
        if current_time - self._last_cleanup < 60:  # Only cleanup every 60 seconds
            return 0
            
        self._last_cleanup = current_time
        old_patterns = {p for p in self.patterns if p.age > max_age}
        
        for pattern in old_patterns:
            self.remove_pattern(pattern)
            
        return len(old_patterns)
    
    def serialize(self) -> Dict[str, Any]:
        """Export space state to dictionary"""
        return {
            'dimensions': self.dimensions,
            'pattern_count': len(self.patterns),
            'interaction_count': len(self.interaction_history)
        }

class AetherEngine:
    """Core encoding system bridging physical laws and manifestations"""
    
    def __init__(self, physics_constraints: Dict = None):
        # Framework constraints
        self.physics = PhysicsConstraints(**physics_constraints) if physics_constraints else PhysicsConstraints()
        
        # Pattern space
        self.space = AetherSpace(dimensions=3)
        
        # Encoding protocol registry
        self.encoders = {
            EncodingType.BINARY: self._encode_binary,
            EncodingType.SYMBOLIC: self._encode_symbolic,
            EncodingType.VOXEL: self._encode_voxel,
            EncodingType.GLYPH: self._encode_glyph,
            EncodingType.QUANTUM: self._encode_quantum,
            EncodingType.FRACTAL: self._encode_fractal,
            EncodingType.WAVE: self._encode_wave
        }
        
        # Interaction protocol registry
        self.interaction_handlers = {
            InteractionProtocol.COMBINE: self._handle_combine,
            InteractionProtocol.ENTANGLE: self._handle_entangle,
            InteractionProtocol.TRANSFORM: self._handle_transform,
            InteractionProtocol.CASCADE: self._handle_cascade,
            InteractionProtocol.RESONATE: self._handle_resonate,
            InteractionProtocol.ANNIHILATE: self._handle_annihilate,
            InteractionProtocol.CATALYZE: self._handle_catalyze
        }
        
        # Performance metrics
        self.metrics = {
            'patterns_created': 0,
            'patterns_mutated': 0,
            'interactions_processed': 0,
            'start_time': time.time()
        }
        
        # Observer callbacks for events
        self.observers: List[Callable[[Dict[str, Any]], None]] = []
        
        logger.info(f"AetherEngine initialized with constraints: {self.physics.get_all_constraints()}")

    def register_observer(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register an observer callback to receive event notifications.
        Ensures that the same callback is not added multiple times.
        Args:
            callback: A callable that takes a dictionary of event data as input.
        """
        if callback not in self.observers:
            self.observers.append(callback)
            logger.info(f"Observer registered: {getattr(callback, '__name__', repr(callback))}")
        else:
            logger.info(f"Observer already registered: {getattr(callback, '__name__', repr(callback))}")

    def unregister_observer(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unregister an observer callback.
        Args:
            callback: The callable to remove from the observer list.
        """
        try:
            self.observers.remove(callback)
            logger.info(f"Observer unregistered: {getattr(callback, '__name__', repr(callback))}")
        except ValueError:
            logger.warning(f"Attempted to unregister a non-existent observer: {getattr(callback, '__name__', repr(callback))}")

    def notify_observers(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Notify all registered observers of a specific event type.

        Args:
            event_type: A string describing the type of event (e.g., "pattern_created", "interaction_processed").
            event_data: A dictionary containing data specific to the event.
        """
        if not self.observers:
            # logger.debug("No observers registered to notify.") # Can be too verbose
            return

        # Prepare a full event payload
        full_event_payload = {
            "event_type": event_type,
            "timestamp": time.time(),
            **event_data  # Merge specific event data
        }
        
        # Iterate over a copy of the list in case observers modify the list during notification
        for callback in list(self.observers): 
            try:
                # Potentially run callbacks in separate threads or use an event queue for robustness
                callback(full_event_payload)
                logger.debug(f"Notified observer {getattr(callback, '__name__', repr(callback))} of event {event_type}")
            except Exception as e:
                logger.error(f"Error notifying observer {getattr(callback, '__name__', repr(callback))} of event {event_type}: {e}", exc_info=True)

    def connect_physics_engine(self, components: Dict[str, Any]) -> None:
        """
        Connect the AetherEngine to other physics components.
        
        Args:
            components: A dictionary of components to connect, such as:
                - 'constants': PhysicsConstants object
                - 'field': QuantumField object
                - 'monte_carlo': QuantumMonteCarlo object
                - 'ethical_manifold': EthicalGravityManifold object
        """
        self.physics_components = components
        logger.info(f"Connected physics components: {list(components.keys())}")

    def create_pattern(self, data: bytes) -> AetherPattern:
        # Create a new pattern
        new_pattern = AetherPattern(core=data, mutations=(), interactions={}, encoding_type=EncodingType.BINARY)
        self.space.add_pattern(new_pattern, position=(0.0, 0.0, 0.0))

        # Notify observers
        self.notify_observers(
            event_type="pattern_created",
            event_data={
                "pattern_id": new_pattern.pattern_id,
                "encoding_type": new_pattern.encoding_type.name,
                "position": (0.0, 0.0, 0.0) # Example position
            }
        )
        return new_pattern

    # --------------------------
    # Encoding Protocols
    # --------------------------
    @lru_cache(maxsize=128)
    def _encode_binary(self, data: bytes) -> bytes:
        """Boolean state representation"""
        if not data:
            raise ValueError("Input data for binary encoding cannot be empty.")
        return hashlib.sha3_512(data).digest()

    def _encode_symbolic(self, data: bytes) -> bytes:
        """Abstract character-based encoding"""
        return data.ljust(self.physics.get('min_pattern_size', 64), b'\x00')

    def _encode_voxel(self, data: bytes) -> bytes:
        """3D volumetric encoding"""
        arr = np.frombuffer(data, dtype=np.uint8)
        size = max(int(len(arr) ** (1/3)) + 1, 1)
        return arr.tobytes().ljust(size**3, b'\x00')

    def _encode_glyph(self, data: bytes) -> bytes:
        """Visual symbolic encoding"""
        return data.center(self.physics.get('min_pattern_size', 64), b'\x01')
    
    def _encode_quantum(self, data: bytes) -> bytes:
        """Probabilistic quantum state encoding"""
        # Generate quantum-like state vector
        seed = int.from_bytes(hashlib.md5(data).digest()[:4], 'big')
        rng = np.random.RandomState(seed)
        
        # Create normalized complex state vector
        state_size = max(64, len(data)) # Ensure a minimum size for the state vector
        
        # Generate real and imaginary parts for a complex vector
        real_parts = rng.normal(0, 1, state_size)
        imag_parts = rng.normal(0, 1, state_size)
        complex_vector = real_parts + 1j * imag_parts
        
        # Normalize the complex vector
        norm = np.linalg.norm(complex_vector)
        if norm == 0: # Avoid division by zero
            norm = 1.0 
        normalized_vector = complex_vector / norm
        
        # Extract magnitudes and phases
        magnitudes = np.abs(normalized_vector)
        phases = np.angle(normalized_vector)
        
        # Serialize magnitudes and phases.
        # Ensure they are in a consistent byte representation.
        # Using float32 for precision and size.
        wave_bytes = np.concatenate((magnitudes.astype(np.float32), phases.astype(np.float32))).tobytes()
        
        # Hash for consistency and to ensure fixed output size
        return hashlib.sha3_512(wave_bytes).digest()

    def _encode_fractal(self, data: bytes) -> bytes:
        """
        Self-similar recursive encoding for fractal patterns.
        This method generates a fractal-like structure by recursively hashing the input data
        and combining the results to create a self-similar pattern.
        
        Args:
            data: Input data to encode as a fractal pattern.
        
        Returns:
            bytes: Encoded fractal pattern.
        """
        # Define recursion depth for fractal generation
        recursion_depth = self.physics.get('max_recursion_depth', 3)

        # Initialize fractal data with the input data
        fractal_data = data

        # Recursively hash and combine data
        for _ in range(recursion_depth):
            # Hash the current fractal data
            hashed_data = hashlib.sha256(fractal_data).digest()

            # Combine the hashed data with the original data
            fractal_data = bytes((a + b) % 256 for a, b in zip(fractal_data, hashed_data))

        # Ensure the result meets the minimum pattern size
        min_size = self.physics.get('min_pattern_size', 64)
        if len(fractal_data) < min_size:
            fractal_data = fractal_data.ljust(min_size, b'\x00')

        return fractal_data

    def _encode_wave(self, data: bytes) -> bytes:
        """
        Frequency/amplitude-based encoding for wave patterns.
        This method generates a wave-like structure by applying a Fourier transform
        to the input data and encoding the result as a frequency/amplitude representation.
        
        Args:
            data: Input data to encode as a wave pattern.
        
        Returns:
            bytes: Encoded wave pattern.
        """
        # Convert input data to a NumPy array
        data_array = np.frombuffer(data, dtype=np.uint8)
        
        # Apply a Fourier transform to generate frequency components
        frequency_components = np.fft.fft(data_array)
        
        # Extract amplitude and phase information
        amplitudes = np.abs(frequency_components)
        phases = np.angle(frequency_components)
        
        # Normalize amplitudes to fit within a byte range
        normalized_amplitudes = (amplitudes / amplitudes.max() * 255).astype(np.uint8)
        
        # Combine amplitudes and phases into a single byte array
        wave_pattern = np.concatenate((normalized_amplitudes, phases)).tobytes()
        
        # Ensure the result meets the minimum pattern size
        min_size = self.physics.get('min_pattern_size', 64)
        if len(wave_pattern) < min_size:
            wave_pattern = wave_pattern.ljust(min_size, b'\x00')
        
        return wave_pattern

    def _handle_combine(self, patterns: List[AetherPattern]) -> AetherPattern:
        """
        Combine multiple patterns into a single new pattern.
        
        Args:
            patterns: List of AetherPattern objects to combine.
        
        Returns:
            AetherPattern: The resulting combined pattern.
        """
        if len(patterns) < 2:
            raise ValueError("At least two patterns are required to combine.")

        # Combine cores by hashing them together
        combined_core = hashlib.sha3_512(b''.join(p.core for p in patterns)).digest()

        # Merge mutations
        combined_mutations = tuple(set(m for p in patterns for m in p.mutations))

        # Merge interactions
        combined_interactions = {}
        for p in patterns:
            combined_interactions.update(p.interactions)

        # Determine the highest recursion level
        max_recursion_level = max(p.recursion_level for p in patterns)

        # Create the combined pattern
        combined_pattern = AetherPattern(
            core=combined_core,
            mutations=combined_mutations,
            interactions=combined_interactions,
            encoding_type=patterns[0].encoding_type,  # Assume all patterns have the same encoding type
            recursion_level=max_recursion_level + 1,
            metadata={"combined_from": [p.pattern_id for p in patterns]}
        )

        # Update metrics
        self.metrics['patterns_created'] += 1
        self.metrics['interactions_processed'] += 1

        logger.info(f"Combined {len(patterns)} patterns into new pattern {combined_pattern.pattern_id}")
        return combined_pattern

    def _handle_entangle(self, patterns: List[AetherPattern]) -> AetherPattern:
        """
        Create quantum-like entanglement between patterns.
        """
        if len(patterns) < 2:
            raise ValueError("At least two patterns are required to entangle.")

        # Generate a shared entanglement key
        entanglement_key = hashlib.sha256(b''.join(p.core for p in patterns)).digest()

        # Update metadata to reflect entanglement
        entangled_patterns = []
        for p in patterns:
            new_metadata = {**p.metadata, 'entanglement_key': entanglement_key.hex()}
            entangled_pattern = AetherPattern(
                core=p.core,
                mutations=p.mutations,
                interactions=p.interactions,
                encoding_type=p.encoding_type,
                recursion_level=p.recursion_level, # Entanglement might not increase recursion
                metadata=new_metadata,
                creation_timestamp=p.creation_timestamp # Preserve original creation time
            )
            entangled_patterns.append(entangled_pattern)
            # Since patterns are replaced, we should notify about the "new" (entangled) patterns
            # This might be better handled by the caller (_process_interaction_result)
            # For now, the handler returns the new states.

        logger.info(f"Entangled {len(patterns)} patterns with key {entanglement_key.hex()}. New pattern instances created.")
        # The _process_interaction_result in AetherSpace needs to handle replacing old patterns with these new ones.
        return entangled_patterns # Return list of new, entangled patterns

    def _handle_transform(self, pattern_or_patterns: Union[AetherPattern, List[AetherPattern]], transformation: Optional[Callable] = None) -> Union[AetherPattern, List[AetherPattern]]:
        """
        Apply a state transformation to a pattern or list of patterns.
        If a list is given, assumes the first is the primary pattern to transform,
        or applies transformation to all if no specific target pattern for transformation.
        For this subtask, the original logic was to transform a single pattern.
        The interaction system calls handlers with List[AetherPattern].
        We'll assume the first pattern in the list is the one to be transformed.
        The `transformation` callable must be provided if this handler is used directly.
        In the context of `AetherSpace.trigger_interactions`, this handler would be called with a list.
        """
        if not isinstance(pattern_or_patterns, list) or not pattern_or_patterns:
            raise ValueError("Input must be a non-empty list of patterns for _handle_transform.")
        
        pattern = pattern_or_patterns[0] # Primary pattern to transform

        if transformation is None:
            # Default transformation: simple re-hash or minor modification
            # This part needs to be defined based on expected default behavior
            def default_transformation(core_data: bytes) -> bytes:
                return hashlib.sha3_256(core_data + b"_transformed").digest()
            transformation = default_transformation
            logger.info(f"No transformation callable provided for pattern {pattern.pattern_id}. Applying default transformation.")
            
        Apply a state transformation to a pattern.
        
        Args:
            pattern: The pattern to transform.
            transformation: A callable that modifies the pattern's core.
        
        Returns:
            AetherPattern: The transformed pattern.
        """
        transformed_core = transformation(pattern.core)

        # Create a new pattern with the transformed core
        # Ensure the core data passed to transformation is appropriate
        transformed_core = transformation(pattern.core)

        transformed_pattern = AetherPattern(
            core=transformed_core,
            mutations=pattern.mutations, # Mutations might also change, or be an outcome of transformation
            interactions=pattern.interactions, # Interactions might also change
            encoding_type=pattern.encoding_type,
            recursion_level=pattern.recursion_level + 1,
            metadata={**pattern.metadata, "transformed_by_rule": transformation.__name__ if hasattr(transformation, '__name__') else "custom_transform"},
            # creation_timestamp will be new for the transformed pattern by default
        )
        
        self.metrics['patterns_mutated'] = self.metrics.get('patterns_mutated', 0) + 1
        self.metrics['interactions_processed'] = self.metrics.get('interactions_processed', 0) + 1

        logger.info(f"Transformed pattern {pattern.pattern_id} into new pattern {transformed_pattern.pattern_id} using {transformation.__name__ if hasattr(transformation, '__name__') else 'custom_transform'}.")
        # This handler now returns the new pattern.
        # The _process_interaction_result in AetherSpace should remove the old pattern.
        return transformed_pattern

    def _handle_cascade(self, patterns: List[AetherPattern]) -> List[AetherPattern]:
        """
        Trigger a chain reaction between patterns.
        
        Args:
            patterns: List of AetherPattern objects to cascade.
        
        Returns:
            List[AetherPattern]: The resulting patterns after the cascade.
        """
        cascaded_patterns = []
        for i, pattern in enumerate(patterns):
            # Create a new pattern for each step in the cascade
            # The cascade effect could be more complex, e.g., affecting nearby patterns or based on specific rules
            # For now, it generates a sequence of new patterns from the first pattern in the list
            source_pattern = patterns[0] # Assume cascade originates from the first pattern if multiple are passed (e.g. for context)
            
            # If other patterns in the list are meant to be part of the cascade chain:
            # This example will make each pattern in the input list generate a successor
            # for i, pattern in enumerate(patterns):
            
            # Current logic: one pattern (first in list) generates a chain.
            # Let's refine to make the input `patterns` list the basis for the cascade
            
            current_core = source_pattern.core
            for i in range(len(patterns)): # Number of steps in cascade can be length of input list or a fixed number
                new_core = hashlib.sha256(current_core + bytes([i]) + source_pattern.pattern_id.encode()).digest()
                
                # Metadata could carry information about the cascade sequence
                cascade_metadata = {
                    **source_pattern.metadata, 
                    "cascade_step": i, 
                    "cascade_origin": source_pattern.pattern_id
                }
                if i > 0 and cascaded_patterns: # Link to previous pattern in this cascade
                    cascade_metadata["cascade_previous_pattern"] = cascaded_patterns[-1].pattern_id

                new_pattern = AetherPattern(
                    core=new_core,
                    mutations=source_pattern.mutations, # Mutations could propagate or change
                    interactions=source_pattern.interactions, # Interactions could also change
                    encoding_type=source_pattern.encoding_type,
                    recursion_level=source_pattern.recursion_level + i + 1, # Recursion increases with each step
                    metadata=cascade_metadata
                    # New creation_timestamp for each new pattern
                )
                cascaded_patterns.append(new_pattern)
                current_core = new_core # Next pattern in cascade builds upon the previous one's core

        self.metrics['patterns_created'] += len(cascaded_patterns)
        self.metrics['interactions_processed'] += 1
        logger.info(f"Cascaded from pattern {patterns[0].pattern_id} into {len(cascaded_patterns)} new patterns.")
        return cascaded_patterns

    def _handle_resonate(self, patterns: List[AetherPattern]) -> AetherPattern:
        """
        Create harmonic resonance between patterns.
        
        Args:
            patterns: List of AetherPattern objects to resonate.
        
        Returns:
            AetherPattern: The resulting resonated pattern.
        """
        if len(patterns) < 2:
            raise ValueError("At least two patterns are required to resonate.")

        # Calculate the harmonic mean of the cores
        combined_core = bytes(
            int(sum(p.core[i] for p in patterns) / len(patterns)) % 256
            for i in range(len(patterns[0].core))
        )

        # Merge mutations and amplify shared mutations
        combined_mutations = tuple(set(m for p in patterns for m in p.mutations))

        # Merge interactions and amplify shared interactions
        combined_interactions = {}
        for p in patterns:
            for key, value in p.interactions.items():
                if key in combined_interactions:
                    combined_interactions[key] = str(
                        float(combined_interactions[key]) + float(value)
                    )
                else:
                    combined_interactions[key] = value

        # Determine the highest recursion level
        max_recursion_level = max(p.recursion_level for p in patterns)

        # Create the resonated pattern
        resonated_pattern = AetherPattern(
            core=combined_core,
            mutations=combined_mutations,
            interactions=combined_interactions,
            encoding_type=patterns[0].encoding_type,  # Assume all patterns have the same encoding type
            recursion_level=max_recursion_level + 1,
            metadata={"resonated_from": [p.pattern_id for p in patterns]}
        )

        # Update metrics
        self.metrics['patterns_created'] += 1
        self.metrics['interactions_processed'] += 1

        logger.info(f"Resonated {len(patterns)} patterns into new pattern {resonated_pattern.pattern_id}")
        return resonated_pattern

    def _handle_annihilate(self, patterns: List[AetherPattern]) -> Optional[AetherPattern]:
        """
        Perform mutual destruction or cancellation of patterns.
        
        Args:
            patterns: List of AetherPattern objects to annihilate.
        
        Returns:
            Optional[AetherPattern]: The resulting annihilated pattern, or None if all are destroyed.
        """
        if len(patterns) < 2:
            raise ValueError("At least two patterns are required to annihilate.")

        # Check if patterns cancel each other out
        annihilation_result = sum(hash(p.core) for p in patterns) % 256 == 0

        if annihilation_result:
            logger.info(f"Patterns {', '.join(p.pattern_id for p in patterns)} annihilated each other.")
            return None  # All patterns are destroyed

        # If not fully annihilated, create a residual pattern
        residual_core = hashlib.sha256(b''.join(p.core for p in patterns)).digest()
        residual_pattern = AetherPattern(
            core=residual_core,
            mutations=(),
            interactions={},
            encoding_type=patterns[0].encoding_type,
            recursion_level=max(p.recursion_level for p in patterns) + 1,
            metadata={"annihilated_from": [p.pattern_id for p in patterns]}
        )

        logger.info(f"Patterns {', '.join(p.pattern_id for p in patterns)} partially annihilated into {residual_pattern.pattern_id}.")
        return residual_pattern

    def _handle_catalyze(self, catalyst: AetherPattern, patterns: List[AetherPattern]) -> List[AetherPattern]:
        """
        Facilitate interactions between patterns using a catalyst.
        
        Args:
            catalyst: The catalyst pattern.
            patterns: List of AetherPattern objects to catalyze.
        
        Returns:
            List[AetherPattern]: The resulting patterns after catalysis.
        """
        if not patterns:
            raise ValueError("At least one pattern is required to catalyze.")

        # Modify patterns based on the catalyst's core
        # The first pattern in the list is the catalyst, the rest are reactants
        if len(patterns) < 2: # Need at least one catalyst and one reactant
            logger.warning("Catalyze interaction requires at least two patterns (catalyst + reactant(s)).")
            return patterns # Or raise error, or return empty list

        catalyst = patterns[0]
        reactants = patterns[1:]
        
        catalyzed_patterns = []
        for reactant_pattern in reactants:
            # Example catalytic effect: XORing cores or a more complex rule
            # Rule: catalyst's core bytes "influence" reactant's core bytes
            # This is a simple example; real catalysis can be very complex.
            catalyzed_core_list = list(reactant_pattern.core)
            catalyst_core_len = len(catalyst.core)
            for i in range(len(catalyzed_core_list)):
                catalyzed_core_list[i] = (catalyzed_core_list[i] + catalyst.core[i % catalyst_core_len]) % 256
            catalyzed_core = bytes(catalyzed_core_list)

            new_metadata = {
                **reactant_pattern.metadata, 
                "catalyzed_by": catalyst.pattern_id,
                "original_pattern_id": reactant_pattern.pattern_id 
            }
            
            # Mutations and interactions could also be affected by the catalyst
            # For example, catalyst might add specific new mutation paths or enable interactions
            new_mutations = reactant_pattern.mutations + tuple([m for m in catalyst.mutations if m.startswith(b"catalyst_effect:")])
            new_interactions = {**reactant_pattern.interactions, **catalyst.interactions}


            catalyzed_pattern = AetherPattern(
                core=catalyzed_core,
                mutations=new_mutations, # Or some combination/modification
                interactions=new_interactions, # Or some combination
                encoding_type=reactant_pattern.encoding_type,
                recursion_level=reactant_pattern.recursion_level + 1, # Catalysis implies a transformation
                metadata=new_metadata
            )
            catalyzed_patterns.append(catalyzed_pattern)
            self.metrics['patterns_mutated'] += 1

        self.metrics['interactions_processed'] += 1
        logger.info(f"Catalyst {catalyst.pattern_id} transformed {len(reactants)} patterns into {len(catalyzed_patterns)} new patterns.")
        # The original reactant patterns should be replaced by these catalyzed_patterns.
        # This list of new patterns is returned to _process_interaction_result.
        return catalyzed_patterns

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the engine's state to a dictionary.
        
        Returns:
            Dict[str, Any]: Serialized state of the engine.
        """
        return {
            "physics_constraints": self.physics.__dict__,
            "metrics": self.metrics,
            "patterns": [p.serialize() for p in self.space.patterns],
            "observers": len(self.observers)
        }