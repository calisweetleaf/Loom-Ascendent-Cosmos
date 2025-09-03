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
import threading
from abc import ABC, abstractmethod

# ================================================================
# CUSTOM EXCEPTION CLASSES FOR AETHER OPERATIONS
# ================================================================

class AetherEngineError(Exception):
    """Base exception for all Aether Engine related errors"""
    pass

class PatternValidationError(AetherEngineError):
    """Raised when pattern validation fails"""
    pass

class EncodingError(AetherEngineError):
    """Raised when pattern encoding fails"""
    pass

class InteractionError(AetherEngineError):
    """Raised when pattern interaction fails"""
    pass

class MutationError(AetherEngineError):
    """Raised when pattern mutation fails"""
    pass

class RecursionLimitError(AetherEngineError):
    """Raised when recursion depth exceeds allowed limits"""
    pass

class PhysicsConstraintViolation(AetherEngineError):
    """Raised when operation violates physics constraints"""
    pass

class PatternCompatibilityError(AetherEngineError):
    """Raised when patterns are incompatible for interaction"""
    pass

class ThreadSafetyError(AetherEngineError):
    """Raised when thread safety operations fail"""
    pass

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

class AetherSpace:
    """Multi-dimensional manifold containing patterns with thread safety"""
    def __init__(self, dimensions: int = 3):
        self.dimensions = dimensions
        self.patterns = set()
        self.spatial_index = defaultdict(set)  # Simple spatial partitioning
        self.interaction_history = []
        self._last_cleanup = time.time()
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._pattern_positions = {}  # Track pattern positions for spatial queries
    
    def add_pattern(self, pattern: AetherPattern, position: Tuple[float, ...]) -> None:
        """Add pattern to space at specific position (thread-safe)"""
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} dimensions")
            
        with self._lock:
            try:
                self.patterns.add(pattern)
                self._pattern_positions[pattern.pattern_id] = position
                
                # Add to spatial index (coarse grid)
                grid_pos = tuple(int(p/10) for p in position)
                self.spatial_index[grid_pos].add(pattern)
                
                logger.debug(f"Added pattern {pattern.pattern_id} at position {position}")
            except Exception as e:
                logger.error(f"Failed to add pattern {pattern.pattern_id}: {e}")
                raise ThreadSafetyError(f"Failed to add pattern: {e}")
    
    def remove_pattern(self, pattern: AetherPattern) -> None:
        """Remove pattern from space (thread-safe)"""
        with self._lock:
            try:
                if pattern in self.patterns:
                    self.patterns.remove(pattern)
                    
                    # Remove from position tracking
                    if pattern.pattern_id in self._pattern_positions:
                        del self._pattern_positions[pattern.pattern_id]
                    
                    # Remove from spatial index
                    for cell, patterns in list(self.spatial_index.items()):
                        if pattern in patterns:
                            patterns.remove(pattern)
                            if not patterns:
                                del self.spatial_index[cell]
                    
                    logger.debug(f"Removed pattern {pattern.pattern_id}")
            except Exception as e:
                logger.error(f"Failed to remove pattern {pattern.pattern_id}: {e}")
                raise ThreadSafetyError(f"Failed to remove pattern: {e}")
    
    def get_nearby_patterns(self, position: Tuple[float, ...], radius: float) -> Set[AetherPattern]:
        """Find patterns within radius of position (thread-safe)"""
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} dimensions")
            
        with self._lock:
            try:
                # Get grid cells that might contain nearby patterns
                nearby = set()
                grid_pos = tuple(int(p/10) for p in position)
                
                # Check surrounding grid cells
                for offset in self._generate_neighbor_offsets(radius/10):
                    neighbor_pos = tuple(g + o for g, o in zip(grid_pos, offset))
                    if neighbor_pos in self.spatial_index:
                        nearby.update(self.spatial_index[neighbor_pos])
                
                return nearby
            except Exception as e:
                logger.error(f"Failed to get nearby patterns: {e}")
                raise ThreadSafetyError(f"Failed to get nearby patterns: {e}")
    
    def _generate_neighbor_offsets(self, radius: int) -> List[Tuple[int, ...]]:
        """Generate offsets for neighboring grid cells"""
        # For simplicity, just return a cube of offsets
        offsets = []
        for i in range(-int(radius), int(radius)+1):
            for j in range(-int(radius), int(radius)+1):
                if self.dimensions == 2:
                    offsets.append((i, j))
                elif self.dimensions == 3:
                    for k in range(-int(radius), int(radius)+1):
                        offsets.append((i, j, k))
        return offsets
    
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
        
        # Thread safety locks
        self._engine_lock = threading.RLock()  # Main engine lock
        self._metrics_lock = threading.Lock()  # Metrics-specific lock
        self._observer_lock = threading.Lock()  # Observer-specific lock
        
        # Pattern space
        self.space = AetherSpace(dimensions=3)
        
        # Pattern cache for performance optimization
        self._pattern_cache = {}
        self._cache_lock = threading.Lock()
        
        # Physics component connections
        self.physics_components = {}
        
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
            'patterns_validated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'start_time': time.time()
        }
        
        # Observer callbacks for events
        self.observers = []  # Initialize an empty list of observers
        
        logger.info(f"AetherEngine initialized with constraints: {self.physics.__dict__}")

    def register_observer(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register an observer callback to receive event notifications.

        Args:
            callback: A callable that takes a dictionary of event data as input.
        """
        self.observers.append(callback)
        logger.info(f"Observer registered: {callback.__name__}")

    def notify_observers(self, event_data: Dict[str, Any]) -> None:
        """
        Notify all registered observers of an event.

        Args:
            event_data: A dictionary containing event data to pass to observers.
        """
        if not self.observers:
            logger.warning("No observers registered to notify.")
            return

        for callback in self.observers:
            try:
                callback(event_data)
                logger.info(f"Notified observer: {callback.__name__} with event data: {event_data}")
            except Exception as e:
                logger.error(f"Error notifying observer {callback.__name__}: {e}")

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
        event_data = {
            "event_type": "pattern_created",
            "pattern": new_pattern
        }
        self.notify_observers(event_data)

        return new_pattern

    def mutate_pattern(self, pattern: AetherPattern, mutation_vector: bytes, 
                      conscious_agent_id: Optional[str] = None) -> AetherPattern:
        """
        Apply mutation vector to pattern according to Genesis Framework.
        
        Args:
            pattern: The pattern to mutate
            mutation_vector: Byte sequence defining the mutation
            conscious_agent_id: Optional ID of conscious agent performing mutation
            
        Returns:
            AetherPattern: The mutated pattern
            
        Raises:
            MutationError: If mutation fails validation
            RecursionLimitError: If recursion depth exceeds limits
        """
        with self._engine_lock:
            try:
                # Validate recursion depth
                max_depth = self.physics.get('max_recursion_depth', 10)
                if pattern.recursion_level >= max_depth:
                    raise RecursionLimitError(f"Pattern recursion depth {pattern.recursion_level} exceeds limit {max_depth}")
                
                # Validate mutation vector
                if not mutation_vector:
                    raise MutationError("Mutation vector cannot be empty")
                
                # Apply mutation using XOR operation for controlled modification
                mutated_core = bytes(a ^ b for a, b in 
                                   zip(pattern.core, 
                                       (mutation_vector * (len(pattern.core) // len(mutation_vector) + 1))[:len(pattern.core)]))
                
                # Create new mutations tuple including the applied mutation
                new_mutations = pattern.mutations + (mutation_vector,)
                
                # Update metadata with mutation info
                mutation_metadata = {
                    **pattern.metadata,
                    'last_mutation': time.time(),
                    'mutation_count': len(new_mutations),
                    'mutated_by': conscious_agent_id or 'system'
                }
                
                # Create mutated pattern
                mutated_pattern = AetherPattern(
                    core=mutated_core,
                    mutations=new_mutations,
                    interactions=pattern.interactions.copy(),
                    encoding_type=pattern.encoding_type,
                    recursion_level=pattern.recursion_level + 1,
                    metadata=mutation_metadata
                )
                
                # Validate the mutated pattern
                if not self.validate_pattern_integrity(mutated_pattern):
                    raise MutationError("Mutated pattern failed integrity validation")
                
                # Update metrics
                with self._metrics_lock:
                    self.metrics['patterns_mutated'] += 1
                
                # Notify observers
                self.notify_observers({
                    'event_type': 'pattern_mutated',
                    'original_pattern_id': pattern.pattern_id,
                    'mutated_pattern_id': mutated_pattern.pattern_id,
                    'conscious_agent_id': conscious_agent_id,
                    'mutation_size': len(mutation_vector)
                })
                
                logger.info(f"Successfully mutated pattern {pattern.pattern_id} -> {mutated_pattern.pattern_id}")
                return mutated_pattern
                
            except Exception as e:
                logger.error(f"Pattern mutation failed: {e}")
                raise MutationError(f"Pattern mutation failed: {e}")

    def validate_pattern_integrity(self, pattern: AetherPattern) -> bool:
        """
        Comprehensive validation of pattern integrity and consistency.
        
        Args:
            pattern: The pattern to validate
            
        Returns:
            bool: True if pattern passes all validation checks
            
        Raises:
            PatternValidationError: If validation encounters critical errors
        """
        try:
            with self._metrics_lock:
                self.metrics['patterns_validated'] += 1
            
            # Check core integrity
            if not pattern.core:
                logger.warning(f"Pattern {pattern.pattern_id} has empty core")
                return False
            
            # Validate minimum size requirements
            min_size = self.physics.get('min_pattern_size', 32)
            if len(pattern.core) < min_size:
                logger.warning(f"Pattern {pattern.pattern_id} core size {len(pattern.core)} below minimum {min_size}")
                return False
            
            # Check maximum complexity
            max_complexity = self.physics.get('max_pattern_complexity', 1000.0)
            if pattern.complexity > max_complexity:
                logger.warning(f"Pattern {pattern.pattern_id} complexity {pattern.complexity} exceeds maximum {max_complexity}")
                return False
            
            # Validate encoding type
            if not isinstance(pattern.encoding_type, EncodingType):
                logger.error(f"Pattern {pattern.pattern_id} has invalid encoding type")
                return False
            
            # Check recursion depth limits
            max_depth = self.physics.get('max_recursion_depth', 10)
            if pattern.recursion_level > max_depth:
                logger.warning(f"Pattern {pattern.pattern_id} recursion level {pattern.recursion_level} exceeds maximum {max_depth}")
                return False
            
            # Validate mutation consistency
            for i, mutation in enumerate(pattern.mutations):
                if not isinstance(mutation, bytes) or len(mutation) == 0:
                    logger.error(f"Pattern {pattern.pattern_id} has invalid mutation at index {i}")
                    return False
            
            # Check interaction protocol consistency
            for protocol, signature in pattern.interactions.items():
                if not isinstance(protocol, str) or not isinstance(signature, str):
                    logger.error(f"Pattern {pattern.pattern_id} has invalid interaction protocol")
                    return False
            
            # Validate against physics constraints if available
            if hasattr(self, 'physics_components') and 'constants' in self.physics_components:
                constants = self.physics_components['constants']
                # Add physics-based validation here if needed
            
            logger.debug(f"Pattern {pattern.pattern_id} passed integrity validation")
            return True
            
        except Exception as e:
            logger.error(f"Pattern validation failed: {e}")
            raise PatternValidationError(f"Pattern validation failed: {e}")

    def apply_recursive_modification(self, pattern: AetherPattern, 
                                   modification_func: Callable[[bytes], bytes],
                                   conscious_agent_id: str,
                                   permission_level: str = "user") -> AetherPattern:
        """
        Apply recursive modification with proper permission control.
        
        Args:
            pattern: The pattern to modify
            modification_func: Function to apply to pattern core
            conscious_agent_id: ID of the conscious agent requesting modification
            permission_level: Permission level ("user", "admin", "system")
            
        Returns:
            AetherPattern: The recursively modified pattern
            
        Raises:
            RecursionLimitError: If recursion depth exceeds limits
            PermissionError: If agent lacks required permissions
        """
        with self._engine_lock:
            try:
                # Check permissions
                allowed_levels = {"system", "admin"}
                if permission_level == "user":
                    # Users can only modify patterns they created or own
                    pattern_creator = pattern.metadata.get('created_by', '')
                    if pattern_creator != conscious_agent_id:
                        raise PermissionError(f"Agent {conscious_agent_id} lacks permission to modify pattern {pattern.pattern_id}")
                elif permission_level not in allowed_levels:
                    raise PermissionError(f"Invalid permission level: {permission_level}")
                
                # Check recursion limits
                max_depth = self.physics.get('max_recursion_depth', 10)
                if pattern.recursion_level >= max_depth:
                    raise RecursionLimitError(f"Cannot modify pattern at recursion depth {pattern.recursion_level}")
                
                # Apply modification function
                try:
                    modified_core = modification_func(pattern.core)
                except Exception as e:
                    raise MutationError(f"Modification function failed: {e}")
                
                # Create recursively modified pattern
                modified_pattern = AetherPattern(
                    core=modified_core,
                    mutations=pattern.mutations,
                    interactions=pattern.interactions.copy(),
                    encoding_type=pattern.encoding_type,
                    recursion_level=pattern.recursion_level + 1,
                    metadata={
                        **pattern.metadata,
                        'last_modified': time.time(),
                        'modified_by': conscious_agent_id,
                        'permission_level': permission_level,
                        'recursive_modification': True
                    }
                )
                
                # Validate the modified pattern
                if not self.validate_pattern_integrity(modified_pattern):
                    raise MutationError("Recursively modified pattern failed integrity validation")
                
                # Add to space if original was in space
                if pattern in self.space.patterns:
                    original_position = self.space._pattern_positions.get(pattern.pattern_id, (0.0, 0.0, 0.0))
                    self.space.add_pattern(modified_pattern, original_position)
                
                # Update metrics
                with self._metrics_lock:
                    self.metrics['patterns_mutated'] += 1
                
                # Notify observers
                self.notify_observers({
                    'event_type': 'recursive_modification_applied',
                    'original_pattern_id': pattern.pattern_id,
                    'modified_pattern_id': modified_pattern.pattern_id,
                    'conscious_agent_id': conscious_agent_id,
                    'permission_level': permission_level
                })
                
                logger.info(f"Applied recursive modification to pattern {pattern.pattern_id} by agent {conscious_agent_id}")
                return modified_pattern
                
            except Exception as e:
                logger.error(f"Recursive modification failed: {e}")
                if isinstance(e, (RecursionLimitError, PermissionError, MutationError)):
                    raise
                else:
                    raise MutationError(f"Recursive modification failed: {e}")

    def batch_process_patterns(self, patterns: List[AetherPattern], 
                             operations: List[Callable], 
                             max_workers: int = 4) -> List[AetherPattern]:
        """
        Process multiple patterns in parallel for performance optimization.
        
        Args:
            patterns: List of patterns to process
            operations: List of functions to apply to each pattern
            max_workers: Maximum number of worker threads
            
        Returns:
            List[AetherPattern]: List of processed patterns
            
        Raises:
            InteractionError: If batch processing fails
        """
        if not patterns:
            return []
        
        if len(operations) != len(patterns):
            raise ValueError("Number of operations must match number of patterns")
        
        processed_patterns = []
        failed_operations = []
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_pattern = {
                    executor.submit(self._safe_pattern_operation, pattern, operation): (pattern, operation)
                    for pattern, operation in zip(patterns, operations)
                }
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_pattern):
                    pattern, operation = future_to_pattern[future]
                    try:
                        result = future.result(timeout=30)  # 30 second timeout per operation
                        if result is not None:
                            processed_patterns.append(result)
                        else:
                            failed_operations.append((pattern.pattern_id, "Operation returned None"))
                    except Exception as e:
                        failed_operations.append((pattern.pattern_id, str(e)))
                        logger.error(f"Batch operation failed for pattern {pattern.pattern_id}: {e}")
            
            # Log batch processing results
            success_count = len(processed_patterns)
            failure_count = len(failed_operations)
            logger.info(f"Batch processing completed: {success_count} succeeded, {failure_count} failed")
            
            if failed_operations:
                logger.warning(f"Failed operations: {failed_operations}")
            
            # Notify observers
            self.notify_observers({
                'event_type': 'batch_processing_completed',
                'total_patterns': len(patterns),
                'successful_operations': success_count,
                'failed_operations': failure_count,
                'processed_patterns': [p.pattern_id for p in processed_patterns]
            })
            
            return processed_patterns
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise InteractionError(f"Batch processing failed: {e}")

    def _safe_pattern_operation(self, pattern: AetherPattern, operation: Callable) -> Optional[AetherPattern]:
        """
        Safely execute a pattern operation with error handling.
        
        Args:
            pattern: Pattern to operate on
            operation: Operation to apply
            
        Returns:
            Optional[AetherPattern]: Result pattern or None if failed
        """
        try:
            # Validate pattern before operation
            if not self.validate_pattern_integrity(pattern):
                logger.warning(f"Pattern {pattern.pattern_id} failed pre-operation validation")
                return None
            
            # Apply operation
            result = operation(pattern)
            
            # Validate result
            if isinstance(result, AetherPattern) and self.validate_pattern_integrity(result):
                return result
            else:
                logger.warning(f"Operation result for pattern {pattern.pattern_id} failed validation")
                return None
                
        except Exception as e:
            logger.error(f"Safe pattern operation failed for {pattern.pattern_id}: {e}")
            return None

    def cleanup_expired_patterns(self, max_age_seconds: float = 3600, 
                                max_complexity_threshold: float = 500.0) -> int:
        """
        Clean up expired and overly complex patterns for memory management.
        
        Args:
            max_age_seconds: Maximum age in seconds before cleanup
            max_complexity_threshold: Maximum complexity before cleanup
            
        Returns:
            int: Number of patterns cleaned up
        """
        with self._engine_lock:
            try:
                current_time = time.time()
                patterns_to_remove = []
                
                # Find expired patterns
                for pattern in list(self.space.patterns):
                    should_remove = False
                    
                    # Check age
                    if pattern.age > max_age_seconds:
                        should_remove = True
                        logger.debug(f"Pattern {pattern.pattern_id} expired (age: {pattern.age:.2f}s)")
                    
                    # Check complexity
                    elif pattern.complexity > max_complexity_threshold:
                        should_remove = True
                        logger.debug(f"Pattern {pattern.pattern_id} too complex (complexity: {pattern.complexity:.2f})")
                    
                    # Check if pattern is orphaned (no interactions recently)
                    elif (pattern.metadata.get('last_interaction', pattern.creation_timestamp) + 
                          max_age_seconds < current_time):
                        should_remove = True
                        logger.debug(f"Pattern {pattern.pattern_id} is orphaned")
                    
                    if should_remove:
                        patterns_to_remove.append(pattern)
                
                # Remove identified patterns
                for pattern in patterns_to_remove:
                    self.space.remove_pattern(pattern)
                    
                    # Remove from cache if present
                    with self._cache_lock:
                        if pattern.pattern_id in self._pattern_cache:
                            del self._pattern_cache[pattern.pattern_id]
                
                # Update metrics
                cleanup_count = len(patterns_to_remove)
                logger.info(f"Cleaned up {cleanup_count} expired/complex patterns")
                
                # Notify observers
                if cleanup_count > 0:
                    self.notify_observers({
                        'event_type': 'patterns_cleaned_up',
                        'cleanup_count': cleanup_count,
                        'cleanup_criteria': {
                            'max_age_seconds': max_age_seconds,
                            'max_complexity_threshold': max_complexity_threshold
                        }
                    })
                
                return cleanup_count
                
            except Exception as e:
                logger.error(f"Pattern cleanup failed: {e}")
                raise AetherEngineError(f"Pattern cleanup failed: {e}")

    def calculate_pattern_compatibility(self, pattern1: AetherPattern, 
                                      pattern2: AetherPattern) -> float:
        """
        Calculate compatibility score between two patterns for interaction validation.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            float: Compatibility score between 0.0 (incompatible) and 1.0 (fully compatible)
            
        Raises:
            PatternCompatibilityError: If compatibility calculation fails
        """
        try:
            # Initialize compatibility score
            compatibility = 0.0
            
            # Encoding type compatibility (40% weight)
            if pattern1.encoding_type == pattern2.encoding_type:
                compatibility += 0.4
            elif self._are_encoding_types_compatible(pattern1.encoding_type, pattern2.encoding_type):
                compatibility += 0.2
            
            # Complexity compatibility (20% weight)
            complexity_diff = abs(pattern1.complexity - pattern2.complexity)
            max_complexity = max(pattern1.complexity, pattern2.complexity)
            if max_complexity > 0:
                complexity_compatibility = 1.0 - (complexity_diff / max_complexity)
                compatibility += 0.2 * max(0.0, complexity_compatibility)
            
            # Interaction protocol overlap (20% weight)
            common_protocols = set(pattern1.interactions.keys()) & set(pattern2.interactions.keys())
            total_protocols = len(set(pattern1.interactions.keys()) | set(pattern2.interactions.keys()))
            if total_protocols > 0:
                protocol_compatibility = len(common_protocols) / total_protocols
                compatibility += 0.2 * protocol_compatibility
            
            # Mutation compatibility (10% weight)
            common_mutations = set(pattern1.mutations) & set(pattern2.mutations)
            total_mutations = len(set(pattern1.mutations) | set(pattern2.mutations))
            if total_mutations > 0:
                mutation_compatibility = len(common_mutations) / total_mutations
                compatibility += 0.1 * mutation_compatibility
            
            # Recursion level compatibility (10% weight)
            recursion_diff = abs(pattern1.recursion_level - pattern2.recursion_level)
            max_recursion = self.physics.get('max_recursion_depth', 10)
            recursion_compatibility = 1.0 - (recursion_diff / max_recursion)
            compatibility += 0.1 * max(0.0, recursion_compatibility)
            
            # Ensure compatibility is within bounds
            compatibility = max(0.0, min(1.0, compatibility))
            
            logger.debug(f"Compatibility between {pattern1.pattern_id} and {pattern2.pattern_id}: {compatibility:.3f}")
            return compatibility
            
        except Exception as e:
            logger.error(f"Pattern compatibility calculation failed: {e}")
            raise PatternCompatibilityError(f"Pattern compatibility calculation failed: {e}")

    def _are_encoding_types_compatible(self, type1: EncodingType, type2: EncodingType) -> bool:
        """Check if two encoding types are compatible for interactions"""
        # Define compatible encoding type pairs
        compatible_pairs = {
            (EncodingType.BINARY, EncodingType.SYMBOLIC),
            (EncodingType.VOXEL, EncodingType.FRACTAL),
            (EncodingType.QUANTUM, EncodingType.WAVE),
            (EncodingType.GLYPH, EncodingType.SYMBOLIC)
        }
        
        return (type1, type2) in compatible_pairs or (type2, type1) in compatible_pairs

    def get_cached_pattern(self, pattern_id: str) -> Optional[AetherPattern]:
        """
        Retrieve pattern from cache for performance optimization.
        
        Args:
            pattern_id: ID of pattern to retrieve
            
        Returns:
            Optional[AetherPattern]: Cached pattern or None if not found
        """
        with self._cache_lock:
            if pattern_id in self._pattern_cache:
                with self._metrics_lock:
                    self.metrics['cache_hits'] += 1
                logger.debug(f"Cache hit for pattern {pattern_id}")
                return self._pattern_cache[pattern_id]
            else:
                with self._metrics_lock:
                    self.metrics['cache_misses'] += 1
                logger.debug(f"Cache miss for pattern {pattern_id}")
                return None

    def cache_pattern(self, pattern: AetherPattern, max_cache_size: int = 1000) -> None:
        """
        Cache pattern for performance optimization.
        
        Args:
            pattern: Pattern to cache
            max_cache_size: Maximum number of patterns to cache
        """
        with self._cache_lock:
            # Remove oldest patterns if cache is full
            if len(self._pattern_cache) >= max_cache_size:
                # Remove 20% of oldest entries
                remove_count = max_cache_size // 5
                oldest_patterns = sorted(
                    self._pattern_cache.items(),
                    key=lambda x: x[1].creation_timestamp
                )[:remove_count]
                
                for pattern_id, _ in oldest_patterns:
                    del self._pattern_cache[pattern_id]
                
                logger.debug(f"Evicted {remove_count} patterns from cache")
            
            self._pattern_cache[pattern.pattern_id] = pattern
            logger.debug(f"Cached pattern {pattern.pattern_id}")

    # --------------------------
    # Encoding Protocols
    # --------------------------
    @lru_cache(maxsize=128)
    def _encode_binary(self, data: bytes) -> bytes:
        """Boolean state representation with comprehensive error handling"""
        try:
            if not data:
                raise EncodingError("Input data for binary encoding cannot be empty")
            
            if len(data) > self.physics.get('max_pattern_size', 1048576):  # 1MB limit
                raise EncodingError(f"Input data size {len(data)} exceeds maximum allowed size")
            
            result = hashlib.sha3_512(data).digest()
            logger.debug(f"Binary encoding completed for {len(data)} bytes -> {len(result)} bytes")
            return result
            
        except Exception as e:
            logger.error(f"Binary encoding failed: {e}")
            raise EncodingError(f"Binary encoding failed: {e}")

    def _encode_symbolic(self, data: bytes) -> bytes:
        """Abstract character-based encoding with validation"""
        try:
            if not data:
                raise EncodingError("Input data for symbolic encoding cannot be empty")
            
            min_size = self.physics.get('min_pattern_size', 64)
            max_size = self.physics.get('max_pattern_size', 1048576)
            
            if len(data) > max_size:
                raise EncodingError(f"Input data size {len(data)} exceeds maximum {max_size}")
            
            result = data.ljust(min_size, b'\x00')[:max_size]  # Truncate if too large
            logger.debug(f"Symbolic encoding completed: {len(data)} -> {len(result)} bytes")
            return result
            
        except Exception as e:
            logger.error(f"Symbolic encoding failed: {e}")
            raise EncodingError(f"Symbolic encoding failed: {e}")

    def _encode_voxel(self, data: bytes) -> bytes:
        """3D volumetric encoding with bounds checking and proper array handling"""
        try:
            if not data:
                raise EncodingError("Input data for voxel encoding cannot be empty")
            
            # Convert to numpy array with proper error handling
            try:
                arr = np.frombuffer(data, dtype=np.uint8)
            except ValueError as e:
                raise EncodingError(f"Failed to convert data to numpy array: {e}")
            
            if len(arr) == 0:
                raise EncodingError("Cannot create voxel array from empty data")
            
            # Calculate optimal cube size with performance considerations
            cube_size = max(int(np.ceil(len(arr) ** (1/3))), 2)  # Minimum cube size of 2
            max_cube_size = self.physics.get('max_voxel_dimension', 100)
            
            if cube_size > max_cube_size:
                logger.warning(f"Voxel cube size {cube_size} exceeds maximum {max_cube_size}, clamping")
                cube_size = max_cube_size
            
            target_size = cube_size ** 3
            
            # Reshape data to fit cube dimensions with proper padding/truncation
            if len(arr) >= target_size:
                # Truncate if data is larger than target
                voxel_data = arr[:target_size]
            else:
                # Pad with pattern-based filling instead of zeros for better distribution
                padding_pattern = arr[:min(len(arr), 256)]  # Use first 256 bytes as pattern
                pad_length = target_size - len(arr)
                
                # Create repeating pattern for padding
                pattern_repeats = (pad_length // len(padding_pattern)) + 1
                padding = np.tile(padding_pattern, pattern_repeats)[:pad_length]
                
                voxel_data = np.concatenate([arr, padding])
            
            # Reshape to 3D cube and apply volumetric transformation
            try:
                voxel_cube = voxel_data.reshape(cube_size, cube_size, cube_size)
                
                # Apply 3D convolution-like operation for spatial encoding
                kernel = np.array([[[0.125, 0.25, 0.125],
                                  [0.25, 0.5, 0.25],
                                  [0.125, 0.25, 0.125]]])
                
                # Simple 3D filtering operation
                filtered_cube = np.zeros_like(voxel_cube, dtype=np.float32)
                for i in range(1, cube_size-1):
                    for j in range(1, cube_size-1):
                        for k in range(1, cube_size-1):
                            neighborhood = voxel_cube[i-1:i+2, j-1:j+2, k-1:k+2].astype(np.float32)
                            filtered_cube[i, j, k] = np.sum(neighborhood * kernel)
                
                # Convert back to uint8 and flatten
                result_data = np.clip(filtered_cube, 0, 255).astype(np.uint8).tobytes()
                
            except (ValueError, MemoryError) as e:
                # Fallback to simple reshape if 3D operations fail
                logger.warning(f"3D operations failed, using simple reshape: {e}")
                result_data = voxel_data.tobytes()
            
            logger.debug(f"Voxel encoding completed: {len(data)} -> {len(result_data)} bytes ({cube_size}³)")
            return result_data
            
        except Exception as e:
            logger.error(f"Voxel encoding failed: {e}")
            raise EncodingError(f"Voxel encoding failed: {e}")

    def _encode_glyph(self, data: bytes) -> bytes:
        """Visual symbolic encoding with advanced glyph generation"""
        try:
            if not data:
                raise EncodingError("Input data for glyph encoding cannot be empty")
            
            target_size = self.physics.get('min_pattern_size', 64)
            max_size = self.physics.get('max_pattern_size', 1048576)
            
            if len(data) > max_size:
                raise EncodingError(f"Input data size {len(data)} exceeds maximum {max_size}")
            
            # Generate glyph symbols based on data entropy
            data_array = np.frombuffer(data, dtype=np.uint8)
            entropy_map = self._calculate_local_entropy(data_array)
            
            # Create glyph patterns based on entropy distribution
            glyph_patterns = {
                'low_entropy': b'\x01\x01\x02\x02',    # Repetitive pattern
                'med_entropy': b'\x03\x05\x07\x0B',    # Prime sequence
                'high_entropy': b'\x0F\x33\x55\xAA',  # Alternating bits
                'ultra_entropy': b'\xFF\x00\x5A\xA5'  # Maximum contrast
            }
            
            # Apply entropy-based glyph mapping
            glyph_data = bytearray()
            chunk_size = max(1, len(data_array) // target_size) if len(data_array) > target_size else 1
            
            for i in range(0, len(data_array), chunk_size):
                chunk = data_array[i:i + chunk_size]
                chunk_entropy = np.std(chunk.astype(np.float32))
                
                if chunk_entropy < 10:
                    pattern = glyph_patterns['low_entropy']
                elif chunk_entropy < 30:
                    pattern = glyph_patterns['med_entropy']
                elif chunk_entropy < 60:
                    pattern = glyph_patterns['high_entropy']
                else:
                    pattern = glyph_patterns['ultra_entropy']
                
                # XOR chunk data with selected pattern
                for j, byte_val in enumerate(chunk):
                    glyph_data.append(byte_val ^ pattern[j % len(pattern)])
            
            # Ensure target size with sophisticated padding
            if len(glyph_data) < target_size:
                # Create fractal padding pattern
                seed_pattern = bytes(glyph_data[-min(16, len(glyph_data)):])
                padding_needed = target_size - len(glyph_data)
                
                # Generate fractal-like padding
                fractal_padding = self._generate_fractal_padding(seed_pattern, padding_needed)
                glyph_data.extend(fractal_padding)
            
            elif len(glyph_data) > target_size:
                # Compress using sliding window with overlap preservation
                compression_ratio = len(glyph_data) / target_size
                compressed_data = bytearray()
                
                for i in range(target_size):
                    source_idx = int(i * compression_ratio)
                    if source_idx < len(glyph_data):
                        compressed_data.append(glyph_data[source_idx])
                    else:
                        compressed_data.append(0)
                
                glyph_data = compressed_data
            
            result = bytes(glyph_data[:target_size])
            logger.debug(f"Glyph encoding completed: {len(data)} -> {len(result)} bytes")
            return result
            
        except Exception as e:
            logger.error(f"Glyph encoding failed: {e}")
            raise EncodingError(f"Glyph encoding failed: {e}")
    
    def _encode_quantum(self, data: bytes) -> bytes:
        """Advanced quantum state encoding with proper normalization"""
        try:
            if not data:
                raise EncodingError("Input data for quantum encoding cannot be empty")
            
            # Generate deterministic quantum-like state vector
            seed = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')
            rng = np.random.RandomState(seed % (2**32))  # Ensure valid seed
            
            # Dynamic state size based on data complexity
            data_complexity = len(set(data)) / 256.0  # Normalized complexity
            base_size = 128
            complexity_factor = max(0.5, min(2.0, data_complexity * 2))
            state_size = int(base_size * complexity_factor)
            state_size = min(state_size, self.physics.get('max_quantum_states', 2048))
            
            try:
                # Generate complex quantum amplitudes
                real_parts = rng.normal(0, 1, state_size)
                imag_parts = rng.normal(0, 1, state_size)
                
                # Create complex amplitudes
                amplitudes = real_parts + 1j * imag_parts
                
                # Proper quantum state normalization
                norm_squared = np.sum(np.abs(amplitudes) ** 2)
                if norm_squared > 0:
                    amplitudes = amplitudes / np.sqrt(norm_squared)
                else:
                    # Fallback to uniform superposition
                    amplitudes = np.ones(state_size, dtype=complex) / np.sqrt(state_size)
                
                # Extract phases and magnitudes
                phases = np.angle(amplitudes)
                magnitudes = np.abs(amplitudes)
                
                # Quantum entanglement simulation through correlation matrix
                correlation_matrix = np.outer(magnitudes, magnitudes)
                eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
                
                # Use top eigenvalues for encoding
                top_k = min(32, len(eigenvals))
                top_eigenvals = eigenvals[-top_k:]
                
                # Combine quantum features into byte representation
                quantum_features = np.concatenate([
                    magnitudes,
                    phases,
                    top_eigenvals
                ])
                
                # Normalize to byte range with quantum discretization
                feature_min, feature_max = quantum_features.min(), quantum_features.max()
                if feature_max > feature_min:
                    normalized_features = ((quantum_features - feature_min) / 
                                         (feature_max - feature_min) * 255).astype(np.uint8)
                else:
                    normalized_features = np.full(len(quantum_features), 128, dtype=np.uint8)
                
                # Apply quantum hash for final encoding
                quantum_bytes = normalized_features.tobytes()
                result = hashlib.blake2b(
                    quantum_bytes, 
                    digest_size=64,
                    key=data[:32].ljust(32, b'\x00')
                ).digest()
                
                logger.debug(f"Quantum encoding completed: {len(data)} -> {len(result)} bytes (state_size: {state_size})")
                return result
                
            except (np.linalg.LinAlgError, MemoryError) as e:
                raise EncodingError(f"Quantum state computation failed: {e}")
                
        except Exception as e:
            logger.error(f"Quantum encoding failed: {e}")
            raise EncodingError(f"Quantum encoding failed: {e}")

    def _encode_fractal(self, data: bytes) -> bytes:
        """Advanced fractal encoding with multi-scale self-similarity"""
        try:
            if not data:
                raise EncodingError("Input data for fractal encoding cannot be empty")
            
            max_depth = self.physics.get('max_fractal_depth', 6)
            recursion_depth = min(max_depth, self.physics.get('max_recursion_depth', 4))
            
            if recursion_depth <= 0:
                raise EncodingError("Invalid recursion depth for fractal encoding")
            
            # Initialize multi-scale fractal encoding
            fractal_scales = []
            current_data = np.frombuffer(data, dtype=np.uint8)
            max_size = self.physics.get('max_pattern_size', 1048576)
            
            # Generate fractal at multiple scales
            for scale_level in range(recursion_depth):
                try:
                    # Apply fractal transformation at current scale
                    scale_factor = 2 ** scale_level
                    if len(current_data) > max_size // scale_factor:
                        current_data = current_data[:max_size // scale_factor]
                    
                    # Mandelbrot-inspired transformation
                    complex_data = self._data_to_complex_plane(current_data)
                    fractal_result = self._mandelbrot_transform(complex_data, iterations=8)
                    
                    # Convert back to bytes with proper scaling
                    scaled_result = (fractal_result * 255).astype(np.uint8)
                    fractal_scales.append(scaled_result)
                    
                    # Prepare for next iteration
                    if len(scaled_result) > 0:
                        # Create feedback loop for next scale
                        hash_feedback = hashlib.sha256(scaled_result.tobytes()).digest()
                        current_data = np.frombuffer(hash_feedback, dtype=np.uint8)
                    
                except (MemoryError, OverflowError) as e:
                    logger.warning(f"Fractal scale {scale_level} failed: {e}, using fallback")
                    # Fallback to simple hash iteration
                    hash_result = hashlib.sha256(current_data.tobytes()).digest()
                    current_data = np.frombuffer(hash_result, dtype=np.uint8)
                    fractal_scales.append(current_data)
            
            # Combine multi-scale results
            if not fractal_scales:
                raise EncodingError("No fractal scales generated")
            
            # Weighted combination of scales
            combined_length = max(len(scale) for scale in fractal_scales)
            combined_fractal = np.zeros(combined_length, dtype=np.float32)
            
            for i, scale_data in enumerate(fractal_scales):
                weight = 1.0 / (2 ** i)  # Exponential weighting
                if len(scale_data) < combined_length:
                    # Tile to match length
                    repetitions = (combined_length // len(scale_data)) + 1
                    extended_scale = np.tile(scale_data, repetitions)[:combined_length]
                else:
                    extended_scale = scale_data[:combined_length]
                
                combined_fractal += extended_scale.astype(np.float32) * weight
            
            # Normalize and convert to bytes
            if combined_fractal.max() > combined_fractal.min():
                normalized_fractal = ((combined_fractal - combined_fractal.min()) / 
                                    (combined_fractal.max() - combined_fractal.min()) * 255)
            else:
                normalized_fractal = np.full_like(combined_fractal, 128)
            
            result_data = normalized_fractal.astype(np.uint8).tobytes()
            
            # Ensure minimum size requirements
            min_size = self.physics.get('min_pattern_size', 64)
            if len(result_data) < min_size:
                # Extend using self-similar tiling
                tiles_needed = (min_size // len(result_data)) + 1
                result_data = (result_data * tiles_needed)[:min_size]
            
            logger.debug(f"Fractal encoding completed: {len(data)} -> {len(result_data)} bytes ({recursion_depth} scales)")
            return result_data
            
        except Exception as e:
            logger.error(f"Fractal encoding failed: {e}")
            raise EncodingError(f"Fractal encoding failed: {e}")

    def _encode_wave(self, data: bytes) -> bytes:
        """Advanced wave encoding with multi-frequency analysis"""
        try:
            if not data:
                raise EncodingError("Input data for wave encoding cannot be empty")
            
            # Convert to floating point for better precision
            data_array = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
            
            if len(data_array) == 0:
                raise EncodingError("Cannot create wave array from empty data")
            
            # Adaptive array sizing for optimal FFT performance
            max_wave_size = self.physics.get('max_wave_samples', 8192)
            if len(data_array) > max_wave_size:
                # Decimate intelligently instead of truncation
                decimation_factor = len(data_array) // max_wave_size
                data_array = data_array[::decimation_factor][:max_wave_size]
                logger.debug(f"Wave data decimated by factor {decimation_factor}")
            
            # Pad to next power of 2 for efficient FFT
            next_pow2 = 2 ** int(np.ceil(np.log2(len(data_array))))
            if next_pow2 > len(data_array):
                # Use mirroring for natural continuation
                mirror_pad = next_pow2 - len(data_array)
                if mirror_pad <= len(data_array):
                    padding = data_array[-mirror_pad:][::-1]
                else:
                    # Repeat mirroring if needed
                    repetitions = (mirror_pad // len(data_array)) + 1
                    padding = np.tile(data_array[::-1], repetitions)[:mirror_pad]
                data_array = np.concatenate([data_array, padding])
            
            try:
                # Apply windowing to reduce spectral leakage
                window = np.hanning(len(data_array))
                windowed_data = data_array * window
                
                # Multi-resolution frequency analysis
                frequency_components = np.fft.fft(windowed_data)
                
                # Extract comprehensive spectral features
                amplitudes = np.abs(frequency_components)
                phases = np.angle(frequency_components)
                power_spectrum = amplitudes ** 2
                
                # Validate spectral data
                if np.any(np.isnan(amplitudes)) or np.any(np.isnan(phases)):
                    raise EncodingError("FFT produced NaN values")
                
                # Spectral centroid and spread for additional features
                freqs = np.fft.fftfreq(len(data_array))
                spectral_centroid = np.sum(freqs * power_spectrum) / (np.sum(power_spectrum) + 1e-10)
                spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power_spectrum) / 
                                        (np.sum(power_spectrum) + 1e-10))
                
                # Mel-scale frequency transformation for perceptual encoding
                mel_filters = self._create_mel_filterbank(len(amplitudes) // 2, 26)
                mel_spectrum = np.dot(mel_filters, amplitudes[:len(amplitudes)//2])
                
                # Combine all spectral features
                spectral_features = np.concatenate([
                    self._normalize_to_uint8(amplitudes),
                    self._normalize_to_uint8(phases + np.pi),  # Shift phases to positive
                    self._normalize_to_uint8(mel_spectrum),
                    [int(spectral_centroid * 255) % 256,
                     int(spectral_spread * 255) % 256]
                ])
                
                wave_pattern = spectral_features.astype(np.uint8).tobytes()
                
                # Ensure size requirements with intelligent truncation/padding
                min_size = self.physics.get('min_pattern_size', 64)
                max_size = self.physics.get('max_pattern_size', 1048576)
                
                # Ensure min_size and max_size are valid integers
                if min_size is None or not isinstance(min_size, int) or min_size <= 0:
                    min_size = 64
                if max_size is None or not isinstance(max_size, int) or max_size <= 0:
                    max_size = 1048576
                
                # Ensure min_size doesn't exceed max_size
                if min_size > max_size:
                    min_size = max_size
                
                if len(wave_pattern) < min_size:
                    # Pad with harmonics of the original pattern
                    harmonic_pattern = self._generate_harmonic_padding(wave_pattern, min_size - len(wave_pattern))
                    wave_pattern = wave_pattern + harmonic_pattern
                elif len(wave_pattern) > max_size:
                    # Intelligent compression preserving key spectral features
                    wave_pattern = self._compress_spectral_data(wave_pattern, max_size)
                
                logger.debug(f"Wave encoding completed: {len(data)} -> {len(wave_pattern)} bytes")
                return wave_pattern
                
            except np.fft._fftpack.error as e:
                raise EncodingError(f"FFT computation failed: {e}")
                
        except Exception as e:
            logger.error(f"Wave encoding failed: {e}")
            raise EncodingError(f"Wave encoding failed: {e}")

    def _calculate_local_entropy(self, data_array: np.ndarray, window_size: int = 8) -> np.ndarray:
        """Calculate local entropy for glyph pattern selection"""
        if len(data_array) == 0:
            return np.array([])
        
        entropy_values = []
        for i in range(0, len(data_array), window_size):
            window = data_array[i:i + window_size]
            if len(window) > 0:
                # Calculate Shannon entropy
                unique_vals, counts = np.unique(window, return_counts=True)
                probabilities = counts / len(window)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                entropy_values.append(entropy)
        
        return np.array(entropy_values)

    def _generate_fractal_padding(self, seed_pattern: bytes, length: int) -> bytearray:
        """Generate fractal-like padding pattern"""
        if not seed_pattern or length <= 0:
            return bytearray(length)
        
        padding = bytearray()
        pattern_array = np.frombuffer(seed_pattern, dtype=np.uint8)
        
        while len(padding) < length:
            # Apply simple fractal transformation
            next_iteration = ((pattern_array * 1.5) % 256).astype(np.uint8)
            padding.extend(next_iteration.tobytes())
            pattern_array = next_iteration
        
        return padding[:length]

    def _data_to_complex_plane(self, data: np.ndarray) -> np.ndarray:
        """Map byte data to complex plane for fractal processing"""
        if len(data) % 2 == 1:
            data = np.append(data, [0])
        
        real_parts = data[::2].astype(np.float32) / 127.5 - 1.0  # Scale to [-1, 1]
        imag_parts = data[1::2].astype(np.float32) / 127.5 - 1.0
        
        return real_parts + 1j * imag_parts

    def _mandelbrot_transform(self, complex_data: np.ndarray, iterations: int = 8) -> np.ndarray:
        """Apply Mandelbrot-inspired transformation"""
        c = complex_data
        z = np.zeros_like(c)
        result = np.zeros(len(c), dtype=np.float32)
        
        for i in range(iterations):
            mask = np.abs(z) <= 2
            z[mask] = z[mask] ** 2 + c[mask]
            result[mask] = i / iterations
        
        return result

    def _normalize_to_uint8(self, data: np.ndarray) -> np.ndarray:
        """Normalize array to uint8 range with proper handling"""
        if len(data) == 0:
            return np.array([], dtype=np.uint8)
        
        data_min, data_max = np.min(data), np.max(data)
        if data_max > data_min:
            normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            normalized = np.full(len(data), 128, dtype=np.uint8)
        
        return normalized

    def _create_mel_filterbank(self, n_fft: int, n_mels: int) -> np.ndarray:
        """Create mel-scale filterbank for perceptual frequency encoding"""
        # Simplified mel filterbank creation
        mel_filters = np.zeros((n_mels, n_fft))
        mel_points = np.linspace(0, n_fft - 1, n_mels + 2, dtype=int)
        
        for i in range(n_mels):
            left, center, right = mel_points[i:i+3]
            # Triangular filters
            mel_filters[i, left:center] = np.linspace(0, 1, center - left)
            if right > center:
                mel_filters[i, center:right] = np.linspace(1, 0, right - center)
        
        return mel_filters

    def _generate_harmonic_padding(self, pattern: bytes, length: int) -> bytes:
        """Generate harmonic padding based on spectral analysis"""
        if length <= 0 or not pattern:
            return b'\x00' * length
        
        # Create harmonic series based on pattern
        pattern_array = np.frombuffer(pattern, dtype=np.uint8).astype(np.float32)
        fundamental_freq = np.mean(pattern_array) / 255.0
        
        harmonic_pattern = []
        for i in range(length):
            # Generate harmonic series
            phase = (i * fundamental_freq * 2 * np.pi) % (2 * np.pi)
            harmonic_value = int((np.sin(phase) + 1) * 127.5) % 256
            harmonic_pattern.append(harmonic_value)
        
        return bytes(harmonic_pattern)

    def _compress_spectral_data(self, data: bytes, target_size: int) -> bytes:
        """Compress spectral data while preserving key features"""
        if len(data) <= target_size:
            return data
        
        data_array = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        compression_ratio = len(data_array) / target_size
        
        compressed = []
        for i in range(target_size):
            # Sample with anti-aliasing
            source_idx = i * compression_ratio
            left_idx = int(np.floor(source_idx))
            right_idx = min(left_idx + 1, len(data_array) - 1)
            
            if left_idx == right_idx:
                value = data_array[left_idx]
            else:
                # Linear interpolation
                alpha = source_idx - left_idx
                value = data_array[left_idx] * (1 - alpha) + data_array[right_idx] * alpha
            
            compressed.append(int(value) % 256)
        
        return bytes(compressed)