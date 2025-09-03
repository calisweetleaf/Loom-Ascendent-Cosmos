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
            
            max_pattern_size = self.physics.get('max_pattern_size', 1048576)
            if not isinstance(max_pattern_size, (int, float)) or max_pattern_size is None:
                max_pattern_size = 1048576  # 1MB default limit
            max_pattern_size = int(max_pattern_size)
            
            if len(data) > max_pattern_size:
                raise EncodingError(f"Input data size {len(data)} exceeds maximum allowed size {max_pattern_size}")
            
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
            
            # Ensure min_size and max_size are valid integers
            if not isinstance(min_size, (int, float)) or min_size is None:
                min_size = 64
            min_size = int(min_size)
            
            if not isinstance(max_size, (int, float)) or max_size is None:
                max_size = 1048576
            max_size = int(max_size)
            
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
            cube_size = max(int(np.ceil(len(arr) ** (1/3))), 3)  # Minimum cube size of 3 for 3x3x3 operations
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
                if len(arr) > 0:
                    padding_pattern = arr[:min(len(arr), 256)]  # Use first 256 bytes as pattern
                    pad_length = target_size - len(arr)
                    
                    if len(padding_pattern) > 0:
                        # Create repeating pattern for padding
                        pattern_repeats = (pad_length // len(padding_pattern)) + 1
                        padding = np.tile(padding_pattern, pattern_repeats)[:pad_length]
                        voxel_data = np.concatenate([arr, padding])
                    else:
                        # Fallback if pattern is empty
                        voxel_data = np.pad(arr, (0, pad_length), 'constant', constant_values=128)
                else:
                    # Fallback for empty array
                    voxel_data = np.full(target_size, 128, dtype=np.uint8)
            
            # Reshape to 3D cube and apply volumetric transformation
            try:
                voxel_cube = voxel_data.reshape(cube_size, cube_size, cube_size)
                
                # Apply 3D convolution-like operation for spatial encoding with proper kernel
                # Create a proper 3x3x3 Gaussian-like kernel
                kernel_1d = np.array([0.25, 0.5, 0.25])
                kernel_3d = np.outer(np.outer(kernel_1d, kernel_1d).flatten(), kernel_1d).reshape(3, 3, 3)
                kernel_3d = kernel_3d / np.sum(kernel_3d)  # Normalize kernel
                
                # Apply 3D filtering with boundary handling
                filtered_cube = np.zeros_like(voxel_cube, dtype=np.float32)
                
                # Process interior points with full kernel
                for i in range(1, cube_size - 1):
                    for j in range(1, cube_size - 1):
                        for k in range(1, cube_size - 1):
                            neighborhood = voxel_cube[i-1:i+2, j-1:j+2, k-1:k+2].astype(np.float32)
                            filtered_cube[i, j, k] = np.sum(neighborhood * kernel_3d)
                
                # Handle boundary conditions by copying edge values
                filtered_cube[0, :, :] = voxel_cube[0, :, :].astype(np.float32)
                filtered_cube[-1, :, :] = voxel_cube[-1, :, :].astype(np.float32)
                filtered_cube[:, 0, :] = voxel_cube[:, 0, :].astype(np.float32)
                filtered_cube[:, -1, :] = voxel_cube[:, -1, :].astype(np.float32)
                filtered_cube[:, :, 0] = voxel_cube[:, :, 0].astype(np.float32)
                filtered_cube[:, :, -1] = voxel_cube[:, :, -1].astype(np.float32)
                
                # Convert back to uint8 with proper clipping
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
            
            if len(data_array) == 0:
                raise EncodingError("Cannot process empty data array for glyph encoding")
            
            # Calculate entropy map with robust error handling
            try:
                entropy_map = self._calculate_local_entropy(data_array)
            except Exception as e:
                logger.warning(f"Entropy calculation failed, using fallback: {e}")
                entropy_map = np.array([np.std(data_array.astype(np.float32))])
            
            # Create glyph patterns based on entropy distribution
            glyph_patterns = {
                'low_entropy': b'\x01\x01\x02\x02',    # Repetitive pattern
                'med_entropy': b'\x03\x05\x07\x0B',    # Prime sequence
                'high_entropy': b'\x0F\x33\x55\xAA',  # Alternating bits
                'ultra_entropy': b'\xFF\x00\x5A\xA5'  # Maximum contrast
            }
            
            # Apply entropy-based glyph mapping with robust chunking
            glyph_data = bytearray()
            
            # Determine chunk size with better bounds checking
            if len(data_array) > target_size and target_size > 0:
                chunk_size = max(1, len(data_array) // target_size)
            else:
                chunk_size = max(1, len(data_array) // 8) if len(data_array) > 8 else 1
            
            entropy_idx = 0
            for i in range(0, len(data_array), chunk_size):
                chunk = data_array[i:i + chunk_size]
                
                if len(chunk) == 0:
                    continue
                    
                # Get entropy value with bounds checking
                if entropy_idx < len(entropy_map):
                    chunk_entropy = entropy_map[entropy_idx]
                    entropy_idx += 1
                else:
                    # Fallback to local entropy calculation
                    chunk_entropy = np.std(chunk.astype(np.float32))
                
                # Select pattern based on entropy thresholds
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
                    pattern_byte = pattern[j % len(pattern)]
                    glyph_data.append(byte_val ^ pattern_byte)
            
            # Ensure target size with sophisticated padding/compression
            if len(glyph_data) < target_size:
                # Create fractal padding pattern
                if len(glyph_data) > 0:
                    seed_pattern = bytes(glyph_data[-min(16, len(glyph_data)):])
                    padding_needed = target_size - len(glyph_data)
                    
                    # Generate fractal-like padding
                    try:
                        fractal_padding = self._generate_fractal_padding(seed_pattern, padding_needed)
                        glyph_data.extend(fractal_padding)
                    except Exception as e:
                        logger.warning(f"Fractal padding failed, using simple padding: {e}")
                        glyph_data.extend(b'\x80' * padding_needed)
                else:
                    # Fallback for empty glyph data
                    glyph_data.extend(b'\x80' * target_size)
            
            elif len(glyph_data) > target_size:
                # Compress using sliding window with overlap preservation
                if target_size > 0:
                    compression_ratio = len(glyph_data) / target_size
                    compressed_data = bytearray()
                    
                    for i in range(target_size):
                        source_idx = int(i * compression_ratio)
                        if source_idx < len(glyph_data):
                            compressed_data.append(glyph_data[source_idx])
                        else:
                            compressed_data.append(128)  # Default value
                    
                    glyph_data = compressed_data
                else:
                    glyph_data = bytearray([128])  # Single byte fallback
            
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
            unique_bytes = len(set(data)) if data else 1
            data_complexity = unique_bytes / 256.0  # Normalized complexity
            base_size = 128
            complexity_factor = max(0.5, min(2.0, data_complexity * 2))
            state_size = int(base_size * complexity_factor)
            
            # Ensure max_quantum_states is a valid integer
            max_quantum_states = self.physics.get('max_quantum_states', 2048)
            if not isinstance(max_quantum_states, (int, float)) or max_quantum_states is None:
                max_quantum_states = 2048
            max_quantum_states = int(max_quantum_states)
            
            state_size = max(16, min(state_size, max_quantum_states))
            
            try:
                # Generate complex quantum amplitudes with better conditioning
                real_parts = rng.normal(0, 0.5, state_size)  # Reduced variance for better conditioning
                imag_parts = rng.normal(0, 0.5, state_size)
                
                # Create complex amplitudes
                amplitudes = real_parts + 1j * imag_parts
                
                # Proper quantum state normalization with numerical stability
                norm_squared = np.sum(np.abs(amplitudes) ** 2)
                if norm_squared > 1e-10:  # Numerical stability threshold
                    amplitudes = amplitudes / np.sqrt(norm_squared)
                else:
                    # Fallback to uniform superposition
                    amplitudes = np.ones(state_size, dtype=complex) / np.sqrt(state_size)
                
                # Extract phases and magnitudes
                phases = np.angle(amplitudes)
                magnitudes = np.abs(amplitudes)
                
                # Quantum entanglement simulation through correlation matrix with regularization
                try:
                    correlation_matrix = np.outer(magnitudes, magnitudes)
                    # Add regularization for numerical stability
                    regularization = 1e-6 * np.eye(correlation_matrix.shape[0])
                    regularized_matrix = correlation_matrix + regularization
                    
                    eigenvals, eigenvecs = np.linalg.eigh(regularized_matrix)
                    
                    # Use top eigenvalues for encoding with bounds checking
                    top_k = min(32, len(eigenvals), state_size // 4)
                    if top_k > 0:
                        top_eigenvals = eigenvals[-top_k:]
                    else:
                        top_eigenvals = np.array([1.0])  # Fallback eigenvalue
                        
                except np.linalg.LinAlgError as e:
                    logger.warning(f"Eigenvalue decomposition failed, using fallback: {e}")
                    top_eigenvals = magnitudes[:min(32, len(magnitudes))]
                
                # Combine quantum features into byte representation
                try:
                    quantum_features = np.concatenate([
                        magnitudes,
                        phases + np.pi,  # Shift phases to positive range [0, 2π]
                        top_eigenvals
                    ])
                except ValueError as e:
                    # Fallback if concatenation fails
                    logger.warning(f"Feature concatenation failed, using magnitudes only: {e}")
                    quantum_features = magnitudes
                
                # Normalize to byte range with quantum discretization
                if len(quantum_features) > 0:
                    feature_min, feature_max = quantum_features.min(), quantum_features.max()
                    if feature_max > feature_min and not (np.isnan(feature_min) or np.isnan(feature_max)):
                        normalized_features = ((quantum_features - feature_min) / 
                                             (feature_max - feature_min) * 255).astype(np.uint8)
                    else:
                        normalized_features = np.full(len(quantum_features), 128, dtype=np.uint8)
                else:
                    normalized_features = np.array([128], dtype=np.uint8)
                
                # Apply quantum hash for final encoding
                quantum_bytes = normalized_features.tobytes()
                key_bytes = data[:32] if len(data) >= 32 else data.ljust(32, b'\x00')
                
                result = hashlib.blake2b(
                    quantum_bytes, 
                    digest_size=64,
                    key=key_bytes
                ).digest()
                
                logger.debug(f"Quantum encoding completed: {len(data)} -> {len(result)} bytes (state_size: {state_size})")
                return result
                
            except (np.linalg.LinAlgError, MemoryError) as e:
                logger.warning(f"Advanced quantum computation failed, using simplified approach: {e}")
                # Simplified fallback quantum encoding
                simple_hash = hashlib.sha3_256(data).digest()
                return simple_hash + hashlib.blake2b(data, digest_size=32).digest()
                
        except Exception as e:
            logger.error(f"Quantum encoding failed: {e}")
            raise EncodingError(f"Quantum encoding failed: {e}")

    def _encode_fractal(self, data: bytes) -> bytes:
        """Self-similar recursive encoding with Mandelbrot-like iterations"""
        try:
            if not data:
                raise EncodingError("Input data for fractal encoding cannot be empty")
            
            # Convert data to complex plane coordinates
            data_array = np.frombuffer(data, dtype=np.uint8)
            if len(data_array) == 0:
                raise EncodingError("Cannot process empty data array for fractal encoding")
            
            # Create complex coordinates from data bytes
            # Use pairs of bytes for real and imaginary parts
            if len(data_array) % 2 != 0:
                data_array = np.append(data_array, [data_array[-1]])  # Duplicate last byte if odd length
            
            real_parts = data_array[::2].astype(np.float32) / 255.0 * 4.0 - 2.0  # Map to [-2, 2]
            imag_parts = data_array[1::2].astype(np.float32) / 255.0 * 4.0 - 2.0
            
            complex_coords = real_parts + 1j * imag_parts
            
            # Mandelbrot-like iteration with data-dependent parameters
            max_iterations = min(256, max(16, len(data) // 4))
            escape_radius = 2.0 + (np.mean(data_array) / 255.0)  # Data-dependent escape radius
            
            # Initialize iteration arrays
            z = np.zeros_like(complex_coords, dtype=np.complex128)
            iteration_counts = np.zeros(len(complex_coords), dtype=np.int32)
            
            # Perform fractal iterations with vectorized operations
            for iteration in range(max_iterations):
                # Compute z^2 + c for all points simultaneously
                z_new = z * z + complex_coords
                
                # Find points that haven't escaped
                mask = np.abs(z_new) <= escape_radius
                z[mask] = z_new[mask]
                iteration_counts[mask] = iteration
            
            # Create fractal signature from iteration patterns
            fractal_features = np.concatenate([
                iteration_counts.astype(np.float32),
                np.abs(z).astype(np.float32),
                np.angle(z).astype(np.float32) + np.pi  # Shift to positive range
            ])
            
            # Normalize to byte range with enhanced dynamic range
            if len(fractal_features) > 0:
                feature_min, feature_max = np.percentile(fractal_features, [5, 95])  # Use percentiles for robustness
                if feature_max > feature_min:
                    normalized_features = ((fractal_features - feature_min) / 
                                         (feature_max - feature_min) * 255).astype(np.uint8)
                else:
                    normalized_features = np.full(len(fractal_features), 128, dtype=np.uint8)
            else:
                normalized_features = np.array([128], dtype=np.uint8)
            
            # Apply fractal hash for dimensional reduction and consistency
            fractal_bytes = normalized_features.tobytes()
            result = hashlib.blake2b(
                fractal_bytes,
                digest_size=64,
                key=data[:32] if len(data) >= 32 else data.ljust(32, b'\x00')
            ).digest()
            
            logger.debug(f"Fractal encoding completed: {len(data)} -> {len(result)} bytes (iterations: {max_iterations})")
            return result
            
        except Exception as e:
            logger.error(f"Fractal encoding failed: {e}")
            raise EncodingError(f"Fractal encoding failed: {e}")

    def _encode_wave(self, data: bytes) -> bytes:
        """Frequency/amplitude based encoding using FFT analysis"""
        try:
            if not data:
                raise EncodingError("Input data for wave encoding cannot be empty")
            
            data_array = np.frombuffer(data, dtype=np.uint8)
            if len(data_array) == 0:
                raise EncodingError("Cannot process empty data array for wave encoding")
            
            # Prepare data for FFT analysis with proper windowing
            signal_length = len(data_array)
            
            # Apply windowing function to reduce spectral leakage
            if signal_length > 1:
                window = np.hanning(signal_length)
                windowed_signal = data_array.astype(np.float64) * window
            else:
                windowed_signal = data_array.astype(np.float64)
            
            # Pad to next power of 2 for efficient FFT
            padded_length = 1 << (signal_length - 1).bit_length()
            if padded_length > signal_length:
                padded_signal = np.pad(windowed_signal, (0, padded_length - signal_length), 'constant')
            else:
                padded_signal = windowed_signal
            
            # Perform FFT analysis
            try:
                fft_result = np.fft.fft(padded_signal)
                frequencies = np.fft.fftfreq(len(padded_signal))
            except Exception as e:
                logger.warning(f"FFT computation failed, using DFT fallback: {e}")
                # Simple DFT fallback for small signals
                N = len(padded_signal)
                fft_result = np.array([
                    np.sum(padded_signal * np.exp(-2j * np.pi * k * np.arange(N) / N))
                    for k in range(min(N, 64))  # Limit computation for performance
                ])
                frequencies = np.arange(len(fft_result)) / len(fft_result)
            
            # Extract meaningful spectral features
            magnitudes = np.abs(fft_result)
            phases = np.angle(fft_result)
            
            # Compute spectral centroid and bandwidth
            if np.sum(magnitudes) > 0:
                spectral_centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
                spectral_bandwidth = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * magnitudes) / np.sum(magnitudes))
            else:
                spectral_centroid = 0.5
                spectral_bandwidth = 0.1
            
            # Compute spectral rolloff (frequency below which 85% of energy lies)
            cumulative_energy = np.cumsum(magnitudes ** 2)
            total_energy = cumulative_energy[-1] if len(cumulative_energy) > 0 else 1.0
            
            if total_energy > 0:
                rolloff_threshold = 0.85 * total_energy
                rolloff_index = np.argmax(cumulative_energy >= rolloff_threshold)
                spectral_rolloff = frequencies[rolloff_index] if rolloff_index < len(frequencies) else 0.85
            else:
                spectral_rolloff = 0.85
            
            # Compute zero crossing rate in time domain
            if len(data_array) > 1:
                zero_crossings = np.sum(np.abs(np.diff(np.sign(data_array.astype(np.float32) - np.mean(data_array))))) / (2.0 * len(data_array))
            else:
                zero_crossings = 0.0
            
            # Create comprehensive wave signature
            # Use top frequency components for encoding
            top_k_components = min(32, len(magnitudes))
            top_indices = np.argpartition(magnitudes, -top_k_components)[-top_k_components:]
            
            wave_features = np.concatenate([
                magnitudes[top_indices],
                phases[top_indices] + np.pi,  # Shift phases to positive range
                [spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossings]
            ])
            
            # Normalize features to byte range
            if len(wave_features) > 0:
                # Use robust normalization
                feature_median = np.median(wave_features)
                feature_mad = np.median(np.abs(wave_features - feature_median))  # Median Absolute Deviation
                
                if feature_mad > 0:
                    normalized_features = np.clip(
                        ((wave_features - feature_median) / (feature_mad * 6) + 1) * 127.5,
                        0, 255
                    ).astype(np.uint8)
                else:
                    normalized_features = np.full(len(wave_features), 128, dtype=np.uint8)
            else:
                normalized_features = np.array([128], dtype=np.uint8)
            
            # Generate final wave encoding with spectral hash
            wave_bytes = normalized_features.tobytes()
            result = hashlib.blake2b(
                wave_bytes,
                digest_size=64,
                key=data[:32] if len(data) >= 32 else data.ljust(32, b'\x00')
            ).digest()
            
            logger.debug(f"Wave encoding completed: {len(data)} -> {len(result)} bytes (spectral components: {top_k_components})")
            return result
            
        except Exception as e:
            logger.error(f"Wave encoding failed: {e}")
            raise EncodingError(f"Wave encoding failed: {e}")

    # --------------------------
    # Interaction Protocol Handlers
    # --------------------------
    
    def _handle_combine(self, pattern1: AetherPattern, pattern2: AetherPattern) -> AetherPattern:
        """Combine two patterns through weighted fusion"""
        try:
            # Calculate combination weights based on pattern properties
            total_complexity = pattern1.complexity + pattern2.complexity
            if total_complexity > 0:
                weight1 = pattern1.complexity / total_complexity
                weight2 = pattern2.complexity / total_complexity
            else:
                weight1 = weight2 = 0.5
            
            # Combine core data using weighted XOR
            min_length = min(len(pattern1.core), len(pattern2.core))
            max_length = max(len(pattern1.core), len(pattern2.core))
            
            # Align cores and combine
            core1_padded = pattern1.core.ljust(max_length, b'\x00')
            core2_padded = pattern2.core.ljust(max_length, b'\x00')
            
            combined_core = bytes(
                int(a * weight1 + b * weight2) & 0xFF 
                for a, b in zip(core1_padded, core2_padded)
            )
            
            # Merge mutations
            combined_mutations = tuple(set(pattern1.mutations) | set(pattern2.mutations))
            
            # Merge interactions
            combined_interactions = {**pattern1.interactions, **pattern2.interactions}
            
            # Select encoding type based on complexity
            encoding_type = pattern1.encoding_type if pattern1.complexity >= pattern2.complexity else pattern2.encoding_type
            
            # Create combined pattern
            result = AetherPattern(
                core=combined_core,
                mutations=combined_mutations,
                interactions=combined_interactions,
                encoding_type=encoding_type,
                recursion_level=max(pattern1.recursion_level, pattern2.recursion_level) + 1,
                metadata={
                    'combination_timestamp': time.time(),
                    'parent_patterns': [pattern1.pattern_id, pattern2.pattern_id],
                    'combination_weights': [weight1, weight2],
                    'operation': 'combine'
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern combination failed: {e}")
            raise InteractionError(f"Pattern combination failed: {e}")

    def _handle_entangle(self, pattern1: AetherPattern, pattern2: AetherPattern) -> AetherPattern:
        """Create quantum-like entanglement between patterns"""
        try:
            # Create entanglement signature using quantum-inspired correlations
            entanglement_key = hashlib.blake2b(
                pattern1.core + pattern2.core,
                digest_size=32,
                key=b'quantum_entanglement'
            ).digest()
            
            # Generate correlated transformations
            np.random.seed(int.from_bytes(entanglement_key[:8], 'big') % (2**32))
            
            # Create entangled core through correlated operations
            min_length = min(len(pattern1.core), len(pattern2.core))
            entangled_data = bytearray()
            
            for i in range(min_length):
                # Quantum-like correlation with controlled randomness
                correlation_factor = np.random.uniform(0.7, 1.0)  # Strong correlation
                byte1, byte2 = pattern1.core[i], pattern2.core[i]
                
                # Create entangled byte through correlation
                entangled_byte = int((byte1 + byte2 * correlation_factor) / (1 + correlation_factor)) & 0xFF
                entangled_data.append(entangled_byte)
            
            # Add entanglement-specific mutations
            entanglement_mutations = (entanglement_key[:16], entanglement_key[16:])
            combined_mutations = pattern1.mutations + pattern2.mutations + entanglement_mutations
            
            # Create entanglement interactions
            entangled_interactions = {
                **pattern1.interactions,
                **pattern2.interactions,
                'entanglement_partner_1': pattern1.pattern_id,
                'entanglement_partner_2': pattern2.pattern_id,
                'entanglement_strength': str(correlation_factor)
            }
            
            result = AetherPattern(
                core=bytes(entangled_data),
                mutations=combined_mutations,
                interactions=entangled_interactions,
                encoding_type=EncodingType.QUANTUM,  # Force quantum encoding for entangled patterns
                recursion_level=max(pattern1.recursion_level, pattern2.recursion_level) + 1,
                metadata={
                    'entanglement_timestamp': time.time(),
                    'entangled_patterns': [pattern1.pattern_id, pattern2.pattern_id],
                    'entanglement_key_hash': hashlib.sha256(entanglement_key).hexdigest(),
                    'correlation_factor': correlation_factor,
                    'operation': 'entangle'
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern entanglement failed: {e}")
            raise InteractionError(f"Pattern entanglement failed: {e}")

    def _handle_transform(self, pattern1: AetherPattern, pattern2: AetherPattern) -> AetherPattern:
        """Transform pattern1 using pattern2 as transformation matrix"""
        try:
            # Use pattern2 as transformation template
            transform_matrix = np.frombuffer(pattern2.core, dtype=np.uint8).astype(np.float32) / 255.0
            source_data = np.frombuffer(pattern1.core, dtype=np.uint8).astype(np.float32)
            
            # Apply transformation with proper size handling
            if len(transform_matrix) == 0:
                raise InteractionError("Transform pattern cannot be empty")
            
            transformed_data = []
            for i, value in enumerate(source_data):
                transform_idx = i % len(transform_matrix)
                # Apply non-linear transformation
                transformed_value = (value * transform_matrix[transform_idx] * 
                                   np.sin(transform_matrix[transform_idx] * np.pi)) % 256
                transformed_data.append(int(transformed_value) & 0xFF)
            
            # Inherit transformation properties
            result = AetherPattern(
                core=bytes(transformed_data),
                mutations=pattern1.mutations + (pattern2.core[:32],),  # Add transform signature
                interactions={**pattern1.interactions, 'transformed_by': pattern2.pattern_id},
                encoding_type=pattern1.encoding_type,
                recursion_level=pattern1.recursion_level + 1,
                metadata={
                    'transform_timestamp': time.time(),
                    'source_pattern': pattern1.pattern_id,
                    'transform_pattern': pattern2.pattern_id,
                    'operation': 'transform'
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern transformation failed: {e}")
            raise InteractionError(f"Pattern transformation failed: {e}")

    def _handle_cascade(self, pattern1: AetherPattern, pattern2: AetherPattern) -> AetherPattern:
        """Trigger cascading chain reaction between patterns"""
        try:
            # Calculate cascade probability based on pattern similarity
            compatibility = self.calculate_pattern_compatibility(pattern1, pattern2)
            cascade_threshold = 0.6
            
            if compatibility < cascade_threshold:
                raise InteractionError(f"Patterns incompatible for cascade (compatibility: {compatibility:.3f})")
            
            # Generate cascade iterations
            cascade_iterations = min(int(compatibility * 10), 5)  # Max 5 iterations
            current_core = pattern1.core
            
            for iteration in range(cascade_iterations):
                # Apply cascade transformation using pattern2 as catalyst
                catalyst_data = np.frombuffer(pattern2.core, dtype=np.uint8)
                current_data = np.frombuffer(current_core, dtype=np.uint8)
                
                # Cascade reaction simulation
                min_length = min(len(catalyst_data), len(current_data))
                cascaded_data = bytearray()
                
                for i in range(len(current_data)):
                    catalyst_byte = catalyst_data[i % len(catalyst_data)] if len(catalyst_data) > 0 else 128
                    current_byte = current_data[i]
                    
                    # Cascade reaction with amplification
                    reaction_product = ((current_byte ^ catalyst_byte) + 
                                      int(iteration * compatibility * 50)) & 0xFF
                    cascaded_data.append(reaction_product)
                
                current_core = bytes(cascaded_data)
            
            # Create cascade result with accumulated changes
            cascade_mutations = pattern1.mutations + tuple(
                current_core[i:i+32] for i in range(0, min(len(current_core), 96), 32)
            )
            
            result = AetherPattern(
                core=current_core,
                mutations=cascade_mutations,
                interactions={
                    **pattern1.interactions,
                    'cascade_catalyst': pattern2.pattern_id,
                    'cascade_iterations': str(cascade_iterations)
                },
                encoding_type=pattern1.encoding_type,
                recursion_level=pattern1.recursion_level + cascade_iterations,
                metadata={
                    'cascade_timestamp': time.time(),
                    'initiator_pattern': pattern1.pattern_id,
                    'catalyst_pattern': pattern2.pattern_id,
                    'cascade_iterations': cascade_iterations,
                    'compatibility_score': compatibility,
                    'operation': 'cascade'
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern cascade failed: {e}")
            raise InteractionError(f"Pattern cascade failed: {e}")

    def _handle_resonate(self, pattern1: AetherPattern, pattern2: AetherPattern) -> AetherPattern:
        """Create harmonic resonance between patterns"""
        try:
            # Analyze frequency components of both patterns
            data1 = np.frombuffer(pattern1.core, dtype=np.uint8).astype(np.float64)
            data2 = np.frombuffer(pattern2.core, dtype=np.uint8).astype(np.float64)
            
            if len(data1) == 0 or len(data2) == 0:
                raise InteractionError("Cannot resonate with empty patterns")
            
            # Find harmonic frequencies
            fft1 = np.fft.fft(data1)
            fft2 = np.fft.fft(data2)
            
            # Align FFT lengths
            min_fft_length = min(len(fft1), len(fft2))
            fft1_aligned = fft1[:min_fft_length]
            fft2_aligned = fft2[:min_fft_length]
            
            # Create resonance through constructive interference
            resonance_fft = fft1_aligned + fft2_aligned * np.exp(1j * np.pi / 4)  # 45° phase shift
            
            # Convert back to spatial domain
            try:
                resonance_signal = np.fft.ifft(resonance_fft).real
            except Exception as e:
                logger.warning(f"IFFT failed, using direct combination: {e}")
                resonance_signal = (data1[:min_fft_length] + data2[:min_fft_length]) / 2
            
            # Normalize and convert to bytes
            if len(resonance_signal) > 0:
                resonance_signal = np.clip(resonance_signal, 0, 255).astype(np.uint8)
                resonance_core = resonance_signal.tobytes()
            else:
                resonance_core = pattern1.core  # Fallback
            
            # Calculate resonance frequency
            dominant_freq_idx = np.argmax(np.abs(resonance_fft))
            resonance_frequency = dominant_freq_idx / len(resonance_fft)
            
            result = AetherPattern(
                core=resonance_core,
                mutations=pattern1.mutations + pattern2.mutations,
                interactions={
                    **pattern1.interactions,
                    **pattern2.interactions,
                    'resonance_partner_1': pattern1.pattern_id,
                    'resonance_partner_2': pattern2.pattern_id,
                    'resonance_frequency': str(resonance_frequency)
                },
                encoding_type=EncodingType.WAVE,  # Wave encoding for resonance
                recursion_level=max(pattern1.recursion_level, pattern2.recursion_level) + 1,
                metadata={
                    'resonance_timestamp': time.time(),
                    'resonating_patterns': [pattern1.pattern_id, pattern2.pattern_id],
                    'resonance_frequency': resonance_frequency,
                    'phase_shift': np.pi / 4,
                    'operation': 'resonate'
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern resonance failed: {e}")
            raise InteractionError(f"Pattern resonance failed: {e}")

    def _handle_annihilate(self, pattern1: AetherPattern, pattern2: AetherPattern) -> AetherPattern:
        """Mutual destruction/cancellation of patterns"""
        try:
            # Calculate annihilation energy based on pattern differences
            data1 = np.frombuffer(pattern1.core, dtype=np.uint8).astype(np.int16)
            data2 = np.frombuffer(pattern2.core, dtype=np.uint8).astype(np.int16)
            
            # Align data lengths
            max_length = max(len(data1), len(data2))
            data1_padded = np.pad(data1, (0, max_length - len(data1)), 'wrap')
            data2_padded = np.pad(data2, (0, max_length - len(data2)), 'wrap')
            
            # Annihilation through destructive interference
            annihilation_result = data1_padded - data2_padded
            
            # Calculate energy release
            energy_released = np.sum(np.abs(annihilation_result))
            total_energy = np.sum(np.abs(data1_padded)) + np.sum(np.abs(data2_padded))
            
            if total_energy > 0:
                annihilation_efficiency = energy_released / total_energy
            else:
                annihilation_efficiency = 0.0
            
            # Create residual pattern from annihilation
            residual_data = np.abs(annihilation_result) % 256
            residual_core = residual_data.astype(np.uint8).tobytes()
            
            # Minimal residual pattern
            result = AetherPattern(
                core=residual_core if len(residual_core) > 0 else b'\x00' * 64,
                mutations=(hashlib.blake2b(pattern1.core + pattern2.core, digest_size=16).digest(),),
                interactions={
                    'annihilated_pattern_1': pattern1.pattern_id,
                    'annihilated_pattern_2': pattern2.pattern_id,
                    'annihilation_efficiency': str(annihilation_efficiency)
                },
                encoding_type=EncodingType.BINARY,  # Simple encoding for residual
                recursion_level=0,  # Reset recursion for residual
                metadata={
                    'annihilation_timestamp': time.time(),
                    'annihilated_patterns': [pattern1.pattern_id, pattern2.pattern_id],
                    'energy_released': float(energy_released),
                    'annihilation_efficiency': annihilation_efficiency,
                    'operation': 'annihilate'
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern annihilation failed: {e}")
            raise InteractionError(f"Pattern annihilation failed: {e}")

    def _handle_catalyze(self, pattern1: AetherPattern, pattern2: AetherPattern) -> AetherPattern:
        """Use pattern2 to catalyze transformation of pattern1"""
        try:
            # Pattern2 acts as catalyst - it facilitates change but remains unchanged conceptually
            catalyst_signature = hashlib.blake2b(pattern2.core, digest_size=16).digest()
            
            # Apply catalytic transformation to pattern1
            source_data = np.frombuffer(pattern1.core, dtype=np.uint8)
            catalyst_data = np.frombuffer(pattern2.core, dtype=np.uint8)
            
            if len(catalyst_data) == 0:
                raise InteractionError("Catalyst pattern cannot be empty")
            
            # Catalytic reaction simulation
            catalyzed_data = bytearray()
            catalyst_strength = np.mean(catalyst_data) / 255.0  # Normalized catalyst strength
            
            for i, byte_val in enumerate(source_data):
                catalyst_byte = catalyst_data[i % len(catalyst_data)]
                
                # Catalytic enhancement with controlled reaction rate
                reaction_rate = catalyst_strength * (1 + np.sin(i * np.pi / len(catalyst_data)))
                catalyzed_byte = int((byte_val + catalyst_byte * reaction_rate) % 256)
                catalyzed_data.append(catalyzed_byte)
            
            # Add catalyst signature to mutations
            catalyzed_mutations = pattern1.mutations + (catalyst_signature,)
            
            result = AetherPattern(
                core=bytes(catalyzed_data),
                mutations=catalyzed_mutations,
                interactions={
                    **pattern1.interactions,
                    'catalyst_pattern': pattern2.pattern_id,
                    'catalyst_strength': str(catalyst_strength)
                },
                encoding_type=pattern1.encoding_type,
                recursion_level=pattern1.recursion_level + 1,
                metadata={
                    'catalysis_timestamp': time.time(),
                    'substrate_pattern': pattern1.pattern_id,
                    'catalyst_pattern': pattern2.pattern_id,
                    'catalyst_strength': catalyst_strength,
                    'operation': 'catalyze'
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern catalysis failed: {e}")
            raise InteractionError(f"Pattern catalysis failed: {e}")