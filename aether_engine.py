# ================================================================
#  AETHER ENGINE - Advanced Pattern Processing System
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
# ================================================================
import numpy as np
import threading
import hashlib
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Set, TypeVar
from enum import Enum
import logging
import asyncio
from collections import defaultdict, deque
import time
import concurrent.futures

logger = logging.getLogger("AetherEngine")

class EncodingParadigm(Enum):
    BINARY = "binary"
    SYMBOLIC = "symbolic"
    VOXEL = "voxel"
    GLYPH = "glyph"

class PatternComplexity(Enum):
    ELEMENTARY = 1
    COMPOUND = 2
    RECURSIVE = 3
    EMERGENT = 4

@dataclass
class AetherPattern:
    """Advanced pattern structure with mutation and interaction capabilities"""
    pattern_id: str
    core_pattern: np.ndarray
    mutation_vectors: List[np.ndarray] = field(default_factory=list)
    interaction_protocols: Dict[str, Callable] = field(default_factory=dict)
    recursive_hooks: List[Callable] = field(default_factory=list)
    
    encoding_paradigm: EncodingParadigm = EncodingParadigm.SYMBOLIC
    complexity_level: PatternComplexity = PatternComplexity.ELEMENTARY
    
    # Advanced pattern properties
    stability_metric: float = 1.0
    resonance_frequency: float = 1.0
    coherence_threshold: float = 0.8
    
    # Performance tracking
    activation_count: int = 0
    last_mutation_time: float = 0.0
    interaction_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def __post_init__(self):
        if not self.pattern_id:
            # Generate unique pattern ID based on core pattern
            pattern_hash = hashlib.sha256(self.core_pattern.tobytes()).hexdigest()[:16]
            self.pattern_id = f"{self.encoding_paradigm.value}_{pattern_hash}"
    
    def mutate(self, mutation_rate: float = 0.01, preserve_stability: bool = True) -> 'AetherPattern':
        """Create mutated version of pattern with stability preservation"""
        if not self.mutation_vectors:
            return self
        
        # Select mutation vector based on pattern history
        mutation_vector = self._select_optimal_mutation_vector()
        
        # Apply mutation with rate control
        mutated_core = self.core_pattern.copy()
        mutation_mask = np.random.random(mutated_core.shape) < mutation_rate
        
        # Apply selected mutation vector
        mutated_core[mutation_mask] += mutation_vector[mutation_mask] * np.random.normal(0, 0.1, mutated_core.shape)[mutation_mask]
        
        # Preserve stability if requested
        if preserve_stability:
            mutated_core = self._apply_stability_constraints(mutated_core)
        
        # Create mutated pattern
        mutated_pattern = AetherPattern(
            pattern_id="",  # Will be auto-generated
            core_pattern=mutated_core,
            mutation_vectors=self.mutation_vectors.copy(),
            interaction_protocols=self.interaction_protocols.copy(),
            recursive_hooks=self.recursive_hooks.copy(),
            encoding_paradigm=self.encoding_paradigm,
            complexity_level=self.complexity_level,
            stability_metric=self.stability_metric * 0.95,  # Slight stability decay
            resonance_frequency=self.resonance_frequency * (1.0 + np.random.normal(0, 0.05)),
            coherence_threshold=self.coherence_threshold
        )
        
        mutated_pattern.last_mutation_time = time.time()
        return mutated_pattern
    
    def _select_optimal_mutation_vector(self) -> np.ndarray:
        """Select optimal mutation vector based on pattern history and performance"""
        if len(self.mutation_vectors) == 1:
            return self.mutation_vectors[0]
        
        # Score mutation vectors based on historical performance
        vector_scores = []
        for i, vector in enumerate(self.mutation_vectors):
            # Calculate score based on stability maintenance and successful interactions
            stability_score = self.stability_metric
            interaction_score = len([h for h in self.interaction_history if h.get("success", False)]) / max(1, len(self.interaction_history))
            
            # Combine scores with random exploration factor
            total_score = stability_score * 0.6 + interaction_score * 0.3 + np.random.random() * 0.1
            vector_scores.append(total_score)
        
        # Select highest scoring vector
        best_vector_idx = np.argmax(vector_scores)
        return self.mutation_vectors[best_vector_idx]
    
    def _apply_stability_constraints(self, pattern_data: np.ndarray) -> np.ndarray:
        """Apply constraints to maintain pattern stability"""
        # Normalize to prevent runaway growth
        if np.max(np.abs(pattern_data)) > 10.0:
            pattern_data = pattern_data / np.max(np.abs(pattern_data)) * 10.0
        
        # Apply coherence filtering
        if self.coherence_threshold > 0:
            # Remove high-frequency noise that reduces coherence
            from scipy.signal import medfilt
            if pattern_data.ndim == 1:
                pattern_data = medfilt(pattern_data, kernel_size=3)
            elif pattern_data.ndim == 2:
                from scipy.ndimage import median_filter
                pattern_data = median_filter(pattern_data, size=3)
        
        return pattern_data
    
    def interact_with(self, other_pattern: 'AetherPattern', interaction_type: str = "default") -> Optional['AetherPattern']:
        """Interact with another pattern to produce emergent pattern"""
        self.activation_count += 1
        other_pattern.activation_count += 1
        
        # Check for compatible interaction protocols
        if interaction_type not in self.interaction_protocols:
            logger.warning(f"No interaction protocol '{interaction_type}' found for pattern {self.pattern_id}")
            return None
        
        try:
            # Execute interaction protocol
            interaction_func = self.interaction_protocols[interaction_type]
            result_pattern = interaction_func(self, other_pattern)
            
            # Record interaction
            interaction_record = {
                "timestamp": time.time(),
                "partner_id": other_pattern.pattern_id,
                "interaction_type": interaction_type,
                "success": result_pattern is not None
            }
            
            self.interaction_history.append(interaction_record)
            other_pattern.interaction_history.append(interaction_record)
            
            return result_pattern
            
        except Exception as e:
            logger.error(f"Error in pattern interaction: {e}", exc_info=True)
            return None
    
    def calculate_resonance(self, other_pattern: 'AetherPattern') -> float:
        """Calculate resonance coefficient with another pattern"""
        # Frequency matching component
        freq_diff = abs(self.resonance_frequency - other_pattern.resonance_frequency)
        freq_resonance = np.exp(-freq_diff / 0.1)  # Exponential decay
        
        # Pattern similarity component
        min_size = min(self.core_pattern.size, other_pattern.core_pattern.size)
        self_flat = self.core_pattern.flatten()[:min_size]
        other_flat = other_pattern.core_pattern.flatten()[:min_size]
        
        pattern_correlation = np.corrcoef(self_flat, other_flat)[0, 1]
        if np.isnan(pattern_correlation):
            pattern_correlation = 0.0
        
        # Combine components
        total_resonance = (freq_resonance * 0.4 + 
                          abs(pattern_correlation) * 0.6)
        
        return total_resonance

class AetherSpace:
    """Multi-dimensional space containing Aether patterns with advanced indexing"""
    
    def __init__(self, dimensions: Tuple[int, ...] = (64, 64, 64), max_patterns: int = 10000):
        self.dimensions = dimensions
        self.max_patterns = max_patterns
        
        # Pattern storage and indexing
        self.patterns: Dict[str, AetherPattern] = {}
        self.spatial_index: Dict[Tuple[int, ...], Set[str]] = defaultdict(set)
        self.frequency_index: Dict[float, Set[str]] = defaultdict(set)
        self.complexity_index: Dict[PatternComplexity, Set[str]] = defaultdict(set)
        
        # Advanced caching system
        self.interaction_cache: Dict[Tuple[str, str], AetherPattern] = {}
        self.resonance_cache: Dict[Tuple[str, str], float] = {}
        self.mutation_cache: Dict[str, List[AetherPattern]] = {}
        
        # Performance optimization
        self.update_lock = threading.RLock()
        self.background_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Pattern evolution system
        self.evolution_enabled = True
        self.evolution_rate = 0.01
        self.selection_pressure = 0.1
        
    def add_pattern(self, pattern: AetherPattern, position: Optional[Tuple[int, ...]] = None):
        """Add pattern to AetherSpace with automatic indexing"""
        with self.update_lock:
            # Auto-assign position if not provided
            if position is None:
                position = self._find_optimal_position(pattern)
            
            # Validate position
            if not all(0 <= p < d for p, d in zip(position, self.dimensions)):
                logger.warning(f"Invalid position {position} for pattern {pattern.pattern_id}")
                return
            
            # Add to main storage
            self.patterns[pattern.pattern_id] = pattern
            
            # Update indices
            self.spatial_index[position].add(pattern.pattern_id)
            
            freq_key = round(pattern.resonance_frequency, 2)
            self.frequency_index[freq_key].add(pattern.pattern_id)
            
            self.complexity_index[pattern.complexity_level].add(pattern.pattern_id)
            
            # Manage pattern limit
            if len(self.patterns) > self.max_patterns:
                self._evict_least_active_patterns()
            
            logger.debug(f"Added pattern {pattern.pattern_id} at position {position}")
    
    def _find_optimal_position(self, pattern: AetherPattern) -> Tuple[int, ...]:
        """Find optimal position for pattern based on resonance with neighbors"""
        best_position = tuple(np.random.randint(0, d) for d in self.dimensions)
        best_resonance = -1.0
        
        # Sample several positions and choose best
        for _ in range(min(20, np.prod(self.dimensions))):
            candidate_pos = tuple(np.random.randint(0, d) for d in self.dimensions)
            
            # Calculate total resonance with nearby patterns
            nearby_patterns = self.get_patterns_in_radius(candidate_pos, radius=5)
            total_resonance = 0.0
            
            for nearby_pattern in nearby_patterns:
                resonance = pattern.calculate_resonance(nearby_pattern)
                total_resonance += resonance
            
            if total_resonance > best_resonance:
                best_resonance = total_resonance
                best_position = candidate_pos
        
        return best_position
    
    def get_patterns_in_radius(self, center: Tuple[int, ...], radius: int = 3) -> List[AetherPattern]:
        """Get all patterns within radius of center position"""
        patterns_in_radius = []
        
        # Generate all positions within radius
        ranges = [range(max(0, c - radius), min(d, c + radius + 1)) 
                 for c, d in zip(center, self.dimensions)]
        
        import itertools
        for position in itertools.product(*ranges):
            pattern_ids = self.spatial_index.get(position, set())
            for pattern_id in pattern_ids:
                if pattern_id in self.patterns:
                    patterns_in_radius.append(self.patterns[pattern_id])
        
        return patterns_in_radius
    
    def find_resonant_patterns(self, target_pattern: AetherPattern, 
                              min_resonance: float = 0.5, limit: int = 10) -> List[Tuple[AetherPattern, float]]:
        """Find patterns with high resonance to target pattern"""
        resonant_patterns = []
        
        # Check frequency index first for efficiency
        target_freq = round(target_pattern.resonance_frequency, 2)
        candidate_freqs = [target_freq - 0.2, target_freq - 0.1, target_freq, 
                          target_freq + 0.1, target_freq + 0.2]
        
        candidate_pattern_ids = set()
        for freq in candidate_freqs:
            if freq in self.frequency_index:
                candidate_pattern_ids.update(self.frequency_index[freq])
        
        # Calculate resonance for candidates
        for pattern_id in candidate_pattern_ids:
            if pattern_id == target_pattern.pattern_id:
                continue
            
            pattern = self.patterns[pattern_id]
            
            # Check cache first
            cache_key = (target_pattern.pattern_id, pattern_id)
            if cache_key in self.resonance_cache:
                resonance = self.resonance_cache[cache_key]
            else:
                resonance = target_pattern.calculate_resonance(pattern)
                self.resonance_cache[cache_key] = resonance
                
                # Limit cache size
                if len(self.resonance_cache) > 1000:
                    oldest_key = next(iter(self.resonance_cache))
                    del self.resonance_cache[oldest_key]
            
            if resonance >= min_resonance:
                resonant_patterns.append((pattern, resonance))
        
        # Sort by resonance and return top results
        resonant_patterns.sort(key=lambda x: x[1], reverse=True)
        return resonant_patterns[:limit]
    
    def evolve_patterns(self, time_step: float = 0.01):
        """Evolve all patterns in the space using selection pressure"""
        if not self.evolution_enabled:
            return
        
        with self.update_lock:
            patterns_to_evolve = list(self.patterns.values())
            
            # Submit evolution tasks to thread pool
            evolution_futures = []
            for pattern in patterns_to_evolve:
                if np.random.random() < self.evolution_rate:
                    future = self.background_executor.submit(self._evolve_single_pattern, pattern)
                    evolution_futures.append(future)
            
            # Collect results
            new_patterns = []
            for future in concurrent.futures.as_completed(evolution_futures, timeout=1.0):
                try:
                    evolved_pattern = future.result()
                    if evolved_pattern:
                        new_patterns.append(evolved_pattern)
                except Exception as e:
                    logger.error(f"Error in pattern evolution: {e}")
            
            # Add evolved patterns to space
            for pattern in new_patterns:
                self.add_pattern(pattern)
            
            # Apply selection pressure
            self._apply_selection_pressure()
    
    def _evolve_single_pattern(self, pattern: AetherPattern) -> Optional[AetherPattern]:
        """Evolve a single pattern through mutation and interaction"""
        try:
            # Mutation-based evolution
            if np.random.random() < 0.7:
                return pattern.mutate(mutation_rate=0.05)
            
            # Interaction-based evolution
            else:
                resonant_patterns = self.find_resonant_patterns(pattern, min_resonance=0.3, limit=3)
                if resonant_patterns:
                    partner_pattern, _ = resonant_patterns[0]
                    return pattern.interact_with(partner_pattern, "fusion")
            
            return None
            
        except Exception as e:
            logger.error(f"Error evolving pattern {pattern.pattern_id}: {e}")
            return None
    
    def _apply_selection_pressure(self):
        """Apply selection pressure to remove least fit patterns"""
        if len(self.patterns) <= self.max_patterns:
            return
        
        # Calculate fitness scores for all patterns
        pattern_fitness = []
        for pattern in self.patterns.values():
            # Fitness based on activation count, stability, and recent interactions
            fitness = (pattern.activation_count * 0.4 + 
                      pattern.stability_metric * 0.3 +
                      len(pattern.interaction_history) * 0.3)
            pattern_fitness.append((pattern.pattern_id, fitness))
        
        # Sort by fitness and remove lowest performers
        pattern_fitness.sort(key=lambda x: x[1])
        patterns_to_remove = int(len(self.patterns) * self.selection_pressure)
        
        for pattern_id, _ in pattern_fitness[:patterns_to_remove]:
            self.remove_pattern(pattern_id)
    
    def _evict_least_active_patterns(self):
        """Evict patterns with lowest activation count"""
        if len(self.patterns) <= self.max_patterns:
            return
        
        # Sort patterns by activation count
        sorted_patterns = sorted(
            self.patterns.values(), 
            key=lambda p: p.activation_count
        )
        
        # Remove least active patterns
        patterns_to_remove = len(self.patterns) - self.max_patterns
        for pattern in sorted_patterns[:patterns_to_remove]:
            self.remove_pattern(pattern.pattern_id)
    
    def remove_pattern(self, pattern_id: str):
        """Remove pattern from space and all indices"""
        if pattern_id not in self.patterns:
            return
        
        pattern = self.patterns[pattern_id]
        
        # Remove from spatial index
        for position, pattern_set in self.spatial_index.items():
            pattern_set.discard(pattern_id)
        
        # Remove from frequency index
        freq_key = round(pattern.resonance_frequency, 2)
        if freq_key in self.frequency_index:
            self.frequency_index[freq_key].discard(pattern_id)
        
        # Remove from complexity index
        self.complexity_index[pattern.complexity_level].discard(pattern_id)
        
        # Remove from main storage
        del self.patterns[pattern_id]
        
        # Clean up caches
        cache_keys_to_remove = [
            key for key in self.resonance_cache.keys() 
            if pattern_id in key
        ]
        for key in cache_keys_to_remove:
            del self.resonance_cache[key]
    
    def get_space_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the AetherSpace"""
        with self.update_lock:
            complexity_distribution = {
                complexity.name: len(patterns) 
                for complexity, patterns in self.complexity_index.items()
            }
            
            avg_activation = np.mean([p.activation_count for p in self.patterns.values()]) if self.patterns else 0
            avg_stability = np.mean([p.stability_metric for p in self.patterns.values()]) if self.patterns else 0
            
            return {
                "total_patterns": len(self.patterns),
                "spatial_positions_occupied": len([pos for pos, patterns in self.spatial_index.items() if patterns]),
                "complexity_distribution": complexity_distribution,
                "average_activation_count": avg_activation,
                "average_stability": avg_stability,
                "cache_sizes": {
                    "interaction_cache": len(self.interaction_cache),
                    "resonance_cache": len(self.resonance_cache),
                    "mutation_cache": len(self.mutation_cache)
                },
                "evolution_enabled": self.evolution_enabled,
                "evolution_rate": self.evolution_rate
            }

class AetherEngine:
    """Main engine managing AetherSpace and pattern operations"""
    
    def __init__(self, space_dimensions: Tuple[int, ...] = (64, 64, 64)):
        self.aether_space = AetherSpace(dimensions=space_dimensions)
        self.pattern_factories: Dict[EncodingParadigm, Callable] = {}
        self.processing_queue = deque()
        self.is_running = False
        
        # Performance monitoring
        self.performance_metrics = {
            "patterns_processed": 0,
            "interactions_completed": 0,
            "mutations_performed": 0,
            "evolution_cycles": 0
        }
        
        self._setup_default_factories()
    
    def _setup_default_factories(self):
        """Setup default pattern factories for different encoding paradigms"""
        self.pattern_factories[EncodingParadigm.BINARY] = self._create_binary_pattern
        self.pattern_factories[EncodingParadigm.SYMBOLIC] = self._create_symbolic_pattern
        self.pattern_factories[EncodingParadigm.VOXEL] = self._create_voxel_pattern
        self.pattern_factories[EncodingParadigm.GLYPH] = self._create_glyph_pattern
    
    def _create_binary_pattern(self, data: Any, **kwargs) -> AetherPattern:
        """Create binary encoding pattern"""
        if isinstance(data, str):
            binary_data = np.array([ord(c) for c in data], dtype=np.uint8)
        elif isinstance(data, (list, np.ndarray)):
            binary_data = np.array(data, dtype=np.uint8)
        else:
            binary_data = np.array([hash(str(data)) % 256], dtype=np.uint8)
        
        return AetherPattern(
            pattern_id="",
            core_pattern=binary_data,
            encoding_paradigm=EncodingParadigm.BINARY,
            **kwargs
        )
    
    def _create_symbolic_pattern(self, data: Any, **kwargs) -> AetherPattern:
        """Create symbolic encoding pattern"""
        if isinstance(data, str):
            # Convert string to symbolic representation
            symbolic_data = np.array([hash(word) % 1000 for word in data.split()], dtype=np.float32)
        else:
            symbolic_data = np.array([hash(str(data)) % 1000], dtype=np.float32)
        
        return AetherPattern(
            pattern_id="",
            core_pattern=symbolic_data,
            encoding_paradigm=EncodingParadigm.SYMBOLIC,
            complexity_level=PatternComplexity.COMPOUND,
            **kwargs
        )
    
    def _create_voxel_pattern(self, data: Any, dimensions: Tuple[int, ...] = (8, 8, 8), **kwargs) -> AetherPattern:
        """Create voxel encoding pattern"""
        voxel_data = np.random.random(dimensions).astype(np.float32)
        
        # Encode data into voxel structure
        if isinstance(data, str):
            data_hash = hash(data)
            for i in range(min(len(data), np.prod(dimensions))):
                idx = np.unravel_index(i, dimensions)
                voxel_data[idx] = (ord(data[i]) + data_hash) % 256 / 255.0
        
        return AetherPattern(
            pattern_id="",
            core_pattern=voxel_data,
            encoding_paradigm=EncodingParadigm.VOXEL,
            complexity_level=PatternComplexity.RECURSIVE,
            **kwargs
        )
    
    def _create_glyph_pattern(self, data: Any, **kwargs) -> AetherPattern:
        """Create glyph encoding pattern"""
        # Create 2D glyph representation
        glyph_size = kwargs.get("glyph_size", 16)
        glyph_data = np.zeros((glyph_size, glyph_size), dtype=np.float32)
        
        if isinstance(data, str):
            # Create visual representation of string
            for i, char in enumerate(data[:glyph_size*glyph_size]):
                row, col = divmod(i, glyph_size)
                glyph_data[row, col] = ord(char) / 255.0
        
        return AetherPattern(
            pattern_id="",
            core_pattern=glyph_data,
            encoding_paradigm=EncodingParadigm.GLYPH,
            complexity_level=PatternComplexity.EMERGENT,
            **kwargs
        )
    
    def create_pattern(self, data: Any, encoding: EncodingParadigm = EncodingParadigm.SYMBOLIC, **kwargs) -> AetherPattern:
        """Create new Aether pattern with specified encoding"""
        if encoding not in self.pattern_factories:
            logger.error(f"No factory registered for encoding {encoding}")
            return None
        
        try:
            factory = self.pattern_factories[encoding]
            pattern = factory(data, **kwargs)
            self.aether_space.add_pattern(pattern)
            return pattern
        except Exception as e:
            logger.error(f"Error creating pattern: {e}", exc_info=True)
            return None
    
    def start_evolution(self, evolution_interval: float = 1.0):
        """Start background pattern evolution"""
        self.is_running = True
        
        async def evolution_loop():
            while self.is_running:
                try:
                    self.aether_space.evolve_patterns()
                    self.performance_metrics["evolution_cycles"] += 1
                    await asyncio.sleep(evolution_interval)
                except Exception as e:
                    logger.error(f"Error in evolution loop: {e}", exc_info=True)
                    await asyncio.sleep(evolution_interval)
        
        # Start evolution loop
        asyncio.create_task(evolution_loop())
    
    def stop_evolution(self):
        """Stop background pattern evolution"""
        self.is_running = False
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        space_stats = self.aether_space.get_space_statistics()
        
        return {
            "is_running": self.is_running,
            "aether_space": space_stats,
            "performance_metrics": self.performance_metrics.copy(),
            "processing_queue_size": len(self.processing_queue),
            "registered_encodings": list(self.pattern_factories.keys())
        }