# ================================================================
#  REALITY KERNEL - DATA CLASSES MODULE
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
#  Version: 1.0.0
#  Date: Current Date
# ================================================================

from dataclasses import dataclass, field, asdict, fields
from typing import Dict, List, Tuple, Optional, Any, ClassVar
from collections import deque
import numpy as np
import time
import uuid
import logging

logger = logging.getLogger(__name__)

# Attempt to import external types; provide dummies if not found for standalone usability
try:
    from aether_engine import AetherPattern
except ImportError:
    logger.warning("AetherPattern not found, using dummy class for RealityAnchor.")
    @dataclass
    class AetherPattern:
        id: str = field(default_factory=lambda: f"DummyPattern_{uuid.uuid4().hex[:4]}")
        complexity: float = 0.0
        # Add other fields AetherPattern is expected to have if used by RealityAnchor methods
        def to_dict(self): return asdict(self)
        @classmethod
        def from_dict(cls, data): return cls(**{k:v for k,v in data.items() if k in cls.__annotations__})


try:
    from quantum_physics import QuantumStateVector
except ImportError:
    logger.warning("QuantumStateVector not found, using dummy class for RealityAnchor.")
    @dataclass
    class QuantumStateVector:
        num_qubits: int = 1
        vector: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=complex))
        def to_dict(self): return {"num_qubits": self.num_qubits, "vector_data_placeholder": self.vector.tolist()[:4]}
        @classmethod
        def from_dict(cls,data): 
            nq = data.get('num_qubits',1)
            vec_data = data.get('vector_data_placeholder', [1.0] + [0.0]*( (2**nq) -1))
            qsv = cls(nq)
            if len(vec_data) == 2**nq : qsv.vector = np.array(vec_data, dtype=complex) # basic restore
            return qsv


@dataclass
class RealityAnchor:
    """Data structure for reality anchors, representing stable points or patterns within the reality fabric."""
    anchor_id: str = field(default_factory=lambda: f"RA_{uuid.uuid4().hex[:8]}")
    position: Optional[Tuple[float, ...]] = None  # N-dimensional position
    stability: float = 1.0  # 0.0 (highly unstable) to 1.0 (perfectly stable)
    resonance_signature: Optional[np.ndarray] = None  # Unique vibrational pattern
    connected_realities: List[str] = field(default_factory=list)  # IDs of other realities or dimensions
    integrity_level: float = 1.0  # 0.0 (compromised) to 1.0 (perfect integrity)
    aether_pattern_id: Optional[str] = None  # ID of the associated AetherPattern
    quantum_state_vector: Optional[QuantumStateVector] = None # Associated QuantumStateVector object
    perceptual_interface: Dict[str, Any] = field(default_factory=dict) # Configuration for how this anchor is perceived
    temporal_signature: Optional[np.ndarray] = None  # Temporal characteristics or behavior
    creation_timestamp: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time) # Renamed from last_update_timestamp for consistency with subtask
    metadata: Dict[str, Any] = field(default_factory=dict) # For additional descriptive data

    def __post_init__(self):
        # Ensure numpy arrays are correctly initialized if default_factory is used elsewhere for them
        if self.position is not None and not isinstance(self.position, tuple): # Example if it could be list
             self.position = tuple(self.position)
        # Similar checks for resonance_signature and temporal_signature if they could be lists initially

    def update_stability(self, change: float):
        self.stability = np.clip(self.stability + change, 0.0, 1.0)
        self.last_update = time.time()
        logger.debug(f"RealityAnchor {self.anchor_id} stability updated to {self.stability:.3f}")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the RealityAnchor to a dictionary."""
        data = asdict(self)
        if isinstance(self.position, tuple): # Already a tuple, which is JSON serializable
            pass
        if isinstance(self.resonance_signature, np.ndarray):
            data['resonance_signature'] = self.resonance_signature.tolist()
        if isinstance(self.temporal_signature, np.ndarray):
            data['temporal_signature'] = self.temporal_signature.tolist()
        
        if self.quantum_state_vector:
            if hasattr(self.quantum_state_vector, 'to_dict') and callable(getattr(self.quantum_state_vector, 'to_dict')):
                data['quantum_state_vector'] = self.quantum_state_vector.to_dict()
            else: 
                data['quantum_state_vector'] = str(self.quantum_state_vector) # Fallback
        
        if self.aether_pattern_id is None and isinstance(data.get('pattern'), AetherPattern): # Handle 'pattern' attribute if it exists from old spec
            pattern_obj = data.pop('pattern')
            if hasattr(pattern_obj, 'id'): data['aether_pattern_id'] = pattern_obj.id


        # Compatibility for 'stability_index' and 'last_update' if loading old data via from_dict
        if 'stability_index' in data: data.pop('stability_index') # Use 'stability'
        
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RealityAnchor':
        """Deserializes a RealityAnchor from a dictionary."""
        field_names = {f.name for f in fields(cls)}
        
        # Handle potential old field names for compatibility
        if 'pattern' in data and 'aether_pattern_id' not in data: # If old 'pattern' field exists
             pattern_data = data.pop('pattern')
             if isinstance(pattern_data, dict) and 'id' in pattern_data :
                 data['aether_pattern_id'] = pattern_data['id']
             # If pattern_data is AetherPattern object, its ID might be directly accessible.
             # This part depends on how AetherPattern was serialized if it wasn't just an ID.
                 
        if 'quantum_link' in data and 'quantum_state_vector' not in data:
            data['quantum_state_vector'] = data.pop('quantum_link')

        if 'stability_index' in data and 'stability' not in data:
            data['stability'] = data.pop('stability_index')

        filtered_data = {k: v for k, v in data.items() if k in field_names}

        for key in ['position', 'resonance_signature', 'temporal_signature']:
            if key in filtered_data and isinstance(filtered_data[key], list):
                try:
                    filtered_data[key] = np.array(filtered_data[key])
                except Exception as e:
                    logger.error(f"Error converting field {key} to numpy array: {e}. Keeping as list or None.")
                    filtered_data[key] = None # Or keep as list if appropriate for the type hint (Optional[Tuple] for position)
        
        if 'position' in filtered_data and isinstance(filtered_data['position'], np.ndarray):
            filtered_data['position'] = tuple(filtered_data['position'])


        qsv_data = filtered_data.get('quantum_state_vector')
        if isinstance(qsv_data, dict):
            try:
                # If QuantumStateVector has from_dict, use it:
                # filtered_data['quantum_state_vector'] = QuantumStateVector.from_dict(qsv_data)
                # Else, basic reconstruction if it's a dummy or simple structure:
                filtered_data['quantum_state_vector'] = QuantumStateVector(num_qubits=qsv_data.get('num_qubits',1)) 
            except Exception as e:
                logger.error(f"Error reconstructing QuantumStateVector: {e}")
                filtered_data['quantum_state_vector'] = None
        elif not isinstance(qsv_data, QuantumStateVector): # If it's not a dict or QSV, nullify
             filtered_data['quantum_state_vector'] = None
            
        return cls(**filtered_data)

@dataclass
class RealityMetrics:
    """Tracks and analyzes reality simulation performance and integrity."""
    sampling_rate: int = 1000 # Default sampling rate for history deques
    coherence_level: float = 1.0 
    entropy_rate: float = 0.0 
    paradox_count: int = 0 
    computational_load: float = 0.0 
    event_throughput: float = 0.0 
    temporal_stability: float = 1.0 
    quantum_entanglement_density: float = 0.0 
    aetheric_flux_intensity: float = 0.0 
    narrative_cohesion: float = 1.0 
    ethical_balance: float = 0.0 # Changed from ethical_tension for consistency with example
    last_snapshot_time: Optional[float] = None # Changed from last_updated_timestamp
    total_energy_observed: float = 0.0

    coherence_history: deque = field(init=False)
    qbit_efficiency_history: deque = field(init=False)
    timeline_divergence_history: deque = field(init=False)
    perception_latency_history: deque = field(init=False)
    entropy_rate_history: deque = field(init=False)
    paradox_count_history: deque = field(init=False)
        
    def __post_init__(self):
        # Initialize deques with sampling_rate as maxlen
        self.coherence_history = deque(maxlen=self.sampling_rate)
        self.qbit_efficiency_history = deque(maxlen=self.sampling_rate)
        self.timeline_divergence_history = deque(maxlen=self.sampling_rate)
        self.perception_latency_history = deque(maxlen=self.sampling_rate)
        self.entropy_rate_history = deque(maxlen=self.sampling_rate)
        self.paradox_count_history = deque(maxlen=self.sampling_rate)

    def update(self, kernel_state: Dict[str, Any]): 
        """Update metrics based on current kernel state snapshot."""
        self.coherence_level = float(kernel_state.get('overall_coherence', self.coherence_level))
        self.entropy_rate = float(kernel_state.get('current_entropy_rate', self.entropy_rate))
        self.paradox_count = int(kernel_state.get('active_paradox_count', self.paradox_count))
        self.computational_load = float(kernel_state.get('current_comp_load', self.computational_load))
        self.event_throughput = float(kernel_state.get('events_per_cycle', self.event_throughput))
        
        # For metrics that might be nested in kernel_state
        timeline_metrics_data = kernel_state.get('timeline_metrics', {})
        self.temporal_stability = float(timeline_metrics_data.get('stability', self.temporal_stability) if isinstance(timeline_metrics_data, dict) else self.temporal_stability)
        self.timeline_divergence_history.append(float(timeline_metrics_data.get('divergence_metric', 0.0) if isinstance(timeline_metrics_data, dict) else 0.0)) # Example key

        entanglement_metrics_data = kernel_state.get('entanglement_metrics', {})
        self.quantum_entanglement_density = float(entanglement_metrics_data.get('density', self.quantum_entanglement_density) if isinstance(entanglement_metrics_data, dict) else self.quantum_entanglement_density)
        self.qbit_efficiency_history.append(float(entanglement_metrics_data.get('qbit_efficiency', 0.95) if isinstance(entanglement_metrics_data, dict) else 0.95)) # Example key

        aether_metrics_data = kernel_state.get('aether_metrics', {})
        self.aetheric_flux_intensity = float(aether_metrics_data.get('flux_intensity', self.aetheric_flux_intensity) if isinstance(aether_metrics_data, dict) else self.aetheric_flux_intensity)
        
        narrative_metrics_data = kernel_state.get('narrative_metrics',{})
        self.narrative_cohesion = float(narrative_metrics_data.get('cohesion', self.narrative_cohesion) if isinstance(narrative_metrics_data, dict) else self.narrative_cohesion)
        
        ethical_metrics_data = kernel_state.get('ethical_metrics', {})
        self.ethical_balance = float(ethical_metrics_data.get('balance', self.ethical_balance) if isinstance(ethical_metrics_data, dict) else self.ethical_balance) # Changed from tension

        self.total_energy_observed = float(kernel_state.get('total_energy_observed', self.total_energy_observed))
        self.perception_latency_history.append(float(kernel_state.get('current_perception_latency_ms', 0.0) if isinstance(kernel_state.get('current_perception_latency_ms'), (int,float)) else 0.0))


        self.coherence_history.append(self.coherence_level)
        self.entropy_rate_history.append(self.entropy_rate)
        self.paradox_count_history.append(self.paradox_count) 
        self.last_snapshot_time = time.time() # Update timestamp
        logger.debug(f"RealityMetrics updated at {self.last_snapshot_time}")

    def get_summary(self) -> Dict[str, Any]: 
        """Return a dictionary of current metric values, including averages from history."""
        summary = {
            "coherence_level": self.coherence_level, "entropy_rate": self.entropy_rate,
            "paradox_count": self.paradox_count, "computational_load": self.computational_load,
            "event_throughput": self.event_throughput, "temporal_stability": self.temporal_stability,
            "quantum_entanglement_density": self.quantum_entanglement_density,
            "aetheric_flux_intensity": self.aetheric_flux_intensity,
            "narrative_cohesion": self.narrative_cohesion, "ethical_balance": self.ethical_balance,
            "total_energy_observed": self.total_energy_observed,
            "last_snapshot_time": self.last_snapshot_time
        }
        if self.coherence_history: summary["avg_coherence_history"] = float(np.mean(list(self.coherence_history)))
        if self.entropy_rate_history: summary["avg_entropy_rate_history"] = float(np.mean(list(self.entropy_rate_history)))
        if self.paradox_count_history: summary["total_paradoxes_in_history_window"] = sum(self.paradox_count_history)
        if self.qbit_efficiency_history: summary["avg_qbit_efficiency_history"] = float(np.mean(list(self.qbit_efficiency_history)))
        if self.timeline_divergence_history: summary["avg_timeline_divergence_history"] = float(np.mean(list(self.timeline_divergence_history)))
        if self.perception_latency_history: summary["avg_perception_latency_history_ms"] = float(np.mean(list(self.perception_latency_history)))
        return summary
        
    def to_dict(self) -> Dict[str, Any]: 
        data = asdict(self)
        # Convert deques to lists for serialization
        for key, value in data.items():
            if isinstance(value, deque): data[key] = list(value)
        data.pop('_history_maxlen', None) # Ensure ClassVar is not in dict
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RealityMetrics':
        init_data = data.copy()
        # Use the sampling_rate from the data if present, otherwise use default
        maxlen = init_data.get('sampling_rate', 1000)
        
        history_fields = {
            "coherence_history", "entropy_rate_history", "paradox_count_history",
            "qbit_efficiency_history", "timeline_divergence_history", "perception_latency_history"
        }
        for key in history_fields:
            if key in init_data and isinstance(init_data[key], list): 
                init_data[key] = deque(init_data[key], maxlen=maxlen)
        
        # Ensure only fields defined in the dataclass are passed to constructor
        defined_field_names = {f.name for f in fields(cls)}
        
        filtered_init_data = {k: v for k, v in init_data.items() if k in defined_field_names}
        
        return cls(**filtered_init_data)

logger.info("reality_kernel_dataclasses.py defined with RealityAnchor and RealityMetrics.")
        return cls(**filtered_init_data)

logger.info("reality_kernel_dataclasses.py defined with RealityAnchor and RealityMetrics.")
