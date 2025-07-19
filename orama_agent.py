# ================================================================
#  ORAMA - Observation, Reasoning, And Memory Agent
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
# ================================================================
import os
import re
import json
import logging
import datetime
import hashlib
import subprocess
import time
import argparse
import threading
import asyncio
import signal
import traceback
import math
import sys # For sys.exit in main
from typing import Dict, Any, List, Union, Optional, Tuple, Callable, Set
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field, asdict

# Attempt to import core engine components
try:
    # Assuming these are top-level modules or correctly in PYTHONPATH
    from timeline_engine import TimelineEngine, TemporalEvent, TimelineMetrics
    from quantum_physics import QuantumField, PhysicsConstants, EthicalGravityManifold
    from aether_engine import AetherEngine, AetherPattern, AetherSpace
    from reality_kernel import RealityKernel, RealityAnchor
    from universe_engine import UniverseEngine
    from paradox_engine import ParadoxEngine
    from mind_seed import MemoryEcho, IdentityMatrix, BreathCycle, NarrativeManifold as MindSeedNarrativeManifold, PerceptionIntegrator as MindSeedPerceptionIntegrator, BehaviorEngine as MindSeedBehaviorEngine, RecursiveSimulator as MindSeedRecursiveSimulator
    from cosmic_scroll import DimensionalRealityManager as CosmicDRM 
    COSMOS_ENGINE_AVAILABLE = True
except ImportError as e:
    COSMOS_ENGINE_AVAILABLE = False
    logging.warning(f"Core Genesis Cosmos Engine components not found: {e}. ORAMA will run with limited engine interaction functionality.")
    # Define dummy/placeholder classes if core components are missing
    class TimelineEngine: imported = True,
    class QuantumField: imported = True
    class PhysicsConstants: imported = True
    class EthicalGravityManifold: pass
    class AetherEngine: pass
    class RealityKernel: pass
    class UniverseEngine: pass
    class ParadoxEngine: pass
    class MemoryEcho: pass
    class IdentityMatrix: pass
    class BreathCycle: pass
    class MindSeedNarrativeManifold: pass
    class MindSeedPerceptionIntegrator: pass
    class MindSeedBehaviorEngine: pass
    class MindSeedRecursiveSimulator: pass
    class CosmicDRM: pass


# ================================================================
#  Configuration and Logging Setup
# ================================================================
LOG_LEVEL = os.environ.get("ORAMA_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] 
)

# Logger instances
system_logger = logging.getLogger("Orama.System")
perception_logger = logging.getLogger("Orama.Perception")
memory_logger = logging.getLogger("Orama.Memory")
knowledge_logger = logging.getLogger("Orama.Knowledge")
truth_logger = logging.getLogger("Orama.TruthValidator")
terminal_logger = logging.getLogger("Orama.Terminal")

# Directories and files
LOG_DIR = Path(os.environ.get("ORAMA_LOG_DIR", "orama_logs"))
MEMORY_FILE = LOG_DIR / os.environ.get("ORAMA_MEMORY_FILE", "orama_memory.json")
KNOWLEDGE_FILE = LOG_DIR / os.environ.get("ORAMA_KNOWLEDGE_FILE", "orama_knowledge.json")
SYSTEM_LOG_FILE = LOG_DIR / "orama_system.log"
PERCEPTION_LOG_FILE = LOG_DIR / "perception_stream.log"

PERCEPTION_BUFFER_SIZE = 1000
MAX_OUTPUT_LENGTH = 2000 

LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_file_handler(logger_instance, file_path, level=LOG_LEVEL, max_bytes=10*1024*1024, backup_count=5):
    handler = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(level)
    # Check if handlers are already present to avoid duplication if this function is called multiple times
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == str(file_path) for h in logger_instance.handlers):
        logger_instance.addHandler(handler)
    logger_instance.propagate = False 

setup_file_handler(system_logger, SYSTEM_LOG_FILE)
setup_file_handler(perception_logger, PERCEPTION_LOG_FILE)
setup_file_handler(memory_logger, MEMORY_FILE.with_suffix(".log"))
setup_file_handler(knowledge_logger, KNOWLEDGE_FILE.with_suffix(".log"))
setup_file_handler(truth_logger, LOG_DIR / "truth_validator.log")
setup_file_handler(terminal_logger, LOG_DIR / "terminal_agent.log")

# ================================================================
#  Data Models
# ================================================================

@dataclass
class MemoryEvent:
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    event_type: str = "UNKNOWN_EVENT"
    content: str = ""
    source: str = "INTERNAL_ORAMA"
    importance: float = 0.5 
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash_id: Optional[str] = None
    access_count: int = 0
    last_accessed: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def __post_init__(self):
        if not self.hash_id:
            data_to_hash = f"{self.timestamp}-{self.event_type}-{self.source}-{self.content}"
            for k, v in sorted(self.metadata.items()): 
                data_to_hash += f"-{k}:{v}"
            self.hash_id = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()[:16]
        if not self.keywords and self.content:
            self.keywords = list(set(re.findall(r'\b[a-zA-Z]{4,}\b', self.content.lower())))[:10] # Basic keyword extraction

    def accessed(self):
        self.access_count += 1
        self.last_accessed = datetime.datetime.now(datetime.timezone.utc).isoformat()

@dataclass
class KnowledgeEntity:
    entity_id: str 
    entity_type: str 
    name: str 
    attributes: Dict[str, Any] = field(default_factory=dict) 
    relationships: List[Dict[str, str]] = field(default_factory=list) 
    confidence: float = 0.5 # Initial confidence might be lower, updated with more evidence
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    source_memory_ids: Set[str] = field(default_factory=set) # Use set for uniqueness
    summary: Optional[str] = None 
    tags: List[str] = field(default_factory=list)

    def update_timestamp(self):
        self.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def add_relationship(self, relationship_type: str, target_entity_id: str, source: str = "inference"):
        self.relationships.append({"type": relationship_type, "target_id": target_entity_id, "source": source})
        self.update_timestamp()

    def add_attribute(self, key: str, value: Any, source: str = "inference"):
        self.attributes[key] = {"value": value, "source": source, "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        self.update_timestamp()

@dataclass
class SimulationPerception:
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    content: str = ""
    source: str = "SIMULATION_CORE"
    perception_type: str = "GENERAL_OBSERVATION"
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[Any] = None 
    
    def to_memory_event(self) -> MemoryEvent:
        mem_content = self.content
        if self.raw_data and not self.content: # If content is empty but raw_data exists
            mem_content = str(self.raw_data)

        keywords = list(set(re.findall(r'\b[a-zA-Z]{4,}\b', mem_content.lower()) + list(self.metadata.keys())))[:15]
        return MemoryEvent(
            timestamp=self.timestamp,
            event_type=f"PERCEPTION_{self.perception_type.upper().replace(' ','_')}",
            content=mem_content,
            source=self.source,
            metadata=self.metadata,
            keywords=keywords
        )

@dataclass
class QueryContext:
    query_text: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    user_context: Dict[str, Any] = field(default_factory=dict) 
    relevant_memories: List[MemoryEvent] = field(default_factory=list)
    relevant_knowledge: List[KnowledgeEntity] = field(default_factory=list)
    recent_perceptions: List[SimulationPerception] = field(default_factory=list)
    conversation_history: List[Tuple[str,str]] = field(default_factory=list)

@dataclass
class OramaState:
    is_initialized: bool = False
    last_query_time: Optional[str] = None
    last_perception_time: Optional[str] = None
    perception_count: int = 0
    query_count: int = 0
    known_entity_ids: Set[str] = field(default_factory=set)
    error_count: int = 0
    engine_status: str = "UNINITIALIZED"
    last_knowledge_synthesis_tick: int = 0 # Using tick or timestamp consistently
    last_memory_save_tick: int = 0
    system_start_time: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    active_threads: Dict[str, Any] = field(default_factory=dict) # For managing background tasks


# ================================================================
#  ORAMA Components
# ================================================================

class OracleMemoryManager:
    """Manages persistent memory storage, retrieval, and maintenance"""
    def __init__(self, memory_file: Union[str, Path] = MEMORY_FILE, max_memories: int = 10000):
        self.memory_file = Path(memory_file)
        self.max_memories = max_memories
        self.memories: List[MemoryEvent] = []
        self.memory_index: Dict[str, int] = {} 
        self._lock = threading.Lock() # For thread-safe operations
        self.load_memories()
        memory_logger.info(f"Memory manager initialized. Loaded {len(self.memories)} memories from {self.memory_file if self.memory_file.exists() else 'new memory store'}.")
    
    def load_memories(self) -> None:
        with self._lock:
            try:
                if self.memory_file.exists():
                    with open(self.memory_file, 'r', encoding='utf-8') as f:
                        memory_data = json.load(f)
                        self.memories = [MemoryEvent(**mem_dict) for mem_dict in memory_data]
                        self.memory_index = {mem.hash_id: i for i, mem in enumerate(self.memories) if mem.hash_id}
                        memory_logger.info(f"Loaded {len(self.memories)} memories.")
                else:
                    memory_logger.info(f"No memory file found at {self.memory_file}, starting fresh.")
            except json.JSONDecodeError:
                memory_logger.error(f"Failed to decode JSON from {self.memory_file}. Starting with empty memory.")
                self.memories = []
                self.memory_index = {}
            except Exception as e:
                memory_logger.error(f"Error loading memories: {e}", exc_info=True)
                self.memories = []
                self.memory_index = {}

    def save_memories(self) -> None:
        with self._lock:
            try:
                # Ensure directory exists
                self.memory_file.parent.mkdir(parents=True, exist_ok=True)
                memory_data = [asdict(mem) for mem in self.memories]
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    json.dump(memory_data, f, indent=2, ensure_ascii=False)
                memory_logger.info(f"Saved {len(self.memories)} memories to {self.memory_file}")
            except Exception as e:
                memory_logger.error(f"Error saving memories: {e}", exc_info=True)
    
    def add_memory(self, memory: MemoryEvent) -> str:
        with self._lock:
            if not memory.hash_id: # Should be generated by __post_init__
                memory_logger.error(f"MemoryEvent missing hash_id: {memory.content[:50]}")
                return "" # Or raise error
            if memory.hash_id in self.memory_index:
                # Update existing memory's access count and timestamp if duplicate content
                existing_mem_idx = self.memory_index[memory.hash_id]
                self.memories[existing_mem_idx].accessed()
                self.memories[existing_mem_idx].importance = max(self.memories[existing_mem_idx].importance, memory.importance) # Keep higher importance
                memory_logger.debug(f"Memory {memory.hash_id} already exists, updated access stats.")
                return memory.hash_id
            
            self.memories.append(memory)
            self.memory_index[memory.hash_id] = len(self.memories) - 1
            
            if len(self.memories) > self.max_memories:
                self._prune_memories()
            memory_logger.debug(f"Added memory {memory.hash_id}: {memory.content[:50]}...")
            return memory.hash_id
    
    def get_memory(self, hash_id: str) -> Optional[MemoryEvent]:
        with self._lock:
            idx = self.memory_index.get(hash_id)
            if idx is not None and idx < len(self.memories):
                mem = self.memories[idx]
                mem.accessed()
                return mem
            return None
    
    def get_recent_memories(self, count: int = 10, event_type: Optional[str] = None) -> List[MemoryEvent]:
        with self._lock:
            temp_memories = self.memories
            if event_type:
                temp_memories = [mem for mem in temp_memories if mem.event_type == event_type]
            # Already sorted by insertion time (newest at end)
            return temp_memories[-count:] 
    
    def search_memories(self, query: str, limit: int = 10, search_type: str = "keyword") -> List[MemoryEvent]:
        with self._lock:
            query_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
            if not query_terms: return []

            results_with_scores = []
            for mem in self.memories:
                score = 0
                # Keyword scoring
                matched_keywords = query_terms.intersection(set(k.lower() for k in mem.keywords))
                score += len(matched_keywords) * 2 # Higher weight for keyword match

                # Content scoring (simple substring)
                if query.lower() in mem.content.lower():
                    score += 1
                
                # Metadata scoring (check if query terms are in metadata values)
                for k, v_obj in mem.metadata.items():
                    if isinstance(v_obj, str) and query.lower() in v_obj.lower():
                        score += 0.5
                        break 
                
                if score > 0:
                    # Boost by importance and recency
                    recency = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(mem.timestamp)).total_seconds()
                    score += mem.importance * 1.5 
                    score -= math.log1p(recency / (3600*24)) * 0.1 # Decay score by days old
                    results_with_scores.append((mem, score))
            
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            return [mem_score[0] for mem_score in results_with_scores[:limit]]

    def _prune_memories(self) -> None:
        # This method is called from within add_memory, which already holds the lock
        if len(self.memories) <= self.max_memories: return
        
        num_to_remove = len(self.memories) - self.max_memories
        memory_logger.info(f"Pruning {num_to_remove} memories. Current count: {len(self.memories)}")
        
        now_ts = datetime.datetime.now(datetime.timezone.utc)
        scored_memories = []
        for i, mem in enumerate(self.memories):
            age_seconds = (now_ts - datetime.datetime.fromisoformat(mem.last_accessed)).total_seconds()
            age_days = age_seconds / (3600 * 24)
            
            # Score: lower is better for pruning. High importance, high access, recent access = higher score (less prunable)
            # Base score on inverse importance (low importance = high score)
            score = (1.0 - mem.importance) * 100.0 
            score -= math.log1p(mem.access_count + 1) * 10 # More accesses = lower score
            score += math.log1p(age_days + 1) * 5 # Older (last_accessed) = higher score
            scored_memories.append((score, mem.hash_id))
        
        scored_memories.sort(key=lambda x: x[0], reverse=True) # Sort so highest scores (most prunable) are first
        
        hashes_to_remove = {scored_mem[1] for scored_mem in scored_memories[:num_to_remove]}
        
        self.memories = [mem for mem in self.memories if mem.hash_id not in hashes_to_remove]
        self.memory_index = {mem.hash_id: i for i, mem in enumerate(self.memories) if mem.hash_id}
        memory_logger.info(f"Pruned. New memory count: {len(self.memories)}")


class TruthValidator:
    """Validates and cross-references information for consistency and truthfulness"""
    def __init__(self, memory_manager: OracleMemoryManager):
        self.memory_manager = memory_manager
        self.validation_patterns = [
            r'(?i)\b(?:true|false|correct|incorrect|valid|invalid)\b',
            r'(?i)\b(?:fact|fiction|real|fake|authentic|bogus)\b',
            r'(?i)\b(?:confirmed|denied|verified|disputed)\b'
        ]
        truth_logger.info("TruthValidator initialized.")
    
    def validate_statement(self, statement: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate a statement against known facts and context"""
        validation_result = {
            "statement": statement,
            "confidence": 0.5,
            "contradictions": [],
            "supporting_evidence": [],
            "validation_score": 0.0,
            "sources": [],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
        try:
            # Search for related memories
            related_memories = self.memory_manager.search_memories(statement, limit=20)
            
            # Analyze contradictions and support
            for memory in related_memories:
                similarity_score = self._calculate_similarity(statement, memory.content)
                if similarity_score > 0.3:  # Threshold for relevance
                    if self._detect_contradiction(statement, memory.content):
                        validation_result["contradictions"].append({
                            "memory_id": memory.hash_id,
                            "content": memory.content,
                            "confidence": memory.importance,
                            "similarity": similarity_score
                        })
                    else:
                        validation_result["supporting_evidence"].append({
                            "memory_id": memory.hash_id,
                            "content": memory.content,
                            "confidence": memory.importance,
                            "similarity": similarity_score
                        })
                    
                    validation_result["sources"].append(memory.source)
            
            # Calculate validation score
            support_score = sum(ev["confidence"] * ev["similarity"] for ev in validation_result["supporting_evidence"])
            contradiction_score = sum(con["confidence"] * con["similarity"] for con in validation_result["contradictions"])
            
            validation_result["validation_score"] = support_score - contradiction_score
            validation_result["confidence"] = min(0.95, max(0.05, 0.5 + validation_result["validation_score"] * 0.1))
            
            truth_logger.debug(f"Validated statement with score {validation_result['validation_score']:.3f}")
            
        except Exception as e:
            truth_logger.error(f"Error validating statement: {e}", exc_info=True)
            validation_result["error"] = str(e)
        
        return validation_result
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate basic similarity between two texts"""
        words1 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text1.lower()))
        words2 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _detect_contradiction(self, statement1: str, statement2: str) -> bool:
        """Detect basic contradictions between statements"""
        # Simple contradiction detection patterns
        contradiction_patterns = [
            (r'\bis\b', r'\bis not\b'),
            (r'\bwas\b', r'\bwas not\b'),
            (r'\bwill\b', r'\bwill not\b'),
            (r'\btrue\b', r'\bfalse\b'),
            (r'\byes\b', r'\bno\b'),
            (r'\bexists\b', r'\bdoes not exist\b')
        ]
        
        stmt1_lower = statement1.lower()
        stmt2_lower = statement2.lower()
        
        for pos_pattern, neg_pattern in contradiction_patterns:
            if (re.search(pos_pattern, stmt1_lower) and re.search(neg_pattern, stmt2_lower)) or \
               (re.search(neg_pattern, stmt1_lower) and re.search(pos_pattern, stmt2_lower)):
                return True
        
        return False


class PerceptionParser:
    """Parses and processes simulation perceptions from various engine sources"""
    def __init__(self, buffer_size: int = PERCEPTION_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.perception_buffer: List[SimulationPerception] = []
        self.parsing_patterns = {
            'entity_detection': r'\b(?:entity|object|being|construct|form)\s+(?:id|name|type):\s*(\w+)',
            'event_detection': r'\b(?:event|action|process|transition):\s*([^,\n]+)',
            'state_change': r'\b(?:state|status|condition)\s+(?:changed|updated|modified)\s+(?:to|from)\s*([^,\n]+)',
            'error_pattern': r'\b(?:error|exception|failure|critical):\s*([^,\n]+)',
            'metric_pattern': r'\b(\w+):\s*([+-]?\d*\.?\d+)'
        }
        self._lock = threading.Lock()
        perception_logger.info("PerceptionParser initialized.")
    
    def parse_perception(self, raw_data: Any, source: str = "UNKNOWN") -> SimulationPerception:
        """Parse raw perception data into structured perception object"""
        perception = SimulationPerception(
            content="",
            source=source,
            perception_type="GENERAL_OBSERVATION",
            raw_data=raw_data
        )
        
        try:
            if isinstance(raw_data, str):
                perception.content = raw_data
            elif isinstance(raw_data, dict):
                perception.content = json.dumps(raw_data, indent=2)
                perception.metadata.update(raw_data)
            else:
                perception.content = str(raw_data)
            
            # Extract structured information
            perception.metadata.update(self._extract_patterns(perception.content))
            
            # Determine perception type
            perception.perception_type = self._classify_perception(perception.content, perception.metadata)
            
            perception_logger.debug(f"Parsed perception from {source}: {perception.perception_type}")
            
        except Exception as e:
            perception_logger.error(f"Error parsing perception: {e}", exc_info=True)
            perception.metadata["parse_error"] = str(e)
        
        return perception
    
    def add_to_buffer(self, perception: SimulationPerception) -> None:
        """Add perception to buffer with size management"""
        with self._lock:
            self.perception_buffer.append(perception)
            if len(self.perception_buffer) > self.buffer_size:
                self.perception_buffer = self.perception_buffer[-self.buffer_size:]
            perception_logger.debug(f"Added perception to buffer. Buffer size: {len(self.perception_buffer)}")
    
    def get_recent_perceptions(self, count: int = 10, perception_type: Optional[str] = None) -> List[SimulationPerception]:
        """Get recent perceptions from buffer"""
        with self._lock:
            perceptions = self.perception_buffer
            if perception_type:
                perceptions = [p for p in perceptions if p.perception_type == perception_type]
            return perceptions[-count:]
    
    def _extract_patterns(self, content: str) -> Dict[str, Any]:
        """Extract structured patterns from content"""
        extracted = {}
        
        for pattern_name, pattern in self.parsing_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                extracted[pattern_name] = matches
        
        return extracted
    
    def _classify_perception(self, content: str, metadata: Dict[str, Any]) -> str:
        """Classify the type of perception"""
        content_lower = content.lower()
        
        if 'error_pattern' in metadata:
            return "ERROR_DETECTION"
        elif 'entity_detection' in metadata:
            return "ENTITY_OBSERVATION"
        elif 'event_detection' in metadata:
            return "EVENT_OBSERVATION"
        elif 'state_change' in metadata:
            return "STATE_CHANGE"
        elif 'metric_pattern' in metadata:
            return "METRIC_OBSERVATION"
        elif any(word in content_lower for word in ['quantum', 'field', 'wave']):
            return "QUANTUM_OBSERVATION"
        elif any(word in content_lower for word in ['timeline', 'temporal', 'time']):
            return "TEMPORAL_OBSERVATION"
        elif any(word in content_lower for word in ['reality', 'kernel', 'anchor']):
            return "REALITY_OBSERVATION"
        else:
            return "GENERAL_OBSERVATION"


class TerminalAccessAgent:
    """Handles terminal command execution and system interaction"""
    def __init__(self, working_directory: Optional[Path] = None):
        self.working_directory = working_directory or Path.cwd()
        self.command_history: List[Dict[str, Any]] = []
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        terminal_logger.info(f"TerminalAccessAgent initialized. Working directory: {self.working_directory}")
    
    def execute_command(self, command: str, timeout: int = 30, capture_output: bool = True) -> Dict[str, Any]:
        """Execute a terminal command and return results"""
        execution_result = {
            "command": command,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": None,
            "execution_time": 0.0,
            "error": None
        }
        
        start_time = time.time()
        
        try:
            terminal_logger.info(f"Executing command: {command}")
            
            with self._lock:
                self.command_history.append({
                    "command": command,
                    "timestamp": execution_result["timestamp"],
                    "working_directory": str(self.working_directory)
                })
            
            # Execute command
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                cwd=self.working_directory,
                text=True,
                encoding='utf-8'
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                execution_result["stdout"] = stdout or ""
                execution_result["stderr"] = stderr or ""
                execution_result["return_code"] = process.returncode
                execution_result["success"] = process.returncode == 0
                
            except subprocess.TimeoutExpired:
                process.kill()
                execution_result["error"] = f"Command timed out after {timeout} seconds"
                terminal_logger.warning(f"Command timed out: {command}")
            
        except Exception as e:
            execution_result["error"] = str(e)
            terminal_logger.error(f"Error executing command '{command}': {e}", exc_info=True)
        
        execution_result["execution_time"] = time.time() - start_time
        
        # Truncate output if too long
        if len(execution_result["stdout"]) > MAX_OUTPUT_LENGTH:
            execution_result["stdout"] = execution_result["stdout"][:MAX_OUTPUT_LENGTH] + "\n... (output truncated)"
        if len(execution_result["stderr"]) > MAX_OUTPUT_LENGTH:
            execution_result["stderr"] = execution_result["stderr"][:MAX_OUTPUT_LENGTH] + "\n... (output truncated)"
        
        terminal_logger.debug(f"Command completed. Success: {execution_result['success']}, Time: {execution_result['execution_time']:.2f}s")
        
        return execution_result
    
    def start_background_process(self, command: str, process_id: str) -> bool:
        """Start a background process"""
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=self.working_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            with self._lock:
                self.active_processes[process_id] = process
            
            terminal_logger.info(f"Started background process '{process_id}': {command}")
            return True
            
        except Exception as e:
            terminal_logger.error(f"Error starting background process: {e}", exc_info=True)
            return False
    
    def stop_background_process(self, process_id: str) -> bool:
        """Stop a background process"""
        with self._lock:
            process = self.active_processes.get(process_id)
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                del self.active_processes[process_id]
                terminal_logger.info(f"Stopped background process '{process_id}'")
                return True
            return False
    
    def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a background process"""
        with self._lock:
            process = self.active_processes.get(process_id)
            if process:
                return {
                    "process_id": process_id,
                    "pid": process.pid,
                    "is_running": process.poll() is None,
                    "return_code": process.returncode
                }
            return None
    
    def get_command_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent command history"""
        with self._lock:
            return self.command_history[-limit:]


class KnowledgeSynthesizer:
    """Synthesizes knowledge entities from memories and maintains knowledge graph"""
    def __init__(self, knowledge_file: Union[str, Path] = KNOWLEDGE_FILE, memory_manager: Optional[OracleMemoryManager] = None):
        self.knowledge_file = Path(knowledge_file)
        self.memory_manager = memory_manager
        self.knowledge_entities: Dict[str, KnowledgeEntity] = {}
        self.entity_relationships: Dict[str, Set[str]] = {}
        self._lock = threading.Lock()
        self.synthesis_patterns = {
            'entity_patterns': [
                r'\b(?:the|a|an)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|was|has|does)',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:engine|system|module|component|agent)'
            ],
            'relationship_patterns': [
                r'([A-Za-z]+)\s+(?:connects to|links with|relates to|depends on)\s+([A-Za-z]+)',
                r'([A-Za-z]+)\s+(?:is part of|belongs to|contains)\s+([A-Za-z]+)'
            ]
        }
        self.load_knowledge()
        knowledge_logger.info(f"KnowledgeSynthesizer initialized. Loaded {len(self.knowledge_entities)} entities.")
    
    def load_knowledge(self) -> None:
        """Load knowledge entities from file"""
        with self._lock:
            try:
                if self.knowledge_file.exists():
                    with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                        knowledge_data = json.load(f)
                        for entity_data in knowledge_data.get('entities', []):
                            # Convert source_memory_ids from list to set if needed
                            if 'source_memory_ids' in entity_data:
                                entity_data['source_memory_ids'] = set(entity_data['source_memory_ids'])
                            entity = KnowledgeEntity(**entity_data)
                            self.knowledge_entities[entity.entity_id] = entity
                        
                        # Load relationships
                        self.entity_relationships = knowledge_data.get('relationships', {})
                        # Convert lists to sets
                        for entity_id, related_ids in self.entity_relationships.items():
                            if isinstance(related_ids, list):
                                self.entity_relationships[entity_id] = set(related_ids)
                        
                        knowledge_logger.info(f"Loaded {len(self.knowledge_entities)} knowledge entities.")
                else:
                    knowledge_logger.info("No knowledge file found, starting fresh.")
            except Exception as e:
                knowledge_logger.error(f"Error loading knowledge: {e}", exc_info=True)
                self.knowledge_entities = {}
                self.entity_relationships = {}
    
    def save_knowledge(self) -> None:
        """Save knowledge entities to file"""
        with self._lock:
            try:
                self.knowledge_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert sets to lists for JSON serialization
                entities_data = []
                for entity in self.knowledge_entities.values():
                    entity_dict = asdict(entity)
                    entity_dict['source_memory_ids'] = list(entity_dict['source_memory_ids'])
                    entities_data.append(entity_dict)
                
                relationships_data = {
                    entity_id: list(related_ids) 
                    for entity_id, related_ids in self.entity_relationships.items()
                }
                
                knowledge_data = {
                    'entities': entities_data,
                    'relationships': relationships_data,
                    'saved_at': datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
                
                with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                    json.dump(knowledge_data, f, indent=2, ensure_ascii=False)
                
                knowledge_logger.info(f"Saved {len(self.knowledge_entities)} knowledge entities.")
                
            except Exception as e:
                knowledge_logger.error(f"Error saving knowledge: {e}", exc_info=True)
    
    def synthesize_from_memories(self, memories: List[MemoryEvent]) -> List[str]:
        """Synthesize knowledge entities from a list of memories"""
        new_entity_ids = []
        
        try:
            for memory in memories:
                entities = self._extract_entities_from_memory(memory)
                for entity in entities:
                    entity_id = self._add_or_update_entity(entity, memory.hash_id)
                    if entity_id:
                        new_entity_ids.append(entity_id)
                
                # Extract relationships
                relationships = self._extract_relationships_from_memory(memory)
                for rel in relationships:
                    self._add_relationship(rel['source'], rel['target'], rel['type'])
            
            knowledge_logger.info(f"Synthesized {len(new_entity_ids)} knowledge entities from {len(memories)} memories.")
            
        except Exception as e:
            knowledge_logger.error(f"Error synthesizing knowledge: {e}", exc_info=True)
        
        return new_entity_ids
    
    def get_entity(self, entity_id: str) -> Optional[KnowledgeEntity]:
        """Get a knowledge entity by ID"""
        with self._lock:
            return self.knowledge_entities.get(entity_id)
    
    def search_entities(self, query: str, limit: int = 10) -> List[KnowledgeEntity]:
        """Search for knowledge entities"""
        with self._lock:
            query_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
            if not query_terms:
                return []
            
            scored_entities = []
            for entity in self.knowledge_entities.values():
                score = 0
                
                # Name matching
                entity_name_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', entity.name.lower()))
                score += len(query_terms.intersection(entity_name_terms)) * 3
                
                # Tag matching
                entity_tag_terms = set(' '.join(entity.tags).lower().split())
                score += len(query_terms.intersection(entity_tag_terms)) * 2
                
                # Summary matching
                if entity.summary:
                    summary_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', entity.summary.lower()))
                    score += len(query_terms.intersection(summary_terms))
                
                if score > 0:
                    score *= entity.confidence
                    scored_entities.append((entity, score))
            
            scored_entities.sort(key=lambda x: x[1], reverse=True)
            return [entity for entity, score in scored_entities[:limit]]
    
    def _extract_entities_from_memory(self, memory: MemoryEvent) -> List[Dict[str, Any]]:
        """Extract potential entities from memory content"""
        entities = []
        
        # Use patterns to find entities
        for pattern in self.synthesis_patterns['entity_patterns']:
            matches = re.findall(pattern, memory.content)
            for match in matches:
                entity_name = match.strip()
                if len(entity_name) > 2:  # Minimum length filter
                    entities.append({
                        'name': entity_name,
                        'type': self._classify_entity_type(entity_name, memory.content),
                        'confidence': min(0.8, memory.importance + 0.3),
                        'source_memory': memory.hash_id,
                        'content_context': memory.content
                    })
        
        # Extract from keywords
        for keyword in memory.keywords:
            if len(keyword) > 3 and keyword.istitle():
                entities.append({
                    'name': keyword,
                    'type': 'CONCEPT',
                    'confidence': memory.importance,
                    'source_memory': memory.hash_id,
                    'content_context': memory.content
                })
        
        return entities
    
    def _extract_relationships_from_memory(self, memory: MemoryEvent) -> List[Dict[str, str]]:
        """Extract relationships from memory content"""
        relationships = []
        
        for pattern in self.synthesis_patterns['relationship_patterns']:
            matches = re.findall(pattern, memory.content, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    relationships.append({
                        'source': match[0].strip(),
                        'target': match[1].strip(),
                        'type': 'RELATES_TO',
                        'source_memory': memory.hash_id
                    })
        
        return relationships
    
    def _classify_entity_type(self, entity_name: str, context: str) -> str:
        """Classify the type of entity"""
        name_lower = entity_name.lower()
        context_lower = context.lower()
        
        if any(word in name_lower for word in ['engine', 'system', 'module']):
            return 'SYSTEM_COMPONENT'
        elif any(word in name_lower for word in ['agent', 'manager', 'processor']):
            return 'AGENT'
        elif any(word in context_lower for word in ['quantum', 'field', 'wave']):
            return 'QUANTUM_ENTITY'
        elif any(word in context_lower for word in ['timeline', 'temporal', 'time']):
            return 'TEMPORAL_ENTITY'
        elif any(word in context_lower for word in ['reality', 'dimension', 'space']):
            return 'REALITY_ENTITY'
        else:
            return 'CONCEPT'
    
    def _add_or_update_entity(self, entity_data: Dict[str, Any], memory_id: str) -> Optional[str]:
        """Add or update a knowledge entity"""
        with self._lock:
            entity_id = hashlib.sha256(entity_data['name'].encode()).hexdigest()[:12]
            
            if entity_id in self.knowledge_entities:
                # Update existing entity
                existing = self.knowledge_entities[entity_id]
                existing.confidence = max(existing.confidence, entity_data['confidence'])
                existing.source_memory_ids.add(memory_id)
                existing.update_timestamp()
            else:
                # Create new entity
                new_entity = KnowledgeEntity(
                    entity_id=entity_id,
                    entity_type=entity_data['type'],
                    name=entity_data['name'],
                    confidence=entity_data['confidence'],
                    source_memory_ids={memory_id}
                )
                self.knowledge_entities[entity_id] = new_entity
            
            return entity_id
    
    def _add_relationship(self, source_name: str, target_name: str, rel_type: str) -> None:
        """Add a relationship between entities"""
        source_id = hashlib.sha256(source_name.encode()).hexdigest()[:12]
        target_id = hashlib.sha256(target_name.encode()).hexdigest()[:12]
        
        with self._lock:
            if source_id not in self.entity_relationships:
                self.entity_relationships[source_id] = set()
            self.entity_relationships[source_id].add(target_id)


class OramaSystem:
    """Main ORAMA system orchestrating all components"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.state = OramaState()
        
        # Initialize components
        self.memory_manager = OracleMemoryManager()
        self.truth_validator = TruthValidator(self.memory_manager)
        self.perception_parser = PerceptionParser()
        self.terminal_agent = TerminalAccessAgent()
        self.knowledge_synthesizer = KnowledgeSynthesizer(memory_manager=self.memory_manager)
        
        # Engine components (if available)
        self.engines = {}
        self.engine_status = {}
        
        self._shutdown_event = threading.Event()
        self._background_tasks = []
        
        system_logger.info("OramaSystem initialized.")
    
    def initialize_engines(self) -> bool:
        """Initialize cosmos engine components if available"""
        if not COSMOS_ENGINE_AVAILABLE:
            system_logger.warning("Cosmos engines not available. Running in limited mode.")
            self.state.engine_status = "ENGINES_UNAVAILABLE"
            return False
        
        try:
            # Initialize engines
            self.engines['timeline'] = TimelineEngine()
            self.engines['quantum'] = QuantumField()
            self.engines['aether'] = AetherEngine()
            self.engines['reality'] = RealityKernel()
            self.engines['universe'] = UniverseEngine()
            self.engines['paradox'] = ParadoxEngine()
            
            for engine_name, engine in self.engines.items():
                self.engine_status[engine_name] = "INITIALIZED"
            
            self.state.engine_status = "ENGINES_ACTIVE"
            system_logger.info("All cosmos engines initialized successfully.")
            return True
            
        except Exception as e:
            system_logger.error(f"Error initializing engines: {e}", exc_info=True)
            self.state.engine_status = "ENGINE_ERROR"
            return False
    
    def process_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """Process a user query and return response"""
        try:
            self.state.query_count += 1
            self.state.last_query_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            # Create query context
            context = QueryContext(
                query_text=query,
                user_context=user_context or {}
            )
            
            # Search for relevant memories
            context.relevant_memories = self.memory_manager.search_memories(query, limit=15)
            
            # Search for relevant knowledge
            context.relevant_knowledge = self.knowledge_synthesizer.search_entities(query, limit=10)
            
            # Get recent perceptions
            context.recent_perceptions = self.perception_parser.get_recent_perceptions(count=5)
            
            # Validate query for truth/consistency
            validation_result = self.truth_validator.validate_statement(query, context.user_context)
            
            # Generate response
            response = self._generate_response(context, validation_result)
            
            # Store query as memory
            query_memory = MemoryEvent(
                event_type="USER_QUERY",
                content=query,
                source="USER_INPUT",
                importance=0.7,
                metadata={
                    "user_context": context.user_context,
                    "response_generated": True,
                    "validation_score": validation_result.get("validation_score", 0.0)
                }
            )
            self.memory_manager.add_memory(query_memory)
            
            system_logger.info(f"Processed query: {query[:50]}...")
            return response
            
        except Exception as e:
            self.state.error_count += 1
            system_logger.error(f"Error processing query: {e}", exc_info=True)
            return f"I encountered an error processing your query: {str(e)}"
    
    def process_perception(self, raw_perception: Any, source: str = "SIMULATION") -> str:
        """Process a perception from simulation engines"""
        try:
            self.state.perception_count += 1
            self.state.last_perception_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            # Parse perception
            perception = self.perception_parser.parse_perception(raw_perception, source)
            
            # Add to buffer
            self.perception_parser.add_to_buffer(perception)
            
            # Convert to memory
            memory_event = perception.to_memory_event()
            memory_id = self.memory_manager.add_memory(memory_event)
            
            # Synthesize knowledge if significant
            if memory_event.importance > 0.6:
                self.knowledge_synthesizer.synthesize_from_memories([memory_event])
            
            perception_logger.info(f"Processed perception from {source}: {perception.perception_type}")
            
            return f"Perception processed: {perception.perception_type} from {source}"
            
        except Exception as e:
            self.state.error_count += 1
            system_logger.error(f"Error processing perception: {e}", exc_info=True)
            return f"Error processing perception: {str(e)}"
    
    def execute_terminal_command(self, command: str) -> Dict[str, Any]:
        """Execute a terminal command through the terminal agent"""
        try:
            result = self.terminal_agent.execute_command(command)
            
            # Store command execution as memory
            memory_event = MemoryEvent(
                event_type="TERMINAL_COMMAND",
                content=f"Command: {command}\nOutput: {result['stdout'][:200]}",
                source="TERMINAL_AGENT",
                importance=0.6 if result['success'] else 0.8,
                metadata={
                    "command": command,
                    "success": result['success'],
                    "return_code": result['return_code'],
                    "execution_time": result['execution_time']
                }
            )
            self.memory_manager.add_memory(memory_event)
            
            return result
            
        except Exception as e:
            system_logger.error(f"Error executing terminal command: {e}", exc_info=True)
            return {
                "command": command,
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "orama_state": asdict(self.state),
            "memory_count": len(self.memory_manager.memories),
            "knowledge_entity_count": len(self.knowledge_synthesizer.knowledge_entities),
            "perception_buffer_size": len(self.perception_parser.perception_buffer),
            "active_processes": list(self.terminal_agent.active_processes.keys()),
            "engine_status": self.engine_status,
            "uptime_seconds": (
                datetime.datetime.now(datetime.timezone.utc) - 
                datetime.datetime.fromisoformat(self.state.system_start_time)
            ).total_seconds()
        }
    
    def save_all_data(self) -> None:
        """Save all persistent data"""
        try:
            self.memory_manager.save_memories()
            self.knowledge_synthesizer.save_knowledge()
            system_logger.info("All data saved successfully.")
        except Exception as e:
            system_logger.error(f"Error saving data: {e}", exc_info=True)
    
    def shutdown(self) -> None:
        """Graceful shutdown of the system"""
        system_logger.info("Initiating ORAMA system shutdown...")
        
        self._shutdown_event.set()
        
        # Stop background processes
        for process_id in list(self.terminal_agent.active_processes.keys()):
            self.terminal_agent.stop_background_process(process_id)
        
        # Wait for background tasks
        for task in self._background_tasks:
            try:
                if hasattr(task, 'join'):
                    task.join(timeout=5)
            except Exception as e:
                system_logger.warning(f"Error stopping background task: {e}")
        
        # Save all data
        self.save_all_data()
        
        system_logger.info("ORAMA system shutdown complete.")
    
    def _generate_response(self, context: QueryContext, validation: Dict[str, Any]) -> str:
        """Generate a response based on query context and validation"""
        response_parts = []
        
        # Add validation information if relevant
        if validation.get("validation_score", 0) < -0.5:
            response_parts.append("⚠️  I found some contradictory information regarding your query.")
        elif validation.get("validation_score", 0) > 0.5:
            response_parts.append("✓ This appears to be consistent with my knowledge.")
        
        # Add knowledge context
        if context.relevant_knowledge:
            knowledge_names = [e.name for e in context.relevant_knowledge[:3]]
            response_parts.append(f"Related concepts: {', '.join(knowledge_names)}")
        
        # Add memory context
        if context.relevant_memories:
            recent_memory = context.relevant_memories[-1]
            response_parts.append(f"Recent relevant memory: {recent_memory.content[:100]}...")
        
        # Add perception context
        if context.recent_perceptions:
            recent_perception = context.recent_perceptions[-1]
            response_parts.append(f"Current observation: {recent_perception.content[:100]}...")
        
        # Basic response generation
        query_lower = context.query_text.lower()
        if any(word in query_lower for word in ['status', 'state', 'how are']):
            status = self.get_system_status()
            response_parts.append(f"System Status: {status['orama_state']['engine_status']}")
            response_parts.append(f"Memory: {status['memory_count']} events")
            response_parts.append(f"Knowledge: {status['knowledge_entity_count']} entities")
        
        elif any(word in query_lower for word in ['help', 'what can', 'capabilities']):
            response_parts.append("I can help you with:")
            response_parts.append("- Analyzing simulation data and perceptions")
            response_parts.append("- Maintaining persistent memory and knowledge")
            response_parts.append("- Executing terminal commands")
            response_parts.append("- Validating information consistency")
        
        else:
            response_parts.append(f"I understand you're asking about: {context.query_text}")
            if context.relevant_memories or context.relevant_knowledge:
                response_parts.append("I've found relevant information in my memory and knowledge base.")
            else:
                response_parts.append("I don't have specific information about this topic yet.")
        
        return "\n".join(response_parts) if response_parts else "I'm processing your query..."


def main():
    """Main entry point for ORAMA"""
    parser = argparse.ArgumentParser(description="ORAMA - Observation, Reasoning, And Memory Agent")
    parser.add_argument("--mode", choices=["interactive", "continuous", "query"], default="interactive",
                       help="Operation mode")
    parser.add_argument("--query", type=str, help="Single query to process (for query mode)")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Update log level
    logging.getLogger().setLevel(args.log_level)
    
    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            system_logger.error(f"Error loading config: {e}")
    
    # Initialize ORAMA system
    orama = OramaSystem(config)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        system_logger.info(f"Received signal {signum}, shutting down...")
        orama.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize engines
        orama.initialize_engines()
        orama.state.is_initialized = True
        
        # Run in specified mode
        if args.mode == "query" and args.query:
            response = orama.process_query(args.query)
            print(response)
        elif args.mode == "continuous":
            continuous_mode(orama)
        else:  # interactive
            interactive_chat_mode(orama)
    
    except KeyboardInterrupt:
        system_logger.info("Interrupted by user")
    except Exception as e:
        system_logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        orama.shutdown()


def continuous_mode(orama: OramaSystem):
    """Run ORAMA in continuous background mode"""
    system_logger.info("Starting ORAMA in continuous mode...")
    
    def background_synthesis():
        """Background task for knowledge synthesis"""
        while not orama._shutdown_event.is_set():
            try:
                # Get recent memories for synthesis
                recent_memories = orama.memory_manager.get_recent_memories(50)
                if recent_memories:
                    orama.knowledge_synthesizer.synthesize_from_memories(recent_memories)
                
                # Save data periodically
                current_time = time.time()
                if current_time - orama.state.last_memory_save_tick > 300:  # Every 5 minutes
                    orama.save_all_data()
                    orama.state.last_memory_save_tick = current_time
                
                time.sleep(30)  # Run every 30 seconds
            except Exception as e:
                system_logger.error(f"Error in background synthesis: {e}")
                time.sleep(60)  # Wait longer on error
    
    # Start background tasks
    synthesis_thread = threading.Thread(target=background_synthesis, daemon=True)
    synthesis_thread.start()
    orama._background_tasks.append(synthesis_thread)
    
    # Main continuous loop
    try:
        while not orama._shutdown_event.is_set():
            # Check for perceptions from engines (if available and connected)
            if COSMOS_ENGINE_AVAILABLE and orama.engines:
                # Here you would poll engines for new perceptions
                # This is a placeholder for actual engine integration
                pass
            
            time.sleep(1)  # Main loop delay
    
    except KeyboardInterrupt:
        system_logger.info("Continuous mode interrupted")
    finally:
        orama._shutdown_event.set()


def interactive_chat_mode(orama: OramaSystem):
    """Run ORAMA in interactive chat mode"""
    print("🧠 ORAMA - Observation, Reasoning, And Memory Agent")
    print("Type 'help' for commands, 'quit' to exit")
    print("-" * 50)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n🔮 Query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! 👋")
                break
            
            if user_input.lower() == 'help':
                print("""
Available commands:
- help: Show this help message
- status: Show system status
- memory <query>: Search memory
- knowledge <query>: Search knowledge entities  
- terminal <command>: Execute terminal command
- save: Save all data
- quit: Exit ORAMA
""")
                continue
            
            if user_input.lower() == 'status':
                status = orama.get_system_status()
                print(f"System Status: {status['orama_state']['engine_status']}")
                print(f"Uptime: {status['uptime_seconds']:.0f} seconds")
                print(f"Memories: {status['memory_count']}")
                print(f"Knowledge Entities: {status['knowledge_entity_count']}")
                print(f"Queries Processed: {status['orama_state']['query_count']}")
                print(f"Perceptions: {status['orama_state']['perception_count']}")
                continue
            
            if user_input.lower().startswith('memory '):
                query = user_input[7:]
                memories = orama.memory_manager.search_memories(query, limit=5)
                print(f"\nFound {len(memories)} relevant memories:")
                for i, mem in enumerate(memories, 1):
                    print(f"{i}. [{mem.event_type}] {mem.content[:100]}...")
                continue
            
            if user_input.lower().startswith('knowledge '):
                query = user_input[10:]
                entities = orama.knowledge_synthesizer.search_entities(query, limit=5)
                print(f"\nFound {len(entities)} relevant knowledge entities:")
                for i, entity in enumerate(entities, 1):
                    print(f"{i}. {entity.name} ({entity.entity_type}) - Confidence: {entity.confidence:.2f}")
                continue
            
            if user_input.lower().startswith('terminal '):
                command = user_input[9:]
                print(f"Executing: {command}")
                result = orama.execute_terminal_command(command)
                if result['success']:
                    print(f"✓ Success (exit code: {result['return_code']})")
                    if result['stdout']:
                        print(f"Output:\n{result['stdout']}")
                else:
                    print(f"✗ Failed (exit code: {result['return_code']})")
                    if result['stderr']:
                        print(f"Error:\n{result['stderr']}")
                continue
            
            if user_input.lower() == 'save':
                orama.save_all_data()
                print("✓ All data saved")
                continue
            
            # Process as regular query
            if user_input:
                response = orama.process_query(user_input)
                print(f"\n🤖 ORAMA: {response}")
                
                conversation_history.append((user_input, response))
                if len(conversation_history) > 20:  # Keep last 20 exchanges
                    conversation_history = conversation_history[-20:]
        
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit properly.")
        except EOFError:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            system_logger.error(f"Interactive mode error: {e}", exc_info=True)


if __name__ == "__main__":
    main()

system_logger.info("orama_agent.py module fully defined.")
