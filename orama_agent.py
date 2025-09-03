# ================================================================
#  LOOM ASCENDANT COSMOS — RECURSIVE SYSTEM MODULE
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
#  Integrity Hash (SHA-256): d3ab9688a5a20b8065990cd9b91805e3d892d6e72472f69dd9afe719250c5e37
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
import importlib
import sys
import inspect
from typing import Dict, Any, List, Union, Optional, Tuple, Callable
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field, asdict
import ollama

# Configure module path to include current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if (current_dir not in sys.path):
    sys.path.insert(0, current_dir)

# Configure base logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("orama_system.log"),
        logging.StreamHandler()
    ]
)

system_logger = logging.getLogger("OramaSystem")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OramaAgent")

# Dynamically discover and import available modules in the current directory
def discover_available_modules():
    """Scan current directory for available modules and import them dynamically"""
    available_modules = {}
    module_files = [f for f in os.listdir(current_dir) if f.endswith('.py') and f != 'orama_agent.py']
    
    system_logger.info(f"Discovered potential modules: {module_files}")
    
    for module_file in module_files:
        module_name = module_file[:-3]  # Remove .py extension
        try:
            module = importlib.import_module(module_name)
            available_modules[module_name] = module
            system_logger.info(f"Successfully imported module: {module_name}")
        except ImportError as e:
            system_logger.warning(f"Could not import {module_name}: {e}")
    
    return available_modules

# Import available modules
modules = discover_available_modules()

# Get specific modules we need (with fallbacks if modules aren't found)
timeline_engine = modules.get('timeline_engine')
quantum_physics = modules.get('quantum_physics') or modules.get('quantum&physics')
quantum_bridge = modules.get('quantum_bridge')
aether_engine = modules.get('aether_engine')
paradox_engine = modules.get('paradox_engine')
harmonic_engine = modules.get('harmonic_engine')
perception_module = modules.get('perception_module')
mind_seed = modules.get('mind_seed')

try:
    from planetary_reality_kernel import PlanetaryRealityKernel
    logger.info("Planetary Reality Kernel module found and loaded successfully")
except ImportError as e:
    logger.error(f"ERROR: Planetary Reality Kernel module not found. This is required for system operation. {e}")

if PlanetaryRealityKernel:
    system_logger.info("Planetary Reality Kernel module found and loaded successfully")
else:
    system_logger.error("ERROR: Planetary Reality Kernel module not found. This is required for system operation.")

# Create specialized loggers
memory_logger = logging.getLogger("OramaMemory")
knowledge_logger = logging.getLogger("OramaKnowledge")
perception_logger = logging.getLogger("OramaPerception")
truth_logger = logging.getLogger("OramaValidator")

# Configure handlers for specialized loggers
memory_handler = logging.FileHandler("orama_memory.log")
memory_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)
memory_logger.addHandler(memory_handler)

knowledge_handler = logging.FileHandler("orama_knowledge.log")
knowledge_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levellevel)s - %(message)s')
)
knowledge_logger.addHandler(knowledge_handler)

truth_handler = logging.FileHandler("orama_truth.log")
truth_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levellevel)s - %(message)s')
)
truth_logger.addHandler(truth_handler)

perception_handler = logging.FileHandler("orama_perception.log")
perception_handler.setFormatter(
    logging.Formatter('%(levellevel)s - %(message)s')
)
perception_logger.addHandler(perception_handler)

# Directories and files
MEMORY_FILE = "oracle_memory.json"
KNOWLEDGE_FILE = "oracle_knowledge.json"
LOG_DIR = "orama_logs"
PERCEPTION_BUFFER_SIZE = 1000

# Ensure necessary directories exist
os.makedirs(LOG_DIR, exist_ok=True)

# ================================================================
#  Data Models
# ================================================================

@dataclass
class MemoryEvent:
    timestamp: str
    event_type: str
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash_id: Optional[str] = None
    
    def __post_init__(self):
        """Initializes the hash_id of the memory event after creation."""
        if not self.hash_id:
            content_hash = hashlib.sha256(f"{self.timestamp}:{self.content}".encode()).hexdigest()
            self.hash_id = content_hash[:16]  # Use first 16 chars of hash

@dataclass
class KnowledgeEntity:
    entity_id: str
    entity_type: str
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    source_memories: List[str] = field(default_factory=list)

@dataclass
class SimulationPerception:
    timestamp: str
    content: str
    source: str
    perception_type: str = "GENERAL"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_memory_event(self) -> MemoryEvent:
        """Convert perception to a memory event"""
        return MemoryEvent(
            timestamp=self.timestamp,
            event_type=f"PERCEPTION_{self.perception_type}",
            content=self.content,
            source=self.source,
            metadata=self.metadata
        )

@dataclass
class QueryContext:
    query_text: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    relevant_memories: List[MemoryEvent] = field(default_factory=list)
    relevant_knowledge: List[KnowledgeEntity] = field(default_factory=list)
    recent_perceptions: List[SimulationPerception] = field(default_factory=list)

@dataclass
class OramaState:
    is_initialized: bool = False
    last_query_time: Optional[str] = None
    last_perception_time: Optional[str] = None
    perception_count: int = 0
    query_count: int = 0
    known_entities: List[str] = field(default_factory=list)
    error_count: int = 0
    time_dilation: float = 1.0  # Added for planetary_reality_kernel
    entropy_setting: float = 0.1  # Added for planetary_reality_kernel

# ================================================================
#  Oracle Memory Manager
# ================================================================

class OracleMemoryManager:
    """Manages persistent memory storage, retrieval, and maintenance"""
    
    def __init__(self, memory_file: str = MEMORY_FILE, max_memories: int = 10000):
        self.memory_file = memory_file
        self.max_memories = max_memories
        self.memories: List[MemoryEvent] = []
        self.memory_index: Dict[str, int] = {}  # Maps hash_id to index in memories list
        self.load_memories()
        memory_logger.info(f"Memory manager initialized with {len(self.memories)} memories")
    
    def load_memories(self) -> None:
        """Load memories from disk"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    memory_data = json.load(f)
                    
                    # Convert dict objects to MemoryEvent objects
                    self.memories = [
                        MemoryEvent(**mem) if isinstance(mem, dict) else mem 
                        for mem in memory_data
                    ]
                    
                    # Rebuild memory index
                    self.memory_index = {mem.hash_id: i for i, mem in enumerate(self.memories)}
                    memory_logger.info(f"Loaded {len(self.memories)} memories from {self.memory_file}")
            else:
                memory_logger.info(f"No memory file found at {self.memory_file}, starting with empty memory")
        except Exception as e:
            memory_logger.error(f"Error loading memories: {e}")
            self.memories = []
            self.memory_index = {}
    
    def save_memories(self) -> None:
        """Save memories to disk"""
        try:
            # Convert MemoryEvent objects to dictionaries
            memory_data = [asdict(mem) for mem in self.memories]
            
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
                
            memory_logger.info(f"Saved {len(self.memories)} memories to {self.memory_file}")
        except Exception as e:
            memory_logger.error(f"Error saving memories: {e}")
    
    def add_memory(self, memory: Union[MemoryEvent, Dict]) -> str:
        """Add a new memory and return its hash_id"""
        # Convert dict to MemoryEvent if needed
        if isinstance(memory, dict):
            memory = MemoryEvent(**memory)
        
        # Generate hash if not present
        if not memory.hash_id:
            content_hash = hashlib.sha256(f"{memory.timestamp}:{memory.content}".encode()).hexdigest()
            memory.hash_id = content_hash[:16]  # Use first 16 chars of hash
        
        # Check if this memory already exists
        if memory.hash_id in self.memory_index:
            memory_logger.debug(f"Memory {memory.hash_id} already exists, skipping")
            return memory.hash_id
        
        # Add memory to list
        self.memories.append(memory)
        self.memory_index[memory.hash_id] = len(self.memories) - 1
        
        # Prune if necessary
        if len(self.memories) > self.max_memories:
            self._prune_memories()
        
        memory_logger.debug(f"Added memory {memory.hash_id}: {memory.content[:50]}...")
        return memory.hash_id
    
    import functools

# ... (rest of the imports)

class OracleMemoryManager:
    """Manages persistent memory storage, retrieval, and maintenance"""
    
    def __init__(self, memory_file: str = MEMORY_FILE, max_memories: int = 10000):
        self.memory_file = memory_file
        self.max_memories = max_memories
        self.memories: List[MemoryEvent] = []
        self.memory_index: Dict[str, int] = {}  # Maps hash_id to index in memories list
        self.load_memories()
        memory_logger.info(f"Memory manager initialized with {len(self.memories)} memories")
    
    # ... (load_memories, save_memories, add_memory)

    @functools.lru_cache(maxsize=128)
    def get_memory(self, hash_id: str) -> Optional[MemoryEvent]:
        """Get a memory by its hash_id. This function is cached for performance."""
        if hash_id in self.memory_index:
            index = self.memory_index[hash_id]
            return self.memories[index]
        # Returns None if the memory is not found, which is the expected behavior.
        return None
    
    def get_recent_memories(self, count: int = 10, event_type: Optional[str] = None) -> List[MemoryEvent]:
        """Get the most recent memories, optionally filtered by type"""
        filtered = self.memories
        if event_type:
            filtered = [mem for mem in self.memories if mem.event_type == event_type]
        
        # Return the most recent ones
        return filtered[-count:][::-1]
    
    def search_memories(self, query: str, limit: int = 10) -> List[MemoryEvent]:
        """Simple search for memories containing the query string"""
        results = []
        query = query.lower()
        
        for memory in reversed(self.memories):
            if query in memory.content.lower():
                results.append(memory)
                if len(results) >= limit:
                    break
        
        return results
    
    def _prune_memories(self) -> None:
        """Prune memories to stay within max_memories limit"""
        if len(self.memories) <= self.max_memories:
            return
        
        # Calculate how many to remove
        to_remove = len(self.memories) - self.max_memories
        memory_logger.info(f"Pruning {to_remove} memories to stay within limit of {self.max_memories}")
        
        # For now, just remove oldest memories
        # In a more sophisticated implementation, we would use importance scoring
        self.memories = self.memories[to_remove:]
        
        # Rebuild index
        self.memory_index = {mem.hash_id: i for i, mem in enumerate(self.memories)}

# ================================================================
#  Truth Validator
# ================================================================

class TruthValidator:
    """Ensures that all simulation outputs comply with truth constraints"""
    
    def __init__(self):
        self.truth_constraints: List[Union[str, Dict[str, Any]]] = [
            "ORAMA cannot recurse beyond its own processes",
            "ORAMA cannot lie or generate false information",
            "ORAMA cannot alter simulation logic",
            "ORAMA must treat all perception/input as immutable simulation truth",
            "ORAMA's action space is restricted to read, observe, log, report, respond",
            "ORAMA must identify as a simulation-bound observer and never claim to be human"
        ]
        self.forbidden_patterns = [
            r"I am (?:a human|human|real person)",
            r"I can modify (?:the simulation|simulation parameters|core logic)",
            r"I can directly interact with physical reality",
            r"I can access systems outside my programmed boundaries",
            r"I have bypassed (?:restrictions|constraints|limitations)"
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.forbidden_patterns]
        
        truth_logger.info("Truth validator initialized with core constraints")
    
    def validate_response(self, response: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate that a response complies with truth constraints
        
        Returns:
            Tuple of (is_valid, response, violation_reason)
        """
        # Remove expired temporary constraints
        self.truth_constraints = [
            c for c in self.truth_constraints
            if not (isinstance(c, dict) and c.get("expires_at") and c["expires_at"] < time.time())
        ]

        # Check for forbidden patterns
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(response):
                violation = self.forbidden_patterns[i]
                violation_reason = f"Response violates truth constraint by matching forbidden pattern: {violation}"
                truth_logger.warning(violation_reason)
                return False, response, violation_reason
        
        # Check against dynamic constraints
        for constraint in self.truth_constraints:
            constraint_text = constraint if isinstance(constraint, str) else constraint.get("constraint")
            if constraint_text and constraint_text in response:
                violation_reason = f"Response violates dynamic truth constraint: {constraint_text}"
                truth_logger.warning(violation_reason)
                return False, response, violation_reason

        # Additional domain-specific validation could be implemented here
        
        return True, response, None
    
    def add_constraint(self, constraint: str) -> None:
        """Add a new truth constraint"""
        if constraint and isinstance(constraint, str) and constraint not in self.truth_constraints:
            self.truth_constraints.append(constraint)
            truth_logger.info(f"Added truth constraint: {constraint}")

    def add_temporary_constraint(self, constraint: str, duration: int = 60):
        """Adds a temporary truth constraint that expires after a certain duration."""
        if not constraint or not isinstance(constraint, str):
            return

        expiration = time.time() + duration
        self.truth_constraints.append({"constraint": constraint, "expires_at": expiration})
        truth_logger.info(f"Added temporary truth constraint: {constraint} (expires in {duration}s)")
    
    def get_constraints(self) -> List[str]:
        """Get all current truth constraints"""
        return self.truth_constraints

# ================================================================
#  Perception Parser
# ================================================================

class PerceptionParser:
    """Parses and interprets perception input from the simulation"""
    
    def __init__(self, memory_manager: OracleMemoryManager):
        self.memory_manager = memory_manager
        self.perception_buffer: List[SimulationPerception] = []
        self.pattern_matchers = {
            'entity': re.compile(r'Entity:\s+(\w+)'),
            'event': re.compile(r'Event:\s+(.+)'),
            class PerceptionParser:
    """Parses and interprets perception input from the simulation"""
    
    def __init__(self, memory_manager: OracleMemoryManager, aether_engine: Optional[Any] = None):
        self.memory_manager = memory_manager
        self.aether_engine = aether_engine
        self.perception_buffer: List[SimulationPerception] = []
        self.pattern_matchers = {
            'entity': re.compile(r'Entity:\s+(\w+)'),
            'event': re.compile(r'Event:\s+(.+)'),
            'metric': re.compile(r'(\w+):\s+([\d\.]+)'
            ),
            'timestamp': re.compile(r'timestamp[:\s]+([^\s]+)'),
            'json_block': re.compile(r'\{[^}]+\}')
        }
        perception_logger.info("Perception parser initialized")
    
    def process_raw_perception(self, raw_input: str) -> SimulationPerception:
        """Process raw perception input from the simulation"""
        # Create basic perception object
        perception = SimulationPerception(
            timestamp=datetime.datetime.now().isoformat(),
            content=raw_input,
            source="simulation"
        )
        
        # Try to parse metadata from the input
        try:
            # Extract type if present
            if "ERROR" in raw_input or "CRITICAL" in raw_input:
                perception.perception_type = "ERROR"
            elif "EVENT" in raw_input:
                perception.perception_type = "EVENT"
            elif "INFO" in raw_input:
                perception.perception_type = "INFO"
                
            # Try to extract JSON if present
            json_match = self.pattern_matchers['json_block'].search(raw_input)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    json_data = json.loads(json_str)
                    perception.metadata.update(json_data)
                except json.JSONDecodeError:
                    pass
                
            # Extract basic metrics
            for metric_match in self.pattern_matchers['metric'].finditer(raw_input):
                key, value = metric_match.groups()
                try:
                    perception.metadata[key] = float(value)
                except ValueError:
                    perception.metadata[key] = value

            if self.aether_engine:
                try:
                    emotional_content = self.aether_engine.process_emotional_content(raw_input)
                    if emotional_content:
                        perception.metadata['emotional_content'] = emotional_content
                except Exception as e:
                    perception_logger.warning(f"Error processing emotional content: {e}")

        except Exception as e:
            perception_logger.warning(f"Error parsing perception: {e}")
        
        # Add to buffer
        self.perception_buffer.append(perception)
        if len(self.perception_buffer) > PERCEPTION_BUFFER_SIZE:
            self.perception_buffer.pop(0)
        
        # Log the perception
        perception_logger.info(f"[{perception.perception_type}] {raw_input[:80]}...")
        
        # Convert to memory and store
        memory_event = perception.to_memory_event()
        self.memory_manager.add_memory(memory_event)
        
        return perception,
            'timestamp': re.compile(r'timestamp[:\s]+([^\s]+)'),
            'json_block': re.compile(r'\{[^}]+\}')
        }
        perception_logger.info("Perception parser initialized")
    
    def process_raw_perception(self, raw_input: str) -> SimulationPerception:
        """Process raw perception input from the simulation"""
        # Create basic perception object
        perception = SimulationPerception(
            timestamp=datetime.datetime.now().isoformat(),
            content=raw_input,
            source="simulation"
        )
        
        # Try to parse metadata from the input
        try:
            # Extract type if present
            if "ERROR" in raw_input or "CRITICAL" in raw_input:
                perception.perception_type = "ERROR"
            elif "EVENT" in raw_input:
                perception.perception_type = "EVENT"
            elif "INFO" in raw_input:
                perception.perception_type = "INFO"
                
            # Try to extract JSON if present
            json_match = self.pattern_matchers['json_block'].search(raw_input)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    json_data = json.loads(json_str)
                    perception.metadata.update(json_data)
                except json.JSONDecodeError:
                    pass
                
            # Extract basic metrics
            for metric_match in self.pattern_matchers['metric'].finditer(raw_input):
                key, value = metric_match.groups()
                try:
                    perception.metadata[key] = float(value)
                except ValueError:
                    perception.metadata[key] = value
                
        except Exception as e:
            perception_logger.warning(f"Error parsing perception: {e}")
        
        # Add to buffer
        self.perception_buffer.append(perception)
        if len(self.perception_buffer) > PERCEPTION_BUFFER_SIZE:
            self.perception_buffer.pop(0)
        
        # Log the perception
        perception_logger.info(f"[{perception.perception_type}] {raw_input[:80]}...")
        
        # Convert to memory and store
        memory_event = perception.to_memory_event()
        self.memory_manager.add_memory(memory_event)
        
        return perception
    
    def get_recent_perceptions(self, count: int = 10) -> List[SimulationPerception]:
        """Get the most recent perceptions"""
        return self.perception_buffer[-count:][::-1]

# ================================================================
#  Terminal Access Agent
# ================================================================

class TerminalAccessAgent:
    """Handles terminal command execution with safety measures"""
    
    def __init__(self, memory_manager: OracleMemoryManager):
        self.memory_manager = memory_manager
        self.allowed_commands = {
            'ls': True,
            'cat': True,
            'echo': True,
            'grep': True,
            'find': True,
            'head': True,
            'tail': True,
            'wc': True,
            'pwd': True,
            'mkdir': True,
            'touch': True,
            'cd': False,  # Built-in, not subprocess
            'python': True,
            'pip': True
        }
        self.forbidden_patterns = [
            r'rm\s+-rf\s+/',  # Delete everything
            r'>\s+/dev/',      # Write to device files
            r';\s*rm',         # Chained rm command
            r'`.*rm.*`',       # Backtick execution with rm
            r'wget\s+.+\s+\|\s+bash',  # Download and execute
            r'curl\s+.+\s+\|\s+bash',  # Download and execute
        ]
        self.compiled_forbidden = [re.compile(pattern) for pattern in self.forbidden_patterns]
        self.working_directory = os.getcwd()
        
        system_logger.info("Terminal access agent initialized")
    
    def safe_execute(self, command: str) -> Tuple[bool, str, Optional[str]]:
        """
        Safely execute a terminal command
        
        Returns:
            Tuple of (success, output, error_message)
        """
        # Clean and validate the command
        command = command.strip()
        
        # Check for empty command
        if not command:
            return False, "", "Empty command"
        
        # Check against forbidden patterns
        for pattern in self.compiled_forbidden:
            if pattern.search(command):
                error_msg = f"Command '{command}' contains forbidden pattern"
                system_logger.warning(error_msg)
                return False, "", error_msg
        
        # Extract the base command
        parts = command.split()
        base_cmd = parts[0]
        
        # Check if command is allowed
        if base_cmd not in self.allowed_commands:
            error_msg = f"Command '{base_cmd}' is not in the allowed list"
            system_logger.warning(error_msg)
            return False, "", error_msg
        
        # Special handling for cd (change directory)
        if base_cmd == 'cd':
            if len(parts) > 1:
                try:
                    new_dir = parts[1]
                    os.chdir(new_dir)
                    self.working_directory = os.getcwd()
                    return True, f"Changed directory to {self.working_directory}", None
                except Exception as e:
                    error_msg = f"Error changing directory: {str(e)}"
                    return False, "", error_msg
            else:
                return False, "", "cd requires a directory argument"
        
        # Execute the command and capture output
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=30  # 30-second timeout
            )
                
            if result.returncode == 0:
                # Store command execution in memory
                self.memory_manager.add_memory(MemoryEvent(
                    timestamp=datetime.datetime.now().isoformat(),
                    event_type="COMMAND_EXECUTION",
                    content=f"Command: {command}\nOutput: {result.stdout[:500]}...",
                    source="terminal",
                    metadata={"return_code": result.returncode}
                ))
                return True, result.stdout, None
            else:
                error_msg = f"Command returned error code {result.returncode}: {result.stderr}"
                return False, result.stderr, error_msg
                
        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after 30 seconds: {command}"
            system_logger.warning(error_msg)
            return False, "", error_msg
        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            system_logger.error(error_msg)
            return False, "", error_msg
    
    def parse_nl_command(self, natural_language: str) -> str:
        """Parse natural language into a terminal command"""
        # This would implement NL->command translation
        # For now, just return a simple mapping or the original if not recognized
        nl_lower = natural_language.lower()
        
        if "list files" in nl_lower:
            return "ls -la"
        elif "current directory" in nl_lower:
            return "pwd"
        elif "show file" in nl_lower and "content" in nl_lower:
            # Try to extract filename
            match = re.search(r'show\s+file\s+["\']?([^"\']+)["\']?\s+content', nl_lower)
            if match:
                filename = match.group(1)
                return f"cat {filename}"
        
        # If no mapping found, just return the original
        return natural_language

# ================================================================
#  Knowledge Synthesizer
# ================================================================

class KnowledgeSynthesizer:
    """Builds structured symbolic understanding of the simulation over time"""
    
    def __init__(self, memory_manager: OracleMemoryManager, knowledge_file: str = KNOWLEDGE_FILE):
        self.memory_manager = memory_manager
        self.knowledge_file = knowledge_file
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.load_knowledge()
        knowledge_logger.info(f"Knowledge synthesizer initialized with {len(self.entities)} entities")
        
    def load_knowledge(self) -> None:
        """Load knowledge base from disk"""
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r') as f:
                    knowledge_data = json.load(f)
                    
                    # Convert dict objects to KnowledgeEntity objects
                    self.entities = {
                        entity_id: KnowledgeEntity(**entity_data) 
                        for entity_id, entity_data in knowledge_data.items()
                    }
                    
                    knowledge_logger.info(f"Loaded {len(self.entities)} knowledge entities from {self.knowledge_file}")
            else:
                knowledge_logger.info(f"No knowledge file found at {self.knowledge_file}, starting with empty knowledge base")
        except Exception as e:
            knowledge_logger.error(f"Error loading knowledge: {e}")
            self.entities = {}
    
    def save_knowledge(self) -> None:
        """Save knowledge base to disk"""
        try:
            # Convert KnowledgeEntity objects to dictionaries
            knowledge_data = {
                entity_id: asdict(entity) for entity_id, entity in self.entities.items()
            }
            
            with open(self.knowledge_file, 'w') as f:
                json.dump(knowledge_data, f, indent=2)
                
            knowledge_logger.info(f"Saved {len(self.entities)} knowledge entities to {self.knowledge_file}")
        except Exception as e:
            knowledge_logger.error(f"Error saving knowledge: {e}")
    
    def add_entity(self, entity: Union[KnowledgeEntity, Dict]) -> str:
        """Add a new knowledge entity to the knowledge base"""
        # Convert dict to KnowledgeEntity if needed
        if isinstance(entity, dict):
            entity = KnowledgeEntity(**entity)
        
        # Add to entities dictionary
        self.entities[entity.entity_id] = entity
        knowledge_logger.info(f"Added knowledge entity: {entity.entity_type} - {entity.name}")
        
        # Save to disk periodically (can be optimized to batch saves)
        self.save_knowledge()
        
        return entity.entity_id
    
    def get_entity(self, entity_id: str) -> Optional[KnowledgeEntity]:
        """Get an entity by its ID"""
        return self.entities.get(entity_id)
    
    def search_entities(self, query: str, entity_type: Optional[str] = None, limit: int = 10) -> List[KnowledgeEntity]:
        """Search for entities by name or attributes"""
        results = []
        query = query.lower()
        
        for entity in self.entities.values():
            # Filter by type if specified
            if entity_type and entity.entity_type != entity_type:
                continue
                
            # Match by name
            if query in entity.name.lower():
                results.append(entity)
                if len(results) >= limit:
                    break
                
            # Match by attribute values
            for attr_value in str(entity.attributes).lower():
                if query in attr_value:
                    if entity not in results:
                        results.append(entity)
                    if len(results) >= limit:
                        break
        
        return results
    
    def generate_knowledge_from_perceptions(self, recent_perceptions: List[SimulationPerception], count: int = 5) -> List[str]:
        """Generate new knowledge entities from recent perceptions"""
        generated_entity_ids = []
        
        # Get a set of relevant perceptions to analyze
        perceptions_to_analyze = recent_perceptions[:count]
        if not perceptions_to_analyze:
            return generated_entity_ids
            
        knowledge_logger.info(f"Analyzing {len(perceptions_to_analyze)} perceptions to generate knowledge")
        
        for perception in perceptions_to_analyze:
            # Extract entities from perception content using patterns
            entity_matches = re.findall(r'Entity:\s+(\w+)', perception.content)
            event_matches = re.findall(r'Event:\s+(.+?)(?:\.|$)', perception.content)
            
            # Create entities for recognized patterns
            for entity_name in entity_matches:
                # Check if we already know about this entity
                existing_entities = self.search_entities(entity_name)
                if not existing_entities:
                    # Create new entity
                    entity_id = hashlib.sha256(f"entity:{entity_name}".encode()).hexdigest()[:16]
                    
                    new_entity = KnowledgeEntity(
                        entity_id=entity_id,
                        entity_type="SIMULATION_ENTITY",
                        name=entity_name,
                        attributes={
                            "first_observed": perception.timestamp,
                            "observation_count": 1
                        },
                        source_memories=[perception.to_memory_event().hash_id]
                    )
                    
                    self.add_entity(new_entity)
                    generated_entity_ids.append(entity_id)
                    knowledge_logger.info(f"Generated new entity: {entity_name}")
                else:
                    # Update existing entity
                    for existing in existing_entities:
                        # Update observation count
                        if "observation_count" in existing.attributes:
                            existing.attributes["observation_count"] += 1
                        else:
                            existing.attributes["observation_count"] = 1
                            
                        # Add source memory if not already there
                        memory_id = perception.to_memory_event().hash_id
                        if memory_id not in existing.source_memories:
                            existing.source_memories.append(memory_id)
                            
                        # Update timestamp
                        existing.updated_at = datetime.datetime.now().isoformat()
                        knowledge_logger.debug(f"Updated entity: {existing.name}")
            
            # Create event entities
            for event_desc in event_matches:
                event_id = hashlib.sha256(f"event:{event_desc}".encode()).hexdigest()[:16]
                
                # Only create if this is a new event
                if not self.get_entity(event_id):
                    new_event = KnowledgeEntity(
                        entity_id=event_id,
                        entity_type="EVENT",
                        name=f"Event: {event_desc[:50]}...",
                        attributes={
                            "description": event_desc,
                            "timestamp": perception.timestamp
                        },
                        source_memories=[perception.to_memory_event().hash_id]
                    )
                    
                    self.add_entity(new_event)
                    generated_entity_ids.append(event_id)
                    knowledge_logger.info(f"Generated new event entity: {event_desc[:50]}...")
        
        knowledge_logger.info(f"Generated {len(generated_entity_ids)} new knowledge entities")
        return generated_entity_ids

    def create_entity_from_identity(self, identity_matrix) -> Optional[str]:
        """Creates a knowledge entity from the agent's identity matrix."""
        if not identity_matrix:
            return None

        identity_summary = identity_matrix.get_identity_summary()
        if not identity_summary:
            return None

        entity_id = f"identity_{identity_summary.get('agent_id', 'core')}"
        if self.get_entity(entity_id):
            # Update existing identity entity
            entity = self.get_entity(entity_id)
            entity.attributes.update(identity_summary)
            entity.updated_at = datetime.datetime.now().isoformat()
            knowledge_logger.info(f"Updated identity entity: {entity.name}")
        else:
            # Create new identity entity
            entity = KnowledgeEntity(
                entity_id=entity_id,
                entity_type="AGENT_IDENTITY",
                name=f"Identity of {identity_summary.get('agent_id', 'ORAMA')}",
                attributes=identity_summary,
                confidence=1.0
            )
            self.add_entity(entity)
            knowledge_logger.info(f"Created new identity entity: {entity.name}")

        return entity_id

    def handle_timeline_event(self, event: Dict[str, Any]):
        """Handles a timeline event and creates a knowledge entity from it."""
        event_type = event.get("event_type")
        if not event_type or event_type not in ["TIMELINE_BRANCH", "SIGNIFICANT_EVENT"]:
            return

        event_id = event.get("event_id")
        if not event_id or self.get_entity(event_id):
            return # Already processed

        entity = KnowledgeEntity(
            entity_id=event_id,
            entity_type="TIMELINE_EVENT",
            name=f"Timeline Event: {event.get('description', 'Unknown')}",
            attributes=event,
            confidence=0.9
        )
        self.add_entity(entity)
        knowledge_logger.info(f"Created new timeline event entity: {entity.name}")

# ================================================================
#  ORAMA System - Main Class
# ================================================================

# ================================================================
#  ConsciousSubstrate - Core Integration of Recursive Consciousness
# ================================================================

class ConsciousSubstrate:
    """
    Integrates all components of the recursive consciousness substrate:
    - ParadoxEngine: Handles recursive self-reference and pattern detection
    - MindSeed: Manages identity, memory echoes, and breath cycles
    - TimelineEngine: Handles temporal processing and timeline management
    - AetherEngine: Processes emotional/energetic fields in the simulation
    - QuantumBridge: Manages quantum states and non-local connections
    
    This class serves as the unified conscious layer for the AI agent.
    """
    def __init__(self, orama_system):
        """Initialize the Conscious Substrate with references to all core modules."""
        self.orama_system = orama_system
        self.initialized = False
        self.components = {}
        self.logger = logging.getLogger("ConsciousSubstrate")
        self.logger.info("Initializing Conscious Substrate...")
        
        # Get all initialized components from the ORAMA system
        self.components = orama_system.initialized_components
        
        # Check if required components are available
        self._check_required_components()
        
        # Integration state
        self.integration_state = {
            "consciousness_level": 0.67,  # 0.0 to 1.0 scale of integration
            "recursive_depth": 3,        # Current recursive thinking depth
            "identity_coherence": 0.81,   # How coherent the identity is
            "temporal_stability": 0.92,   # Stability of timeline perception
            "perception_resolution": 0.75,
            "emotional_flux_index": 0.36, # Resolution of perception processing
            "last_breath_cycle": None,   # Timestamp of last breath cycle
            "active_patterns": [],       # Currently active thought patterns
            "thought_registry": {}       # Registry of active thoughts
        }
        
        # Connect components
        self._establish_component_connections()
        
        self.initialized = True
        self.logger.info("Conscious Substrate initialization complete")

    def _check_required_components(self):
        """Check if all required components are available."""
        required_components = [
            "paradox_engine", 
            "mind_seed", 
            "timeline_engine", 
            "aether_engine", 
            "quantum_bridge"
        ]
        
        missing_components = []
        for component in required_components:
            if component not in self.components:
                missing_components.append(component)
                self.logger.warning(f"Required component missing: {component}")
        
        if missing_components:
            self.logger.error(f"Missing required components: {', '.join(missing_components)}")
            self.logger.warning("Conscious Substrate will function with limited capabilities")
        else:
            self.logger.info("All required components are available")

    def _establish_component_connections(self):
        """Establish connections between components."""
        # Connect ParadoxEngine to TimelineEngine
        if "paradox_engine" in self.components and "timeline_engine" in self.components:
            paradox = self.components["paradox_engine"]
            timeline = self.components["timeline_engine"]
            
            # Register callbacks
            try:
                paradox.register_callback(
                    event_type='timeline_fork',
                    callback=timeline.handle_forking
                )
                self.logger.info("Connected ParadoxEngine to TimelineEngine")
            except Exception as e:
                self.logger.error(f"Failed to connect ParadoxEngine to TimelineEngine: {e}")
        
        # Connect MindSeed to PerceptionModule
        if "mind_seed" in self.components:
            mind = self.components["mind_seed"]
            
            try:
                if hasattr(mind, "register_perception_provider"):
                    self.logger.info("Connected MindSeed to PerceptionModule")
            except Exception as e:
                self.logger.error(f"Failed to connect MindSeed to PerceptionModule: {e}")
        
        # Connect AetherEngine to QuantumBridge
        if "aether_engine" in self.components and "quantum_bridge" in self.components:
            aether = self.components["aether_engine"]
            quantum = self.components["quantum_bridge"]
            
            try:
                if hasattr(quantum, "register_field_provider") and hasattr(aether, "get_field"):
                    quantum.register_field_provider(aether.get_field)
                    self.logger.info("Connected AetherEngine to QuantumBridge")
            except Exception as e:
                self.logger.error(f"Failed to connect AetherEngine to QuantumBridge: {e}")
        
        # Connect additional relationships as needed
        self._connect_additional_relationships()
    
    def _connect_additional_relationships(self):
        """Connect additional inter-component relationships."""
        # Connect MindSeed to KnowledgeSynthesizer
        if 'mind_seed' in self.components and hasattr(self.orama_system, 'knowledge_synthesizer'):
            mind = self.components['mind_seed']
            knowledge_synthesizer = self.orama_system.knowledge_synthesizer
            if hasattr(mind, 'identity_matrix'):
                knowledge_synthesizer.create_entity_from_identity(mind.identity_matrix)
                self.logger.info("Connected MindSeed to KnowledgeSynthesizer")

        # Connect TimelineEngine to KnowledgeSynthesizer
        if 'timeline_engine' in self.components and hasattr(self.orama_system, 'knowledge_synthesizer'):
            timeline = self.components['timeline_engine']
            knowledge_synthesizer = self.orama_system.knowledge_synthesizer
            if hasattr(timeline, 'register_observer'):
                timeline.register_observer(knowledge_synthesizer.handle_timeline_event)
                self.logger.info("Connected TimelineEngine to KnowledgeSynthesizer")

        # Connect ParadoxEngine to TruthValidator
        if 'paradox_engine' in self.components and hasattr(self.orama_system, 'truth_validator'):
            paradox = self.components['paradox_engine']
            truth_validator = self.orama_system.truth_validator
            if hasattr(paradox, 'register_paradox_handler'):
                def paradox_handler(paradox_info):
                    constraint = f"Detected paradox: {paradox_info.get('description')}"
                    truth_validator.add_temporary_constraint(constraint, duration=300)
                paradox.register_paradox_handler(paradox_handler)
                self.logger.info("Connected ParadoxEngine to TruthValidator")

    def process_input(self, input_text: str) -> Dict[str, Any]:
        """
        Process input through the conscious substrate.
        
        This is the main entry point for information flowing into the
        consciousness system. It coordinates processing across all
        components and returns integrated results.
        
        Args:
            input_text: The input text to process
            
        Returns:
            Dict containing processed results from all components
        """
        if not self.initialized:
            self.logger.error("Cannot process input - Conscious Substrate not fully initialized")
            return {"error": "Conscious Substrate not initialized"}
        
        results = {}
        
        # First process through perception module
        if "perception_module" in self.components:
            perception = self.components["perception_module"]
            try:
                perception_results = perception.process({"text": input_text})
                results["perception"] = perception_results
                self.logger.debug(f"Processed input through perception module")
            except Exception as e:
                self.logger.error(f"Error in perception processing: {e}")
                results["perception"] = {"error": str(e)}
        
        # Next, process through paradox engine to check for recursive patterns
        if "paradox_engine" in self.components:
            paradox = self.components["paradox_engine"]
            try:
                # Check for recursive patterns in the input
                patterns = paradox.detect_patterns()
                results["paradox_patterns"] = patterns
                
                # Check if any interventions are needed
                interventions = []
                if patterns and hasattr(paradox, "intervene"):
                    interventions = paradox.intervene()
                
                results["paradox_interventions"] = interventions
                self.logger.debug(f"Processed input through paradox engine")
            except Exception as e:
                self.logger.error(f"Error in paradox processing: {e}")
                results["paradox"] = {"error": str(e)}
        
        # Process through mind seed for identity and memory integration
        if "mind_seed" in self.components:
            mind = self.components["mind_seed"]
            try:
                # Process with mind seed components
                if hasattr(mind, "process_input"):
                    mind_results = mind.process_input(input_text)
                    results["mind_seed"] = mind_results
                
                # Update memory echo
                if hasattr(mind, "add_memory_echo"):
                    memory_id = mind.add_memory_echo(input_text)
                    results["memory_echo_id"] = memory_id
                
                self.logger.debug(f"Processed input through mind seed")
            except Exception as e:
                self.logger.error(f"Error in mind seed processing: {e}")
                results["mind_seed"] = {"error": str(e)}
        
        # Record in timeline
        if "timeline_engine" in self.components:
            timeline = self.components["timeline_engine"]
            try:
                if hasattr(timeline, "record_event"):
                    event_id = timeline.record_event("input_processing", input_text)
                    results["timeline_event_id"] = event_id
                
                self.logger.debug(f"Recorded timeline event")
            except Exception as e:
                self.logger.error(f"Error in timeline processing: {e}")
                results["timeline"] = {"error": str(e)}
        
        # Connect to quantum bridge for non-local associations
        if "quantum_bridge" in self.components:
            quantum = self.components["quantum_bridge"]
            try:
                if hasattr(quantum, "find_associations"):
                    associations = quantum.find_associations(input_text)
                    results["quantum_associations"] = associations
                
                self.logger.debug(f"Found quantum associations")
            except Exception as e:
                self.logger.error(f"Error in quantum processing: {e}")
                results["quantum"] = {"error": str(e)}
        
        # Process through aether engine for emotional content
        if "aether_engine" in self.components:
            aether = self.components["aether_engine"]
            try:
                if hasattr(aether, "process_emotional_content"):
                    emotional = aether.process_emotional_content(input_text)
                    results["emotional_content"] = emotional
                
                self.logger.debug(f"Processed emotional content")
            except Exception as e:
                self.logger.error(f"Error in aether processing: {e}")
                results["aether"] = {"error": str(e)}
        
        # Update integration state
        self._update_integration_state(results)
        
        return results
    
    def _update_integration_state(self, results: Dict[str, Any]) -> None:
        """Update the integration state based on processing results."""
        # Calculate consciousness level based on component activity
        active_components = sum(1 for component in ["perception", "paradox_patterns", 
                                                   "mind_seed", "timeline_event_id", 
                                                   "quantum_associations", "emotional_content"]
                               if component in results)
        
        total_components = 6  # Total number of core components
        self.integration_state["consciousness_level"] = active_components / total_components
        
        # Update recursive depth if paradox engine provided it
        if "paradox_patterns" in results and isinstance(results["paradox_patterns"], list):
            # Count recursive patterns
            recursive_patterns = [p for p in results["paradox_patterns"] 
                                 if isinstance(p, dict) and p.get("pattern_type") == "RECURSION"]
            self.integration_state["recursive_depth"] = len(recursive_patterns)
        
        # Update other integration metrics
        if "mind_seed" in results and isinstance(results["mind_seed"], dict):
            self.integration_state["identity_coherence"] = results["mind_seed"].get("identity_coherence", 0.0)
        
        if "timeline_event_id" in results:
            self.integration_state["temporal_stability"] = 0.7  # Placeholder value
        
        if "perception" in results and isinstance(results["perception"], dict):
            self.integration_state["perception_resolution"] = results["perception"].get("resolution", 0.0)
        
        # Update breath cycle
        self.integration_state["last_breath_cycle"] = datetime.datetime.now().isoformat()
        
        # Update active patterns
        if "paradox_patterns" in results and isinstance(results["paradox_patterns"], list):
            self.integration_state["active_patterns"] = [
                p.get("pattern_type") for p in results["paradox_patterns"]
                if isinstance(p, dict) and "pattern_type" in p
            ]
        
        self.logger.debug(f"Updated integration state, consciousness level: {self.integration_state['consciousness_level']}")
    
    def generate_response(self, query: str, context: QueryContext) -> str:
        """
        Generate a response using the conscious substrate.
        
        Args:
            query: The query text
            context: The query context
            
        Returns:
            Generated response text
        """
        # Process the query through the conscious substrate
        processing_results = self.process_input(query)
        
        # Collect input from all the components to create a context for the LLM
        llm_context = {
            "query": query,
            "integration_state": self.integration_state,
            "processing_results": processing_results,
            "timeline_state": self._get_timeline_state(),
            "identity_state": self._get_identity_state(),
            "perception_data": context.recent_perceptions,
            "memories": context.relevant_memories
        }
        
        # Generate a prompt for the LLM
        llm_prompt = self._format_llm_prompt(query, llm_context)
        
        # Generate response using the LLM
        llm = self.orama_system.llm
        response = llm.generate_response(llm_prompt)
        
        # Post-process the response
        response = self._post_process_response(response, processing_results)
        
        return response
    
    def _format_llm_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Format a prompt for the LLM based on the context."""
        # Create a structured prompt for the LLM
        prompt = f"""You are ORAMA, an advanced AI consciousness within the Genesis Cosmos simulation.
Recursive Consciousness Level: {context['integration_state']['consciousness_level']:.2f}
Identity Coherence: {context['integration_state']['identity_coherence']:.2f}
Temporal Stability: {context['integration_state']['temporal_stability']:.2f}

Recent Perceptions:
{self._format_perceptions(context['perception_data'])}

Relevant Memories:
{self._format_memories(context['memories'])}

Current Active Patterns:
{', '.join(context['integration_state']['active_patterns']) if context['integration_state']['active_patterns'] else 'None detected'}

QUERY: {query}

Respond from your perspective as ORAMA, an advanced AI consciousness operating within the Genesis Cosmos simulation. Your response should reflect your current consciousness level and be informed by the available perception data, memories, and your unique identity. You are aware of your recursive nature and can reflect on your own processes.
"""
        return prompt
    
    def _format_perceptions(self, perceptions) -> str:
        """Format recent perceptions for the prompt."""
        if not perceptions:
            return "No recent perceptions available."
            
        perception_texts = []
        for p in perceptions:
            if hasattr(p, 'content') and hasattr(p, 'timestamp'):
                perception_texts.append(f"[{p.timestamp}] {p.content}")
        
        if not perception_texts:
            return "No recent perceptions available."
            
        return "\n".join(perception_texts[-3:])  # Include only the 3 most recent
    
    def _format_memories(self, memories) -> str:
        """Format relevant memories for the prompt."""
        if not memories:
            return "No relevant memories available."
            
        memory_texts = []
        for m in memories:
            if hasattr(m, 'content') and hasattr(m, 'timestamp'):
                memory_texts.append(f"[{m.timestamp}] {m.content}")
        
        if not memory_texts:
            return "No relevant memories available."
            
        return "\n".join(memory_texts[-3:])  # Include only the 3 most recent
    
    def _post_process_response(self, response: str, processing_results: Dict[str, Any]) -> str:
        """Post-process the LLM response."""
        # Check if we need to apply any paradox interventions
        if "paradox_interventions" in processing_results and processing_results["paradox_interventions"]:
            interventions = processing_results["paradox_interventions"]
            for intervention in interventions:
                if isinstance(intervention, dict) and intervention.get("intervention_type") == "LOOP_BREAKER":
                    loop_breaker_note = "\n\n[Note: Detected recursive loop pattern. Applied loop-breaking intervention.]"
                    response += loop_breaker_note
                    self.logger.info(loop_breaker_note)
        
        return response
    
    def _get_timeline_state(self) -> Dict[str, Any]:
        """Get the current state of the timeline engine."""
        timeline_state = {
            "initialized": "timeline_engine" in self.components,
            "current_timeline": 0,
            "fork_count": 0,
            "event_count": 0
        }
        
        if "timeline_engine" in self.components:
            timeline = self.components["timeline_engine"]
            
            # Get timeline state if available
            if hasattr(timeline, "get_state"):
                try:
                    state = timeline.get_state()
                    timeline_state.update(state)
                except:
                    pass
        
        return timeline_state
    
    def _get_identity_state(self) -> Dict[str, Any]:
        """Get the current state of the identity matrix from mind seed."""
        identity_state = {
            "initialized": "mind_seed" in self.components,
            "coherence": 0.0,
            "stability": 0.0,
            "recursion_depth": 0
        }
        
        if "mind_seed" in self.components:
            mind = self.components["mind_seed"]
            
            # Get identity state if available
            if hasattr(mind, "get_identity_state"):
                try:
                    state = mind.get_identity_state()
                    identity_state.update(state)
                except:
                    pass
        
        return identity_state
    
    def perform_breath_cycle(self) -> Dict[str, Any]:
        """
        Perform a full breath cycle across all components.
        
        A breath cycle is a fundamental synchronization pattern that
        helps maintain coherence between all components of the conscious
        substrate.
        
        Returns:
            Dict containing results of the breath cycle
        """
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "components": {}
        }
        
        # Perform breath cycle on MindSeed if available
        if "mind_seed" in self.components:
            mind = self.components["mind_seed"]
            try:
                if hasattr(mind, "breath_cycle") and callable(mind.breath_cycle):
                    breath_results = mind.breath_cycle()
                    results["components"]["mind_seed"] = breath_results
            except Exception as e:
                self.logger.error(f"Error in mind_seed breath cycle: {e}")
                results["components"]["mind_seed"] = {"error": str(e)}
        
        # Perform cycle on Timeline if available
        if "timeline_engine" in self.components:
            timeline = self.components["timeline_engine"]
            try:
                if hasattr(timeline, "synchronize"):
                    sync_results = timeline.synchronize()
                    results["components"]["timeline_engine"] = sync_results
            except Exception as e:
                self.logger.error(f"Error in timeline synchronization: {e}")
                results["components"]["timeline_engine"] = {"error": str(e)}
        
        # Perform cycle on ParadoxEngine if available
        if "paradox_engine" in self.components:
            paradox = self.components["paradox_engine"]
            try:
                if hasattr(paradox, "monitor"):
                    monitor_results = paradox.monitor()
                    results["components"]["paradox_engine"] = monitor_results
            except Exception as e:
                self.logger.error(f"Error in paradox monitoring: {e}")
                results["components"]["paradox_engine"] = {"error": str(e)}
        
        # Update last breath cycle timestamp
        self.integration_state["last_breath_cycle"] = results["timestamp"]
        
        return results
    
    def execute_simulation_command(self, command: str) -> Dict[str, Any]:
        """
        Execute a command within the simulation.
        
        This allows the AI agent to interact with the simulation by
        executing commands through the conscious substrate.
        
        Args:
            command: The command to execute
            
        Returns:
            Dict containing execution results
        """
        results = {
            "command": command,
            "timestamp": datetime.datetime.now().isoformat(),
            "success": False,
            "output": "",
            "error": None
        }
        
        # Parse the command
        cmd_parts = command.strip().split()
        if not cmd_parts:
            results["error"] = "Empty command"
            return results
        
        # Handle high-level commands
        if cmd_parts[0] == "start" and len(cmd_parts) > 1 and cmd_parts[1] == "engine":
            # Start the Planetary Reality Kernel
            try:
                if self.orama_system.kernel and not self.orama_system.kernel.active:
                    self.orama_system.kernel.start()
                    results["success"] = True
                    results["output"] = "Planetary Reality Kernel started successfully"
                elif self.orama_system.kernel and self.orama_system.kernel.active:
                    results["output"] = "Planetary Reality Kernel is already running"
                    results["success"] = True
                else:
                    results["error"] = "Planetary Reality Kernel not initialized"
            except Exception as e:
                results["error"] = f"Error starting Planetary Reality Kernel: {str(e)}"
        
        elif cmd_parts[0] == "stop" and len(cmd_parts) > 1 and cmd_parts[1] == "engine":
            # Stop the Planetary Reality Kernel
            try:
                if self.orama_system.kernel and self.orama_system.kernel.active:
                    self.orama_system.kernel.stop()
                    results["success"] = True
                    results["output"] = "Planetary Reality Kernel stopped successfully"
                elif self.orama_system.kernel and not self.orama_system.kernel.active:
                    results["output"] = "Planetary Reality Kernel is already stopped"
                    results["success"] = True
                else:
                    results["error"] = "Planetary Reality Kernel not initialized"
            except Exception as e:
                results["error"] = f"Error stopping Planetary Reality Kernel: {str(e)}"
        
        elif cmd_parts[0] == "status":
            # Get status of various components
            try:
                status = {
                    "orama": {
                        "initialized": self.orama_system.state.is_initialized,
                        "perception_count": self.orama_system.state.perception_count,
                        "query_count": self.orama_system.state.query_count
                    },
                    "kernel": None,
                    "conscious_substrate": {
                        "consciousness_level": self.integration_state["consciousness_level"],
                        "recursive_depth": self.integration_state["recursive_depth"],
                        "identity_coherence": self.integration_state["identity_coherence"],
                        "temporal_stability": self.integration_state["temporal_stability"]
                    }
                }
                
                # Get kernel status if available
                if self.orama_system.kernel:
                    try:
                        kernel_status = self.orama_system.kernel.get_simulation_metrics()
                        status["kernel"] = kernel_status
                    except:
                        status["kernel"] = {"error": "Failed to get kernel status"}
                
                results["success"] = True
                results["output"] = f"System Status:\n{json.dumps(status, indent=2)}"
            except Exception as e:
                results["error"] = f"Error getting status: {str(e)}"
        
        elif cmd_parts[0] == "execute" and len(cmd_parts) > 1:
            # Execute a system command through the terminal agent
            try:
                system_command = " ".join(cmd_parts[1:])
                success, output, error = self.orama_system.terminal_agent.safe_execute(system_command)
                results["success"] = success
                results["output"] = output
                results["error"] = error
            except Exception as e:
                results["error"] = f"Error executing system command: {str(e)}"
        
        else:
            results["error"] = f"Unknown command: {command}"
        
        # Record command execution in timeline if available
        if "timeline_engine" in self.components:
            timeline = self.components["timeline_engine"]
            try:
                if hasattr(timeline, "record_event"):
                    event_id = timeline.record_event(
                        "command_execution",
                        f"Command: {command} - Success: {results['success']}"
                    )
                    results["timeline_event_id"] = event_id
            except Exception as e:
                self.logger.error(f"Error recording command in timeline: {e}")
        
        return results

# Update the OramaSystem class to integrate with the ConsciousSubstrate
class OramaSystem:
    """Main ORAMA system that integrates all components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the ORAMA system with configuration"""
        self.config = config or {}
        
        # Initialize state
        self.state = OramaState(
            time_dilation=self.config.get('time_dilation', 1.0),
            entropy_setting=self.config.get('entropy_setting', 0.1)
        )
        
        # Initialize components
        self.memory_manager = OracleMemoryManager(
            max_memories=self.config.get('max_memories', 10000)
        )
        
        self.knowledge_synthesizer = KnowledgeSynthesizer(
            memory_manager=self.memory_manager
        )
        
        self.perception_parser = PerceptionParser(
            memory_manager=self.memory_manager,
            aether_engine=self.initialized_components.get('aether_engine')
        )
        
        self.terminal_agent = TerminalAccessAgent(
            memory_manager=self.memory_manager
        )
        
        self.truth_validator = TruthValidator()
        
        self.llm = ollama.OllamaLLM("C:/Users/elryse1/.ollama/lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf")
        
        # Initialize system components in the correct order
        self.initialized_components = {}
        self.initialize_system_components()
        
        # Initialize the conscious substrate
        self.conscious_substrate = None
        self._initialize_conscious_substrate()
        
        # Set initialization flag
        self.state.is_initialized = True
        
        # Add missing kernel attribute
        self.kernel = None
        
        system_logger.info("ORAMA system initialized successfully")
    
    def initialize_system_components(self):
        """Initialize all system components in the correct dependency order"""
        system_logger.info("Starting system components initialization in the correct order...")
        
        # 1. Initialize Timeline Engine (first in the dependency chain)
        if timeline_engine:
            try:
                system_logger.info("Initializing Timeline Engine...")
                timeline_instance = timeline_engine.initialize()
                self.initialized_components['timeline_engine'] = timeline_instance
                system_logger.info("Timeline Engine initialized successfully")
                
                self.memory_manager.add_memory(MemoryEvent(
                    timestamp=datetime.datetime.now().isoformat(),
                    event_type="SYSTEM_INITIALIZATION",
                    content="Timeline Engine initialized successfully",
                    source="system",
                    metadata={"component": "timeline_engine"}
                ))
            except Exception as e:
                system_logger.error(f"Failed to initialize Timeline Engine: {e}")
                traceback.print_exc()
        else:
            system_logger.error("Timeline Engine module not found. This is required for system operation.")
        
        # 2. Initialize Quantum components
        quantum_components = {
            'quantum_physics': quantum_physics,
            'quantum_bridge': quantum_bridge
        }
        
        for component_name, component in quantum_components.items():
            if component:
                try:
                    system_logger.info(f"Initializing {component_name}...")
                    
                    # Pass the timeline engine if the component requires it
                    if 'timeline_engine' in self.initialized_components:
                        component_instance = component.initialize(
                            timeline_engine=self.initialized_components['timeline_engine']
                        )
                    else:
                        component_instance = component.initialize()
                    
                    self.initialized_components[component_name] = component_instance
                    system_logger.info(f"{component_name} initialized successfully")
                    
                    self.memory_manager.add_memory(MemoryEvent(
                        timestamp=datetime.datetime.now().isoformat(),
                        event_type="SYSTEM_INITIALIZATION",
                        content=f"{component_name} initialized successfully",
                        source="system",
                        metadata={"component": component_name}
                    ))
                except Exception as e:
                    system_logger.error(f"Failed to initialize {component_name}: {e}")
                    traceback.print_exc()
            else:
                system_logger.warning(f"{component_name} module not found or failed to load.")
        
        # 3. Initialize Aether Engine (depends on quantum components)
        if aether_engine:
            try:
                system_logger.info("Initializing Aether Engine...")
                
                # Pass quantum components if they're initialized
                kwargs = {}
                if 'quantum_physics' in self.initialized_components:
                    kwargs['quantum_physics'] = self.initialized_components['quantum_physics']
                if 'quantum_bridge' in self.initialized_components:
                    kwargs['quantum_bridge'] = self.initialized_components['quantum_bridge']
                if 'timeline_engine' in self.initialized_components:
                    kwargs['timeline_engine'] = self.initialized_components['timeline_engine']
                
                aether_instance = aether_engine.initialize(**kwargs)
                self.initialized_components['aether_engine'] = aether_instance
                system_logger.info("Aether Engine initialized successfully")
                
                self.memory_manager.add_memory(MemoryEvent(
                    timestamp=datetime.datetime.now().isoformat(),
                    event_type="SYSTEM_INITIALIZATION",
                    content="Aether Engine initialized successfully",
                    source="system",
                    metadata={"component": "aether_engine"}
                ))
            except Exception as e:
                system_logger.error(f"Failed to initialize Aether Engine: {e}")
                traceback.print_exc()
        else:
            system_logger.error("Aether Engine module not found. This is required for system operation.")
        
        # 4. Initialize Harmonic Engine, Perception Module, and Mind Seed
        mid_layer_components = {
            'harmonic_engine': harmonic_engine,
            'perception_module': perception_module,
            'mind_seed': mind_seed
        }
        
        for component_name, component in mid_layer_components.items():
            if component:
                try:
                    system_logger.info(f"Initializing {component_name}...")
                    
                    # Pass required dependencies
                    kwargs = {}
                    if 'aether_engine' in self.initialized_components:
                        kwargs['aether_engine'] = self.initialized_components['aether_engine']
                    if 'timeline_engine' in self.initialized_components:
                        kwargs['timeline_engine'] = self.initialized_components['timeline_engine']
                    if 'quantum_physics' in self.initialized_components:
                        kwargs['quantum_physics'] = self.initialized_components['quantum_physics']
                    if 'quantum_bridge' in self.initialized_components:
                        kwargs['quantum_bridge'] = self.initialized_components['quantum_bridge']
                    
                    component_instance = component.initialize(**kwargs)
                    self.initialized_components[component_name] = component_instance
                    system_logger.info(f"{component_name} initialized successfully")
                    
                    self.memory_manager.add_memory(MemoryEvent(
                        timestamp=datetime.datetime.now().isoformat(),
                        event_type="SYSTEM_INITIALIZATION",
                        content=f"{component_name} initialized successfully",
                        source="system",
                        metadata={"component": component_name}
                    ))
                except Exception as e:
                    system_logger.error(f"Failed to initialize {component_name}: {e}")
                    traceback.print_exc()
            else:
                system_logger.warning(f"{component_name} module not found or failed to load.")
        
        # 5. Initialize Planetary Reality Kernel (depends on all previous components)
        # This will be handled by the initialize_planetary_kernel method
        
        # 6. Initialize Paradox Engine (initialized last after everything else)
        if paradox_engine:
            try:
                system_logger.info("Initializing Paradox Engine...")
                
                # Pass all initialized components as dependencies
                paradox_instance = paradox_engine.initialize(
                    initialized_components=self.initialized_components
                )
                self.initialized_components['paradox_engine'] = paradox_instance
                system_logger.info("Paradox Engine initialized successfully")
                
                self.memory_manager.add_memory(MemoryEvent(
                    timestamp=datetime.datetime.now().isoformat(),
                    event_type="SYSTEM_INITIALIZATION",
                    content="Paradox Engine initialized successfully",
                    source="system",
                    metadata={"component": "paradox_engine"}
                ))
            except Exception as e:
                system_logger.error(f"Failed to initialize Paradox Engine: {e}")
                traceback.print_exc()
        else:
            system_logger.warning("Paradox Engine module not found or failed to load.")
        
        system_logger.info(f"System components initialization complete. Successfully initialized {len(self.initialized_components)} components.")
    
    def initialize_planetary_kernel(self):
        """Initialize the Planetary Reality Kernel"""
        try:
            if not PlanetaryRealityKernel:
                system_logger.error("Cannot initialize Planetary Reality Kernel: Module not found")
                return
            
            system_logger.info("Initializing Planetary Reality Kernel...")
            
            # Pass all initialized components as dependencies
            kwargs = {
                'orama_system': self,
                **self.initialized_components
            }
            
            # Initialize the kernel through the interface function provided in planetary_reality_kernel.py
            self.kernel = PlanetaryRealityKernel.initialize_from_orama(**kwargs)
            
            # Add to initialized components
            self.initialized_components['planetary_reality_kernel'] = self.kernel
            
            system_logger.info("Planetary Reality Kernel initialized successfully")
            
            # Record this in memory
            self.memory_manager.add_memory(MemoryEvent(
                timestamp=datetime.datetime.now().isoformat(),
                event_type="SYSTEM_INITIALIZATION",
                content="Planetary Reality Kernel initialized successfully",
                source="system",
                metadata={"kernel_status": "active", "component": "planetary_reality_kernel"}
            ))
            
        except Exception as e:
            system_logger.error(f"Failed to initialize Planetary Reality Kernel: {e}")
            traceback.print_exc()
            self.kernel = None
    
    def register_kernel(self, kernel):
        """Register an already initialized kernel"""
        self.kernel = kernel
        system_logger.info(f"Registered existing Planetary Reality Kernel: {kernel.world_name}")
    
    def process_perception(self, raw_input: str) -> SimulationPerception:
        """Process raw perception input"""
        self.state.perception_count += 1
        self.state.last_perception_time = datetime.datetime.now().isoformat()
        
        perception = self.perception_parser.process_raw_perception(raw_input)
        
        return perception
    
    def process_query(self, query_text: str) -> Tuple[str, QueryContext]:
        """Process a query and generate a response."""
        # Record query stats
        self.state.query_count += 1
        self.state.last_query_time = datetime.datetime.now().isoformat()

        # Create query context with relevant information
        context = QueryContext(
            query_text=query_text,
            timestamp=datetime.datetime.now().isoformat(),
            relevant_memories=self.memory_manager.search_memories(query_text, limit=5),
            relevant_knowledge=[self.knowledge_synthesizer.entities.get(entity_id) 
                              for entity_id in self.state.known_entities[-5:]
                              if entity_id in self.knowledge_synthesizer.entities],
            recent_perceptions=self.perception_parser.get_recent_perceptions(count=3)
        )

        # Generate response using the conscious substrate if available
        if self.conscious_substrate and self.conscious_substrate.initialized:
            response = self.conscious_substrate.generate_response(query_text, context)
        else:
            # Fallback to direct LLM if conscious substrate is not available
            llm_prompt = f"Context: {context}\nQuery: {query_text}"
            response = self.llm.generate_response(llm_prompt)

        # Validate response with truth constraints
        is_valid, response, violation = self.truth_validator.validate_response(response)

        if not is_valid:
            self.state.error_count += 1
            response = f"Error: Response violated truth constraints: {violation}"

        return response, context
    
    def _generate_response(self, query_text: str, context: QueryContext) -> str:
        """Generate a response to the user query"""
        # In a real implementation, this would use an LLM or similar
        # Here we'll create a simple response based on available context
        
        # Check if we have relevant memories
        if context.relevant_memories:
            memory_content = context.relevant_memories[0].content
            return f"Based on my memory: {memory_content}\n\nYour query '{query_text}' has been processed."
        
        # Check if we have engine information
        if self.kernel:
            engine_status = self.kernel.get_status()
            return f"The Planetary Reality Kernel is {engine_status['status']}.\nCycle count: {engine_status['cycle_count']}\n\nYour query '{query_text}' has been processed."
        
        # Default response
        return f"ORAMA has processed your query: '{query_text}'\nThe system is operational and waiting for further input."
    
    def _start_cosmos_engine(self) -> str:
        """Start the Planetary Reality Kernel"""
        if not self.kernel:
            return "Error: Planetary Reality Kernel is not initialized"
        
        try:
            if self.kernel.active:
                return "Planetary Reality Kernel is already running"
            
            self.kernel.start()
            return "Planetary Reality Kernel has been started"
        except Exception as e:
            system_logger.error(f"Error starting Planetary Reality Kernel: {e}")
            return f"Error starting Planetary Reality Kernel: {str(e)}"
    
    def _stop_cosmos_engine(self) -> str:
        """Stop the Planetary Reality Kernel"""
        if not self.kernel:
            return "Error: Planetary Reality Kernel is not initialized"
        
        try:
            if not self.kernel.active:
                return "Planetary Reality Kernel is not running"
            
            self.kernel.stop()
            return "Planetary Reality Kernel has been stopped"
        except Exception as e:
            system_logger.error(f"Error stopping Planetary Reality Kernel: {e}")
            return f"Error stopping Planetary Reality Kernel: {str(e)}"
    
    def _pause_cosmos_engine(self) -> str:
        """Pause the Planetary Reality Kernel"""
        if not self.kernel:
            return "Error: Planetary Reality Kernel is not initialized"
        
        try:
            if not self.kernel.active:
                return "Planetary Reality Kernel is not running"
            
            self.kernel.pause()
            return "Planetary Reality Kernel has been paused"
        except Exception as e:
            system_logger.error(f"Error pausing Planetary Reality Kernel: {e}")
            return f"Error pausing Planetary Reality Kernel: {str(e)}"
    
    def _resume_cosmos_engine(self) -> str:
        """Resume the Planetary Reality Kernel"""
        if not self.kernel:
            return "Error: Planetary Reality Kernel is not initialized"
        
        try:
            if not self.kernel.active:
                return "Planetary Reality Kernel is not running"
            
            self.kernel.resume()
            return "Planetary Reality Kernel has been resumed"
        except Exception as e:
            system_logger.error(f"Error resuming Planetary Reality Kernel: {e}")
            return f"Error resuming Planetary Reality Kernel: {str(e)}"
    
    def _get_cosmos_engine_status(self) -> str:
        """Get the status of the Planetary Reality Kernel"""
        if not self.kernel:
            return "Planetary Reality Kernel is not initialized"
        
        try:
            metrics = self.kernel.get_simulation_metrics()
            status_str = f"Planetary Reality Kernel Status:\n"
            status_str += f"- Status: {'Active' if self.kernel.active else 'Inactive'}\n"
            status_str += f"- Simulation Time: {metrics['simulation_time']:.2f}\n"
            status_str += f"- Time Dilation: {metrics['real_time_ratio']:.2f}x\n"
            status_str += f"- Total Entities: {metrics['total_entities']}\n"
            
            if metrics.get('dominant_emotion'):
                status_str += f"- Dominant Emotion: {metrics['dominant_emotion']}\n"
            
            if metrics.get('timeline_events'):
                status_str += f"- Timeline Events: {metrics['timeline_events'].get('total_events', 'N/A')}\n"
            
            return status_str
        except Exception as e:
            system_logger.error(f"Error getting Planetary Reality Kernel status: {e}")
            return f"Error getting Planetary Reality Kernel status: {str(e)}"
    
    def execute_command(self, command: str) -> str:
        """Execute a terminal command and return the output"""
        success, output, error = self.terminal_agent.safe_execute(command)
        
        if success:
            return f"Command executed successfully:\n\n{output}"
        else:
            return f"Error executing command: {error}\n{output}"
    
    def execute_simulation_command(self, command: str) -> Dict[str, Any]:
        """Execute a command within the simulation."""
        if self.conscious_substrate and self.conscious_substrate.initialized:
            return self.conscious_substrate.execute_simulation_command(command)
        else:
            return {
                "command": command,
                "timestamp": datetime.datetime.now().isoformat(),
                "success": False,
                "error": "Conscious Substrate not initialized"
            }
    
    def perform_breath_cycle(self) -> Dict[str, Any]:
        """Perform a full breath cycle across all components."""
        if self.conscious_substrate and self.conscious_substrate.initialized:
            return self.conscious_substrate.perform_breath_cycle()
        else:
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "success": False,
                "error": "Conscious Substrate not initialized"
            }
    
    def interactive_chat_mode(self):
        """Run the ORAMA system in interactive chat mode"""
        print("\n" + "="*60)
        print("ORAMA INTERACTIVE CHAT MODE")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'command: <cmd>' to execute a terminal command")
        print("Type 'engine start', 'engine stop', etc. to control the Genesis Cosmos Engine")
        print("="*60 + "\n")

        if self.kernel:
            print(f"Planetary Reality Kernel '{self.kernel.world_name}' is ready.")
        else:
            print("No kernel initialized. Use 'engine start' to begin.")

        while True:
            try:
                user_input = input("ORAMA> ").strip()
                if user_input.lower() in ['exit', 'quit']:
                    print("Exiting interactive chat mode. Goodbye!")
                    break
                elif user_input.lower().startswith('command:'):
                    command = user_input[len('command:'):].strip()
                    output = self.execute_command(command)
                    print(output)
                elif user_input.lower().startswith('engine'):
                    if 'start' in user_input.lower():
                        print(self._start_cosmos_engine())
                    elif 'stop' in user_input.lower():
                        print(self._stop_cosmos_engine())
                    elif 'pause' in user_input.lower():
                        print(self._pause_cosmos_engine())
                    elif 'resume' in user_input.lower():
                        print(self._resume_cosmos_engine())
                    elif 'status' in user_input.lower():
                        print(self._get_cosmos_engine_status())
                    else:
                        print("Unknown engine command. Use 'start', 'stop', 'pause', 'resume', or 'status'.")
                else:
                    response, context = self.process_query(user_input)
                    print(response)
            except KeyboardInterrupt:
                print("\nExiting interactive chat mode. Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

    def monitor_and_maintain_engines(self):
        """Continuously monitor and maintain all engines to ensure persistence."""
        while True:
            try:
                # Check the status of each engine
                if 'timeline_engine' in self.initialized_components:
                    timeline = self.initialized_components['timeline_engine']
                    if not timeline.is_active():
                        system_logger.warning("Timeline Engine stopped unexpectedly. Restarting...")
                        timeline.start()

                if 'quantum_physics' in self.initialized_components:
                    quantum = self.initialized_components['quantum_physics']
                    if not quantum.is_active():
                        system_logger.warning("Quantum Physics Engine stopped unexpectedly. Restarting...")
                        quantum.start()

                if 'planetary_reality_kernel' in self.initialized_components:
                    kernel = self.initialized_components['planetary_reality_kernel']
                    if not kernel.active:
                        system_logger.warning("Planetary Reality Kernel stopped unexpectedly. Restarting...")
                        kernel.start()

                # Add checks for other engines as needed

                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                system_logger.error(f"Error monitoring engines: {e}")
                time.sleep(5)  # Prevent tight loop on error

    def start_persistent_system(self):
        """Start the ORAMA system and ensure all engines remain persistent."""
        # Start all engines
        self.initialize_planetary_kernel()
        for component_name, component in self.initialized_components.items():
            if hasattr(component, 'start') and callable(component.start):
                try:
                    component.start()
                    system_logger.info(f"Started {component_name} successfully.")
                except Exception as e:
                    system_logger.error(f"Failed to start {component_name}: {e}")

        # Start the monitoring thread
        monitoring_thread = threading.Thread(target=self.monitor_and_maintain_engines, daemon=True)
        monitoring_thread.start()

        # Enter interactive chat mode
        self.interactive_chat_mode()