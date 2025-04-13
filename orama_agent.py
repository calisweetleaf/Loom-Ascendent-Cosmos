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
from typing import Dict, Any, List, Union, Optional, Tuple, Callable
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field, asdict

# ================================================================
#  Configuration and Logging
# ================================================================

# Configure base logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("orama_system.log"),
        logging.StreamHandler()
    ]
)

# Create logger instances
system_logger = logging.getLogger("OramaSystem")
perception_logger = logging.getLogger("OramaPerception")
memory_logger = logging.getLogger("OramaMemory")
knowledge_logger = logging.getLogger("OramaKnowledge")
truth_logger = logging.getLogger("OramaTruthValidator")

# Set up specific perception logger
perception_handler = RotatingFileHandler(
    "perception_stream.log", 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
perception_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
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
    
    def get_memory(self, hash_id: str) -> Optional[MemoryEvent]:
        """Get a memory by its hash_id"""
        if hash_id in self.memory_index:
            index = self.memory_index[hash_id]
            return self.memories[index]
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
        self.truth_constraints = [
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
        # Check for forbidden patterns
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(response):
                violation = self.forbidden_patterns[i]
                violation_reason = f"Response violates truth constraint by matching forbidden pattern: {violation}"
                truth_logger.warning(violation_reason)
                return False, response, violation_reason
        
        # Additional domain-specific validation could be implemented here
        
        return True, response, None
    
    def add_constraint(self, constraint: str) -> None:
        """Add a new truth constraint"""
        self.truth_constraints.append(constraint)
        truth_logger.info(f"Added truth constraint: {constraint}")
    
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
            'metric': re.compile(r'(\w+):\s+([\d\.]+)'),
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

# ================================================================
#  ORAMA System - Main Class
# ================================================================

class OramaSystem:
    """Main ORAMA system that integrates all components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the ORAMA system with configuration"""
        self.config = config or {}
        
        # Initialize state
        self.state = OramaState()
        
        # Initialize components
        self.memory_manager = OracleMemoryManager(
            max_memories=self.config.get('max_memories', 10000)
        )
        
        self.knowledge_synthesizer = KnowledgeSynthesizer(
            memory_manager=self.memory_manager
        )
        
        self.perception_parser = PerceptionParser(
            memory_manager=self.memory_manager
        )
        
        self.terminal_agent = TerminalAccessAgent(
            memory_manager=self.memory_manager
        )
        
        self.truth_validator = TruthValidator()
        
        # Set initialization flag
        self.state.is_initialized = True
        
        system_logger.info("ORAMA system initialized successfully")
    
    def process_perception(self, raw_input: str) -> SimulationPerception:
        """Process raw perception input"""
        self.state.perception_count += 1
        self.state.last_perception_time = datetime.datetime.now().isoformat()
        
        perception = self.perception_parser.process_raw_perception(raw_input)
        
        return perception
    
    def process_query(self, query_text: str) -> Tuple[str, QueryContext]:
        """Process a query and generate a response"""
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
        
        # Process the query and generate a response
        # This is where a real LLM would be used
        response = f"ORAMA has processed your query: {query_text}"
        
        # Validate response with truth constraints
        is_valid, response, violation = self.truth_validator.validate_response(response)
        
        if not is_valid:
            self.state.error_count += 1
            response = f"Error: Response violated truth constraints: {violation}"
        
        return response, context
    
    def execute_command(self, command: str) -> Tuple[bool, str]:
        """Safely execute a terminal command"""
        success, output, error = self.terminal_agent.safe_execute(command)
        
        if not success:
            self.state.error_count += 1
            return False, f"Command failed: {error}"
        
        return True, output
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current system state"""
        return asdict(self.state)
    
    def shutdown(self) -> None:
        """Safely shut down the system, saving all data"""
        system_logger.info("Shutting down ORAMA system")
        self.memory_manager.save_memories()
        self.knowledge_synthesizer.save_knowledge()
        system_logger.info("Shutdown complete")

# ================================================================
#  Main Entry Point
# ================================================================

def main():
    """Main entry point for the ORAMA system"""
    import argparse
    import time
    import random
    
    parser = argparse.ArgumentParser(description="ORAMA - Observation, Reasoning, And Memory Agent")
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--perception-file', type=str, help='Process perceptions from a file')
    parser.add_argument('--memory-file', type=str, default=MEMORY_FILE, help=f'Memory storage file (default: {MEMORY_FILE})')
    parser.add_argument('--knowledge-file', type=str, default=KNOWLEDGE_FILE, help=f'Knowledge storage file (default: {KNOWLEDGE_FILE})')
    parser.add_argument('--log-dir', type=str, default=LOG_DIR, help=f'Log directory (default: {LOG_DIR})')
    parser.add_argument('--max-memories', type=int, default=10000, help='Maximum number of memories to keep')
    parser.add_argument('--continuous', action='store_true', help='Run in continuous mode, generating and processing data')
    parser.add_argument('--interval', type=float, default=5.0, help='Interval between perception generation in seconds (default: 5.0)')
    parser.add_argument('--runtime', type=int, default=0, help='How long to run in continuous mode in seconds (0 for indefinite)')
    
    args = parser.parse_args()
    
    # Configure with args
    config = {
        'memory_file': args.memory_file,
        'knowledge_file': args.knowledge_file,
        'log_dir': args.log_dir,
        'max_memories': args.max_memories
    }
    
    # Initialize system
    orama = OramaSystem(config)
    
    # Process perceptions from file if specified
    if args.perception_file:
        try:
            with open(args.perception_file, 'r') as f:
                for line in f:
                    orama.process_perception(line.strip())
            print(f"Processed {orama.state.perception_count} perceptions from {args.perception_file}")
        except Exception as e:
            print(f"Error processing perception file: {e}")
    
    # Run interactive mode if specified
    if args.interactive:
        interactive_mode(orama)
    # Run in continuous mode if specified
    elif args.continuous:
        continuous_mode(orama, interval=args.interval, runtime=args.runtime)
    # If no mode is specified, run in continuous mode by default
    else:
        print("No operation mode specified. Running in continuous mode by default.")
        continuous_mode(orama, interval=args.interval, runtime=args.runtime)
    
    # Shutdown properly
    orama.shutdown()

def continuous_mode(orama: OramaSystem, interval: float = 5.0, runtime: int = 0) -> None:
    """
    Run ORAMA in continuous mode, automatically generating perceptions and knowledge
    
    Args:
        orama: The OramaSystem instance
        interval: Time between perception generation in seconds
        runtime: How long to run in seconds (0 for indefinite)
    """
    import time
    import random
    
    system_logger.info(f"Starting continuous mode (interval={interval}s, runtime={runtime if runtime > 0 else 'indefinite'}s)")
    print(f"ORAMA Continuous Mode (Press Ctrl+C to stop)")
    
    # Sample perception templates for simulation
    perception_templates = [
        "Entity: {entity} observed in {location} with state {state}",
        "Event: {entity} changed from {old_state} to {new_state}",
        "INFO: System is analyzing {entity} with metric {metric}: {value}",
        "Entity: {entity} interacting with {other_entity} in {context}",
        "Event: New pattern detected in {entity} behavior: {pattern}"
    ]
    
    # Sample data for templates
    entities = ["Particle", "Wave", "Field", "Anomaly", "Structure", "Dimension", "Recursion", "Pattern", "Cycle", "Nexus"]
    locations = ["Quadrant1", "MainLoop", "CoreMemory", "PerceptionField", "DataStream", "VoidSpace", "BoundaryLayer"]
    states = ["Stable", "Unstable", "Expanding", "Contracting", "Resonating", "Diverging", "Converging", "Fluctuating"]
    metrics = ["Coherence", "Stability", "Symmetry", "Complexity", "Entropy", "Recursion", "Amplitude", "Frequency"]
    patterns = ["Recursive", "Symmetric", "Oscillating", "Divergent", "Convergent", "Self-similar", "Fractal", "Chaotic"]
    contexts = ["Information exchange", "Energy transfer", "Structure formation", "Pattern recognition", "Memory encoding"]
    
    start_time = time.time()
    cycle_count = 0
    last_knowledge_gen_time = start_time
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if runtime has been exceeded
            if runtime > 0 and elapsed > runtime:
                system_logger.info(f"Runtime of {runtime}s reached, exiting continuous mode")
                break
            
            # Generate a random perception
            template = random.choice(perception_templates)
            perception_text = ""
            
            # Generate the appropriate perception based on template type
            if "Entity:" in template and "interacting" in template:
                entity = random.choice(entities)
                other_entity = random.choice([e for e in entities if e != entity])
                context = random.choice(contexts)
                perception_text = template.format(entity=entity, other_entity=other_entity, context=context)
            elif "Entity:" in template:
                entity = random.choice(entities)
                location = random.choice(locations)
                state = random.choice(states)
                perception_text = template.format(entity=entity, location=location, state=state)
            elif "Event:" in template and "pattern" in template:
                entity = random.choice(entities)
                pattern = random.choice(patterns)
                perception_text = template.format(entity=entity, pattern=pattern)
            elif "Event:" in template:
                entity = random.choice(entities)
                old_state = random.choice(states)
                # Ensure new state is different from old state
                new_state = random.choice([s for s in states if s != old_state])
                perception_text = template.format(entity=entity, old_state=old_state, new_state=new_state)
            elif "INFO:" in template:
                entity = random.choice(entities)
                metric = random.choice(metrics)
                value = round(random.uniform(0, 100), 2)
                perception_text = template.format(entity=entity, metric=metric, value=value)
            else:
                # Fallback
                perception_text = f"Event: Generic observation cycle {cycle_count}"
            
            # Process the generated perception
            perception = orama.process_perception(perception_text)
            print(f"[{perception.timestamp}] Generated perception: {perception_text}")
            
            # Every 5 cycles, generate knowledge from recent perceptions
            if cycle_count % 5 == 0 and cycle_count > 0:
                recent_perceptions = orama.perception_parser.get_recent_perceptions(count=10)
                generated_entities = orama.knowledge_synthesizer.generate_knowledge_from_perceptions(recent_perceptions)
                
                if generated_entities:
                    print(f"Generated {len(generated_entities)} new knowledge entities")
                    # Add entities to known entities list in state
                    orama.state.known_entities.extend(generated_entities)
                
                # Save current state periodically
                orama.memory_manager.save_memories()
                orama.knowledge_synthesizer.save_knowledge()
                last_knowledge_gen_time = current_time
            
            # Increment cycle and wait for next interval
            cycle_count += 1
            time_to_wait = interval - (time.time() - current_time)
            if time_to_wait > 0:
                time.sleep(time_to_wait)
                
    except KeyboardInterrupt:
        system_logger.info("Received interrupt, exiting continuous mode")
        print("\nExiting continuous mode")

def interactive_mode(orama: OramaSystem) -> None:
    """Run ORAMA in interactive CLI mode"""
    print("ORAMA Interactive Mode")
    print("Type 'exit' or 'quit' to end session")
    print("Commands starting with ! will be executed in the terminal")
    print("Commands starting with ? will be treated as queries")
    print("All other input will be treated as perception data")
    
    while True:
        try:
            user_input = input("\nORAMA> ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                break
                
            elif user_input.startswith('!'):
                # Terminal command
                cmd = user_input[1:].strip()
                success, output = orama.execute_command(cmd)
                print(f"Command {'succeeded' if success else 'failed'}")
                print(output)
                
            elif user_input.startswith('?'):
                # Query
                query = user_input[1:].strip()
                response, _ = orama.process_query(query)
                print(f"Response: {response}")
                
            else:
                # Perception data
                perception = orama.process_perception(user_input)
                print(f"Processed perception of type {perception.perception_type}")
                
        except KeyboardInterrupt:
            print("\nReceived interrupt, exiting...")
            break
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("Exiting interactive mode")

if __name__ == "__main__":
    main()

