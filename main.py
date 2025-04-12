# ================================================================
#  LOOM ASCENDANT COSMOS — RECURSIVE SYSTEM MODULE
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
#  Integrity Hash (SHA-256): d3ab9688a5a20b8065990cd9b91805e3d892d6e72472f69dd9afe719250c5e37
# ================================================================
import os
import time
import numpy as np
import traceback
import matplotlib.pyplot as plt
import logging
from logging.handlers import RotatingFileHandler
import json
import argparse
import sys
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

# Configure general logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("genesis_cosmos.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GenesisMain")

# Set up specific universe evolution logger
universe_logger = logging.getLogger("UniverseEvolution")
universe_logger.setLevel(logging.INFO)
universe_log_handler = RotatingFileHandler(
    "universe_evolution.log", 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
universe_log_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)
universe_logger.addHandler(universe_log_handler)

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def check_dependencies():
    """Check and install required dependencies"""
    try:
        import ollama
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "ollama", "numpy", "matplotlib"])
        print("Dependencies installed. Restarting...")
        os.execv(sys.executable, ['python'] + sys.argv)

def check_ollama_status(model_name="qwen2.5:3b", model_path=None):
    """Check if Ollama is running and the specified model is available"""
    try:
        import ollama
        models = ollama.list()
        model_names = [model.get('name', '') for model in models.get('models', [])]
        
        print(f"Available Ollama models: {', '.join(model_names)}")
        
        # If model_path is provided, use the local GGUF file
        if model_path and os.path.exists(model_path):
            print(f"Using local model path: {model_path}")
            # Configure ollama to use the local model
            # In a production environment, you would register this model with ollama
            return model_name
            
        if (model_name not in model_names):
            print(f"Warning: Model {model_name} not found in available models.")
            print(f"Available models: {', '.join(model_names)}")
            
            if len(model_names) > 0:
                response = input(f"Would you like to use {model_names[0]} instead? (y/n): ")
                if response.lower() == 'y':
                    return model_names[0]
                else:
                    print(f"Please pull the model with: ollama pull {model_name}")
                    sys.exit(1)
            else:
                print("No models available. Please pull a model with: ollama pull qwen2.5:3b")
                sys.exit(1)
        
        return model_name
    except Exception as e:
        print(f"Error checking Ollama: {e}")
        print("Make sure Ollama is installed and running.")
        print("You can install Ollama from: https://ollama.com/")
        print("Then start it with the 'ollama serve' command.")
        sys.exit(1)

def initialize_cosmos_engine():
    """Initialize the Genesis Cosmos Engine components"""
    try:
        # Import all engine components
        from reality_kernel import RealityKernel
        from universe_engine import UniverseEngine
        from timeline_engine import TimelineEngine 
        from harmonic_engine import HarmonicEngine
        from aether_engine import AetherEngine
        from quantum_physics import QuantumField
        
        # Log initialization
        universe_logger.info(f"Initializing Genesis Cosmos Engine components at {datetime.now().isoformat()}")
        
        # Create basic config (adjust based on your actual implementation)
        config = {
            'reality_cycles_per_second': 1.0,
            'timeline_branches': 3,
            'ethical_dimensions': 3,
            'aether_resolution': 64,
            'quantum_precision': 1e-30
        }
        
        # Initialize reality kernel (if available)
        try:
            kernel = RealityKernel(config)
            universe_logger.info("RealityKernel initialized successfully")
            return kernel
        except Exception as e:
            universe_logger.error(f"Failed to initialize RealityKernel: {e}")
            
            # Fall back to individual components if available
            try:
                # Create minimal engine components
                timeline = TimelineEngine()
                universe_logger.info("TimelineEngine initialized with 1 timeline(s) at 1.0Hz breath frequency")
                
                # Get an observer in the timeline
                timeline.register_observer("test_observer")
                # Test with a sample event to make sure timeline works
                timeline.notify_observers("test_event", 0)
                
                aether = AetherEngine()
                universe_logger.info("AetherEngine initialized")
                
                try:
                    # Create universe engine with proper parameters
                    physics = {'c': 299792458.0, 'G': 6.67430e-11, 'hubble_constant': 70.0}
                    initial_conditions = {'initial_temperature': 1e32, 'initial_density': 1e96}
                    
                    universe = UniverseEngine(
                        aether_space=aether.space, 
                        physics=physics,
                        timeline=timeline,
                        initial_conditions=initial_conditions
                    )
                    universe_logger.info("UniverseEngine initialized")
                except Exception as ue_error:
                    universe_logger.error(f"Failed to initialize UniverseEngine: {ue_error}")
                    universe = None
                
                # Create a minimal kernel-like object to hold components
                class MinimalKernel:
                    def __init__(self):
                        self.timeline = timeline
                        self.aether = aether
                        self.universe = universe
                        self.config = config
                        self.tick_count = 0
                        
                    def breathe(self, cycles=1):
                        """Run simulation for specified cycles"""
                        universe_logger.info(f"Running {cycles} breath cycles")
                        for i in range(cycles):
                            self.tick_count += 1
                            
                            # Log universe evolution data
                            universe_data = {
                                "tick": self.tick_count,
                                "timestamp": datetime.now().isoformat(),
                                "entropy": 0.1 * self.tick_count,
                                "structures": self.tick_count * 3,
                                "complexity": min(1.0, 0.05 * self.tick_count),
                                "phase": "expansion"
                            }
                            universe_logger.info(f"Universe tick {self.tick_count}: {json.dumps(universe_data)}")
                        
                        return True
                
                kernel = MinimalKernel()
                universe_logger.info("MinimalKernel created as fallback")
                return kernel
                
            except Exception as nested_e:
                universe_logger.error(f"Failed to initialize minimal components: {nested_e}")
                raise
    
    except ImportError as e:
        universe_logger.error(f"Failed to import engine components: {e}")
        
        # Create mock engine for demonstration
        class MockEngine:
            def __init__(self):
                self.tick_count = 0
                universe_logger.info("MockEngine initialized (no actual engine components available)")
            
            def breathe(self, cycles=1):
                """Mock simulation function"""
                universe_logger.info(f"MockEngine: Running {cycles} simulated breath cycles")
                for i in range(cycles):
                    self.tick_count += 1
                    
                    # Generate mock universe evolution data
                    universe_data = {
                        "tick": self.tick_count,
                        "timestamp": datetime.now().isoformat(),
                        "entropy": 0.1 * self.tick_count,
                        "structures": max(1, self.tick_count * 2),
                        "complexity": min(1.0, 0.05 * self.tick_count),
                        "phase": "mock_simulation",
                        "note": "This is simulated data - no actual engine running"
                    }
                    
                    # Log the universe evolution data
                    universe_logger.info(f"MockEngine tick {self.tick_count}: {json.dumps(universe_data)}")
                
                return True
        
        return MockEngine()

def run_ollama_chat(engine, model_name):
    """Run an interactive chat session with Ollama and the Cosmos Engine"""
    try:
        import ollama
        
        print(f"\n{'=' * 60}")
        print(f"Genesis Cosmos Engine - Ollama Chat Interface")
        print(f"Connected to model: {model_name}")
        print(f"Type 'exit' to end the session, 'help' for commands")
        print(f"{'=' * 60}\n")
        
        # Initial breathe cycle to get things started
        print("Running initial simulation cycle...")
        engine.breathe(cycles=1)
        
        # Record chat history for context
        history = [
            {
                "role": "system", 
                "content": "You are an AI connected to the Genesis Cosmos Engine, "
                           "a simulation of emergent realities through a sophisticated "
                           "multi-layered architecture. Respond as if you are exploring "
                           "and interacting with this cosmos."
            }
        ]
        
        # Set up automatic universe evolution tracking
        universe_tracker = {
            "last_tick_time": time.time(),
            "tick_interval": 30,  # Seconds between automatic ticks
            "total_ticks": 0,
            "active": True
        }
        
        while True:
            # Check if we should run a cosmos tick
            current_time = time.time()
            if universe_tracker["active"] and (current_time - universe_tracker["last_tick_time"]) > universe_tracker["tick_interval"]:
                try:
                    # Run a cosmic tick
                    print("\n[System] Running automatic simulation cycle...")
                    universe_logger.info(f"Automatic cosmic tick triggered after {current_time - universe_tracker['last_tick_time']:.1f} seconds")
                    engine.breathe(cycles=1)
                    universe_tracker["total_ticks"] += 1
                    universe_tracker["last_tick_time"] = current_time
                    
                    # Log some basic metrics if available
                    if hasattr(engine, 'universe') and hasattr(engine.universe, 'evolution_metrics'):
                        metrics = engine.universe.evolution_metrics
                        universe_logger.info(f"Evolution metrics: {json.dumps(metrics, default=str)}")
                except Exception as e:
                    print(f"\n[Error] Simulation cycle failed: {e}")
                    universe_logger.error(f"Error during automatic cosmic tick: {e}")
            
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Ending session...")
                break
            
            # Check for help command
            if user_input.lower() == 'help':
                print("\n--- Available Commands ---")
                print("help       : Show this help message")
                print("exit       : End the session")
                print("tick       : Run a simulation cycle manually")
                print("pause      : Pause automatic simulation cycles")
                print("resume     : Resume automatic simulation cycles")
                print("status     : Show current engine status")
                print("log [n]    : Show the last n log entries (default: 5)")
                continue
            
            # Check for tick command
            if user_input.lower() == 'tick':
                try:
                    print("Running simulation cycle...")
                    universe_logger.info("Manual cosmic tick triggered by user")
                    engine.breathe(cycles=1)
                    universe_tracker["total_ticks"] += 1
                    universe_tracker["last_tick_time"] = time.time()
                    print("Simulation cycle completed")
                    continue
                except Exception as e:
                    print(f"Error during simulation cycle: {e}")
                    universe_logger.error(f"Error during manual cosmic tick: {e}")
                    continue
            
            # Check for pause/resume commands
            if user_input.lower() == 'pause':
                universe_tracker["active"] = False
                print("Automatic simulation cycles paused")
                universe_logger.info("Automatic cosmic ticks paused by user")
                continue
            
            if user_input.lower() == 'resume':
                universe_tracker["active"] = True
                print("Automatic simulation cycles resumed")
                universe_logger.info("Automatic cosmic ticks resumed by user")
                continue
            
            # Check for status command
            if user_input.lower() == 'status':
                print("\n--- Engine Status ---")
                print(f"Total simulation cycles: {universe_tracker['total_ticks']}")
                print(f"Auto-cycle mode: {'Active' if universe_tracker['active'] else 'Paused'}")
                
                # Show engine-specific status if available
                if hasattr(engine, 'universe') and hasattr(engine.universe, 'current_time'):
                    print(f"Universe time: {engine.universe.current_time:.2e}")
                
                if hasattr(engine, 'timeline') and hasattr(engine.timeline, 'master_tick'):
                    print(f"Timeline master tick: {engine.timeline.master_tick}")
                
                continue
            
            # Check for log command
            if user_input.lower().startswith('log'):
                try:
                    # Parse number of entries to show
                    parts = user_input.split()
                    entries = 5  # Default
                    if len(parts) > 1 and parts[1].isdigit():
                        entries = int(parts[1])
                    
                    # Read the log file
                    with open("universe_evolution.log", "r") as f:
                        lines = f.readlines()
                    
                    # Show the last n entries
                    print(f"\n--- Last {min(entries, len(lines))} Universe Evolution Log Entries ---")
                    for line in lines[-entries:]:
                        print(line.strip())
                    
                    continue
                except Exception as e:
                    print(f"Error reading log: {e}")
                    continue
            
            # Add user message to history
            history.append({"role": "user", "content": user_input})
            
            # Call Ollama for response
            print("\nAI: ", end="", flush=True)
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=history,
                    options={"temperature": 0.7}
                )
                
                # Extract the assistant's message
                assistant_message = response['message']['content']
                
                # Add to history
                history.append({"role": "assistant", "content": assistant_message})
                
                # Print the response
                print(assistant_message)
                
                # Limit history size to prevent context overflow
                if len(history) > 20:
                    # Keep system prompt and last 10 exchanges
                    history = [history[0]] + history[-19:]
                
            except Exception as e:
                print(f"Error communicating with Ollama: {e}")
    
    except Exception as e:
        print(f"Error in chat session: {e}")
        traceback.print_exc()

def main():
    """Main function to run the Genesis Cosmos Engine with Ollama integration"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Genesis Cosmos Engine with Ollama Integration")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", 
                       help="Ollama model to use (default: qwen2.5:3b)")
    parser.add_argument("--model_path", type=str, default=None, 
                       help="Path to local GGUF model file (default: None)")
    parser.add_argument("--cycles", type=int, default=1,
                       help="Initial simulation cycles to run (default: 1)")
    args = parser.parse_args()
    
    try:
        print("Initializing Genesis Cosmos Engine...")
        
        # Log startup
        universe_logger.info("=" * 60)
        universe_logger.info(f"Genesis Cosmos Engine startup at {datetime.now().isoformat()}")
        
        # Check dependencies
        check_dependencies()
        
        # Check Ollama status and get confirmed model name
        model_name = check_ollama_status(args.model, args.model_path)
        
        # Initialize the Cosmos Engine
        engine = initialize_cosmos_engine()
        
        # Run initial simulation cycles
        if args.cycles > 0:
            print(f"Running {args.cycles} initial simulation cycles...")
            engine.breathe(cycles=args.cycles)
        
        # Start the Ollama chat interface
        run_ollama_chat(engine, model_name)
        
        # Log shutdown
        universe_logger.info(f"Genesis Cosmos Engine shutdown at {datetime.now().isoformat()}")
        universe_logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Unhandled error in main: {e}")
        traceback.print_exc()
        print(f"\nError: {e}")
        print("Please check the logs for more details.")
        
        # Log shutdown with error
        universe_logger.error(f"Genesis Cosmos Engine abnormal shutdown at {datetime.now().isoformat()}: {e}")
        universe_logger.info("=" * 60)
        
        sys.exit(1)

if __name__ == "__main__":
    main()