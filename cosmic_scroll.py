# ================================================================
#  LOOM ASCENDANT COSMOS — RECURSIVE SYSTEM MODULE
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
#  Integrity Hash (SHA-256): d3ab9688a5a20b8065990cd9b91805e3d892d6e72472f69dd9afe719250c5e37
# ================================================================
"""
CosmicScroll Engine
------------------
Procedural cosmic entity generation and management system for a comprehensive reality simulation.
Designed to integrate with quantum physics, timeline, aether, universe engines and reality kernel.
"""

import random
import math
import uuid
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# Physical constants from the Planetary Framework
G = 6.67430e-11  # Gravitational constant
C = 299792458    # Speed of light
H = 6.62607015e-34  # Planck constant
ALPHA = 7.2973525693e-3  # Fine-structure constant
OMEGA_LAMBDA = 0.6889  # Dark energy density parameter

class EntityType(Enum):
    """Classification of cosmic entities for the DRM system"""
    UNIVERSE = "universe"
    GALAXY_CLUSTER = "galaxy_cluster"
    GALAXY = "galaxy"
    STAR = "star"
    PLANET = "planet"
    MOON = "moon"
    ASTEROID = "asteroid"
    CIVILIZATION = "civilization"
    ANOMALY = "anomaly"


class DimensionalRealityManager:
    """
    Central registry for all entities in the simulation.
    Acts as an interface between the cosmic scroll and other engine components.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DimensionalRealityManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the DRM singleton"""
        self.entities = {}  # id -> entity
        self.entity_sector_map = defaultdict(set)  # entity_id -> set of sectors
        self.sector_entity_map = defaultdict(lambda: defaultdict(set))  # sector -> {entity_type: set(entity_ids)}
        self.query_cache = {}  # (query_params) -> results
        self.time_dilation_factor = 1.0
        self.reality_coherence = 1.0
        self.active_observers = set()
        
    def store_entity(self, entity_id: str, entity: Any, sectors: Optional[List[Tuple]] = None):
        """Register an entity in the DRM system"""
        self.entities[entity_id] = entity
        
        if hasattr(entity, 'entity_type'):
            entity_type = entity.entity_type.value
        else:
            entity_type = EntityType.ANOMALY.value
            
        if sectors:
            for sector in sectors:
                self.entity_sector_map[entity_id].add(sector)
                self.sector_entity_map[sector][entity_type].add(entity_id)
    
    def query_entities(self, entity_type: str, sector: Optional[Tuple] = None) -> List[Any]:
        """Retrieve entities matching criteria"""
        cache_key = (entity_type, sector)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
            
        if sector is None:
            # Return all entities of this type
            results = [e for e in self.entities.values() 
                      if hasattr(e, 'entity_type') and e.entity_type.value == entity_type]
        else:
            # Return entities in the specific sector
            entity_ids = self.sector_entity_map[sector].get(entity_type, set())
            results = [self.entities[eid] for eid in entity_ids if eid in self.entities]
            
        self.query_cache[cache_key] = results
        return results
    
    def invalidate_cache(self, sector: Optional[Tuple] = None):
        """Clear query cache for updated sectors"""
        if sector is None:
            self.query_cache.clear()
        else:
            keys_to_remove = [k for k in self.query_cache if k[1] == sector]
            for key in keys_to_remove:
                del self.query_cache[key]
    
    def get_entity(self, entity_id: str) -> Optional[Any]:
        """Retrieve a specific entity by ID"""
        return self.entities.get(entity_id)
    
    def update_entity_sectors(self, entity_id: str, old_sector: Tuple, new_sector: Tuple):
        """Update entity location when it moves between sectors"""
        if entity_id not in self.entities:
            return False
            
        entity = self.entities[entity_id]
        entity_type = entity.entity_type.value if hasattr(entity, 'entity_type') else EntityType.ANOMALY.value
        
        # Remove from old sector
        if old_sector in self.sector_entity_map and entity_type in self.sector_entity_map[old_sector]:
            self.sector_entity_map[old_sector][entity_type].discard(entity_id)
            self.entity_sector_map[entity_id].discard(old_sector)
            
        # Add to new sector
        self.sector_entity_map[new_sector][entity_type].add(entity_id)
        self.entity_sector_map[entity_id].add(new_sector)
        
        # Invalidate affected caches
        self.invalidate_cache(old_sector)
        self.invalidate_cache(new_sector)
        
        return True
    
    def register_observer(self, observer_id: str, position: Tuple):
        """Register an active observer in the simulation"""
        self.active_observers.add(observer_id)
        # Increase simulation fidelity around observer position
        self._adjust_reality_coherence(position)
    
    def _adjust_reality_coherence(self, focal_point: Tuple, radius: int = 3):
        """Adjust reality coherence based on observer proximity"""
        affected_sectors = self._get_sectors_in_radius(focal_point, radius)
        # Implementation details for coherence adjustment would go here
        pass
        
    def _get_sectors_in_radius(self, center: Tuple, radius: int) -> List[Tuple]:
        """Get all sectors within a radius of a central point"""
        sectors = []
        x, y, z = center
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx*dx + dy*dy + dz*dz <= radius*radius:
                        sectors.append((x + dx, y + dy, z + dz))
        return sectors


# Initialize the DRM singleton
DRM = DimensionalRealityManager()


class QuantumSeedGenerator:
    """
    Generates procedural content using quantum-inspired permutations
    Ensures consistent entity generation across multiple simulation runs
    """
    
    _seeds_cache = {}
    
    @staticmethod
    def generate_planet_seed(galaxy_id: str, sector: tuple) -> int:
        """Generate a deterministic seed for a planet"""
        cache_key = f"planet|{galaxy_id}|{sector}"
        if cache_key in QuantumSeedGenerator._seeds_cache:
            return QuantumSeedGenerator._seeds_cache[cache_key]
            
        seed = hash(f"{galaxy_id}|{sector}") % 2**32
        QuantumSeedGenerator._seeds_cache[cache_key] = seed
        return seed
    
    @staticmethod
    def generate_star_seed(galaxy_id: str, sector: tuple, index: int) -> int:
        """Generate a deterministic seed for a star"""
        cache_key = f"star|{galaxy_id}|{sector}|{index}"
        if cache_key in QuantumSeedGenerator._seeds_cache:
            return QuantumSeedGenerator._seeds_cache[cache_key]
            
        seed = hash(f"{galaxy_id}|{sector}|{index}") % 2**32
        QuantumSeedGenerator._seeds_cache[cache_key] = seed
        return seed
    
    @staticmethod
    def generate_galaxy_seed(universe_id: str, position: tuple) -> int:
        """Generate a deterministic seed for a galaxy"""
        cache_key = f"galaxy|{universe_id}|{position}"
        if cache_key in QuantumSeedGenerator._seeds_cache:
            return QuantumSeedGenerator._seeds_cache[cache_key]
            
        seed = hash(f"{universe_id}|{position[0]}|{position[1]}|{position[2]}") % 2**32
        QuantumSeedGenerator._seeds_cache[cache_key] = seed
        return seed
    
    @staticmethod
    def generate_anomaly_seed(sector: tuple, timestamp: float) -> int:
        """Generate a non-deterministic seed for anomalies"""
        # Anomalies are unique events that shouldn't be cached
        return hash(f"{sector}|{timestamp}|{uuid.uuid4()}") % 2**32
    
    @staticmethod
    def clear_cache():
        """Clear the seed cache"""
        QuantumSeedGenerator._seeds_cache.clear()


class CosmicEntity:
    """Base class for all entities in the cosmic simulation"""
    
    def __init__(self, entity_id: str = None):
        self.entity_id = entity_id or f"{self.__class__.__name__.lower()}_{uuid.uuid4().hex}"
        self.traits = {}
        self.motifs = []
        self.creation_time = 0
        self.last_update_time = 0
        self.sectors = set()
        self.entity_type = EntityType.ANOMALY  # Default
        
    def evolve(self, time_delta: float):
        """Update entity state based on time progression"""
        self.last_update_time += time_delta
        # Base implementation does nothing, subclasses will override
        
    def add_to_reality(self, sectors: List[Tuple] = None):
        """Register entity with the DRM system"""
        if sectors:
            self.sectors.update(sectors)
        DRM.store_entity(self.entity_id, self, list(self.sectors))
        
    def get_trait(self, trait_name: str, default=None):
        """Get a trait value with optional default"""
        return self.traits.get(trait_name, default)
    
    def set_trait(self, trait_name: str, value):
        """Set a trait value"""
        self.traits[trait_name] = value
        
    def has_motif(self, motif_name: str) -> bool:
        """Check if entity has a specific motif"""
        return motif_name in self.motifs


class GalaxyType(Enum):
    """Classification of galaxy morphologies"""
    SPIRAL = "spiral"
    ELLIPTICAL = "elliptical"
    IRREGULAR = "irregular"
    PECULIAR = "peculiar"
    DWARF = "dwarf"


class Galaxy(CosmicEntity):
    """Representation of a galaxy with star systems"""
    
    def __init__(self, galaxy_id: str = None):
        super().__init__(galaxy_id)
        self.entity_type = EntityType.GALAXY
        self.galaxy_type = None
        self.stars = []
        self.age = 0
        self.size = 0
        self.active_regions = set()
        self.metallicity = 0
        self.black_hole_mass = 0
        self.star_formation_rate = 0
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve galaxy over time - affects star formation"""
        super().evolve(time_delta)
        self.age += time_delta
        
        # Only evolve stars in active regions
        if self.active_regions:
            for region in self.active_regions:
                stars = DRM.query_entities(EntityType.STAR.value, region)
                for star in stars:
                    if star.galaxy_id == self.entity_id:
                        star.evolve(time_delta)
                        
    def add_star(self, star):
        """Add a star to this galaxy"""
        star.galaxy_id = self.entity_id
        self.stars.append(star.entity_id)


class StarType(Enum):
    """Stellar classification system"""
    O = "O"  # Hot, massive, blue
    B = "B"  # Blue-white
    A = "A"  # White
    F = "F"  # Yellow-white
    G = "G"  # Yellow (Sun-like)
    K = "K"  # Orange
    M = "M"  # Red dwarf
    L = "L"  # Brown dwarf
    T = "T"  # Methane dwarf
    Y = "Y"  # Ultra-cool brown dwarf
    NEUTRON = "neutron"
    WHITE_DWARF = "white_dwarf"
    BLACK_HOLE = "black_hole"


class Star(CosmicEntity):
    """Stellar object that can host planetary systems"""
    
    def __init__(self, star_id: str = None):
        super().__init__(star_id)
        self.entity_type = EntityType.STAR
        self.star_type = None
        self.planets = []  # List of planet IDs
        self.luminosity = 0
        self.mass = 0
        self.radius = 0
        self.temperature = 0
        self.age = 0
        self.life_expectancy = 0
        self.color = (0, 0, 0)  # RGB
        self.galaxy_id = None
        self.position = (0, 0, 0)  # Position in galaxy
        self.habitable_zone = (0, 0)  # Inner and outer radii
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve star over time - affects luminosity, temperature"""
        super().evolve(time_delta)
        self.age += time_delta
        
        # Calculate evolution based on stellar models
        self._update_stellar_properties()
        
        # Evolve planets
        for planet_id in self.planets:
            planet = DRM.get_entity(planet_id)
            if planet:
                planet.evolve(time_delta)
                
    def _update_stellar_properties(self):
        """Update star properties based on its age and type"""
        # Simplified stellar evolution model
        if self.age > self.life_expectancy:
            self._initiate_stellar_death()
        else:
            # Main sequence evolution - gradual changes
            age_fraction = self.age / self.life_expectancy
            
            # Stars get slightly hotter and more luminous over time
            luminosity_factor = 1 + (0.1 * age_fraction)
            self.luminosity *= luminosity_factor
            
            # Recalculate habitable zone based on luminosity
            self._update_habitable_zone()
    
    def _update_habitable_zone(self):
        """Update the habitable zone based on current luminosity"""
        # Simplified model based on stellar luminosity
        # HZ proportional to square root of luminosity
        luminosity_factor = math.sqrt(self.luminosity)
        self.habitable_zone = (
            0.95 * luminosity_factor,  # Inner boundary
            1.37 * luminosity_factor   # Outer boundary
        )
        
    def _initiate_stellar_death(self):
        """Handle star's end-of-life transition"""
        if self.mass < 0.5:  # Low mass
            # Becomes white dwarf
            self.star_type = StarType.WHITE_DWARF
            self.temperature *= 0.5
            self.radius *= 0.01
            self.luminosity *= 0.001
        elif self.mass < 8:  # Medium mass
            # Also becomes white dwarf after red giant phase
            self.star_type = StarType.WHITE_DWARF
            self.temperature *= 0.7
            self.radius *= 0.01
            self.luminosity *= 0.01
        elif self.mass < 20:  # High mass
            # Becomes neutron star
            self.star_type = StarType.NEUTRON
            self.radius *= 0.0001
            self.temperature *= 10
            self.luminosity *= 0.0001
        else:  # Very high mass
            # Becomes black hole
            self.star_type = StarType.BLACK_HOLE
            self.radius *= 0.00001
            self.luminosity = 0
            
            # Create accretion disk effects
            if random.random() < 0.3:
                self.motifs.append('event_horizon')
                self.motifs.append('hawking_radiation')


class PlanetType(Enum):
    """Classification of planetary bodies"""
    TERRESTRIAL = "terrestrial"
    GAS_GIANT = "gas_giant"
    ICE_GIANT = "ice_giant"
    DWARF = "dwarf"
    SUPER_EARTH = "super_earth"
    HOT_JUPITER = "hot_jupiter"
    OCEAN_WORLD = "ocean_world"
    DESERT_WORLD = "desert_world"
    ICE_WORLD = "ice_world"
    LAVA_WORLD = "lava_world"


class Planet(CosmicEntity):
    """Planetary body that can host life and civilizations"""
    
    def __init__(self, planet_id: str = None):
        super().__init__(planet_id)
        self.entity_type = EntityType.PLANET
        self.planet_type = None
        self.star_id = None
        self.moons = []
        self.mass = 0
        self.radius = 0
        self.density = 0
        self.gravity = 0
        self.orbital_period = 0
        self.rotation_period = 0
        self.axial_tilt = 0
        self.orbital_eccentricity = 0
        self.albedo = 0
        self.atmosphere = {}  # Component -> percentage
        self.surface = {}  # Feature -> percentage
        self.climate = {}
        self.has_magnetosphere = False
        self.temperature = 0
        self.civilizations = []
        self.habitability_index = 0
        self.biosphere_complexity = 0
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve planet over time - affects climate, geology"""
        super().evolve(time_delta)
        
        # Geological processes
        self._evolve_geology(time_delta)
        
        # Climate evolution
        self._evolve_climate(time_delta)
        
        # Biosphere evolution
        if self.biosphere_complexity > 0:
            self._evolve_biosphere(time_delta)
            
        # Civilization evolution
        for civ_id in self.civilizations:
            civ = DRM.get_entity(civ_id)
            if civ:
                civ.evolve(time_delta)
                
    def _evolve_geology(self, time_delta: float):
        """Evolve geological features over time"""
        # Simplified plate tectonics and erosion model
        if self.has_motif('tectonic_dreams') and random.random() < 0.01 * time_delta:
            # Geological event
            event_type = random.choice(['volcanic', 'earthquake', 'erosion'])
            if event_type == 'volcanic':
                self.surface['volcanic'] = self.surface.get('volcanic', 0) + 0.01
                self.atmosphere['sulfur'] = self.atmosphere.get('sulfur', 0) + 0.005
            elif event_type == 'earthquake':
                if 'mountains' in self.surface:
                    self.surface['mountains'] += 0.005
            elif event_type == 'erosion':
                if 'mountains' in self.surface and self.surface['mountains'] > 0.01:
                    self.surface['mountains'] -= 0.005
                    self.surface['sedimentary'] = self.surface.get('sedimentary', 0) + 0.005
    
    def _evolve_climate(self, time_delta: float):
        """Evolve climate patterns over time"""
        # Get host star
        star = DRM.get_entity(self.star_id)
        if not star:
            return
            
        # Update temperature based on star's luminosity and orbital distance
        star_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, star.position)))
        luminosity_factor = star.luminosity / (star_distance ** 2)
        
        # Basic greenhouse effect from atmosphere
        greenhouse_factor = 1.0
        if 'carbon_dioxide' in self.atmosphere:
            greenhouse_factor += self.atmosphere['carbon_dioxide'] * 10
        if 'methane' in self.atmosphere:
            greenhouse_factor += self.atmosphere['methane'] * 30
            
        # Calculate temperature
        base_temp = -270 + (340 * luminosity_factor)  # Simple model in Kelvin
        self.temperature = base_temp * greenhouse_factor - 273.15  # Convert to Celsius
        
        # Update climate zones based on temperature and axial tilt
        self._update_climate_zones()
        
    def _update_climate_zones(self):
        """Update climate zone distribution based on current parameters"""
        # Reset climate zones
        self.climate = {}
        
        # Temperature-based assignments
        if self.temperature < -50:
            self.climate['polar'] = 0.8
            self.climate['tundra'] = 0.2
        elif self.temperature < -10:
            self.climate['polar'] = 0.4
            self.climate['tundra'] = 0.4
            self.climate['taiga'] = 0.2
        elif self.temperature < 5:
            self.climate['tundra'] = 0.3
            self.climate['taiga'] = 0.4
            self.climate['temperate'] = 0.3
        elif self.temperature < 15:
            self.climate['taiga'] = 0.2
            self.climate['temperate'] = 0.6
            self.climate['mediterranean'] = 0.2
        elif self.temperature < 25:
            self.climate['temperate'] = 0.3
            self.climate['mediterranean'] = 0.3
            self.climate['subtropical'] = 0.4
        elif self.temperature < 35:
            self.climate['mediterranean'] = 0.2
            self.climate['subtropical'] = 0.3
            self.climate['tropical'] = 0.3
            self.climate['desert'] = 0.2
        else:
            self.climate['subtropical'] = 0.1
            self.climate['tropical'] = 0.3
            self.climate['desert'] = 0.6
            
        # Adjust for water coverage (if present in surface)
        water_percentage = self.surface.get('water', 0)
        if water_percentage > 0.3:
            # Reduce desert, increase humid zones
            if 'desert' in self.climate:
                desert_reduction = min(self.climate['desert'], water_percentage * 0.3)
                self.climate['desert'] -= desert_reduction
                self.climate['tropical'] = self.climate.get('tropical', 0) + desert_reduction
                
            # Add ocean climate
            self.climate['oceanic'] = water_percentage * 0.7
            
    def _evolve_biosphere(self, time_delta: float):
        """Evolve planetary biosphere if present"""
        # Check habitability
        if self.habitability_index > 0:
            growth_rate = 0.001 * time_delta * self.habitability_index
            
            # Biosphere complexity grows over time
            self.biosphere_complexity = min(1.0, self.biosphere_complexity + growth_rate)
            
            # Chance for civilization emergence
            if (self.biosphere_complexity > 0.8 and 
                'exogenesis' in self.motifs and 
                not self.civilizations and 
                random.random() < 0.001 * time_delta):
                self._spawn_civilization()
                
    def _spawn_civilization(self):
        """Generate a new civilization on this planet"""
        civ = Civilization()
        civ.planet_id = self.entity_id
        civ.home_sector = list(self.sectors)[0] if self.sectors else None
        
        # Set initial properties based on planet type
        civ.traits['adaptations'] = []
        
        # Adapt to planet conditions
        if self.temperature < -10:
            civ.traits['adaptations'].append('cold_resistance')
        if self.temperature > 30:
            civ.traits['adaptations'].append('heat_resistance')
        if self.gravity > 1.3:
            civ.traits['adaptations'].append('high_gravity')
        
        # Set initial development level
        civ.development_level = 0.1  # Stone age
        
        # Register with DRM
        civ.add_to_reality(list(self.sectors))
        
        # Add to planet's civilization list
        self.civilizations.append(civ.entity_id)


class CivilizationType(Enum):
    """Classification of civilization types"""
    TYPE_0 = "pre_industrial"
    TYPE_1 = "planetary"
    TYPE_2 = "stellar"
    TYPE_3 = "galactic"
    TYPE_4 = "cosmic"
    
    
class DevelopmentArea(Enum):
    """Areas of technological development"""
    ENERGY = "energy"
    COMPUTATION = "computation"
    MATERIALS = "materials"
    BIOLOGY = "biology"
    SPACE_TRAVEL = "space_travel"
    WEAPONRY = "weaponry"
    COMMUNICATION = "communication"
    SOCIAL_ORGANIZATION = "social_organization"


class Civilization(CosmicEntity):
    """An intelligent species and its technological development"""
    
    def __init__(self, civ_id: str = None):
        super().__init__(civ_id)
        self.entity_type = EntityType.CIVILIZATION
        self.planet_id = None
        self.civ_type = CivilizationType.TYPE_0
        self.development_level = 0  # 0 to 1 within its type
        self.population = 0
        self.tech_focus = None
        self.tech_levels = {area: 0 for area in DevelopmentArea}
        self.colonized_planets = []
        self.home_sector = None
        self.communication_range = 0
        self.ftl_capability = False
        self.quantum_understanding = 0
        self.known_civilizations = []
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve civilization over time"""
        super().evolve(time_delta)
        
        # Development rate decreases as civilization advances
        # This creates a logistic growth curve
        dev_rate = 0.01 * time_delta * (1 - self.development_level)
        self.development_level = min(1.0, self.development_level + dev_rate)
        
        # Check for civ type transition
        self._check_advancement()
        
        # Expand population
        self._evolve_population(time_delta)
        
        # Advance technology
        self._evolve_technology(time_delta)
        
        # Check for expansion to other planets
        self._check_expansion()
        
        # Check for contact with other civilizations
        if random.random() < 0.01 * time_delta:
            self._check_contact()
            
    def _check_advancement(self):
        """Check if civilization should advance to next type"""
        if (self.development_level >= 0.99):
            if self.civ_type == CivilizationType.TYPE_0:
                self.civ_type = CivilizationType.TYPE_1
                self.development_level = 0.1
                self.motifs.append('industrial_revolution')
            elif self.civ_type == CivilizationType.TYPE_1:
                self.civ_type = CivilizationType.TYPE_2
                self.development_level = 0.1
                self.motifs.append('stellar_expansion')
                self.ftl_capability = True
            elif self.civ_type == CivilizationType.TYPE_2:
                self.civ_type = CivilizationType.TYPE_3
                self.development_level = 0.1
                self.motifs.append('galactic_network')
            # Type 4 is theoretical and extremely rare
    
    def _evolve_population(self, time_delta: float):
        """Evolve population size and distribution"""
        if self.population == 0:
            # Initialize population for new civilization
            self.population = 1000
        else:
            # Logistic growth model
            growth_rate = 0.02 * time_delta
            carrying_capacity = 10**10 * (1 + len(self.colonized_planets))
            
            growth = growth_rate * self.population * (1 - self.population / carrying_capacity)
            self.population = max(1000, int(self.population + growth))
    
    def _evolve_technology(self, time_delta: float):
        """Advance technology levels"""
        # Choose focus if none exists
        if not self.tech_focus:
            self.tech_focus = random.choice(list(DevelopmentArea))
            
        # Advance all tech areas with focus getting bonus
        for area in DevelopmentArea:
            advance_rate = 0.005 * time_delta
            
            # Focus area advances faster
            if area == self.tech_focus:
                advance_rate *= 2
                
            # Higher civilization types advance faster
            advance_rate *= (1 + 0.5 * self.civ_type.value.count('_'))
            
            self.tech_levels[area] = min(1.0, self.tech_levels[area] + advance_rate)
            
        # Occasionally change focus (every ~100 time units)
        if random.random() < 0.01 * time_delta:
            self.tech_focus = random.choice(list(DevelopmentArea))
            
        # Update FTL capability
        if (self.tech_levels[DevelopmentArea.SPACE_TRAVEL] > 0.7 and
            self.tech_levels[DevelopmentArea.ENERGY] > 0.8):
            self.ftl_capability = True
            
        # Update communication range
        self.communication_range = 10 * self.tech_levels[DevelopmentArea.COMMUNICATION]
        if self.ftl_capability:
            self.communication_range *= 100
            
        # Update quantum understanding
        self.quantum_understanding = (
            self.tech_levels[DevelopmentArea.COMPUTATION] * 0.5 +
            self.tech_levels[DevelopmentArea.ENERGY] * 0.3 +
            self.tech_levels[DevelopmentArea.MATERIALS] * 0.2
        )
    
    def _check_expansion(self):
        """Check if civilization should expand to new planets"""
        # Only Type 1+ civs with space travel can expand
        if (self.civ_type.value != CivilizationType.TYPE_0.value and 
            self.tech_levels[DevelopmentArea.SPACE_TRAVEL] > 0.5 and
            random.random() < 0.05 * self.tech_levels[DevelopmentArea.SPACE_TRAVEL]):
            
            # Get home planet
            home_planet = DRM.get_entity(self.planet_id)
            if not home_planet:
                return
                
            # Get star of home planet
            star = DRM.get_entity(home_planet.star_id)
            if not star:
                return
                
            # Search for habitable planets in the star system first
            habitable_candidates = []
            for planet_id in star.planets:
                planet = DRM.get_entity(planet_id)
                if (planet and 
                    planet.entity_id != self.planet_id and
                    planet.entity_id not in self.colonized_planets and
                    planet.habitability_index > 0.3):
                    habitable_candidates.append(planet)
                    
            # If no candidates in home system and civilization has FTL,
            # search nearby star systems
            if not habitable_candidates and self.ftl_capability:
                nearby_stars = self._find_nearby_stars(star, 10)  # 10 light year radius
                
                for nearby_star in nearby_stars:
                    for planet_id in nearby_star.planets:
                        planet = DRM.get_entity(planet_id)
                        if (planet and 
                            planet.habitability_index > 0.4 and  # Higher threshold for interstellar travel
                            planet.entity_id not in self.colonized_planets):
                            habitable_candidates.append(planet)
            
            # Colonize a random candidate
            if habitable_candidates:
                target_planet = random.choice(habitable_candidates)
                
                # Add to colonized planets
                self.colonized_planets.append(target_planet.entity_id)
                
                # Add civilization to planet
                if self.entity_id not in target_planet.civilizations:
                    target_planet.civilizations.append(self.entity_id)
                    
                # Update sectors covered by civilization
                for sector in target_planet.sectors:
                    if sector not in self.sectors:
                        self.sectors.add(sector)
                        
                # Update DRM
                DRM.store_entity(self.entity_id, self, list(self.sectors))
                
                # Add colonization motif
                self.motifs.append('interplanetary_colonization' if target_planet.star_id == star.entity_id 
                                  else 'interstellar_colonization')
    
    def _find_nearby_stars(self, reference_star, max_distance: float) -> List[Star]:
        """Find stars within max_distance light years of reference_star"""
        nearby_stars = []
        ref_pos = reference_star.position
        
        # Get galaxy
        galaxy = DRM.get_entity(reference_star.galaxy_id)
        if not galaxy:
            return nearby_stars
            
        # Query all stars in galaxy
        all_stars = DRM.query_entities(EntityType.STAR.value)
        
        for star in all_stars:
            if star.entity_id == reference_star.entity_id:
                continue
                
            if star.galaxy_id != reference_star.galaxy_id:
                continue
                
            # Calculate distance
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(ref_pos, star.position)))
            
            if distance <= max_distance:
                nearby_stars.append(star)
                
        return nearby_stars
    
    def _check_contact(self):
        """Check for contact with other civilizations"""
        # Get all civilizations
        all_civs = DRM.query_entities(EntityType.CIVILIZATION.value)
        
        for civ in all_civs:
            if civ.entity_id == self.entity_id:
                continue
                
            if civ.entity_id in self.known_civilizations:
                continue
                
            # Check if in communication range
            home_planet = DRM.get_entity(self.planet_id)
            other_home_planet = DRM.get_entity(civ.planet_id)
            
            if not home_planet or not other_home_planet:
                continue
                
            # Get stars
            home_star = DRM.get_entity(home_planet.star_id)
            other_star = DRM.get_entity(other_home_planet.star_id)
            
            if not home_star or not other_star:
                continue
                
            # Calculate distance between stars
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(home_star.position, other_star.position)))
            
            # Check if in range
            civ_range = self.communication_range + civ.communication_range
            
            if distance <= civ_range:
                # Contact established
                self.known_civilizations.append(civ.entity_id)
                civ.known_civilizations.append(self.entity_id)
                
                # Add motif
                contact_type = 'first_contact' if not self.known_civilizations else 'alien_contact'
                self.motifs.append(contact_type)
                civ.motifs.append(contact_type)


class AnomalyType(Enum):
    """Classification of cosmic anomalies"""
    WORMHOLE = "wormhole"
    BLACK_HOLE = "black_hole"
    NEUTRON_STAR = "neutron_star"
    DARK_MATTER_CLOUD = "dark_matter_cloud"
    QUANTUM_FLUCTUATION = "quantum_fluctuation"
    TIME_DILATION = "time_dilation"
    DIMENSIONAL_RIFT = "dimensional_rift"
    COSMIC_STRING = "cosmic_string"
    STRANGE_MATTER = "strange_matter"
    REALITY_BUBBLE = "reality_bubble"


class Anomaly(CosmicEntity):
    """Rare cosmic phenomena with unique properties"""
    
    def __init__(self, anomaly_id: str = None):
        super().__init__(anomaly_id)
        self.entity_type = EntityType.ANOMALY
        self.anomaly_type = None
        self.intensity = 0
        self.stability = 0
        self.radius = 0
        self.effects = []
        self.danger_level = 0
        self.is_traversable = False
        self.connects_to = None  # For wormholes, dimensional rifts
        self.discovery_chance = 0.1
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve anomaly over time"""
        super().evolve(time_delta)
        
        # Stability changes over time
        stability_change = (random.random() - 0.5) * 0.1 * time_delta
        self.stability = max(0, min(1, self.stability + stability_change))
        
        # Check for collapse if unstable
        if self.stability < 0.2 and random.random() < 0.01 * time_delta:
            self._trigger_collapse()
            
        # Intensity fluctuations
        intensity_change = (random.random() - 0.5) * 0.05 * time_delta
        self.intensity = max(0.1, min(1, self.intensity + intensity_change))
        
        # Effect on nearby entities
        self._apply_proximity_effects()
        
    def _trigger_collapse(self):
        """Handle anomaly collapse or transformation"""
        if self.anomaly_type == AnomalyType.WORMHOLE:
            # Wormhole collapse can create a black hole
            if random.random() < 0.3:
                self.anomaly_type = AnomalyType.BLACK_HOLE
                self.intensity *= 2
                self.stability = 0.7
                self.is_traversable = False
                self.connects_to = None
                self.motifs.append('wormhole_collapse')
            else:
                # Complete dissipation
                self._mark_for_removal()
                
        elif self.anomaly_type == AnomalyType.DIMENSIONAL_RIFT:
            # Rift collapse can cause quantum side effects
            self.anomaly_type = AnomalyType.QUANTUM_FLUCTUATION
            self.intensity *= 0.5
            self.stability = 0.4
            self.motifs.append('dimensional_collapse')
            
        elif self.anomaly_type == AnomalyType.REALITY_BUBBLE:
            # Reality bubble collapse can be catastrophic
            self._create_collapse_shockwave()
            self._mark_for_removal()
            
        else:
            # Generic dissipation
            self._mark_for_removal()
            
    def _mark_for_removal(self):
        """Mark anomaly for removal from simulation"""
        # This would be implemented by the reality engine
        # For now we just add a motif to indicate pending removal
        self.motifs.append('marked_for_removal')
        
    def _create_collapse_shockwave(self):
        """Create a shockwave effect when anomaly collapses"""
        # This would normally spawn a temporary effect entity
        # For now we just add the motif
        self.motifs.append('collapse_shockwave')
        
    def _apply_proximity_effects(self):
        """Apply effects to entities near the anomaly"""
        # Get nearby entities
        nearby_entities = {}
        
        for sector in self.sectors:
            for entity_type in EntityType:
                entities = DRM.query_entities(entity_type.value, sector)
                
                for entity in entities:
                    if entity.entity_id != self.entity_id:
                        nearby_entities[entity.entity_id] = entity
        
        # Apply effects based on anomaly type
        for entity in nearby_entities.values():
            self._apply_anomaly_effect(entity)
            
    def _apply_anomaly_effect(self, entity: CosmicEntity):
        """Apply type-specific anomaly effect to an entity"""
        effect_strength = self.intensity * 0.2  # Base effect strength
        
        if self.anomaly_type == AnomalyType.TIME_DILATION:
            # Time flows differently near this anomaly
            if hasattr(entity, 'time_dilation_factor'):
                entity.time_dilation_factor = 1 + (effect_strength * (random.random() - 0.5) * 2)
                
        elif self.anomaly_type == AnomalyType.DARK_MATTER_CLOUD:
            # Affects gravity and energy calculations
            if hasattr(entity, 'mass'):
                entity.effective_mass = entity.mass * (1 + effect_strength)
                
        elif self.anomaly_type == AnomalyType.QUANTUM_FLUCTUATION:
            # Can cause unpredictable behavior or even mutations
            if random.random() < 0.1 * effect_strength:
                if 'quantum_affected' not in entity.motifs:
                    entity.motifs.append('quantum_affected')
                    
                    # Special effect on civilizations: boost quantum understanding
                    if entity.entity_type == EntityType.CIVILIZATION:
                        if hasattr(entity, 'quantum_understanding'):
                            entity.quantum_understanding = min(1.0, entity.quantum_understanding + 0.1)
                
        elif self.anomaly_type == AnomalyType.DIMENSIONAL_RIFT:
            # Can connect to other parts of space or even realities
            if random.random() < 0.05 * effect_strength:
                if 'dimensional_exposure' not in entity.motifs:
                    entity.motifs.append('dimensional_exposure')


class Universe(CosmicEntity):
    """Top-level container for all cosmic entities"""
    
    def __init__(self, universe_id: str = None):
        super().__init__(universe_id)
        self.entity_type = EntityType.UNIVERSE
        self.galaxies = []
        self.age = 0
        self.expansion_rate = 67.8  # Hubble constant
        self.dark_energy_density = OMEGA_LAMBDA
        self.dark_matter_density = 0.2589
        self.baryonic_matter_density = 0.0486
        self.fundamental_constants = {
            'G': G,
            'c': C,
            'h': H,
            'alpha': ALPHA
        }
        self.dimensions = 3
        self.active_sectors = set()
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve universe over time"""
        super().evolve(time_delta)
        self.age += time_delta
        
        # Universe expansion affects distances between galaxies
        self._apply_cosmic_expansion(time_delta)
        
        # Only evolve active sectors to save computational resources
        self._evolve_active_sectors(time_delta)
        
        # Occasionally spawn anomalies
        if random.random() < 0.01 * time_delta:
            self._spawn_anomaly()
            
    def _apply_cosmic_expansion(self, time_delta: float):
        """Apply cosmic expansion to galaxy positions"""
        # Simplified model: expansion increases with distance from origin
        expansion_factor = self.expansion_rate * time_delta / 1000
        
        # Apply to galaxies
        for galaxy_id in self.galaxies:
            galaxy = DRM.get_entity(galaxy_id)
            if galaxy:
                # Scale positions outward from origin
                old_pos = galaxy.position
                new_pos = tuple(p * (1 + expansion_factor * abs(p)) for p in old_pos)
                galaxy.position = new_pos
                
                # Update sectors if needed
                old_sector = tuple(int(p // 100) for p in old_pos)
                new_sector = tuple(int(p // 100) for p in new_pos)
                
                if old_sector != new_sector:
                    DRM.update_entity_sectors(galaxy.entity_id, old_sector, new_sector)
    
    def _evolve_active_sectors(self, time_delta: float):
        """Evolve only the active sectors in the universe"""
        for sector in self.active_sectors:
            # Evolve galaxies in this sector
            galaxies = DRM.query_entities(EntityType.GALAXY.value, sector)
            for galaxy in galaxies:
                galaxy.evolve(time_delta)
                
            # Evolve anomalies in this sector
            anomalies = DRM.query_entities(EntityType.ANOMALY.value, sector)
            for anomaly in anomalies:
                anomaly.evolve(time_delta)
                
    def _spawn_anomaly(self):
        """Randomly spawn a new anomaly in the universe"""
        # Select a random active sector
        if not self.active_sectors:
            return
            
        sector = random.choice(list(self.active_sectors))
        
        # Create a new anomaly
        anomaly = Anomaly()
        anomaly.anomaly_type = random.choice(list(AnomalyType))
        anomaly.intensity = random.uniform(0.3, 1.0)
        anomaly.stability = random.uniform(0.5, 1.0)
        anomaly.radius = random.uniform(0.1, 10.0)  # Light years
        
        # Set danger level based on type and intensity
        if anomaly.anomaly_type in [AnomalyType.BLACK_HOLE, AnomalyType.STRANGE_MATTER]:
            anomaly.danger_level = 0.7 + (0.3 * anomaly.intensity)
        elif anomaly.anomaly_type in [AnomalyType.WORMHOLE, AnomalyType.DIMENSIONAL_RIFT]:
            anomaly.danger_level = 0.5 + (0.4 * anomaly.intensity)
        else:
            anomaly.danger_level = 0.2 + (0.3 * anomaly.intensity)
            
        # Determine if traversable (for wormholes and rifts)
        if anomaly.anomaly_type in [AnomalyType.WORMHOLE, AnomalyType.DIMENSIONAL_RIFT]:
            anomaly.is_traversable = random.random() < 0.3
            
            if anomaly.is_traversable:
                # Connect to another sector
                connected_sector = self._find_connection_target(sector)
                anomaly.connects_to = connected_sector
                
        # Add to reality
        anomaly.sectors.add(sector)
        anomaly.add_to_reality([sector])
        
    def _find_connection_target(self, source_sector: Tuple) -> Tuple:
        """Find a valid target sector for wormhole/rift connections"""
        # Usually connect to a distant part of the same universe
        x, y, z = source_sector
        
        # Random offset at least 10 sectors away
        min_distance = 10
        while True:
            dx = random.randint(-100, 100)
            dy = random.randint(-100, 100)
            dz = random.randint(-100, 100)
            
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            if distance >= min_distance:
                break
                
        return (x + dx, y + dy, z + dz)
    
    def add_galaxy(self, galaxy: Galaxy):
        """Add a galaxy to this universe"""
        galaxy.universe_id = self.entity_id
        self.galaxies.append(galaxy.entity_id)


class GalaxyCluster(CosmicEntity):
    """A collection of gravitationally bound galaxies"""
    
    def __init__(self, cluster_id: str = None):
        super().__init__(cluster_id)
        self.entity_type = EntityType.GALAXY_CLUSTER
        self.galaxies = []
        self.size = 0  # Megaparsecs
        self.mass = 0  # Solar masses
        self.dark_matter_ratio = 0.85
        self.center_position = (0, 0, 0)
        self.universe_id = None
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve galaxy cluster over time"""
        super().evolve(time_delta)
        
        # Gravitational effects on member galaxies
        self._update_galaxy_positions(time_delta)
        
        # Merger chance
        self._check_galaxy_mergers()
        
    def _update_galaxy_positions(self, time_delta: float):
        """Update positions of galaxies based on gravity"""
        # Simplified model: galaxies orbit around cluster center
        # and sometimes move closer or further based on interactions
        
        for galaxy_id in self.galaxies:
            galaxy = DRM.get_entity(galaxy_id)
            if not galaxy:
                continue
                
            # Calculate vector from center to galaxy
            galaxy_pos = galaxy.position
            center = self.center_position
            
            # Vector from center to galaxy
            offset = tuple(g - c for g, c in zip(galaxy_pos, center))
            distance = math.sqrt(sum(o**2 for o in offset))
            
            if distance == 0:
                continue  # At center, no movement
                
            # Normalized direction vector
            direction = tuple(o / distance for o in offset)
            
            # Orbital velocity (decreases with distance)
            orbit_speed = math.sqrt(G * self.mass / distance) * time_delta / 1000
            
            # Perpendicular vector for orbit (simplified to 2D orbit in xy plane)
            orbit_vector = (-direction[1], direction[0], 0)
            
            # Random perturbation
            perturbation = tuple(random.uniform(-0.1, 0.1) for _ in range(3))
            
            # Update position: apply orbital motion and small random shift
            new_pos = tuple(
                g + (orbit_vector[i] * orbit_speed) + (perturbation[i] * orbit_speed / 10)
                for i, g in enumerate(galaxy_pos)
            )
            
            # Update galaxy position
            galaxy.position = new_pos
            
            # Update sectors if needed
            old_sector = tuple(int(p // 100) for p in galaxy_pos)
            new_sector = tuple(int(p // 100) for p in new_pos)
            
            if old_sector != new_sector:
                DRM.update_entity_sectors(galaxy.entity_id, old_sector, new_sector)
    
    def _check_galaxy_mergers(self):
        """Check for and handle galaxy mergers"""
        # This would be a complex calculation in reality
        # Simplified version: check for galaxies that are very close to each other
        
        galaxy_objects = []
        for galaxy_id in self.galaxies:
            galaxy = DRM.get_entity(galaxy_id)
            if galaxy:
                galaxy_objects.append(galaxy)
                
        # Check all pairs of galaxies
        for i, galaxy1 in enumerate(galaxy_objects):
            for galaxy2 in galaxy_objects[i+1:]:
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(galaxy1.position, galaxy2.position)))
                
                # If galaxies are very close, consider a merger
                merger_threshold = (galaxy1.size + galaxy2.size) / 2
                if distance < merger_threshold * 0.8:
                    self._merge_galaxies(galaxy1, galaxy2)
                    break  # Only handle one merger at a time
                    
    def _merge_galaxies(self, galaxy1: Galaxy, galaxy2: Galaxy):
        """Merge two galaxies together"""
        # Create a new merged galaxy
        merged_galaxy = Galaxy()
        
        # Combine properties
        merged_galaxy.galaxy_type = GalaxyType.PECULIAR  # Mergers produce peculiar galaxies
        merged_galaxy.size = galaxy1.size + galaxy2.size
        merged_galaxy.metallicity = (galaxy1.metallicity + galaxy2.metallicity) / 2
        merged_galaxy.black_hole_mass = galaxy1.black_hole_mass + galaxy2.black_hole_mass
        
        # Position at center of mass
        total_mass = (galaxy1.size + galaxy1.black_hole_mass) + (galaxy2.size + galaxy2.black_hole_mass)
        mass1 = galaxy1.size + galaxy1.black_hole_mass
        mass2 = galaxy2.size + galaxy2.black_hole_mass
        
        merged_galaxy.position = tuple(
            (p1 * mass1 + p2 * mass2) / total_mass
            for p1, p2 in zip(galaxy1.position, galaxy2.position)
        )
        
        # Inherit stars from both galaxies
        merged_galaxy.stars = galaxy1.stars + galaxy2.stars
        
        # Update star's galaxy_id
        for star_id in merged_galaxy.stars:
            star = DRM.get_entity(star_id)
            if star:
                star.galaxy_id = merged_galaxy.entity_id
                
        # Add merger motif
        merged_galaxy.motifs.append('galaxy_merger')
        
        # Register with DRM
        sector = tuple(int(p // 100) for p in merged_galaxy.position)
        merged_galaxy.sectors.add(sector)
        merged_galaxy.add_to_reality([sector])
        
        # Add to cluster
        self.galaxies.append(merged_galaxy.entity_id)
        
        # Remove old galaxies from cluster
        self.galaxies.remove(galaxy1.entity_id)
        self.galaxies.remove(galaxy2.entity_id)
        
        # Mark old galaxies for removal
        galaxy1.motifs.append('marked_for_removal')
        galaxy2.motifs.append('marked_for_removal')


class Moon(CosmicEntity):
    """Natural satellite orbiting a planet"""
    
    def __init__(self, moon_id: str = None):
        super().__init__(moon_id)
        self.entity_type = EntityType.MOON
        self.planet_id = None
        self.radius = 0
        self.mass = 0
        self.orbital_distance = 0
        self.orbital_period = 0
        self.rotation_period = 0
        self.surface = {}
        self.habitability_index = 0
        self.has_atmosphere = False
        self.atmosphere = {}
        self.temperature = 0
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve moon over time"""
        super().evolve(time_delta)
        
        # Get parent planet
        planet = DRM.get_entity(self.planet_id)
        if not planet:
            return
            
        # Geological processes
        if random.random() < 0.005 * time_delta:
            self._evolve_geology()
            
        # Tidal effects
        self._apply_tidal_effects(planet, time_delta)
        
    def _evolve_geology(self):
        """Evolve geological features"""
        # Chance for geological activity
        if random.random() < 0.2:
            # Add or modify surface features
            feature = random.choice(['craters', 'plains', 'mountains', 'canyons', 'volcanic'])
            self.surface[feature] = self.surface.get(feature, 0) + random.uniform(0.05, 0.1)
            
            # Normalize percentages
            total = sum(self.surface.values())
            if total > 1:
                for key in self.surface:
                    self.surface[key] /= total
                    
    def _apply_tidal_effects(self, planet: Planet, time_delta: float):
        """Apply tidal effects between moon and planet"""
        # Tidal locking - rotation period gradually matches orbital period
        if self.rotation_period != self.orbital_period:
            adjustment = (self.orbital_period - self.rotation_period) * 0.001 * time_delta
            self.rotation_period += adjustment
            
        # Tidal forces can cause heating in moon's core
        if self.orbital_distance < planet.radius * 5:
            # Close orbits cause stronger tidal heating
            heating = 0.1 * time_delta * (planet.radius * 5 / self.orbital_distance)
            
            # Add volcanic activity if heating is significant
            if heating > 0.05 and random.random() < heating:
                self.surface['volcanic'] = self.surface.get('volcanic', 0) + 0.05
                
                # Normalize percentages
                total = sum(self.surface.values())
                if total > 1:
                    for key in self.surface:
                        self.surface[key] /= total


class Asteroid(CosmicEntity):
    """Small rocky body in space"""
    
    def __init__(self, asteroid_id: str = None):
        super().__init__(asteroid_id)
        self.entity_type = EntityType.ASTEROID
        self.size = 0
        self.mass = 0
        self.composition = {}  # material -> percentage
        self.orbit = None  # Star ID or planet ID
        self.orbital_distance = 0
        self.orbital_period = 0
        self.is_hazardous = False
        self.trajectory = []  # List of positions
        
    @property
    def scroll_id(self):
        return self.entity_id
        
    def evolve(self, time_delta: float):
        """Evolve asteroid over time"""
        super().evolve(time_delta)
        
        # Update position based on orbital parameters
        self._update_position(time_delta)
        
        # Check for collisions
        self._check_collisions()
        
    def _update_position(self, time_delta: float):
        """Update asteroid position based on orbit"""
        # Get the object being orbited
        orbit_center = DRM.get_entity(self.orbit)
        if not orbit_center:
            return
            
        # Calculate new position
        old_pos = self.position
        
        # Simple circular orbit for now
        orbit_angle = (2 * math.pi / self.orbital_period) * time_delta
        
        # Rotation matrix for orbit_angle around z-axis
        cos_theta = math.cos(orbit_angle)
        sin_theta = math.sin(orbit_angle)
        
        # Vector from center to asteroid
        offset = tuple(a - b for a, b in zip(old_pos, orbit_center.position))
        
        # Apply rotation
        new_offset = (
            offset[0] * cos_theta - offset[1] * sin_theta,
            offset[0] * sin_theta + offset[1] * cos_theta,
            offset[2]
        )
        
        # New position
        new_pos = tuple(c + o for c, o in zip(orbit_center.position, new_offset))
        self.position = new_pos
        
        # Record trajectory point
        self.trajectory.append(new_pos)
        if len(self.trajectory) > 10:
            self.trajectory.pop(0)
            
        # Update sectors if needed
        old_sector = tuple(int(p // 100) for p in old_pos)
        new_sector = tuple(int(p // 100) for p in new_pos)
        
        if old_sector != new_sector:
            DRM.update_entity_sectors(self.entity_id, old_sector, new_sector)
    
    def _check_collisions(self):
        """Check for collisions with planets"""
        if not self.is_hazardous:
            return
            
        # Get planets in current sector
        sector = tuple(int(p // 100) for p in self.position)
        planets = DRM.query_entities(EntityType.PLANET.value, sector)
        
        for planet in planets:
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, planet.position)))
            
            # If asteroid is closer than planet radius, collision occurs
            if distance < planet.radius:
                self._handle_collision(planet)
                break
                
    def _handle_collision(self, planet: Planet):
        """Handle asteroid collision with a planet"""
        # Impact effects depend on asteroid size relative to planet
        size_ratio = self.size / planet.radius
        
        if size_ratio < 0.001:  # Tiny asteroid
            # Minimal impact, maybe a small crater
            if 'craters' in planet.surface:
                planet.surface['craters'] += 0.001
            else:
                planet.surface['craters'] = 0.001
            planet.motifs.append('minor_impact')
                
        elif size_ratio < 0.01:  # Small asteroid
            # Notable impact, might affect climate briefly
            if 'craters' in planet.surface:
                planet.surface['craters'] += 0.01
            else:
                planet.surface['craters'] = 0.01
            planet.motifs.append('moderate_impact')
            # Slight climate change
            planet.temperature += random.uniform(-1, 1)
            
        elif size_ratio < 0.1:  # Medium asteroid
            # Significant impact, climate effects
            planet.surface['craters'] = planet.surface.get('craters', 0) + 0.05
            planet.motifs.append('major_impact')
            planet.temperature += random.uniform(-5, 5)
            # Possible mass extinction
            if planet.biosphere_complexity > 0.5:
                planet.biosphere_complexity *= 0.8
                planet.motifs.append('mass_extinction')
                
        else:  # Large asteroid
            # Catastrophic impact
            planet.surface['craters'] = planet.surface.get('craters', 0) + 0.2
            planet.motifs.append('cataclysmic_impact')
            planet.temperature += random.uniform(-10, 10)
            # Drastic reduction in biosphere and civilizations
            planet.biosphere_complexity *= 0.2
            for civ_id in planet.civilizations:
                civ = DRM.get_entity(civ_id)
                if civ:
                    civ.population = max(0, int(civ.population * 0.1))
                    civ.development_level = max(0, civ.development_level - 0.2)
        
        # Mark asteroid for removal
        self.motifs.append('marked_for_removal')