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


# -------------------------------------------------------------------------
# MotifSeeder System
# -------------------------------------------------------------------------
class MotifCategory(Enum):
    """Categories of symbolic motifs that can be applied to entities"""
    PRIMORDIAL = "primordial"  # Ancient/original patterns
    ELEMENTAL = "elemental"    # Related to fundamental forces/elements
    HARMONIC = "harmonic"      # Pattern relationships and resonance
    CHAOTIC = "chaotic"        # Disorder and unpredictability
    LUMINOUS = "luminous"      # Light, radiation, visibility
    SHADOW = "shadow"          # Darkness, obscurity, mystery
    RECURSIVE = "recursive"    # Self-referential patterns
    ASCENDANT = "ascendant"    # Evolution, growth, transcendence
    DIMENSIONAL = "dimensional"  # Spatial and dimensional properties
    TEMPORAL = "temporal"      # Time-related patterns
    VITAL = "vital"            # Life and consciousness
    ENTROPIC = "entropic"      # Decay, dissolution, heat death
    CRYSTALLINE = "crystalline"  # Order, structure, lattice
    ABYSSAL = "abyssal"        # Depth, void, emptiness


class MotifSeeder:
    """
    Symbolic initialization system that assigns thematic motifs to cosmic entities.
    Creates meaningful patterns and relationships between entities.
    """
    
    # Motif pools for different entity types
    _motif_pools = {
        EntityType.UNIVERSE: {
            MotifCategory.PRIMORDIAL: ["first_breath", "genesis_point", "all_potential", "undivided_unity", "original_void"],
            MotifCategory.DIMENSIONAL: ["hyperbolic_manifold", "dimensional_flux", "space_weave", "boundary_condition"],
            MotifCategory.RECURSIVE: ["fractal_seed", "nested_reality", "self_similar_pattern", "recursive_boundary"],
            MotifCategory.TEMPORAL: ["timestream_source", "eternal_now", "temporal_ocean", "omega_point"],
        },
        EntityType.GALAXY: {
            MotifCategory.LUMINOUS: ["starfire_spiral", "radiant_halo", "stellar_tapestry", "cosmic_lighthouse"],
            MotifCategory.CHAOTIC: ["stellar_tempest", "void_turbulence", "dark_flow", "random_scatter"],
            MotifCategory.CRYSTALLINE: ["stellar_lattice", "harmonic_arrangement", "symmetric_pattern"],
            MotifCategory.ENTROPIC: ["heat_death_advance", "dispersal_pattern", "expansion_drift"],
        },
        EntityType.STAR: {
            MotifCategory.ELEMENTAL: ["plasma_heart", "fusion_crucible", "energy_fountain", "elemental_forge"],
            MotifCategory.LUMINOUS: ["light_bearer", "radiation_pulse", "constant_dawn", "warmth_giver"],
            MotifCategory.VITAL: ["life_catalyst", "habitable_anchor", "biosphere_enabler"],
            MotifCategory.ENTROPIC: ["stellar_decay", "supernova_potential", "fuel_consumer"],
        },
        EntityType.PLANET: {
            MotifCategory.ELEMENTAL: ["earth_dreamer", "water_memory", "air_whisper", "fire_core"],
            MotifCategory.CRYSTALLINE: ["tectonic_dreams", "mineral_consciousness", "geometric_growth"],
            MotifCategory.VITAL: ["life_cradle", "sentience_potential", "evolutionary_canvas", "gaia_mind"],
            MotifCategory.RECURSIVE: ["ecosystem_cycle", "weather_patterns", "geological_layers"],
        },
        EntityType.CIVILIZATION: {
            MotifCategory.VITAL: ["collective_consciousness", "dream_weaver", "thought_network", "memory_keeper"],
            MotifCategory.ASCENDANT: ["transcendence_path", "knowledge_seeker", "pattern_recognizer"],
            MotifCategory.RECURSIVE: ["cultural_iteration", "technological_ladder", "historical_cycle"],
            MotifCategory.TEMPORAL: ["future_dreamer", "past_keeper", "now_experiencer", "legacy_builder"],
        },
        EntityType.ANOMALY: {
            MotifCategory.CHAOTIC: ["pattern_breaker", "uncertainty_spike", "probability_storm", "anomalous_field"],
            MotifCategory.DIMENSIONAL: ["space_fold", "reality_tear", "boundary_transgression"],
            MotifCategory.SHADOW: ["dark_secret", "unknown_variable", "void_whisper", "cosmos_blindspot"],
            MotifCategory.ABYSSAL: ["bottomless_depth", "infinite_recursion", "meaning_void", "null_state"],
        },
    }
    
    # Counter-motifs create opposing forces and dramatic tension
    _counter_motifs = {
        "radiant_halo": "void_shadow",
        "fusion_crucible": "cold_silence",
        "light_bearer": "darkness_bringer",
        "life_cradle": "death_harbor",
        "knowledge_seeker": "mystery_keeper",
        "pattern_recognizer": "chaos_embracer",
        "self_similar_pattern": "fractal_disruption",
        "timestream_source": "entropy_sink",
        "dimensional_flux": "spatial_stasis",
    }
    
    # Resonant motifs that reinforce and amplify each other
    _resonant_motifs = {
        "first_breath": ["genesis_point", "all_potential"],
        "stellar_tempest": ["void_turbulence", "dark_flow"],
        "plasma_heart": ["fusion_crucible", "energy_fountain"],
        "tectonic_dreams": ["mineral_consciousness", "geometric_growth"],
        "transcendence_path": ["knowledge_seeker", "pattern_recognizer"],
    }
    
    @classmethod
    def seed_motifs(cls, entity: CosmicEntity, seed_value: int = None):
        """
        Initialize an entity with symbolic motifs based on its type.
        
        Args:
            entity: The cosmic entity to initialize
            seed_value: Optional seed for deterministic motif generation
        
        Returns:
            List of applied motifs
        """
        if not hasattr(entity, 'entity_type'):
            return []
            
        # Use provided seed or entity's hash
        if seed_value is None:
            seed_value = hash(entity.entity_id)
        
        # Seed random generator for deterministic results
        random.seed(seed_value)
        
        # Get motif pool for this entity type
        entity_pools = cls._motif_pools.get(entity.entity_type, {})
        if not entity_pools:
            # Use generic pool if no specific one exists
            entity_pools = {
                MotifCategory.ELEMENTAL: ["generic_pattern", "basic_form", "standard_structure"],
                MotifCategory.CHAOTIC: ["random_element", "unpredictable_aspect", "complex_behavior"],
            }
        
        # Select 2-5 categories based on seed
        available_categories = list(entity_pools.keys())
        num_categories = min(len(available_categories), random.randint(2, 5))
        selected_categories = random.sample(available_categories, num_categories)
        
        # Select 1-3 motifs from each category
        selected_motifs = []
        for category in selected_categories:
            motifs_in_category = entity_pools[category]
            num_motifs = min(len(motifs_in_category), random.randint(1, 3))
            selected_motifs.extend(random.sample(motifs_in_category, num_motifs))
        
        # Add motifs to entity
        entity.motifs = selected_motifs.copy()
        
        # Add counter-motifs with some probability
        cls._add_counter_motifs(entity, seed_value)
        
        # Reset random seed to avoid affecting other random processes
        random.seed()
        
        return entity.motifs
    
    @classmethod
    def _add_counter_motifs(cls, entity: CosmicEntity, seed_value: int):
        """Add opposing motifs for dramatic tension"""
        random.seed(seed_value + 1)  # Different seed from main motifs
        
        for motif in entity.motifs.copy():
            if motif in cls._counter_motifs and random.random() < 0.3:
                counter = cls._counter_motifs[motif]
                entity.motifs.append(counter)
                
    @classmethod
    def find_resonance(cls, entity1: CosmicEntity, entity2: CosmicEntity) -> float:
        """
        Calculate motif resonance between two entities.
        Returns a value between 0 (no resonance) and 1 (perfect resonance).
        """
        if not hasattr(entity1, 'motifs') or not hasattr(entity2, 'motifs'):
            return 0.0
            
        # Direct motif matches
        common_motifs = set(entity1.motifs).intersection(set(entity2.motifs))
        direct_score = len(common_motifs) / max(len(entity1.motifs) + len(entity2.motifs), 1)
        
        # Resonant motif groups
        resonance_score = 0
        for motif1 in entity1.motifs:
            if motif1 in cls._resonant_motifs:
                for resonant in cls._resonant_motifs[motif1]:
                    if resonant in entity2.motifs:
                        resonance_score += 0.5  # Half point for resonant matches
        
        resonance_score = min(1.0, resonance_score / max(len(entity1.motifs) + len(entity2.motifs), 1))
        
        # Counter-motif tension (reduces resonance)
        tension_score = 0
        for motif1 in entity1.motifs:
            if motif1 in cls._counter_motifs and cls._counter_motifs[motif1] in entity2.motifs:
                tension_score += 1
                
        for motif2 in entity2.motifs:
            if motif2 in cls._counter_motifs and cls._counter_motifs[motif2] in entity1.motifs:
                tension_score += 1
                
        tension_score = min(1.0, tension_score / max(len(entity1.motifs) + len(entity2.motifs), 1))
        
        # Combine scores (direct matches + resonance - tension)
        return max(0, min(1.0, direct_score + resonance_score - tension_score))


# -------------------------------------------------------------------------
# ScrollMemory System
# -------------------------------------------------------------------------
class ScrollMemoryEvent:
    """Represents a significant event in an entity's timeline"""
    
    def __init__(self, timestamp: float, event_type: str, description: str, importance: float = 0.5,
                entities_involved: List[str] = None, motifs_added: List[str] = None, location=None):
        self.timestamp = timestamp
        self.event_type = event_type
        self.description = description
        self.importance = importance  # 0.0 to 1.0
        self.entities_involved = entities_involved or []
        self.motifs_added = motifs_added or []
        self.location = location  # Coordinates or sector
        
    def to_dict(self) -> Dict:
        """Convert event to dictionary for storage"""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'description': self.description,
            'importance': self.importance,
            'entities_involved': self.entities_involved,
            'motifs_added': self.motifs_added,
            'location': self.location
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ScrollMemoryEvent':
        """Create event from dictionary"""
        return cls(
            timestamp=data.get('timestamp', 0),
            event_type=data.get('event_type', 'unknown'),
            description=data.get('description', ''),
            importance=data.get('importance', 0.5),
            entities_involved=data.get('entities_involved', []),
            motifs_added=data.get('motifs_added', []),
            location=data.get('location')
        )
    
    def __str__(self) -> str:
        """String representation of event"""
        return f"[T:{self.timestamp:.2f}] {self.event_type}: {self.description}"


class ScrollMemory:
    """
    Memory system that records significant events in an entity's history.
    Provides continuity and emergence of meaningful narratives.
    """
    
    def __init__(self, owner_id: str, capacity: int = 100):
        self.owner_id = owner_id
        self.events = []
        self.capacity = capacity
        self.last_consolidation = 0
        self.thematic_summary = {}  # Event type -> frequency count
        
    def record_event(self, event: ScrollMemoryEvent):
        """
        Add a new event to the memory scroll
        
        Args:
            event: The event to record
        """
        self.events.append(event)
        
        # Update thematic summary
        self.thematic_summary[event.event_type] = self.thematic_summary.get(event.event_type, 0) + 1
        
        # Ensure we don't exceed capacity by consolidating or pruning
        if len(self.events) > self.capacity:
            self._consolidate_memory()
    
    def get_events_by_type(self, event_type: str) -> List[ScrollMemoryEvent]:
        """Retrieve all events of a specific type"""
        return [e for e in self.events if e.event_type == event_type]
    
    def get_events_by_timeframe(self, start_time: float, end_time: float) -> List[ScrollMemoryEvent]:
        """Retrieve events within a specific timeframe"""
        return [e for e in self.events if start_time <= e.timestamp <= end_time]
    
    def get_events_involving_entity(self, entity_id: str) -> List[ScrollMemoryEvent]:
        """Retrieve events involving a specific entity"""
        return [e for e in self.events if entity_id in e.entities_involved]
    
    def get_most_important_events(self, count: int = 10) -> List[ScrollMemoryEvent]:
        """Retrieve the most important events"""
        sorted_events = sorted(self.events, key=lambda e: e.importance, reverse=True)
        return sorted_events[:count]
    
    def get_narrative_arc(self) -> Dict[str, List[ScrollMemoryEvent]]:
        """
        Identify narrative arcs in the entity's history
        Returns a dict mapping arc type to sequence of events
        """
        arcs = {}
        
        # Group events by type
        event_types = set(e.event_type for e in self.events)
        for etype in event_types:
            type_events = sorted(self.get_events_by_type(etype), key=lambda e: e.timestamp)
            if len(type_events) >= 3:  # Minimum 3 events to form an arc
                arcs[etype] = type_events
                
        return arcs
    
    def _consolidate_memory(self):
        """
        Consolidate memory by merging or removing less important events
        This is called when the scroll exceeds capacity
        """
        if len(self.events) <= self.capacity:
            return
            
        # Sort events by importance
        sorted_events = sorted(self.events, key=lambda e: e.importance)
        
        # Identify candidates for consolidation (lowest importance events)
        consolidation_candidates = sorted_events[:len(sorted_events) // 3]
        
        # Group candidates by type and timeframe
        by_type_and_time = {}
        for event in consolidation_candidates:
            time_bucket = int(event.timestamp / 10) * 10  # Group by 10 time units
            key = (event.event_type, time_bucket)
            if key not in by_type_and_time:
                by_type_and_time[key] = []
            by_type_and_time[key].append(event)
        
        # Merge groups that have multiple events
        events_to_remove = []
        for (event_type, _), group in by_type_and_time.items():
            if len(group) > 1:
                # Create a consolidated event
                earliest = min(e.timestamp for e in group)
                importance = max(e.importance for e in group)
                entities = set()
                motifs = set()
                descriptions = []
                
                for e in group:
                    entities.update(e.entities_involved)
                    motifs.update(e.motifs_added)
                    descriptions.append(e.description)
                    events_to_remove.append(e)
                
                consolidated = ScrollMemoryEvent(
                    timestamp=earliest,
                    event_type=event_type,
                    description=f"Multiple events: {'; '.join(descriptions[:3])}",
                    importance=importance,
                    entities_involved=list(entities),
                    motifs_added=list(motifs)
                )
                
                self.events.append(consolidated)
        
        # Remove consolidated events
        for event in events_to_remove:
            self.events.remove(event)
            
        # If still over capacity, remove least important events
        if len(self.events) > self.capacity:
            self.events = sorted(self.events, key=lambda e: e.importance, reverse=True)[:self.capacity]
        
        # Update last consolidation time
        self.last_consolidation = max(e.timestamp for e in self.events) if self.events else 0
        
        # Rebuild thematic summary
        self._rebuild_thematic_summary()
    
    def _rebuild_thematic_summary(self):
        """Rebuild the thematic summary after consolidation"""
        self.thematic_summary = {}
        for event in self.events:
            self.thematic_summary[event.event_type] = self.thematic_summary.get(event.event_type, 0) + 1
    
    def get_memory_keywords(self) -> List[str]:
        """Extract keywords that define this entity's identity based on memory"""
        keywords = []
        
        # Add most frequent event types
        for event_type, count in sorted(self.thematic_summary.items(), key=lambda x: x[1], reverse=True)[:3]:
            keywords.append(event_type)
            
        # Add most common motifs
        motif_count = {}
        for event in self.events:
            for motif in event.motifs_added:
                motif_count[motif] = motif_count.get(motif, 0) + 1
                
        for motif, _ in sorted(motif_count.items(), key=lambda x: x[1], reverse=True)[:3]:
            keywords.append(motif)
            
        return keywords
    
    def generate_timeline_summary(self, max_events: int = 5) -> str:
        """Generate a textual summary of the entity's history"""
        if not self.events:
            return "No recorded history."
        
        # Sort by timestamp
        sorted_events = sorted(self.events, key=lambda e: e.timestamp)
        
        # Select key events based on importance and distribution across time
        events_count = len(sorted_events)
        if events_count <= max_events:
            key_events = sorted_events
        else:
            # Choose some top importance events plus some distributed across time
            by_importance = sorted(sorted_events, key=lambda e: e.importance, reverse=True)
            top_half = max_events // 2
            
            # Take top events by importance
            key_events = by_importance[:top_half]
            
            # Take some distributed across timeline
            time_range = sorted_events[-1].timestamp - sorted_events[0].timestamp
            if time_range > 0:
                step = time_range / (max_events - top_half)
                target_times = [sorted_events[0].timestamp + i * step for i in range(max_events - top_half)]
                
                for target in target_times:
                    closest = min(sorted_events, key=lambda e: abs(e.timestamp - target))
                    if closest not in key_events:
                        key_events.append(closest)
            
            # Sort by timestamp
            key_events = sorted(key_events, key=lambda e: e.timestamp)
        
        # Generate summary text
        summary = []
        for event in key_events:
            summary.append(f"T{event.timestamp:.1f}: {event.description}")
        
        return "\n".join(summary)


# Augment the CosmicEntity class with ScrollMemory
def add_scroll_memory_to_entity(entity: CosmicEntity):
    """Add a ScrollMemory to a cosmic entity if it doesn't already have one"""
    if not hasattr(entity, 'scroll_memory'):
        entity.scroll_memory = ScrollMemory(entity.entity_id)
        return entity.scroll_memory
    return getattr(entity, 'scroll_memory')


# Add scroll_memory attribute and methods to CosmicEntity
def augment_entities_with_scroll_memory():
    """Augment all existing entities with ScrollMemory"""
    for entity_id, entity in DRM.entities.items():
        if isinstance(entity, CosmicEntity) and not hasattr(entity, 'scroll_memory'):
            add_scroll_memory_to_entity(entity)


# Add record_event method to CosmicEntity
def record_scroll_event(self, event_type: str, description: str, importance: float = 0.5,
                      entities_involved: List[str] = None, motifs_added: List[str] = None,
                      location=None, timestamp: float = None):
    """
    Record an event in this entity's memory scroll
    
    Args:
        event_type: Category of event
        description: Text description
        importance: How important this event is (0.0-1.0)
        entities_involved: List of entity IDs involved
        motifs_added: List of motifs related to this event
        location: Where the event occurred
        timestamp: When the event occurred (defaults to current time)
    """
    if not hasattr(self, 'scroll_memory'):
        add_scroll_memory_to_entity(self)
        
    if timestamp is None:
        timestamp = self.last_update_time
        
    # Create the event
    event = ScrollMemoryEvent(
        timestamp=timestamp,
        event_type=event_type,
        description=description,
        importance=importance,
        entities_involved=entities_involved or [],
        motifs_added=motifs_added or [],
        location=location
    )
    
    # Record in scroll memory
    self.scroll_memory.record_event(event)
    
    # If motifs were added, update entity's motifs
    if motifs_added:
        for motif in motifs_added:
            if motif not in self.motifs:
                self.motifs.append(motif)
    
    # For important events, ensure they're shared with related entities
    if importance >= 0.7 and entities_involved:
        self._propagate_event_to_related_entities(event)
            
    return event


def _propagate_event_to_related_entities(self, event: ScrollMemoryEvent):
    """Share important events with related entities"""
    for entity_id in event.entities_involved:
        if entity_id != self.entity_id:
            entity = DRM.get_entity(entity_id)
            if isinstance(entity, CosmicEntity):
                # Create a copy of the event with reduced importance
                related_event = ScrollMemoryEvent(
                    timestamp=event.timestamp,
                    event_type=event.event_type,
                    description=f"Related: {event.description}",
                    importance=event.importance * 0.8,  # Less important for related entities
                    entities_involved=event.entities_involved,
                    motifs_added=event.motifs_added,
                    location=event.location
                )
                
                # Add to other entity's scroll
                if hasattr(entity, 'scroll_memory'):
                    entity.scroll_memory.record_event(related_event)
                else:
                    add_scroll_memory_to_entity(entity).record_event(related_event)


# Bind new methods to CosmicEntity
CosmicEntity.record_scroll_event = record_scroll_event
CosmicEntity._propagate_event_to_related_entities = _propagate_event_to_related_entities


# -------------------------------------------------------------------------
# CultureEngine System
# -------------------------------------------------------------------------
class BeliefType(Enum):
    """Types of beliefs that shape cultural identity"""
    COSMOLOGY = "cosmology"  # Origin/structure of universe
    MORALITY = "morality"    # Right and wrong
    ONTOLOGY = "ontology"    # Nature of being/reality
    EPISTEMOLOGY = "epistemology"  # Knowledge acquisition
    ESCHATOLOGY = "eschatology"  # Ultimate destiny
    AXIOLOGY = "axiology"   # Values and value judgments
    THEOLOGY = "theology"   # Divine/transcendent entities
    TELEOLOGY = "teleology" # Purpose and design


class SocialStructure(Enum):
    """Organizational patterns for civilizations"""
    TRIBAL = "tribal"          # Kinship-based groups
    FEUDAL = "feudal"          # Hierarchical land ownership
    IMPERIAL = "imperial"      # Centralized empire
    REPUBLIC = "republic"      # Representative governance
    DIRECT_DEMOCRACY = "direct_democracy"  # Citizen voting
    TECHNOCRACY = "technocracy"  # Rule by technical experts
    OLIGARCHY = "oligarchy"    # Rule by small group
    HIVE_MIND = "hive_mind"    # Collective consciousness
    DISTRIBUTED = "distributed"  # Decentralized networks
    QUANTUM_CONSENSUS = "quantum_consensus"  # Superposition of choices


class SymbolicArchetype(Enum):
    """Cultural archetypes that appear across civilizations"""
    CREATOR = "creator"        # Creative/generative force
    DESTROYER = "destroyer"    # Destructive/transformative force
    TRICKSTER = "trickster"    # Chaos agent, boundary-crosser
    SAGE = "sage"              # Wisdom figure
    HERO = "hero"              # Challenger of adversity
    RULER = "ruler"            # Authority figure
    CAREGIVER = "caregiver"    # Nurturing force
    EXPLORER = "explorer"      # Seeker of knowledge/territory
    INNOCENT = "innocent"      # Pure, untainted perspective
    SHADOW = "shadow"          # Hidden/suppressed aspects


class CultureEngine:
    """
    Simulates the emergence and evolution of cultures within civilizations,
    including belief systems, archetypes, naming conventions, and social structures.
    """
    
    def __init__(self, civilization: Civilization):
        self.civilization = civilization
        self.belief_systems = {}  # BeliefType -> value (0.0-1.0 scale)
        self.social_structure = None
        self.archetypes = []  # List of active archetypes
        self.naming_patterns = {}  # Category -> pattern
        self.languages = []
        self.values = {}  # Value name -> importance (0.0-1.0)
        self.taboos = []
        self.rituals = []
        self.cultural_motifs = []
        self.cultural_age = 0
        self.adaptations = []
        self.cultural_coherence = 0.8  # How unified the culture is (0.0-1.0)
        self.divergent_subcultures = []  # List of emerging subcultures
        
        # Name components for procedural naming
        self.name_components = {
            'prefixes': [],
            'roots': [],
            'suffixes': []
        }
        
        # Dynamic variables that track cultural changes over time
        self.cultural_shift_momentum = {}  # Aspect -> direction and strength
        self.external_influences = []  # Other cultures influencing this one
        
        # Initialize culture based on civilization traits and planet
        self._initialize_culture()
    
    def _initialize_culture(self):
        """Initialize culture based on civilization traits and environment"""
        # Get planet data for environmental influences
        planet = DRM.get_entity(self.civilization.planet_id) if self.civilization.planet_id else None
        
        # Initialize belief systems with random tendencies
        for belief_type in BeliefType:
            self.belief_systems[belief_type] = random.random()
        
        # Environmental influences on beliefs and values
        if planet:
            # Planet with extreme climate affects cosmology
            if hasattr(planet, 'temperature'):
                if planet.temperature < -30:
                    # Cold world - more structured, ordered cosmology
                    self.belief_systems[BeliefType.COSMOLOGY] = max(0.7, self.belief_systems[BeliefType.COSMOLOGY])
                    self.values['endurance'] = 0.9
                    self.values['community'] = 0.8
                elif planet.temperature > 40:
                    # Hot world - more chaotic cosmology
                    self.belief_systems[BeliefType.COSMOLOGY] = min(0.3, self.belief_systems[BeliefType.COSMOLOGY])
                    self.values['adaptation'] = 0.9
                    self.values['resourcefulness'] = 0.8
            
            # Geological activity affects ontology (nature of being)
            if 'tectonic_dreams' in planet.motifs:
                self.belief_systems[BeliefType.ONTOLOGY] = max(0.6, self.belief_systems[BeliefType.ONTOLOGY])
                self.archetypes.append(SymbolicArchetype.DESTROYER)
            
            # Water worlds influence values
            if hasattr(planet, 'surface') and planet.surface.get('water', 0) > 0.7:
                self.values['flow'] = 0.8
                self.values['depth'] = 0.7
                self.archetypes.append(SymbolicArchetype.EXPLORER)
        
        # Choose initial social structure based on civ development
        self._select_social_structure()
        
        # Select initial archetypes (2-4 primary ones)
        self._initialize_archetypes()
        
        # Generate naming patterns
        self._generate_naming_conventions()
        
        # Initialize cultural motifs (separate from entity motifs)
        self._initialize_cultural_motifs()
    
    def _select_social_structure(self):
        """Select appropriate social structure based on civilization traits"""
        if not self.civilization:
            self.social_structure = SocialStructure.TRIBAL
            return
            
        # Choose based on development level and tech focus
        if self.civilization.development_level < 0.2:
            # Early civilization
            self.social_structure = SocialStructure.TRIBAL
        elif self.civilization.development_level < 0.4:
            # Emerging civilization
            self.social_structure = random.choice([
                SocialStructure.TRIBAL, 
                SocialStructure.FEUDAL
            ])
        elif self.civilization.development_level < 0.6:
            # Established civilization
            if hasattr(self.civilization, 'tech_focus'):
                if self.civilization.tech_focus == DevelopmentArea.SOCIAL_ORGANIZATION:
                    self.social_structure = random.choice([
                        SocialStructure.REPUBLIC,
                        SocialStructure.DIRECT_DEMOCRACY
                    ])
                else:
                    self.social_structure = random.choice([
                        SocialStructure.FEUDAL,
                        SocialStructure.IMPERIAL,
                        SocialStructure.OLIGARCHY
                    ])
        elif self.civilization.development_level < 0.8:
            # Advanced civilization
            if hasattr(self.civilization, 'tech_focus'):
                if self.civilization.tech_focus == DevelopmentArea.COMPUTATION:
                    self.social_structure = SocialStructure.TECHNOCRACY
                elif self.civilization.quantum_understanding > 0.7:
                    self.social_structure = SocialStructure.QUANTUM_CONSENSUS
                else:
                    self.social_structure = random.choice([
                        SocialStructure.REPUBLIC,
                        SocialStructure.DIRECT_DEMOCRACY,
                        SocialStructure.TECHNOCRACY
                    ])
        else:
            # Highly advanced civilization
            if self.civilization.quantum_understanding > 0.9:
                if random.random() < 0.3:
                    self.social_structure = SocialStructure.HIVE_MIND
                else:
                    self.social_structure = SocialStructure.QUANTUM_CONSENSUS
            else:
                self.social_structure = random.choice([
                    SocialStructure.TECHNOCRACY,
                    SocialStructure.DISTRIBUTED
                ])
    
    def _initialize_archetypes(self):
        """Select initial cultural archetypes"""
        # Always include CREATOR archetype
        self.archetypes.append(SymbolicArchetype.CREATOR)
        
        # Select 1-3 additional archetypes
        available_archetypes = [a for a in SymbolicArchetype if a != SymbolicArchetype.CREATOR]
        num_additional = random.randint(1, 3)
        self.archetypes.extend(random.sample(available_archetypes, num_additional))
        
        # Make sure archetypes list contains Enum values, not plain strings
        self.archetypes = [a if isinstance(a, SymbolicArchetype) else a for a in self.archetypes]
    
    def _generate_naming_conventions(self):
        """Generate naming patterns for this culture"""
        # Initialize name components based on culture characteristics
        consonants = 'bcdfghjklmnpqrstvwxyz'
        vowels = 'aeiou'
        
        # Adjust sound palette based on environment and beliefs
        if self.belief_systems.get(BeliefType.COSMOLOGY, 0) > 0.7:
            # Ordered cosmos - more structured names
            consonants = 'kptbdgmnsr'
            vowels = 'aeiou'
        elif self.belief_systems.get(BeliefType.COSMOLOGY, 0) < 0.3:
            # Chaotic cosmos - more varied sounds
            consonants = 'bcdfghjklmnpqrstvwxz'
            vowels = 'aeiouy'
        
        # Generate specific components
        num_prefixes = random.randint(5, 12)
        num_roots = random.randint(10, 20)
        num_suffixes = random.randint(5, 12)
        
        for _ in range(num_prefixes):
            length = random.randint(1, 3)
            prefix = ''
            for i in range(length):
                if i % 2 == 0:
                    prefix += random.choice(consonants)
                else:
                    prefix += random.choice(vowels)
            self.name_components['prefixes'].append(prefix)
            
        for _ in range(num_roots):
            length = random.randint(2, 4)
            root = ''
            start_with = random.choice([0, 1])  # 0: consonant, 1: vowel
            for i in range(length):
                if (i + start_with) % 2 == 0:
                    root += random.choice(consonants)
                else:
                    root += random.choice(vowels)
            self.name_components['roots'].append(root)
            
        for _ in range(num_suffixes):
            length = random.randint(1, 3)
            suffix = ''
            for i in range(length):
                if i % 2 == 0:
                    suffix += random.choice(vowels)
                else:
                    suffix += random.choice(consonants)
            self.name_components['suffixes'].append(suffix)
        
        # Define naming patterns for different entity types
        self.naming_patterns = {
            'person': lambda: self._generate_name('person'),
            'place': lambda: self._generate_name('place'),
            'concept': lambda: self._generate_name('concept'),
            'deity': lambda: self._generate_name('deity')
        }
    
    def _generate_name(self, entity_type: str) -> str:
        """Generate a name based on cultural patterns"""
        result = ''
        
        if entity_type == 'person':
            # Person names can have prefix + root or root + suffix
            if random.random() < 0.5 and self.name_components['prefixes']:
                result += random.choice(self.name_components['prefixes'])
            
            result += random.choice(self.name_components['roots'])
            
            if random.random() < 0.5 and self.name_components['suffixes']:
                result += random.choice(self.name_components['suffixes'])
                
        elif entity_type == 'place':
            # Places often have compound structure: root + root or root + suffix
            result += random.choice(self.name_components['roots'])
            
            if random.random() < 0.7:
                if random.random() < 0.5 and self.name_components['roots']:
                    result += random.choice(self.name_components['roots'])
                elif self.name_components['suffixes']:
                    result += random.choice(self.name_components['suffixes'])
                    
        elif entity_type == 'concept':
            # Concepts can be more abstract: prefix + root or root + suffix
            if random.random() < 0.6 and self.name_components['prefixes']:
                result += random.choice(self.name_components['prefixes'])
                
            result += random.choice(self.name_components['roots'])
            
            if random.random() < 0.6 and self.name_components['suffixes']:
                result += random.choice(self.name_components['suffixes'])
                
        elif entity_type == 'deity':
            # Deities often have grander names: prefix + root + suffix
            if self.name_components['prefixes']:
                result += random.choice(self.name_components['prefixes'])
                
            result += random.choice(self.name_components['roots'])
            
            if self.name_components['suffixes']:
                result += random.choice(self.name_components['suffixes'])
        
        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:]
            
        return result
    
    def _initialize_cultural_motifs(self):
        """Initialize cultural motifs based on environment and beliefs"""
        # Carry over relevant motifs from civilization
        if hasattr(self.civilization, 'motifs'):
            for motif in self.civilization.motifs:
                if any(term in motif for term in ['consciousness', 'mind', 'dream', 'thought', 'memory']):
                    self.cultural_motifs.append(motif)
        
        # Add basic cultural motifs based on social structure
        if self.social_structure == SocialStructure.TRIBAL:
            self.cultural_motifs.extend(['kinship_bonds', 'ancestral_wisdom', 'seasonal_cycles'])
        elif self.social_structure == SocialStructure.FEUDAL:
            self.cultural_motifs.extend(['hierarchical_order', 'loyalty_chains', 'land_connection'])
        elif self.social_structure == SocialStructure.IMPERIAL:
            self.cultural_motifs.extend(['centralized_power', 'expansionist_destiny', 'glory_cult'])
        elif self.social_structure == SocialStructure.REPUBLIC:
            self.cultural_motifs.extend(['collective_wisdom', 'balanced_powers', 'civic_duty'])
        elif self.social_structure == SocialStructure.DIRECT_DEMOCRACY:
            self.cultural_motifs.extend(['voice_of_all', 'consensus_seeking', 'public_discourse'])
        elif self.social_structure == SocialStructure.TECHNOCRACY:
            self.cultural_motifs.extend(['optimization_ideal', 'expertise_value', 'system_thinking'])
        elif self.social_structure == SocialStructure.QUANTUM_CONSENSUS:
            self.cultural_motifs.extend(['probability_thinking', 'superposition_identity', 'entangled_fate'])
        
        # Add motifs based on archetypes
        for archetype in self.archetypes:
            if archetype == SymbolicArchetype.CREATOR:
                self.cultural_motifs.append('genesis_narrative')
            elif archetype == SymbolicArchetype.DESTROYER:
                self.cultural_motifs.append('renewal_through_ending')
            elif archetype == SymbolicArchetype.TRICKSTER:
                self.cultural_motifs.append('wisdom_through_disruption')
            elif archetype == SymbolicArchetype.HERO:
                self.cultural_motifs.append('individual_transcendence')
    
    def evolve_culture(self, time_delta: float):
        """
        Evolve culture over time, responding to internal and external pressures
        
        Args:
            time_delta: Time increment for evolution
        """
        self.cultural_age += time_delta
        
        # Evolve belief systems
        self._evolve_beliefs(time_delta)
        
        # Check for social structure transitions
        self._check_social_structure_transition(time_delta)
        
        # Evolve cultural motifs
        self._evolve_cultural_motifs(time_delta)
        
        # Handle subculture emergence and divergence
        self._handle_cultural_divergence(time_delta)
        
        # Update naming conventions periodically
        if random.random() < 0.05 * time_delta:
            self._evolve_naming_conventions()
        
        # Record significant cultural events
        self._generate_cultural_events(time_delta)
    
    def _evolve_beliefs(self, time_delta: float):
        """Evolve belief systems over time"""
        # Get external influences
        external_pressure = self._calculate_external_pressure()
        
        # Update each belief system
        for belief_type in self.belief_systems:
            # Natural drift - beliefs gradually change
            drift = (random.random() - 0.5) * 0.05 * time_delta
            
            # External influence pushes belief systems
            ext_influence = 0
            if external_pressure.get(belief_type):
                target, strength = external_pressure[belief_type]
                ext_influence = (target - self.belief_systems[belief_type]) * strength * time_delta
            
            # Internal consistency pressure
            internal_pressure = 0
            if belief_type == BeliefType.COSMOLOGY and BeliefType.ONTOLOGY in self.belief_systems:
                # Cosmology and ontology tend to align
                ontology_val = self.belief_systems[BeliefType.ONTOLOGY]
                internal_pressure = (ontology_val - self.belief_systems[belief_type]) * 0.02 * time_delta
            
            # Development level can push certain beliefs
            dev_pressure = 0
            if hasattr(self.civilization, 'development_level'):
                if belief_type == BeliefType.EPISTEMOLOGY:
                    # Higher development pushes toward rational epistemology
                    dev_target = min(0.8, self.civilization.development_level)
                    dev_pressure = (dev_target - self.belief_systems[belief_type]) * 0.03 * time_delta
                    
                elif belief_type == BeliefType.THEOLOGY:
                    # Higher technical development can reduce theological focus
                    if hasattr(self.civilization, 'tech_levels') and 'computation' in self.civilization.tech_levels:
                        comp_level = self.civilization.tech_levels['computation']
                        if comp_level > 0.7:
                            dev_target = max(0.2, 1.0 - comp_level)
                            dev_pressure = (dev_target - self.belief_systems[belief_type]) * 0.02 * time_delta
            
            # Combine all influences
            total_change = drift + ext_influence + internal_pressure + dev_pressure
            
            # Apply change with limits
            self.belief_systems[belief_type] = max(0.01, min(0.99, self.belief_systems[belief_type] + total_change))
    
    def _calculate_external_pressure(self) -> Dict:
        """Calculate external cultural influences"""
        pressure = {}
        
        # Consider influences from known civilizations
        if hasattr(self.civilization, 'known_civilizations'):
            for civ_id in self.civilization.known_civilizations:
                other_civ = DRM.get_entity(civ_id)
                if not other_civ or not hasattr(other_civ, 'culture_engine'):
                    continue
                    
                # Calculate influence strength based on development difference
                dev_diff = other_civ.development_level - self.civilization.development_level
                influence_strength = 0.1  # Base influence
                
                if dev_diff > 0.2:
                    # More advanced civilizations have stronger influence
                    influence_strength += dev_diff * 0.3
                    
                # Add influence for each belief system
                for belief_type, value in other_civ.culture_engine.belief_systems.items():
                    if belief_type not in pressure:
                        pressure[belief_type] = (value, influence_strength)
                    else:
                        # Average with existing influences
                        current_target, current_strength = pressure[belief_type]
                        new_strength = current_strength + influence_strength
                        new_target = (current_target * current_strength + value * influence_strength) / new_strength
                        pressure[belief_type] = (new_target, new_strength)
        
        return pressure
    
    def _check_social_structure_transition(self, time_delta: float):
        """Check if social structure should evolve"""
        if not hasattr(self.civilization, 'development_level'):
            return
            
        current = self.social_structure
        dev_level = self.civilization.development_level
        
        # Development thresholds that might trigger transitions
        if current == SocialStructure.TRIBAL and dev_level > 0.3:
            if random.random() < 0.1 * time_delta:
                self.social_structure = SocialStructure.FEUDAL
                self._record_social_transition(current, self.social_structure)
                
        elif current == SocialStructure.FEUDAL and dev_level > 0.5:
            if random.random() < 0.1 * time_delta:
                choices = [SocialStructure.IMPERIAL, SocialStructure.REPUBLIC]
                self.social_structure = random.choice(choices)
                self._record_social_transition(current, self.social_structure)
                
        elif current == SocialStructure.IMPERIAL and dev_level > 0.7:
            if random.random() < 0.1 * time_delta:
                if hasattr(self.civilization, 'tech_focus') and self.civilization.tech_focus == DevelopmentArea.COMPUTATION:
                    self.social_structure = SocialStructure.TECHNOCRACY
                else:
                    self.social_structure = SocialStructure.REPUBLIC
                self._record_social_transition(current, self.social_structure)
                
        elif current == SocialStructure.REPUBLIC and dev_level > 0.8:
            if random.random() < 0.1 * time_delta:
                choices = [SocialStructure.DIRECT_DEMOCRACY, SocialStructure.TECHNOCRACY]
                self.social_structure = random.choice(choices)
                self._record_social_transition(current, self.social_structure)
                
        elif dev_level > 0.9 and hasattr(self.civilization, 'quantum_understanding'):
            if self.civilization.quantum_understanding > 0.8 and random.random() < 0.1 * time_delta:
                self.social_structure = SocialStructure.QUANTUM_CONSENSUS
                self._record_social_transition(current, self.social_structure)
    
    def _record_social_transition(self, old_structure, new_structure):
        """Record a social structure transition event"""
        if hasattr(self.civilization, 'record_scroll_event'):
            self.civilization.record_scroll_event(
                event_type="social_evolution",
                description=f"Society transformed from {old_structure.value} to {new_structure.value}",
                importance=0.7,
                motifs_added=[f"social_{new_structure.value}"]
            )
    
    def _evolve_cultural_motifs(self, time_delta: float):
        """Evolve cultural motifs over time"""
        # Chance to add new motifs
        if random.random() < 0.1 * time_delta:
            # Potential new motifs based on current state
            potential_motifs = []
            
            # Add motifs based on belief strengths
            for belief_type, value in self.belief_systems.items():
                if value > 0.7:
                    if belief_type == BeliefType.COSMOLOGY:
                        potential_motifs.extend(['cosmic_order', 'celestial_harmony'])
                    elif belief_type == BeliefType.ONTOLOGY:
                        potential_motifs.extend(['being_awareness', 'essence_focus'])
                    elif belief_type == BeliefType.EPISTEMOLOGY:
                        potential_motifs.extend(['truth_seeking', 'knowledge_path'])
                    elif belief_type == BeliefType.THEOLOGY:
                        potential_motifs.extend(['divine_connection', 'transcendent_aspiration'])
            
            # Add motifs based on current social structure
            if self.social_structure == SocialStructure.QUANTUM_CONSENSUS:
                potential_motifs.extend(['quantum_identity', 'probability_ethics'])
            elif self.social_structure == SocialStructure.TECHNOCRACY:
                potential_motifs.extend(['optimization_mandate', 'efficiency_pursuit'])
                
            # Select a new motif if any are available
            if potential_motifs and random.random() < 0.3:
                new_motif = random.choice(potential_motifs)
                if new_motif not in self.cultural_motifs:
                    self.cultural_motifs.append(new_motif)
                    
                    # Record this cultural development
                    if hasattr(self.civilization, 'record_scroll_event'):
                        self.civilization.record_scroll_event(
                            event_type="cultural_development",
                            description=f"Culture developed new motif: {new_motif}",
                            importance=0.5,
                            motifs_added=[new_motif]
                        )
        
        # Chance to lose motifs that no longer fit
        if self.cultural_motifs and random.random() < 0.05 * time_delta:
            # Find motifs that might be lost based on belief shifts
            for motif in self.cultural_motifs[:]:  # Copy to avoid modification during iteration
                # Examples of checking for obsolete motifs
                if motif == 'divine_connection' and self.belief_systems.get(BeliefType.THEOLOGY, 0.5) < 0.3:
                    self.cultural_motifs.remove(motif)
                elif motif == 'cosmic_order' and self.belief_systems.get(BeliefType.COSMOLOGY, 0.5) < 0.3:
                    self.cultural_motifs.remove(motif)
    
    def _handle_cultural_divergence(self, time_delta: float):
        """Handle emergence of subcultures and cultural divergence"""
        # Factors that increase chance of subculture formation:
        # 1. Population size
        # 2. Multiple colonized planets
        # 3. Low cultural coherence
        # 4. High development level
        
        divergence_chance = 0.01 * time_delta  # Base chance
        
        if hasattr(self.civilization, 'population') and self.civilization.population > 1000000:
            divergence_chance += 0.05
            
        if hasattr(self.civilization, 'colonized_planets') and len(self.civilization.colonized_planets) > 1:
            divergence_chance += 0.1 * len(self.civilization.colonized_planets)
            
        divergence_chance *= (2.0 - self.cultural_coherence)  # Lower coherence increases chance
        
        if hasattr(self.civilization, 'development_level'):
            divergence_chance *= (1.0 + self.civilization.development_level)  # Higher development increases chance
            
        # Check for new subculture formation
        if random.random() < divergence_chance:
            self._form_new_subculture()
            
        # Evolve existing subcultures
        for subculture in self.divergent_subcultures:
            subculture['divergence'] = min(1.0, subculture['divergence'] + 0.05 * time_delta)
            
            # Subculture might influence main culture
            if random.random() < 0.1 * time_delta:
                influence_strength = subculture['divergence'] * 0.1
                for belief_type, value in subculture['beliefs'].items():
                    if belief_type in self.belief_systems:
                        self.belief_systems[belief_type] += (value - self.belief_systems[belief_type]) * influence_strength
                
                # Subculture might contribute motifs to main culture
                if random.random() < subculture['divergence'] * 0.2:
                    potential_motifs = [m for m in subculture['motifs'] if m not in self.cultural_motifs]
                    if potential_motifs:
                        self.cultural_motifs.append(random.choice(potential_motifs))
    
    def _form_new_subculture(self):
        """Create a new divergent subculture"""
        # Base new subculture on main culture, but with variations
        new_subculture = {
            'name': self._generate_name('concept'),
            'formation_time': self.cultural_age,
            'divergence': 0.1,  # Starts only slightly different
            'beliefs': {},
            'motifs': [],
            'social_structure': self.social_structure,
            'location': None  # Will be set if specific location exists
        }
        
        # Copy beliefs with variations
        for belief_type, value in self.belief_systems.items():
            variation = (random.random() - 0.5) * 0.4  # -0.2 to 0.2 variation
            new_subculture['beliefs'][belief_type] = max(0.1, min(0.9, value + variation))
            
        # Copy some motifs, add some new ones
        common_motifs = random.sample(self.cultural_motifs, min(len(self.cultural_motifs), 3))
        new_subculture['motifs'].extend(common_motifs)
        
        # Add some unique motifs
        unique_motifs = [
            'identity_seeking',
            'tradition_breaking',
            'path_divergence',
            'alternative_vision',
            'cultural_rebellion'
        ]
        new_subculture['motifs'].append(random.choice(unique_motifs))
        
        # Set location if on multiple planets
        if hasattr(self.civilization, 'colonized_planets') and self.civilization.colonized_planets:
            new_subculture['location'] = random.choice(self.civilization.colonized_planets)
            
        # Add to subcultures list
        self.divergent_subcultures.append(new_subculture)
        
        # Reduce overall cultural coherence
        self.cultural_coherence = max(0.1, self.cultural_coherence - 0.05)
        
        # Record this cultural event
        if hasattr(self.civilization, 'record_scroll_event'):
            self.civilization.record_scroll_event(
                event_type="cultural_divergence",
                description=f"New subculture formed: {new_subculture['name']}",
                importance=0.6,
                motifs_added=['cultural_divergence']
            )
            
        return new_subculture
    
    def _evolve_naming_conventions(self):
        """Evolve naming conventions over time"""
        # Chance to add new components
        component_types = ['prefixes', 'roots', 'suffixes']
        component_type = random.choice(component_types)
        
        # Create a new component
        consonants = 'bcdfghjklmnpqrstvwxyz'
        vowels = 'aeiou'
        
        length = random.randint(1, 3)
        new_component = ''
        for i in range(length):
            if i % 2 == 0:
                new_component += random.choice(consonants)
            else:
                new_component += random.choice(vowels)
                
        self.name_components[component_type].append(new_component)
        
        # Chance to remove old components (but always keep at least 3 of each)
        if len(self.name_components[component_type]) > 5 and random.random() < 0.3:
            removed = random.choice(self.name_components[component_type])
            self.name_components[component_type].remove(removed)
    
    def _generate_cultural_events(self, time_delta: float):
        """Generate cultural events based on current state"""
        # Chance for significant cultural events
        if random.random() < 0.05 * time_delta:
            # Possible event types
            event_types = [
                "artistic_revolution",
                "philosophical_breakthrough",
                "spiritual_awakening",
                "linguistic_evolution",
                "value_shift",
                "ritual_creation",
                "archetype_transformation"
            ]
            
            event_type = random.choice(event_types)
            event_description = ""
            event_importance = 0.5
            event_motifs = []
            
            if event_type == "artistic_revolution":
                art_form = random.choice(["visual", "musical", "literary", "performative", "immersive"])
                movement_name = self._generate_name('concept')
                event_description = f"New artistic movement '{movement_name}' revolutionized {art_form} expression"
                event_motifs = ["artistic_innovation"]
                
            elif event_type == "philosophical_breakthrough":
                philosopher = self._generate_name('person')
                concept = self._generate_name('concept')
                event_description = f"Philosopher {philosopher} introduced concept of '{concept}'"
                event_motifs = ["philosophical_advance"]
                event_importance = 0.7
                
            elif event_type == "spiritual_awakening":
                leader = self._generate_name('person')
                teaching = self._generate_name('concept')
                event_description = f"Spiritual leader {leader} revealed teachings of '{teaching}'"
                event_motifs = ["spiritual_revelation"]
                event_importance = 0.8
                
            elif event_type == "linguistic_evolution":
                language_name = self._generate_name('concept')
                event_description = f"Language evolved into new dialect: {language_name}"
                event_motifs = ["linguistic_drift"]
                
            elif event_type == "value_shift":
                old_value = random.choice(list(self.values.keys())) if self.values else "tradition"
                new_value = self._generate_name('concept')
                event_description = f"Cultural values shifted from '{old_value}' toward '{new_value}'"
                event_motifs = ["value_evolution"]
                # Update values dictionary
                self.values[new_value] = 0.8
                if old_value in self.values:
                    self.values[old_value] = max(0.1, self.values[old_value] - 0.3)
                
            elif event_type == "ritual_creation":
                ritual_name = self._generate_name('concept')
                purpose = random.choice(["harmony", "cleansing", "remembrance", "transition", "union"])
                event_description = f"New ritual '{ritual_name}' established for {purpose}"
                event_motifs = ["ritual_creation"]
                self.rituals.append(ritual_name)
                
            elif event_type == "archetype_transformation":
                old_archetype = random.choice(self.archetypes) if self.archetypes else SymbolicArchetype.CREATOR
                available_archetypes = [a for a in SymbolicArchetype if a not in self.archetypes]
                if available_archetypes:
                    new_archetype = random.choice(available_archetypes)
                    event_description = f"Cultural archetype shifted from {old_archetype.value} to {new_archetype.value}"
                    event_motifs = ["archetype_shift"]
                    # Update archetypes
                    if old_archetype in self.archetypes:
                        self.archetypes.remove(old_archetype)
                    self.archetypes.append(new_archetype)
            
            # Record event in civilization's scroll memory
            if hasattr(self.civilization, 'record_scroll_event') and event_description:
                self.civilization.record_scroll_event(
                    event_type=event_type,
                    description=event_description,
                    importance=event_importance,
                    motifs_added=event_motifs
                )

# Add culture_engine to Civilization class
def _initialize_culture_engine(self):
    """Initialize culture engine for this civilization"""
    if not hasattr(self, 'culture_engine'):
        self.culture_engine = CultureEngine(self)
        
    # Record culture initialization in scroll memory
    if hasattr(self, 'record_scroll_event'):
        self.record_scroll_event(
            event_type="culture_genesis",
            description=f"Culture established with {self.culture_engine.social_structure.value} structure",
            importance=0.8,
            motifs_added=self.culture_engine.cultural_motifs[:3]
        )
    
    return self.culture_engine

# Update evolve method to use culture engine
def _civilization_evolve_with_culture(self, time_delta: float):
    """Evolve civilization with cultural component"""
    # Original evolve method (store reference to original)
    if hasattr(self, '_original_evolve'):
        self._original_evolve(time_delta)
    else:
        super(Civilization, self).evolve(time_delta)
    
    # Ensure culture engine exists
    if not hasattr(self, 'culture_engine'):
        self._initialize_culture_engine()
        
    # Evolve culture
    self.culture_engine.evolve_culture(time_delta)

# Bind culture methods to Civilization
Civilization._initialize_culture_engine = _initialize_culture_engine
Civilization._original_evolve = Civilization.evolve  # Store original method
Civilization.evolve = _civilization_evolve_with_culture  # Replace with new method


# -------------------------------------------------------------------------
# Civilization Interaction System
# -------------------------------------------------------------------------
class InteractionType(Enum):
    """Types of relationships between civilizations"""
    CONFLICT = "conflict"            # Active hostility
    COOPERATION = "cooperation"      # Mutual aid and exchange
    COMPETITION = "competition"      # Non-violent rivalry
    CULTURAL_EXCHANGE = "cultural_exchange"  # Ideas and beliefs flow
    TRADE = "trade"                  # Economic relations
    SUBJUGATION = "subjugation"      # Dominance of one over another
    ISOLATION = "isolation"          # Deliberate separation
    OBSERVATION = "observation"      # One studying the other covertly
    HYBRID = "hybrid"                # Complex mix of relationship types


class CivilizationInteraction:
    """
    Manages relationships and interactions between civilizations,
    including conflict, cooperation, and hybrid states.
    """
    
    def __init__(self, civ1_id: str, civ2_id: str):
        self.civ1_id = civ1_id
        self.civ2_id = civ2_id
        self.relation_id = f"relation_{civ1_id}_{civ2_id}"
        self.interaction_type = InteractionType.OBSERVATION
        self.motif_resonance = 0.0
        self.technological_parity = 0.0
        self.cultural_compatibility = 0.0
        self.tension = 0.0
        self.shared_history = []  # List of historical events
        self.diplomatic_status = "neutral"
        self.treaties = []
        self.war_status = False
        self.trade_volume = 0
        self.last_update_time = 0
        
        # Initialize the relationship
        self._initialize_relationship()
    
    def _initialize_relationship(self):
        """Set initial relationship parameters based on civilizations"""
        civ1 = DRM.get_entity(self.civ1_id)
        civ2 = DRM.get_entity(self.civ2_id)
        
        if not civ1 or not civ2:
            return
            
        # Calculate technological parity (0 = disparate, 1 = equal)
        tech_diff = {}
        for area in DevelopmentArea:
            if hasattr(civ1.tech_levels, area) and hasattr(civ2.tech_levels, area):
                tech_diff[area] = abs(civ1.tech_levels[area] - civ2.tech_levels[area])
                
        if tech_diff:
            avg_tech_diff = sum(tech_diff.values()) / len(tech_diff)
            self.technological_parity = 1.0 - avg_tech_diff
        
        # Calculate motif resonance using MotifSeeder
        self.motif_resonance = MotifSeeder.find_resonance(civ1, civ2)
        
        # Initial cultural compatibility based on belief systems
        if hasattr(civ1, 'culture_engine') and hasattr(civ2, 'culture_engine'):
            belief_compatibility = 0.0
            belief_count = 0
            
            for belief_type in BeliefType:
                if (belief_type in civ1.culture_engine.belief_systems and 
                    belief_type in civ2.culture_engine.belief_systems):
                    diff = abs(civ1.culture_engine.belief_systems[belief_type] - 
                               civ2.culture_engine.belief_systems[belief_type])
                    belief_compatibility += (1.0 - diff)
                    belief_count += 1
                    
            if belief_count > 0:
                self.cultural_compatibility = belief_compatibility / belief_count
        
        # Initialize tension based on inverted compatibility
        self.tension = 0.3 + (1.0 - self.cultural_compatibility) * 0.4 + (1.0 - self.motif_resonance) * 0.3
        
        # Select initial interaction type
        self._update_interaction_type()
    
    def _update_interaction_type(self):
        """Determine interaction type based on current metrics"""
        if self.war_status:
            self.interaction_type = InteractionType.CONFLICT
            return
            
        # Get the civilizations
        civ1 = DRM.get_entity(self.civ1_id)
        civ2 = DRM.get_entity(self.civ2_id)
        
        if not civ1 or not civ2:
            return
            
        # Development gap can lead to subjugation
        dev_gap = abs(civ1.development_level - civ2.development_level)
        if dev_gap > 0.4 and self.tension > 0.6:
            # Significant development gap and high tension
            higher_civ = civ1 if civ1.development_level > civ2.development_level else civ2
            if higher_civ.entity_id == self.civ1_id:
                self.interaction_type = InteractionType.SUBJUGATION
                return
                
        # High tension leads to conflict or competition
        if self.tension > 0.7:
            if self.technological_parity > 0.8:
                # Similar tech levels mean direct conflict is more likely
                self.interaction_type = InteractionType.CONFLICT
            else:
                # Different tech levels lead to competition
                self.interaction_type = InteractionType.COMPETITION
            return
            
        # High cultural compatibility encourages cooperation or exchange
        if self.cultural_compatibility > 0.7:
            if self.technological_parity > 0.6:
                # Similar tech and culture leads to full cooperation
                self.interaction_type = InteractionType.COOPERATION
            else:
                # Cultural similarity but tech difference leads to cultural exchange
                self.interaction_type = InteractionType.CULTURAL_EXCHANGE
            return
            
        # High tech parity and moderate cultural compatibility can lead to trade
        if self.technological_parity > 0.6 and self.cultural_compatibility > 0.4:
            self.interaction_type = InteractionType.TRADE
            return
            
        # Low resonance and low tension can lead to isolation
        if self.motif_resonance < 0.3 and self.tension < 0.4:
            self.interaction_type = InteractionType.ISOLATION
            return
            
        # Complex mix of factors
        if (0.3 < self.motif_resonance < 0.7 and 
            0.3 < self.technological_parity < 0.7 and
            0.3 < self.cultural_compatibility < 0.7):
            self.interaction_type = InteractionType.HYBRID
            return
            
        # Default to observation
        self.interaction_type = InteractionType.OBSERVATION
    
    def update_relationship(self, time_delta: float):
        """Update the relationship between civilizations over time"""
        self.last_update_time += time_delta
        
        # Get the civilizations
        civ1 = DRM.get_entity(self.civ1_id)
        civ2 = DRM.get_entity(self.civ2_id)
        
        if not civ1 or not civ2:
            return
            
        # Update tech parity
        tech_diff = {}
        for area in DevelopmentArea:
            if hasattr(civ1.tech_levels, area) and hasattr(civ2.tech_levels, area):
                tech_diff[area] = abs(civ1.tech_levels[area] - civ2.tech_levels[area])
                
        if tech_diff:
            avg_tech_diff = sum(tech_diff.values()) / len(tech_diff)
            self.technological_parity = 1.0 - avg_tech_diff
            
        # Cultural compatibility evolves based on current interaction
        if hasattr(civ1, 'culture_engine') and hasattr(civ2, 'culture_engine'):
            if self.interaction_type == InteractionType.CULTURAL_EXCHANGE:
                # Cultural exchange increases compatibility
                self.cultural_compatibility = min(1.0, self.cultural_compatibility + 0.05 * time_delta)
            elif self.interaction_type == InteractionType.CONFLICT:
                # Conflict decreases compatibility
                self.cultural_compatibility = max(0.0, self.cultural_compatibility - 0.05 * time_delta)
                
        # Tension evolves based on interaction type and random factors
        tension_change = 0.0
        
        if self.interaction_type == InteractionType.CONFLICT:
            tension_change = 0.05  # Conflicts increase tension
        elif self.interaction_type == InteractionType.COOPERATION:
            tension_change = -0.05  # Cooperation decreases tension
        elif self.interaction_type == InteractionType.TRADE:
            tension_change = -0.02  # Trade slightly decreases tension
        
        # Random factors
        tension_change += (random.random() - 0.5) * 0.04 * time_delta
        
        # Apply tension change
        self.tension = max(0.1, min(0.9, self.tension + tension_change))
        
        # Check for significant events
        if random.random() < 0.1 * time_delta:
            self._generate_interaction_event()
            
        # Re-evaluate interaction type
        self._update_interaction_type()
        
        # Update each civilization based on interaction
        self._apply_interaction_effects(time_delta)
    
    def _generate_interaction_event(self):
        """Generate a significant event in the relationship"""
        civ1 = DRM.get_entity(self.civ1_id)
        civ2 = DRM.get_entity(self.civ2_id)
        
        if not civ1 or not civ2:
            return
            
        event_desc = ""
        event_importance = 0.5
        event_motifs = []
        
        # Different events based on interaction type
        if self.interaction_type == InteractionType.CONFLICT:
            if random.random() < 0.3:
                # Major conflict
                event_desc = f"War broke out between {civ1.entity_id} and {civ2.entity_id}"
                event_importance = 0.9
                event_motifs = ["interspecies_war"]
                self.war_status = True
            else:
                # Minor conflict
                event_desc = f"Border skirmish between {civ1.entity_id} and {civ2.entity_id}"
                event_importance = 0.7
                event_motifs = ["territorial_dispute"]
                
        elif self.interaction_type == InteractionType.COOPERATION:
            if random.random() < 0.3:
                # Major alliance
                event_desc = f"Alliance formed between {civ1.entity_id} and {civ2.entity_id}"
                event_importance = 0.8
                event_motifs = ["interspecies_alliance"]
                self.treaties.append(("Alliance", self.last_update_time))
            else:
                # Joint project
                event_desc = f"Joint technological project between {civ1.entity_id} and {civ2.entity_id}"
                event_importance = 0.6
                event_motifs = ["technological_cooperation"]
                
        elif self.interaction_type == InteractionType.TRADE:
            event_desc = f"Trade agreement established between {civ1.entity_id} and {civ2.entity_id}"
            event_importance = 0.5
            event_motifs = ["interspecies_commerce"]
            self.trade_volume += 0.2
            self.treaties.append(("Trade Agreement", self.last_update_time))
            
        elif self.interaction_type == InteractionType.CULTURAL_EXCHANGE:
            event_desc = f"Cultural exchange program between {civ1.entity_id} and {civ2.entity_id}"
            event_importance = 0.6
            event_motifs = ["cultural_transmission"]
            
        elif self.interaction_type == InteractionType.SUBJUGATION:
            # Determine which civilization is dominant
            dominant = civ1 if civ1.development_level > civ2.development_level else civ2
            subjugated = civ2 if dominant == civ1 else civ1
            
            event_desc = f"{dominant.entity_id} established dominance over {subjugated.entity_id}"
            event_importance = 0.8
            event_motifs = ["power_hierarchy", "empire_building"]
            
        # Record the event in both civilizations' scroll memories
        if event_desc:
            if hasattr(civ1, 'record_scroll_event'):
                civ1.record_scroll_event(
                    event_type="diplomacy",
                    description=event_desc,
                    importance=event_importance,
                    entities_involved=[civ1.entity_id, civ2.entity_id],
                    motifs_added=event_motifs
                )
                
            if hasattr(civ2, 'record_scroll_event'):
                civ2.record_scroll_event(
                    event_type="diplomacy",
                    description=event_desc,
                    importance=event_importance,
                    entities_involved=[civ1.entity_id, civ2.entity_id],
                    motifs_added=event_motifs
                )
                
            # Add to shared history
            self.shared_history.append({
                'time': self.last_update_time,
                'description': event_desc,
                'type': self.interaction_type.value,
                'importance': event_importance
            })
    
    def _apply_interaction_effects(self, time_delta: float):
        """Apply effects of the relationship to both civilizations"""
        civ1 = DRM.get_entity(self.civ1_id)
        civ2 = DRM.get_entity(self.civ2_id)
        
        if not civ1 or not civ2:
            return
            
        # Different effects based on interaction type
        if self.interaction_type == InteractionType.COOPERATION:
            # Boost tech in areas where the other civ is stronger
            for area in DevelopmentArea:
                if (hasattr(civ1.tech_levels, area) and 
                    hasattr(civ2.tech_levels, area)):
                    # Civ 1 learns from Civ 2 in areas where Civ 2 is stronger
                    if civ2.tech_levels[area] > civ1.tech_levels[area]:
                        boost = (civ2.tech_levels[area] - civ1.tech_levels[area]) * 0.1 * time_delta
                        civ1.tech_levels[area] = min(1.0, civ1.tech_levels[area] + boost)
                    
                    # Civ 2 learns from Civ 1
                    if civ1.tech_levels[area] > civ2.tech_levels[area]:
                        boost = (civ1.tech_levels[area] - civ2.tech_levels[area]) * 0.1 * time_delta
                        civ2.tech_levels[area] = min(1.0, civ2.tech_levels[area] + boost)
            
        elif self.interaction_type == InteractionType.CONFLICT:
            # War pushes military technology but drains resources
            if hasattr(civ1.tech_levels, DevelopmentArea.WEAPONRY):
                civ1.tech_levels[DevelopmentArea.WEAPONRY] = min(1.0, civ1.tech_levels[DevelopmentArea.WEAPONRY] + 0.05 * time_delta)
                
            if hasattr(civ2.tech_levels, DevelopmentArea.WEAPONRY):
                civ2.tech_levels[DevelopmentArea.WEAPONRY] = min(1.0, civ2.tech_levels[DevelopmentArea.WEAPONRY] + 0.05 * time_delta)
                
            # Population and development penalties
            civ1.population = max(1000, int(civ1.population * (1 - 0.02 * time_delta)))
            civ2.population = max(1000, int(civ2.population * (1 - 0.02 * time_delta)))
            
            # Check for war resolution
            if random.random() < 0.05 * time_delta:
                self._resolve_conflict()
                
        elif self.interaction_type == InteractionType.CULTURAL_EXCHANGE:
            # Exchange cultural motifs
            if hasattr(civ1, 'culture_engine') and hasattr(civ2, 'culture_engine'):
                if random.random() < 0.1 * time_delta:
                    # Civ 1 learns from Civ 2
                    if civ2.culture_engine.cultural_motifs:
                        motif = random.choice(civ2.culture_engine.cultural_motifs)
                        if motif not in civ1.culture_engine.cultural_motifs:
                            civ1.culture_engine.cultural_motifs.append(motif)
                    
                    # Civ 2 learns from Civ 1
                    if civ1.culture_engine.cultural_motifs:
                        motif = random.choice(civ1.culture_engine.cultural_motifs)
                        if motif not in civ2.culture_engine.cultural_motifs:
                            civ2.culture_engine.cultural_motifs.append(motif)
                            
                # Belief systems influence each other
                for belief_type in BeliefType:
                    if (belief_type in civ1.culture_engine.belief_systems and 
                        belief_type in civ2.culture_engine.belief_systems):
                        # Both belief systems drift toward each other
                        civ1_belief = civ1.culture_engine.belief_systems[belief_type]
                        civ2_belief = civ2.culture_engine.belief_systems[belief_type]
                        
                        drift = (civ2_belief - civ1_belief) * 0.05 * time_delta
                        civ1.culture_engine.belief_systems[belief_type] += drift
                        civ2.culture_engine.belief_systems[belief_type] -= drift
        
        elif self.interaction_type == InteractionType.TRADE:
            # Trade boosts economies and certain technologies
            boost = 0.02 * time_delta * self.trade_volume
            
            # Population growth boost
            civ1.population = int(civ1.population * (1 + 0.01 * boost))
            civ2.population = int(civ2.population * (1 + 0.01 * boost))
            
            # Technology boost in energy, materials, computation
            for area in [DevelopmentArea.ENERGY, DevelopmentArea.MATERIALS, DevelopmentArea.COMPUTATION]:
                if hasattr(civ1.tech_levels, area):
                    civ1.tech_levels[area] = min(1.0, civ1.tech_levels[area] + 0.01 * boost)
                if hasattr(civ2.tech_levels, area):
                    civ2.tech_levels[area] = min(1.0, civ2.tech_levels[area] + 0.01 * boost)
                    
        elif self.interaction_type == InteractionType.SUBJUGATION:
            # Determine dominant civilization
            dominant = civ1 if civ1.development_level > civ2.development_level else civ2
            subjugated = civ2 if dominant == civ1 else civ1
            
            # Resource flow from subjugated to dominant
            # Dominant grows faster, subjugated grows slower
            dominant.population = int(dominant.population * (1 + 0.02 * time_delta))
            subjugated.population = max(1000, int(subjugated.population * (1 - 0.01 * time_delta)))
            
            # Technology transfer (one-way)
            for area in DevelopmentArea:
                if (hasattr(dominant.tech_levels, area) and 
                    hasattr(subjugated.tech_levels, area)):
                    if dominant.tech_levels[area] > subjugated.tech_levels[area]:
                        # Slow tech transfer to subjugated
                        boost = (dominant.tech_levels[area] - subjugated.tech_levels[area]) * 0.05 * time_delta
                        subjugated.tech_levels[area] = min(dominant.tech_levels[area] * 0.8, 
                                                         subjugated.tech_levels[area] + boost)
    
    def _resolve_conflict(self):
        """Resolve an ongoing conflict between civilizations"""
        civ1 = DRM.get_entity(self.civ1_id)
        civ2 = DRM.get_entity(self.civ2_id)
        
        if not civ1 or not civ2:
            return
            
        # Determine victor based on multiple factors
        # - Military technology
        # - Population
        # - Overall development
        # - Random chance
        
        # Calculate military strength
        mil_tech1 = civ1.tech_levels.get(DevelopmentArea.WEAPONRY, 0.1)
        mil_tech2 = civ2.tech_levels.get(DevelopmentArea.WEAPONRY, 0.1)
        
        # Population factor (logarithmic to avoid pure numbers determining outcome)
        pop_factor1 = math.log10(max(1000, civ1.population)) / 10
        pop_factor2 = math.log10(max(1000, civ2.population)) / 10
        
        # Overall development
        dev1 = civ1.development_level
        dev2 = civ2.development_level
        
        # Calculate total strength
        strength1 = mil_tech1 * 0.5 + pop_factor1 * 0.3 + dev1 * 0.2
        strength2 = mil_tech2 * 0.5 + pop_factor2 * 0.3 + dev2 * 0.2
        
        # Add random factor (fog of war)
        strength1 *= random.uniform(0.8, 1.2)
        strength2 *= random.uniform(0.8, 1.2)
        
        # Determine victor
        victor = civ1 if strength1 > strength2 else civ2
        defeated = civ2 if victor == civ1 else civ1
        
        # War resolution effects
        self.war_status = False
        
        # Victor gains, defeated loses
        if victor.development_level > 0.7 and defeated.development_level > 0.7:
            # Advanced civilizations sign peace treaty
            event_desc = f"Peace treaty signed between {civ1.entity_id} and {civ2.entity_id}"
            self.treaties.append(("Peace Treaty", self.last_update_time))
            
            # Reset tension
            self.tension = 0.4
            
            # Both lose some population
            victor.population = int(victor.population * 0.95)
            defeated.population = int(defeated.population * 0.9)
            
        elif victor.development_level - defeated.development_level > 0.3:
            # Significant tech gap leads to subjugation
            event_desc = f"{victor.entity_id} defeated and subjugated {defeated.entity_id}"
            
            # Set interaction to subjugation
            self.interaction_type = InteractionType.SUBJUGATION
            
            # Population effects
            victor.population = int(victor.population * 0.97)
            defeated.population = int(defeated.population * 0.7)
            
        else:
            # Standard victory
            event_desc = f"{victor.entity_id} defeated {defeated.entity_id} in war"
            
            # Population effects
            victor.population = int(victor.population * 0.95)
            defeated.population = int(defeated.population * 0.8)
            
            # Reset tension
            self.tension = 0.5
            
        # Record event in both civilizations' scroll memories
        if hasattr(civ1, 'record_scroll_event'):
            civ1.record_scroll_event(
                event_type="war_resolution",
                description=event_desc,
                importance=0.8,
                entities_involved=[civ1.entity_id, civ2.entity_id],
                motifs_added=["war_conclusion"]
            )
            
        if hasattr(civ2, 'record_scroll_event'):
            civ2.record_scroll_event(
                event_type="war_resolution",
                description=event_desc,
                importance=0.8,
                entities_involved=[civ1.entity_id, civ2.entity_id],
                motifs_added=["war_conclusion"]
            )
            
        # Add to shared history
        self.shared_history.append({
            'time': self.last_update_time,
            'description': event_desc,
            'type': 'war_resolution',
            'importance': 0.8
        })


# Central registry for civilization interactions
class DiplomaticRegistry:
    """Manages all civilization interactions in the simulation"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DiplomaticRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the diplomatic registry"""
        self.relationships = {}  # (civ1_id, civ2_id) -> CivilizationInteraction
        self.alliances = {}  # alliance_id -> [civ_ids]
        self.wars = {}  # war_id -> [civ_ids, start_time, cause]
        self.trade_networks = {}  # network_id -> [civ_ids, trade_volume]
    
    def get_relationship(self, civ1_id: str, civ2_id: str) -> CivilizationInteraction:
        """Get or create relationship between two civilizations"""
        # Sort IDs to ensure consistent key
        if civ1_id > civ2_id:
            civ1_id, civ2_id = civ2_id, civ1_id
            
        key = (civ1_id, civ2_id)
        
        if key not in self.relationships:
            self.relationships[key] = CivilizationInteraction(civ1_id, civ2_id)
            
        return self.relationships[key]
    
    def update_all_relationships(self, time_delta: float):
        """Update all diplomatic relationships"""
        for relationship in self.relationships.values():
            relationship.update_relationship(time_delta)
            
        # Update meta-structures like alliances and trade networks
        self._update_diplomatic_structures()
    
    def _update_diplomatic_structures(self):
        """Update alliance networks, trade blocs, etc."""
        # Reset structures
        self.alliances = {}
        self.trade_networks = {}
        
        # Rebuild based on current relationships
        processed = set()
        
        for (civ1_id, civ2_id), relationship in self.relationships.items():
            if (civ1_id, civ2_id) in processed:
                continue
                
            processed.add((civ1_id, civ2_id))
            
            # Check for alliances
            if relationship.interaction_type == InteractionType.COOPERATION:
                # Find existing alliance to add to
                alliance_id = None
                for aid, members in self.alliances.items():
                    if civ1_id in members or civ2_id in members:
                        alliance_id = aid
                        break
                        
                if alliance_id:
                    # Add to existing alliance
                    if civ1_id not in self.alliances[alliance_id]:
                        self.alliances[alliance_id].append(civ1_id)
                    if civ2_id not in self.alliances[alliance_id]:
                        self.alliances[alliance_id].append(civ2_id)
                else:
                    # Create new alliance
                    alliance_id = f"alliance_{civ1_id}_{civ2_id}"
                    self.alliances[alliance_id] = [civ1_id, civ2_id]
                    
            # Check for trade networks
            elif relationship.interaction_type == InteractionType.TRADE:
                # Find existing trade network
                network_id = None
                for nid, data in self.trade_networks.items():
                    members = data['members']
                    if civ1_id in members or civ2_id in members:
                        network_id = nid
                        break
                        
                if network_id:
                    # Add to existing network
                    if civ1_id not in self.trade_networks[network_id]['members']:
                        self.trade_networks[network_id]['members'].append(civ1_id)
                    if civ2_id not in self.trade_networks[network_id]['members']:
                        self.trade_networks[network_id]['members'].append(civ2_id)
                    # Update volume
                    self.trade_networks[network_id]['volume'] += relationship.trade_volume
                else:
                    # Create new trade network
                    network_id = f"trade_{civ1_id}_{civ2_id}"
                    self.trade_networks[network_id] = {
                        'members': [civ1_id, civ2_id],
                        'volume': relationship.trade_volume
                    }
                    
        # Update wars
        active_wars = {}
        for (civ1_id, civ2_id), relationship in self.relationships.items():
            if relationship.war_status:
                # Find existing war
                war_id = None
                for wid, data in self.wars.items():
                    if civ1_id in data['members'] and civ2_id in data['members']:
                        war_id = wid
                        break
                        
                if not war_id:
                    # New war
                    war_id = f"war_{civ1_id}_{civ2_id}"
                    history = relationship.shared_history
                    cause = "Unknown"
                    
                    # Find war cause in history
                    for event in reversed(history):
                        if event['type'] == InteractionType.CONFLICT.value:
                            cause = event['description']
                            break
                            
                    active_wars[war_id] = {
                        'members': [civ1_id, civ2_id],
                        'start_time': relationship.last_update_time,
                        'cause': cause
                    }
                else:
                    # Existing war
                    active_wars[war_id] = self.wars[war_id]
                    
        self.wars = active_wars
    
    def get_civilization_conflicts(self, civ_id: str) -> List[Dict]:
        """Get all conflicts a civilization is involved in"""
        conflicts = []
        
        for war_id, data in self.wars.items():
            if civ_id in data['members']:
                conflicts.append({
                    'war_id': war_id,
                    'opponents': [m for m in data['members'] if m != civ_id],
                    'start_time': data['start_time'],
                    'cause': data['cause']
                })
                
        return conflicts
    
    def get_civilization_alliances(self, civ_id: str) -> List[Dict]:
        """Get all alliances a civilization is part of"""
        alliances = []
        
        for alliance_id, members in self.alliances.items():
            if civ_id in members:
                alliances.append({
                    'alliance_id': alliance_id,
                    'members': [m for m in members if m != civ_id]
                })
                
        return alliances
    
    def get_civilization_trade_partners(self, civ_id: str) -> List[Dict]:
        """Get all trade partners of a civilization"""
        partners = []
        
        for network_id, data in self.trade_networks.items():
            if civ_id in data['members']:
                partners.append({
                    'network_id': network_id,
                    'partners': [m for m in data['members'] if m != civ_id],
                    'volume': data['volume']
                })
                
        return partners


# Initialize the diplomatic registry singleton
DIPLOMATIC_REGISTRY = DiplomaticRegistry()