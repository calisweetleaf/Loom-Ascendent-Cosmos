# ================================================================
#  LOOM ASCENDANT COSMOS — RECURSIVE SYSTEM MODULE
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
#  Integrity Hash (SHA-256): d3ab9688a5a20b8065990cd9b91805e3d892d6e72472f69dd9afe719250c5e37
# ================================================================
import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.sparse import csr_matrix, diags, kron, eye
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.linalg import eigh
from numba import cuda, njit, prange
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import pickle
import time
import warnings
from tqdm import tqdm
import logging
import importlib.util
import sys

# -------------------------------------------------------------------------
# Custom Exceptions
# -------------------------------------------------------------------------
class QuantumDecoherenceError(Exception):
    """Exception raised when quantum coherence falls below critical threshold causing decoherence."""
    def __init__(self, message="Quantum state coherence failure detected", coherence_value=None, 
                affected_patterns=None, location=None):
        self.message = message
        self.coherence_value = coherence_value
        self.affected_patterns = affected_patterns or []
        self.location = location
        super().__init__(self.message)
        
    def __str__(self):
        details = []
        if self.coherence_value is not None:
            details.append(f"Coherence value: {self.coherence_value:.6f}")
            
        if self.affected_patterns:
            details.append(f"Affected patterns: {', '.join(str(p) for p in self.affected_patterns)}")
            
        if self.location:
            details.append(f"Location: {self.location}")
        
        if details:
            return f"{self.message} - {'; '.join(details)}"
        return self.message

# -------------------------------------------------------------------------
# Constants and Configuration
# -------------------------------------------------------------------------
class PhysicsConstants:
    """Physical constants in natural units (ħ = c = 1)"""
    def __init__(self):
        self.G = 6.67430e-11  # Gravitational constant
        self.hbar = 1.0       # Reduced Planck constant (natural units)
        self.c = 1.0          # Speed of light (natural units)
        self.eps0 = 8.85418782e-12  # Vacuum permittivity
        self.k_B = 1.380649e-23     # Boltzmann constant
        self.alpha = 1/137.035999084  # Fine structure constant
        self.m_e = 9.1093837e-31     # Electron mass
        
        # Planck scale
        self.l_p = 1.616255e-35     # Planck length
        self.t_p = 5.391247e-44     # Planck time
        self.m_p = 2.176434e-8      # Planck mass

# Configuration for simulation parameters
class SimulationConfig:
    def __init__(self):
        self.grid_resolution = 64
        self.spatial_dim = 3
        self.temporal_resolution = 0.01
        self.total_time = 10.0
        
        # QFT parameters
        self.mass = 0.1
        self.coupling = 0.5
        self.vacuum_energy = 1e-6
        
        # Monte Carlo parameters
        self.metropolis_steps = 1000
        self.thermalization_steps = 100
        self.correlation_length = 5
        
        # Numerical parameters
        self.convergence_tolerance = 1e-6
        self.max_iterations = 1000
        self.adaptive_step_size = True
        
        # GPU parameters
        self.use_gpu = cuda.is_available()
        self.block_size = 256
        self.stream_count = 4
        
        # Ethical parameters
        self.ethical_dim = 5
        self.ethical_init = np.array([0.8, -0.2, 0.5, 0.1, -0.4])
        self.ethical_coupling = 0.1
        
        # Output & visualization
        self.save_frequency = 10
        self.output_dir = "./quantum_sim_results"
        self.verbose = True

# -------------------------------------------------------------------------
# Quantum Field Theory Core
# -------------------------------------------------------------------------
class QuantumField:
    """Base class for quantum fields with dynamical evolution"""
    def __init__(self, config):
        self.config = config
        self.constants = PhysicsConstants()
        self.grid_shape = (config.grid_resolution,) * config.spatial_dim
        self.lattice_spacing = 1.0 / config.grid_resolution
        self.dtype = np.float64  # Define the dtype attribute here

        # Initialize field configurations
        self.field = np.zeros(self.grid_shape, dtype=np.complex128)
        self.conjugate_momentum = np.zeros(self.grid_shape, dtype=np.complex128)
        self.potential = np.zeros(self.grid_shape, dtype=self.dtype)  # Ensure potential matches the grid shape

        # Prepare Hamiltonian components
        self._init_hamiltonian()

        # Wavefunction representation
        self.psi = np.zeros(self.grid_shape, dtype=np.complex128)
        self.initialize_vacuum_state()

        # Quantum coherence
        self.coherence = QuantumCoherence(config.grid_resolution)
        
        # Initialize ethical components
        self.ethical_tensor = None
        self.ethical_coupling = getattr(config, 'ethical_coupling', 0.1)
        self.previous_psi = None
        self.previous_previous_psi = None
    
    def _init_hamiltonian(self):
        """Initialize the Hamiltonian operator using sparse matrices and memory-efficient approach"""
        grid_resolution = self.config.grid_resolution
        dim = self.config.spatial_dim

        # Debug: Print grid resolution and spatial dimensions
        print(f"Grid resolution: {grid_resolution}, Spatial dimensions: {dim}")

        # Build 1D Laplacian
        diagonals = [-2.0 * np.ones(grid_resolution, dtype=self.dtype), 
                     np.ones(grid_resolution-1, dtype=self.dtype), 
                     np.ones(grid_resolution-1, dtype=self.dtype)]
        offsets = [0, -1, 1]
        laplacian_1d = diags(diagonals, offsets, shape=(grid_resolution, grid_resolution), 
                             dtype=self.dtype, format='csr')

        # Identity matrix
        identity = eye(grid_resolution, format='csr', dtype=self.dtype)

        # Build Laplacian for higher dimensions
        if dim == 1:
            self.laplacian = laplacian_1d
        elif dim == 2:
            laplacian_x = kron(laplacian_1d, identity, format='csr')
            laplacian_y = kron(identity, laplacian_1d, format='csr')
            self.laplacian = laplacian_x + laplacian_y
        elif dim == 3:
            identity_sq = kron(identity, identity, format='csr')
            laplacian_x = kron(laplacian_1d, identity_sq, format='csr')
            laplacian_y = kron(identity, kron(laplacian_1d, identity, format='csr'), format='csr')
            laplacian_z = kron(identity_sq, laplacian_1d, format='csr')
            self.laplacian = laplacian_x + laplacian_y + laplacian_z

        # Scale Laplacian
        self.laplacian = self.laplacian * (1.0 / self.lattice_spacing**2)

        # Initialize kinetic and potential energy operators
        self.kinetic_operator = -0.5 * self.constants.hbar**2 / self.config.mass * self.laplacian
        potential_values = self.potential.reshape(-1).astype(self.dtype)
        self.potential_operator = diags([potential_values], [0], format='csr')

        # Debug: Print shapes of operators
        print(f"Kinetic operator shape: {self.kinetic_operator.shape}")
        print(f"Potential operator shape: {self.potential_operator.shape}")

        # Full Hamiltonian (H = T + V)
        self.hamiltonian = self.kinetic_operator + self.potential_operator
    
    def initialize_vacuum_state(self):
        """Initialize the field in the vacuum state"""
        # Start with Gaussian vacuum state
        sigma = 1.0 / np.sqrt(2 * self.config.mass)
        
        if self.config.spatial_dim == 1:
            x = np.linspace(-1, 1, self.config.grid_resolution)
            self.psi = np.exp(-x**2 / (2 * sigma**2)) / np.sqrt(sigma * np.sqrt(2 * np.pi))
        elif self.config.spatial_dim == 2:
            x = np.linspace(-1, 1, self.config.grid_resolution)
            y = np.linspace(-1, 1, self.config.grid_resolution)
            X, Y = np.meshgrid(x, y)
            self.psi = np.exp(-(X**2 + Y**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        elif self.config.spatial_dim == 3:
            x = np.linspace(-1, 1, self.config.grid_resolution)
            y = np.linspace(-1, 1, self.config.grid_resolution)
            z = np.linspace(-1, 1, self.config.grid_resolution)
            X, Y, Z = np.meshgrid(x, y, z)
            self.psi = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi)**1.5)
        
        # Normalize
        self.psi = self.psi / np.sqrt(np.sum(np.abs(self.psi)**2) * self.lattice_spacing**self.config.spatial_dim)
    
    def evolve_field(self, dt):
        """Evolve quantum field using split-operator method"""
        # Apply kinetic operator (half step)
        psi_k = np.fft.fftn(self.psi)
        k_values = np.fft.fftfreq(self.config.grid_resolution, d=self.lattice_spacing)
        
        if self.config.spatial_dim == 1:
            K = k_values**2
            psi_k *= np.exp(-0.5j * dt * K)
        elif self.config.spatial_dim == 2:
            kx, ky = np.meshgrid(k_values, k_values)
            K = kx**2 + ky**2
            psi_k *= np.exp(-0.5j * dt * K)
        elif self.config.spatial_dim == 3:
            kx, ky, kz = np.meshgrid(k_values, k_values, k_values)
            K = kx**2 + ky**2 + kz**2
            psi_k *= np.exp(-0.5j * dt * K)
        
        self.psi = np.fft.ifftn(psi_k)
        
        # Apply potential (full step)
        self.psi *= np.exp(-1j * dt * (self.potential + self.config.mass**2 * np.abs(self.psi)**2))
        
        # Apply kinetic operator (half step)
        psi_k = np.fft.fftn(self.psi)
        if self.config.spatial_dim == 1:
            psi_k *= np.exp(-0.5j * dt * K)
        elif self.config.spatial_dim == 2 or self.config.spatial_dim == 3:
            psi_k *= np.exp(-0.5j * dt * K)
        
        self.psi = np.fft.ifftn(psi_k)
        
        # Normalize (accounts for numerical errors)
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.lattice_spacing**self.config.spatial_dim)
        self.psi = self.psi / norm

        # Apply quantum coherence
        self.psi = self.coherence.apply_coherence(self.psi)
    
    def compute_energy(self):
        """Compute total energy of the field"""
        # Kinetic energy (using spectral method)
        psi_k = np.fft.fftn(self.psi)
        k_values = np.fft.fftfreq(self.config.grid_resolution, d=self.lattice_spacing)
        
        if self.config.spatial_dim == 1:
            K = k_values**2
            kinetic = np.sum(np.abs(psi_k)**2 * K) * self.lattice_spacing
        elif self.config.spatial_dim == 2:
            kx, ky = np.meshgrid(k_values, k_values)
            K = kx**2 + ky**2
            kinetic = np.sum(np.abs(psi_k)**2 * K) * self.lattice_spacing**2
        elif self.config.spatial_dim == 3:
            kx, ky, kz = np.meshgrid(k_values, k_values, k_values)
            K = kx**2 + ky**2 + kz**2
            kinetic = np.sum(np.abs(psi_k)**2 * K) * self.lattice_spacing**3
        
        # Potential energy
        potential_energy = np.sum(self.potential * np.abs(self.psi)**2) * self.lattice_spacing**self.config.spatial_dim
        
        # Self-interaction energy (φ⁴ term)
        interaction = 0.5 * self.config.coupling * np.sum(np.abs(self.psi)**4) * self.lattice_spacing**self.config.spatial_dim
        
        return kinetic + potential_energy + interaction

# -------------------------------------------------------------------------
# Quantum Coherence
# -------------------------------------------------------------------------
class QuantumCoherence:
    """Simulate quantum coherence and entanglement"""
    def __init__(self, grid_resolution):
        self.grid_resolution = grid_resolution
        self.entanglement_matrix = np.random.rand(grid_resolution, grid_resolution)

    def apply_coherence(self, wavefunction):
        """Apply coherence and entanglement"""
        return np.dot(self.entanglement_matrix, wavefunction)

# -------------------------------------------------------------------------
# Quantum Monte Carlo Implementation
# -------------------------------------------------------------------------
class QuantumMonteCarlo:
    """Path integral Monte Carlo for quantum systems"""
    def __init__(self, config):
        self.config = config
        self.grid_shape = (config.grid_resolution,) * config.spatial_dim
        self.beta = 1.0 / (config.mass * 0.1)  # Inverse temperature
        self.action = np.zeros(self.grid_shape)
        self.field_config = np.random.normal(0, 1, self.grid_shape)
        
        # RG flow functions for ethical parameters
        self.beta_rg = [
            lambda x: x**2 - x,               # First order effect
            lambda x: 0.5*x**3 - 0.3*x,       # Second order effect
            lambda x: 0.1*x**4 - 0.2*x**2,    # Third order effect
            lambda x: 0.05*x**5 - 0.1*x**3,   # Fourth order effect
            lambda x: 0.01*x**6 - 0.05*x**4   # Fifth order effect
        ]
        
        # Initialize ethical tensor
        self.ethical_tensor = np.zeros((config.ethical_dim, *self.grid_shape))
        
        # Prepare CUDA kernels if GPU is available
        if config.use_gpu and cuda.is_available():
            self._init_cuda_kernels()
            
    def _init_cuda_kernels(self):
        """Initialize CUDA kernels for GPU acceleration"""
        threads_per_block = 256
        blocks_per_grid = (np.prod(self.grid_shape) + threads_per_block - 1) // threads_per_block
        self.cuda_grid = (blocks_per_grid, 1)
        self.cuda_block = (threads_per_block, 1, 1)
        
        # Pre-compile kernels
        self._compile_action_kernel()
        self._compile_metropolis_kernel()
    
    def _compile_action_kernel(self):
        """Compile CUDA kernel for action calculation"""
        @cuda.jit
        def action_kernel(field, ethical_tensor, action_out, mass, coupling, eth_coupling, dims):
            idx = cuda.grid(1)
            if idx < field.size:
                # Convert flat index to multi-dimensional
                coords = np.zeros(dims, dtype=np.int32)
                temp = idx
                for i in range(dims-1, -1, -1):
                    coords[i] = temp % field.shape[i]
                    temp //= field.shape[i]
                
                # Compute kinetic term (discrete Laplacian)
                lap = 0.0
                for dim in range(dims):
                    for offset in [-1, 1]:
                        neigh_coords = coords.copy()
                        neigh_coords[dim] = (neigh_coords[dim] + offset) % field.shape[dim]
                        
                        # Convert back to flat index
                        neigh_idx = 0
                        for i in range(dims):
                            neigh_idx = neigh_idx * field.shape[i] + neigh_coords[i]
                        
                        lap += field[neigh_idx] - field[idx]
                
                # Potential term
                phi = field[idx]
                potential = 0.5 * mass**2 * phi**2 + 0.25 * coupling * phi**4
                
                # Ethical coupling
                eth_term = 0.0
                for e in range(ethical_tensor.shape[0]):
                    eth_term += ethical_tensor[e, idx] * phi
                
                # Total action
                action_out[idx] = (lap + potential + eth_coupling * eth_term)
        
        self.action_kernel = action_kernel
    
    def _compile_metropolis_kernel(self):
        """Compile CUDA kernel for Metropolis updates"""
        @cuda.jit
        def metropolis_kernel(field, ethical_tensor, action, mass, coupling, eth_coupling, 
                             dims, step_size, random_vals, accept_mask):
            idx = cuda.grid(1)
            if idx < field.size:
                # Propose move
                field_new = field[idx] + step_size * (2.0 * random_vals[idx] - 1.0)
                
                # Compute action difference
                delta_S = 0.0
                
                # Kinetic term
                for dim in range(dims):
                    for offset in [-1, 1]:
                        neigh_coords = np.zeros(dims, dtype=np.int32)
                        temp = idx
                        for i in range(dims-1, -1, -1):
                            neigh_coords[i] = temp % field.shape[i]
                            temp //= field.shape[i]
                        
                        neigh_coords[dim] = (neigh_coords[dim] + offset) % field.shape[dim]
                        
                        # Convert back to flat index
                        neigh_idx = 0
                        for i in range(dims):
                            neigh_idx = neigh_idx * field.shape[i] + neigh_coords[i]
                        
                        delta_S += 0.5 * (field_new - field[idx]) * (field[neigh_idx] - field[idx])
                
                # Potential term
                phi_old = field[idx]
                phi_new = field_new
                delta_S += 0.5 * mass**2 * (phi_new**2 - phi_old**2)
                delta_S += 0.25 * coupling * (phi_new**4 - phi_old**4)
                
                # Ethical coupling
                for e in range(ethical_tensor.shape[0]):
                    delta_S += eth_coupling * ethical_tensor[e, idx] * (phi_new - phi_old)
                
                # Accept-reject step
                if random_vals[idx + field.size] < np.exp(-delta_S):
                    field[idx] = field_new
                    accept_mask[idx] = 1
                else:
                    accept_mask[idx] = 0
        
        self.metropolis_kernel = metropolis_kernel
    
    def compute_action(self, field=None):
        """Compute action (energy functional) of field configuration"""
        if field is None:
            field = self.field_config
            
        if self.config.use_gpu and cuda.is_available():
            # GPU implementation
            action_gpu = cuda.device_array(field.shape, dtype=np.float64)
            field_gpu = cuda.to_device(field)
            ethical_gpu = cuda.to_device(self.ethical_tensor)
            
            self.action_kernel[self.cuda_grid, self.cuda_block](
                field_gpu, ethical_gpu, action_gpu, 
                self.config.mass, self.config.coupling, self.config.ethical_coupling,
                self.config.spatial_dim
            )
            
            self.action = action_gpu.copy_to_host()
        else:
            # CPU implementation
            self.action = np.zeros_like(field)
            
            # Kinetic term (discrete Laplacian)
            for dim in range(self.config.spatial_dim):
                for offset in [-1, 1]:
                    # Compute nearest neighbor differences
                    slices_this = [slice(None)] * self.config.spatial_dim
                    slices_neigh = [slice(None)] * self.config.spatial_dim
                    
                    slices_this[dim] = slice(None)
                    slices_neigh[dim] = slice(offset, None) if offset > 0 else slice(None, offset)
                    
                    self.action[tuple(slices_this)] += 0.5 * (
                        field[tuple(slices_this)] - field[tuple(slices_neigh)]
                    )**2
            
            # Potential term
            self.action += 0.5 * self.config.mass**2 * field**2
            self.action += 0.25 * self.config.coupling * field**4
            
            # Ethical coupling
            for e in range(self.config.ethical_dim):
                self.action += self.config.ethical_coupling * self.ethical_tensor[e] * field
            
        return np.sum(self.action)
    
    def metropolis_step(self, ethical_tensor=None):
        """Perform one Metropolis Monte Carlo step"""
        if ethical_tensor is not None:
            self.ethical_tensor = ethical_tensor
            
        accept_ratio = 0.0
        step_size = 1.0 / self.config.mass
        
        if self.config.use_gpu and cuda.is_available():
            # GPU implementation
            field_gpu = cuda.to_device(self.field_config)
            ethical_gpu = cuda.to_device(self.ethical_tensor)
            action_gpu = cuda.to_device(self.action)
            
            # Random numbers for proposal and accept/reject
            random_vals = np.random.random(2 * np.prod(self.grid_shape)).astype(np.float32)
            random_gpu = cuda.to_device(random_vals)
            
            # Acceptance mask
            accept_mask = np.zeros(np.prod(self.grid_shape), dtype=np.int32)
            accept_gpu = cuda.to_device(accept_mask)
            
            self.metropolis_kernel[self.cuda_grid, self.cuda_block](
                field_gpu, ethical_gpu, action_gpu, 
                self.config.mass, self.config.coupling, self.config.ethical_coupling,
                self.config.spatial_dim, step_size, random_gpu, accept_gpu
            )
            
            # Copy results back to host
            self.field_config = field_gpu.copy_to_host()
            accept_mask = accept_gpu.copy_to_host()
            accept_ratio = np.mean(accept_mask)
        else:
            # CPU implementation
            for _ in range(self.config.metropolis_steps):
                # Select a random site
                site_idx = tuple(np.random.randint(0, self.config.grid_resolution, self.config.spatial_dim))
                
                # Current action contribution
                old_action_site = self._site_action(site_idx)
                
                # Propose new value
                old_val = self.field_config[site_idx]
                new_val = old_val + step_size * (2.0 * np.random.random() - 1.0)
                
                # Temporarily update field
                self.field_config[site_idx] = new_val
                
                # New action contribution
                new_action_site = self._site_action(site_idx)
                
                # Metropolis acceptance
                delta_S = new_action_site - old_action_site
                if delta_S < 0 or np.random.random() < np.exp(-delta_S):
                    # Accept
                    accept_ratio += 1.0
                else:
                    # Reject
                    self.field_config[site_idx] = old_val
            
            accept_ratio /= self.config.metropolis_steps
        
        # Recompute full action
        self.compute_action()
        
        return self.field_config, accept_ratio
    
    def _site_action(self, site_idx):
        """Compute action contribution from a single site"""
        action_site = 0.0
        
        # Kinetic term
        for dim in range(self.config.spatial_dim):
            for offset in [-1, 1]:
                neigh_idx = list(site_idx)
                neigh_idx[dim] = (neigh_idx[dim] + offset) % self.config.grid_resolution
                neigh_idx = tuple(neigh_idx)
                
                action_site += 0.5 * (self.field_config[site_idx] - self.field_config[neigh_idx])**2
        
        # Potential term
        phi = self.field_config[site_idx]
        action_site += 0.5 * self.config.mass**2 * phi**2
        action_site += 0.25 * self.config.coupling * phi**4
        
        # Ethical coupling
        for e in range(self.config.ethical_dim):
            action_site += self.config.ethical_coupling * self.ethical_tensor[e][site_idx] * phi
        
        return action_site
    
    def renormalization_flow(self, ethical_params, scale):
        """Solve RG equations for ethical couplings"""
        def _rg_equations(t, params):
            return np.array([self.beta_rg[i](params[i]) for i in range(len(params))])
        
        # Use SciPy's ODE solver
        sol = solve_ivp(
            _rg_equations,
            [0, scale],
            ethical_params,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        
        return sol.y[:, -1]  # Return final values
    
    def thermalize(self):
        """Run Metropolis algorithm to reach thermal equilibrium"""
        acceptance_ratios = []
        
        for step in tqdm(range(self.config.thermalization_steps), desc="Thermalizing QFT vacuum"):
            _, accept_ratio = self.metropolis_step()
            acceptance_ratios.append(accept_ratio)
            
            # Update ethical parameters every few steps
            if step % 10 == 0:
                flat_params = np.mean(self.ethical_tensor, axis=tuple(range(1, self.ethical_tensor.ndim)))
                evolved_params = self.renormalization_flow(flat_params, 0.1)
                
                # Reshape and broadcast
                for e in range(self.config.ethical_dim):
                    self.ethical_tensor[e] = evolved_params[e]
        
        return np.mean(acceptance_ratios)

# -------------------------------------------------------------------------
# Adaptive Mesh Refinement
# -------------------------------------------------------------------------
class AMRGrid:
    """Adaptive Mesh Refinement for field theory"""
    def __init__(self, config):
        self.config = config
        self.max_level = 3
        self.refinement_threshold = 0.1
        self.base_resolution = config.grid_resolution
        
        # Initialize grid hierarchy
        self.grids = {0: np.zeros((self.base_resolution,) * config.spatial_dim)}
        self.refined_regions = {}
        self.interpolation_matrices = {}
        
        # Field data on each level
        self.fields = {0: np.zeros((self.base_resolution,) * config.spatial_dim)}
        
        # Grid spacing at each level
        self.dx = {0: 1.0 / self.base_resolution}
        
    def refine_regions(self, error_estimate):
        """Apply Berger-Oliger AMR based on error estimate"""
        # Start with finest level and work down
        for level in range(self.max_level - 1, -1, -1):
            if level in self.grids:
                # Flag cells for refinement
                flagged = self._flag_cells(level, error_estimate)
                
                if np.any(flagged):
                    # Group flagged cells into rectangular patches
                    patches = self._create_patches(level, flagged)
                    
                    # Create child grid for each patch
                    for patch in patches:
                        self._create_child_grid(level + 1, patch)
    
    def _flag_cells(self, level, error):
        """Flag cells exceeding refinement threshold"""
        if isinstance(error, np.ndarray):
            # Resample error to match this level's resolution
            if error.shape != self.grids[level].shape:
                error = self._resample(error, self.grids[level].shape)
            
            # Apply threshold
            return error > self.refinement_threshold
        else:
            # Gradient-based flagging
            grad = np.gradient(self.fields[level])
            grad_mag = np.sqrt(sum(g**2 for g in grad))
            return grad_mag > self.refinement_threshold
    
    def _create_patches(self, level, flagged):
        """Convert flagged cells to rectangular patches"""
        # Implement clustering algorithm
        # This is a simplified version - in practice would use more sophisticated methods
        patches = []
        
        # Add buffer zone around flagged cells
        buffer_width = 2
        flagged_buffered = flagged.copy()
        
        if self.config.spatial_dim == 1:
            from scipy.ndimage import binary_dilation
            flagged_buffered = binary_dilation(flagged, iterations=buffer_width)
        else:
            for _ in range(buffer_width):
                for dim in range(self.config.spatial_dim):
                    slices_p = [slice(None)] * self.config.spatial_dim
                    slices_n = [slice(None)] * self.config.spatial_dim
                    
                    slices_p[dim] = slice(0, -1)
                    slices_n[dim] = slice(1, None)
                    
                    flagged_temp = flagged_buffered.copy()
                    flagged_buffered[tuple(slices_p)] |= flagged_temp[tuple(slices_n)]
                    flagged_buffered[tuple(slices_n)] |= flagged_temp[tuple(slices_p)]
        
        # Find connected components (simplified)
        from scipy.ndimage import label
        labeled, num_features = label(flagged_buffered)
        
        for i in range(1, num_features + 1):
            # Get bounding box of this connected component
            indices = np.where(labeled == i)
            
            if self.config.spatial_dim == 1:
                xmin, xmax = np.min(indices[0]), np.max(indices[0])
                bbox = [(xmin, xmax)]
            elif self.config.spatial_dim == 2:
                ymin, ymax = np.min(indices[0]), np.max(indices[0])
                xmin, xmax = np.min(indices[1]), np.max(indices[1])
                bbox = [(ymin, ymax), (xmin, xmax)]
            elif self.config.spatial_dim == 3:
                zmin, zmax = np.min(indices[0]), np.max(indices[0])
                ymin, ymax = np.min(indices[1]), np.max(indices[1])
                xmin, xmax = np.min(indices[2]), np.max(indices[2])
                bbox = [(zmin, zmax), (ymin, ymax), (xmin, xmax)]
            
            # Ensure minimum patch size and even dimensions for refinement
            for d in range(len(bbox)):
                min_size = 4  # Minimum patch size
                size = int(bbox[d][1]) - int(bbox[d][0]) + 1

                if size < min_size:
                    # Use explicit arithmetic instead of max/min to avoid mixed int / numpy intp typing issues
                    center = (int(bbox[d][0]) + int(bbox[d][1])) // 2
                    half_size = min_size // 2
                    start = center - half_size
                    if start < 0:
                        start = 0
                    end = center + half_size
                    limit = self.grids[level].shape[d] - 1
                    if end > limit:
                        end = limit
                    bbox[d] = (np.intp(start), np.intp(end))

                # Recompute size after possible adjustment
                size = int(bbox[d][1]) - int(bbox[d][0]) + 1
                if size % 2 == 1:
                    # Make even by extending upper bound within limits
                    end = int(bbox[d][1]) + 1
                    limit = self.grids[level].shape[d] - 1
                    if end > limit:
                        end = limit
                    bbox[d] = (np.intp(bbox[d][0]), np.intp(end))

            patches.append(bbox)
        
        return patches
    
    def _create_child_grid(self, level, patch):
        """Create a refined grid for a specific patch"""
        if level not in self.grids:
            self.grids[level] = np.zeros((self.base_resolution * 2**level,) * self.config.spatial_dim)
            self.fields[level] = np.zeros_like(self.grids[level])
            self.dx[level] = self.dx[level - 1] / 2
        
        # Extract patch bounds and refine
        slices = tuple(slice(b[0], b[1] + 1) for b in patch)
        self.grids[level][slices] = self._refine(self.grids[level - 1][slices])
        self.refined_regions[level] = self.refined_regions.get(level, []) + [patch]
# -------------------------------------------------------------------------
# Symbolic Operators from Genesis Framework
# -------------------------------------------------------------------------
class SymbolicOperators:
    """Implementation of formal symbolic operators from Genesis Framework"""
    
    @staticmethod
    def field_propagation(field, dt, constants):
        """
        Field Propagation (∇ϕ) - Calculates force vector at a given point in space-time
        
        Args:
            field: The quantum field to propagate
            dt: Time step
            constants: Physical constants
            
        Returns:
            Force vectors for the field
        """
        # Calculate gradient of field
        grad = np.gradient(np.abs(field)**2)
        
        # Convert to force vectors (F = -∇V)
        force_vectors = [-g for g in grad]
        
        # Scale by appropriate constants
        scaled_vectors = [constants.hbar * v for v in force_vectors]
        
        return scaled_vectors
    
    @staticmethod
    def quantum_collapse(wavefunction, positions, method="random"):
        """
        Quantum Collapse (Ψ→) - Resolves probabilistic states into definite outcomes
        
        Args:
            wavefunction: Quantum state to collapse
            positions: Possible position states
            method: Collapse method ("random", "max_prob", "ethical_weight")
            
        Returns:
            Collapsed state and position
        """
        # Calculate probability distribution
        prob_density = np.abs(wavefunction)**2
        prob_density = prob_density / np.sum(prob_density)
        
        if method == "random":
            # Standard random collapse according to Born rule
            flat_probs = prob_density.flatten()
            flat_indices = np.arange(len(flat_probs))
            chosen_idx = np.random.choice(flat_indices, p=flat_probs)
            
            # Convert back to original shape indices
            chosen_position = np.unravel_index(chosen_idx, prob_density.shape)
            
        elif method == "max_prob":
            # Collapse to highest probability state (useful for debugging)
            chosen_position = np.unravel_index(np.argmax(prob_density), prob_density.shape)
            
        elif method == "ethical_weight":
            # Weight collapse by ethical tensor (not fully implemented yet)
            chosen_position = np.unravel_index(np.argmax(prob_density), prob_density.shape)
        
        # Create collapsed state (delta function at chosen position)
        collapsed_state = np.zeros_like(wavefunction)
        collapsed_state[chosen_position] = 1.0
        
        return collapsed_state, positions[chosen_position]
    
    @staticmethod
    def conservation_enforcement(before_states, after_states, constants, tolerance=1e-6):
        """
        Conservation Enforcement (≡ℰ) - Balances physical quantities across transformations
        
        Args:
            before_states: System state before transformation
            after_states: System state after transformation
            constants: Physical constants
            tolerance: Allowed deviation
            
        Returns:
            Corrected after_states that conserve required quantities
        """
        # Check energy conservation
        energy_before = np.sum([np.sum(np.abs(state)**2) for state in before_states])
        energy_after = np.sum([np.sum(np.abs(state)**2) for state in after_states])
        
        energy_diff = energy_after - energy_before
        
        if abs(energy_diff) > tolerance:
            # Apply correction factor to enforce conservation
            correction_factor = np.sqrt(energy_before / energy_after)
            after_states = [state * correction_factor for state in after_states]
            
            # Log correction
            print(f"Applied energy conservation correction: {correction_factor:.6f}")
        
        return after_states
    
    @staticmethod
    def ethical_force_application(field, ethical_tensor, coupling_constant):
        """
        Ethical Force Application (⊕ξ) - Applies moral weight as actual physical force
        
        Args:
            field: Quantum field to modify
            ethical_tensor: Ethical force tensor
            coupling_constant: Strength of ethical-physical coupling
            
        Returns:
            Modified field with ethical forces applied
        """
        # Calculate ethical force gradient
        ethical_grad = np.zeros_like(field)
        
        # Apply each ethical dimension
        for ethical_dim in range(ethical_tensor.shape[0]):
            # Calculate gradient of ethical field in this dimension
            eth_grad = np.gradient(ethical_tensor[ethical_dim])
            
            # Apply to corresponding spatial dimensions
            for i, grad_component in enumerate(eth_grad):
                if i < len(ethical_grad.shape):
                    ethical_grad += coupling_constant * grad_component
        
        # Apply ethical force to field (phase modification)
        modified_field = field * np.exp(1j * ethical_grad)
        
        # Normalize to ensure conservation
        norm = np.sqrt(np.sum(np.abs(modified_field)**2))
        if norm > 0:
            modified_field = modified_field / norm
        
        return modified_field

# -------------------------------------------------------------------------
# Timeline Engine Connection
# -------------------------------------------------------------------------
class TemporalFramework:
    """Connects Quantum & Physics Base Engine to Timeline Engine"""
    
    def __init__(self, config):
        self.config = config
        self.timeline = None
        self.breath_phase = 0.0
        self.last_tick_time = 0.0
        self.temporal_event_queue = []
        
    def register_timeline(self, timeline_engine):
        """Connect to Timeline Engine for temporal framework"""
        self.timeline = timeline_engine
        # Register for temporal events
        if hasattr(timeline_engine, 'register_observer'):
            timeline_engine.register_observer(self._handle_temporal_event)
        print("Quantum Physics Engine connected to Timeline Engine")
    
    def _handle_temporal_event(self, event_data):
        """Process temporal events from Timeline Engine"""
        event_type = event_data.get('type')
        
        if event_type == 'breath_pulse':
            # Synchronize field oscillations with breath cycle
            self.breath_phase = event_data.get('phase', 0.0)
            self._synchronize_to_breath_phase()
            
        elif event_type == 'temporal_paradox':
            # Handle temporal paradox
            self.temporal_event_queue.append(event_data)
            return self.resolve_physical_paradox(event_data)
            
        elif event_type == 'tick':
            # Update master time
            self.last_tick_time = event_data.get('time', self.last_tick_time + self.config.temporal_resolution)
    
    def _synchronize_to_breath_phase(self):
        """Synchronize quantum oscillations to breath phase"""
        # Implementation depends on how breath synchronization should affect quantum fields
        # For now, just a placeholder that would modify field phases
        pass
    
    def get_current_tick(self):
        """Get current time from timeline"""
        if self.timeline and hasattr(self.timeline, 'master_tick'):
            return self.timeline.master_tick
        return self.last_tick_time / self.config.temporal_resolution

# -------------------------------------------------------------------------
# Paradox Resolution System
# -------------------------------------------------------------------------
class ParadoxResolver:
    """
    Handles physical contradictions and paradoxes according to the
    'Paradox as Entropy Source' principle from Genesis Framework
    """
    
    def __init__(self, config):
        self.config = config
        self.paradox_history = []
        self.entropy_generated = 0.0
        
    def resolve_physical_paradox(self, contradiction_data):
        """
        Convert physical contradictions into usable energy
        Implements 'Paradox as Entropy Source' principle from Genesis Framework
        """
        # Extract contradiction parameters
        paradox_type = contradiction_data.get('type', 'unknown')
        severity = contradiction_data.get('severity', 1.0)
        location = contradiction_data.get('location', (0, 0, 0, 0))  # 4D spacetime location
        description = contradiction_data.get('description', 'Unspecified paradox')
        
        # Calculate contradiction energy based on severity
        energy_density = self._calculate_contradiction_energy(severity)
        
        # Log the paradox
        self.paradox_history.append({
            'time': time.time(),
            'type': paradox_type,
            'severity': severity,
            'location': location,
            'description': description,
            'energy_generated': energy_density
        })
        
        # Calculate entropy increase (simplified Shannon entropy)
        entropy_increase = np.log(1 + severity)
        self.entropy_generated += entropy_increase
        
        # Return resolution status
        return {
            'resolved': True,
            'energy_generated': energy_density,
            'entropy_increase': entropy_increase,
            'resolution_method': f"Converted {paradox_type} paradox to {energy_density:.2e} energy units"
        }
    
    def _calculate_contradiction_energy(self, severity):
        """Calculate usable energy from paradox severity"""
        # E = mc² analog for paradoxes: paradox severity converts to energy
        # More severe paradoxes generate more energy
        base_energy = self.config.vacuum_energy * 100  # Base conversion rate
        return base_energy * severity * (1 + 0.1 * np.random.random())  # Add some randomness
    
    def _inject_energy_to_field(self, field, energy_density, location=None):
        """Inject paradox-derived energy into a quantum field"""
        if location is None:
            # Distribute energy throughout the field
            shape = field.shape
            energy_field = np.ones(shape) * energy_density / np.prod(shape)
        else:
            # Create localized energy injection
            energy_field = np.zeros_like(field)
            # Create a Gaussian distribution around the location
            indices = np.indices(field.shape)
            for dim, loc in enumerate(location[:len(field.shape)]):
                energy_field += np.exp(-0.5 * ((indices[dim] - loc) / 2)**2)
            # Normalize and scale by energy density
            energy_field = energy_field / np.sum(energy_field) * energy_density
        
        # Add energy to field (amplitude increase)
        modified_field = field + np.sqrt(energy_field) * np.exp(1j * 2 * np.pi * np.random.random(field.shape))
        
        # Normalize
        return modified_field / np.sqrt(np.sum(np.abs(modified_field)**2))
    
    def get_total_entropy_generated(self):
        """Return total entropy generated from paradox resolution"""
        return self.entropy_generated

# -------------------------------------------------------------------------
# Methods to add to QuantumField class
# -------------------------------------------------------------------------
# You can add these methods to your existing QuantumField class
def apply_unified_field_equation(self, dt: float) -> None:
    """
    Implement the Quantum-Ethical Unified Field Equation:
    ∇²Φ(x,t) = δ²Φ(x,t)/δt² + η∇·[ξ(x,t)Φ(x,t)] + V(x,t)Φ(x,t)
    """
    # Compute Laplacian (∇²Φ)
    laplacian = self._compute_laplacian()
    
    # Compute second time derivative (δ²Φ/δt²)
    time_deriv = self._compute_time_derivatives()
    
    # Compute ethical force term (η∇·[ξ(x,t)Φ(x,t)])
    ethical_term = self._ethical_divergence_term()
    
    # Compute potential term (V(x,t)Φ(x,t))
    potential_term = self.potential * self.psi
    
    # Combine terms according to the unified equation
    d_psi = laplacian - time_deriv - ethical_term - potential_term
    
    # Update field
    self.psi = self.psi + dt * d_psi
    
    # Enforce normalization (conservation)
    self._enforce_normalization()

def _compute_laplacian(self):
    """Compute Laplacian of the field"""
    return self.laplacian.dot(self.psi.reshape(-1)).reshape(self.grid_shape)

def _compute_time_derivatives(self):
    """Compute second time derivative of field"""
    # For now, use a simplified approach
    # In a more advanced implementation, we would track previous states
    if self.previous_psi is None:
        self.previous_psi = self.psi.copy()
        self.previous_previous_psi = self.psi.copy()
        return np.zeros_like(self.psi)
    
    # Finite difference approximation of second derivative
    dt = self.config.temporal_resolution
    second_deriv = (self.psi - 2 * self.previous_psi + self.previous_previous_psi) / (dt**2)
    
    # Update history
    self.previous_previous_psi = self.previous_psi.copy()
    self.previous_psi = self.psi.copy()
    
    return second_deriv

def _ethical_divergence_term(self):
    """Compute the ethical force term: η∇·[ξ(x,t)Φ(x,t)]"""
    if self.ethical_tensor is None:
        # Initialize ethical tensor if not present
        ethical_dim = getattr(self.config, 'ethical_dim', 5)
        self.ethical_tensor = np.zeros((ethical_dim, *self.grid_shape))
    
    # Initialize result
    result = np.zeros_like(self.psi)
    
    # For each ethical dimension
    for dim in range(self.ethical_tensor.shape[0]):
        # Compute ethical field * psi
        ethical_field_psi = self.ethical_tensor[dim] * self.psi
        
        # Compute divergence
        div = np.sum(np.gradient(ethical_field_psi), axis=0)
        
        # Add to result with coupling constant
        result += self.ethical_coupling * div
    
    return result

def _enforce_normalization(self):
    """Ensure the wavefunction remains normalized"""
    norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.lattice_spacing**self.config.spatial_dim)
    if norm > 0:
        self.psi = self.psi / norm

def set_ethical_tensor(self, ethical_tensor):
    """Set the ethical force tensor field"""
    if ethical_tensor.shape[1:] != self.grid_shape:
        raise ValueError(f"Ethical tensor shape {ethical_tensor.shape} doesn't match field shape {self.grid_shape}")
    
    self.ethical_tensor = ethical_tensor
    
    # Log ethical field properties
    ethical_magnitude = np.mean(np.abs(ethical_tensor))
    ethical_gradient = np.mean([np.mean(np.abs(np.gradient(ethical_tensor[dim]))) 
                              for dim in range(ethical_tensor.shape[0])])
    
    print(f"Ethical tensor set with magnitude {ethical_magnitude:.4f}, gradient {ethical_gradient:.4f}")

# -------------------------------------------------------------------------
# Quantum State Vector Implementation
# -------------------------------------------------------------------------
class QuantumStateVector:
    """
    Represents a quantum state vector in Hilbert space, providing a 
    mathematical representation of quantum superposition and entanglement.
    """
    
    def __init__(self, n_qubits=1, state=None):
        """
        Initialize a quantum state vector.
        
        Args:
            n_qubits: Number of qubits in the system
            state: Initial state vector (if None, initialize to |0>^⊗n)
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        
        # Initialize state vector
        if state is not None:
            if len(state) != self.dim:
                raise ValueError(f"State vector dimension {len(state)} doesn't match expected {self.dim}")
            self.state = np.array(state, dtype=complex)
            self.normalize()
        else:
            # Initialize to |0...0> state
            self.state = np.zeros(self.dim, dtype=complex)
            self.state[0] = 1.0
    
    def normalize(self):
        """Normalize the state vector"""
        norm = np.sqrt(np.sum(np.abs(self.state)**2))
        if norm > 0:
            self.state /= norm
    
    def amplitude(self, bitstring):
        """
        Get the amplitude for a specific basis state.
        
        Args:
            bitstring: String of 0s and 1s representing the basis state
            
        Returns:
            Complex amplitude
        """
        if len(bitstring) != self.n_qubits:
            raise ValueError(f"Bitstring length {len(bitstring)} doesn't match qubit count {self.n_qubits}")
        
        # Convert bitstring to index
        index = int(bitstring, 2)
        return self.state[index]
    
    def probability(self, bitstring):
        """
        Get the probability for a specific basis state.
        
        Args:
            bitstring: String of 0s and 1s representing the basis state
            
        Returns:
            Probability (0 to 1)
        """
        return np.abs(self.amplitude(bitstring))**2
    
    def apply_gate(self, gate, target_qubits):
        """
        Apply a quantum gate to the state vector.
        
        Args:
            gate: Unitary matrix representing the gate
            target_qubits: List of qubits the gate acts on
        """
        # Validate gate shape
        gate_qubits = int(np.log2(gate.shape[0]))
        if len(target_qubits) != gate_qubits:
            raise ValueError(f"Gate acts on {gate_qubits} qubits but {len(target_qubits)} targets specified")
        
        # Validate target qubits
        if any(q >= self.n_qubits for q in target_qubits):
            raise ValueError(f"Target qubits {target_qubits} out of range for {self.n_qubits} qubits")
        
        # For single qubit gates, we can use a more efficient implementation
        if gate_qubits == 1:
            target = target_qubits[0]
            
            # Reshape state for easier processing
            new_shape = [2] * self.n_qubits
            state_tensor = self.state.reshape(new_shape)
            
            # Apply gate along the target axis
            # For numpy's tensordot to work correctly with complex matrices
            state_tensor = np.tensordot(gate, state_tensor, axes=([1], [target]))
            
            # Transpose to get the correct order
            axes = list(range(self.n_qubits + 1))
            axes.remove(0)
            axes.insert(target, 0)
            state_tensor = np.transpose(state_tensor, axes)
            
            # Reshape back to vector
            self.state = state_tensor.reshape(self.dim)
            
        else:
            # For multi-qubit gates, we'll use a simple but less efficient approach
            # Building the full unitary is exponential in the number of qubits!
            full_gate = self._expand_gate(gate, target_qubits)
            self.state = np.dot(full_gate, self.state)
        
        self.normalize()
    
    def _expand_gate(self, gate, target_qubits):
        """
        Expand a gate to act on the full Hilbert space.
        
        Args:
            gate: Gate matrix
            target_qubits: List of target qubits
        
        Returns:
            Full unitary matrix
        """
        # This is an inefficient implementation for illustration
        n = self.n_qubits
        full_gate = np.eye(2**n, dtype=complex)
        
        # For each basis state, apply the gate
        for i in range(2**n):
            # Convert i to bitstring
            bitstring = format(i, f'0{n}b')
            
            # Extract the bits corresponding to target_qubits
            target_bits = ''.join(bitstring[q] for q in target_qubits)
            target_idx = int(target_bits, 2)
            
            # For each output of the gate
            for j in range(gate.shape[1]):
                # Convert j to bitstring for target qubits
                j_bits = format(j, f'0{len(target_qubits)}b')
                
                # Create new bitstring with j_bits inserted at target positions
                new_bits = list(bitstring)
                for q_idx, q in enumerate(target_qubits):
                    new_bits[q] = j_bits[q_idx]
                new_bitstring = ''.join(new_bits)
                new_idx = int(new_bitstring, 2)
                
                # Apply gate
                full_gate[new_idx, i] = gate[j, target_idx]
        
        return full_gate
    
    def measure(self, qubit):
        """
        Measure a specific qubit and collapse the state.
        
        Args:
            qubit: Index of qubit to measure
            
        Returns:
            Measurement result (0 or 1)
        """
        # Calculate probabilities for 0 and 1 outcomes
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i in range(self.dim):
            # Convert i to bitstring
            bitstring = format(i, f'0{self.n_qubits}b')
            
            # Check if the qubit is 0 or 1
            if bitstring[qubit] == '0':
                prob_0 += np.abs(self.state[i])**2
            else:
                prob_1 += np.abs(self.state[i])**2
        
        # Random outcome based on probabilities
        outcome = np.random.choice([0, 1], p=[prob_0, prob_1])
        
        # Collapse the state
        new_state = np.zeros_like(self.state)
        norm = 0.0
        
        for i in range(self.dim):
            # Convert i to bitstring
            bitstring = format(i, f'0{self.n_qubits}b')
            
            # Keep only amplitudes consistent with measurement
            if int(bitstring[qubit]) == outcome:
                new_state[i] = self.state[i]
                norm += np.abs(self.state[i])**2
        
        # Normalize the collapsed state
        if norm > 0:
            self.state = new_state / np.sqrt(norm)
        
        return outcome
    
    def entanglement_entropy(self, subsystem_qubits):
        """
        Calculate the entanglement entropy between subsystem and its complement.
        
        Args:
            subsystem_qubits: List of qubit indices in the subsystem
            
        Returns:
            Entanglement entropy value
        """
        # Convert state vector to density matrix
        density = np.outer(self.state, np.conj(self.state))
        
        # Compute partial trace over complement subsystem
        reduced_density = self._partial_trace(density, subsystem_qubits)
        
        # Calculate von Neumann entropy
        eigenvalues = np.linalg.eigvalsh(reduced_density)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Ignore zero eigenvalues
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        return float(entropy)
    
    def _partial_trace(self, density, subsystem_qubits):
        """
        Compute the partial trace of the density matrix.
        
        Args:
            density: Full density matrix
            subsystem_qubits: Qubits to keep
            
        Returns:
            Reduced density matrix
        """
        # This is a simple but inefficient implementation
        n = self.n_qubits
        subsystem_dim = 2 ** len(subsystem_qubits)
        complement_qubits = [i for i in range(n) if i not in subsystem_qubits]
        complement_dim = 2 ** len(complement_qubits)
        
        # Reshape density matrix for partial trace
        density_tensor = density.reshape([2] * (2 * n))
        
        # Trace over complement qubits
        reduced_density = np.zeros((subsystem_dim, subsystem_dim), dtype=complex)
        
        # For each basis state of subsystem and complement
        for i in range(subsystem_dim):
            i_bits = format(i, f'0{len(subsystem_qubits)}b')
            for j in range(subsystem_dim):
                j_bits = format(j, f'0{len(subsystem_qubits)}b')
                
                # Sum over complement subsystem
                val = 0.0
                for k in range(complement_dim):
                    k_bits = format(k, f'0{len(complement_qubits)}b')
                    
                    # Create full indices for bra and ket
                    bra_idx = ['0'] * n
                    ket_idx = ['0'] * n
                    
                    # Fill in subsystem bits
                    for idx, q in enumerate(subsystem_qubits):
                        bra_idx[q] = i_bits[idx]
                        ket_idx[q] = j_bits[idx]
                    
                    # Fill in complement bits
                    for idx, q in enumerate(complement_qubits):
                        bra_idx[q] = k_bits[idx]
                        ket_idx[q] = k_bits[idx]
                    
                    # Convert to tensor indices
                    bra_tensor_idx = tuple(int(bit) for bit in bra_idx)
                    ket_tensor_idx = tuple(int(bit) for bit in ket_idx)
                    
                    # Combine for full density matrix index
                    tensor_idx = bra_tensor_idx + ket_tensor_idx
                    
                    # Add to sum
                    val += density_tensor[tensor_idx]
                
                reduced_density[i, j] = val
        
        return reduced_density
    
    def to_bloch_vector(self, qubit):
        """
        Convert a single qubit's state to Bloch sphere coordinates.
        
        Args:
            qubit: Index of qubit
            
        Returns:
            (x, y, z) coordinates on Bloch sphere
        """
        if self.n_qubits == 1:
            # Simple case, just use the state directly
            rho = np.outer(self.state, np.conj(self.state))
        else:
            # Compute reduced density matrix for the qubit
            rho = self._partial_trace(np.outer(self.state, np.conj(self.state)), [qubit])
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        
        # Calculate Bloch coordinates
        x = np.real(np.trace(np.dot(rho, sigma_x)))
        y = np.real(np.trace(np.dot(rho, sigma_y)))
        z = np.real(np.trace(np.dot(rho, sigma_z)))
        
        return (x, y, z)
    
    def visualize_bloch(self, qubit=0):
        """
        Visualize a qubit state on the Bloch sphere.
        
        Args:
            qubit: Index of qubit to visualize
        
        Returns:
            Matplotlib figure
        """
        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            warnings.warn("Cannot import Axes3D for 3D plotting")
            return None
        
        # Get Bloch coordinates
        x, y, z = self.to_bloch_vector(qubit)
        
        # Create figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw Bloch sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        sphere_x = np.cos(u) * np.sin(v)
        sphere_y = np.sin(u) * np.sin(v)
        sphere_z = np.cos(v)
        ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="lightgray", alpha=0.2)
        
        # Draw coordinate axes
        ax.quiver(-1.3, 0, 0, 2.6, 0, 0, color='gray', arrow_length_ratio=0.05, linewidth=0.5)
        ax.quiver(0, -1.3, 0, 0, 2.6, 0, color='gray', arrow_length_ratio=0.05, linewidth=0.5)
        ax.quiver(0, 0, -1.3, 0, 0, 2.6, color='gray', arrow_length_ratio=0.05, linewidth=0.5)
        
        # Draw state vector
        ax.quiver(0, 0, 0, x, y, z, color='r', arrow_length_ratio=0.05)
        
        # Label axes
        ax.text(1.5, 0, 0, r'$|0\rangle$', color='black')
        ax.text(-1.5, 0, 0, r'$|1\rangle$', color='black')
        ax.text(0, 1.5, 0, r'$|+\rangle$', color='black')
        ax.text(0, -1.5, 0, r'$|-\rangle$', color='black')
        ax.text(0, 0, 1.5, r'$|i+\rangle$', color='black')
        ax.text(0, 0, -1.5, r'$|i-\rangle$', color='black')
        
        # Set figure properties
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(f'Qubit {qubit} State on Bloch Sphere')
        
        # Add text showing coordinates
        state_text = f"Bloch Vector: ({x:.3f}, {y:.3f}, {z:.3f})"
        fig.text(0.5, 0.02, state_text, ha='center')
        
        return fig
    
    # Common gate definitions as class methods
    @classmethod
    def hadamard_gate(cls):
        """Create Hadamard gate matrix"""
        return (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    
    @classmethod
    def pauli_x(cls):
        """Create Pauli X (NOT) gate matrix"""
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @classmethod
    def pauli_y(cls):
        """Create Pauli Y gate matrix"""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @classmethod
    def pauli_z(cls):
        """Create Pauli Z gate matrix"""
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @classmethod
    def phase_gate(cls, phi):
        """Create phase gate matrix with phase phi"""
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)
    
    @classmethod
    def rotation_x(cls, theta):
        """Create rotation around X-axis gate matrix"""
        return np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    @classmethod
    def rotation_y(cls, theta):
        """Create rotation around Y-axis gate matrix"""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    @classmethod
    def rotation_z(cls, theta):
        """Create rotation around Z-axis gate matrix"""
        return np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ], dtype=complex)
    
    @classmethod
    def cnot_gate(cls):
        """Create CNOT (Controlled-NOT) gate matrix"""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
    
    @classmethod
    def swap_gate(cls):
        """Create SWAP gate matrix"""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    def create_bell_pair(self):
        """
        Create a Bell pair (maximally entangled two-qubit state)
        Requires that n_qubits >= 2
        """
        if self.n_qubits < 2:
            raise ValueError("Need at least 2 qubits to create a Bell pair")
            
        # Start with |00> state
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0
        
        # Apply Hadamard to first qubit
        self.apply_gate(self.hadamard_gate(), [0])
        
        # Apply CNOT with control=first qubit, target=second qubit
        self.apply_gate(self.cnot_gate(), [0, 1])
        
        return self