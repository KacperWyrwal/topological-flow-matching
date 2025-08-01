# Class Structure Design for Topological Flow Matching

## Overview
This document outlines the class hierarchy for modeling controlled SDEs and bridges in the topological flow matching project.

## 1. Core Abstract Base Classes

### SDE (Base Class)
```python
from abc import ABC, abstractmethod
import torch
from typing import Callable, Optional, Tuple, Union, Type

class SDE(ABC):
    """Abstract base class for Stochastic Differential Equations"""
    
    def __init__(self, dim: int, device: str = "cpu"):
        self.dim = dim
        self.device = device
        self._bridge = None  # Will be set by subclasses
    
    @abstractmethod
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute drift term f(t, x)"""
        pass
    
    @abstractmethod
    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute diffusion term g(t, x)"""
        pass
    
    @property
    def bridge(self) -> 'Bridge':
        """Return the bridge instance for this SDE"""
        if self._bridge is None:
            raise ValueError(f"No bridge defined for {self.__class__.__name__}")
        return self._bridge
```

### LinearSDE (Updated Base Class)
```python
class LinearSDE(SDE):
    """Base class for linear SDEs: dX_t = (A(t)X_t + b(t))dt + σ(t)dW_t"""
    
    def __init__(self, dim: int, A: Callable, b: Callable, sigma: Callable):
        super().__init__(dim)
        self.A = A  # A(t) matrix function
        self.b = b  # b(t) vector function  
        self.sigma = sigma  # σ(t) matrix function
        self._bridge = LinearGaussianBridge(self)
    
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.A(t), x) + self.b(t)
    
    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.sigma(t)
```

### DiagonalizableLinearSDE (New Base Class)
```python
class DiagonalizableLinearSDE(LinearSDE):
    """Base class for linear SDEs that can be diagonalized"""
    
    def __init__(self, dim: int, A: Callable, b: Callable, sigma: Callable):
        super().__init__(dim, A, b, sigma)
        self._diagonalized = False
        self._diagonal_form = None
        self.eigenvalues = None
        self.eigenvectors = None
    
    def diagonalize(self) -> 'DiagonalizableLinearSDE':
        """Return diagonalized version of the SDE"""
        if self._diagonalized:
            return self._diagonal_form
        
        # Compute eigenvalues and eigenvectors
        self._compute_eigenvalues_eigenvectors()
        
        # Create diagonal form
        diagonal_sde = self._create_diagonal_form()
        diagonal_sde._diagonalized = True
        diagonal_sde.eigenvalues = self.eigenvalues
        diagonal_sde.eigenvectors = self.eigenvectors
        self._diagonal_form = diagonal_sde
        return diagonal_sde
    
    def _compute_eigenvalues_eigenvectors(self):
        """Compute eigenvalues and eigenvectors for diagonalization"""
        # Implementation depends on specific linear SDE structure
        pass
    
    def _create_diagonal_form(self) -> 'DiagonalizableLinearSDE':
        """Create diagonal form of the SDE"""
        # Implementation depends on specific linear SDE structure
        pass
    
    def transform_to_original_space(self, x_diagonal: torch.Tensor) -> torch.Tensor:
        """Transform from diagonal space back to original space"""
        if self.eigenvectors is None:
            raise ValueError("Must diagonalize first")
        return torch.matmul(self.eigenvectors, x_diagonal)
    
    def transform_to_diagonal_space(self, x_original: torch.Tensor) -> torch.Tensor:
        """Transform from original space to diagonal space"""
        if self.eigenvectors is None:
            raise ValueError("Must diagonalize first")
        return torch.matmul(self.eigenvectors.T, x_original)
```

### TopologicalSDE (Updated)
```python
class TopologicalSDE(DiagonalizableLinearSDE):
    """SDE with topological structure that can be diagonalized"""
    
    def __init__(self, dim: int, L: torch.Tensor, device: str = "cpu"):
        # Create linear SDE with Laplacian as drift matrix
        def A(t): return -L  # Linear drift based on Laplacian
        def b(t): return torch.zeros(dim, device=device)
        def sigma(t): return torch.eye(dim, device=device)
        
        super().__init__(dim, A, b, sigma)
        self.L = L  # Laplacian matrix defining topological structure
    
    def _compute_eigenvalues_eigenvectors(self):
        """Compute eigenvalues and eigenvectors from Laplacian"""
        # Diagonalize the Laplacian L: L = U^T D U
        self.eigenvalues, self.eigenvectors = torch.linalg.eigh(self.L)
    
    def _create_diagonal_form(self) -> 'DiagonalizableLinearSDE':
        """Create diagonal form using Laplacian eigenvalues"""
        # Create diagonal linear SDE based on eigenvalues
        # Implementation depends on specific topological structure
        pass
```

### TopologicalOrnsteinUhlenbeckSDE (Updated)
```python
class TopologicalOrnsteinUhlenbeckSDE(TopologicalSDE):
    """Topological OU process that can be diagonalized"""
    
    def __init__(self, alpha: float, beta: float, L: torch.Tensor, dim: int = None):
        if dim is None:
            dim = L.shape[0]
        super().__init__(dim, L)
        self.alpha = alpha
        self.beta = beta
        self._bridge = TopologicalSchrodingerBridge(self)
    
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Override to use alpha parameter
        return -self.alpha * torch.matmul(self.L, x)
    
    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.ones_like(x)
    
    def _create_diagonal_form(self) -> 'DiagonalizableLinearSDE':
        """Create diagonal form using Laplacian eigenvalues"""
        # Ensure eigenvalues/eigenvectors are computed
        if self.eigenvalues is None:
            self._compute_eigenvalues_eigenvectors()
        
        # Create diagonal OU SDE with eigenvalues as parameters
        # Each component evolves as: dX_i = -α * λ_i * X_i dt + β dW_i
        # where λ_i are eigenvalues of L
        diagonal_sde = DiagonalTopologicalOU(
            alpha=self.alpha, 
            beta=self.beta, 
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors
        )
        return diagonal_sde
```

### DiagonalTopologicalOU (Updated)
```python
class DiagonalTopologicalOU(DiagonalizableLinearSDE):
    """Diagonalized version of topological OU process"""
    
    def __init__(self, alpha: float, beta: float, eigenvalues: torch.Tensor, 
                 eigenvectors: torch.Tensor):
        # Create diagonal linear SDE
        def A(t): return -alpha * torch.diag(eigenvalues)
        def b(t): return torch.zeros(eigenvalues.shape[0], device=eigenvalues.device)
        def sigma(t): return beta * torch.eye(eigenvalues.shape[0], device=eigenvalues.device)
        
        super().__init__(eigenvalues.shape[0], A, b, sigma)
        self.alpha = alpha
        self.beta = beta
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self._diagonalized = True  # Already diagonalized
```

### OrnsteinUhlenbeckSDE (Updated)
```python
class OrnsteinUhlenbeckSDE(DiagonalizableLinearSDE):
    """OU process: dX_t = -αX_t dt + βdW_t"""
    
    def __init__(self, alpha: float, beta: float, dim: int = 1):
        def A(t): return -alpha * torch.eye(dim, device=self.device)
        def b(t): return torch.zeros(dim, device=self.device)
        def sigma(t): return beta * torch.eye(dim, device=self.device)
        
        super().__init__(dim, A, b, sigma)
        self.alpha = alpha
        self.beta = beta
        # Override with specific OU bridge
        self._bridge = OUBridge(alpha, beta)
    
    def _compute_eigenvalues_eigenvectors(self):
        """OU process is already diagonal, so eigenvectors are identity"""
        self.eigenvalues = -self.alpha * torch.ones(self.dim, device=self.device)
        self.eigenvectors = torch.eye(self.dim, device=self.device)
    
    def _create_diagonal_form(self) -> 'DiagonalizableLinearSDE':
        """OU process is already in diagonal form"""
        return self
```

### ConstantDiffusionSDE (Updated)
```python
class ConstantDiffusionSDE(LinearSDE):
    """Constant diffusion: dX_t = βdW_t"""
    
    def __init__(self, beta: float, dim: int = 1):
        def A(t): return torch.zeros(dim, dim, device=self.device)
        def b(t): return torch.zeros(dim, device=self.device)
        def sigma(t): return beta * torch.eye(dim, device=self.device)
        
        super().__init__(dim, A, b, sigma)
        self.beta = beta
        self._bridge = LinearGaussianBridge(self)
```

## Updated Class Hierarchy

```python
SDE (base)
├── LinearSDE (linear SDEs)
│   ├── DiagonalizableLinearSDE (diagonalizable linear SDEs)
│   │   ├── TopologicalSDE (topological structure)
│   │   │   └── TopologicalOrnsteinUhlenbeckSDE
│   │   └── OrnsteinUhlenbeckSDE
│   └── ConstantDiffusionSDE
└── (other non-linear SDEs if needed)
```

## Usage Examples

```python
# Create Laplacian for topological structure
L = torch.tensor([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])  # 3-node graph

# Create topological OU SDE
topological_ou = TopologicalOrnsteinUhlenbeckSDE(alpha=1.0, beta=1.0, L=L)

# Get bridge
bridge = topological_ou.bridge()

# Diagonalize (computes eigenvalues/eigenvectors)
diagonal_sde = topological_ou.diagonalize()

# Access eigenvalues/eigenvectors
print(f"Eigenvalues: {diagonal_sde.eigenvalues}")
print(f"Eigenvectors shape: {diagonal_sde.eigenvectors.shape}")

# Transform between spaces
x_original = torch.randn(3)
x_diagonal = diagonal_sde.transform_to_diagonal_space(x_original)
x_back = diagonal_sde.transform_to_original_space(x_diagonal)

# Work in diagonal space (each component evolves independently)
drift_diagonal = diagonal_sde.drift(t, x_diagonal)  # Component-wise drift
```

## 2. SDE Implementations

### LinearSDE
```python
class LinearSDE(SDE):
    """Base class for linear SDEs: dX_t = (A(t)X_t + b(t))dt + σ(t)dW_t"""
    
    def __init__(self, dim: int, A: Callable, b: Callable, sigma: Callable):
        super().__init__(dim)
        self.A = A  # A(t) matrix function
        self.b = b  # b(t) vector function  
        self.sigma = sigma  # σ(t) matrix function
        self._bridge = LinearGaussianBridge(self)
    
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.A(t), x) + self.b(t)
    
    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.sigma(t)
```

### OrnsteinUhlenbeckSDE
```python
class OrnsteinUhlenbeckSDE(LinearSDE):
    """OU process: dX_t = -αX_t dt + βdW_t"""
    
    def __init__(self, alpha: float, beta: float, dim: int = 1):
        def A(t): return -alpha * torch.eye(dim, device=self.device)
        def b(t): return torch.zeros(dim, device=self.device)
        def sigma(t): return beta * torch.eye(dim, device=self.device)
        
        super().__init__(dim, A, b, sigma)
        self.alpha = alpha
        self.beta = beta
        # Override with specific OU bridge
        self._bridge = OUBridge(alpha, beta)
```

### TopologicalOrnsteinUhlenbeckSDE
```python
class TopologicalOrnsteinUhlenbeckSDE(TopologicalSDE):
    """Topological OU process that can be diagonalized"""
    
    def __init__(self, alpha: float, beta: float, L: torch.Tensor, dim: int = None):
        if dim is None:
            dim = L.shape[0]
        super().__init__(dim, L)
        self.alpha = alpha
        self.beta = beta
        self._bridge = TopologicalSchrodingerBridge(self)
    
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -self.alpha * x
    
    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.ones_like(x)
    
    def _create_diagonal_form(self) -> 'DiagonalizedSDE':
        """Create diagonal form using Laplacian eigenvalues"""
        # Diagonalize L: L = U^T D U
        eigenvalues, eigenvectors = torch.linalg.eigh(self.L)
        
        # Create diagonal OU SDE with eigenvalues as parameters
        # Each component evolves as: dX_i = -α_i X_i dt + β_i dW_i
        # where α_i and β_i depend on the eigenvalues
        diagonal_sde = DiagonalizedTopologicalOU(
            alpha=self.alpha, 
            beta=self.beta, 
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors
        )
        return diagonal_sde
```

### ConstantDiffusionSDE
```python
class ConstantDiffusionSDE(SDE):
    """Constant diffusion: dX_t = βdW_t"""
    
    def __init__(self, beta: float, dim: int = 1):
        super().__init__(dim)
        self.beta = beta
        self._bridge = LinearGaussianBridge(self)
    
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
    
    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.ones_like(x)
```

### ControlledSDE
```python
class ControlledSDE(SDE):
    """SDE with control: dX_t = (f(t,X_t) + u(t,X_t))dt + g(t,X_t)dW_t"""
    
    def __init__(self, base_sde: SDE, control: OptimalControl):
        super().__init__(base_sde.dim, base_sde.device)
        self.base_sde = base_sde
        self.control = control
        # Inherit bridge from base SDE
        self._bridge = base_sde._bridge
    
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        base_drift = self.base_sde.drift(t, x)
        control_term = self.control(t, x)
        return base_drift + control_term
    
    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.base_sde.diffusion(t, x)
```

## 3. Bridge Implementations

### SchrodingerBridge
```python
class SchrodingerBridge(Bridge):
    """Abstract base for Schrodinger Bridge processes"""
    
    def __init__(self, sde: SDE, noise_level: float = 0.0):
        super().__init__(sde)
        self.noise_level = noise_level
    
    @abstractmethod
    def static_sbp_solution(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Solve the static SBP to find transference plan"""
        pass
```

### TopologicalSchrodingerBridge
```python
class TopologicalSchrodingerBridge(SchrodingerBridge):
    """Topological Schrodinger Bridge with Dirac delta boundary marginals"""
    
    def __init__(self, sde: TopologicalSDE, noise_level: float = 0.0):
        super().__init__(sde, noise_level)
        self.topological_sde = sde
    
    def static_sbp_solution(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Explicit solution for topological SB with Dirac marginals"""
        # Implementation based on [1] for multivariate case
        # Simplified to 1D case since SDE can be diagonalized
        pass
    
    def _optimal_control_impl(self, t: torch.Tensor, x: torch.Tensor, 
                             x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Optimal control for topological SB"""
        # Implementation from [1]
        pass
```

### LinearGaussianBridge
```python
class LinearGaussianBridge(Bridge):
    """Linear Gaussian Bridge processes"""
    
    def _optimal_control_impl(self, t: torch.Tensor, x: torch.Tensor, 
                             x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Optimal control for linear Gaussian bridges"""
        # Closed-form solution for linear Gaussian bridges
        pass
```

### OUBridge
```python
class OUBridge(LinearGaussianBridge):
    """Ornstein-Uhlenbeck Bridge"""
    
    def __init__(self, alpha: float, beta: float):
        sde = OrnsteinUhlenbeckSDE(alpha, beta)
        super().__init__(sde)
        self.alpha = alpha
        self.beta = beta
    
    def _optimal_control_impl(self, t: torch.Tensor, x: torch.Tensor, 
                             x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Optimal control for OU bridge"""
        # Closed-form solution from current implementation
        sinh_term = torch.sinh(self.alpha * (1 - t))
        exp_term = torch.exp(self.alpha * (1 - t))
        return (self.alpha / sinh_term) * (x1 - exp_term * x)
```

## 4. Set Bridge Classes After Definition

```python
# Set bridge classes after all classes are defined
LinearSDE.bridge_class = LinearGaussianBridge
TopologicalSDE.bridge_class = TopologicalSchrodingerBridge
ConstantDiffusionSDE.bridge_class = LinearGaussianBridge

# Specific bridge classes for concrete SDEs
OrnsteinUhlenbeckSDE.bridge_class = OUBridge
TopologicalOrnsteinUhlenbeckSDE.bridge_class = TopologicalSchrodingerBridge
```

## 4. Optimal Control Classes

### BridgeOptimalControl
```python
class BridgeOptimalControl(OptimalControl):
    """Abstract base for bridge optimal controls"""
    
    def __init__(self, bridge: Bridge):
        self.bridge = bridge
    
    @abstractmethod
    def __call__(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pass
```

### NeuralControl
```python
class NeuralControl(OptimalControl):
    """Neural network-based control (for learning)"""
    
    def __init__(self, network: torch.nn.Module):
        self.network = network
    
    def __call__(self, t: torch.Tensor, x: torch.Tensor, 
                 x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # Concatenate time, state, and endpoints as input
        inputs = torch.cat([t.unsqueeze(-1), x, x0, x1], dim=-1)
        return self.network(inputs)
```

## 5. Solver Classes

### Solver (Base Class)
```python
class Solver(ABC):
    """Abstract base class for numerical SDE solvers"""
    
    @abstractmethod
    def solve(self, sde: SDE, x0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """Solve SDE and return trajectory"""
        pass
```

### EulerMaruyamaSolver
```python
class EulerMaruyamaSolver(Solver):
    """Euler-Maruyama numerical solver"""
    
    def __init__(self, dt: float = 0.01):
        self.dt = dt
    
    def solve(self, sde: SDE, x0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """Solve SDE using Euler-Maruyama method"""
        n_steps = len(t_span)
        x = torch.zeros(n_steps, *x0.shape, device=sde.device)
        x[0] = x0
        
        for i in range(1, n_steps):
            t = t_span[i-1]
            dt = t_span[i] - t_span[i-1]
            
            drift, diffusion = sde.forward(t, x[i-1])
            noise = torch.randn_like(x[i-1]) * torch.sqrt(dt)
            
            x[i] = x[i-1] + drift * dt + diffusion * noise
        
        return x
```

## 6. Training and Loss Classes

### FlowMatchingLoss
```python
class FlowMatchingLoss:
    """Conditional flow matching loss for training"""
    
    def __init__(self, bridge: Bridge, solver: Solver):
        self.bridge = bridge
        self.solver = solver
    
    def __call__(self, network: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        """Compute flow matching loss"""
        # Implementation of conditional flow matching loss
        pass
```

### WassersteinDistance
```python
class WassersteinDistance:
    """1- and 2-Wasserstein distance computation"""
    
    @staticmethod
    def wasserstein_1d(p_samples: torch.Tensor, q_samples: torch.Tensor, 
                       p: int = 1) -> torch.Tensor:
        """Compute p-Wasserstein distance between empirical distributions"""
        # Sort samples
        p_sorted = torch.sort(p_samples, dim=0)[0]
        q_sorted = torch.sort(q_samples, dim=0)[0]
        
        # Compute Wasserstein distance
        return torch.mean(torch.abs(p_sorted - q_sorted) ** p) ** (1/p)
```

## 7. Training Manager

### TrainingManager
```python
class TrainingManager:
    """Manages training of parametrized control models"""
    
    def __init__(self, bridge: Bridge, solver: Solver, 
                 loss_fn: FlowMatchingLoss, optimizer: torch.optim.Optimizer):
        self.bridge = bridge
        self.solver = solver
        self.loss_fn = loss_fn
        self.optimizer = optimizer
    
    def train_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Single training step"""
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.network, batch)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def evaluate(self, test_samples: torch.Tensor) -> dict:
        """Evaluate model performance"""
        # Compute Wasserstein distances and other metrics
        pass
```

## 8. Path Sampling and Evaluation

### PathSampler
```python
class PathSampler:
    """Samples paths from bridges and SDEs"""
    
    def __init__(self, solver: Solver):
        self.solver = solver
    
    def sample_bridge_paths(self, bridge: Bridge, n_samples: int, 
                           n_steps: int, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Sample multiple bridge paths"""
        # Implementation for sampling bridge paths with endpoints
        pass
    
    def sample_sde_paths(self, sde: SDE, x0: torch.Tensor, n_samples: int,
                        t_span: torch.Tensor) -> torch.Tensor:
        """Sample multiple SDE paths"""
        # Implementation for sampling SDE paths
        pass
```

## Usage Examples

```python
# Create Laplacian for topological structure
L = torch.tensor([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])  # 3-node graph

# Create topological OU SDE
topological_ou = TopologicalOrnsteinUhlenbeckSDE(alpha=1.0, beta=1.0, L=L)

# Get bridge
bridge = topological_ou.bridge()

# Diagonalize (computes eigenvalues/eigenvectors)
diagonal_sde = topological_ou.diagonalize()

# Access eigenvalues/eigenvectors
print(f"Eigenvalues: {diagonal_sde.eigenvalues}")
print(f"Eigenvectors shape: {diagonal_sde.eigenvectors.shape}")

# Transform between spaces
x_original = torch.randn(3)
x_diagonal = diagonal_sde.transform_to_diagonal_space(x_original)
x_back = diagonal_sde.transform_to_original_space(x_diagonal)

# Work in diagonal space (each component evolves independently)
drift_diagonal = diagonal_sde.drift(t, x_diagonal)  # Component-wise drift
```

## Key Improvements

1. **Simplified API**: No more `condition()` method, just `bridge()` property
2. **Direct Access**: `sde.bridge()` returns the bridge instance directly
3. **No Bridge Classes**: No need to set bridge classes after definition
4. **Cleaner Inheritance**: Each SDE type creates its appropriate bridge in `__init__`
5. **Type Safety**: Bridge instances are created at construction time
6. **Functional Style**: Bridges remain stateless with endpoints as method arguments 