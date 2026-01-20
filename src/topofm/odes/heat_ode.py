import torch
from torch import Tensor

from topofm.odes.ode import ODE, SpectralBaseODE
from topofm.odes.trivial_ode import TrivialODE
from topofm.spaces import Space
from topofm.distributions.covariance import Covariance


class _PositiveEigenvalueSpectralHeatODE(ODE):
    def __init__(self, kappa: float, eigvals: Tensor) -> None:
        """
        Args:
            kappa: float
            eigvals: (E)
        """
        super().__init__()
        self.D = -kappa * eigvals

    def b(self, t: Tensor, xt: Tensor) -> Tensor:
        """
        b(t, x_t) = D x_t

        Args:
            t: (..., 1)
            x: (..., d)
        Returns:
            b: (..., d)
        """
        return self.D * xt

    def s(self, t: Tensor, v: Tensor) -> Tensor:
        """
        s(t)v(t, x_t) = D exp(-t D) / sinh(D) v(t, x_t)

        Args:
            t: (..., 1)
            v: (..., d)
        Returns:
            sv: (..., d)
        """
        return (
            (self.D * torch.exp(-t * self.D) / torch.sinh(self.D)) * v
        )

    def v(self, x0: Tensor, x1: Tensor) -> Tensor:
        """
        v(t, x_t) = x_1 - exp(D) x_0

        Args:
            x0: (..., d)
            x1: (..., d)
        Returns:
            v: (..., d)
        """
        return (
            x1 - 
            torch.exp(self.D) * x0
        )

    def sv(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """
        s(t) v(t, x_t) = D exp(-t D) / sinh(D) v(t, x_t)

        Args:
            t: (..., 1)
            x0: (..., d)
            x1: (..., d)
        Returns:
            sv: (..., d)
        """
        return (
            ((self.D * torch.exp(-t * self.D) / torch.sinh(self.D)) * x1) - 
            ((self.D * torch.exp((1.0 - t) * self.D) / torch.sinh(self.D)) * x0)
        )
    
    def x(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """
        x_t = (sinh((1 - t) D) x_0 + sinh(t D) x_1) / sinh(D)

        Args:
            t: (..., 1)
            x0: (..., d)
            x1: (..., d)
        Returns:
            x: (..., d)
        """
        return (
            (
                (torch.sinh((1 - t) * self.D) * x0) + 
                (torch.sinh(t * self.D) * x1)
            ) /
            torch.sinh(self.D)
        )
    
    def _c_transform(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x0: (..., d)
            x1: (..., d)
        Returns:
            z0: (..., d)
            z1: (..., d)
        """
        return (
            (self.D * torch.exp(self.D) / torch.sinh(self.D)).sqrt() * x0,
            (self.D * torch.exp(-self.D) / torch.sinh(self.D)).sqrt() * x1
        )

    def c(self, x0: Tensor, x1: Tensor) -> Tensor:
        """
        c(x_0, x_1) = ||x_1 - exp(D) x_0||^2_{2 D exp(-D) / sinh(D)}

        Args:
            x0: (..., n, d)
            x1: (..., m, d)
        Returns:
            c: (..., n, m)
        """
        z0, z1 = self._c_transform(x0, x1)
        return torch.cdist(z0, z1, p=2).square()

    def _Phi10_stable(self, x: Covariance):
        # D is your vector of -kappa * eigvals
        # We want to transport from t=1 to t=0 using exp(-D)
        
        # 1. Determine precision-specific constants
        if x.matrix.dtype == torch.float64:
            eps = 1e-300      # Much smaller log-offset for double precision
            max_log = 700.0   # Safe limit for exp() in float64
        else:
            eps = 1e-40       # Standard for float32
            max_log = 88.0    # Safe limit for exp() in float32

        # 2. Decompose into Log-Absolute and Signs
        # matrix: (D, D)
        log_abs_Sigma = torch.log(x.matrix.abs() + eps)
        signs = x.matrix.sign()
        
        # 3. Perform the 'Sandwich' in log-space (Addition/Subtraction)
        # log(exp(-Di) * Sigma_ij * exp(-Dj)) = log(Sigma_ij) - Di - Dj
        # self.D is (D,) -> unsqueeze to (D, 1) and (1, D) for broadcasting
        log_res = log_abs_Sigma - self.D.unsqueeze(1) - self.D.unsqueeze(0)
        
        # 4. Optional: Clamp to prevent Inf if you need to return to linear space
        # This acts as a spectral filter for extremely high-heat components
        log_res = torch.clamp(log_res, max=max_log)
        
        # Reconstruct the matrix
        transported_mat = log_res.exp() * signs
    
        # REPAIR STEP: Force symmetry and clip negative eigenvalues
        # This is essential because the transport can "tilt" the matrix 
        # out of the PSD cone due to floating point noise.
        
        # 1. Force Symmetry
        transported_mat = 0.5 * (transported_mat + transported_mat.mT)
        
        # 2. Spectral Clipping
        # Use eigh because we know the matrix is symmetric now
        vals, vecs = torch.linalg.eigh(transported_mat)
        
        # Use the threshold we discussed earlier based on dtype
        tol = 1e-6 if transported_mat.dtype == torch.float32 else 1e-15
        
        # Clamp to your floor (tol) to ensure it stays invertible/PSD
        vals = torch.clamp(vals, min=tol)
        
        # 3. Reconstruct
        fixed_mat = vecs @ (vals.unsqueeze(-1) * vecs.mT)
        
        return Covariance(fixed_mat)


    def Phi10(self, x: Tensor | Covariance) -> Tensor | Covariance:
        Phi_st = torch.exp(-self.D)
        if isinstance(x, Covariance):
            # return self._Phi10_stable(x)
            return Covariance(
                torch.einsum('i,ij,j->ij', Phi_st, x.matrix, Phi_st)
            )
        else:
            return Phi_st * x


class _SpectralHeatODE(ODE):

    def __init__(self, kappa: float, eigvals: Tensor) -> None:
        """
        Args:
            kappa: float
            eigvals: (E,)
        """
        super().__init__()
        self.zero_eigenvalue_ode = TrivialODE()
        self.unsafe_positive_eigenvalue_ode = _PositiveEigenvalueSpectralHeatODE(kappa=kappa, eigvals=eigvals)

        # Create safe variant
        self.zero_eigenvalue_mask = ((kappa * eigvals) == 0.0)
        safe_eigvals = torch.where(eigvals == 0.0, torch.ones_like(eigvals), eigvals)
        self.safe_positive_eigenvalue_ode = _PositiveEigenvalueSpectralHeatODE(kappa=kappa, eigvals=safe_eigvals)


    def b(self, t: Tensor, xt: Tensor) -> Tensor:
        return torch.where(
            self.zero_eigenvalue_mask,
            self.zero_eigenvalue_ode.b(t, xt),
            self.safe_positive_eigenvalue_ode.b(t, xt)
        )

    def s(self, t: Tensor, v: Tensor) -> Tensor:
        return torch.where(
            self.zero_eigenvalue_mask,
            self.zero_eigenvalue_ode.s(t, v),
            self.safe_positive_eigenvalue_ode.s(t, v)
        )

    def v(self, x0: Tensor, x1: Tensor) -> Tensor:
        return torch.where(
            self.zero_eigenvalue_mask,
            self.zero_eigenvalue_ode.v(x0, x1),
            self.safe_positive_eigenvalue_ode.v(x0, x1)
        )

    def x(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        return torch.where(
            self.zero_eigenvalue_mask,
            self.zero_eigenvalue_ode.x(t, x0, x1),
            self.safe_positive_eigenvalue_ode.x(t, x0, x1)
        )

    def c(self, x0: Tensor, x1: Tensor) -> Tensor:
        z0, z1 = self.safe_positive_eigenvalue_ode._c_transform(x0, x1)
        z0 = torch.where(self.zero_eigenvalue_mask, x0, z0)
        z1 = torch.where(self.zero_eigenvalue_mask, x1, z1)
        return torch.cdist(z0, z1, p=2).square()

    def Phi10(self, x: Tensor | Covariance) -> Tensor | Covariance:
        return self.unsafe_positive_eigenvalue_ode.Phi10(x)


class HeatODE(SpectralBaseODE):
    def __init__(self, kappa: float, space: Space) -> None:
        """
        Args:
            kappa: float
            space: The space.
        """
        base_ode = _SpectralHeatODE(kappa=kappa, eigvals=space.eigvals)
        super().__init__(base_ode=base_ode, space=space)
        