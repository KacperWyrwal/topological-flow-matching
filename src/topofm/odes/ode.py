from abc import ABC, abstractmethod
from torch import Tensor
from topofm.frames.frame import Frame
from topofm.distributions.covariance import Covariance


class ODE(ABC):
    """
    Models the ODE dx_t = b(t, x_t) dt, 
    and its bridge dx_t = [b(t, x_t) + s(t)v(t, x_t)] dt.
    """
    
    @abstractmethod
    def b(self, t: Tensor, xt: Tensor) -> Tensor:
        """
        b(t, x_t) in the ODE dx_t = b(t, x_t) dt.

        Args:
            t: (..., 1)
            xt: (..., d)
        Returns:
            b: (..., d)
        """
        raise NotImplementedError

    def s(self, t: Tensor, v: Tensor) -> Tensor:
        """
        Apply to v(t, x_t) to get s(t) v(t, x_t) in the bridge dx_t = [b(t, x_t) + s(t)v(t, x_t)] dt.

        Args:
            t: (..., 1)
            v: (..., d)
        Returns:
            sv: (..., d)
        """
        raise NotImplementedError

    def v(self, x0: Tensor, x1: Tensor) -> Tensor:
        """
        v(t, x_t) in the bridge dx_t = [b(t, x_t) + s(t)v(t, x_t)] dt.

        Args:
            x0: (..., d)
            x1: (..., d)
        Returns:
            v: (..., d)
        """
        raise NotImplementedError

    def sv(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """
        s(t) v(t, x_t) in the bridge dx_t = [b(t, x_t) + s(t)v(t, x_t)] dt.

        Args:
            t: (..., 1)
            x0: (..., d)
            x1: (..., d)
        Returns:
            sv: (..., d)
        """
        v = self.v(x0=x0, x1=x1)
        sv = self.s(t=t, v=v)
        return sv

    def x(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """
        x_t in the flow of x_0 to x_1 at time t.
        
        Args:
            t: (..., 1)
            x0: (..., d)
            x1: (..., d)
        Returns:
            x: (..., d)
        """
        raise NotImplementedError

    def c(self, x0: Tensor, x1: Tensor) -> Tensor:
        """
        Returns the cost of transforming x_0 to x_1.

        c(x_0, x_1) propto -log p(x_1 | x_0)

        Args:
            x0: (..., n, d)
            x1: (..., m, d)
        Returns:
            c: (..., n, m)
        """
        raise NotImplementedError

    def Phi10(self, x: Tensor | Covariance) -> Tensor | Covariance:
        """
        Apply the flow od the ODE to transport x from time 1 to time 0.

        Args:
            x: (..., d)
        Returns:
            Phi: (..., d)
        """
        raise NotImplementedError


class SpectralBaseODE(ODE):
    """
    Wraps a base ODE performing computations in spectral coordinates
    to perform computations in standard coordinates.

    Args:
        base_ode: The base ODE which operates in the spectral coordinates.
        frame: The frame.
    """
    def __init__(self, base_ode: ODE, frame: Frame) -> None:
        super().__init__()
        self.base_ode = base_ode
        self.frame = frame

    def b(self, t: Tensor, xt: Tensor) -> Tensor:
        xt_spectral = self.frame.ambient_to_spectral(xt)
        b_spectral = self.base_ode.b(t=t, xt=xt_spectral)
        return self.frame.spectral_to_ambient(b_spectral)

    def s(self, t: Tensor, v: Tensor) -> Tensor:
        v_spectral = self.frame.ambient_to_spectral(v)
        s_spectral = self.base_ode.s(t=t, v=v_spectral)
        return self.frame.spectral_to_ambient(s_spectral)

    def v(self, x0: Tensor, x1: Tensor) -> Tensor:
        x0_spectral = self.frame.ambient_to_spectral(x0)
        x1_spectral = self.frame.ambient_to_spectral(x1)
        v_spectral = self.base_ode.v(x0=x0_spectral, x1=x1_spectral)
        return self.frame.spectral_to_ambient(v_spectral)

    def sv(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        x0_spectral = self.frame.ambient_to_spectral(x0)
        x1_spectral = self.frame.ambient_to_spectral(x1)
        sv_spectral = self.base_ode.sv(t=t, x0=x0_spectral, x1=x1_spectral)
        return self.frame.spectral_to_ambient(sv_spectral)

    def x(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        x0_spectral = self.frame.ambient_to_spectral(x0)
        x1_spectral = self.frame.ambient_to_spectral(x1)
        x_spectral = self.base_ode.x(t=t, x0=x0_spectral, x1=x1_spectral)
        return self.frame.spectral_to_ambient(x_spectral)

    def c(self, x0: Tensor, x1: Tensor) -> Tensor:
        x0_spectral = self.frame.ambient_to_spectral(x0)
        x1_spectral = self.frame.ambient_to_spectral(x1)
        # cost is preserved under coordinate change
        c = self.base_ode.c(x0=x0_spectral, x1=x1_spectral)
        return c

    def Phi10(self, x: Tensor | Covariance) -> Tensor | Covariance:
        x_spectral = self.frame.ambient_to_spectral(x)
        Phi_spectral = self.base_ode.Phi10(x=x_spectral)
        return self.frame.spectral_to_ambient(Phi_spectral)
