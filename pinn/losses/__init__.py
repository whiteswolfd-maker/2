"""Four-loss PINN: L_IC, L_PDE, L_BC, L_RH."""

from .ic_loss  import ICLoss
from .pde_loss import PDELoss
from .bc_loss  import BCLoss
from .rh_loss  import RHLoss

__all__ = ["ICLoss", "PDELoss", "BCLoss", "RHLoss"]
