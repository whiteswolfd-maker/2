"""Loss modules for the spherical two-network PINN.

Phase A (DetonationNet):
    * DetonationDataLoss     L_data,A
    * DetonationPDELoss      L_PDE,A
    * DetonationICLoss       L_IC,A
    * DetonationSlopeBCLoss  L_BC,A,slope

Phase B (AirShockNet):
    * AirShockDataLoss       L_data,B
    * AirShockICLoss         L_IC,B
    * AirShockPDELoss        L_PDE,B
    * AirShockRHLoss         L_RH,B
    * AirShockOutflowLoss    L_BC,B,outflow
"""

from .air_shock_loss import (
    AirShockDataLoss,
    AirShockICLoss,
    AirShockOutflowLoss,
    AirShockPDELoss,
    AirShockRHLoss,
)
from .detonation_loss import (
    DetonationDataLoss,
    DetonationICLoss,
    DetonationPDELoss,
    DetonationSlopeBCLoss,
)

__all__ = [
    # Phase A
    "DetonationDataLoss",
    "DetonationICLoss",
    "DetonationPDELoss",
    "DetonationSlopeBCLoss",
    # Phase B
    "AirShockDataLoss",
    "AirShockICLoss",
    "AirShockOutflowLoss",
    "AirShockPDELoss",
    "AirShockRHLoss",
]
