"""L_IC — Initial-condition loss at t = t_sep.

Supervises the MainNet output against the IC field (ρ, u, P)(r, t_sep)
provided by :class:`data.ic_builder.ICBuilder`.

Density and pressure are compared in log-space (log-MSE) to handle the
large dynamic range near the shock; velocity is compared linearly.

    L_IC = (1/N) Σ_k [w_rho * (log_rho_err)^2
                      + w_u   * (u_err / u_ref)^2
                      + w_P   * (log_P_err)^2]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ICLoss(nn.Module):
    """Initial-condition loss.

    Parameters
    ----------
    main_net : MainNet
        The primary flow-field network.
    ic_data : dict
        Output of :meth:`data.ic_builder.ICBuilder.as_tensors`; keys:
        'r', 't', 'rho', 'u', 'P'.
    rho_ref, u_ref, P_ref : float
        Normalisation scales for log-MSE stabilisation.
    w_rho, w_u, w_P : float
        Per-variable MSE weights.
    """

    def __init__(
        self,
        main_net: nn.Module,
        ic_data: dict,
        rho_ref: float = 1.225,
        u_ref:   float = 340.0,
        P_ref:   float = 101325.0,
        w_rho:   float = 1.0,
        w_u:     float = 0.5,
        w_P:     float = 1.0,
    ) -> None:
        super().__init__()
        self.main_net = main_net
        self.ic_data  = ic_data
        self.rho_ref  = rho_ref
        self.u_ref    = u_ref
        self.P_ref    = P_ref
        self.w_rho    = w_rho
        self.w_u      = w_u
        self.w_P      = w_P

    def forward(self) -> torch.Tensor:
        r = self.ic_data["r"].requires_grad_(True)
        t = self.ic_data["t"].requires_grad_(True)

        rho_pred, u_pred, P_pred = self.main_net(r, t)

        rho_true = self.ic_data["rho"]
        u_true   = self.ic_data["u"]
        P_true   = self.ic_data["P"]

        eps = 1e-10
        log_rho_err = torch.log(rho_pred + eps) - torch.log(rho_true + eps)
        log_P_err   = torch.log(P_pred   + eps) - torch.log(P_true   + eps)
        u_err       = (u_pred - u_true) / self.u_ref

        loss = (
            self.w_rho * (log_rho_err ** 2).mean()
            + self.w_u  * (u_err       ** 2).mean()
            + self.w_P  * (log_P_err   ** 2).mean()
        )
        return loss
