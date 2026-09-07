"""Dataset wrapper around the ``extracted/`` artefacts produced by
``data.extract_d3plot``.

Loads ``raw_line.npz`` (full 1D state on (t, x) grid) plus the three
trajectory CSVs and ``metadata.json`` into torch tensors.  Provides:

* bilinear ``state_at(t, x) → (rho, u, P)`` for arbitrary tensor queries
* linear-in-time ``R_c(t)``, ``R_s(t)``, ``Pbar_p(t)``
* ``dPbar_p_dt(t)`` from a 4th-order central-difference buffer pre-built at
  init time (avoids per-step recomputation)

All tensors live on CPU by default with dtype float32; move with ``.to(device)``
or pass ``device`` to the constructor.

Use ``--check`` mode for a quick visual sanity report:

    python -m data.d3plot_dataset --check extracted/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ============================================================ helpers


def _central_diff_4th(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Fourth-order central difference dy/dt on a (possibly non-uniform) grid.

    Interior (i in [2, N-3]):  4th-order central
    Edges (i = 0, 1, N-2, N-1): np.gradient fallback (2nd-order)

    For a non-uniform grid the 4-stencil weights would technically depend on
    spacing.  Since extracted d3plot states are typically uniform in time, we
    use the uniform-grid weights and the largest local error stays O(h^4).
    """
    n = len(t)
    if n < 5:
        return np.gradient(y, t, edge_order=min(n - 1, 2))
    dy = np.gradient(y, t, edge_order=2)
    h = np.diff(t)
    h_min, h_max = h.min(), h.max()
    if (h_max - h_min) / max(h_max, 1e-30) < 1e-3:
        # Uniform grid: apply 4th-order central to interior
        h_uniform = float(h.mean())
        for i in range(2, n - 2):
            dy[i] = (-y[i + 2] + 8 * y[i + 1] - 8 * y[i - 1] + y[i - 2]) / (12.0 * h_uniform)
    return dy


def _interp1d_torch(x_query: torch.Tensor, x_table: torch.Tensor, y_table: torch.Tensor) -> torch.Tensor:
    """Linear interpolation: ``y_table`` indexed by ``x_table`` evaluated at ``x_query``.

    All tensors 1D in the table dimension; ``x_query`` may have any shape.
    Out-of-range queries are clamped to the boundary value (``np.interp``-style).
    ``x_query`` is auto-moved to the table's device so callers do not have to
    juggle CPU/GPU placement.
    """
    if x_query.device != x_table.device:
        x_query = x_query.to(x_table.device)
    out_shape = x_query.shape
    xq = x_query.flatten()
    n = x_table.numel()
    # searchsorted returns the right insertion index in [0, n]
    idx = torch.searchsorted(x_table, xq).clamp(1, n - 1)
    x0 = x_table[idx - 1]; x1 = x_table[idx]
    y0 = y_table[idx - 1]; y1 = y_table[idx]
    w = ((xq - x0) / (x1 - x0).clamp(min=1e-30)).clamp(0.0, 1.0)
    yq = (1 - w) * y0 + w * y1
    return yq.view(out_shape)


# ============================================================ Dataset


class D3plotLineDataset:
    """Torch-friendly view of ``extracted/*`` artefacts.

    Stores the 1D state along the +x axis (radial direction in the
    spherical-equivalent model) as ``t_arr``, ``x_arr`` (= r), and the
    ``rho_grid`` / ``u_grid`` / ``P_grid`` (N_t, N_x) tensors.  Also reads
    the Sec.4.2.1 separation metadata and the Taylor-Sadovsky slope curve
    ``r_c_taylor_sadovsky.csv`` when present (written by the updated
    extractor); these are required by Phase-A losses but optional for
    legacy callers.
    """

    REQUIRED_FILES = (
        "raw_line.npz",
        "shock.csv",
        "contact.csv",
        "product_pressure.csv",
        "metadata.json",
    )

    def __init__(self, extracted_dir: str | Path, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        self.dir = Path(extracted_dir)
        self.device = torch.device(device)
        self.dtype = dtype

        for f in self.REQUIRED_FILES:
            p = self.dir / f
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing {p}. Run `python -m data.extract_d3plot --d3plot "
                    f"<sim_data> --out {self.dir}` first."
                )

        with open(self.dir / "metadata.json", encoding="utf-8") as f:
            self.metadata: dict = json.load(f)

        raw = np.load(self.dir / "raw_line.npz")
        self.t_arr   = torch.as_tensor(raw["t"], dtype=dtype, device=device)
        self.x_arr   = torch.as_tensor(raw["x"], dtype=dtype, device=device)
        self.rho_grid = torch.as_tensor(raw["rho"], dtype=dtype, device=device)
        self.u_grid   = torch.as_tensor(raw["u"],   dtype=dtype, device=device)
        self.P_grid   = torch.as_tensor(raw["P"],   dtype=dtype, device=device)

        sh = np.loadtxt(self.dir / "shock.csv", delimiter=",", skiprows=1)
        ct = np.loadtxt(self.dir / "contact.csv", delimiter=",", skiprows=1)
        pp = np.loadtxt(self.dir / "product_pressure.csv", delimiter=",", skiprows=1)
        if sh.ndim == 1: sh = sh[None, :]
        if ct.ndim == 1: ct = ct[None, :]
        if pp.ndim == 1: pp = pp[None, :]

        self.t_traj = torch.as_tensor(sh[:, 0], dtype=dtype, device=device)
        self.R_s_traj = torch.as_tensor(sh[:, 1], dtype=dtype, device=device)
        self.R_c_traj = torch.as_tensor(ct[:, 1], dtype=dtype, device=device)
        self.Pbar_p_traj = torch.as_tensor(pp[:, 1], dtype=dtype, device=device)

        # 2nd-order gradient for D_s — more stable than 4th-order central diff
        # on SG-smoothed trajectories.  Precomputed once, stored as buffer.
        Rs_np = sh[:, 1]; t_np = sh[:, 0]
        dRs_dt_np = np.gradient(Rs_np, t_np, edge_order=2)
        self._dRs_dt_traj = torch.as_tensor(dRs_dt_np, dtype=dtype, device=device)

        # dP̄_p/dt buffer (4th-order central diff)
        dPbar_dt_np = _central_diff_4th(pp[:, 1], pp[:, 0])
        self.dPbar_p_dt_traj = torch.as_tensor(dPbar_dt_np, dtype=dtype, device=device)

        # Geometry / timing metadata
        # ``R_0`` is the spherical TNT charge radius (m) -- the natural
        # primary geometric parameter; older snapshots may still carry the
        # planar-style ``L_0`` (cube half-edge), in which case the extractor
        # should be re-run.  We accept either key for backwards compatibility.
        if "R_0" in self.metadata:
            self.R_0 = float(self.metadata["R_0"])
        elif "L_0" in self.metadata:
            # Legacy cube-style snapshot (L_0 = cube half-edge).  Treating it as
            # the spherical charge radius gives ~1.9x wrong mass / R_c / t_sep.
            print(f"[WARN] {self.dir}/metadata.json has legacy 'L_0' "
                  f"(cube half-edge) but no 'R_0'; using R_0 = L_0 = "
                  f"{self.metadata['L_0']:.4g} m.  Re-run the extractor for a "
                  f"spherical R_0.")
            self.R_0 = float(self.metadata["L_0"])
        else:
            self.R_0 = None
        # Charge-surface coordinate in this dataset's units (physical metres).
        # MultiRadiusDataset redefines Z_R0 in Hopkinson-scaled units; the
        # single-radius dataset is physical, so the surface is just R_0.
        self.Z_R0 = None if self.R_0 is None else float(self.R_0)
        self.x_end  = float(self.metadata["x_end"])
        self.t_0    = float(self.metadata["t_0"])
        self.t_end  = float(self.metadata["t_end"])
        self.rho_TNT = float(self.metadata.get("rho_TNT", 1630.0))
        self.rho_CJ = float(self.metadata.get("rho_CJ", 0.0))
        self.P_CJ   = float(self.metadata.get("P_CJ", 0.0))
        self.u_CJ   = float(self.metadata.get("u_CJ", 0.0))

        # Sec.4.2.1 separation state at the connection point (t_sep, R_c).
        # ``R_c_sep`` avoids shadowing the linear-interp R_c(t) trajectory
        # method below.
        self.R_c_sep = self.metadata.get("R_c",   None)
        self.t_sep   = self.metadata.get("t_sep", None)
        self.P_x     = self.metadata.get("P_x",   None)
        self.u_x     = self.metadata.get("u_x",   None)
        self.rho_x   = self.metadata.get("rho_x", None)
        self.V_x     = self.metadata.get("V_x",   None)

        # Taylor-Sadovsky slope table (optional)
        ts_path = self.dir / "r_c_taylor_sadovsky.csv"
        if ts_path.exists():
            ts = np.loadtxt(ts_path, delimiter=",", skiprows=1)
            if ts.ndim == 1:
                ts = ts[None, :]
            self.t_traj_rc = torch.as_tensor(ts[:, 0], dtype=dtype, device=device)
            self.r_traj_rc = torch.as_tensor(ts[:, 1], dtype=dtype, device=device)
        else:
            self.t_traj_rc = None
            self.r_traj_rc = None

    # ============================================================ accessors

    def state_at(self, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bilinear (t, x) → (rho, u, P).  ``t`` and ``x`` must broadcast.

        Returns three tensors of the broadcast shape.  Out-of-range clamped.
        Inputs are auto-moved to ``self.device``; outputs stay on that device.
        """
        if t.device != self.device:
            t = t.to(self.device)
        if x.device != self.device:
            x = x.to(self.device)
        if t.shape != x.shape:
            t, x = torch.broadcast_tensors(t, x)
        out_shape = t.shape

        t_flat = t.flatten()
        x_flat = x.flatten()

        n_t = self.t_arr.numel()
        n_x = self.x_arr.numel()
        i_t = torch.searchsorted(self.t_arr, t_flat).clamp(1, n_t - 1)
        i_x = torch.searchsorted(self.x_arr, x_flat).clamp(1, n_x - 1)

        t0 = self.t_arr[i_t - 1]; t1 = self.t_arr[i_t]
        x0 = self.x_arr[i_x - 1]; x1 = self.x_arr[i_x]
        wt = ((t_flat - t0) / (t1 - t0).clamp(min=1e-30)).clamp(0.0, 1.0)
        wx = ((x_flat - x0) / (x1 - x0).clamp(min=1e-30)).clamp(0.0, 1.0)

        def bilinear(field: torch.Tensor) -> torch.Tensor:
            f00 = field[i_t - 1, i_x - 1]; f01 = field[i_t - 1, i_x]
            f10 = field[i_t,     i_x - 1]; f11 = field[i_t,     i_x]
            f_t0 = (1 - wx) * f00 + wx * f01
            f_t1 = (1 - wx) * f10 + wx * f11
            return ((1 - wt) * f_t0 + wt * f_t1).view(out_shape)

        return bilinear(self.rho_grid), bilinear(self.u_grid), bilinear(self.P_grid)

    def R_c(self, t: torch.Tensor) -> torch.Tensor:
        """Linear-in-time R_c(t).  Out-of-range clamped."""
        return _interp1d_torch(t, self.t_traj, self.R_c_traj)

    def R_s(self, t: torch.Tensor) -> torch.Tensor:
        """Linear-in-time R_s(t)."""
        return _interp1d_torch(t, self.t_traj, self.R_s_traj)

    def dPbar_p_dt(self, t: torch.Tensor) -> torch.Tensor:
        """dP̄_p/dt(t) (Pa/s); 4th-order central diff buffer, linearly interp."""
        return _interp1d_torch(t, self.t_traj, self.dPbar_p_dt_traj)

    def r_c_taylor_sadovsky(self, t: torch.Tensor) -> torch.Tensor:
        """Linear-interp r_c(t) from the Taylor-Sadovsky CSV.

        Raises if the table has not been written by the extractor.
        """
        if self.t_traj_rc is None:
            raise RuntimeError(
                f"Taylor-Sadovsky table missing.  Re-run the extractor with the "
                f"updated `data.extract_d3plot` so it writes "
                f"{self.dir}/r_c_taylor_sadovsky.csv."
            )
        return _interp1d_torch(t, self.t_traj_rc, self.r_traj_rc)

    # ============================================================ training-sample helpers

    def sample_grid_points(self, n: int, seed: Optional[int] = None
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random sample ``n`` (t, x) points from the full grid.

        Returns (t, x, rho, u, P), each shape (n, 1).  All tensors on the
        dataset's device/dtype.  Useful for L_data,main during stage 1.
        """
        if seed is not None:
            g = torch.Generator(device="cpu").manual_seed(seed)
            i_t = torch.randint(0, self.t_arr.numel(), (n,), generator=g)
            i_x = torch.randint(0, self.x_arr.numel(), (n,), generator=g)
        else:
            i_t = torch.randint(0, self.t_arr.numel(), (n,))
            i_x = torch.randint(0, self.x_arr.numel(), (n,))
        t   = self.t_arr[i_t].view(-1, 1)
        x   = self.x_arr[i_x].view(-1, 1)
        rho = self.rho_grid[i_t, i_x].view(-1, 1)
        u   = self.u_grid[i_t,   i_x].view(-1, 1)
        P   = self.P_grid[i_t,   i_x].view(-1, 1)
        return t, x, rho, u, P


# ============================================================ CLI check


def _check(extracted_dir: Path) -> None:
    """Print structural / physical sanity for the extracted dataset."""
    ds = D3plotLineDataset(extracted_dir)
    print(f"\n=== D3plotLineDataset @ {extracted_dir} ===")
    print(f"  t:   shape={tuple(ds.t_arr.shape)}  "
          f"range=[{ds.t_arr.min():.3e}, {ds.t_arr.max():.3e}] s")
    print(f"  x:   shape={tuple(ds.x_arr.shape)}  "
          f"range=[{ds.x_arr.min():.4f}, {ds.x_arr.max():.4f}] m")
    print(f"  rho: shape={tuple(ds.rho_grid.shape)}  "
          f"range=[{ds.rho_grid.min():.3e}, {ds.rho_grid.max():.3e}] kg/m³")
    print(f"  u:   shape={tuple(ds.u_grid.shape)}  "
          f"range=[{ds.u_grid.min():.3e}, {ds.u_grid.max():.3e}] m/s")
    print(f"  P:   shape={tuple(ds.P_grid.shape)}  "
          f"range=[{ds.P_grid.min():.3e}, {ds.P_grid.max():.3e}] Pa")

    print(f"\n  R_c trajectory: [{ds.R_c_traj.min()*1e3:.2f}, "
          f"{ds.R_c_traj.max()*1e3:.2f}] mm  monotone? "
          f"{(torch.diff(ds.R_c_traj) >= -1e-9).all().item()}")
    print(f"  R_s trajectory: [{ds.R_s_traj.min()*1e3:.2f}, "
          f"{ds.R_s_traj.max()*1e3:.2f}] mm  monotone? "
          f"{(torch.diff(ds.R_s_traj) >= -1e-9).all().item()}")
    print(f"  P̄_p trajectory: [{ds.Pbar_p_traj.min():.3e}, "
          f"{ds.Pbar_p_traj.max():.3e}] Pa  decreasing? "
          f"{(torch.diff(ds.Pbar_p_traj) <= 1e-9).all().item()}")

    print(f"\n  metadata: R_0={ds.R_0*1e3:.1f} mm  t_0={ds.t_0:.3e} s  "
          f"t_end={ds.t_end:.3e} s")
    print(f"            rho_CJ={ds.rho_CJ:.1f} kg/m³  P_CJ={ds.P_CJ/1e9:.2f} GPa")

    # Polytropic exponent regression: log P̄_p vs log R_c
    ok = (ds.R_c_traj > 0) & (ds.Pbar_p_traj > 0)
    if ok.sum() > 5:
        log_R = torch.log(ds.R_c_traj[ok]).cpu().numpy()
        log_P = torch.log(ds.Pbar_p_traj[ok]).cpu().numpy()
        slope, intercept = np.polyfit(log_R, log_P, 1)
        print(f"\n  log P̄_p ~ {slope:+.3f} log R_c + {intercept:.3f}")
        if -3.5 < slope < -2.5:
            print(f"  γ_eff ≈ {-slope:.2f} -- consistent with Landau-Stanyukovich γ_eff = 3 ✓")
        else:
            print(f"  WARNING: slope outside [-3.5, -2.5]; γ_eff = 3 may not hold")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", type=Path, required=True,
                   help="Directory with raw_line.npz / shock.csv / contact.csv / "
                        "product_pressure.csv / metadata.json")
    args = p.parse_args()
    _check(args.check)


if __name__ == "__main__":
    main()
