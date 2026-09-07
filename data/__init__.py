"""d3plot extraction and PyTorch datasets for the spherical TNT PINN.

Modules:
    - extract_d3plot      : LS-DYNA d3plot -> 1D radial (rho, u, P) profiles
    - d3plot_dataset      : torch view of extracted/ (bilinear state_at, R_s/R_c)
    - multi_radius_dataset: Hopkinson-Cranz scaled unified multi-radius view
"""
