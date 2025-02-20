def psi_norm(psi, psi_axis, psi_boundary):
    """Normalize psi to [0, 1] between axis and boundary."""
    return (psi - psi_axis) / (psi_boundary - psi_axis)