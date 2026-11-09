"""Elastic components referenced to their connection deformation at formation."""

import numpy as np
from scipy.sparse import csr_matrix


class FormedComponent:
    """Energy (H u - e_birth)^T D (H u - e_birth)/2 on a soil mesh.

    H samples connection deformations, not the tunnel displacement alone.
    D and the deformation coordinates must be work conjugate: for axial
    extension in metres, D is axial stiffness in kN/m and force is in kN.
    The stored reference is fixed after formation. Existing prestressed
    components require their own reference; resetting them is not formation.
    A changed mesh requires rebuilding H while retaining this reference.
    """

    def __init__(self, compatibility, stiffness, reference):
        self.compatibility = csr_matrix(compatibility, dtype=float)
        self.stiffness = np.array(stiffness, dtype=float, copy=True)
        self.reference = np.array(reference, dtype=float, copy=True)
        n = self.compatibility.shape[0]
        if self.stiffness.shape != (n, n) or self.reference.shape != (n,):
            raise ValueError('Component stiffness and reference must match its deformation coordinates.')
        if (not np.all(np.isfinite(self.stiffness)) or
                not np.all(np.isfinite(self.reference)) or
                not np.all(np.isfinite(self.compatibility.data))):
            raise ValueError('Component data must be finite.')
        if not np.allclose(self.stiffness, self.stiffness.T, rtol=1e-12, atol=1e-12):
            raise ValueError('Elastic component stiffness must be symmetric.')
        if np.linalg.eigvalsh(self.stiffness)[0] < -1e-12 * max(np.linalg.norm(self.stiffness), 1.):
            raise ValueError('Elastic component stiffness must be positive semidefinite.')

    @classmethod
    def from_state(cls, compatibility, stiffness, displacement):
        """Install without prestress in the recovered current soil state."""
        H = csr_matrix(compatibility, dtype=float)
        return cls(H, stiffness, H @ displacement)

    def assembled_terms(self):
        """Return added global stiffness and the equivalent reference load."""
        H = self.compatibility
        return (H.T @ csr_matrix(self.stiffness) @ H,
                np.asarray(H.T @ (self.stiffness @ self.reference)))

    def force(self, displacement):
        """Return component restoring forces in its deformation coordinates."""
        return self.stiffness @ (self.compatibility @ displacement - self.reference)
