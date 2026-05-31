"""
3D Source Triangulation from DOA Direction Vectors
====================================================
Finds the single 3D point that is closest (in a least-squares sense)
to all direction rays produced by estimate_directions() in the original
file (tmpcodeSayeretProject.py).

Usage
-----
    python triangulate_source.py

The module also exposes `closest_point_to_rays` for use as a library.
"""

import numpy as np
from tmpcodeSayeretProject import simulate_arrays, estimate_directions


# ---------------------------------------------------------------------------
# Core geometry: closest point to a set of 3D rays
# ---------------------------------------------------------------------------

def closest_point_to_rays(
    origins: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """
    Find the 3D point P that minimises the sum of squared distances
    to a set of infinite lines (rays).

    Each ray i is defined as:
        R_i(t) = origins[i] + t * directions[i]    (t ∈ ℝ)

    The squared distance from a point P to ray i is:
        d²(P, ray_i) = ‖(P - origins[i]) - [(P - origins[i])·d̂_i] d̂_i‖²

    Minimising the total over all rays yields a linear system  A · P = b.

    Parameters
    ----------
    origins    : (K, 3) array  — one origin per ray (array centre)
    directions : (K, 3) array  — unit direction vectors (already normalised)

    Returns
    -------
    P : (3,) array — the closest point
    """
    origins    = np.asarray(origins,    dtype=float)
    directions = np.asarray(directions, dtype=float)

    # Re-normalise just in case
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.where(norms > 0, norms, 1.0)

    K = len(origins)
    # I - d̂ d̂ᵀ  projects out the component along the ray direction.
    # Summing these gives the matrix A and vector b for the normal equations.
    I = np.eye(3)
    A = np.zeros((3, 3))
    b = np.zeros(3)

    for o, d in zip(origins, directions):
        M = I - np.outer(d, d)   # 3×3 projection matrix
        A += M
        b += M @ o

    # Solve A·P = b  (least-squares, handles near-degenerate cases)
    P, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return P


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TRUE_SOURCE = np.array([4.0, 5.0, 3.0])
    C = 343.0

    print("=" * 60)
    print("3D Triangulation — Closest Point to All DOA Rays")
    print("=" * 60)

    # --- Generate the same synthetic arrays as the original demo ---
    arrays = simulate_arrays(
        true_source=TRUE_SOURCE,
        n_arrays=3,
        mics_per_array=8,
        noise_std=1e-6,
        c=C,
        seed=123,
    )

    # --- Get per-array directions from the original solver ---
    directions = estimate_directions(arrays, c=C)

    # --- Print individual DOA results (mirrors the original demo) ---
    print("\nPer-array DOA results:")
    origins    = []
    unit_vecs  = []

    for data in directions:
        idx = data["array_index"]
        true_vec      = TRUE_SOURCE - data["array_center"]
        true_unit_vec = true_vec / np.linalg.norm(true_vec)
        true_az  = np.degrees(np.arctan2(true_unit_vec[1], true_unit_vec[0]))
        true_el  = np.degrees(np.arcsin(true_unit_vec[2]))

        print(f"\n  Array {idx + 1}")
        print(f"    Array Centre   : {np.round(data['array_center'],   2)}")
        print(f"    Est. Azimuth   : {data['azimuth_deg']:>7.3f}°  |  True: {true_az:>7.3f}°")
        print(f"    Est. Elevation : {data['elevation_deg']:>7.3f}°  |  True: {true_el:>7.3f}°")

        origins.append(data["array_center"])
        unit_vecs.append(data["direction_vector"])

    origins   = np.array(origins)
    unit_vecs = np.array(unit_vecs)

    # --- Triangulate ---
    estimated_source = closest_point_to_rays(origins, unit_vecs)
    error = np.linalg.norm(estimated_source - TRUE_SOURCE)

    print("\n" + "=" * 60)
    print("Triangulation Result")
    print("=" * 60)
    print(f"  True source        : {TRUE_SOURCE}")
    print(f"  Estimated source   : {np.round(estimated_source, 4)}")
    print(f"  Euclidean error    : {error:.6f} m")
