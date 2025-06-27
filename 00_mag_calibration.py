import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, svd
from scipy.linalg import sqrtm
import json
import sys
import os

def load_file(path):
    mag_df = pd.read_csv(path, sep=r'\s+', header=0, names=["timestamp", "mag_x", "mag_y", "mag_z"])
    return mag_df[['mag_x', 'mag_y', 'mag_z']].values

# === Fit ellipsoid: algebraic method ===

def fit_ellipsoid(raw_data):
    x, y, z = raw_data[:, 0], raw_data[:, 1], raw_data[:, 2]
    d = np.column_stack([
        x**2, y**2, z**2,
        x*y, x*z, y*z,
        x, y, z,
        np.ones_like(x)
    ])
    S = np.dot(d.T, d)
    _, _, V = svd(S)
    return V[-1]

def ellipsoid_center_and_transform(v):
    # Extract matrix and vector
    A = np.array([
        [v[0], v[3]/2, v[4]/2],
        [v[3]/2, v[1], v[5]/2],
        [v[4]/2, v[5]/2, v[2]]
    ])
    b = np.array([v[6], v[7], v[8]])
    d = v[9]

    # Compute ellipsoid center
    center = -0.5 * inv(A).dot(b)

    # Translate the quadratic form to center the ellipsoid at the origin
    T = np.eye(4)
    T[3, :3] = center

    # Calculate the translated constant
    r = np.dot(center, np.dot(A, center)) - d

    # Normalize the matrix A
    A_norm = A / r

    return center, A_norm


def compute_calibration_parameters(raw_data):
    ellipsoid = fit_ellipsoid(raw_data)
    return ellipsoid_center_and_transform(ellipsoid)

def store_calibration_parameters(input_file, local_bias, local_scale_matrix):
    # === Save calibration parameters to JSON ===
    calibration_data = {
        "bias": local_bias.tolist(),
        "scale_matrix": local_scale_matrix.tolist()
    }

    # Save JSON next to input file
    output_path = os.path.splitext(input_file)[0] + "_parameters.json"
    with open(output_path, "w") as f:
        json.dump(calibration_data, f, indent=4)

    print(f"✅ Calibration parameters saved to: {output_path}")


def calibrate(raw_data):
    # Center the data
    x_centered = raw_data - bias
    # Apply whitening (sphering) transformation
    m_inv = inv(sqrtm(scale_matrix))
    return (m_inv @ x_centered.T).T


def plot(raw_data, calibrated_data):
    # === Plot before and after calibration ===
    fig = plt.figure(figsize=(14, 6))
    # Raw data
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(*raw_data.T, s=2, alpha=0.5)
    ax1.set_title("Raw Magnetometer Data")
    ax1.set_xlabel("X"), ax1.set_ylabel("Y"), ax1.set_zlabel("Z")
    # Calibrated data
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(*calibrated_data.T, s=2, alpha=0.5, color='green')
    ax2.set_title("Calibrated Magnetometer Data (Normalized Sphere)")
    ax2.set_xlabel("X"), ax2.set_ylabel("Y"), ax2.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


#  === Main Execution ===

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calibrate_mag.py <path/to/mag.data>")
        sys.exit(1)
    input_file = sys.argv[1]
    
    X = load_file(input_file)
    
    bias, scale_matrix = compute_calibration_parameters(X)
    store_calibration_parameters(input_file, bias, scale_matrix)
    
    plot(X, calibrate(X))
    
