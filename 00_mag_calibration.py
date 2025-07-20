import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
import os

from lib.mag import compute_calibration_parameters, calibrate_array


def load_file(path):
    mag_df = pd.read_csv(path, sep=r'\s+', header=0, names=["timestamp", "mag_x", "mag_y", "mag_z"])
    return mag_df[['mag_x', 'mag_y', 'mag_z']].values

# === Fit ellipsoid: algebraic method ===

def store_calibration_parameters(input_file, local_bias, local_scale_matrix):
    # === Save calibration parameters to JSON ===
    calibration_data = {
        "bias": local_bias,
        "scale_matrix": local_scale_matrix
    }

    # Save JSON next to input file
    output_path = os.path.splitext(input_file)[0] + "_parameters.json"
    with open(output_path, "w") as f:
        json.dump(calibration_data, f, indent=4)

    print(f"✅ Calibration parameters saved to: {output_path}")


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
    
    plot(X, calibrate_array(X, bias, scale_matrix))
    
