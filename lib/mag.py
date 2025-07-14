
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, svd
from scipy.linalg import sqrtm
import json
import sys
import os
from scipy.signal import medfilt

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

    return center.tolist(), A_norm.tolist()


def compute_calibration_parameters(raw_data):
    # Apply median filter to each column (axis)
    filtered_data = np.apply_along_axis(lambda m: medfilt(m, kernel_size=5), axis=0, arr=raw_data)
    ellipsoid = fit_ellipsoid(filtered_data)
    return ellipsoid_center_and_transform(ellipsoid)


def calibrate_array(raw_data, bias, scale_matrix):
    # Center the data
    x_centered = raw_data - bias
    # Apply whitening (sphering) transformation
    m_inv = inv(sqrtm(scale_matrix))
    return (m_inv @ x_centered.T).T

def calibrate_np(raw_data_np, bias, scale_matrix):
    bias = np.array(bias)
    scale_matrix = np.array(scale_matrix)
    from scipy.linalg import sqrtm
    scale_inv = np.linalg.inv(sqrtm(scale_matrix))
    # Apply calibration to MAG
    mag_raw = raw_data_np[["mag_x", "mag_y", "mag_z"]].values
    mag_corrected = (scale_inv @ (mag_raw - bias).T).T
    mag_df_copy = raw_data_np.copy()
    mag_df_copy[["mag_x", "mag_y", "mag_z"]] = mag_corrected
    return mag_df_copy

