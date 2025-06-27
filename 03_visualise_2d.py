import pandas as pd
import matplotlib.pyplot as plt

# === 1. Load Euler angles ===
euler_df = pd.read_csv("02_out_orientation_output.csv")  # Columns: roll_deg, pitch_deg, yaw_deg, time

# === 2. Plot roll, pitch, yaw over time ===
plt.figure(figsize=(12, 6))

plt.plot(euler_df["time"], euler_df["roll_deg"], label="Roll (°)")
plt.plot(euler_df["time"], euler_df["pitch_deg"], label="Pitch (°)")
plt.plot(euler_df["time"], euler_df["yaw_deg"], label="Yaw (°)")

plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title("Orientation Over Time (Roll, Pitch, Yaw)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
