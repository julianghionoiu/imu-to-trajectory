# IMU data from Polar Verity Sense to trajectory

Setup
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Quick run:
```shell
python 01_sensor_calibration_ekf.py
```

## IMU integration algo assessment

### Algorithms Tested with the Gyro calibration

| AHRS Short Name | Algorithm Name       | Description                                                  | Obs                                                                   |
|-----------------|----------------------|--------------------------------------------------------------|-----------------------------------------------------------------------|
| ChatGPT gen     | Vibe coded Kalman    | It seems to be like a Kalman but ignoring Mag                | Surprisingly well but ignores MAG                                     |
| Complementary   | Complementary        | Combines accelerometer and gyroscope data for orientation.   | Works pretty well, low linear acc, but struggles with one Z dimension |
| EKF             | Extended Kalman      | Probabilistic approach to estimate system states.            | Very good attitude detection, but picks up ACC from rotation          |
| Madgwick        | Madgwick Filter      | Sensor fusion algorithm specifically for IMU data.           | ACC rotating around 2.5. Has a gain that allows us to tweak.          |


## Rejected algos

| AHRS Short Name | Algorithm Name       | Description                                                  | Obs                               |
|-----------------|----------------------|--------------------------------------------------------------|-----------------------------------|
| Aqua            | Algebraic Quaternion | Estimates a quaternion from inertial+magnetic observations   | Too much linear acceleration      |
| Fourati         | Fourati              | Estimates based on the time integral of the angular velocity | Nope. Has growing error gain      |
| Mahony          | Mahony Filter        | Quaternion-based algorithm for attitude estimation.          | ACC flapping around X                                                 |
