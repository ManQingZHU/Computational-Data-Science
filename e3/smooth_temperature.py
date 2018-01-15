import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter

filename = sys.argv[1]
cpu_data = pd.read_csv(filename, parse_dates=[3])

loess_smoothed = lowess(cpu_data['temperature'], cpu_data['timestamp'], frac = 0.05)

kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1']]
initial_state = kalman_data.iloc[0]
ob_stddev = 1.5
trans_stddev = 0.5
observation_covariance = np.diag([ob_stddev, ob_stddev, ob_stddev]) ** 2 # TODO: shouldn't be zero
transition_covariance = np.diag([trans_stddev, trans_stddev, trans_stddev]) ** 2 # TODO: shouldn't be zero
transition = [[1.0, 0, -0.27], [0, 0.85, -1.14], [0, 0.06, 0.37]] # TODO: shouldn't (all) be zero
kf = KalmanFilter(
    initial_state_mean=initial_state,
    initial_state_covariance=observation_covariance,
    observation_covariance=observation_covariance,
    transition_covariance=transition_covariance,
    transition_matrices=transition
)
kalman_smoothed, _ = kf.smooth(kalman_data)
plt.figure(figsize=(12, 10))
plt.subplot(2,1,1)
plt.title("LOWESS: frac = 0.05")
plt.ylabel('temperature')
plt.xlabel('timestamp')
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5)
plt.plot(cpu_data['timestamp'], loess_smoothed[:, 1], 'r-',linewidth=2)
plt.legend(['Sensor Observation', 'After Process'])

plt.subplot(2,1,2)
plt.title("Kalman Filtering")
plt.ylabel('temperature')
plt.xlabel('timestamp')
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5)
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-',linewidth=2)
plt.legend(['Sensor Observation', 'After Process'])
#plt.show() # easier for testing
plt.savefig('cpu.svg') # for final submission