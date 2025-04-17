import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dv import AedatFile
import numpy as np

# Load the .aedat4 file
filename = "/Users/joshuaj27/Desktop/Physical Pendulum Test/Single_Pendulum1.aedat4"

# Empty lists to store event data
timestamps = []
x_coords = []
y_coords = []
polarities = []
# Load the data from the file
with AedatFile(filename) as f:
    if 'events' in f.names:
        for event in f['events']:
            timestamps.append(event.timestamp)
            x_coords.append(event.x)
            y_coords.append(event.y)
            polarities.append(event.polarity)

# Convert data to arrays
timestamps = np.array(timestamps)
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)
polarities = np.array(polarities)

# Convert timestamps to seconds
timestamps = timestamps / 1_000_000  # Convert from microseconds to seconds

# Convert polarities to colors for visualization
colors = ['green' if p else 'red' for p in polarities]

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d')

# Set the axis background to black
ax.set_facecolor('black')

# Set grid and tick colors for better visibility
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')

# Scatter plot
scatter = ax.scatter(x_coords, timestamps - 1733264779, y_coords, c=colors, s=1, alpha=0.7)

# Labels and title
ax.set_title("3D Scatter Plot of Events Over Time", color='white')
ax.set_xlabel("X Coordinate", color='white')
ax.set_ylabel("Time (seconds)", color='white')
ax.set_zlabel("Y Coordinate", color='white')

ax.invert_xaxis()
ax.invert_zaxis()

# Set viewing angle
ax.view_init(elev=0, azim=0)

plt.show()