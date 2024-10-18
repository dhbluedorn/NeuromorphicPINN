import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def PCA(A):
    # Center the data
    Xavg = np.mean(A, axis=0)
    B = A - Xavg
    
    # Compute covariance matrix
    cov_matrix = np.cov(B, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, V = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues and corresponding eigenvectors in descending order
    i = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[i]
    V = V[:, i]
    
    return Xavg, V, eigenvalues

# 1. Generate noisy Gaussian data cloud
xC = np.array([2, 1])
sig = np.array([2, 0.5])
theta = np.pi / 3
R = np.array([[np.cos(theta), np.sin(theta)],
              [-np.sin(theta), np.cos(theta)]])

nPoints = 10000
X = np.random.randn(nPoints, 2) @ np.diag(sig) @ R + np.ones((nPoints, 2)) * xC
plt.figure()
plt.scatter(X[:, 0], X[:, 1], color='black', s=2)
plt.title('PCA Plot of Noisy Gaussian Data Cloud')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axis('equal')
plt.grid(True)

# PCA function call
Aavg, V, eigenvalues = PCA(X)

theta = np.linspace(0, 2 * np.pi, 100)
Astd = np.array([np.cos(theta), np.sin(theta)]).T @ np.sqrt(np.diag(eigenvalues)) @ V.T
plt.plot(Aavg[0] + Astd[:, 0], Aavg[1] + Astd[:, 1], 'r-')
plt.plot(Aavg[0] + 2 * Astd[:, 0], Aavg[1] + 2 * Astd[:, 1], 'r-')
plt.plot(Aavg[0] + 3 * Astd[:, 0], Aavg[1] + 3 * Astd[:, 1], 'r-')
plt.show()

# 2. Generate a circle of data (centered at origin)
nPoints = 10000
r = 1
theta = 2 * np.pi * np.random.rand(nPoints)
radius = r * np.sqrt(np.random.rand(nPoints))
A = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))

# PCA function call
Aavg, V, eigenvalues = PCA(A)

# Plotting the 2D data
plt.figure()
plt.scatter(A[:, 0], A[:, 1], color='black', s=2)

# Plot the principal axes using PCA results
plt.quiver(Aavg[0], Aavg[1], V[0, 0], V[1, 0], scale=np.sqrt(eigenvalues[0]), color='red', width=0.005)
plt.quiver(Aavg[0], Aavg[1], V[0, 1], V[1, 1], scale=np.sqrt(eigenvalues[1]), color='green', width=0.005)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('PCA Plot of 2D Circle of Data')
plt.axis('equal')
plt.grid(True)
plt.show()

# 3. Generate a circle of data (offset)
center = np.array([2, 3])
X = center[0] + radius * np.cos(theta)
Y = center[1] + radius * np.sin(theta)
A = np.column_stack((X, Y))

# PCA function call
Aavg, V, eigenvalues = PCA(A)

# Plotting the 2D data
plt.figure()
plt.scatter(A[:, 0], A[:, 1], color='black', s=2)

# Plot the principal axes using PCA results
plt.quiver(Aavg[0], Aavg[1], V[0, 0], V[1, 0], scale=np.sqrt(eigenvalues[0]), color='red', width=0.005)
plt.quiver(Aavg[0], Aavg[1], V[0, 1], V[1, 1], scale=np.sqrt(eigenvalues[1]), color='green', width=0.005)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('PCA Plot of 2D Circle of Data (Offset)')
plt.axis('equal')
plt.grid(True)
plt.show()

# 4. Generate a sphere of data
nPoints = 10000
r = 1
theta = 2 * np.pi * np.random.rand(nPoints)
phi = np.arccos(2 * np.random.rand(nPoints) - 1)

# Convert spherical coordinates to Cartesian coordinates
A = np.column_stack((r * np.sin(phi) * np.cos(theta), 
                     r * np.sin(phi) * np.sin(theta), 
                     r * np.cos(phi)))

# PCA function call
Aavg, V, eigenvalues = PCA(A)

# Plotting the 3D data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(A[:, 0], A[:, 1], A[:, 2], color='black', s=2)

# Plot the principal axes using PCA results
ax.quiver(Aavg[0], Aavg[1], Aavg[2], V[0, 0], V[1, 0], V[2, 0], length=np.sqrt(eigenvalues[0]), color='red')
ax.quiver(Aavg[0], Aavg[1], Aavg[2], V[0, 1], V[1, 1], V[2, 1], length=np.sqrt(eigenvalues[1]), color='green')
ax.quiver(Aavg[0], Aavg[1], Aavg[2], V[0, 2], V[1, 2], V[2, 2], length=np.sqrt(eigenvalues[2]), color='blue')

# Labels and view setup
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('PCA Plot of 3D Sphere of Data')
plt.show()
