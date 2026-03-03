import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ── Spiral Galaxy Parameters
N = 500
M = 4
G = 4.302e-6
Md = 1e11
a = 6.0
b = 0.5
pitch = 0.28
r_min = 3.0
dr = 0.15
dt = 0.5
steps = 500  # fewer steps for faster animation

# ── Initialize stars
r_init = r_min + dr * np.arange(N)
arm_off = 2 * np.pi * np.arange(N)/N * M
theta = pitch * r_init + arm_off
omega_c = np.sqrt(G * Md / (r_init**2 + (a+b)**2)**1.5)

x = r_init * np.cos(theta + 0.02*(np.arange(N)/N - 0.5))
y = r_init * np.sin(theta + 0.02*(np.arange(N)/N - 0.5))
z = 0.07*(np.arange(N)-N/2)/N + 0.01*(np.arange(N)/N - 0.5)

vx = -np.sin(theta) * r_init * omega_c
vy =  np.cos(theta) * r_init * omega_c
vz = np.zeros(N)

# ── Prepare trajectory arrays
x_traj = np.zeros((steps, N))
y_traj = np.zeros((steps, N))
z_traj = np.zeros((steps, N))

# ── Euler integration
for s in range(steps):
    r2 = x**2 + y**2
    sz = np.sqrt(z**2 + b**2)
    B = a + sz
    denom = (r2 + B**2)**1.5

    ax = -G*Md * x / denom
    ay = -G*Md * y / denom
    az = -G*Md * B * z / (sz * denom)

    vx += ax * dt
    vy += ay * dt
    vz += az * dt

    x += vx * dt
    y += vy * dt
    z += vz * dt

    x_traj[s] = x
    y_traj[s] = y
    z_traj[s] = z

# ── 3D Animation
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(x_traj[0], y_traj[0], z_traj[0], s=10, c='white')

ax.set_facecolor('black')
ax.set_xlim([-30,30])
ax.set_ylim([-30,30])
ax.set_zlim([-5,5])
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_zlabel('Z [kpc]')
ax.set_title('Spiral Galaxy Simulation')

def update(frame):
    scat._offsets3d = (x_traj[frame], y_traj[frame], z_traj[frame])
    ax.view_init(elev=30, azim=frame*0.5)  # rotate view
    return scat,

ani = FuncAnimation(fig, update, frames=steps, interval=30, blit=False)
plt.show()