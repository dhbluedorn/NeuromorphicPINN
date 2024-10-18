# Triple Pendulum in Python
# Kyle Brown
# NMSU — ENGR 401

# Code logic based off of Ten-Minutes Physics Tutorial #06 for triple pendulum in HTML.
# https://matthias-research.github.io/pages/tenMinutePhysics/index.html

# To change pendulum starting conditions, alter `angles` variable. 
#    angles = [0, 0, 0] will begin the simulation with a still pendulum

# Added keyboard inputs to introduce impulses into each pendulum mass.
#   - Press 1, 2, or 3 to introduce a clockwise torque on each respective pendulum mass.
#   - Press Shift + 1, 2, or 3 to introduce a ccounter-clockwise torque on each respective pendulum mass.


import pygame
import math

# Initialize pygame
pygame.init()

# Define simulation parameters
lengths = [0.15, 0.15, 0.15]
masses = [1.0, 1.0, 1.0]
angles = [1.5, -1.5, 1.5]          # In radians

width, height = 1600, 600
cScale = min(width, height) / 1.0
gravity = -9.81
dt = 0.01
num_substeps = 100

# Screen setup
screen = pygame.display.set_mode((width, height))           # Screen dimensions
pygame.display.set_caption("Triple Pendulum Simulation")    # Window caption
font = pygame.font.SysFont(None, 36)                        # For FPS counter display

def cX(pos):
    return width // 2 + pos[0] * cScale

def cY(pos):
    return 0.4 * height - pos[1] * cScale

class Pendulum:
    def __init__(self, masses, lengths, angles):
        self.masses = [0.0]
        self.lengths = [0.0]
        self.pos = [(0.0, 0.0)]
        self.prev_pos = [(0.0, 0.0)]
        self.vel = [(0.0, 0.0)]

        x, y = 0.0, 0.0
        for i in range(len(masses)):
            self.masses.append(masses[i])
            self.lengths.append(lengths[i])
            x += lengths[i] * math.sin(angles[i])
            y += lengths[i] * -math.cos(angles[i])
            self.pos.append((x, y))
            self.prev_pos.append((x, y))
            self.vel.append((0.0, 0.0))

    def simulate(self, dt, gravity):
        for i in range(1, len(self.masses)):
            self.vel[i] = (self.vel[i][0], self.vel[i][1] + dt * gravity)
            self.prev_pos[i] = self.pos[i]
            self.pos[i] = (self.pos[i][0] + self.vel[i][0] * dt, self.pos[i][1] + self.vel[i][1] * dt)

        for i in range(1, len(self.masses)):
            dx = self.pos[i][0] - self.pos[i-1][0]
            dy = self.pos[i][1] - self.pos[i-1][1]
            d = math.sqrt(dx ** 2 + dy ** 2)
            w0 = 1.0 / self.masses[i - 1] if self.masses[i - 1] > 0.0 else 0.0
            w1 = 1.0 / self.masses[i] if self.masses[i] > 0.0 else 0.0
            corr = (self.lengths[i] - d) / d / (w0 + w1)
            self.pos[i - 1] = (self.pos[i - 1][0] - w0 * corr * dx, self.pos[i - 1][1] - w0 * corr * dy)
            self.pos[i] = (self.pos[i][0] + w1 * corr * dx, self.pos[i][1] + w1 * corr * dy)

        for i in range(1, len(self.masses)):
            self.vel[i] = ((self.pos[i][0] - self.prev_pos[i][0]) / dt, (self.pos[i][1] - self.prev_pos[i][1]) / dt)

    def draw(self):
        for i in range(1, len(self.masses)):
            pygame.draw.line(screen, (48, 48, 48), (cX(self.pos[i-1]), cY(self.pos[i-1])), (cX(self.pos[i]), cY(self.pos[i])), 2)
            r = 0.05 * math.sqrt(self.masses[i])
            pygame.draw.circle(screen, (255, 255, 255), (int(cX(self.pos[i])), int(cY(self.pos[i]))), int(cScale * r))

    def apply_torque(self, index, clockwise=True):
        # Apply a torque impulse to rotate the pendulum clockwise or counterclockwise
        if 1 <= index < len(self.vel):
            direction = 1 if clockwise else -1
            vx, vy = self.vel[index]
            # Adjust both x and y velocities for rotational effect
            self.vel[index] = (vx + direction * 0.1, vy + direction * 0.05)

# Create the pendulum object
pendulum = Pendulum(masses, lengths, angles)

# Simulation loop
running = True
clock = pygame.time.Clock()
frame_times = []

running = True
clock = pygame.time.Clock()
frame_times = []
accumulator = 0.0  # Used to accumulate time for fixed-step physics simulation

while running:
    screen.fill((0, 0, 0))
    
    # Process input (key presses)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    
    # Torque impulses for clockwise and counterclockwise torques
    if keys[pygame.K_1] and not keys[pygame.K_LSHIFT]:  # Torque clockwise on the first pendulum
        pendulum.apply_torque(1, clockwise=True)
    if keys[pygame.K_2] and not keys[pygame.K_LSHIFT]:  # Torque clockwise on the second pendulum
        pendulum.apply_torque(2, clockwise=True)
    if keys[pygame.K_3] and not keys[pygame.K_LSHIFT]:  # Torque clockwise on the third pendulum
        pendulum.apply_torque(3, clockwise=True)
    
    if keys[pygame.K_1] and keys[pygame.K_LSHIFT]:  # Torque counterclockwise on the first pendulum
        pendulum.apply_torque(1, clockwise=False)
    if keys[pygame.K_2] and keys[pygame.K_LSHIFT]:  # Torque counterclockwise on the second pendulum
        pendulum.apply_torque(2, clockwise=False)
    if keys[pygame.K_3] and keys[pygame.K_LSHIFT]:  # Torque counterclockwise on the third pendulum
        pendulum.apply_torque(3, clockwise=False)

    # Time management for physics simulation
    frame_time = clock.tick() / 1000.0  # Get the frame time in seconds
    accumulator += frame_time

    # Run physics simulation steps for each accumulated dt slice
    while accumulator >= dt:
        sdt = dt / num_substeps
        for _ in range(num_substeps):
            pendulum.simulate(sdt, gravity)
        accumulator -= dt

    # Draw the pendulum
    pendulum.draw()

    # FPS display (optional)
    fps_text = font.render(f"Effective FPS: {clock.get_fps():.2f}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))

    # Update display
    pygame.display.flip()

# Quit pygame
pygame.quit()
