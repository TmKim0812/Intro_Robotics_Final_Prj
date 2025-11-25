import numpy as np
import genesis as gs

np.random.seed(25) # fix the random seed / working when 25 / failure when 42
gs.init(backend=gs.gpu)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, gravity=(0,0,0)),
    show_viewer=True,
)

# Entities
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

# Random goal + obstacles
def sample_pos_xy_ring(r_min=0.3, r_max=0.6, z_range=(0.25, 0.6)):
    while True:
        x, y = np.random.uniform(-r_max, r_max, 2)
        r = np.hypot(x, y)
        if r_min <= r <= r_max:
            z = np.random.uniform(*z_range)
            return (x, y, z)

goal_pos = sample_pos_xy_ring(0.75, 0.85, (0.45, 0.75))
goal_sphere = scene.add_entity(
    gs.morphs.Sphere(radius=0.05, pos=goal_pos, fixed=True),
    surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0,1,0)))
)

def sample_blocking_obstacles(goal_pos, n_blocking=1, r_offset=0.1, z_range=(0.4, 0.8)):
    obstacles = []
    base = np.array([0, 0, 1.4])
    goal = np.array(goal_pos)
    dir_vec = goal - base
    dir_vec /= np.linalg.norm(dir_vec)
    for i in range(n_blocking):
        t = np.random.uniform(0.45, 0.55)
        point_on_line = base + t * dir_vec
        offset = np.random.uniform(-r_offset, r_offset, size=3)
        offset[2] = np.clip(offset[2], -0.05, 0.05)
        pos = point_on_line + offset
        pos[2] = np.clip(pos[2], *z_range)
        obstacles.append(tuple(pos))
    return obstacles

cube_poses = sample_blocking_obstacles(goal_pos, n_blocking=2)
def red(): return gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(1,0,0)))
cubes = [scene.add_entity(gs.morphs.Box(size=(0.1,0.1,0.1), pos=p, fixed=True), surface=red()) for p in cube_poses]

scene.build()

# Setup
arm_joints = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
dofs = [franka.get_joint(j).dof_idx_local for j in arm_joints]
franka.set_dofs_position(np.array([0, 0, 0, 0, 0, np.pi, np.pi]), dofs)
scene.step()
ee = franka.get_link("hand")

# Compute via point
base = np.array([0,0,1.4])
goal = np.array(goal_pos)
dir_vec = goal - base
dir_vec /= np.linalg.norm(dir_vec)
obs_pos = np.array(cube_poses[0])
tangent = np.cross(np.array([0,0,1]), dir_vec)
tangent /= np.linalg.norm(tangent)

via_pos = obs_pos + 0.4 * tangent
via_pos[2] += -0.1  # lift under obstacle

# Inverse Kinematics
q_start = franka.get_dofs_position()
q_via = franka.inverse_kinematics(link=ee, pos=via_pos, quat=np.array([1,0,0,0]))
q_goal = franka.inverse_kinematics(link=ee, pos=goal_pos, quat=np.array([1,0,0,0]))
print(q_start, q_via, q_goal)

# Linear joint interpolation
def move_linear(franka, q_start, q_end, steps=150):
    for α in np.linspace(0, 1, steps):
        q = (1 - α) * q_start + α * q_end
        franka.control_dofs_position(q)
        scene.step()

# Execute motion
move_linear(franka, q_start, q_via, steps=250)
move_linear(franka, q_via, q_goal, steps=250)
for i in range(50):
    scene.step() # give more time to reach goal point