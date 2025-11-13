import numpy as np
import genesis as gs

gs.init(backend=gs.gpu)

# ---------- helper: sample obstacle far from base (0,0) ----------
def sample_pos_far_from_base(min_xy_rad=0.3, z_range=(0.2, 0.8)):
    # ensure sqrt(x^2 + y^2) >= min_xy_rad so it won't touch the robot at start
    while True:
        x = np.random.uniform(-0.4, 0.4)
        y = np.random.uniform(-0.4, 0.4)
        if np.hypot(x, y) >= min_xy_rad:
            z = np.random.uniform(*z_range)
            return (x, y, z)

scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        res           = (960, 640),
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
        gravity = (0.0, 0.0, 0.0),  # off for pure velocity control
    ),
    show_viewer = True,
)

# ---------- entities (spawn obstacles far away up front) ----------
plane = scene.add_entity(gs.morphs.Plane())

franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
)

# 3 cubes and 1 goal, all sampled outside a radius from origin
cube_poses = [sample_pos_far_from_base() for _ in range(3)]
goal_pos   = sample_pos_far_from_base(min_xy_rad=0.4, z_range=(0.3, 0.9))

def red():  return gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.0, 0.0)))
def green():return gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0)))

for pos in cube_poses:
    scene.add_entity(gs.morphs.Box(size=(0.2,0.2,0.2), pos=pos, fixed=True), surface=red())

scene.add_entity(gs.morphs.Sphere(radius=0.05, pos=goal_pos, fixed=True), surface=green())

# ---------- build ----------
scene.build()

# ---------- joints indexing ----------
jnt_names = [
    'joint1','joint2','joint3','joint4','joint5','joint6','joint7',
    'finger_joint1','finger_joint2'
]
dofs_idx = [franka.get_joint(n).dof_idx_local for n in jnt_names]

# ---------- gains / limits (optional, as you had) ----------
franka.set_dofs_kp(kp=np.array([4500,4500,3500,3500,2000,2000,2000,100,100]), dofs_idx_local=dofs_idx)
franka.set_dofs_kv(kv=np.array([ 450, 450, 350, 350, 200, 200, 200, 10, 10]), dofs_idx_local=dofs_idx)
franka.set_dofs_force_range(
    lower=np.array([-87,-87,-87,-87,-12,-12,-12,-100,-100]),
    upper=np.array([ 87, 87, 87, 87, 12, 12, 12, 100, 100]),
    dofs_idx_local=dofs_idx
)

# ---------- start from exactly ZERO pose ----------
zero_dofs = np.zeros(9)  # 7 arm + 2 fingers = 0 rad/0 m
franka.set_dofs_position(zero_dofs, dofs_idx)
franka.set_dofs_velocity(np.zeros(9), dofs_idx)
scene.step()
# sync the velocity controller once to avoid the first-frame impulse
franka.control_dofs_velocity(np.zeros(9), dofs_idx)
scene.step()

# ---------- hold still ----------
for _ in range(1000):
    franka.control_dofs_velocity(np.zeros(9), dofs_idx)
    scene.step()