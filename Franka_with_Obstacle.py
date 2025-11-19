import numpy as np
import genesis as gs

# ------------------- Init -------------------
gs.init(backend=gs.gpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0,0,0)
    ),
    show_viewer=True
)

plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

# Helper: sample obstacle/goal positions reasonably close but not at origin
def sample_pos_xy_ring(r_min=0.30, r_max=0.50, z_range=(0.20, 0.60)):
    while True:
        x = np.random.uniform(-r_max, r_max)
        y = np.random.uniform(-r_max, r_max)
        r = np.hypot(x, y)
        if r_min <= r <= r_max:
            z = np.random.uniform(*z_range)
            return (x, y, z)
        
# Obstacles (3 red cubes)
cube_poses = [sample_pos_xy_ring() for _ in range(3)]

def red():
    return gs.surfaces.Rough(
        diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.0, 0.0))
    )

cubes = []
for pos in cube_poses:
    cubes.append(
        scene.add_entity(
            gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=pos, fixed=True),
            surface=red(),
        )
    )

# goal point (green sphere)
goal_pos = sample_pos_xy_ring(r_min=0.55, r_max=0.65, z_range=(0.25, 0.75))
goal_sphere = scene.add_entity(
    gs.morphs.Sphere(radius=0.05, pos=goal_pos, fixed=True),
    surface=gs.surfaces.Rough(
        diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0))
    ),
)
scene.build()

# joints
arm_joints = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
dofs = [franka.get_joint(j).dof_idx_local for j in arm_joints]

# PD (needed for torque solver stability)
franka.set_dofs_kp( np.array([500]*7), dofs )
franka.set_dofs_kv( np.array([40]*7), dofs )

# reset
franka.set_dofs_position(np.zeros(7), dofs)
scene.step()

ee = franka.get_link("hand")


# ------------------- Control -------------------
λ = 0.05          # damping
α = 1.0           # step size
q_null_weight = 0.1

for t in range(2000):

    # EE pos
    x = ee.get_pos().cpu().numpy()
    e = goal_pos - x

    # stop near goal
    if np.linalg.norm(e) < 0.02:
        franka.control_dofs_velocity(np.zeros(7), dofs)
        scene.step()
        continue

    # full 6x7 jacobian
    J = franka.get_jacobian(ee)[0:3,0:7].cpu().numpy()

    # damped least squares inverse
    JJt = J @ J.T + λ*np.eye(3)
    pinv = J.T @ np.linalg.inv(JJt) # pseudoinverse of Jacobian

    # primary task
    qdot = α * pinv @ e

    # nullspace centering to zero posture
    N = np.eye(7) - pinv @ J
    q = franka.get_dofs_position(dofs).cpu().numpy()
    qdot += q_null_weight * (N @ (-q))

    franka.control_dofs_velocity(qdot, dofs)
    scene.step()