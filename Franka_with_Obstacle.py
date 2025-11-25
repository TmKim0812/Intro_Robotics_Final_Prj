import numpy as np
import genesis as gs

#fix the random seed
np.random.seed(25) # successful seed: 42, 29, 14 / failure but meaningful: 25, 6
# Initialization
gs.init(backend=gs.gpu)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, gravity=(0,0,0)),
    show_viewer=True,
)

plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

# random goal + obstacles
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

def sample_blocking_obstacles(goal_pos, n_blocking=1, n_random=2,
                              r_offset=0.1, z_range=(0.2, 0.6)):
    obstacles = []

    # Main blocking obstacles 
    base = np.array([0, 0, 1.4]) # initial end effector pose
    goal = np.array(goal_pos)
    dir_vec = goal - base
    dir_vec /= np.linalg.norm(dir_vec)

    for i in range(n_blocking):
        t = np.random.uniform(0.45, 0.55)  # fraction along path
        point_on_line = base + t * goal
        offset = np.random.uniform(-r_offset, r_offset, size=3)
        offset[2] = np.clip(offset[2], -0.05, 0.05)  # small vertical variation
        pos = point_on_line + offset
        pos[2] = np.clip(pos[2], *z_range)
        obstacles.append(tuple(pos))

    # Additional random obstacles
    for i in range(n_random):
        obstacles.append(sample_pos_xy_ring(r_min=0.3, r_max=0.6, z_range=z_range))

    return obstacles

cube_poses = sample_blocking_obstacles(goal_pos, n_blocking=2, n_random=0)
def red(): return gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(1,0,0)))
cubes = [scene.add_entity(gs.morphs.Box(size=(0.15,0.15,0.15), pos=p, fixed=True), surface=red()) for p in cube_poses]

scene.build()

# joints
arm_joints = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
dofs = [franka.get_joint(j).dof_idx_local for j in arm_joints]
franka.set_dofs_kp(np.array([500]*7), dofs)
franka.set_dofs_kv(np.array([40]*7), dofs)

franka.set_dofs_position(np.array([0, 0, 0, 0, 0, np.pi, np.pi]), dofs)
scene.step()

ee = franka.get_link("hand")
links = [franka.get_link(name) for name in [
    "link1","link2","link3","link4","link5","link6","link7"
]]
links.append(ee)

# Parameters 
λ = 0.05
α = 1.0
k_goal = 1.5
k_avoid = 0.4          # stronger for tangential motion
influence_radius = 0.3
q_null_weight = 0.1

for t in range(2000):
    ee_pos = ee.get_pos().cpu().numpy()
    goal_vec = np.array(goal_pos) - ee_pos

    # Goal attraction
    v_goal = k_goal * goal_vec
    J_ee = franka.get_jacobian(ee)[0:3,0:7].cpu().numpy()
    JJt = J_ee @ J_ee.T + λ*np.eye(3)
    J_pinv = J_ee.T @ np.linalg.inv(JJt)
    qdot_goal = α * J_pinv @ v_goal

    # Whole-body tangential obstacle avoidance 
    qdot_avoid = np.zeros(7)
    for link in links:
        link_pos = link.get_pos().cpu().numpy()
        for cube in cubes:
            obs = cube.get_pos().cpu().numpy()
            d_vec = link_pos - obs
            dist = np.linalg.norm(d_vec)
            if 1e-6 < dist < influence_radius:
                dir_vec = d_vec / dist   # from obstacle to link
                goal_dir = (np.array(goal_pos) - link_pos)
                goal_dir /= np.linalg.norm(goal_dir) + 1e-6

                # Tangential direction (orthogonal to both obstacle direction and goal direction)
                tangent = np.cross(dir_vec, np.cross(goal_dir, dir_vec))
                tangent /= np.linalg.norm(tangent) + 1e-6

                # Scale by proximity (closer → stronger tangential influence)
                strength = k_avoid * (1.0/dist - 1.0/influence_radius)
                v_tan = strength * tangent

                # Map to joint space
                J_link = franka.get_jacobian(link)[0:3,0:7].cpu().numpy()
                w = 1.0 / (1.0 + dist**2)
                # qdot_avoid += w * (J_link.T @ v_tan) -> transpose is just simple implementation
                JJt = J_link @ J_link.T + λ*np.eye(3)
                J_pinv = J_link.T @ np.linalg.inv(JJt)
                qdot_avoid += w * (J_pinv @ v_tan)

    # Combine
    qdot = qdot_goal + qdot_avoid

    # Nullspace posture
    N = np.eye(7) - J_pinv @ J_ee
    q = franka.get_dofs_position(dofs).cpu().numpy()
    qdot += q_null_weight * (N @ (-q))

    # Apply
    franka.control_dofs_velocity(qdot, dofs)
    scene.step()  