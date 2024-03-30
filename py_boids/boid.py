import taichi as ti

ti.init(arch=ti.gpu)

N = 10000
length_bound = 100
centering_factor = 0.01
repulsion_factor = 0.05
matching_factor = 0.3
speed_limit = 5.0
perception_radius = 40.0
avoidance_radius = 5.0

boundary_lines = 12 


particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
particles_vel = ti.Vector.field(3, dtype=ti.f32, shape=N)
line_starts = ti.Vector.field(3, dtype=ti.f32, shape=boundary_lines)
line_ends = ti.Vector.field(3, dtype=ti.f32, shape=boundary_lines)

window_x = 800
windows_y = 600
length_bound = 100
camera_move_speed = window_x / length_bound * 0.1

@ti.kernel
def init_particles():
    for i in range(N):
        particles_pos[i] = ti.Vector([ti.random() * length_bound for _ in range(3)])
        particles_vel[i] = ti.Vector([ti.random() * 2 - 1 for _ in range(3)]) * speed_limit

@ti.func
def limit_velocity(velocity):
    result_velocity = velocity  # Default to the original velocity
    if velocity.norm() > speed_limit:
        result_velocity = velocity.normalized() * speed_limit
    return result_velocity

@ti.func
def avoid_other_boids(i):
    repulsion = ti.Vector([0.0, 0.0, 0.0])
    for j in range(N):
        if i != j:
            distance = (particles_pos[i] - particles_pos[j]).norm()
            if distance < avoidance_radius:  # Avoidance radius
                repulsion += (particles_pos[i] - particles_pos[j]) / distance
    return repulsion * repulsion_factor

@ti.func
def align_velocity(i):
    average_vel = ti.Vector([0.0, 0.0, 0.0])
    count = 0
    for j in range(N):
        if i != j:
            average_vel += particles_vel[j]
            count += 1
    if count > 0:
        average_vel /= count
    return (average_vel - particles_vel[i]) * matching_factor


@ti.func
def find_flock_center(i):
    center = ti.Vector([0.0, 0.0, 0.0])
    count = 0
    for j in range(N):
        if i != j:
            distance = (particles_pos[j] - particles_pos[i]).norm()
            if distance < perception_radius:
                center += particles_pos[j]
                count += 1
            # center += particles_pos[j]
            # count += 1
    if count > 0:
        center /= count
    return (center - particles_pos[i]) * centering_factor

@ti.kernel
def update_particles():
    for i in range(N):
        center_offset = find_flock_center(i)
        avoidance_offset = avoid_other_boids(i)
        alignment_offset = align_velocity(i)
        
        velocity = particles_vel[i] + center_offset + avoidance_offset + alignment_offset
        velocity = limit_velocity(velocity)
        
        particles_pos[i] += velocity * 0.1  # Assuming 0.1 as a time step equivalent
        particles_vel[i] = velocity
        
        for j in ti.static(range(3)):  
            if particles_pos[i][j] < 0:
                particles_pos[i][j] += length_bound  
            elif particles_pos[i][j] > length_bound:
                particles_pos[i][j] -= length_bound  
        
        
# @ti.kernel
# def update_particles():
#     temp_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
#     temp_vel = ti.Vector.field(3, dtype=ti.f32, shape=N)
    
#     for i in range(N):
#         center_offset = find_flock_center(i)
#         avoidance_offset = avoid_other_boids(i)
#         alignment_offset = align_velocity(i)
#         velocity = particles_vel[i] + center_offset + avoidance_offset + alignment_offset
        
#         # Limit the velocity to a maximum speed
#         if velocity.norm() > speed_limit:
#             velocity = velocity.normalized() * speed_limit
        
#         temp_vel[i] = velocity
#         temp_pos[i] = particles_pos[i] + temp_vel[i] * 0.1  # Assuming 0.1 as a time step equivalent
        
#         for j in ti.static(range(3)):  
#             if particles_pos[i][j] < 0:
#                 particles_pos[i][j] += length_bound  
#             elif particles_pos[i][j] > length_bound:
#                 particles_pos[i][j] -= length_bound  


def draw_bounds(x_min=0, y_min=0, z_min=0, x_max=1, y_max=1, z_max=1):
    box_anchors = ti.Vector.field(3, dtype=ti.f32, shape = 8)
    box_anchors[0] = ti.Vector([x_min, y_min, z_min])
    box_anchors[1] = ti.Vector([x_min, y_max, z_min])
    box_anchors[2] = ti.Vector([x_max, y_min, z_min])
    box_anchors[3] = ti.Vector([x_max, y_max, z_min])
    box_anchors[4] = ti.Vector([x_min, y_min, z_max])
    box_anchors[5] = ti.Vector([x_min, y_max, z_max])
    box_anchors[6] = ti.Vector([x_max, y_min, z_max])
    box_anchors[7] = ti.Vector([x_max, y_max, z_max])

    box_lines_indices = ti.field(int, shape=(2 * 12))
    for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
        box_lines_indices[i] = val
    return box_anchors, box_lines_indices




        
init_particles()

# Create a window for rendering
window = ti.ui.Window("3D Moving Particles", (window_x, windows_y))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

camera.position(length_bound/2, length_bound/2, -1.5 *length_bound)  
camera.lookat(length_bound/2, length_bound/2, 0)    
box_anchors, box_lines_indices = draw_bounds(x_min=0, y_min=0, z_min=0, x_max=length_bound, y_max=length_bound, z_max=length_bound)


while window.running:
    # camera.position(5, 5, -15)  # Set camera position
    # camera.lookat(5, 5, 0)     # Camera looks at the center
    # camera.fov(60)             # Field of view
    camera.track_user_inputs(window, movement_speed=camera_move_speed, hold_key=ti.ui.RMB)
    scene.set_camera(camera) 
    
    update_particles() 

    scene.ambient_light((0.5, 0.5, 0.5)) 
    # scene.point_light(pos=(0, 10, -10), color=(1, 1, 1)) 

    # Render particles
    scene.particles(particles_pos, radius=0.5, color=(0.4, 0.6, 0.9))
    scene.lines(box_anchors, indices=box_lines_indices, color = (0.99, 0.99, 0.01), width = 2.0)
    
    
    canvas.scene(scene)
    window.show()
