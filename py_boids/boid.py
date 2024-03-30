import taichi as ti

ti.init(arch=ti.gpu)

N = 100
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
        particles_vel[i] = ti.Vector([ti.random() * 2 - 1 for _ in range(3)]) * 0.1

@ti.kernel
def update_particles():
    for i in range(N):
        particles_pos[i] += particles_vel[i]
        
        for j in ti.static(range(3)):  
            if particles_pos[i][j] < 0:
                particles_pos[i][j] += length_bound  
            elif particles_pos[i][j] > length_bound:
                particles_pos[i][j] -= length_bound  


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
