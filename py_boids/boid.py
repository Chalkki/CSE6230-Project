import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)

N = 10000
length_bound = 100
centering_factor = 0.1
repulsion_factor = 0.03
matching_factor = 0.1
speed_limit = 5.0
perception_radius = 7.0
avoidance_radius =2.0

boundary_lines = 12 


particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
particles_vel = ti.Vector.field(3, dtype=ti.f32, shape=N)
line_starts = ti.Vector.field(3, dtype=ti.f32, shape=boundary_lines)
line_ends = ti.Vector.field(3, dtype=ti.f32, shape=boundary_lines)


particleArrayIndices = ti.field(dtype=ti.i32, shape=N) # What index in pars_pos and parts_vel represents this particle?
particleGridIndices = ti.field( dtype=ti.i32, shape=N) #  What grid cell is this particle in?
# assuming the particleArrayIndices is already sorted
gridCellWidth = 2.0 * perception_radius
halfSideCount = int((length_bound / gridCellWidth)) + 1
gridSideCount = 2 * halfSideCount
gridCellCount = gridSideCount * gridSideCount * gridSideCount
gridInverseCellWidth = 1.0 / gridCellWidth
gridStartIndices = ti.field(dtype=ti.i32, shape=gridCellCount)
gridEndIndices = ti.field(dtype=ti.i32, shape=gridCellCount)


window_x = 800
windows_y = 600
length_bound = 100
camera_move_speed = window_x / length_bound * 0.1

@ti.kernel
def init_particles():
    for i in range(N):
        particles_pos[i] = ti.Vector([ti.random() * length_bound for _ in range(3)])
        particles_vel[i] = ti.Vector([ti.random() * 2 - 1 for _ in range(3)]) * speed_limit

# @ti.kernel
# def init_simulation():
#     init_particles()
#     global gridStartIndices, gridEndIndices
#     gridCellWidth = 2.0 * perception_radius
#     halfSideCount = int((length_bound / gridCellWidth)) + 1
#     gridSideCount = 2 * halfSideCount
#     gridCellCount = gridSideCount * gridSideCount * gridSideCount
#     gridInverseCellWidth = 1.0
#     gridStartIndices = ti.Vector.field(1, dtype=ti.i32,shape=gridCellCount)
#     gridEndIndices = ti.Vector.field(1, dtype=ti.i32, shape=gridCellCount)


@ti.func
def gridIndex3Dto1D(x: int, y: int, z: int, gridResolution: int):
    return x + y * gridResolution + z * gridResolution * gridResolution;

@ ti.kernel
def IdentifyCellStartEnd():
    # Identify the start point of each cell in the gridIndices array
    # This is basically a parallel unrolling of a loop that goes
    # "this index doesn't match the one before it, must be a new cell!"
    for i in range(N):
        cellIndex = particleGridIndices[i]
        if i == 0:
            gridStartIndices[cellIndex] = i
        if i == N-1:
            gridEndIndices[cellIndex] = i
        cellBefore = particleGridIndices[i-1]
        if cellBefore != cellIndex:
            gridEndIndices[cellBefore] = i-1
            gridStartIndices[cellIndex] = i


@ti.kernel
def ComputeIndices():
    # compute grid and boid indices
    for i in range(N):
        cell_index_3D = ti.math.floor(particles_pos[i] * gridInverseCellWidth)
        particleGridIndices[i] = gridIndex3Dto1D(cell_index_3D[0],cell_index_3D[1],cell_index_3D[2], gridSideCount)
        particleArrayIndices[i] = i

# this function is used to indicate that a cell does not have any boid inside
@ti.kernel
def resetIntBuffer():
    for i in range(gridCellCount):
        gridStartIndices[i] = -1
        gridEndIndices[i] = -1

@ti.kernel
def kernUpdatePos():
    # update position after grid search
    for i in range(N):
        velocity = particles_vel[i]
        velocity = limit_velocity(velocity)

        particles_pos[i] += velocity * 0.1  # Assuming 0.1 as a time step equivalent
        particles_vel[i] = velocity

        for j in ti.static(range(3)):
            if particles_pos[i][j] < 0:
                particles_pos[i][j] += length_bound
            elif particles_pos[i][j] > length_bound:
                particles_pos[i][j] -= length_bound

@ti.kernel
def updateVelScatteredSearch():
    for i in range(N):
        velocity_change = ti.Vector([0.,0.,0.])
        center = ti.Vector([0.,0.,0.])
        avoid = ti.Vector([0.,0.,0.])
        follow = ti.Vector([0.,0.,0.])
        n_neighbor_r1 = 0
        n_neighbor_r3 = 0
        pos_self = particles_pos[i]
        cur_cell_index_3D = ti.math.floor(particles_pos[i] * gridInverseCellWidth, dtype= ti.i32)
        # these two variables are used to check the boundary conditions
        check_neg = ti.Vector([0, 0, 0])
        check_pos = ti.Vector([0, 0, 0])
        if cur_cell_index_3D[0] - 1 > 0:
            check_neg[0] = 1
        else:
            check_neg[0] = 0
        if cur_cell_index_3D[1] - 1 > 0:
            check_neg[1] = 1
        else:
            check_neg[1] = 0
        if cur_cell_index_3D[2] - 1 > 0:
            check_neg[2] = 1
        else:
            check_neg[2] = 0

        if cur_cell_index_3D[0] + 1 < gridSideCount:
            check_pos[0] = 1
        else:
            check_pos[0] = 0
        if cur_cell_index_3D[1] + 1 < gridSideCount:
            check_pos[1] = 1
        else:
            check_pos[1] = 0
        if cur_cell_index_3D[2] + 1 < gridSideCount:
            check_pos[2] = 1
        else:
            check_pos[2] = 0    
        for z in range(cur_cell_index_3D[2] - check_neg[2], cur_cell_index_3D[2]+check_pos[2]+1):
            for y in range(cur_cell_index_3D[1] - check_neg[1], cur_cell_index_3D[1] + check_pos[1] + 1):
                for x in range(cur_cell_index_3D[0] - check_neg[0], cur_cell_index_3D[0] + check_pos[0] + 1):
                    neighbor_cell_1D = gridIndex3Dto1D(x,y,z,gridSideCount)

                    if gridStartIndices[neighbor_cell_1D] == -1 or gridEndIndices[neighbor_cell_1D] == -1:
                        # no bird in the cell
                        continue
                    for idx in range(gridStartIndices[neighbor_cell_1D], gridEndIndices[neighbor_cell_1D]):
                        index_other = particleArrayIndices[idx]
                        pos_other = particles_pos[index_other]
                        if i != index_other:
                            dis = (pos_other - pos_self).norm()
                            # center rule
                            if dis < perception_radius:
                                center += pos_other
                                n_neighbor_r1 += 1
                            # avoid rule
                            if dis < avoidance_radius:
                                avoid -= (pos_other - pos_self)
                            if dis < perception_radius:
                                follow += particles_vel[index_other]
                                n_neighbor_r3 +=1
            if n_neighbor_r1 > 0:
                velocity_change += (center/float(n_neighbor_r1) - pos_self) * centering_factor
            velocity_change += avoid * repulsion_factor
            if n_neighbor_r3 > 0:
                velocity_change += (follow / float(n_neighbor_r3)) * matching_factor
                # update velocity
                particles_vel[i] += velocity_change



def sort_keys_scattered():
    particleArrayIndices_np = particleArrayIndices.to_numpy()
    particleGridIndices_np = particleGridIndices.to_numpy()

    # Get the indices that would sort the particleGridIndices array.
    sorted_indices = np.argsort(particleGridIndices_np)

    # Use the sorted indices to reorder the arrays.
    sorted_particleGridIndices = particleGridIndices_np[sorted_indices]
    sorted_particleArrayIndices = particleArrayIndices_np[sorted_indices]
    particleArrayIndices.from_numpy(sorted_particleArrayIndices)
    particleGridIndices.from_numpy(sorted_particleGridIndices)

def sort_keys_coherent():
    particleArrayIndices_np = particleArrayIndices.to_numpy()
    particleGridIndices_np = particleGridIndices.to_numpy()
    particlesPos_np = particles_pos.to_numpy()
    particlesVel_np = particles_vel.to_numpy()
    # Get the indices that would sort the particleGridIndices array.
    sorted_indices = np.argsort(particleGridIndices_np)

    # Use the sorted indices to reorder the arrays.
    sorted_particleGridIndices = particleGridIndices_np[sorted_indices]
    sorted_particleArrayIndices = particleArrayIndices_np[sorted_indices]
    sorted_particlesPos = particlesPos_np[sorted_indices]
    sorted_particlesVel = particlesVel_np[sorted_indices]
    particleArrayIndices.from_numpy(sorted_particleArrayIndices)
    particleGridIndices.from_numpy(sorted_particleGridIndices)
    particles_pos.from_numpy(sorted_particlesPos)
    particles_vel.from_numpy(sorted_particlesVel)

def stepSimulationScatteredGrid():
    ComputeIndices()
    sort_keys_scattered()
    resetIntBuffer()
    IdentifyCellStartEnd()
    updateVelScatteredSearch()
    kernUpdatePos()

def stepSimulationCoherentGrid():
    ComputeIndices()
    sort_keys_coherent()
    resetIntBuffer()
    IdentifyCellStartEnd()
    updateVelScatteredSearch()
    kernUpdatePos()

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
                repulsion -= (particles_pos[j] - particles_pos[i])
    return repulsion * repulsion_factor

@ti.func
def align_velocity(i):
    average_vel = ti.Vector([0.0, 0.0, 0.0])
    count = 0
    for j in range(N):
        if i != j:
            distance = (particles_pos[j] - particles_pos[i]).norm()
            if distance < perception_radius:
                average_vel += particles_vel[j]
                count += 1
    if count > 0:
        average_vel /= count
    return average_vel* matching_factor


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



SetScatteredGrid = 0
SetCoherentGrid = 1
init_particles()

# Create a window for rendering
window = ti.ui.Window("3D Moving Particles", (window_x, windows_y))
canvas = window.get_canvas()
scene = window.get_scene()
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
    if SetScatteredGrid == 1:
        stepSimulationScatteredGrid()
    elif SetCoherentGrid == 1:
        stepSimulationCoherentGrid()
    else:
        update_particles()

    scene.ambient_light((0.5, 0.5, 0.5)) 
    # scene.point_light(pos=(0, 10, -10), color=(1, 1, 1)) 

    # Render particles
    scene.particles(particles_pos, radius=0.1, color=(0.4, 0.6, 0.9))
    scene.lines(box_anchors, indices=box_lines_indices, color = (0.99, 0.99, 0.01), width = 2.0)
    
    
    canvas.scene(scene)
    window.show()
