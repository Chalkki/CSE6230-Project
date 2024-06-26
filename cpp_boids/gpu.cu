#include "common.h"
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
//#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <cuda_runtime.h>
#define NUM_THREADS 256
///

// Put any static global variables here that you will use throughout the simulation.
int blks;
Vec3* dev_vel1;
Vec3* dev_vel2;


int* dev_particleArrayIndices; // What tid in dev_pos and dev_velX represents this particle?
int* dev_particleGridIndices; // What grid cell is this particle in?
int* dev_atomicCount;   // for prefix sum, how many particles is filled into a block
int* dev_gridParticleCount; // for prefix sum, how many particles within this block?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;
int* dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int* dev_gridCellEndIndices;   // to this cell?

// the position and velocity data to be coherent within cells.
Vec3* dev_pos2;

int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
Vec3 gridMinimum;

__device__ Vec3 computeVelocityChange(int N, int iSelf, const Vec3 *pos, const Vec3 *vel) {
    // compute velocity change in brute force
    Vec3 pos_self = pos[iSelf];
    Vec3 velocity_change = Vec3();
    Vec3 perceived_center = Vec3();
    Vec3 c = Vec3() ;
    Vec3 perceived_velocity = Vec3();

    unsigned int num_neighbors_r1 = 0;
    unsigned int num_neighbors_r3 = 0;

    for (int i = 0; i < N; i++)
    {
        if (i == iSelf)
        {
            continue;
        }

        Vec3 pos_other = pos[i];
        float dist_to_other = norm(pos_other - pos_self);

        // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
        if (dist_to_other < perception_radius)
        {
            perceived_center += pos_other;
            num_neighbors_r1++;
        }

        // Rule 2: boids try to stay a distance d away from each other
        if (dist_to_other < avoidance_radius)
        {
            c -= (pos_other - pos_self);
        }

        // Rule 3: boids try to match the speed of surrounding boids
        if (dist_to_other < perception_radius)
        {
            perceived_velocity += vel[i];
            num_neighbors_r3++;
        }
    }

    if (num_neighbors_r1 > 0)
    {
        velocity_change += (perceived_center / (float) num_neighbors_r1 - pos_self) * centering_factor;
    }

    velocity_change += c * repulsion_factor;

    if (num_neighbors_r3 > 0)
    {
        velocity_change += (perceived_velocity / (float) num_neighbors_r3) * matching_factor;
    }

    return velocity_change;
}
  

__global__ void move_gpu_pos(int N, Vec3 *pos, Vec3 *vel) {
  // Update position by velocity
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= N) {
    return;
  }
  Vec3 thisPos = pos[tid];
  thisPos += vel[tid] * dt;

  // boundary condition
  thisPos.x = thisPos.x < -scale ? scale : thisPos.x;
  thisPos.y = thisPos.y < -scale ? scale : thisPos.y;
  thisPos.z = thisPos.z < -scale ? scale : thisPos.z;

  thisPos.x = thisPos.x > scale ? -scale : thisPos.x;
  thisPos.y = thisPos.y > scale ? -scale : thisPos.y;
  thisPos.z = thisPos.z > scale ? -scale : thisPos.z;

  pos[tid] = thisPos;
}

__device__ int gridtid3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernUpdateVelocityBruteForce(int N, Vec3 *pos,
    Vec3 *vel1, Vec3 *vel2) {
    // Compute Boid associated with thread
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid >= N) {
        return;
    }

    // Compute a new velocity based on pos and vel1
    Vec3 new_velocity = vel1[tid] + computeVelocityChange(N, tid, pos, vel1);

    // Clamp the speed
    if (norm(new_velocity) > speed_limit)
    {
        new_velocity = speed_limit * normalize(new_velocity);
    }

    // Record the new velocity into vel2.
    vel2[tid] = new_velocity;
}


__global__ void kernUpdateVelocityScattered(int N, int gridRes, Vec3 gridMin, float inverseCW, float cW, int* startIndex, int* endIndex, int* particleArrayIndex, Vec3 * pos, Vec3* vel1, Vec3* vel2){
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid >= N) {
        return;
    }
    
    Vec3 pos_self = pos[tid];
    Vec3 grid_c = inverseCW * (pos_self - gridMin);
    Vec3 grid_c_int = floor(grid_c);
    
    Vec3 velocity_change = Vec3();
    Vec3 perceived_center = Vec3();
    Vec3 c = Vec3();
    Vec3 perceived_velocity = Vec3();
    
    unsigned int num_neighbors_r1 = 0;
    unsigned int num_neighbors_r3 = 0;
    
    for(int delta_z = -1;  delta_z <= 1; delta_z++){
        for(int delta_y = -1;  delta_y <= 1; delta_y++){
            for(int delta_x = -1;  delta_x <= 1; delta_x++){
                int neigh_id = gridtid3Dto1D(grid_c_int.x + delta_x,
                                             grid_c_int.y + delta_y,
                                             grid_c_int.z + delta_z, gridRes);
                if(startIndex[neigh_id] == -1){
                    continue;
                }
                for(int c_ind = startIndex[neigh_id]; c_ind <= endIndex[neigh_id]; c_ind++){
                    int other_boids  =  particleArrayIndex[c_ind];
                    Vec3 pos_other = pos[other_boids];
                    
                    if(other_boids != tid){
                    
                        Vec3 dist = pos_other - pos_self;
                        float dist_mag2 = dot(dist, dist);

                        if (dist_mag2 < perception_radius2)
                        {
                            perceived_center += pos_other;
                            num_neighbors_r1++;
                        }
                
                        // Rule 2: boids try to stay a distance d away from each other
                        if (dist_mag2 < avoidance_radius2)
                        {
                            c -= (pos_other - pos_self);
                        }
                
                        // Rule 3: boids try to match the speed of surrounding boids
                        if (dist_mag2 < perception_radius2)
                        {
                            perceived_velocity += vel1[other_boids];
                            num_neighbors_r3++;
                        }
                    }
                }
            }
        }
    }
        
    if (num_neighbors_r1 > 0)
    {
        velocity_change += (perceived_center / (float) num_neighbors_r1 - pos_self) * centering_factor;
    }

    velocity_change += c * repulsion_factor;

    if (num_neighbors_r3 > 0)
    {
        velocity_change += (perceived_velocity / (float) num_neighbors_r3) * matching_factor;
    }
    
    
    Vec3 new_velocity = vel1[tid] + velocity_change;
    
    if (norm(new_velocity) > speed_limit)
    {
        new_velocity = speed_limit * normalize(new_velocity);
    }

    // Record the new velocity into vel2.
    vel2[tid] =  new_velocity;

}

__global__ void posReshuffle(
    int num_parts, Vec3* pos1, Vec3* pos2, Vec3* vel1, Vec3* vel2,
    int* particleArrayIndices) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid >= num_parts) {
        return;
    }

    int p_arr_idx = particleArrayIndices[tid];
    pos2[tid] = pos1[p_arr_idx];
    vel2[tid] = vel1[p_arr_idx];
}


__global__ void kernUpdateVelNeighborCoherent(int N, int gridRes, Vec3 gridMin, float inverseCW, float cW, int* startIndex, int* endIndex, Vec3 * pos, Vec3* vel1, Vec3* vel2){
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid >= N) {
        return;
    }
    
     Vec3 pos_self = pos[tid];
    Vec3 grid_c = inverseCW * (pos_self - gridMin);
    Vec3 grid_c_int = floor(grid_c);
    
    Vec3 velocity_change = Vec3();
    Vec3 perceived_center = Vec3();
    Vec3 c = Vec3();
    Vec3 perceived_velocity = Vec3();
    
    unsigned int num_neighbors_r1 = 0;
    unsigned int num_neighbors_r3 = 0;
    
    for(int delta_z = -1;  delta_z <= 1; delta_z++){
        for(int delta_y = -1;  delta_y <= 1; delta_y++){
            for(int delta_x = -1;  delta_x <= 1; delta_x++){
                int x = grid_c_int.x + delta_x, y = grid_c_int.y + delta_y, z = grid_c_int.z + delta_z;
                if (x < 0 || y < 0 || z < 0 || x >= gridRes || y >= gridRes || z >= gridRes) continue;
                int neigh_id = gridtid3Dto1D(x, y, z, gridRes);

                if(startIndex[neigh_id] == -1){
                    continue;
                }
                for(int c_ind = startIndex[neigh_id]; c_ind <= endIndex[neigh_id]; c_ind++){
                    Vec3 pos_other = pos[c_ind];
                    
                    if(c_ind != tid){
                    
                        Vec3 dist = pos_other - pos_self;
                        float dist_mag2 = dot(dist, dist);

                        if (dist_mag2 < perception_radius2)
                        {
                            perceived_center += pos_other;
                            num_neighbors_r1++;
                        }
                
                        // Rule 2: boids try to stay a distance d away from each other
                        if (dist_mag2 < avoidance_radius2)
                        {
                            c -= (pos_other - pos_self);
                        }
                
                        // Rule 3: boids try to match the speed of surrounding boids
                        if (dist_mag2 < perception_radius2)
                        {
                            perceived_velocity += vel1[c_ind];
                            num_neighbors_r3++;
                        }
                    }
                }
            }
        }
    }
        
    if (num_neighbors_r1 > 0)
    {
        velocity_change += (perceived_center / (float) num_neighbors_r1 - pos_self) * centering_factor;
    }

    velocity_change += c * repulsion_factor;

    if (num_neighbors_r3 > 0)
    {
        velocity_change += (perceived_velocity / (float) num_neighbors_r3) * matching_factor;
    }
    
    
    Vec3 new_velocity = vel1[tid] + velocity_change;
    
    if (norm(new_velocity) > speed_limit)
    {
        new_velocity = speed_limit * normalize(new_velocity);
    }

    // Record the new velocity into vel2.
    vel2[tid] =  new_velocity;

}

__global__ void kernUpdateVelocityScattered_prefix(int N, int gridRes, Vec3 gridMin, float inverseCW, float cW, int* startIndex, int* endIndex, int* particleArrayIndex, Vec3 * pos, Vec3* vel1, Vec3* vel2){
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid >= N) {
        return;
    }
    
    Vec3 pos_self = pos[tid];
    Vec3 grid_c = inverseCW * (pos_self - gridMin);
    Vec3 grid_c_int = floor(grid_c);
    
    Vec3 velocity_change = Vec3();
    Vec3 perceived_center = Vec3();
    Vec3 c = Vec3();
    Vec3 perceived_velocity = Vec3();
    
    unsigned int num_neighbors_r1 = 0;
    unsigned int num_neighbors_r3 = 0;
    
    for(int delta_z = -1;  delta_z <= 1; delta_z++){
        for(int delta_y = -1;  delta_y <= 1; delta_y++){
            for(int delta_x = -1;  delta_x <= 1; delta_x++){
                int neigh_id = gridtid3Dto1D(grid_c_int.x + delta_x,
                                             grid_c_int.y + delta_y,
                                             grid_c_int.z + delta_z, gridRes);

                for(int c_ind = startIndex[neigh_id]; c_ind < endIndex[neigh_id]; c_ind++){
                    int other_boids  =  particleArrayIndex[c_ind];
                    Vec3 pos_other = pos[other_boids];
                    
                    if(other_boids != tid){
                        Vec3 dist = pos_other - pos_self;
                        float dist_mag2 = dot(dist, dist);

                        if (dist_mag2 < perception_radius2)
                        {
                            perceived_center += pos_other;
                            num_neighbors_r1++;
                        }
                
                        // Rule 2: boids try to stay a distance d away from each other
                        if (dist_mag2 < avoidance_radius2)
                        {
                            c -= (pos_other - pos_self);
                        }
                
                        // Rule 3: boids try to match the speed of surrounding boids
                        if (dist_mag2 < perception_radius2)
                        {
                            perceived_velocity += vel1[other_boids];
                            num_neighbors_r3++;
                        }
                    }
                }
            }
        }
    }
        
    if (num_neighbors_r1 > 0)
    {
        velocity_change += (perceived_center / (float) num_neighbors_r1 - pos_self) * centering_factor;
    }

    velocity_change += c * repulsion_factor;

    if (num_neighbors_r3 > 0)
    {
        velocity_change += (perceived_velocity / (float) num_neighbors_r3) * matching_factor;
    }
    
    
    Vec3 new_velocity = vel1[tid] + velocity_change;
    if (dot(new_velocity, new_velocity) > speed_limit * speed_limit)
    {
        new_velocity = speed_limit * normalize(new_velocity);
    }

    // Record the new velocity into vel2.
    vel2[tid] =  new_velocity;

}

__global__ void kernUpdateVelNeighborCoherent_prefix(int N, int gridRes, Vec3 gridMin, float inverseCW, float cW, int* startIndex, int* endIndex, Vec3 * pos, Vec3* vel1, Vec3* vel2){
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid >= N) {
        return;
    }
    
    Vec3 pos_self = pos[tid];
    Vec3 grid_c = inverseCW * (pos_self - gridMin);
    Vec3 grid_c_int = floor(grid_c);
    
    Vec3 velocity_change = Vec3();
    Vec3 perceived_center = Vec3();
    Vec3 c = Vec3();
    Vec3 perceived_velocity = Vec3();

    unsigned int num_neighbors_r1 = 0;
    unsigned int num_neighbors_r3 = 0;
    
    for(int delta_z = -1;  delta_z <= 1; delta_z++){
        for(int delta_y = -1;  delta_y <= 1; delta_y++){
            for(int delta_x = -1;  delta_x <= 1; delta_x++){
                int x = grid_c_int.x + delta_x, y = grid_c_int.y + delta_y, z = grid_c_int.z + delta_z;
                if (x < 0 || y < 0 || z < 0 || x >= gridRes || y >= gridRes || z >= gridRes) continue;
                int neigh_id = gridtid3Dto1D(x, y, z, gridRes);


                for(int c_ind = startIndex[neigh_id]; c_ind <= endIndex[neigh_id]; c_ind++){
                    Vec3 pos_other = pos[c_ind];
                    
                    if(c_ind != tid){
                    
                        Vec3 dist = pos_other - pos_self;
                        float dist_mag2 = dot(dist, dist);

                        if (dist_mag2 < perception_radius2)
                        {
                            perceived_center += pos_other;
                            num_neighbors_r1++;
                        }
                
                        // Rule 2: boids try to stay a distance d away from each other
                        if (dist_mag2 < avoidance_radius2)
                        {
                            c -= (pos_other - pos_self);
                        }
                
                        // Rule 3: boids try to match the speed of surrounding boids
                        if (dist_mag2 < perception_radius2)
                        {
                            perceived_velocity += vel1[c_ind];
                            num_neighbors_r3++;
                        }
                    }
                }
            }
        }
    }
        
    if (num_neighbors_r1 > 0)
    {
        velocity_change += (perceived_center / (float) num_neighbors_r1 - pos_self) * centering_factor;
    }

    velocity_change += c * repulsion_factor;

    if (num_neighbors_r3 > 0)
    {
        velocity_change += (perceived_velocity / (float) num_neighbors_r3) * matching_factor;
    }
    
    
    Vec3 new_velocity = vel1[tid] + velocity_change;
    
    if (norm(new_velocity) > speed_limit)
    {
        new_velocity = speed_limit * normalize(new_velocity);
    }

    // Record the new velocity into vel2.
    vel2[tid] =  new_velocity;

}


void init_simulation(Vec3 * pos, int num_parts) {

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    cudaMalloc((void**)&dev_vel1, num_parts * sizeof(Vec3));
    cudaMalloc((void**)&dev_vel2, num_parts * sizeof(Vec3));

    // computing grid params
    gridCellWidth = 2.0f * perception_radius;

    int halfSideCount = (int)(scale / gridCellWidth) + 1;
    gridSideCount = 2 * halfSideCount;
    gridCellCount = gridSideCount * gridSideCount * gridSideCount;
    gridInverseCellWidth = 1.0f / gridCellWidth;
    float halfGridWidth = gridCellWidth * halfSideCount;
    gridMinimum.x -= halfGridWidth;
    gridMinimum.y -= halfGridWidth;
    gridMinimum.z -= halfGridWidth;
    //  Allocate additional buffers here.
    cudaMalloc((void**)&dev_particleArrayIndices, num_parts * sizeof(int));
    cudaMalloc((void**)&dev_particleGridIndices, num_parts * sizeof(int));
    cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
    cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
    cudaMalloc((void**)&dev_pos2, num_parts * sizeof(Vec3));

    cudaMalloc((void**)&dev_atomicCount, gridCellCount * sizeof(int));
    cudaMalloc((void**)&dev_gridParticleCount, gridCellCount * sizeof(int));
    cudaDeviceSynchronize();
        
}


__global__ void bufferReset(int num_parts,int* buf, int val){
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid < num_parts) {
        buf[tid] = val;
    }
}

__global__ void identifyCellInfo(int num_parts, int* gridIndex, int* gridCellStart, int* gridCellEnd){
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid >= num_parts) {
        return;
    }
    
    int part_index = gridIndex[tid];
    if((tid ==0) || (part_index != gridIndex[tid-1])){
        gridCellStart[part_index] = tid;
    }
    if((tid == num_parts-1) || (part_index != gridIndex[tid+1])){
        gridCellEnd[part_index] = tid;
    }

}

__global__ void computeIndices(int num_parts, int gridres, Vec3 gridmin, float gridInverseCellWidth, Vec3* position, int* indices, int* gridIndex){
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid >= num_parts) {
        return;
    }
    
    Vec3 cell_3D = floor(gridInverseCellWidth * (position[tid]-gridmin));
    gridIndex[tid] = gridtid3Dto1D(cell_3D.x, cell_3D.y, cell_3D.z, gridres);
    
    indices[tid] = tid;

}


void simulate_one_step_naive(Vec3 * pos, int num_parts) {
    kernUpdateVelocityBruteForce <<<blks, NUM_THREADS >>>(num_parts, pos, dev_vel1, dev_vel2);
    move_gpu_pos<<<blks, NUM_THREADS>>>(num_parts, pos, dev_vel2);
    // ping-pong the velocity buffers
    Vec3* temp = dev_vel2;
    dev_vel2 = dev_vel1;
    dev_vel1 = temp;
}


void stepSimulationCoherentGrid(Vec3 * pos, int num_parts) {
    
    dim3 block_per_grid((num_parts + NUM_THREADS - 1) / NUM_THREADS);
    dim3 block_per_cell((gridCellCount + NUM_THREADS - 1) / NUM_THREADS);
    computeIndices <<<block_per_grid, NUM_THREADS>>>(num_parts, gridSideCount, gridMinimum,gridInverseCellWidth,pos, dev_particleArrayIndices,dev_particleGridIndices);
    dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
    dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);

thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + num_parts, dev_thrust_particleArrayIndices);
    
    thrust::fill(thrust::device, dev_gridCellStartIndices, dev_gridCellStartIndices + gridCellCount, -1);
    identifyCellInfo<<<block_per_grid, NUM_THREADS>>> (num_parts, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    // posReshuffle<<<block_per_grid, NUM_THREADS>>> (num_parts, pos, dev_pos2, dev_vel1, dev_vel2, dev_particleArrayIndices);
    thrust::device_ptr<Vec3>thrust_pos(pos);
    thrust::device_ptr<Vec3>thrust_vel1(dev_vel1);
    thrust::device_ptr<Vec3>thrust_pos2(dev_pos2);
    thrust::device_ptr<Vec3>thrust_vel2(dev_vel2);

    thrust::gather(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + num_parts, thrust_pos,
                    thrust_pos2);
    thrust::gather(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + num_parts, thrust_vel1,
                    thrust_vel2);
    kernUpdateVelNeighborCoherent<<<block_per_grid, NUM_THREADS>>>(
        num_parts, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_pos2, dev_vel2, dev_vel1);
    move_gpu_pos<<<block_per_grid, NUM_THREADS>>>(num_parts, dev_pos2, dev_vel1);
    // ping-pong the position buffers
    std::swap(pos,dev_pos2);
}


void stepSimulationScatteredGrid(Vec3 * pos, int num_parts) {
    dim3 block_per_grid((num_parts + NUM_THREADS - 1) / NUM_THREADS);
    dim3 block_per_cell((gridCellCount + NUM_THREADS - 1) / NUM_THREADS);
    computeIndices <<<block_per_grid, NUM_THREADS>>>(num_parts, gridSideCount, gridMinimum,gridInverseCellWidth,pos, dev_particleArrayIndices,dev_particleGridIndices);
    
    dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
    dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + num_parts, dev_thrust_particleArrayIndices);

    bufferReset<<<block_per_cell, NUM_THREADS>>>(gridCellCount, dev_gridCellStartIndices, -1);
    bufferReset<<<block_per_cell, NUM_THREADS>>>(gridCellCount, dev_gridCellEndIndices, -1);
    // thrust::fill(thrust::device, dev_gridCellStartIndices, dev_gridCellStartIndices + gridCellCount, -1);
    identifyCellInfo<<<block_per_grid, NUM_THREADS>>> (num_parts, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    
    kernUpdateVelocityScattered <<<block_per_grid, NUM_THREADS >>>(num_parts, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, pos, dev_vel1, dev_vel2);
    move_gpu_pos<<<block_per_grid, NUM_THREADS>>>(num_parts, pos, dev_vel2);
    // ping-pong the velocity buffers
    Vec3* temp = dev_vel2;
    dev_vel2 = dev_vel1;
    dev_vel1 = temp;
}

// compute the particle block index and count number of particle within a given block
__global__ void prefixCount(int num_parts, int gridres, Vec3 gridmin, float gridInverseCellWidth, Vec3* position, int* partCount, int* gridIndex) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid >= num_parts) {
        return;
    }
    
    Vec3 cell_3D = floor(gridInverseCellWidth * (position[tid]-gridmin));
    int gid = gridtid3Dto1D(cell_3D.x, cell_3D.y, cell_3D.z, gridres);
    gridIndex[tid] = gid;
    atomicAdd(&partCount[gid], 1);
}

__global__ void prefixSort(int num_parts, int* prefixAtomicCount, int* sort_particles, int* gridCellStart, int* gridIndex) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    int gid = gridIndex[tid];

    int target = gridCellStart[gid] + atomicAdd(&prefixAtomicCount[gid], 1);
    sort_particles[target] = tid;
}


void stepSimulationCoherentGrid_prefix(Vec3 * pos, int num_parts) {
    thrust::fill(thrust::device, dev_atomicCount, dev_atomicCount + gridCellCount, 0);
    thrust::fill(thrust::device, dev_gridParticleCount, dev_gridParticleCount + gridCellCount, 0);

    dim3 block_per_grid((num_parts + NUM_THREADS - 1) / NUM_THREADS);
    dim3 block_per_cell((gridCellCount + NUM_THREADS - 1) / NUM_THREADS);
    prefixCount <<<block_per_grid, NUM_THREADS>>>(num_parts, gridSideCount, gridMinimum, gridInverseCellWidth,pos, dev_gridParticleCount, dev_particleGridIndices);

    thrust::exclusive_scan(thrust::device, dev_gridParticleCount, dev_gridParticleCount + gridCellCount, dev_gridCellStartIndices);
    thrust::transform(thrust::device, dev_gridParticleCount, dev_gridParticleCount + gridCellCount, dev_gridCellStartIndices, dev_gridCellEndIndices, thrust::plus<int>());
    prefixSort<<<block_per_grid, NUM_THREADS >>>(num_parts, dev_atomicCount, dev_particleArrayIndices, dev_gridCellStartIndices, dev_particleGridIndices);
    
    // posReshuffle<<<block_per_grid, NUM_THREADS>>> (num_parts, pos, dev_pos2, dev_vel1, dev_vel2, dev_particleArrayIndices);
    thrust::gather(thrust::device, dev_particleArrayIndices, dev_particleArrayIndices + num_parts, pos, dev_pos2);
    thrust::gather(thrust::device, dev_particleArrayIndices, dev_particleArrayIndices + num_parts, dev_vel1, dev_vel2);

    kernUpdateVelNeighborCoherent_prefix<<<block_per_grid, NUM_THREADS>>>(
        num_parts, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_pos2, dev_vel2, dev_vel1);
    move_gpu_pos<<<block_per_grid, NUM_THREADS>>>(num_parts, dev_pos2, dev_vel1);
    // ping-pong the velocity buffers
    Vec3* temp = dev_pos2;
    dev_pos2 = pos;
    pos = temp;
}

void stepSimulationScatteredGrid_prefix(Vec3 * pos, int num_parts) {
    dim3 block_per_grid((num_parts + NUM_THREADS - 1) / NUM_THREADS);
    dim3 block_per_cell((gridCellCount + NUM_THREADS - 1) / NUM_THREADS);

    bufferReset<<<block_per_cell, NUM_THREADS>>>(gridCellCount, dev_atomicCount, 0);
    bufferReset<<<block_per_cell, NUM_THREADS>>>(gridCellCount, dev_gridParticleCount, 0);
    
    // thrust::fill(thrust::device, dev_atomicCount, dev_atomicCount + gridCellCount, 0);
    // thrust::fill(thrust::device, dev_gridParticleCount, dev_gridParticleCount + gridCellCount, 0);

    prefixCount <<<block_per_grid, NUM_THREADS>>>(num_parts, gridSideCount, gridMinimum, gridInverseCellWidth,pos, dev_gridParticleCount, dev_particleGridIndices);

    thrust::exclusive_scan(thrust::device, dev_gridParticleCount, dev_gridParticleCount + gridCellCount, dev_gridCellStartIndices);
    thrust::transform(thrust::device, dev_gridParticleCount, dev_gridParticleCount + gridCellCount, dev_gridCellStartIndices, dev_gridCellEndIndices, thrust::plus<int>());
    prefixSort<<<block_per_grid, NUM_THREADS >>>(num_parts, dev_atomicCount, dev_particleArrayIndices, dev_gridCellStartIndices, dev_particleGridIndices);
    
    kernUpdateVelocityScattered_prefix <<<block_per_grid, NUM_THREADS >>>(num_parts, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, 
                                                                   dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, 
                                                                   pos, dev_vel1, dev_vel2);
    move_gpu_pos<<<block_per_grid, NUM_THREADS>>>(num_parts, pos, dev_vel2);
    // ping-pong the velocity buffers
    Vec3* temp = dev_vel2;
    dev_vel2 = dev_vel1;
    dev_vel1 = temp;

}

__device__ void bitonicCompare(int* keys, int* values, int i, int j, int dir) {
    int key_i = keys[i];
    int key_j = keys[j];
    int value_i = values[i];
    int value_j = values[j];

    if ((dir == 1 && key_i > key_j) || (dir == 0 && key_i < key_j)) {
        // Swap keys
        keys[i] = key_j;
        keys[j] = key_i;
        // Swap values
        values[i] = value_j;
        values[j] = value_i;
    }
}

__global__ void bitonicSortKernel(int* keys, int* values, int j, int k) {
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i^j;

    // The threads with the lowest ids sort the array.
    if ((ixj)>i) {
        if ((i&k) == 0) {
            // Ascending order
            bitonicCompare(keys, values, i, ixj, 1);
        }
        else {
            // Descending order
            bitonicCompare(keys, values, i, ixj, 0);
        }
    }
}

void bitonicSort(int* dev_keys, int* dev_values, int num_elements) {
    dim3 blocks((num_elements + 511)/512); // Number of blocks
    dim3 threads(512); // Number of threads per block

    int j, k;
    for (k = 2; k <= num_elements; k <<= 1) {
        for (j=k>>1; j>0; j=j>>1) {
            bitonicSortKernel<<<blocks, threads>>>(dev_keys, dev_values, j, k);
            cudaDeviceSynchronize();
        }
    }
}



void stepSimulationCoherentBoitGrid(Vec3 * pos, int num_parts) {
    dim3 block_per_grid((num_parts + NUM_THREADS - 1) / NUM_THREADS);
    dim3 block_per_cell((gridCellCount + NUM_THREADS - 1) / NUM_THREADS);

    computeIndices<<<block_per_grid, NUM_THREADS>>>(num_parts, gridSideCount, gridMinimum,gridInverseCellWidth,pos, dev_particleArrayIndices,dev_particleGridIndices);

    // Prepare for bitonic sort
    bitonicSort(dev_particleGridIndices, dev_particleArrayIndices, num_parts); // Custom bitonic sort for key-value pairs
    
    bufferReset<<<block_per_cell, NUM_THREADS>>>(gridCellCount, dev_gridCellStartIndices, -1);

    identifyCellInfo<<<block_per_grid, NUM_THREADS>>> (num_parts, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    posReshuffle<<<block_per_grid, NUM_THREADS>>> (num_parts, pos, dev_pos2, dev_vel1, dev_vel2, dev_particleArrayIndices);
//   thrust::device_ptr<Vec3>thrust_pos(pos);
//   thrust::device_ptr<Vec3>thrust_vel1(dev_vel1);
//   thrust::device_ptr<Vec3>thrust_pos2(dev_pos2);
//   thrust::device_ptr<Vec3>thrust_vel2(dev_vel2);

//   thrust::gather(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + num_parts, thrust_pos,
//                  thrust_pos2);
//   thrust::gather(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + num_parts, thrust_vel1,
//                  thrust_vel2);    
    kernUpdateVelNeighborCoherent<<<block_per_grid, NUM_THREADS>>>(
        num_parts, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_pos2, dev_vel2, dev_vel1);

    move_gpu_pos<<<block_per_grid, NUM_THREADS>>>(num_parts, dev_pos2, dev_vel1);
    // ping-pong the position buffers
    Vec3* temp = dev_pos2;
    dev_pos2 = pos;
    pos = temp;
}




// Clear allocations
void clear_simulation() {
    //cudaFree();
    //gpu memory of pos2, vel1, vel2, start indices, end indices, boid id array, gridid array
    cudaFree(dev_vel1);
    cudaFree(dev_vel2);
    cudaFree(dev_pos2);
    cudaFree(dev_gridCellStartIndices);
    cudaFree(dev_gridCellEndIndices);
    cudaFree(dev_particleArrayIndices);
    cudaFree(dev_particleGridIndices);
    cudaFree(dev_atomicCount);
    cudaFree(dev_gridParticleCount);
}