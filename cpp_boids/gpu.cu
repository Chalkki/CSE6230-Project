#include "common.h"
#include <cuda.h>

#define NUM_THREADS 256

///

// Put any static global variables here that you will use throughout the simulation.
int blks;
Vec3* dev_vel1;
Vec3* dev_vel2;


int* dev_particleArrayIndices; // What tid in dev_pos and dev_velX represents this particle?
int* dev_particleGridIndices; // What grid cell is this particle in?
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
    Vec3 velocity_change;
    Vec3 perceived_center;
    Vec3 c;
    Vec3 perceived_velocity;

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
    cudaMalloc((void**)&dev_pos2, N * sizeof(Vec3));
    cudaDeviceSynchronize();
        
}

__device__ int gridtid3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

void simulate_one_step_naive(Vec3 * pos, int num_parts) {
    kernUpdateVelocityBruteForce <<<blks, NUM_THREADS >>>(num_parts, pos, dev_vel1, dev_vel2);
    move_gpu_pos<<<blks, NUM_THREADS>>>(num_parts, dev_pos, dev_vel2);
    // ping-pong the velocity buffers
    Vec3* temp = dev_vel2;
    dev_vel2 = dev_vel1;
    dev_vel1 = temp;
}


void stepSimulationCoherentGrid(Vec3 * pos, int num_parts) {
    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
    // Rule 2: boids try to stay a distance d away from each other
    // Rule 3: boids try to match the speed of surrounding boids
}



void stepSimulationScatteredGrid(Vec3 * pos, int num_parts) {
  

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
}
