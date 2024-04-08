#include "common.h"
#include <cuda.h>

#define NUM_THREADS 256

///

// Put any static global variables here that you will use throughout the simulation.
int blks;

// __device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
//     double dx = neighbor.x - particle.x;
//     double dy = neighbor.y - particle.y;
//     double r2 = dx * dx + dy * dy;
//     if (r2 > cutoff * cutoff)
//         return;
//     // r2 = fmax( r2, min_r*min_r );
//     r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
//     double r = sqrt(r2);

//     //
//     //  very simple short-range repulsive force
//     //
//     double coef = (1 - cutoff / r) / r2 / mass;
//     particle.ax += coef * dx;
//     particle.ay += coef * dy;
// }


__device__ void computeVelocityChange(particle_t& particle, particle_t& neighbor) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids
  //return glm::vec3(0.0f, 0.0f, 0.0f); set boids velocity

  
  //get new velocity
}

__global__ void compute_forces_gpu_brute_force(particle_t* particles, int num_parts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = particles[tid].az = 0;
    for (int j = 0; j < num_parts; j++)
        // apply_force_gpu(particles[tid], particles[j]);

        //atomicexcp()
        //replace old velcoity new velocity
}

__global__ void move_gpu_pos(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->vz += p->az * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    p->z += p->vz * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
    while(p->z < 0 || p->z > size){
        p->z = p->z < 0 ? -(p->z) : 2 * size - p->z;
        p->vz = -(p->vz);
    }
}

void init_simulation(int num_parts) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
}

void simulate_one_step_naive(float dt) {
    // parts live in GPU memory
    // Rewrite this function

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);

    // Move particles
    move_gpu_pos<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}


void stepSimulationCoherentGrid(float dt) {
    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
    // Rule 2: boids try to stay a distance d away from each other
    // Rule 3: boids try to match the speed of surrounding boids
}



void stepSimulationScatteredGrid(float dt) {
  

}




// Clear allocations
void clear_simulation() {
    //cudaFree();
    //gpu memory of pos, vel1, vel2
    cudaFree(dev_vel1);
    cudaFree(dev_vel2);
    cudaFree(dev_pos);

    //gpu to cpu final result
    cudaFree(dev_intKeys);
    cudaFree(dev_intValues);
}
