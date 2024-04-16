#include "common.h"
#include "happly.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <iostream>
#include <random>
#include <vector>
#include <thrust/random.h>
// toggles for UNIFORM_GRID and COHERENT_GRID
#define NUM_THREADS 256
// change this to adjust particle count in the simulation
const int N_FOR_VIS = 5000;
std::string _method;
__host__ __device__ unsigned int hash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}


// Function for generating a random vec3.
__host__ __device__ Vec3 generateRandomVec3(float time, int index) {
	thrust::default_random_engine rng(hash((int)(index * time)));
	thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

	return Vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

// CUDA kernel for generating boids with a specified mass randomly around the star.

__global__ void kernGenerateRandomPosArray(int time, int N, Vec3* arr) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		Vec3 rand = generateRandomVec3(time, index);
		arr[index].x = scale * rand.x;
		arr[index].y = scale * rand.y;
		arr[index].z = scale * rand.z;
	}
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

void runCUDA(Vec3* pos, int num_parts, int i) {
  if(i == 1){
    simulate_one_step_naive(pos, num_parts);
    _method = "naive";
  }else if(i==2){
    stepSimulationScatteredGrid(pos, num_parts);
    _method = "scatteredGrid";
  }else if(i==3){
    stepSimulationCoherentGrid(pos, num_parts);
    _method = "coherentGrid";
  } else if (i==5) {
    stepSimulationScatteredGrid_prefix(pos, num_parts);
    _method = "scatteredGrid with prefix";
  }else if(i == 4){
    stepSimulationCoherentBoitGrid(pos, num_parts);
    _method = "coherentGrid";
  } else if (i==6) {
    stepSimulationCoherentGrid_prefix(pos, num_parts);
    _method = "coherentGrid with prefix";
  }
}

void save_boid_data( const Vec3* pos, int num_parts, const std::string& path) {
    happly::PLYData plyOut;
    std::vector<std::array<double, 3>> points;

    for (int i = 0; i < num_parts; ++i) {
      const Vec3& p = pos[i];
      points.push_back({p.x, p.y, p.z});
    }
    plyOut.addVertexPositions(points);

    plyOut.write(path, happly::DataFormat::ASCII);
}

int main(int argc, char* argv[]) {
    // Initialize Particles
    int num_parts = find_int_arg(argc, argv, "-n", N_FOR_VIS);
    int method = find_int_arg(argc, argv, "-m", 1);
    int save = find_int_arg(argc, argv, "-s", 0);
    dim3 fullBlocksPerGrid((num_parts+NUM_THREADS-1)/ NUM_THREADS);
    Vec3* host_pos = new Vec3[num_parts];
    Vec3* gpu_pos ;
    cudaMalloc((void**)&gpu_pos, num_parts * sizeof(Vec3));
    kernGenerateRandomPosArray <<<fullBlocksPerGrid, NUM_THREADS >> > (1, num_parts, gpu_pos);
    // Initialize Simulation
    init_simulation(gpu_pos, num_parts);
    std::string prefix = "../boid_ply_data/";
    auto start_time = std::chrono::steady_clock::now();
      int frame = 0;
      for (int step = 0; step < nsteps; ++step) {
          runCUDA(gpu_pos, num_parts, method);
          //cudaDeviceSynchronize();
        // Save state if necessary
        if(save == 1 && (step % savefreq == 0|| step == nsteps-1)){
          cudaMemcpy(host_pos, gpu_pos, num_parts * sizeof(Vec3), cudaMemcpyDeviceToHost);
          save_boid_data(host_pos, num_parts,prefix+ std::to_string(frame)+".ply");
          frame++;
        }
    }


    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();
    // Finalize
    std::cout << "Simulation Time = " << seconds << " seconds for " << num_parts << " boids.\n";
     std::cout << "Average FPS: " << float(nsteps / seconds) << " using " << _method << std::endl;
    clear_simulation();
    cudaFree(gpu_pos); 
    delete[] host_pos;
}





// void mainLoop() {
//     double fps = 0;
//     double timebase = 0;
//     int frame = 0;

//     // Boids::unitTest(); 

//     while (true) {

//       frame++;

//       if (time - timebase > 1.0) {
//         fps = frame / (time - timebase);
//         timebase = time;
//         frame = 0;
//       }

//       runCUDA();

//       std::ostringstream ss;
//       ss << "[";
//       ss.precision(1);
//       ss << std::fixed << fps;
//       ss << " fps] " << deviceName;

//       #if VISUALIZE
//       #endif
//     }

//   }
