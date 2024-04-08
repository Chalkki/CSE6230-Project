#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>

#include <cuda.h>

#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// LOOK-2.1 LOOK-2.3 - toggles for UNIFORM_GRID and COHERENT_GRID
#define VISUALIZE 1
#define UNIFORM_GRID 0
#define COHERENT_GRID 0

// LOOK-1.2 - change this to adjust particle count in the simulation
const int N_FOR_VIS = 5000;


int main(int argc, char* argv[]) {
  if (init(argc, argv)) {
    mainLoop();
    clear_simulation();
    return 0;
  } else {
    return 1;
  }
}


std::string deviceName;

bool init(int argc, char **argv) {
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count) {
        return false;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    int major = deviceProp.major;å
    int minor = deviceProp.minor;

    std::ostringstream ss;
    ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
    deviceName = ss.str()
    init_simulation();
    return true;
}




void runCUDA() {

    #if UNIFORM_GRID && COHERENT_GRID
    stepSimulationCoherentGrid(dt);
    #elif UNIFORM_GRID
    stepSimulationScatteredGrid(dt);
    #else
    simulate_one_step_naive(dt);
    #endif

    // #if VISUALIZE
    // Boids::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
    // #endif
}

void mainLoop() {
    double fps = 0;
    double timebase = 0;
    int frame = 0;

    // Boids::unitTest(); 

    while (true) {

      frame++;

      if (time - timebase > 1.0) {
        fps = frame / (time - timebase);
        timebase = time;
        frame = 0;
      }

      runCUDA();

      std::ostringstream ss;
      ss << "[";
      ss.precision(1);
      ss << std::fixed << fps;
      ss << " fps] " << deviceName;

      #if VISUALIZE
      #endif
    }

  }
