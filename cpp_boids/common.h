#ifndef __COMMON_H__
#define __COMMON_H__

#include <cstdint>
#include <cmath>
// Program Constants
#define nsteps   1000
#define savefreq 10
#define dt       0.2f
#define scale 100
#define centering_factor 0.01f
#define repulsion_factor 0.1f
#define matching_factor 0.1f
#define speed_limit 1.0f
#define perception_radius 5.0f
#define avoidance_radius 3.0f



struct Vec3 {
    float x;
    float y;
    float z;

     __host__ __device__ Vec3():x(0),y(0),z(0){}

     __host__ __device__ Vec3(float x_val, float y_val, float z_val): x(x_val),y(y_val),z(z_val){}

    __device__ void operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
    }

    __device__ void operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
    }
};

__device__ inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float normSquared(const Vec3& r) {
    return r.x * r.x + r.y * r.y + r.z * r.z;
}

__device__ inline float norm(const Vec3& r) {
    return sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
}
__device__ inline Vec3 normalize(const Vec3& a) {
    float mag = norm(a);
    return {a.x / mag, a.y / mag, a.z / mag};
}

__device__ inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ inline Vec3 operator*(float a, const Vec3& b) {
    return {b.x * a, b.y * a, b.z * a};
}

__device__ inline Vec3 operator*(const Vec3& a, float b) {
    return {a.x * b, a.y * b, a.z * b};
}

__device__ inline Vec3 operator/(const Vec3& a, float b) {
    return {a.x / b, a.y / b, a.z / b};
}




// Simulation routine
void init_simulation(Vec3 * pos, int num_parts);
void simulate_one_step_naive(Vec3 * pos, int num_parts);
void stepSimulationCoherentGrid(Vec3 * pos, int num_parts);
void stepSimulationScatteredGrid(Vec3 * pos, int num_parts);
void clear_simulation();

#endif
