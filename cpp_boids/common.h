#ifndef __COMMON_H__
#define __COMMON_H__

// Program Constants
#define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01
#define min_r    (cutoff / 100)
#define dt       0.2

#define centering_factor 0.1
#define epulsion_factor 0.03
#define matching_factor 0.1
#define speed_limit 5.0
#define perception_radius 7.0
#define avoidance_radius 2.0



typedef struct vec3{
    double x;
    double y;
    double z;

    vec3(double x, double y, double z){
        this->x = x; 
        this->y = y;
        this->z = z;
    }

}vec3;




// Simulation routine
void init_simulation();
void simulate_one_step(float dt);
void stepSimulationCoherentGrid(float dt);
void stepSimulationScatteredGrid(float dt);
void clear_simulation();

#endif
