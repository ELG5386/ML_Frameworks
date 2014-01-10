#ifndef PSO_H
#define PSO_H
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <stdint.h>
#include <math.h>
#include "ml_kernel.h"

#define FLT_MAX numeric_limits<float>::max()

class PSO
{
public:
    PSO();
    PSO(int _n_particles, int _p_size,float _c1,float _c2,float _v_max,
        float _x_max, float _v_min, float _x_min);
    virtual ~PSO();

    struct particle{
        float* position;
        float* velocity;
        float* best_positoin;
        float best_cost;

        void init_particle(int _p_size){
            best_cost = FLT_MAX;
            position = (float*)malloc(_p_size*sizeof(float));
            velocity = (float*)malloc(_p_size*sizeof(float));
            best_positoin = (float*)malloc(_p_size*sizeof(float));
        }
    };

    particle* p_swarm;
    unsigned p_size;


    void set_parameters(int _n_particles,int _p_size,float _c1,float _c2,float _v_max,
                        float _x_max, float _v_min, float _x_min);

    void update();

    void init_swarm();

    particle* get_particles();

private:
    float v_max,v_min;
    float x_max,x_min;
    float c1, c2;
    unsigned n_particles;

};

#endif // PSO_H
