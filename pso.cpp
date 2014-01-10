#include "pso.h"

PSO::PSO(): c1(1.5f), c2(1.5f) , v_max(100) , x_max(100) ,v_min(-100) , x_min(-100)
{ 
}

PSO::PSO(int _n_particles, int _p_size, float _c1, float _c2, float _v_max,
         float _x_max, float _v_min, float _x_min)
{
    c1=_c1; c2=_c2;
    v_max = _v_max; v_min = _v_min;
    x_max = _x_max; x_min = _x_min;
    n_particles = _n_particles;
    p_size =_p_size;
}

void PSO::set_parameters(int _n_particles,int _p_size,float _c1,float _c2,
                         float _v_max, float _x_max, float _v_min, float _x_min)
{
    c1=_c1; c2=_c2;
    v_max = _v_max; v_min = _v_min;
    x_max = _x_max; x_min = _x_min;
    n_particles = _n_particles;
    p_size =_p_size;
}


void PSO::init_swarm()
{
    p_swarm = (particle*)malloc(n_particles*sizeof(particle));

    for(unsigned i=0 ; i < p_size  ; i++ )
    {
        p_swarm->init_particle(p_size);
        p_swarm->position[i]=NN_Kernel::get_rand(-1,1);
        p_swarm->velocity[i]=NN_Kernel::get_rand(-1,1);
    }
}

void PSO::update()
{
    for(unsigned i=0 ; i < n_particles  ; i++ )
    {
        for(unsigned j=0 ; j < p_size  ; j++ )
        {
            p_swarm[i].velocity[j] += c1*rand()*(p_swarm[i].best_positoin[j]-p_swarm[i].position[j])+
                                      c2*rand()*(p_swarm[i].best_cost-p_swarm[i].position[j]);
            p_swarm[i].position += p_swarm[i].velocity[j];
        }
    }
}

PSO::~PSO()
{
    for (unsigned i=0; i< n_particles ;i++)
    {
        free(p_swarm->position);
        free(p_swarm->velocity);
        free(p_swarm->best_positoin);
    }
    free(p_swarm);

}
