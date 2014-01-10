#ifndef ML_KERNEL_H
#define ML_KERNEL_H

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <stdint.h>
#include <math.h>


using namespace std;

template<typename T>
struct vector2D : public std::vector< std::vector<T> > {};

#define FLT_MAX numeric_limits<float>::max()

#define MAX_LAYERS 4
#define MAX_INPUTS 256
#define MAX_NEURONS 1000
#define MAX_OUTPUTS 20

#define MAX(a,b) a<b ? b : a
#define MIN(a,b) a>b ? b : a

const int MAX_ITER = 20000;
const float EPSILON = 1e-5;

/// Termination Criteria

#define ML_TERMCRIT_ITER    1
#define ML_TERMCRIT_EPS     2

typedef struct ML_TerminationCriteria
{
    int    type;
    int    max_iter;
    double epsilon;    

}ML_TerminationCriteria;


inline ML_TerminationCriteria ml_TerminationCriteria( int _type, int _max_iter, float _epsilon )
{    
    ML_TerminationCriteria mlt;
    mlt.type = _type;
    mlt.max_iter = _max_iter;
    mlt.epsilon =_epsilon;
    return mlt;
}
/// Training parameters
struct ML_TrainParams{

    ML_TrainParams();
    ML_TrainParams( ML_TerminationCriteria term_criteria, int train_method, int _weight_init_method,
                    float param1=0, float param2=0 );

    enum { BACKPROP=0, RPROP=1 };
    enum { HARD=0, RANDOM=1, NGUYEN=2};


    ML_TerminationCriteria term_crit;
    int train_method;
    int weight_init_method;

    // backpropagation parameters
    float bp_etha, bp_moment_scale;
};

class NN_Kernel
{
public:

    float error;
    enum {LINEAR=0,SYGMOID=1,GAUSSIAN=2};
    NN_Kernel(){}

    virtual ~NN_Kernel(){}

    virtual void create(const vector<int> &netSizes,int activateFunc,float param1, float param2,
                        const vector2D<float> &initial_weights) =0;

    virtual int train(const vector2D<float> &inputs, const vector2D<float> &outputs,
                      ML_TrainParams &params, int flags=0 ) =0;

    virtual void update_weights(void) =0;

    virtual void predict(const vector<float> &inputs, vector<float> &outputs)=0;

    virtual vector2D<float> getNetWeights(void) =0;

    virtual void calcLayerOutput(unsigned layer) =0;

    virtual void calcLayerInput(unsigned layer) =0;

    virtual float getError(void){return error;}

    template<typename T>
    T SUM(unsigned first,unsigned last,T* f){
        T t =f[first];
        for(unsigned i=first+1 ; i < last ; i++)
        {
            t = t + f[i];
        } return t;
    }

    template<typename F>
    float SUM(unsigned first,unsigned last,F* f,bool isfunc){
        if(!isfunc)
            return -1;
        float t =f(first);
        for(unsigned i=first+1 ; i < last ; i++)
        {
            t = t + f(i);
        } return t;
    }

    static float get_rand(float a,float b)
    {
        return (fabs(b-a)*(float)(rand()-((float)RAND_MAX/2.f))/((float)RAND_MAX));
    }
};

#endif // ML_KERNEL_H
