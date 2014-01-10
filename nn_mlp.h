#ifndef NN_MLP_H
#define NN_MLP_H
#include "ml_kernel.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include "pso.h"
using namespace std;

#define n(l) net_size[l]

ML_TrainParams::ML_TrainParams()
{

    /// Default training parameters
    term_crit = ml_TerminationCriteria( ML_TERMCRIT_ITER + ML_TERMCRIT_EPS , 1000, 0.01 );
    train_method = BACKPROP;
    weight_init_method = NGUYEN;
    bp_etha = bp_moment_scale = 0.1;
    rp_dw0 = 0.1; rp_dw_plus = 1.2; rp_dw_minus = 0.5;    
    rp_dw_min = EPSILON; rp_dw_max = 50.;    
}

ML_TrainParams::ML_TrainParams( ML_TerminationCriteria _term_crit, int _train_method, int _weight_init_method,
                                float _param1, float _param2 )
{
    term_crit = _term_crit;
    train_method = _train_method;
    weight_init_method = _weight_init_method;
    bp_etha = bp_moment_scale = 0.1;
    rp_dw0 = 1.; rp_dw_plus = 1.2; rp_dw_minus = 0.5;
    rp_dw_min = EPSILON; rp_dw_max = 50.;

    if( train_method == RPROP )
    {
        rp_dw0 = _param1;
        if( rp_dw0 < EPSILON )
            rp_dw0 = 1.;
        rp_dw_min = _param2;
        rp_dw_min = MAX( rp_dw_min, 0 );
    }
    else if( train_method == BACKPROP )
    {
        bp_etha = _param1;
        if( bp_etha <= 0 )
            bp_etha = 0.1;
        bp_etha = MAX( bp_etha, 1e-3 );
        bp_etha = MIN( bp_etha, 1 );
        bp_moment_scale = _param2;
        if( bp_moment_scale < 0 )
            bp_moment_scale = 0.1;
        bp_moment_scale = MIN( bp_moment_scale, 1 );
    }
    else
        train_method = BACKPROP;
}



class NN_MLP : public NN_Kernel
{
public:
    NN_MLP(){

    }

    NN_MLP( const vector<int> &_net_size,int _activation_func=SYGMOID,float _param1=0,float _param2=0,
            const vector2D<float> &initial_weights=vector2D<float>()){

        activ_func=_activation_func;
        this->create(_net_size,_activation_func,_param1,_param2,initial_weights);

    }

    virtual ~NN_MLP()
    {
        free(inputs_Net);
        free(outputs_Net);
        free(net_weights);
        free(err_signal);        
    }


    /// virtual functions
    void create(const vector<int> &_net_size,int activateFunc=SYGMOID, float _param1=0,float _param2=0,const vector2D<float> &initial_weights=vector2D<float>())
    {        
        if( _net_size.size() < 2 )
        {
            printf("ERR: Number of layers should be at least more than two layers. \n");
            return;
        }
        L = _net_size.size();
        net_size = _net_size;

        input_size=output_size=SUM<int>(0,L,net_size.data());
        err_size = output_size - n(0);
        weights_size= get_size_W();

        inputs_Net  = (float*)malloc(sizeof(float)*input_size);
        outputs_Net = (float*)malloc(sizeof(float)*output_size);
        err_signal  = (float*)malloc(sizeof(float)*err_size);
        net_weights = (float*)malloc(sizeof(float)*weights_size);



        /// Number of neurons in input, hidden and output layers must be less than MAX numbers

        for( unsigned l=0 ; l < L ; l++)
        {
            int num_neurons=n(l);
            if(num_neurons==0)
            {
                printf("ERR: Each layer must have at least one neuron. \n");
                return;
            }
            if(l==0 || l==L-1)
                continue;
            n(l) = MIN( num_neurons, MAX_NEURONS );
        }



        n(0) = MIN(n(0),MAX_INPUTS);
        n(L-1) = MIN(n(L-1),MAX_OUTPUTS);

        weight_indices.push_back(0); //l=0
        unsigned index=0;
        for(unsigned l=1 ; l < L ; l++)
        {
            index += get_size_W_l(l-1);
            weight_indices.push_back((float)index);
        }

        net_indices.push_back(0);
        for(unsigned i=1 ;i < L;i++)
        {
            unsigned index= SUM(0,i,net_size.data());
            net_indices.push_back(index);
        }        

#define w(i,j,l) net_weights[ weight_indices[l] + (n(l-1)+1)*j + i ]  // i:from   j:to
#define w_t(i,j,l) net_weights[ weight_indices[l+1] + (n(l-1)+1)*j + i ]

        param1=_param1;
        param2=_param2;

        init_activ_func(activateFunc,param1,param2);

    }

    bool readFromFile(std::string& inputFilePath, vector2D<float> &_patterns, vector2D<float> &_desired)
    {
        // read data from from file

        ifstream infile(inputFilePath.c_str());
        if (!infile.is_open())
        {
            cout << "Error opening input points file: " << inputFilePath << endl;
            return false;
        }
        cout<<inputFilePath.c_str()<<endl;


        vector<float> p(net_size[0],0);
        vector<float> d(net_size[L-1],0);

        bool p_d=true;
        int cnt=0;
        string s;
        bool ex=false;
        while (getline(infile,s))          // loop while extraction from file is possible
        {

            while(s.find(",")!=string::npos){
                if(atoi(s.substr(0,s.find(",")).c_str())<net_size[0])
                {
                    p[atoi(s.substr(0,s.find(",")).c_str())]=1;
                    s=s.substr(s.find(",")+1,s.length());
                }
                else{
                    ex=true;
                    break;
                }


            }
            if(!ex)
            {
                p[atoi(s.substr(0,s.find(";")).c_str())]=1;
                _patterns.push_back(p);
                string chl=(s.substr(s.find(";")+1,s.length()));
                d[(atoi((s.substr(s.find(";")+1,s.length())).c_str()))]=1;
                _desired.push_back(d);
            }

            p=vector<float>(net_size[0],0);
            d=vector<float>(net_size[L-1],0);           

        }
        infile.close();

        return true;
    }   

    int train(const vector2D<float> &patterns, const vector2D<float> &desired,
              ML_TrainParams &_params, int flags=0)
    {                        
        int max_iter;
        float epsilon;
        int iter = -1;

        if(net_size.empty())
        {
            printf("ERR: The Network must be first created. \n");
            return iter;
        }

        if( patterns.size() != desired.size() )
        {
            printf("ERR: Input and output training sets must be of the same size \n");
            return iter;
        }



        inputs_pattern  = patterns;

        outputs_desired = desired;

        train_params = _params;

        /// TO DO: take care of flag

        max_iter = _params.term_crit.type & ML_TERMCRIT_ITER ? _params.term_crit.max_iter : MAX_ITER;
        max_iter = MIN( max_iter, MAX_ITER );
        max_iter = MAX( max_iter, 1 );

        epsilon = _params.term_crit.type & ML_TERMCRIT_EPS ? _params.term_crit.epsilon : EPSILON;
        epsilon = MAX(epsilon, EPSILON);

        train_params.term_crit.type = ML_TERMCRIT_ITER + ML_TERMCRIT_EPS;
        train_params.term_crit.max_iter = max_iter;
        train_params.term_crit.epsilon = epsilon;

        init_method = _params.weight_init_method;


        if( _params.train_method == ML_TrainParams::BACKPROP )
        {            
            iter = train_backprop();           
        }

    }


    void predict(const vector<float> &inputs, vector<float> &outputs)
    {        
        if(inputs.size() != n(0))
        {
            printf("ERR: Input size mismatch");
            return;
        }

        memcpy(inputs_Net,inputs.data(),sizeof(float)*inputs.size());

        feed_forward();

        outputs.resize(n(L-1));
        for(unsigned i=0;i < n(L-1) ; i++)
        {
            float* out= get_output_l(L-1);
            outputs[i]=out[i];
        }        
    }


    vector2D<float> getNetWeights()
    {
        vector2D<float> temp;
        for(unsigned l=1;l<L;l++)
        {
            temp.push_back(vector<float>());
            for(unsigned j=0; j < n(l) ; j++)
                for(unsigned i=0; i < n(l-1)+1 ; i++)
                    temp[l-1].push_back(w(i,j,l));
        }
        return temp;
    }

    void update_weights()
    {
        for(unsigned l=1 ; l < L ; l++)
        {
            float* out_l = get_output_l(l-1);

            float norm_sqr = 0.0;
            for(unsigned j=0 ; j < n(l-1) ; j++)
                norm_sqr += out_l[j]*out_l[j];
            norm_sqr = MAX(norm_sqr,0.1);

            int index1=net_indices[l];
            for(unsigned j=0 ; j < n(l) ; j++)
            {
                float err_s=err_signal[index1-n(0)+j];

                for(unsigned i=0 ; i < n(l-1) ; i++)
                {
                    float x= out_l[i];
                    w(i,j,l) += train_params.bp_etha * x * err_s / norm_sqr; /** f_deriv*/;     


                }
                w(n(l-1),j,l)+= train_params.bp_etha * -1 * err_s / norm_sqr; /** f_deriv*/;
            }

        }
    }


    void calcLayerOutput(unsigned layer)
    {
        int index = net_indices[layer];
        for(unsigned i=0;i < n(layer) ;i++)
        {
            float sum = inputs_Net[index+i];
            outputs_Net[index+i] = layer==0 ? sum : calc_activ_func(sum);         
        }        
    }

    void calcLayerInput(unsigned layer)
    {
        if(layer==0)
            return;

        unsigned index=net_indices[layer];

        int I = n(layer-1);

        float* out_l = get_output_l(layer-1);

        for(unsigned j=0 ; j < n(layer) ; j++)
        {
            float input= -1.f * w(I,j,layer);

            for (unsigned i=0; i < I ;i++)                            
                input += w(i,j,layer)* out_l[i];

            inputs_Net[index + j]= input;
        }
    }

    void set_Initial_weights();

    // Get number of weights between layer l and l-1
    int get_size_W_l(unsigned l)
    {
        if(l==0)
            return 0;
        return (n(l-1)+1)*n(l);
    }

    // Get number of the whole network's weights
    int get_size_W()
    {
        int sum=0;
        for(unsigned l=1 ; l < L ; l++)
            sum += get_size_W_l(l);
        return sum;
    }



protected:

    void initialize_Weights(int method)
    {
        if(!sample_weights.empty())
        {

            /// TO DO
            return;
        }

        switch(method)
        {
        case ML_TrainParams::HARD :
            for(unsigned l=1 ; l < L ; l++)
                for(unsigned j=0 ; j < n(l) ; j++)
                    for(unsigned i=0 ; i < n(l-1)+1 ; i++)
                        w(i,j,l)=get_rand(-1.,1.);

            break;

        case ML_TrainParams::RANDOM :
            /// TO DO
        case ML_TrainParams::NGUYEN :

            float beta = 0.7 * pow((float)SUM(1,L-1,net_size.data()),(1.f/(float)n(0)));

            for(unsigned l=1 ; l < L ; l++)
            {                
                for(unsigned j=0 ; j < n(l) ; j++)
                {
                    float norm=0.f;
                    float w_w;
                    for(unsigned i=0 ; i < n(l-1)+1 ; i++)
                    {                        
                        w(i,j,l) = get_rand(-1.,1.);
                        w_w = w(i,j,l);
                        w_w *=w_w;
                        norm += w_w;
                    }
                    norm = sqrt(norm);
                    for(unsigned i=0 ; i < n(l-1)+1 ; i++)
                        w(i,j,l) *= (beta/norm);
                }
            }

            break;
        }
    }


    void init_activ_func(int _activ_type,float _param1,float _param2)
    {
        if( _activ_type < 0 || _activ_type > GAUSSIAN )
            printf("ERR: Unknown activation function" );

        activ_func = _activ_type;

        switch( activ_func )
        {
        case SYGMOID:

            if( fabs(_param1) < EPSILON )
                _param1 = 2./3;
            if( fabs(_param2) < EPSILON )
                _param2 = 1.7159;
            break;

        case GAUSSIAN:
            if( fabs(_param1) < EPSILON )
                _param1 = 1.;
            if( fabs(_param2) < EPSILON )
                _param2 = 1.;
            break;

        default:
            _param1 = 1.;
            _param2 = 0.;
        }

        param1=_param1;
        param2=_param2;
    }

    float calc_activ_func(float sum)
    {
        float result =0.f;

        switch (activ_func)
        {
        case LINEAR :
            break;
        case GAUSSIAN:
            break;
        case SYGMOID :
            float e = exp(-1.f*sum);
            result = 1./(1.f+e);
            break;
        }
        return result;
    }

    float calc_active_deriv(float sum)
    {
        float result =0.f;

        switch (activ_func)
        {
        case LINEAR :
            break;
        case GAUSSIAN:
            break;
        case SYGMOID :            
            float e = exp(-1.f*param1*(float)sum);

            result = 1./(1.f+e);

            result *= (1.-result);            
            break;
        }
        return result;
    }

    float* get_input_l(unsigned layer)
    {
        int index=net_indices[layer];
        return &inputs_Net[index];
    }

    float* get_output_l(unsigned layer)
    {
        int index=net_indices[layer];
        return &outputs_Net[index];
    }

    float* get_weight_l(unsigned layer)
    {
        if (layer==0)
            return NULL;

        int size_w_l = get_size_W_l(layer);
        return &net_weights[ layer*size_w_l];

    }

    int train_backprop()
    {
        int iter=0;        
        float Error=0;
        Error_total = 0.;      

        desired=outputs_desired;

        initialize_Weights(init_method);


        for( iter ; iter < train_params.term_crit.max_iter ; iter++)
        {            

            float last_Error =0;// Error;
            float err_p=0;

            bool termin = 1;
            for(unsigned p = 0 ; p < inputs_pattern.size() ; p++)
            {

                memcpy(inputs_Net,inputs_pattern[p].data(),sizeof(float)*inputs_pattern[p].size());

                feed_forward();                    

                termin &= compute_err_signal(desired[p]);

                Error_total += err_p;
                Error_vec.push_back(err_p);
                last_Error = Error;
                Error = err_p;

                update_weights();


            }
            if(termin)
                break;
        }

        return iter;
    }

    void feed_forward()
    {
        for(unsigned l=0 ; l<L ; l++)
        {
            calcLayerInput(l);
            calcLayerOutput(l);
        }
    }

    bool compute_err_signal(const vector<float> &desired)
    {
        float err_p=0.;


        bool termin = 1;
        for(unsigned l=L-1 ; l > 0 ; l--)
        {            
            int index=net_indices[l]-n(0);
            int last_index;         
            float* out = get_output_l(l);            
            float* in  = get_input_l(l);
            for(unsigned n=0; n < n(l) ; n++)
            {

                float deriv=calc_active_deriv(in[n]);

                if(l==L-1)               
                {                    
                    float err = desired[n] - out[n];

                    err_signal[index+n] = err /** deriv*/;                 

                    if(fabs(err)>train_params.term_crit.epsilon)
                        termin=0;
                    err_p += fabs(err);
                }
                else
                {

                    float sum = 0.f;
                    for(unsigned nn=0; nn < n(l+1) ; nn++)                    
                        sum += err_signal[last_index+nn] * w_t(n,nn,l);                    

                    err_signal[index+n] = sum*deriv;

                }
            }
            last_index=index;
        }        
        return termin;///2.f;
    }



    vector<int> net_size;
    vector<int> net_indices;
    vector<int> weight_indices;

    float* outputs_Net;                          // Store each layer's output
    float* inputs_Net;                          // Store each layer's input    
    float* net_weights;                     // Matrix of all weight (3 dimensions)
    float* err_signal;

    unsigned L;                       // Number of layers
    unsigned output_size;             // Number of all outputs in the net
    unsigned input_size;             // Number of all inputs in the net
    unsigned err_size;
    unsigned weights_size;            // Number of all weights in the net  

    vector2D<float> sample_weights;
    vector2D<float> inputs_pattern;
    vector2D<float> outputs_desired;
    vector<vector<float> > desired;
    float param1, param2;
    float Error_total;
    vector<float> Error_vec;

    int activ_func;
    int init_method;
    ML_TrainParams train_params;    
};

#endif // NN_MLP_H

