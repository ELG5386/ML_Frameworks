
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "CONV_NT.h"

using namespace cv;
using namespace std;

CONV_NT::CONV_NT(){

    Kernel_Initializer();
};
void F_print2(float* x, int R, int C){
    for (int i=0;i<R;i++)
    {
        for (int j=0;j<C;j++){
            cout<<*(x+i*C+j)<<",";
        }
        cout<<endl;
    }
}


void CONV_NT::Conv2D(const float* input, float* output, const float* kernel,
                     int input_R,int input_C, int kernel_R, int kernel_C){

    int	kCenterX = kernel_C / 2;
    int kCenterY = kernel_R / 2;

    memset(output,0,input_R*input_C*sizeof(float));
    for(int i=0; i < input_R; i++)
    {
        for(int j=0; j < input_C; j++)
        {

            for(int m=0; m < kernel_R; m++)
            {
                int mm = kernel_R - 1 - m;

                for(int n=0; n < kernel_C; n++)
                {
                    int nn = kernel_C - 1 - n;
                    int ii = i + (m - kCenterY);
                    int jj = j + (n - kCenterX);

                    if( ii >= 0 && ii < input_R && jj >= 0 && jj < input_C )
                        output[i*input_R+j] +=  input[ii*input_R+jj] * kernel[mm*kernel_R+nn];
                }
            }
        }
    }

}
void CONV_NT::Kernel_Initializer(){

    float KERNEL_0f[9]={-1.f, -1.f, -1.f, 2.f, 2.f, 2.f, -1.f , -1.f , -1.f};
    float KERNEL_45f[9]={-1, -1, 2, -1, 2, -1, 2 , -1 , -1};
    float KERNEL_90f[9]={-1, 2, -1, -1, 2, -1, -1 , 2 , -1};
    float KERNEL_135f[9]={2, -1, -1, -1, 2, -1, -1 , -1 , 2};
    float KERNEL_sobelfy[9]={-1, 0, 1, -2, 0, 2, -1 , 0 , 1};
    float KERNEL_sobelfx[9]={-1, -2, -1, 0, 0, 0, 1 , 2 , 1};
    for (int i=0;i<9;i++){
        *(KERNEL_0+i)=*(KERNEL_0f+i);
        *(KERNEL_45+i)=*(KERNEL_45f+i);
        *(KERNEL_90+i)=*(KERNEL_90f+i);
        *(KERNEL_135+i)=*(KERNEL_135f+i);
        *(KERNEL_sobely+i)=*(KERNEL_sobelfy+i);
        *(KERNEL_sobelx+i)=*(KERNEL_sobelfx+i);
    }

    KERNELS.push_back(KERNEL_0);
    KERNELS.push_back(KERNEL_45);
    KERNELS.push_back(KERNEL_90);
    KERNELS.push_back(KERNEL_135);
   // KERNELS.push_back(KERNEL_sobelx);
   // KERNELS.push_back(KERNEL_sobely);
}

void CONV_NT::Down_Sampling(float* input,float* output, string method, int factor, int input_R,int input_C){

    memset(output,0,input_R/factor*input_C/factor*sizeof(float));
    for(int i=0;i<input_R/factor;i++){
        int step_i=i*factor*input_C;
        for(int j=0;j<input_C/factor;j++){
            int step_j=j*factor;

            for(int m=0;m<factor;m++){
                int step_m = step_j+m;
                for(int n=0;n<factor;n++){
                    output[i*input_C/factor+j] += input[step_i+ n*input_C + step_m] ;
                }
            }
        }
    }
}
