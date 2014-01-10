#ifndef CONV_NT_H
#define CONV_NT_H
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
using namespace cv;
using namespace std;
class CONV_NT{
public:
	CONV_NT();

        void Conv2D(const float* input, float* output, const float* kernel, int input_R,int input_C, int kernel_R, int kernel_C);
	void Down_Sampling(float* input,float* output, string method, int factor, int input_R,int input_C);	
	vector<float*> KERNELS;
private:
	void Kernel_Initializer();
	float KERNEL_0[9];
	float KERNEL_45[9];
	float KERNEL_90[9];
	float KERNEL_135[9];
        float KERNEL_sobelx[9];
        float KERNEL_sobely[9];
};

#endif // CONV_NT_H
