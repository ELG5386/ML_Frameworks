#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <windows.h>
#include "nn_mlp.h"
#include <time.h>
#include "CONV_NT.h"


using namespace std;
using namespace cv;



vector<float> readFromImage(std::string inputFilePath,int rows,int cols)
{

    Mat image = imread(inputFilePath,0);

    vector<float> output(rows*cols,0);
    int width = image.cols;
    int height = image.rows;

    unsigned y_ind=0;
    for(unsigned r=0; r < height- height/rows/2 ; r = r + height/rows , y_ind++)
    {
        int y = r + height/rows/2;

        unsigned x_ind=0;
        for(unsigned c=0; c < width - width/cols/2 ; c = c + width/cols , x_ind++)
        {
            int x = c + width/cols/2;
            int pixel = (int)image.at<uchar>(y,x);
            if(pixel==255)
                continue;
            output[y_ind*rows+x_ind]=1;
        }
    }

    return output;
}



void Conv_System(const vector<float> input, vector<float> &output, int input_R, int input_C, int factor)
{
    CONV_NT* cc=new CONV_NT();

    vector<float> input_b = input;
    float* output1=(float*)malloc(input_R*input_C * sizeof(float));
    float* output2=(float*)malloc((input_R*input_C)/(factor*factor) * sizeof(float));

    for (int i=0;i<cc->KERNELS.size();i++)
    {
        cc->Conv2D((float*)input_b.data(),output1,cc->KERNELS[i],input_R,input_C,3,3);


        cc->Down_Sampling(output1,output2,"AVERAGE",factor,input_R,input_C);


        for(int t=0;t<((input_R*input_C)/(factor*factor));t++)
            output.push_back(0.5f + output2[t]/(float)(factor*factor)*0.5f);

    }

    free(output1);
    free(output2);

}
int main(int argc, char *argv[])
{

    NN_MLP mlp;

    srand(time(NULL));
    int _n[]={64,10,5,5};
    vector<int> Net(_n,_n+3);

    mlp.create(Net,NN_MLP::SYGMOID,1,1);

    string filepath="C:\\Users\\ASUS\\Database.txt";

    vector2D<float> patterns;
    vector2D<float> desireds;

    mlp.readFromFile(filepath,patterns,desireds);



// These methods are used to load patterns from file or image
//    vector<float> p=readFromImage("1_16_16.png",16,16);
//    mlp.readFromFile(filepath,patterns,desireds);



    ML_TrainParams train_param(ml_TerminationCriteria(ML_TERMCRIT_ITER+ML_TERMCRIT_EPS,
                                                      20000,
                                                      .5),
                               ML_TrainParams::BACKPROP,
                               ML_TrainParams::HARD,
                               0.5,
                               1);


    int iter=mlp.train(patterns,desireds,train_param);    

    vector<float> output;
    mlp.predict(patterns[5],output);


    return 1;
}

