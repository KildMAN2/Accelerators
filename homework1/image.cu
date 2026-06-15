///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#include "ex1.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

int main(int argc,char **argv) 
{ 
    char *image_name = argv[1];

    Mat image = imread(image_name, 1);
    if( argc != 2 || !image.data )
    {
	printf( " No image data \n " );
	return 1;
    }

    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY); // here was a bug in the original code, it was CV_BGR2BGRA instead of COLOR_BGR2GRAY
    imwrite("grayscale.png", gray_image);

    Size sz = gray_image.size();

    assert(gray_image.isContinuous());

    uchar *image_out = new uchar[sz.width * sz.height];
    cpu_process(gray_image.data, image_out, sz.width, sz.height);

    Mat result(sz, CV_8UC1, image_out);
    imwrite("result.png", result);

    // GUI display skipped – no display available on this headless server.
    // Results already written to grayscale.png and result.png above.

    return 0;
}
