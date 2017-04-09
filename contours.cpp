/**
 * @function findContours_Demo.cpp
 * @brief Demo code to find contours in an image
 * @author OpenCV team
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src;
Mat channel[3];
Mat src_gray;
RNG rng(12345);
const int steps = 5;

/// Function header
void thresh_callback(int, void* );

/**
 * @function main
 */
int main( int, char** argv )
{
    /// Load source image and convert it to gray
    src = imread( argv[1], 1 );

    /// Convert image to gray and blur it
    resize(src, src, Size((1000.0/src.rows)*src.cols,1000));
    split(src, channel);
    src_gray = 255 - (channel[1] - (channel[0] + channel[2])/2);
    blur( src_gray, src_gray, Size(3,3) );

    /// Create Window
    const char* source_window = "Source";
    namedWindow( source_window, WINDOW_AUTOSIZE );
    imshow( source_window, src_gray );

    Mat canny_output;
    Mat thresh_output;
    vector<vector<Point> > canny_contours;
    vector<vector<Point> > thresh_contours;
    vector<Vec4i> canny_hierarchy;
    vector<Vec4i> thresh_hierarchy;

    for(int l = 0; l < steps; l++) {
        /// Detect edges using canny
//        Canny( src_gray, canny_output, (255/steps)*l, (255/steps)*(l+1), 3 );
        inRange(src_gray, 0, Scalar((255/steps)*(l+1)), thresh_output);
        namedWindow( "Thresh", WINDOW_AUTOSIZE );
        imshow( "Thresh", thresh_output );

        /// Find contours
        //findContours( canny_output, canny_contours, canny_hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        findContours( thresh_output, thresh_contours, thresh_hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

        /// Draw contours
  		//Mat canny_drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
        //for( size_t i = 0; i< canny_contours.size(); i++ )
        //{
        //    if(fabs(contourArea(Mat(canny_contours[i]))) > 1000) {
        //        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        //        drawContours( canny_drawing, canny_contours, (int)i, color, 2, 8, canny_hierarchy, 0, Point() );
        //    }
        //}
        //namedWindow( "Canny", WINDOW_AUTOSIZE );
        //imshow( "Canny", canny_output );

        //namedWindow( "Canny Contours", WINDOW_AUTOSIZE );
        //imshow( "Canny Contours", canny_drawing );

  		Mat thresh_drawing = Mat::zeros( thresh_output.size(), CV_8UC3 );
        for( size_t i = 0; i< thresh_contours.size(); i++ )
        {
            if(fabs(contourArea(Mat(thresh_contours[i]))) > 1000) {
                Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                drawContours( thresh_drawing, thresh_contours, (int)i, color, 2, 8, thresh_hierarchy, 0, Point() );
            }
        }

        namedWindow( "Thresh Contours", WINDOW_AUTOSIZE );
        imshow( "Thresh Contours", thresh_drawing );

        waitKey(0);
    }
    return(0);
}
