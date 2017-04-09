// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <math.h>
#include <string.h>
#include <climits>

using namespace cv;
using namespace std;

static void help()
{
    cout <<
    "Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}


RNG rng(12345);
const char* wndname = "Lego Panel Detection";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findLargestContour( const Mat& image, vector<Point> &maxContour )
{
    Mat pyr, timg, gray0(image.size(), CV_8U), gray;
    Mat channel[3];

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    split(image, channel);
    gray0 = max(channel[1] - (channel[0] + channel[2])/2,0);
    //blur( gray0, gray0, Size(3,3) );

    inRange(gray0, 15, 255, gray);
    imshow("Gray", gray);

    // find contours and store them all as a list
    findContours(gray, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);


    vector<Point> approx;

    int maxArea = 0;
    // test each contour
    for( size_t i = 0; i < contours.size(); i++ )
    {
        // approximate contour with accuracy proportional
        // to the contour perimeter
        approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);
        if(fabs(contourArea(Mat(approx))) > maxArea)
        {
            maxArea = contourArea(Mat(approx));
            maxContour = Mat(approx);
        }
    }
}


// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, CV_AA);
    }

    imshow(wndname, image);
}

static void getAnchors(vector<Point> points, vector<Point2f> &anchors) {
    anchors.clear();

    Point tl(0,0);
    Point br(0,0);

    int minarea = INT_MAX;
    int maxarea = INT_MIN;
    for( int i = 0; i < points.size(); i++) {
        int area = points[i].x * points[i].y;
        if(area > maxarea) {
            maxarea = area;
            tl = points[i];
        }
        if(area < minarea) {
            minarea = area;
            br = points[i];
        }
    }
    int avgx = (tl.x+br.x)/2;
    int avgy = (tl.y+br.y)/2;

    Point tr(avgx,avgy);
    Point bl(avgx,avgy);
    int maxareatr = INT_MIN;
    int maxareabl = INT_MIN;
    for( int i = 0; i < points.size(); i++) {
        int area = abs(points[i].x-avgx) * abs(points[i].y-avgy);
        if(points[i].x < avgx && points[i].y > avgy && area > maxareabl) {
            maxareabl = area;
            bl = points[i];
        }
        if(points[i].x > avgx && points[i].y < avgy && area > maxareatr) {
            maxareatr = area;
            tr = points[i];
        }
    }
    anchors.push_back(tl);
    anchors.push_back(tr);
    anchors.push_back(br);
    anchors.push_back(bl);
}

int main(int argc, char** argv)
{
    help();
    namedWindow( wndname, 1 );

    for( int i = 1; i < argc; i++ )
    {
        vector<Point> object;
        vector<Point2f> src_anchors;
        vector<Point2f> dst_anchors = {Point(500,1000), Point(500,0), Point(0,0), Point(0,1000)};

        Mat image = imread(argv[i], 1);
        Mat out_image = Mat::zeros( Size(500,1000), image.type() );
        if( image.empty() )
        {
            cout << "Couldn't load " << argv[i] << endl;
            continue;
        }

        resize(image, image, Size((1000.0/image.rows)*image.cols,1000));
        findLargestContour(image, object);
        getAnchors(object, src_anchors);

//        cout << src_anchors << endl;
//        cout << dst_anchors << endl;

        Mat transform = findHomography(src_anchors,dst_anchors);
//        Mat transform = getPerspectiveTransform(src, dst);
        warpPerspective(image, out_image, transform, out_image.size());

        const Point* p = &object[0];
        int n = (int)object.size();
        polylines(image, &p, &n, 1, true, Scalar(0,0,255), 3, CV_AA);

        for(int i = 0; i < src_anchors.size(); i++) {
            circle(image, src_anchors[i], 5, Scalar(255,0,0), 3);
        }

        imshow(wndname,image);
        imshow("Result",out_image);

        int c = waitKey();
        if( (char)c == 27 )
            break;
    }

    return 0;
}
