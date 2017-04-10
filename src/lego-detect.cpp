// Detects a green lego backplane,
//  then crops and warps the image to isolate the backplane.

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

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


const char* wndname = "Lego Panel Detection";

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

    /// Convert image to grayscale using formula green - avg(red,blue).
    ///  This isolates green and discounts yellow and cyan.
    split(image, channel);
    gray0 = max(channel[1] - (channel[0] + channel[2])/2,0);

    // Threshold out any lower values.
    inRange(gray0, 15, 255, gray);

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

        // Find largest contour
        if(fabs(contourArea(Mat(approx))) > maxArea)
        {
            maxArea = contourArea(Mat(approx));
            maxContour = Mat(approx);
        }
    }
}


static void getAnchors(vector<Point> points, vector<Point2f> &anchors) {
    anchors.clear();

    // Find top left and bottom right corner
    Point tl(0,0);
    Point br(0,0);

    int mindist = INT_MAX;
    int maxdist = INT_MIN;
    for( int i = 0; i < points.size(); i++) {
        int dist = norm(points[i]);
        // If farthest from origin, then top left.
        if(dist > maxdist) {
            maxdist = dist;
            tl = points[i];
        }
        // If closest to origin, then bottom right.
        if(dist < mindist) {
            mindist = dist;
            br = points[i];
        }
    }

    // Find midpoint by averaging top right and bottom left corners
    Point mid((tl.x+br.x)/2,(tl.y+br.y)/2);

    // Find top right and bottom left corner
    Point tr(mid.x,mid.y);
    Point bl(mid.x,mid.y);

    int maxareatr = INT_MIN;
    int maxareabl = INT_MIN;
    for( int i = 0; i < points.size(); i++) {
        // Calculate area between this point and midpoint.
        int area = abs(points[i].x-mid.x)*abs(points[i].y-mid.y);
        // If in bottom left quadrant, and has most area,
        //  Then it is the bottom left corner
        if(points[i].x < mid.x && points[i].y > mid.y && area > maxareabl) {
            maxareabl = area;
            bl = points[i];
        }
        // If in top right quadrant, and has most area,
        //  Then it is the top right corner
        if(points[i].x > mid.x && points[i].y < mid.y && area > maxareatr) {
            maxareatr = area;
            tr = points[i];
        }
    }
    anchors.push_back(br);
    anchors.push_back(bl);
    anchors.push_back(tl);
    anchors.push_back(tr);
}

int main(int argc, char** argv)
{
    help();
    namedWindow( wndname, 1 );

    for( int i = 1; i < argc; i++ )
    {
        vector<Point> object;
        vector<Point2f> src_anchors;
        vector<Point2f> dst_anchors;

        // Read image
        Mat image = imread(argv[i], 1);
        Mat out_image;
        if( image.empty() )
        {
            cout << "Couldn't load " << argv[i] << endl;
            continue;
        }

        // Detect LEGO backplane
        findLargestContour(image, object);
        getAnchors(object, src_anchors);

        // Determine width and height of result image.
        int width  = norm(src_anchors[0] - src_anchors[1]);
        int height = norm(src_anchors[1] - src_anchors[2]);
        out_image  = Mat::zeros( Size(height,width), image.type() );
        dst_anchors.push_back(Point(0,0));
        dst_anchors.push_back(Point(0,width));
        dst_anchors.push_back(Point(height,width));
        dst_anchors.push_back(Point(height,0));

        // Warp LEGO backplane into its own image and save.
        Mat transform = findHomography(src_anchors,dst_anchors);
        warpPerspective(image, out_image, transform, out_image.size());

        char filename[14];
        sprintf(filename,"result-%03d.jpg",i);
        imwrite(filename, out_image);


        // Draw debug markers on original image
        const Point* p = &object[0];
        int n = (int)object.size();
        polylines(image, &p, &n, 1, true, Scalar(0,0,255), 3, CV_AA);

        circle(image, src_anchors[0], 5, Scalar(0,255,0), 5);
        for(int i = 1; i < src_anchors.size(); i++) {
            circle(image, src_anchors[i], 5, Scalar(255,0,0), 5);
        }

        // Resize debug previews to reasonable size.
        resize(image, image, Size((1000.0/image.rows)*image.cols,1000));
        resize(out_image, out_image, Size((500.0/out_image.rows)*out_image.cols,500));

        // Show debug previews.
        imshow(wndname,image);
        imshow("Result",out_image);

        // Wait for key to advance.
        int c = waitKey();
        if( (char)c == 27 )
            break;
    }

    return 0;
}
