// Uses a normalized image of green LEGO backplane
//  then converts it to a text schedule.

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string.h>
#include <climits>

#define XRES 32
#define YRES 64

using namespace cv;
using namespace std;

enum Color {WHITE, RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE};

const Scalar color_map[] = {Scalar(255,255,255), Scalar(255,  0,  0),
                            Scalar(255,127,  0), Scalar(  0,255,  0),
                            Scalar(  0,  0,255), Scalar(  0,  0,255),
                            Scalar(255,  0,255)};

static void help()
{
    cout <<
    "Using OpenCV version " << CV_VERSION << endl << endl;
}

const char* wndname = "LEGO Schedule";

// returns closest color of given cell.
static Color getCellColor( const Mat& image, const Rect& cell) {
    Mat roiImg = image(cell);

    Mat hsv;
    cvtColor(roiImg, hsv, CV_BGR2HSV);

    // Nearest colors are red (0 deg) and orange(30 deg).
    //  Need 15 deg of accuracy. 360/15 = 24.
    int hbins=24;

    // Need saturation histogram to detect white.
    //  If most pixels have saturation less than 32 (256/8)
    //  then cell is white.
    int sbins=8;

    // Hue varies from 0 to 179 NOT 0 to 255.
    const float hrange0[] = {0, 180};
    const float srange0[] = {0, 256};

    const float* hrange = {hrange0};
    const float* srange = {srange0};

    const int hchan = 0;
    const int schan = 1;

    Mat h_hist, s_hist;
    calcHist(&hsv, 1, &hchan, Mat(), h_hist, 1, &hbins, &hrange);
    calcHist(&hsv, 1, &schan, Mat(), s_hist, 1, &sbins, &srange);

    Point huePt, satPt;
    minMaxLoc(h_hist, NULL, NULL, NULL, &huePt);
    minMaxLoc(s_hist, NULL, NULL, NULL, &satPt);

    int hue = huePt.y;
    int sat = satPt.y;

    cerr << s_hist << endl;
    cerr << 'H' << hue << 'S' << sat << ' ' << cell << endl;

    if(sat == 0 || sat == 1)
    {
        return WHITE;
    }
    else
    {
        if(hue == 22 || hue == 23 || hue == 0)
        {
            return RED;
        }
        else if(hue == 1 || hue == 2)
        {
            return ORANGE;
        }
        else if(hue >= 3 && hue <= 5)
        {
            return YELLOW;
        }
        else if(hue >= 6 && hue <= 11)
        {
            return GREEN;
        }
        else if(hue >= 12 && hue <= 17)
        {
            return BLUE;
        }
        else if(hue >= 18 && hue <= 21)
        {
            return PURPLE;
        }
    }
}

int main( int argc, char** argv) {
    help();
    namedWindow( wndname, 1 );

    for( int i = 1; i < argc; i++ )
    {
        vector<Point> object;
        vector<Point2f> src_anchors;
        vector<Point2f> dst_anchors;

        // Read image
        Mat out_image;
        Mat image = imread(argv[i], 1);
        if( image.empty() )
        {
            cout << "Couldn't load " << argv[i] << endl;
            continue;
        }
    
        cout << argv[i] << endl;
        Size cellSize = Size(image.size().width/XRES, image.size().height/YRES);
        // Generate grid
        for( int y = 0; y < YRES; y += 1)
        {
            for( int x = 0; x < XRES; x += 1)
            {
                Rect cell = Rect(Point(x*cellSize.width,y*cellSize.height), cellSize);
                Color color = getCellColor(image, cell);
                cout << color;
            }
            cout << endl;
        }
        cout << endl;
    }

    return 0;
}
