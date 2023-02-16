// Apply boundary following to segment a fish from a simple background

// refer to https://docs.opencv.org/4.1.0/da/d5c/tutorial_canny_detector.html

// Run it using White crappie.png
#include	<opencv2/opencv.hpp>
#include	<opencv2/highgui/highgui.hpp>
#include	<opencv2/imgproc.hpp>
#include	<iostream>
#include	"Supp.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	string	ss = "Inputs/Images/White crappie.png";
	Mat		srcI = imread(ss), tmp, cannyEdge;

	if (srcI.empty()) {
		cout << "cannot open " << ss << endl;
		return -1;
	}

	int const	noOfImagePerCol = 3, noOfImagePerRow = 3; // create a 3X3 window partition
	Mat			largeWin, win[noOfImagePerRow * noOfImagePerCol], // create the new window
		legend[noOfImagePerRow * noOfImagePerCol]; // and the means to each sub-window
	int			winI = 0;

	int			ratio = 3, kernelSize = 3; // set parameters for Canny
	Mat			B = (Mat_<unsigned char>(3, 3) << 1, 1, 1,  // define Bel 1
		1, 1, 1,
		1, 1, 1);

	createWindowPartition(srcI, largeWin, win, legend, noOfImagePerCol, noOfImagePerRow);

	srcI.copyTo(win[0]); // copy the input to the first subwindow
	putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1); // place text in the first legend window
	Canny(srcI, cannyEdge, 60, 60 * ratio, kernelSize);
	cvtColor(cannyEdge, win[1], COLOR_GRAY2BGR);
	putText(legend[1], "Canny edge", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	dilate(win[1], win[2], B); // Make edge thicker to fill gap/break
	putText(legend[2], "Dilate(edge)", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	erode(win[2], win[3], B, Point(-1, -1)); // Make edge thinner to get back the original edge as much as possible
	putText(legend[3], "Erode(previous)", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	/// Section 2: Find contours
	// The input parameters are the segmented image, the result as an array of contours, 
	// hierarachy of edges connected together, edge retrieval mode, line fitting method
	// We only need to pay attention to the first 2. Contours is an array (vector) of contours.
	// Each element of Contours is one contour. Each contour is an array (vector) of 2D points.
	// Note that the content of the first input is destroyed
	vector<vector<Point> >	contours;

	cvtColor(win[3], tmp, COLOR_BGR2GRAY);
	// findContours(tmp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE); // extract only the outer contour
	findContours(tmp, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE); // extract all contours

																   /// Section 3: Access each contour
	RNG				rng(12345);
	Scalar			color;
	int				index, max = 0;

	cvtColor(tmp, tmp, COLOR_GRAY2BGR);
	tmp = Scalar(0, 0, 0);

	static int	t1, t2, t3, t4;
	for (int i = 0; i < contours.size(); i++) { // Just in case there is more than one object in image
		for (;;) { // get random colors that are not too dim
			t1 = rng.uniform(0, 255); // blue
			t2 = rng.uniform(0, 255); // green
			t3 = rng.uniform(0, 255); // red
			t4 = t1 + t2 + t3;
			if (t4 > 255) break;
		}
		color = Scalar(t1, t2, t3);
		if (max < contours[i].size()) { // Find the longest contour as fish boundary
			max = contours[i].size();
			index = i;
		}
		// Draw contours using drawContours() which has the following input parameters
		// 1. image object to draw on, 2. an array of contours, 3. the index of contour to draw
		// 4. color, 5. ... not important. You can check them from the web site.
		//		drawContours(result, contours, i, color); // draw boundaries on original image with distinct colors
		drawContours(tmp, contours, i, color); // draw boundaries
	}
	tmp.copyTo(win[4]);
	putText(legend[4], "All contours", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	tmp = Scalar(0, 0, 0); // reset the canvas to black to draw only the longest contour
	drawContours(tmp, contours, index, Scalar(255, 255, 255));
	tmp.copyTo(win[5]);
	putText(legend[5], "Longest contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	vector<Point>	curContour = contours[index]; //access each contour point
	Point2i			p;
	int				count = 0;

	p.x = p.y = 0; // add all contour points to compute the average, i.e. the center of fish
	for (int j = 0; j < curContour.size(); j++) // Add all point coordinates
		p += curContour[j];
	p.x /= curContour.size(); // take average, i.e. center of fish
	p.y /= curContour.size();
	win[5].copyTo(win[6]);
	floodFill(win[6], p, Scalar(255, 255, 255)); // fill inside fish boundary
	putText(legend[6], "Mask of fish", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	win[7] = win[0] & win[6];
	putText(legend[7], "Fish segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	erode(win[6], tmp, B, Point(-1, -1)); // attempt to eliminate empty space around boundary from win[6]
	win[8] = win[0] & tmp;
	putText(legend[8], "Fish tidied up", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
	imshow("Boundary following application", largeWin);

	waitKey();
	destroyAllWindows();
	//	system("pause");
	return 0;
}