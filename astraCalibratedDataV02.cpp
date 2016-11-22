/// @file astraCalibratedDataV02.cpp
///
/// @brief Connect to Orbbec astra, get color and point streams,
/// align the point stream data
///
///
/// Uses linear interpolation to fit data from depths of ~550mm
/// upto 1720mm. FIlters out data past 2000mm. The general
/// alignment operation is a scaling operation in x and y, and
/// offset operation in x and y. x and y direction operations
/// are assumed to be independent of each other. Therefore
///
/// x_adjusted = kx*x + cx
/// y_adjusted = ky*y + cy
///
/// The parameter cx is a function of depth while the others
/// are constant with depth.
///
/// This is based on examples in the html documentation 0.4.0
/// and my tested code astraV07.cpp which included extraction
/// of depth-to-color mapping parameters
///
/// Depth and color frame size is 640 X 480 each
///
/// Information on streams:
/// ColorStream and DepthStream are low level streams, whereas
/// PointStream and HandStreams are high level streams.
///
/// Color Stream: RGB pixel data from the sensor. The data
/// array included in each ColorFrame contains values ranging
/// from 0-255 for each color component of each pixel.
///
/// DepthStream: Depth data from the sensor. The data array
/// included in each DepthFrame contains values in  millimeters
/// for each pixel within the sensor’s field of view.
///
/// PointStream: World coordinates (XYZ) computed from the
/// depth data. The data array included in each PointFrame is
/// an array of astra:Vector3f elements to more easily access
/// the x, y and z values for each pixel.
///
/// HandStream: Hand points computed from the depth data. On
/// each HandFrame, the number of points detected at any given
/// time can be retrieved through the HandFrame::handpoint_count
/// function, and a astra::HandPointList can be retrieved
/// through the HandFrame::handpoints function.
///
/// Created 02 Oct 2016
/// Modified 03 Oct 2016 (V02)
/// -fixed bug in 1D depth array
/// -struct for 3D depth data
/// -synthetic color function
/// -inv UV map created
///
/// @author Mustafa Ghazi
///

// for using printfs etc because MS has its own version
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#endif

#include <iterator> // because not included in Astra libraries by default
// Astra stuff
#include <Astra/Astra.h>
#include <AstraUL/AstraUL.h>
#include <cstdio>
#include <iostream>
// Open CV stuff
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// frame width and height
#define DWIDTH 640
#define DHEIGHT 480
// end program after this number of frames
#define NMAXFRAMES 18000
#define METADATA "20160930_2211"

// for point in 3D space
struct Point3D {
	float x;
	float y;
	float z;
};

// for inverse UV map
struct Map3D {
	int8_t u;
	int8_t v;
	int8_t isMapped;
};

Mat imgDepth = Mat::zeros(DHEIGHT, DWIDTH, CV_16UC1); // image with z-depth information
Mat imgDepthRawMirrored = Mat::zeros(DHEIGHT, DWIDTH, CV_16UC1); // mirror image of imageDepth

Mat imgDepthFakeColor = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3); // synthetic image with colors based on z-depth
Mat imgDepthMirrored = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3); // mirror image of imgDepthFakeColor
Mat imgDepthMirroredCorrected = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3); // corrected depth visualization

Mat imgColor = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3); // image with color data
Mat imgColorMirrored = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3); // mirror image of imgColor
Mat imgOverlay = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3); // color + depth overlay

float DEPTHZ[DWIDTH*DHEIGHT];
Point3D DEPTHRAW[DWIDTH*DHEIGHT];
Map3D UVMAPINV[DWIDTH*DHEIGHT];
char charCheckForKey = 0; // keyboard input
int DEPTHPT[2] = {0,0}; // read off depth from this point

/// @brief Return color intensity for depth value.
int lookupFunc (int depth);

/// @brief Update global flags and variables based on
/// global value that stores the last key stroke
void updateKeyCmds();

/// @brief Save color, depth, and overlay image view as a PNG image.
int saveImages();

static void onMouse(int event, int x, int y, int f, void*);


void depthCorrectionCustom (Mat src, Mat dst);

/// @brief Parameter cx for a given depth.
float depthCorrectionLookup (float d);

/// @brief Generate synthetic blue, green, red values based on
/// depth value.
void fakeDepthColor(float depthVal, int8_t *blue, int8_t *green, int8_t *red);


/// @brief Listens for and processes DepthPoint frames
///
/// PointStream: World coordinates (XYZ) computed from the
/// depth data. The data array included in each PointFrame is
/// an array of astra:Vector3f elements to more easily access
/// the x, y and z values for each pixel.
///
class DepthFrameListener : public astra::FrameReadyListener
{
public:
	DepthFrameListener(int maxFramesToProcess) :
		m_maxFramesToProcess(maxFramesToProcess)
	{

	}

	bool is_finished()
	{
		return m_isFinished;
	}

private:
	/// @brief Do this when frame is available on stream reader
	///
	/// @param reader the stream reader
	/// @param frame the frame being read from the stream
	///
	/// @return Void
	///
	virtual void on_frame_ready(astra::StreamReader& reader,
		astra::Frame& frame) override
	{
		astra::PointFrame depthFrame = frame.get<astra::PointFrame>();

		if (depthFrame.is_valid())
		{
			processDepthFrame(depthFrame);
			++m_framesProcessed;
		}

		if (m_framesProcessed >= m_maxFramesToProcess)
		{
			m_isFinished = true;
		}
	}
	/// @brief Copy depth frame for further processing
	///
	/// @param depthFrame the frame to process
	///
	/// @return Void
	///
	void processDepthFrame(astra::PointFrame& depthFrame)
	{
		int frameIndex = depthFrame.frameIndex();
		int nPixels = depthFrame.numberOfPixels();
		int xRes = depthFrame.resolutionX();
		int yRes = depthFrame.resolutionY();

		astra::Vector3f frameData = depthFrame.data()[153600]; // point frame
		const astra::Vector3f* allFrameData = depthFrame.data();

		imgDepth = Mat::zeros(DHEIGHT, DWIDTH, CV_16UC1);
		imgDepthFakeColor = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3);
		Vec3b intensity;
		int8_t blueVal, greenVal, redVal;
		intensity.val[0] = 0;
		intensity.val[1] = 0;
		intensity.val[2] = 0;
		int i;

		for(i=0;i<DWIDTH*DHEIGHT;i++) {
			imgDepth.at<int16_t>(i/DWIDTH,i%DWIDTH) = (int16_t)allFrameData[i].z;
			fakeDepthColor(allFrameData[i].z, &blueVal, &greenVal, &redVal);
			intensity.val[0] = blueVal; // B
			intensity.val[1] = greenVal; // G
			intensity.val[2] = redVal; // R

			//intensity.val[0] = (int8_t)lookupFunc((int16_t)allFrameData[i].z/3-600);	// B
			//intensity.val[1] = (int8_t)lookupFunc((int16_t)allFrameData[i].z/3-300);	// G
			//intensity.val[2] = (int8_t)lookupFunc((int16_t)allFrameData[i].z/3);	// R
			DEPTHZ[i] = allFrameData[i].z;
			DEPTHRAW[i].x = allFrameData[i].x;
			DEPTHRAW[i].y = allFrameData[i].y;
			DEPTHRAW[i].z = allFrameData[i].z;
			imgDepthFakeColor.at<Vec3b>(i/DWIDTH,i%DWIDTH) = intensity;
		}

	}


	bool m_isFinished{false};
	int m_framesProcessed{0};
	int m_maxFramesToProcess{0};
};

/// @brief Listens for and processes DepthPoint frames
///
/// Color Stream: RGB pixel data from the sensor. The data
/// array included in each ColorFrame contains values ranging
/// from 0-255 for each color component of each pixel.
///
class ColorFrameListener : public astra::FrameReadyListener
{
public:
	ColorFrameListener(int maxFramesToProcess) :
		m_maxFramesToProcess(maxFramesToProcess)
	{

	}

	bool is_finished()
	{
		return m_isFinished;
	}

private:
	/// @brief Do this when frame is available on stream reader
	///
	/// @param reader the stream reader
	/// @param frame the frame being read from the stream
	///
	/// @return Void
	///
	virtual void on_frame_ready(astra::StreamReader& reader,
		astra::Frame& frame) override
	{

		astra::ColorFrame colorFrame = frame.get<astra::ColorFrame>();

		if (colorFrame.is_valid())
		{
			processColorFrame(colorFrame);
			++m_framesProcessed;
		}

		if (m_framesProcessed >= m_maxFramesToProcess)
		{
			m_isFinished = true;
		}
	}
	/// @brief Copy color frame for further processing
	///
	/// @param colorFrame the frame to process
	///
	/// @return Void
	///
	void processColorFrame(astra::ColorFrame& colorFrame)
	{
		int frameIndex = colorFrame.frameIndex();
		int nPixels = colorFrame.numberOfPixels();
		int xRes = colorFrame.resolutionX();
		int yRes = colorFrame.resolutionY();

		astra::RGBPixel frameData = colorFrame.data()[153600];
		const astra::RGBPixel* allFrameData = colorFrame.data();

		imgColor = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3);
		Vec3b intensity;
		intensity.val[0] = 0;
		intensity.val[1] = 0;
		intensity.val[2] = 0;
		int i;

		for(i=0;i<DWIDTH*DHEIGHT;i++) {

			intensity.val[0] = allFrameData[i].b;	// B
			intensity.val[1] = allFrameData[i].g;	// G
			intensity.val[2] = allFrameData[i].r;	// R
			imgColor.at<Vec3b>(i/DWIDTH,i%DWIDTH) = intensity;
		}

	}

	bool m_isFinished{false};
	int m_framesProcessed{0};
	int m_maxFramesToProcess{0};
};


int main(int argc, char** arvg)
{
	namedWindow("depth image", CV_WINDOW_AUTOSIZE);
	namedWindow("color image", CV_WINDOW_AUTOSIZE);
	setMouseCallback("depth image", onMouse, NULL);

	astra::Astra::initialize();
	astra::StreamSet streamSet;
	astra::StreamReader reader = streamSet.create_reader();

	reader.stream<astra::PointStream>().start(); // x,y,z
	reader.stream<astra::ColorStream>().start(); // R,G,B

	int maxFramesToProcess = NMAXFRAMES;
	DepthFrameListener listener(maxFramesToProcess);
	ColorFrameListener listener2(maxFramesToProcess);

	reader.addListener(listener);
	reader.addListener(listener2);

	astra_temp_update();
	int iterationCounter = 0, depthToDisplay = 0;
	double t = (double)getTickCount();
	while (!listener.is_finished() && !listener2.is_finished() && charCheckForKey != 27)
	{
		astra_temp_update();

		flip(imgColor, imgColorMirrored, 1); // mirror color image

		flip(imgDepthFakeColor, imgDepthMirrored,1); // mirror synthetic depth image *** TODO - PHASE OUT
		flip(imgDepth, imgDepthRawMirrored,1); // mirror depth data matrix *** TODO - PHASE OUT
		depthCorrectionCustom(imgDepthMirrored, imgDepthMirroredCorrected); // custom	*** TODO - PHASE OUT

		addWeighted(imgDepthMirroredCorrected, 0.5, imgColorMirrored, 0.5, 0.0, imgOverlay); // corrected distortion

		char TEXT[32]; // for displaying marker range
		depthToDisplay = imgDepth.at<int16_t>(DEPTHPT[1], DEPTHPT[0]);
		sprintf(TEXT,"%dmm",depthToDisplay);
		putText(imgColorMirrored, TEXT, Point(25,25), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,192,55), 2, 8, false );

		imshow("depth image", imgOverlay);
		imshow("color image", imgColorMirrored);

		iterationCounter++;

		// print frame rate, depth info every 30 frames
		if(iterationCounter%30 == 0) {

			t = ((double)getTickCount() - t)/getTickFrequency();
			printf("t= %f, (%d,%d)\n",t/30.0, DEPTHPT[0], DEPTHPT[1]);
			t = (double)getTickCount();;

		}

		charCheckForKey = waitKey(30);
		updateKeyCmds();
	}

	reader.removeListener(listener);
	reader.removeListener(listener2);
	astra::Astra::terminate();

	//cout << imgDepth;

	return 0;
}


/// @brief Return color intensity for depth value.
///
/// Intensity is generated using a triangular function with
/// peak at (255,255) and end points (0,0) and (510,0). This
/// is useful in generating an R/G/B color value for a number
/// between 0 and 510, which peaks at 255. With 3 color
/// channels, it can be used so that different colors fade
/// in and out for different depth values.
///
/// Returns 0 outside the 0-510 range. So the prrovided depth
/// value must always be scaled/shifted to fit this range.
///
/// @param depth the depth value for which to generate color
///
/// @return a color value from 0-255
///
int lookupFunc (int depth) {

	if(depth < 0) {
		return  0;
	} else if( (depth >=0) && (depth <=255)) {
		return depth;
	} else if( (depth > 255) && (depth <= 510) ) {
		return (510 - depth);
	} else {
		return 0;
	}

}

/// @brief Update global flags and variables based on
/// global value that stores the last key stroke
///
/// @return Void
///
void updateKeyCmds() {

	if(charCheckForKey == 'w') { saveImages(); }

}

/// @brief Save color, depth, and overlay image view as a PNG image.
///
/// @return 0 if read successfully, 1 if failed
///
int saveImages() {

	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	char OUTPUT1[75], METADATASTRING1[25], OUTPUT2[75], METADATASTRING2[25], OUTPUT3[75], METADATASTRING3[25];
	sprintf(METADATASTRING1, METADATA);
	sprintf(OUTPUT1, "orbbecColor%s.png",METADATASTRING1);
	sprintf(METADATASTRING2, METADATA);
	sprintf(OUTPUT2, "orbbecDepth%s.png",METADATASTRING2);
	sprintf(METADATASTRING3, METADATA);
	sprintf(OUTPUT3, "orbbecFused%s.png",METADATASTRING3);
	try {
		imwrite(OUTPUT1, imgColorMirrored, compression_params);
	}
	catch (cv::Exception& ex) {
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
		return 1;
	}

	try {
		imwrite(OUTPUT2, imgDepthMirrored, compression_params);
	}
	catch (cv::Exception& ex) {
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
		return 1;
	}

	try {
		imwrite(OUTPUT3, imgOverlay, compression_params);
	}
	catch (cv::Exception& ex) {
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
		return 1;
	}
	fprintf(stdout, "Saved PNG file for color image window.\n");
}

static void onMouse(int event, int x, int y, int f, void*) {

	if (event == CV_EVENT_MOUSEMOVE) {
		DEPTHPT[0] = DWIDTH - x; // undo mirror effect
		DEPTHPT[1] = y;
	}

}


/// @brief Align the distorted depth image to color image
///
void depthCorrectionCustom (Mat src, Mat dst) {

	dst = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3);

	Vec3b intensity;
	intensity.val[0] = 0;
	intensity.val[1] = 0;
	intensity.val[2] = 0;
	int i, x, y,iMirror;
	float kx = 0.887, cx = 15.33, ky = 0.8837, cy = 18; // default depth correction parameters
	int16_t currDepth;
	// go through the color image
	for(i=0;i<DWIDTH*DHEIGHT;i++) {
		//currDepth = imgDepthRawMirrored.at<int16_t>(i/DWIDTH,i%DWIDTH);
		iMirror = i/DWIDTH * DWIDTH + DWIDTH - i%DWIDTH; // z data is mirrored, other is mirror corrected
		currDepth = (int16_t)DEPTHRAW[iMirror].z;

		if(currDepth>500 && currDepth<2000) {

			//imgDepth.at<int16_t>(i/DWIDTH,i%DWIDTH) = 16*(int)(allFrameData[i].z);

			intensity = src.at<Vec3b>(i/DWIDTH,i%DWIDTH); // y,x
			x = (float)(i%DWIDTH)*kx + depthCorrectionLookup(currDepth);  // depthCorrectionLookup(currDepth); //depthCorrectionLookup(DEPTHZ[iMirror]);
			y = (float)(i/DWIDTH)*ky + cy;
			dst.at<Vec3b>(y,x) = intensity;

		}

	}

}

void generateInverseUVMap() {

	int i, xColor, yColor, xDepth, yDepth;
	int16_t currDepth;
	float kx = 0.887, cx = 15.33, ky = 0.8837, cy = 18; // default depth correction parameters
	// clear inverse UV map
	for(i=0;i<DWIDTH*DHEIGHT;i++) {
		UVMAPINV[i].isMapped = 0; // set to unmapped
	}
	// go through the depth image
	for(i=0;i<DWIDTH*DHEIGHT;i++) {

		currDepth = (int16_t)DEPTHRAW[i].z;
		if(currDepth>500 && currDepth<2000) {

			xDepth = i%DWIDTH;
			yDepth = i/DWIDTH;
			xColor = (float)(DWIDTH-xDepth)*kx + depthCorrectionLookup(currDepth); // map the mirror corrected x-coord
			yColor = (float)(yDepth)*ky + cy;
			UVMAPINV[yColor*DWIDTH+xColor].isMapped = 1; // in range, set to mapped
			UVMAPINV[yColor*DWIDTH+xColor].u = xDepth;
			UVMAPINV[yColor*DWIDTH+xColor].v = yDepth;

		}

	}

}

void depthCorrectionUVMap(Mat src, Mat dst) {

	int i;
	dst = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3);

	Vec3b intensity;
	intensity.val[0] = 0;
	intensity.val[1] = 0;
	intensity.val[2] = 0;

	// go through the depth image
	for(i=0;i<DWIDTH*DHEIGHT;i++) {

	}

}


/// @brief Parameter cx for a given depth.
///
/// This is for the alignment of depth image pixels to color
/// image pixels for this model:
/// x_adjusted = kx*x + cx
/// y_adjusted = ky*y + cy
///
/// @param depth (mm)
///
/// @return cx (pixel)
///
float depthCorrectionLookup (float depth) {
	float cx;
	int d = (int)depth;
	if(d>=550 && d<=780) {
		cx = 0.043478*depth - 11.913;
	} else if(d>780 && d<=1020) {
		cx = 0.016667*depth + 8.9997;
	} else if(d>1020 && d<=1724) {
		cx = 0.0042614*depth + 21.653;
	} else if(d>1724) {
		cx = 29; // out of bounds of model
	} else if(d<550) {
		cx = 12; // out of bounds of model
	}

	return cx;

}

/// @brief Generate synthetic blue, green, red values based on
/// depth value.
///
/// @param depthVal depth value [mm]
/// @param *blue blue channel value in range 0-255
/// @param *green green channel value in range 0-255
/// @param *red red channel value in range 0-255
///
/// @return Void
///
void fakeDepthColor(float depthVal, int8_t *blue, int8_t *green, int8_t *red) {

	*blue = (int8_t)lookupFunc((int16_t)depthVal/3-600);	// B
	*green = (int8_t)lookupFunc((int16_t)depthVal/3-300);	// G
	*red = (int8_t)lookupFunc((int16_t)depthVal/3);			// R

}
