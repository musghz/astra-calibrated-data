/// @file astraCalibratedDataV04.cpp
///
/// @brief Connect to Orbbec astra, get color and point streams,
/// align the point stream data, and write point cloud to .ply
/// file.
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
/// Modified 04 Oct 2016 (V03)
/// -inverse UV map implemented
/// -write to .ply file
/// Modified 05 Oct 2016 (V04)
/// -cleanup
/// -getDepthAt()
/// -getPointAt()
/// -proper name for .ply
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
// depth range limits (mm) 500
#define MAXDEPTH 2000
#define MINDEPTH 500
// end program after this number of frames
#define NMAXFRAMES 18000
// for file names
#define METADATA "201611152005"

// for point in 3D space
struct Point3D {
	float x;
	float y;
	float z;
};

// for inverse UV map
struct Map3D {
	int16_t u; // x-coordinate of color image (pixel)
	int16_t v; // y-coordinate of color image (pixel)
	int8_t isMapped; // 0 if not mapped, 1 if mapped
};

Mat imgDepthMirroredCorrected = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3); // synthetic depth image, mirror corrected
Mat imgColor = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3); // image with color data
Mat imgColorMirrored = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3); // imgColor mirror corrected
Mat imgOverlay = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3); // color + depth overlay

Point3D DEPTHRAW[DWIDTH*DHEIGHT]; // xyz depth data
Map3D UVMAPINV[DWIDTH*DHEIGHT]; // map from color to depth pixels/data
char charCheckForKey = 0; // keyboard input
int DEPTHPT[2] = {0,0}; // read off depth from this point

/// @brief Return color intensity for depth value.
int lookupFunc (int depth);

/// @brief Update global flags and variables based on
/// global value that stores the last key stroke
void updateKeyCmds();

/// @brief Save color, depth, and overlay image view as a PNG 
/// image.
int saveImages();

/// @brief Save color, depth, and overlay image view as a 
/// Stanford .ply file.
int savePLYFile();

static void onMouse(int event, int x, int y, int f, void*);

/// @brief Generate an inverse UV map to lookup 3D coordinates 
/// given color pixels.
void generateInverseUVMap();

/// @brief Generate synthetic depth image based on 3D data
/// such that it is aligned with the color image.
void syntheticDepthImage (Mat dst);

/// @brief Parameter cx for a given depth.
float depthCorrectionLookup (float d);

/// @brief Generate synthetic blue, green, red values based on
/// depth value.
void fakeDepthColor(float depthVal, int8_t *blue, int8_t *green, int8_t *red);

/// @brief Get depth corresponding to a specific pixel.
float getDepthAt(int16_t x, int16_t y);

/// @brief Get 3D point corresponding to a specific pixel.
Point3D getPointAt(int16_t x, int16_t y);

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
		const astra::Vector3f* allFrameData = depthFrame.data();
		
		Vec3b intensity;
		int8_t blueVal, greenVal, redVal;
		int i;

		for(i=0;i<DWIDTH*DHEIGHT;i++) {
			
			DEPTHRAW[i].x = allFrameData[i].x;
			DEPTHRAW[i].y = allFrameData[i].y;
			DEPTHRAW[i].z = allFrameData[i].z;
			
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
		const astra::RGBPixel* allFrameData = colorFrame.data();

		imgColor = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3);
		Vec3b intensity;
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


int main(int argc, char** arvg) {

	// setup Open CV stuff
	namedWindow("depth image", CV_WINDOW_AUTOSIZE);
	namedWindow("color image", CV_WINDOW_AUTOSIZE);
	setMouseCallback("depth image", onMouse, NULL);

	// setup Orbbec Astra S camera
	int maxFramesToProcess = NMAXFRAMES;
	int iterationCounter = 0;
	double t = (double)getTickCount();
	astra::Astra::initialize();
	astra::StreamSet streamSet;
	astra::StreamReader reader = streamSet.create_reader();
	reader.stream<astra::PointStream>().start(); // x,y,z
	reader.stream<astra::ColorStream>().start(); // R,G,B	
	DepthFrameListener listener(maxFramesToProcess);
	ColorFrameListener listener2(maxFramesToProcess);
	reader.addListener(listener);
	reader.addListener(listener2);

	astra_temp_update(); // need to "pump" this
	
	while (!listener.is_finished() && !listener2.is_finished() && charCheckForKey != 27) {
	
		astra_temp_update(); // need to "pump" this

		flip(imgColor, imgColorMirrored, 1); // un-mirror color image
		generateInverseUVMap(); // inverse UV map for properly using depth data (required)
		syntheticDepthImage(imgDepthMirroredCorrected); // generate a synthetic depth image (for visualization)
		addWeighted(imgDepthMirroredCorrected, 0.5, imgColorMirrored, 0.5, 0.0, imgOverlay); // for visualization

		char TEXT[32]; // for displaying marker range
		sprintf(TEXT,"%dmm",(int)getDepthAt(DEPTHPT[0],DEPTHPT[1]));
		putText(imgColorMirrored, TEXT, Point(25,25), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,192,55), 2, 8, false );

		char TEXT2[56]; // for displaying xyz
		Point3D currData = getPointAt(DEPTHPT[0],DEPTHPT[1]);
		sprintf(TEXT2,"%d,%d,%d",(int)currData.x,(int)currData.y,(int)currData.z);
		putText(imgColorMirrored, TEXT2, Point(25,55), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,192,55), 2, 8, false );

		imshow("depth image", imgOverlay);
		imshow("color image", imgColorMirrored);

		iterationCounter++;

		// print frame rate, depth info every 30 frames
		if(iterationCounter%30 == 0) {

			t = ((double)getTickCount() - t)/getTickFrequency();
			printf("t= %f, (%d,%d)\n",t/30.0, DEPTHPT[0], DEPTHPT[1]);
			t = (double)getTickCount();
		}

		charCheckForKey = waitKey(30);
		updateKeyCmds();
	}

	reader.removeListener(listener);
	reader.removeListener(listener2);
	astra::Astra::terminate();

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
/// Returns 0 outside the 0-510 range. So the provided depth
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

	if(charCheckForKey == 'w') { 
		saveImages();
		savePLYFile();
	}
}


/// @brief Save color, depth, and overlay image view as a PNG 
/// image.
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
		imwrite(OUTPUT2, imgDepthMirroredCorrected, compression_params);
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


/// @brief Save color, depth, and overlay image view as a 
/// Stanford .ply file.
///
/// @return 0 if read successfully, 1 if failed
///
int savePLYFile() {

	printf("Attempting to write .ply file...\n");
	int i=0, iDepth=0, counter=0;
	Vec3b intensity;

	char OUTPUT[75], METADATASTRING[25];
	sprintf(METADATASTRING, METADATA);
	sprintf(OUTPUT, "orbbecPoints%s.ply",METADATASTRING);

	// file write stuff
	FILE * fp;
	fp = fopen (OUTPUT,"w");
	if(fp==NULL){
		printf(".ply file write error!\n");
		return 1;
	}

	// count elements
	for(i=0;i<DWIDTH*DHEIGHT;i++) {
		if(UVMAPINV[i].isMapped == 1) {
			counter++;
		}
	}

	// header info
	fprintf(fp,"ply\nformat ascii 1.0\ncomment author: Mustafa Ghazi, Intelligent Robotics Lab\ncomment object: Orbbec Astra S data\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",counter);

	// cycle through color pixels
	for(i=0;i<DWIDTH*DHEIGHT;i++) {
		if(UVMAPINV[i].isMapped == 1) {
			iDepth = DWIDTH*UVMAPINV[i].v + UVMAPINV[i].u;
			intensity = imgColorMirrored.at<Vec3b>(i/DWIDTH,i%DWIDTH); // y,x
			fprintf(fp,"%.2f %.2f %.2f %d %d %d\n",DEPTHRAW[iDepth].x,DEPTHRAW[iDepth].y,DEPTHRAW[iDepth].z,intensity.val[2],intensity.val[1],intensity.val[0]);
		}
	}

	fclose(fp);
	printf("Saved .ply file successfully!\n");

	return 0;
}


static void onMouse(int event, int x, int y, int f, void*) {

	if (event == CV_EVENT_MOUSEMOVE) {
		DEPTHPT[0] = x; 
		DEPTHPT[1] = y;
	}
}


/// @brief Generate an inverse UV map to lookup 3D coordinates 
/// given color pixels.
///
/// Every i-th elemnt in the UV map array corresponds to the 
/// i-th element (i%WIDTH,i/WIDTH) in the color image. The 
/// (u,v) values represent the corresponding depth image pixel
/// (i=u+v*WIDTH). The variable isMapped indicates whether
/// this array element is mapped or not.
///
/// An element is mapped only if its z-depth is between MINDEPTH
/// and MAXDEPTH. 
///
/// @return Void
///
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
		if(currDepth>=MINDEPTH && currDepth<=MAXDEPTH) {
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


/// @brief Generate synthetic depth image based on 3D data
/// such that it is aligned with the color image.
///
/// Image is generated based on inverse UV Map. Colors are
/// limited to the predefined min/max depth range.
///
/// @param dst 
///
/// @return Void
///
void syntheticDepthImage(Mat dst) {

	int i, iDepth;
	int8_t blue, green, red;
	float thisDepth;
	Vec3b intensity;
	dst = Mat::zeros(DHEIGHT, DWIDTH, CV_8UC3);	

	// go through the indices corresponding to the color image
	for(i=0;i<DWIDTH*DHEIGHT;i++) {
		if(UVMAPINV[i].isMapped == 1) {
			iDepth = DWIDTH*UVMAPINV[i].v + UVMAPINV[i].u;
			fakeDepthColor(DEPTHRAW[iDepth].z,&blue,&green,&red);
			intensity.val[0] = blue;
			intensity.val[1] = green;
			intensity.val[2] = red;

			dst.at<Vec3b>(i/DWIDTH,i%DWIDTH) = intensity;
		}
	}
}


/// @brief Parameter cx for a given depth.
///
/// This is for the alignment of depth image pixels to color
/// image pixels for this model:
/// x_adjusted = kx*x + cx
/// y_adjusted = ky*y + cy
///
/// As of 02 Oct 2016 it is valid for ~550 to ~1724 mm.
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


/// @brief Get depth corresponding to a specific pixel.
///
/// x and y must lie within frame width and frame height.
///
/// Coordinate systems:
/// -Mirror-corrected color camera image frame, origin at top 
/// left corner, +x right, +y down
/// -Uncorrected depth camera image frame, origin at top left
/// corner, +u right, +v down
/// -camera body coordinate frame, origin at ?? +z out the 
/// image plane, +x ?? +y??
///
/// @param x x-coordinate, mirror corrected color camera image
/// frame [pixels]
/// @param y y-coordinate, mirror corrected color camera image
/// frame [pixels]
///
/// @return z coordinate, camera body frame
///
float getDepthAt(int16_t x, int16_t y) {

	// check if within range	
	if((x<0) || (x>DWIDTH) || (y<0) || (y>DHEIGHT)) {
		return 0.0;
	}
	
	int clrIdx = DWIDTH*y + x;
	int depthIdx = DWIDTH*UVMAPINV[clrIdx].v  + UVMAPINV[clrIdx].u;	
	if((depthIdx >=0) && (depthIdx<=DWIDTH*DHEIGHT)){
		return DEPTHRAW[depthIdx].z;
	} else { 
		return 0.0; 
	}
}


/// @brief Get 3D point corresponding to a specific pixel.
///
/// x and y must lie within frame width and frame height.
///
/// Coordinate systems:
/// -Mirror-corrected color camera image frame, origin at top 
/// left corner, +x right, +y down
/// -Uncorrected depth camera image frame, origin at top left
/// corner, +u right, +v down
/// -camera body coordinate frame, origin at ?? +z out the 
/// image plane along LOS, +x left +y up
///
/// @param x x-coordinate, mirror corrected color camera image
/// frame [pixels]
/// @param y y-coordinate, mirror corrected color camera image
/// frame [pixels]
///
/// @return Point3D, camera body frame
///
Point3D getPointAt(int16_t x, int16_t y) {

	Point3D thisPoint;
	thisPoint.x = 0;
	thisPoint.y = 0;
	thisPoint.z = 0;
	// check if within range	
	if((x<0) || (x>DWIDTH) || (y<0) || (y>DHEIGHT)) {
		return thisPoint;
	}

	int clrIdx = DWIDTH*y + x;
	int depthIdx = DWIDTH*UVMAPINV[clrIdx].v  + UVMAPINV[clrIdx].u;	
	if((depthIdx >=0) && (depthIdx<=DWIDTH*DHEIGHT)){
		thisPoint.x = DEPTHRAW[depthIdx].x;
		thisPoint.y = DEPTHRAW[depthIdx].y;
		thisPoint.z = DEPTHRAW[depthIdx].z;
		return thisPoint;
	} else { 
		return thisPoint; 
	}
}