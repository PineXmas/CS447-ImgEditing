///////////////////////////////////////////////////////////////////////////////
//
//      TargaImage.cpp                          Author:     Stephen Chenney
//                                              Modified:   Eric McDaniel
//                                              Date:       Fall 2004
//                                              Modified:   Feng Liu
//                                              Date:       Winter 2011
//                                              Why:        Change the library file 
//      Implementation of TargaImage methods.  You must implement the image
//  modification functions.
//
///////////////////////////////////////////////////////////////////////////////

#include "Globals.h"
#include "TargaImage.h"
#include "libtarga.h"
#include <stdlib.h>
#include <assert.h>
#include <memory.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace std;

// constants
const int           RED             = 0;                // red channel
const int           GREEN           = 1;                // green channel
const int           BLUE            = 2;                // blue channel
const unsigned char BACKGROUND[3]   = { 0, 0, 0 };      // background color


// Computes n choose s, efficiently
double Binomial(int n, int s)
{
	double        res;

	res = 1;
	for (int i = 1 ; i <= s ; i++)
		res = (n - i + 1) * res / i ;

	return res;
}// Binomial


///////////////////////////////////////////////////////////////////////////////
//
//      Constructor.  Initialize member variables.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage() : width(0), height(0), data(NULL)
{}// TargaImage

///////////////////////////////////////////////////////////////////////////////
//
//      Constructor.  Initialize member variables.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage(int w, int h) : width(w), height(h)
{
	data = new unsigned char[width * height * 4];
	ClearToBlack();
}// TargaImage



///////////////////////////////////////////////////////////////////////////////
//
//      Constructor.  Initialize member variables to values given.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage(int w, int h, unsigned char *d)
{
	int i;

	width = w;
	height = h;
	data = new unsigned char[width * height * 4];

	for (i = 0; i < width * height * 4; i++)
		data[i] = d[i];
}// TargaImage

///////////////////////////////////////////////////////////////////////////////
//
//      Copy Constructor.  Initialize member to that of input
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::TargaImage(const TargaImage& image) 
{
	width = image.width;
	height = image.height;
	data = NULL; 
	if (image.data != NULL) {
		data = new unsigned char[width * height * 4];
		memcpy(data, image.data, sizeof(unsigned char) * width * height * 4);
	}
}


///////////////////////////////////////////////////////////////////////////////
//
//      Destructor.  Free image memory.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage::~TargaImage()
{
	if (data)
		delete[] data;
}// ~TargaImage


///////////////////////////////////////////////////////////////////////////////
//
//      Converts an image to RGB form, and returns the rgb pixel data - 24 
//  bits per pixel. The returned space should be deleted when no longer 
//  required.
//
///////////////////////////////////////////////////////////////////////////////
unsigned char* TargaImage::To_RGB(void)
{
	unsigned char   *rgb = new unsigned char[width * height * 3];
	int		    i, j;

	if (! data)
		return NULL;

	// Divide out the alpha
	for (i = 0 ; i < height ; i++)
	{
		int in_offset = i * width * 4;
		int out_offset = i * width * 3;

		for (j = 0 ; j < width ; j++)
		{
			RGBA_To_RGB(data + (in_offset + j*4), rgb + (out_offset + j*3));
		}
	}

	return rgb;
}// TargaImage


///////////////////////////////////////////////////////////////////////////////
//
//      Save the image to a targa file. Returns 1 on success, 0 on failure.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Save_Image(const char *filename)
{
	TargaImage	*out_image = Reverse_Rows();

	if (! out_image)
		return false;

	if (!tga_write_raw(filename, width, height, out_image->data, TGA_TRUECOLOR_32))
	{
		cout << "TGA Save Error: %s\n", tga_error_string(tga_get_last_error());
		return false;
	}

	delete out_image;

	return true;
}// Save_Image


///////////////////////////////////////////////////////////////////////////////
//
//      Load a targa image from a file.  Return a new TargaImage object which 
//  must be deleted by caller.  Return NULL on failure.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage* TargaImage::Load_Image(char *filename)
{
	unsigned char   *temp_data;
	TargaImage	    *temp_image;
	TargaImage	    *result;
	int		        width, height;

	if (!filename)
	{
		cout << "No filename given." << endl;
		return NULL;
	}// if

	temp_data = (unsigned char*)tga_load(filename, &width, &height, TGA_TRUECOLOR_32);
	if (!temp_data)
	{
		cout << "TGA Error: %s\n", tga_error_string(tga_get_last_error());
		width = height = 0;
		return NULL;
	}
	temp_image = new TargaImage(width, height, temp_data);
	free(temp_data);

	result = temp_image->Reverse_Rows();

	delete temp_image;

	return result;
}// Load_Image


///////////////////////////////////////////////////////////////////////////////
//
//      Convert image to grayscale.  Red, green, and blue channels should all 
//  contain grayscale value.  Alpha channel shoould be left unchanged.  Return
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::To_Grayscale()
{
	int i, j;

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			//convert current pixel from pre-multiplied RGBA back to RGB
			int pixelOffset = i*width*4 + j*4;
			unsigned char rgb[3];
			RGBA_To_RGB(data + pixelOffset, rgb);

			//use the formula I = 0.299r + 0.587g + 0.114b to convert color images to grayscale
			//multiply alpha-value back to RGB before re-assign to data
			unsigned char intensity = CalculateGrayscale(rgb[0], rgb[1], rgb[2]);
			intensity = floor((double)intensity * ((double)*(data + pixelOffset + 3) / 255.0));
			for (int k=0; k<3; k++){
				*(data + pixelOffset + k) = intensity;
			}
		}
	}
	return true;
}// To_Grayscale

//
// Using the formula I = 0.299r + 0.587g + 0.114b to convert color images to grayscale
//
unsigned char TargaImage::CalculateGrayscale(unsigned char r, unsigned char g, unsigned char b){
	return floor( (double)r * 0.299 + (double)g*0.587 + (double)b*0.114);
}


///////////////////////////////////////////////////////////////////////////////
//
//  Convert the image to an 8 bit image using uniform quantization.  Return 
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Quant_Uniform()
{
	int i, j;

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			//convert current pixel from pre-multiplied RGBA back to RGB
			int pixelOffset = i*width*4 + j*4;
			unsigned char rgb[3];
			RGBA_To_RGB(data + pixelOffset, rgb);

			//Use the uniform quantization algorithm to convert the current image from a 24 bit color image to an 8 bit color image.
			//Use 4 levels of blue, 8 levels of red, and 8 levels of green in the quantized image.
			//multiply alpha-value back to RGB before re-assign to data
			double alpha = (double)*(data + pixelOffset + 3) / 255.0;
			*(data + pixelOffset + 0) = floor(double(rgb[0] / 32 * 32) * alpha);
			*(data + pixelOffset + 1) = floor(double(rgb[1] / 32 * 32) * alpha);
			*(data + pixelOffset + 2) = floor(double(rgb[2] / 64 * 64) * alpha);
		}
	}
	return true;
}// Quant_Uniform


///////////////////////////////////////////////////////////////////////////////
//
//      Convert the image to an 8 bit image using populosity quantization.  
//  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Quant_Populosity()
{
	//init arrCounter for histogram, and 3 other arrRed/Green/Blue for keep track of what color at a particular index in the arrCounter
	int i, j;
	const int nColors = 32768;
	int arrCounter[nColors] = {0};
	int arrRed[nColors];
	int arrGreen[nColors];
	int arrBlue[nColors];

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			//convert current pixel from pre-multiplied RGBA back to RGB
			int pixelOffset = i*width*4 + j*4;
			unsigned char rgb[3];
			RGBA_To_RGB(data + pixelOffset, rgb);

			//Before building the color usage histogram, do a uniform quantization step down to 32 levels of each primary
			int iRed;
			int iGreen;
			int iBlue;
			int index;
			RGB2Index(rgb[0],rgb[1], rgb[2], &index, &iRed, &iGreen, &iBlue);

			//update the count for the current color in the arrCounter, and set the coresponding color if not init
			if (index >= nColors || index < 0) {
				int i=0;
			}
			if (arrCounter[index] == 0){
				arrRed[index] = rgb[0];
				arrGreen[index] = rgb[1];
				arrBlue[index] = rgb[2];
			}
			arrCounter[index]++; 
		}
	}

	SortHistogram(arrCounter, arrRed, arrGreen, arrBlue, nColors);

	//map colors to their nearest color in the top 256 colors in the arrCounter
	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			//convert current pixel from pre-multiplied RGBA back to RGB
			int pixelOffset = i*width*4 + j*4;
			unsigned char rgb[3];
			RGBA_To_RGB(data + pixelOffset, rgb);

			int iNearest = FindHistogramNearestColor(arrRed, arrGreen, arrBlue, rgb[0], rgb[1], rgb[2]);

			//update color data
			double alpha = (double)*(data + pixelOffset + 3) / 255.0;
			*(data + pixelOffset + 0) = floor((double)arrRed  [iNearest] * alpha);
			*(data + pixelOffset + 1) = floor((double)arrGreen[iNearest] * alpha);
			*(data + pixelOffset + 2) = floor((double)arrBlue [iNearest] * alpha);
		}
	}

	return true;
}// Quant_Populosity

//
// Calculate the given values in R, G, B to the index in an array of 32*32*32 = 32768 colors, and scale the values of R/G/B to the range [0..31]
//
void TargaImage::RGB2Index(unsigned char r, unsigned char g, unsigned char b, int* index, int* iRed, int* iGreen, int* iBlue){
	*iRed = r / 8;
	*iGreen = g / 8;
	*iBlue = b / 8;
	*index = (*iRed * 32 * 32) + (*iGreen * 32) + (*iBlue);
	*iRed *= 8;
	*iGreen *= 8;
	*iBlue *= 8;
}

//
// Sort the histogram array decreasingly
//
void TargaImage::SortHistogram(int arr[], int arrR[], int arrG[], int arrB[], int size){
	for (int i=0; i<size-1; i++){
		for (int j=i+1; j<size; j++){
			if (arr[i] < arr[j]){
				Swap(arr+i, arr+j);
				Swap(arrR+i, arrR+j);
				Swap(arrG+i, arrG+j);
				Swap(arrB+i, arrB+j);
			}
		}
	}
}

//
// Swap 2 given numbers
//
void TargaImage::Swap(int* a, int* b){
	int t = *a;
	*a = *b;
	*b = t;
}

//
// Find the nearest color among the top 256 colors in the given histogram and return the histogram index of the result
//
int TargaImage::FindHistogramNearestColor(int arrR[], int arrG[], int arrB[], unsigned char r, unsigned char g, unsigned char b){
	int index = 0;
	double min = EuclidDistance(r, g, b, arrR[0], arrG[0], arrB[0]);

	for (int i = 1; i < 256; i++) {
		double tmp = EuclidDistance(r, g, b, arrR[i], arrG[i], arrB[i]);
		if (min > tmp){
			index = i;
			min = tmp;
		}
	}
	return index;
}

//
// Find the distance between 2 colors
//
double TargaImage::EuclidDistance(double r1, double g1, double b1, double r2, double g2, double b2){
	return sqrt(pow(r1-r2, 2) + pow(g1-g2, 2) + pow(b1-b2, 2));
}

///////////////////////////////////////////////////////////////////////////////
//
//      Dither the image using a threshold of 1/2.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Threshold()
{
	return Dither_Threshold(0.5);
}// Dither_Threshold

//
// Dither the image using a the given threshold
//
bool TargaImage::Dither_Threshold(double threshold){
	int i, j;

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			//convert current pixel from pre-multiplied RGBA back to RGB
			int pixelOffset = i*width*4 + j*4;
			unsigned char rgb[3];
			RGBA_To_RGB(data + pixelOffset, rgb);

			//Dither an image to black and white using threshold dithering with a threshold of 0.5.
			double grayscale = CalculateGrayscale(rgb[0], rgb[1], rgb[2]);
			double color = 255;
			if ((grayscale / 255.0) < threshold) {
				color = 0;
			}
			double alpha = (double)*(data + pixelOffset + 3) / 255.0;
			for (int i = 0; i < 3; i++) {
				*(data + pixelOffset + i) = floor(color * alpha);
			}
		}
	}
	return true;
}


///////////////////////////////////////////////////////////////////////////////
//
//      Dither image using random dithering.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Random()
{
	int i, j;

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			//convert current pixel from pre-multiplied RGBA back to RGB
			int pixelOffset = i*width*4 + j*4;
			unsigned char rgb[3];
			RGBA_To_RGB(data + pixelOffset, rgb);

			//Dither an image to black and white using random dithering.
			//Add random values chosen uniformly from the range [-0.2,0.2],
			//assuming that the input image intensity runs from 0 to 1 (scale appropriately)
			double grayscale = CalculateGrayscale(rgb[0], rgb[1], rgb[2]);
			grayscale /= 255.0;
			grayscale += 0.4 / RAND_MAX * rand() - 0.2;
			if (grayscale < 0) {
				grayscale = 0;
			}
			if (grayscale > 1) {
				grayscale = 1;
			}
			double color = 255;
			if (grayscale <= 0.5) {
				color = 0;
			}
			double alpha = (double)*(data + pixelOffset + 3) / 255.0;
			for (int i = 0; i < 3; i++) {
				*(data + pixelOffset + i) = floor(color * alpha);
			}
		}
	}
	return true;
}// Dither_Random


///////////////////////////////////////////////////////////////////////////////
//
//      Perform Floyd-Steinberg dithering on the image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_FS()
{
	ClearToBlack();
	return false;
}// Dither_FS


///////////////////////////////////////////////////////////////////////////////
//
//      Dither the image while conserving the average brightness.  Return 
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Bright()
{
	//count the number of pixel in each brightness degree
	int i, j;
	int brightCounter[256] = {0};
	double brightSum = 0;

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			//convert current pixel from pre-multiplied RGBA back to RGB
			int pixelOffset = i*width*4 + j*4;
			unsigned char rgb[3];
			RGBA_To_RGB(data + pixelOffset, rgb);

			//Dither an image to black and white using threshold dithering with a threshold chosen to keep the average brightness constant.
			int grayscale = CalculateGrayscale(rgb[0], rgb[1], rgb[2]);
			brightSum += grayscale;
			brightCounter[grayscale]++;
		}
	}

	//determine the threshold
	double avgBright = (brightSum / double(width*height)) / 255.0;
	int numDarkPixel = floor((1 - avgBright) * (double)width * (double)height);
	double pixelCount = 0;
	int threshold = 0;
	for (int i = 0; i < 256 && pixelCount < numDarkPixel; i++)
	{
		pixelCount += brightCounter[i];
		threshold = i;
	}

	return Dither_Threshold((double)threshold/255.0);
}// Dither_Bright


///////////////////////////////////////////////////////////////////////////////
//
//      Perform clustered differing of the image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Cluster()
{
	ClearToBlack();
	return false;
}// Dither_Cluster


///////////////////////////////////////////////////////////////////////////////
//
//  Convert the image to an 8 bit image using Floyd-Steinberg dithering over
//  a uniform quantization - the same quantization as in Quant_Uniform.
//  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Dither_Color()
{
	ClearToBlack();
	return false;
}// Dither_Color

//
// Composite the two given pixels with the providen factors f1, f2
//
unsigned char* Composite2Pixels(unsigned char* rgba1, unsigned char* rgba2, double f1, double f2) {
	unsigned char* rgba = new unsigned char [4];
	for (int i = 0; i < 4; i++)
	{
		double comp = floor((double)rgba1[i] * f1 + (double)rgba2[i] * f2);
		if (comp > 255)
		{
			comp = 255;
		} else if (comp < 0)
		{
			comp = 0;
		}
		rgba[i] = comp;
	}
	return rgba;
}

///////////////////////////////////////////////////////////////////////////////
//
//      Composite the current image over the given image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Over(TargaImage* pImage)
{
	if (width != pImage->width || height != pImage->height)
	{
		cout <<  "Comp_Over: Images not the same size\n";
		return false;
	}

	int i, j;

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			//convert current pixel from pre-multiplied RGBA back to RGB
			int pixelOffset = i*width*4 + j*4;
			unsigned char* rgba1 = new unsigned char[4];
			unsigned char* rgba2 = new unsigned char[4];
			for (int k = 0; k < 4; k++)
			{
				rgba1[k] = *(data + pixelOffset + k);
				rgba2[k] = *(pImage->data + pixelOffset + k);
			}

			//calculate composition factors
			double alpha1 = (double)rgba1[3] / 255.0;
			double f1 = 1;
			double f2 = 1 - alpha1;

			//set composited pixel
			unsigned char* composited = Composite2Pixels(rgba1, rgba2, f1, f2);
			for (int k = 0; k < 4; k++)
			{
				*(data + pixelOffset + k) = composited[k];
			}
		}
	}
	return true;
}// Comp_Over


///////////////////////////////////////////////////////////////////////////////
//
//      Composite this image "in" the given image.  See lecture notes for 
//  details.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_In(TargaImage* pImage)
{
	if (width != pImage->width || height != pImage->height)
	{
		cout << "Comp_In: Images not the same size\n";
		return false;
	}

	int i, j;

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			//convert current pixel from pre-multiplied RGBA back to RGB
			int pixelOffset = i*width*4 + j*4;
			unsigned char* rgba1 = new unsigned char[4];
			unsigned char* rgba2 = new unsigned char[4];
			for (int k = 0; k < 4; k++)
			{
				rgba1[k] = *(data + pixelOffset + k);
				rgba2[k] = *(pImage->data + pixelOffset + k);
			}

			//calculate composition factors
			double alpha2 = (double)rgba2[3] / 255.0;
			double f1 = alpha2;
			double f2 = 0;

			//set composited pixel
			unsigned char* composited = Composite2Pixels(rgba1, rgba2, f1, f2);
			for (int k = 0; k < 4; k++)
			{
				*(data + pixelOffset + k) = composited[k];
			}
		}
	}
	return true;
}// Comp_In


///////////////////////////////////////////////////////////////////////////////
//
//      Composite this image "out" the given image.  See lecture notes for 
//  details.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Out(TargaImage* pImage)
{
	if (width != pImage->width || height != pImage->height)
	{
		cout << "Comp_Out: Images not the same size\n";
		return false;
	}

	int i, j;

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			//convert current pixel from pre-multiplied RGBA back to RGB
			int pixelOffset = i*width*4 + j*4;
			unsigned char* rgba1 = new unsigned char[4];
			unsigned char* rgba2 = new unsigned char[4];
			for (int k = 0; k < 4; k++)
			{
				rgba1[k] = *(data + pixelOffset + k);
				rgba2[k] = *(pImage->data + pixelOffset + k);
			}

			//calculate composition factors
			double alpha2 = (double)rgba2[3] / 255.0;
			double f1 = 1 - alpha2;
			double f2 = 0;

			//set composited pixel
			unsigned char* composited = Composite2Pixels(rgba1, rgba2, f1, f2);
			for (int k = 0; k < 4; k++)
			{
				*(data + pixelOffset + k) = composited[k];
			}
		}
	}
	return true;
}// Comp_Out


///////////////////////////////////////////////////////////////////////////////
//
//      Composite current image "atop" given image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Atop(TargaImage* pImage)
{
	if (width != pImage->width || height != pImage->height)
	{
		cout << "Comp_Atop: Images not the same size\n";
		return false;
	}

	int i, j;

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			//convert current pixel from pre-multiplied RGBA back to RGB
			int pixelOffset = i*width*4 + j*4;
			unsigned char* rgba1 = new unsigned char[4];
			unsigned char* rgba2 = new unsigned char[4];
			for (int k = 0; k < 4; k++)
			{
				rgba1[k] = *(data + pixelOffset + k);
				rgba2[k] = *(pImage->data + pixelOffset + k);
			}

			//calculate composition factors
			double alpha1 = (double)rgba1[3] / 255.0;
			double alpha2 = (double)rgba2[3] / 255.0;
			double f1 = alpha2;
			double f2 = 1 - alpha1;

			//set composited pixel
			unsigned char* composited = Composite2Pixels(rgba1, rgba2, f1, f2);
			for (int k = 0; k < 4; k++)
			{
				*(data + pixelOffset + k) = composited[k];
			}
		}
	}
	return true;
}// Comp_Atop


///////////////////////////////////////////////////////////////////////////////
//
//      Composite this image with given image using exclusive or (XOR).  Return
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Comp_Xor(TargaImage* pImage)
{
	if (width != pImage->width || height != pImage->height)
	{
		cout << "Comp_Xor: Images not the same size\n";
		return false;
	}

	int i, j;

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			//convert current pixel from pre-multiplied RGBA back to RGB
			int pixelOffset = i*width*4 + j*4;
			unsigned char* rgba1 = new unsigned char[4];
			unsigned char* rgba2 = new unsigned char[4];
			for (int k = 0; k < 4; k++)
			{
				rgba1[k] = *(data + pixelOffset + k);
				rgba2[k] = *(pImage->data + pixelOffset + k);
			}

			//calculate composition factors
			double alpha1 = (double)rgba1[3] / 255.0;
			double alpha2 = (double)rgba2[3] / 255.0;
			double f1 = 1 - alpha2;
			double f2 = 1 - alpha1;

			//set composited pixel
			unsigned char* composited = Composite2Pixels(rgba1, rgba2, f1, f2);
			for (int k = 0; k < 4; k++)
			{
				*(data + pixelOffset + k) = composited[k];
			}
		}
	}
	return true;
}// Comp_Xor


///////////////////////////////////////////////////////////////////////////////
//
//      Calculate the difference bewteen this imag and the given one.  Image 
//  dimensions must be equal.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Difference(TargaImage* pImage)
{
	if (!pImage)
		return false;

	if (width != pImage->width || height != pImage->height)
	{
		cout << "Difference: Images not the same size\n";
		return false;
	}// if

	for (int i = 0 ; i < width * height * 4 ; i += 4)
	{
		unsigned char        rgb1[3];
		unsigned char        rgb2[3];

		RGBA_To_RGB(data + i, rgb1);
		RGBA_To_RGB(pImage->data + i, rgb2);

		data[i] = abs(rgb1[0] - rgb2[0]);
		data[i+1] = abs(rgb1[1] - rgb2[1]);
		data[i+2] = abs(rgb1[2] - rgb2[2]);
		data[i+3] = 255;
	}

	return true;
}// Difference


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 box filter on this image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Box()
{
	int matrix1D[5] = {1, 1, 1, 1, 1};
	return Filter_Matrix(Gen2DMatrix(matrix1D, 5), 5);
}// Filter_Box

//
// Generate an NxN matrix based on the given Nx1 matrix
//
int** TargaImage::Gen2DMatrix(int* matrix1D, int size){
	int** result = new int*[size];
	for (int i = 0; i < size; i++)
	{
		result[i] = new int[size];
		for (int j = 0; j < size; j++)
		{
			result[i][j] = matrix1D[i] * matrix1D[j];
		}
	}
	return result;
}

//
// Perform NxN matrix filter on this image (N is passed by size).  Return success of operation
//
bool TargaImage::Filter_Matrix(int** matrix, int size)
{
	int i, j;
	double matrixSum = CalculateMatrixSum(matrix, size);

	for (i=0; i<height; i++){
		for (j=0; j<width; j++){
			int pixelOffset = i*width*4 + j*4;
			unsigned char filteredRGB[3];
			ApplyFilter(matrix, size, matrixSum, filteredRGB, i, j);

			double alpha = *(data + pixelOffset + 3) / 255.0;
			for (int k = 0; k < 3; k++)
			{
				*(data + pixelOffset + k) = floor((double)filteredRGB[k] * alpha);
			}
		}
	}

	return true;
}

//
// Apply Filter to the given pixel by the given matrix values. Deal with edge pixels by reflecting it about its edges.
//
void TargaImage::ApplyFilter(int** matrix, int size, double matrixSum, unsigned char* rgb, int currRow, int currCol){
	double sumRGB[3] = {0};
	int boundary = size / 2;

	for (int i = -boundary; i < boundary+1; i++)
	{
		for (int j = -boundary; j < boundary+1; j++)
		{
			//get RGB values
			unsigned char target[4];
			int targetRow = currRow + i;
			int targetCol = currCol + j;
			if (targetRow < 0) {
				targetRow = -targetRow;
			} else if (targetRow >= height){
				targetRow = height * 2 - targetRow - 2;
			}
			if (targetCol < 0) {
				targetCol = -targetCol;
			} else if (targetCol >= width){
				targetCol = width * 2 - targetCol - 2;
			}
			GetOriginPixel(targetRow, targetCol, target);

			//compute & accumulate values
			double currMatrixValue = matrix[i+boundary][j+boundary];
			for (int k = 0; k < 3; k++)
			{
				sumRGB[k] += currMatrixValue * (double)target[k];
			}
		}
	}

	//set filtered value for output
	for (int i = 0; i < 3; i++)
	{
		rgb[i] = floor(sumRGB[i] / matrixSum);
	}
}

//
// Retrieve the ORIGINAL values of R/G/B/Alpha at a given pixel position
//
void TargaImage::GetOriginPixel(int currRow, int currCol, unsigned char* rgba){
	int pixelOffset = currRow*width*4 + currCol*4;
	unsigned char rgb[3];
	RGBA_To_RGB(data + pixelOffset, rgb);
	for (int i = 0; i < 3; i++)
	{
		rgba[i] = rgb[i];
	}
	rgba[3] = *(data + pixelOffset + 3);
}

//
// Calculate matrix sum
//
double TargaImage::CalculateMatrixSum(int** maxtrix, int size){
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			sum += maxtrix[i][j];
		}
	}
	return sum;
}

///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 Bartlett filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Bartlett()
{
	int matrix1D[5] = {1, 2, 3, 2, 1};
	return Filter_Matrix(Gen2DMatrix(matrix1D, 5), 5);
}// Filter_Bartlett


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 Gaussian filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Gaussian()
{
	return Filter_Gaussian_N(5);
}// Filter_Gaussian

//
// Create Gaussian Filter Mask based on the given size
//


///////////////////////////////////////////////////////////////////////////////
//
//      Perform NxN Gaussian filter on this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////

bool TargaImage::Filter_Gaussian_N( unsigned int N )
{
	int* matrix1D = GenGaussian1DMatrix(N);
	return Filter_Matrix(Gen2DMatrix(matrix1D, N), N);
}// Filter_Gaussian_N

//
// Generate Nx1 Gaussian Mask
//
int* TargaImage::GenGaussian1DMatrix(int n){
	int* m1D;

	if (n < 3)
	{
		m1D = new int[n];
		for (int i = 0; i < n; i++)
		{
			m1D[i] = 1;
		}
		
		return m1D;
	}

	int* mPre = GenGaussian1DMatrix(n-1);
	m1D = new int[n];
	m1D[0] = 1;
	m1D[n-1] = 1;
	for (int i = 0; i < n-2; i++)
	{
		m1D[i+1] = mPre[i] + mPre[i+1];
	}
	return m1D;
}


///////////////////////////////////////////////////////////////////////////////
//
//      Perform 5x5 edge detect (high pass) filter on this image.  Return 
//  success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Edge()
{
	ClearToBlack();
	return false;
}// Filter_Edge


///////////////////////////////////////////////////////////////////////////////
//
//      Perform a 5x5 enhancement filter to this image.  Return success of 
//  operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Filter_Enhance()
{
	ClearToBlack();
	return false;
}// Filter_Enhance


///////////////////////////////////////////////////////////////////////////////
//
//      Run simplified version of Hertzmann's painterly image filter.
//      You probably will want to use the Draw_Stroke funciton and the
//      Stroke class to help.
// Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::NPR_Paint()
{
	ClearToBlack();
	return false;
}



///////////////////////////////////////////////////////////////////////////////
//
//      Halve the dimensions of this image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Half_Size()
{
	ClearToBlack();
	return false;
}// Half_Size


///////////////////////////////////////////////////////////////////////////////
//
//      Double the dimensions of this image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Double_Size()
{
	ClearToBlack();
	return false;
}// Double_Size


///////////////////////////////////////////////////////////////////////////////
//
//      Scale the image dimensions by the given factor.  The given factor is 
//  assumed to be greater than one.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Resize(float scale)
{
	ClearToBlack();
	return false;
}// Resize


//////////////////////////////////////////////////////////////////////////////
//
//      Rotate the image clockwise by the given angle.  Do not resize the 
//  image.  Return success of operation.
//
///////////////////////////////////////////////////////////////////////////////
bool TargaImage::Rotate(float angleDegrees)
{
	ClearToBlack();
	return false;
}// Rotate


//////////////////////////////////////////////////////////////////////////////
//
//      Given a single RGBA pixel return, via the second argument, the RGB
//      equivalent composited with a black background.
//
///////////////////////////////////////////////////////////////////////////////
void TargaImage::RGBA_To_RGB(unsigned char *rgba, unsigned char *rgb)
{
	const unsigned char	BACKGROUND[3] = { 0, 0, 0 };

	unsigned char  alpha = rgba[3];

	if (alpha == 0)
	{
		rgb[0] = BACKGROUND[0];
		rgb[1] = BACKGROUND[1];
		rgb[2] = BACKGROUND[2];
	}
	else
	{
		float	alpha_scale = (float)255 / (float)alpha;
		int	val;
		int	i;

		for (i = 0 ; i < 3 ; i++)
		{
			val = (int)floor(rgba[i] * alpha_scale);
			if (val < 0)
				rgb[i] = 0;
			else if (val > 255)
				rgb[i] = 255;
			else
				rgb[i] = val;
		}
	}
}// RGA_To_RGB


///////////////////////////////////////////////////////////////////////////////
//
//      Copy this into a new image, reversing the rows as it goes. A pointer
//  to the new image is returned.
//
///////////////////////////////////////////////////////////////////////////////
TargaImage* TargaImage::Reverse_Rows(void)
{
	unsigned char   *dest = new unsigned char[width * height * 4];
	TargaImage	    *result;
	int 	        i, j;

	if (! data)
		return NULL;

	for (i = 0 ; i < height ; i++)
	{
		int in_offset = (height - i - 1) * width * 4;
		int out_offset = i * width * 4;

		for (j = 0 ; j < width ; j++)
		{
			dest[out_offset + j * 4] = data[in_offset + j * 4];
			dest[out_offset + j * 4 + 1] = data[in_offset + j * 4 + 1];
			dest[out_offset + j * 4 + 2] = data[in_offset + j * 4 + 2];
			dest[out_offset + j * 4 + 3] = data[in_offset + j * 4 + 3];
		}
	}

	result = new TargaImage(width, height, dest);
	delete[] dest;
	return result;
}// Reverse_Rows


///////////////////////////////////////////////////////////////////////////////
//
//      Clear the image to all black.
//
///////////////////////////////////////////////////////////////////////////////
void TargaImage::ClearToBlack()
{
	memset(data, 0, width * height * 4);
}// ClearToBlack


///////////////////////////////////////////////////////////////////////////////
//
//      Helper function for the painterly filter; paint a stroke at
// the given location
//
///////////////////////////////////////////////////////////////////////////////
void TargaImage::Paint_Stroke(const Stroke& s) {
	int radius_squared = (int)s.radius * (int)s.radius;
	for (int x_off = -((int)s.radius); x_off <= (int)s.radius; x_off++) {
		for (int y_off = -((int)s.radius); y_off <= (int)s.radius; y_off++) {
			int x_loc = (int)s.x + x_off;
			int y_loc = (int)s.y + y_off;
			// are we inside the circle, and inside the image?
			if ((x_loc >= 0 && x_loc < width && y_loc >= 0 && y_loc < height)) {
				int dist_squared = x_off * x_off + y_off * y_off;
				if (dist_squared <= radius_squared) {
					data[(y_loc * width + x_loc) * 4 + 0] = s.r;
					data[(y_loc * width + x_loc) * 4 + 1] = s.g;
					data[(y_loc * width + x_loc) * 4 + 2] = s.b;
					data[(y_loc * width + x_loc) * 4 + 3] = s.a;
				} else if (dist_squared == radius_squared + 1) {
					data[(y_loc * width + x_loc) * 4 + 0] = 
						(data[(y_loc * width + x_loc) * 4 + 0] + s.r) / 2;
					data[(y_loc * width + x_loc) * 4 + 1] = 
						(data[(y_loc * width + x_loc) * 4 + 1] + s.g) / 2;
					data[(y_loc * width + x_loc) * 4 + 2] = 
						(data[(y_loc * width + x_loc) * 4 + 2] + s.b) / 2;
					data[(y_loc * width + x_loc) * 4 + 3] = 
						(data[(y_loc * width + x_loc) * 4 + 3] + s.a) / 2;
				}
			}
		}
	}
}


///////////////////////////////////////////////////////////////////////////////
//
//      Build a Stroke
//
///////////////////////////////////////////////////////////////////////////////
Stroke::Stroke() {}

///////////////////////////////////////////////////////////////////////////////
//
//      Build a Stroke
//
///////////////////////////////////////////////////////////////////////////////
Stroke::Stroke(unsigned int iradius, unsigned int ix, unsigned int iy,
			   unsigned char ir, unsigned char ig, unsigned char ib, unsigned char ia) :
radius(iradius),x(ix),y(iy),r(ir),g(ig),b(ib),a(ia)
{
}

