
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include <stdio.h>


__global__ void myKernel(char* inputImg, char* imgOut) 
{
	printf("\nStarting computations on GPU..... :)\n");
	// Process the image into gray scale
	processImage(inputImg, imgOut);
}

// Rectify the image into the desired form them create a new holder image to hold the rectified one
void rectifyImage(char* input_filename, char* output_filename) {

	// Defining variables 
	unsigned err;
	unsigned char* inputImg;
	unsigned char* newImg;
	unsigned imgWidth, imgHeight;

	// Testing to load an image and see if any errors occured;
	err = lodepng_decode32_file(&inputImg, &imgWidth, &imgHeight, input_filename);
	if (err) {
		printf("error occurred %u: %s\n", err, lodepng_error_text(err));
	}
	else {
		//newImg = (unsiged char*)malloc
	}


}

// Converts to gray scale image
void processImage(char* input_filename, char* output_filename)
{
	unsigned error;
	unsigned char* image, * new_image;
	unsigned width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	new_image = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));

	// process image
	unsigned char value;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			value = image[4 * width * i + 4 * j];

			new_image[4 * width * i + 4 * j + 0] = value; // R
			new_image[4 * width * i + 4 * j + 1] = value; // G
			new_image[4 * width * i + 4 * j + 2] = value; // B
			new_image[4 * width * i + 4 * j + 3] = image[4 * width * i + 4 * j + 3]; // A
		}
	}
	lodepng_encode32_file(output_filename, new_image, width, height);
	free(image);
	free(new_image);
}

// Start of program
int main(void) {

	// Image names
	char* inputImg = "./jag.png";
	char* imgOut = "jagGrayScale.png";
	// Host code (CPU) calling device code to run on GPU
	myKernel <<<1, 1 >>> (inputImg, imgOut);
	return 0;
}


