
// Imports 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "wm.h"
#include "A_10.h"
#include "b_10.h"
#include "b_32.h"
#include "b_512.h"
#include "b_1024.h"
#include "X_512.h"
#include "X_1024.h"
#include "X_32.h"
#include "A_32.h"
#include "A_512.h"
#include "A_1024.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace std;
#define max_Threads 1024 // Max num of threads
#define N 32  // Order of the matrix have to change this every time you run a different size matrix

// Final version of convolution kernel
__global__ void convolutionKernelFinalVersion(unsigned char* input, unsigned inputSize, unsigned inputWidth,
	unsigned char* output, unsigned outputSize, unsigned outWidth,
	int kernelSize, int numThreads, float* w) {
    // Finding the offset for where the program is starting to do the convolution
	int rgbaOffset = (blockIdx.x * blockDim.x + threadIdx.x);
	printf("%d\n", rgbaOffset);
    // Looping to do the convolution
	for (int i = rgbaOffset; i < outputSize; i += numThreads) {
		unsigned colSmall = i % (4 * outWidth);
		unsigned rowSmall = i / (4 * outWidth);
		unsigned bRow = rowSmall * 4 * inputWidth;
		unsigned largeOffset = bRow + colSmall;
        // Doing the direct convolution in the 3x3, 5x5, 7x7 window
		float sumRGBA = 0.0;
		for (int j = 0; j < kernelSize; ++j) {
			for (int k = 0; k < kernelSize; ++k) {
				sumRGBA += input[largeOffset + 4 * k + 4 * inputWidth * j] * w[j * kernelSize + k];
			}
		}
        // Clamping the RBGA values
		if (sumRGBA > 255.0 || (i + 1) % 4 == 0) {
			sumRGBA = 255.0;
		}
		if (sumRGBA < 0.0) {
			sumRGBA = 0.0;
		}
        // Storing the R,G,B,A channels directly into the smaller image array indices
		output[i] = (unsigned char)((int)sumRGBA);
	}
}

// Parallelized convolution
__global__ void convolutionKernel1(unsigned char* inputImg, unsigned char* output, unsigned imgWidth, unsigned imgHeight, int numThreads, int kernelSize, int outputCounter, int rowNum) {
	// Thread offset to see where we are starting to convolute
	int rgbaOffset = (((blockIdx.x * blockDim.x) + threadIdx.x) * ((4 * imgWidth * imgHeight) / numThreads));
	// To determine which kernel size to use 
	if (kernelSize == 3) {
		// Convolution
		for (int c = rgbaOffset; c <= ((4 * imgWidth * imgHeight) / numThreads); c = c + 4) {

			// Offsets for all the R channels for the 3x3 blocks 
			unsigned p1R = c; //1
			unsigned p2R = p1R + 4; //2
			unsigned p3R = p2R + 4;
			unsigned p4R = p1R + (4 * imgWidth);
			unsigned p5R = p4R + 4;
			unsigned p6R = p5R + 4;
			unsigned p7R = p4R + (4 * imgWidth);
			unsigned p8R = p7R + 4;
			unsigned p9R = p8R + 4;

			// Extracting RGBA from the input image for the 3x3 convolution
			unsigned R[9] = { inputImg[p1R], inputImg[p2R], inputImg[p3R], inputImg[p4R], inputImg[p5R], inputImg[p6R], inputImg[p7R], inputImg[p8R], inputImg[p9R] };

			unsigned G[9] = { inputImg[p1R + 1], inputImg[p2R + 1], inputImg[p3R + 1], inputImg[p4R + 1], inputImg[p5R + 1], inputImg[p6R + 1], inputImg[p7R + 1], inputImg[p8R + 1], inputImg[p9R + 1] };

			unsigned B[9] = { inputImg[p1R + 2], inputImg[p2R + 2], inputImg[p3R + 2], inputImg[p4R + 2], inputImg[p5R + 2], inputImg[p6R + 2], inputImg[p7R + 2], inputImg[p8R + 2], inputImg[p9R + 2] };

			unsigned A[9] = { inputImg[p1R + 3], inputImg[p2R + 3], inputImg[p3R + 3], inputImg[p4R + 3], inputImg[p5R + 3], inputImg[p6R + 3], inputImg[p7R + 3], inputImg[p8R + 3], inputImg[p9R + 3] };

			// Weights matrix flattened
			float wArray[9] = { 1, 2, -1, 2, 0.25, -2, 1, -2, -1 };

			// tmp to store the calculations
			int sumR = 0;
			int sumG = 0;
			int sumB = 0;
			int sumA = 0;

			// Looping to conlvove and put into the copy image array
			for (int j = 0; j < 9; j++) {
				sumR += (wArray[j] * (int)R[j]);
				sumG += (wArray[j] * (int)G[j]);
				sumB += (wArray[j] * (int)B[j]);
				sumA += (wArray[j] * (int)A[j]);
			}

			// Making sure if it is negative it is set to 0 and if it is above 255 it is set to 255
			// R Channel
			if (sumR < 0) {
				sumR = 0;
			}
			else if (sumR > 255) {
				sumR = 255;
			}

			// G Channel
			if (sumG < 0) {
				sumG = 0;
			}
			else if (sumG > 255) {
				sumG = 255;
			}

			// B Channel
			if (sumB < 0) {
				sumB = 0;
			}
			else if (sumB > 255) {
				sumB = 255;
			}

			// A Channel
			if (sumA < 0) {
				sumA = 0;
			}
			else if (sumA > 255) {
				sumA = 255;
			}

			// Inputing the RGBA values into the new image array
			output[c] = (unsigned)sumR;
			output[c + 1] = (unsigned)sumG;
			output[c + 2] = (unsigned)sumB;
			output[c + 3] = (unsigned)sumA;

			// Making sure the output image is incrementing by 4 each time
			outputCounter = outputCounter + 4;

			// Checking to see if it hit the wall
			if (((c + 12) / rowNum) == (4 * (imgWidth))) {
                
                // To end the convolution algorithm once the top left corner of the kernel hit 2 rows before the full height
				printf("\nRow Number: %d\n", rowNum);
				if (rowNum == (imgHeight - 2)) {
					break;
				}
				rowNum++;
				c += 8;
			}
		}
	}
	else if (kernelSize == 5) {
	}
	else {
	}
}

// Parallelized matrix operations
__global__ void matrixInversionLSSKernel(double* input, double* b, double* x, int width)
{
	// Need to run with # of threads that is at least equal to or greater than the dimensionality or else the multiplication part would fail 
	// Row offset to where the threads are doing matrix multiplications with each row 
	int r = (blockIdx.x * blockDim.x) + threadIdx.x;
	//printf("\n%d\n", r);

	// The row number has to be less than the dimension of the matrix bc we dont want it to go out of bounds
	if (r < width) {
		double sum = 0;
			for (int k = 0; k < width; k++) {
				sum += input[r * width + k] * b[k];
			}
		// Put sum for each row in the designated row number but since the output is just an Nx1 matrix it would follow the normal array indices offsetted by 1 each time
		x[r] = sum;
	}
}

// Helper method to load correct kernel
float* loadKernel(int kernelSize) {
	// Mallocing for the image weights
	float* fw = (float*)malloc(kernelSize * kernelSize * sizeof(float));
	// Finding out which kernel size to use
	if (kernelSize == 3) {
		printf("Using kernel 3x3\n");
		float w3_ar[9] = { 1,2,-1, 2,0.25,-2,1,-2,-1 };
		for (int j = 0; j < kernelSize * kernelSize; j++) {
			fw[j] = w3_ar[j];
			//printf("%f\n", fw[j]);
		}
		return fw;
	}
	else if (kernelSize == 5) {
		printf("Using kernel 5x5\n");
		for (int i = 0; i < kernelSize; i++) {
			for (int j = 0; j < kernelSize; j++) {
				fw[i * kernelSize + j] = w5[i][j];
			}
		}
		return fw;
	}
	else if (kernelSize == 7) {
		printf("Using kernel 7x7\n");
		for (int i = 0; i < kernelSize; i++) {
			for (int j = 0; j < kernelSize; j++) {
				fw[i * kernelSize + j] = w7[i][j];
			}
		}
		return fw;
	}
	else {
		printf("The kernel size is invalid/n");
		return NULL;
	}
	return NULL;
}

void computeConvolution(char* input_filename, char* output_filename, int numThreads, int kernelSize)
{
	// For timing
	clock_t start, end;
	int duration;

	// Start timer
	start = clock();

	// Blocks variable
	int blocks;
	// Checks for num of threads and allocate appropriate number of blocks
	if ((numThreads % max_Threads) == 0) {
		blocks = (numThreads / 1024);
	}
	else if (numThreads <= max_Threads) {
		blocks = 1;
	}
	else if (numThreads <= 0) {
		printf("\nError: number of threads can not be negative!\n");
		return;
	}
	else {
		blocks = (numThreads / max_Threads) + 1;
	}
	// Checking our logic 
	printf("\nNumber of blocks: %d and threads: %d needed!\n", blocks, numThreads);
	// Definining image input variables
	unsigned err;
	unsigned imgWidth;
	unsigned imgHeight;
	// Host & Device variables
	unsigned char* inputImg;
	unsigned char* output;
	// Testing to load an image and see if any errors occured;
	err = lodepng_decode32_file(&inputImg, &imgWidth, &imgHeight, input_filename);
	if (err) {
		printf("error occurred %u: %s\n", err, lodepng_error_text(err));
	}
	// Output image width and height
	unsigned outputWidth = imgWidth - (kernelSize - 1);
	unsigned outputHeight = imgHeight - (kernelSize - 1);
	// Input image and output image size
	unsigned int inputSize = 4 * imgWidth * imgHeight;
	unsigned int outputSize = 4 * outputHeight * outputWidth;
    // Mallocing local memory for the output image array
    output = (unsigned char*)malloc(outputSize*sizeof(unsigned char));
    // Loading the correct kernel to use
    float *fw = loadKernel(kernelSize);
	// Allocating device memory
    unsigned char* cudaInput;
    unsigned char* cudaOutput;
	float* fWeightsCuda;
	cudaMalloc((void**)& cudaInput, inputSize * sizeof(unsigned char));
	cudaMalloc((void**)& cudaOutput, outputSize * sizeof(unsigned char));
	cudaMalloc((void**)& fWeightsCuda, kernelSize * kernelSize * sizeof(float));
	// Coping over data from cpu memory to GPU memory
	cudaMemcpy(cudaInput, inputImg, inputSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(fWeightsCuda, fw, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
	// Calling the GPU to parallelize our convolutions
	convolutionKernelFinalVersion <<< blocks, (numThreads/blocks) >>> (cudaInput, inputSize, imgWidth, cudaOutput, outputSize, outputWidth, kernelSize, numThreads, fWeightsCuda);
	// Copying from device memory back to cpu memory for the output image array
	cudaMemcpy(output, cudaOutput, outputSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // Display and save image to local folder
	lodepng_encode32_file(output_filename, output, outputWidth, outputHeight);
	free(inputImg);
	free(output);
    free(fw);
	cudaFree(cudaInput);
	cudaFree(cudaOutput);
    cudaFree(fWeightsCuda);
    printf("\nFinished the convolution!\n\n");

	// end time
	end = clock();
	duration = (end - start);
	printf("\nTime took to run: %d ms\n", duration);
}


// Function to get cofactor of A[p][q] in temp[][]. n is current 
// dimension of A[][] 
void getCofactor(double A[N][N], double temp[N][N], int p, int q, int n)
{
	int i = 0, j = 0;

	// Looping for each element of the matrix 
	for (int row = 0; row < n; row++)
	{
		for (int col = 0; col < n; col++)
		{
			//  Copying into temporary matrix only those element 
			//  which are not in given row and column 
			if (row != p && col != q)
			{
				temp[i][j++] = A[row][col];

				// Row is filled, so increase row index and 
				// reset col index 
				if (j == n - 1)
				{
					j = 0;
					i++;
				}
			}
		}
	}

}

/* Recursive function for finding determinant of matrix.
   n is current dimension of A[][]. */
int determinant(double A[N][N], int n)
{
	int D = 0; // Initialize result 

	//  Base case : if matrix contains single element 
	if (n == 1)
		return A[0][0];

	double temp[N][N]; // To store cofactors 

	int sign = 1;  // To store sign multiplier 

	 // Iterate for each element of first row 
	for (int f = 0; f < n; f++)
	{
		// Getting Cofactor of A[0][f] 
		getCofactor(A, temp, 0, f, n);
		D += sign * A[0][f] * determinant(temp, n - 1);

		// terms are to be added with alternate sign 
		sign = -sign;
	}

	return D;
}

// Function to get adjoint of A[N][N] in adj[N][N]. 
void adjoint(double A[N][N], double adj[N][N])
{
	if (N == 1)
	{
		adj[0][0] = 1;
		return;
	}

	// temp is used to store cofactors of A[][] 
	int sign = 1;
	double temp[N][N];

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			// Get cofactor of A[i][j] 
			getCofactor(A, temp, i, j, N);

			// sign of adj[j][i] positive if sum of row 
			// and column indexes is even. 
			sign = ((i + j) % 2 == 0) ? 1 : -1;

			// Interchanging rows and columns to get the 
			// transpose of the cofactor matrix 
			adj[j][i] = (sign) * (determinant(temp, N - 1));
		}
	}
}

// Function to calculate and store inverse, returns false if 
// matrix is singular 
bool inverse(double A[N][N], double inverse[N][N])
{
	// Find determinant of A[][] 
	int det = determinant(A, N);
	if (det == 0)
	{
		cout << "Singular matrix, can't find its inverse";
		return false;
	}

	// Find adjoint 
	double adj[N][N];
	adjoint(A, adj);
	printf("\nFinished adjoint\n");

	// Find Inverse using formula "inverse(A) = adj(A)/det(A)" 
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			inverse[i][j] = adj[i][j] / double(det);

	printf("\nFinished inverse\n");
	return true;
}


void GaussJordan(double matrix[N][N], int MatrixOrder, double inverse[N][N]) {

	double r = 0.000;
	//change the size of the matrix here
	double bigMatrix[32][32 * 2];
	// Create gauss jordan matrix: matrix | identity matrix
	for (int i = 0; i < MatrixOrder; i++) {
		for (int j = 0; j < MatrixOrder; j++) {
			bigMatrix[i][j] = matrix[i][j];
		}
	}
	for (int i = 0; i < MatrixOrder; i++) {
		for (int j = 0; j < MatrixOrder; j++) {
			if (i == j) {
				bigMatrix[i][j + MatrixOrder] = 1;
			}
			else {
				bigMatrix[i][j + MatrixOrder] = 0;
			}
		}
	}
	// Apply Gauss Jordan Elimination
	for (int i = 0; i < MatrixOrder; i++) {
		if (bigMatrix[i][i] == 0) {
			exit(0);
		}
		for (int j = 0; j < MatrixOrder; j++) {
			if (i != j) {
				r = bigMatrix[j][i] / bigMatrix[i][i];
				for (int k = 0; k < 2 * MatrixOrder; k++) {
					bigMatrix[j][k] = bigMatrix[j][k] - (bigMatrix[i][k] * r);
				}
			}
		}
	}

	// Row operation to make principal diagonal to 1
	for (int i = 0; i < MatrixOrder; i++) {
		double temp = bigMatrix[i][i];
		for (int j = 0; j < MatrixOrder * 2; j++) {
			bigMatrix[i][j] = bigMatrix[i][j] / temp;
		}
	}

	// Taking the correct values from the augmented matrix
	for (int i = 0; i < MatrixOrder; i++) {
		for (int j = MatrixOrder; j < MatrixOrder * 2; j++) {
			inverse[i][j] = bigMatrix[i][j];
		}
	}
}



// Generic function to display the matrix.  We use it to display 
// both adjoin and inverse. adjoin is integer matrix and inverse 
// is a float. 
template<class T>
void display(T A[N][N])
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			cout << A[i][j] << " ";
		cout << endl;
	}
}

void matrixMultiplication(double inverted[N][N], double b[N][1], int numThreads, int order) {

	// For timing
	clock_t start, end;
	int duration;

	// Start timer
	start = clock();

	// Blocks variable
	int blocks;
	// Checks for num of threads and allocate appropriate number of blocks
	if ((numThreads % max_Threads) == 0) {
		blocks = (numThreads / 1024);
	}
	else if (numThreads <= max_Threads) {
		blocks = 1;
	}
	else if (numThreads <= 0) {
		printf("\nError: number of threads can not be negative!\n");
		return;
	}
	else {
		blocks = (numThreads / max_Threads) + 1;
	}
	// Checking our logic 
	printf("\nNumber of blocks: %d and threads: %d needed!\n", blocks, numThreads);

	// Allocate local memory for the flattened matrices of the original ones
	double* flattenedINV = (double*)malloc(N * N * sizeof(double));
	double* flattenedB = (double*)malloc(N * 1 * sizeof(double));
	double* output = (double*)malloc(N * 1 * sizeof(double));

	// Flattening of the matrices
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			flattenedINV[j * N + i] = inverted[j][i];
			//printf(" %f ", flattenedINV[j * N + i]);
		}
	}

	for (int j = 0; j < N; j++) {
		flattenedB[j] = b[j][0];
	}

	// Allocate device memory
	double* cuda_FlattenedINV;
	double* cuda_FlattenedB;
	double* x;
	cudaMalloc((void**)& cuda_FlattenedB, N*1*sizeof(double));
	cudaMalloc((void**)& cuda_FlattenedINV, N*N*sizeof(double));
	cudaMalloc((void**)& x, N * 1 * sizeof(double));

	// copy memeory from 
	cudaMemcpy(cuda_FlattenedB, flattenedB, N*1*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_FlattenedINV, flattenedINV, N*N*sizeof(double), cudaMemcpyHostToDevice);

	// run in GPU
	matrixInversionLSSKernel <<< blocks, (numThreads / blocks) >>> (cuda_FlattenedINV, cuda_FlattenedB, x, order);

	// copy back to cpu
	cudaMemcpy(output, x, N*1*sizeof(double), cudaMemcpyDeviceToHost);

	// display output
	printf("\nOutput\n");
	for (int i = 0; i < N; i++) {
		printf("\n%lf", output[i]);
	}
	printf("\n");
	// end time
	end = clock();
	duration = (end - start);
	printf("\nTime took to run: %d ms\n", duration);
}


int main(void) {
 //   // Run with minimum 32 threads or else the algorithm times out before the whole convolution is finished
	//// For 3x3 run with minimum 16
	//// For 5x5 run with minimum 32
	//// For 7x7 run with minimum 64
	//int kernelSize = 7;
	//int numThreads = 1024;
 //   char *input = "test.png";
 //   char *output = "convolved7x7.png";
	//computeConvolution(input, output, numThreads, kernelSize);

	// Gauss Jordan Elimination
	// Output matrix
	// If the matrix is of order N then run with minimum N threads or else the matrix multiplication part will not compute the entire X column vector
	// It will compute most of it but then have little bits left over
	// Before running our code you need to make sure N is changed to the appropriate order that is equal to the order of the matrix you are test with 
	/*double inv[N][N];
	cout << "\nThe Inverse is :\n";
	GaussJordan(A_10, N, inv);
	display(inv);*/

	/*if (inverse(A_10, inv))
		display(inv);*/

	/*(invertibleMatrix, bVector, numThreads to run with, order of matrix)*/
	matrixMultiplication(A_32, X_32, 2048, N);
}


