#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <jpeglib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#define BLOCK_SIZE 1024

__global__ void applyCepstralSmoothing(cufftComplex* freqData, int width, int mk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width / 2 + 1) return;

    if (idx >= mk) {
        freqData[idx].x = 0.0f;
        freqData[idx].y = 0.0f;
    }
}

__global__ void normalizeAndCopy(unsigned char* output, float* spatialData, int width, int height, int bPP) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int out_idx = (y * width + x) * bPP;

    float val = spatialData[idx] / width;
    val = fminf(255.0f, fmaxf(0.0f, val));

    output[out_idx + 0] = (unsigned char)val;
    output[out_idx + 1] = (unsigned char)val;
    output[out_idx + 2] = (unsigned char)val;
}

int main(int argc, char **argv) {
    char *inputFile = (argc > 1) ? argv[1] : (char *)"input.jpg";
    char *outputFile = (argc > 2) ? argv[2] : (char *)"output.jpg";
    int mk = (argc > 3) ? atoi(argv[3]) : 10;

    struct jpeg_decompress_struct in;
    struct jpeg_error_mgr jInErr;
    FILE *inFile = fopen(inputFile, "rb");
    if (!inFile) { printf("Can't open input JPEG\n"); return 1; }

    in.err = jpeg_std_error(&jInErr);
    jpeg_create_decompress(&in);
    jpeg_stdio_src(&in, inFile);
    jpeg_read_header(&in, TRUE);
    jpeg_start_decompress(&in);

    int width = in.output_width;
    int height = in.output_height;
    int bPP = in.num_components;
    int bytesPerLine = width * bPP;

    unsigned char *h_input = (unsigned char *)malloc(width * height * bPP);
    unsigned char *h_output = (unsigned char *)malloc(width * height * bPP);
    JSAMPROW row_pointer[1];

    for (int i = 0; i < height; i++) {
        row_pointer[0] = h_input + i * bytesPerLine;
        jpeg_read_scanlines(&in, row_pointer, 1);
    }
    jpeg_finish_decompress(&in);
    jpeg_destroy_decompress(&in);
    fclose(inFile);

    float *h_gray = (float *)malloc(sizeof(float) * width * height);
    for (int i = 0; i < width * height; i++) {
        h_gray[i] = (float)h_input[i * bPP];
    }

    float *d_gray;
    cufftComplex *d_freq;
    cudaMalloc(&d_gray, sizeof(float) * width * height);
    cudaMalloc(&d_freq, sizeof(cufftComplex) * height * (width / 2 + 1));

    cudaMemcpy(d_gray, h_gray, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlanMany(&plan, 1, &width,
                  NULL, 1, width,
                  NULL, 1, width / 2 + 1,
                  CUFFT_R2C, height);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    cufftExecR2C(plan, d_gray, d_freq);

    int threads = 256;
    int blocks = (width / 2 + 1 + threads - 1) / threads;
    for (int y = 0; y < height; y++) {
        applyCepstralSmoothing<<<blocks, threads>>>(d_freq + y * (width / 2 + 1), width, mk);
    }

    cufftHandle iplan;
    cufftPlanMany(&iplan, 1, &width,
                  NULL, 1, width / 2 + 1,
                  NULL, 1, width,
                  CUFFT_C2R, height);

    cufftExecC2R(iplan, d_freq, d_gray);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Czas wykonania GPU (CUFFT + smoothing): %.6f ms\n", milliseconds);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    unsigned char *d_output;
    cudaMalloc(&d_output, width * height * bPP);

    normalizeAndCopy<<<gridSize, blockSize>>>(d_output, d_gray, width, height, bPP);

    cudaMemcpy(h_output, d_output, width * height * bPP, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cufftDestroy(iplan);
    cudaFree(d_gray);
    cudaFree(d_freq);
    cudaFree(d_output);
    free(h_gray);

    struct jpeg_compress_struct out;
    struct jpeg_error_mgr jOutErr;
    FILE *outFile = fopen(outputFile, "wb");
    if (!outFile) { printf("Can't open output JPEG\n"); return 1; }

    out.err = jpeg_std_error(&jOutErr);
    jpeg_create_compress(&out);
    jpeg_stdio_dest(&out, outFile);
    out.image_width = width;
    out.image_height = height;
    out.input_components = bPP;
    out.in_color_space = JCS_RGB;
    jpeg_set_defaults(&out);
    jpeg_start_compress(&out, TRUE);

    for (int i = 0; i < height; i++) {
        row_pointer[0] = h_output + i * bytesPerLine;
        jpeg_write_scanlines(&out, row_pointer, 1);
    }

    jpeg_finish_compress(&out);
    jpeg_destroy_compress(&out);
    fclose(outFile);

    free(h_input);
    free(h_output);

    return 0;
}