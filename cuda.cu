#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <jpeglib.h>
#include <cuda_runtime.h>

#define MAXMK 100
#define BLOCK_SIZE 1024

__device__ float dct_basis(int k, int n, int N) {
    return cosf(M_PI * k * (2 * n + 1) / (2.0f * N));
}

__global__ void cepstralSmoothKernel(unsigned char *in, unsigned char *out, int width, int height, int mk, int bPP) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // piksel w wierszu
    int y = blockIdx.y; // numer wiersza

    if (x >= width || y >= height) return;

    __shared__ float G[MAXMK];

    unsigned char *in_row = in + y * width * bPP;
    unsigned char *out_row = out + y * width * bPP;

    if (threadIdx.x < mk && threadIdx.y == 0) {
        G[threadIdx.x] = 0.0f;
        for (int i = 0; i < width; ++i) {
            int px = in_row[i * bPP];
            G[threadIdx.x] += px * dct_basis(threadIdx.x, i, width);
        }
        G[threadIdx.x] *= (threadIdx.x == 0 ? 1.0f : sqrtf(2.0f)) / sqrtf((float)width);
    }

    __syncthreads();

    float c = 0.0f;
    for (int k = 0; k < mk; ++k)
        c += G[k] * dct_basis(k, x, width);
    c *= sqrtf(2.0f) / sqrtf((float)width);
    c = fminf(255.0f, fmaxf(0.0f, c));
    out_row[x * bPP + 0] = (unsigned char)c;
    out_row[x * bPP + 1] = (unsigned char)c;
    out_row[x * bPP + 2] = (unsigned char)c;
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

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, width * height * bPP);
    cudaMalloc(&d_output, width * height * bPP);
    cudaMemcpy(d_input, h_input, width * height * bPP, cudaMemcpyHostToDevice);

    int threadsPerBlock = (width <= 1024) ? width : 1024;
    int blocksX = (width + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock);
    dim3 grid(blocksX, height);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cepstralSmoothKernel<<<grid, block>>>(d_input, d_output, width, height, mk, bPP);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Czas wykonania GPU (DCT): %.6f sek\n", milliseconds / 1000.0f);

    cudaMemcpy(h_output, d_output, width * height * bPP, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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
