#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <jpeglib.h>
#include <cuda_runtime.h>

#define MAXMK 100
#define BLOCK_SIZE 720

__device__ float dct_basis(int k, int n, int N) {
    return cosf(M_PI * k * (2 * n + 1) / (2.0f * N));
}

__global__ void cepstralSmoothLine(unsigned char *in, unsigned char *out, int width, int mk, int bPP) {
    int h = blockIdx.x;
    in += h * width * bPP;
    out += h * width * bPP;

    __shared__ float G[MAXMK];
    float c = 0.0f;
    int tid = threadIdx.x;

    if (tid < mk) {
        G[tid] = 0.0f;
        for (int i = 0; i < width; i++) {
            int x = in[i * bPP];
            G[tid] += x * dct_basis(tid, i, width);
        }
        G[tid] *= (tid == 0 ? 1.0f : sqrtf(2.0f)) / sqrtf((float)width);
    }
    __syncthreads();

    if (tid < width) {
        for (int k = 0; k < mk; ++k)
            c += G[k] * dct_basis(k, tid, width);
        c *= sqrtf(2.0f) / sqrtf((float)width);
        c = fmaxf(0.0f, fminf(255.0f, c));
        out[tid * bPP + 0] = (unsigned char)c;
        out[tid * bPP + 1] = (unsigned char)c;
        out[tid * bPP + 2] = (unsigned char)c;
    }
}

int main(int argc, char **argv) {
    char *inputFile = (argc > 1) ? argv[1] : (char *)"oko_R.jpg";
    char *outputFile = (argc > 2) ? argv[2] : (char *)"oko_D.jpg";
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

    dim3 block(BLOCK_SIZE);
    dim3 grid(height);
    //cepstralSmoothLine<<<grid, block>>>(d_input, d_output, width, mk, bPP);
    //cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cepstralSmoothLine<<<grid, block>>>(d_input, d_output, width, mk, bPP);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("Czas wykonania GPU (DCT): %.3f ms\n", milliseconds);
    printf("Czas wykonania GPU (DCT): %.6f sek\n", milliseconds / 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_output, d_output, width * height * bPP, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

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
