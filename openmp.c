#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <jpeglib.h>
#include <omp.h>

#define MAXMK 100

float dct_basis(int k, int n, int N) {
    return cos(M_PI * k * (2 * n + 1) / (2.0 * N));
}

void cepstralSmoothLine(unsigned char *in, unsigned char *out, int width, int mk, int bPP) {
    float G[MAXMK] = {0};

    for (int k = 0; k < mk; ++k) {
        for (int i = 0; i < width; ++i) {
            int x = in[i * bPP];
            G[k] += x * dct_basis(k, i, width);
        }
        G[k] *= (k == 0 ? 1.0 : sqrt(2.0)) / sqrt((double)width);
    }

    for (int j = 0; j < width; ++j) {
        float c = 0;
        for (int k = 0; k < mk; ++k) {
            c += G[k] * dct_basis(k, j, width);
        }
        c *= sqrt(2.0) / sqrt((double)width);
        c = fmax(0.0, fmin(255.0, c));
        out[j * bPP + 0] = (unsigned char)c;
        out[j * bPP + 1] = (unsigned char)c;
        out[j * bPP + 2] = (unsigned char)c;
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

    unsigned char *h_input = malloc(width * height * bPP);
    unsigned char *h_output = malloc(width * height * bPP);
    JSAMPROW row_pointer[1];

    for (int i = 0; i < height; i++) {
        row_pointer[0] = h_input + i * bytesPerLine;
        jpeg_read_scanlines(&in, row_pointer, 1);
    }
    jpeg_finish_decompress(&in);
    jpeg_destroy_decompress(&in);
    fclose(inFile);

    double start = omp_get_wtime();
    // Równoległe przetwarzanie wierszy
    #pragma omp parallel for
    for (int h = 0; h < height; ++h) {
        cepstralSmoothLine(
            h_input + h * bytesPerLine,
            h_output + h * bytesPerLine,
            width, mk, bPP
        );
    }
    double end = omp_get_wtime();
    printf("Czas wykonania OpenMP (DCT): %.6f sek\n", end - start);

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
