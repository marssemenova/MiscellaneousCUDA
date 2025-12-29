/**
 * main.cu: Main file for the implementation of a heat transfer simulation.
 * Based on chapter 7 of CUDA by Example: https://edoras.sdsu.edu/~mthomas/docs/cuda/cuda_by_example.book.pdf.
 *
 * @author Mars Semenova
 * @date Dec. 28, 2025
 */

#include "texture_indirect_functions.h"
#include "texture_types.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util/book.h"
#include "util/cpu_anim.h"

 // params
#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

// struct used by update routine
struct DataBlock {
    unsigned char* output_bitmap;
    float* dev_inSrc;
    float* dev_outSrc;
    float* dev_constSrc;
    CPUAnimBitmap* bitmap;
    cudaEvent_t start, stop;
    float totalTime;
    float frames;
};

// GPU in/out buffers
texture<float, 2> texConstSrc;
texture<float, 2> texIn;
texture<float, 2>  texOut;

/**
 * Copy the initial values of the heaters to the updated cells to overwrite any
 * updates to the heater cells (assumption they remain const).
 *
 * @param grid - Pointer to the grid to update.
 */
__global__ void overwrite_heater_cells_kernel(float* grid) {
    // map to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex2D(texConstSrc, x, y);
    if (c != 0) {
        grid[offset] = c;
    }
}

/**
 * Update the grid based on the heat transfer equation.
 *
 * @param dest - Pointer to estination of output.
 * @param destOut - Bool flag that indicates which buffer to use as input + which to use as output.
 */
__global__ void update_grid_kernel(float* dest, bool destOut) {
    // map to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float top, left, curr, right, bot;
    if (destOut) {
        top = tex2D(texIn, x, y - 1);
        left = tex2D(texIn, x - 1, y);
        curr = tex2D(texIn, x, y);
        right = tex2D(texIn, x + 1, y);
        bot = tex2D(texIn, x, y + 1);
    }
    else {
        top = tex2D(texOut, x, y - 1);
        left = tex2D(texOut, x - 1, y);
        curr = tex2D(texOut, x, y);
        right = tex2D(texOut, x + 1, y);
        bot = tex2D(texOut, x, y + 1);
    }
    dest[offset] = curr + SPEED * (top + bot + right + left - 4 * curr);
}

/**
 * Animation helper method from Cuda by Example' support files: https://github.com/tpoisot/CUDA-training/tree/master/utils/cuda_by_example/common.
 */
void anim_gpu(DataBlock* d, int ticks) {
    HANDLE_ERROR(cudaEventRecord(d->start, 0));
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    CPUAnimBitmap* bitmap = d->bitmap;
    volatile bool dstOut = true;
    for (int i = 0; i < 90; i++) {
        float *in, *out;
        if (dstOut) {
            in = d->dev_inSrc;
            out = d->dev_outSrc;
        }
        else {
            out = d->dev_inSrc;
            in = d->dev_outSrc;
        }
        overwrite_heater_cells_kernel<<<blocks, threads >>> (in);
        update_grid_kernel<<<blocks, threads >>> (out, dstOut);
        dstOut = !dstOut;
    }
    float_to_color << <blocks, threads >> > (d->output_bitmap, d->dev_inSrc);
    HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaEventRecord(d->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(d->stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    d->totalTime += elapsedTime;
    ++d->frames;
}

/**
 * Animation helper method from Cuda by Example' support files: https://github.com/tpoisot/CUDA-training/tree/master/utils/cuda_by_example/common.
 */
void anim_exit(DataBlock* d) {
    cudaUnbindTexture(texIn);
    cudaUnbindTexture(texOut);
    cudaUnbindTexture(texConstSrc);
    cudaFree(d->dev_inSrc);
    cudaFree(d->dev_outSrc);
    cudaFree(d->dev_constSrc);
    HANDLE_ERROR(cudaEventDestroy(d->start));
    HANDLE_ERROR(cudaEventDestroy(d->stop));
}

int main(void) {
    // set data
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    HANDLE_ERROR(cudaEventCreate(&data.start));
    HANDLE_ERROR(cudaEventCreate(&data.stop));
    int imageSize = bitmap.image_size();
    HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, imageSize));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, imageSize));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, imageSize));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, imageSize));

    // bind textures
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    HANDLE_ERROR(cudaBindTexture2D(NULL, texConstSrc, data.dev_constSrc, desc, DIM, DIM, sizeof(float) * DIM));

    HANDLE_ERROR(cudaBindTexture2D(NULL, texIn, data.dev_inSrc, desc, DIM, DIM, sizeof(float) * DIM));

    HANDLE_ERROR(cudaBindTexture2D(NULL, texOut, data.dev_outSrc, desc, DIM, DIM, sizeof(float) * DIM));

    // init the const data
    float* temp = (float*)malloc(imageSize);
    for (int i = 0; i < DIM * DIM; i++) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x > 300) && (x < 600) && (y > 310) && (y < 601)) {
            temp[i] = MAX_TEMP;
        }
    }
    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
    temp[DIM * 700 + 100] = MIN_TEMP;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;
    for (int y = 800; y < 900; y++) {
        for (int x = 400; x < 500; x++) {
            temp[x + y * DIM] = MIN_TEMP;
        }
    }

    HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, imageSize, cudaMemcpyHostToDevice));

    for (int y = 800; y < DIM; y++) {
        for (int x = 0; x < 200; x++) {
            temp[x + y * DIM] = MAX_TEMP;
        }
    }

    HANDLE_ERROR(cudaMemcpy(data.dev_inSrc, temp, imageSize, cudaMemcpyHostToDevice));

    free(temp);

    bitmap.anim_and_exit((void (*)(void*, int)) anim_gpu, (void (*)(void*)) anim_exit);
}