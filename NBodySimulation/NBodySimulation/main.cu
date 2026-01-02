/**
 * main.cu: Main file for the implementation of a fast N-body simulation.
 * Based on chapter 31 of GPU Gems 3: https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda.
 *
 * @author Mars Semenova
 * @date Dec. 30, 2025
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "util/book.h"
#include "util/cpu_anim_points.h"
#include "NBodyInit.h"
#include <stdio.h>

 // params
#define N 64
#define p 16 // TODO: how to determine?
#define EPS2 pow(1e-20, 2)

struct DataBlock {
    float4* dev_pos;
    float4* dev_a;
    CPUAnimPoints* ptsAnim;
};
/**
 * Compute the interaction of a body with another body to
 * calculate the updated acceleration of the first body.
 *
 * @param bi - First body represented by a float4 (position, mass).
 * @param bj - Second body represented by a float4 (position, mass).
 * @param ai - Acceleration (float3) of the first body.
 * @return Updated acceleration (float3) of the first body.
 */
__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai) {
  float3 r;
  // r_ij
  r.x = bj.x - bi.x;
  r.y = bj.y - bi.y;
  r.z = bj.z - bi.z;
  if (r.x == 0 && r.y == 0 && r.z == 0) { // TODO: is this correct? I had to add this to avoid division by inf but this isn't in the ch 31 code :')
      return ai;
  }
  // distSqr = dot(r_ij, r_ij) + EPS^2
  float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
  // invDistCube =1/distSqr^(3/2)
  float distSixth = distSqr * distSqr * distSqr;
  float invDistCube = 1.0f/sqrtf(distSixth);
  // s = m_j * invDistCube
  float s = bj.w * invDistCube;
  // a_i =  a_i + s * r_ij
  ai.x += r.x * s;
  ai.y += r.y * s;
  ai.z += r.z * s;
  return ai;
}

/**
 * Calculate the updated acceleration of a body with all other bodies in its tile.
 *
 * @param myPosition Position of the body.
 * @param accel Cumulative acceleration.
 * @return Updated acceleration.
 */
__device__ float3 tile_calculation(float4 myPosition, float3 accel) {
  int i;
  extern __shared__ float4 shPosition[];
  for (i = 0; i < blockDim.x; i++) {
    accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
  }
  return accel;
}

/**
 * Kernel executed by a thread block with P threads to
 * compute the acceleration of P bodies per block after
 * interaction with all N bodies.
 *
 * @param devX - Pointer to global device memory for the positions of all bodies.
 * @param devA - Pointer to global device memory for the acceleration of all bodies.
 */
__global__ void calculate_forces(void* devX, void* devA) {
    extern __shared__ float4 shPosition[];
    float4* globalX = (float4*)devX;
    float4* globalA = (float4*)devA;
    float4 myPosition;
    int i, tile;
    float3 acc = { 0.0f, 0.0f, 0.0f };
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    myPosition = globalX[gtid];
    for (i = 0, tile = 0; i < N; i += p, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        shPosition[threadIdx.x] = globalX[idx];
        __syncthreads();
        acc = tile_calculation(myPosition, acc);
        __syncthreads();
    }
    // Save the result in global memory for the integration step.
    float4 acc4 = { acc.x, acc.y, acc.z, 0.0f };
    globalA[gtid] = acc4;
}

/**
 * Animation helper method from Cuda by Example's support files: https://github.com/tpoisot/CUDA-training/tree/master/utils/cuda_by_example/common.
 */
void anim_gpu(DataBlock* d, int ticks) {
    CPUAnimPoints* ptsAnimator = d->ptsAnim;

    // call kernel
    calculate_forces <<<(N / p), p >>> (d->dev_pos, d->dev_a);

    // retrieve resultes
    float4* res_a = (float4*)malloc(N * sizeof(float4));
    HANDLE_ERROR(cudaMemcpy(res_a, d->dev_a, N * sizeof(float4), cudaMemcpyDeviceToHost));

    // copy data to animator (TODO: overwrite points instead, temp sol to show results)
    ptsAnimator->copyA(ptsAnimator->a, res_a, &(ptsAnimator->init));
}

/**
 * Animation helper method from Cuda by Example' support files: https://github.com/tpoisot/CUDA-training/tree/master/utils/cuda_by_example/common.
 */
void anim_exit(DataBlock* d) {
    cudaFree(d->dev_pos);
    cudaFree(d->dev_a);
}

int main(void) {
    // gen data
    int err;
    double* r = NULL, *v = NULL, *a = NULL, *m = NULL;
    err = allocData3N_NB(N, &r);
    err |= allocData3N_NB(N, &v);
    err |= allocData3N_NB(N, &a);
    err |= allocDataN_NB(N, &m);
    if (err) {
        printf("Could not alloc data for N=%ld, err:%d\n", N, err);
        exit(0);
    }

    err = initData_NB(N, -1, r, v, a, m);
    if (!err) {
        printf("Could not initialize data for N=%ld, err:%d\n", N, err);
        exit(0);
    }

    // copy gened data into temp arrays
    float4* temp_pos = (float4*)malloc(N* sizeof(float4));
    float4* temp_a = (float4*)malloc(N * sizeof(float4));

    for (int x = 0; x < N; x++) {
        temp_pos[x] = make_float4(r[3*x], r[3*x + 1], r[3*x + 2], m[x]);
        temp_a[x] = make_float4(a[3 * x], a[3 * x + 1], a[3 * x + 2], 0.0);
    }

    // allocate data
    DataBlock data;
    double range = computeDomainSize_NB(N, r);
    CPUAnimPoints ptsAnimator(N, range, temp_pos, &data);
    data.ptsAnim = &ptsAnimator;
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_pos, N * sizeof(float4)));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_a, N * sizeof(float4)));

    // copy data to device
    HANDLE_ERROR(cudaMemcpy(data.dev_pos, temp_pos, N * sizeof(float4), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(data.dev_a, temp_a, N * sizeof(float4), cudaMemcpyHostToDevice));
    free(temp_pos);
    free(temp_a);

    // enter loop
    ptsAnimator.anim_and_exit((void (*)(void*, int)) anim_gpu, (void (*)(void*)) anim_exit);
}