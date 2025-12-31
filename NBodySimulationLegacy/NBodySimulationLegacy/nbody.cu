/**
 * nbody.cu: Main file for the implementation of a fast N-body simulation.
 * Based on chapter 31 of GPU Gems 3: https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda.
 *
 * @author Mars Semenova
 * @date Dec. 30, 2025
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util/book.h"
#include "util/cpu_anim.h"
#include "NBodyInit.h"

 // params
#define N 8
#define P 8

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
  extern __shared__ float4[] shPosition;
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
__global__ void calculate_forces(void *devX, void *devA) {
  extern __shared__ float4[] shPosition;
  float4 *globalX = (float4 *)devX;
  float4 *globalA = (float4 *)devA;
  float4 myPosition;
  int i, tile;
  float3 acc = {0.0f, 0.0f, 0.0f};
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
   float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
   globalA[gtid] = acc4;
}

int main(void) {

}