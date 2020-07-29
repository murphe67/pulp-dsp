/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_mat_mult_stride_i32s_rv32im.c
 * Description:  32-bit strided matrix mulitplication kernel for RV32IM
 *
 * $Date:        14. July 2020
 * $Revision:    V0
 *
 * Target Processor: PULP cores
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Author: Tibor Schneider, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plp_math.h"

/**
  @ingroup BasicMatMultStride
 */

/**
  @defgroup BasicMatMultStrideKernels Strided Matrix Multiplication Kernels
  This module contains the kernels for strided matrix matrix multiplication.

  The Matrix Matrix Multiplication computes the product of two matrices with dimensions MxN and NxO.
  The first matrix is accessed row wise, the second column wise, all values form the first are
  multiplied with the values of the second and then sum of the result gives the value for the result
  matrix.

      'pDst[m,o] = pSrcA[m,0]*pSrcB[0,o] + pSrcA[m,1]*pSrcB[1,o] + ... + pSrcA[m,N-1]*pSrcB[N-1,o]`

  There are functions for integer 32- 16- and 8-bit data types. For lower precision integers (16-
  and 8-bit), functions exploiting SIMD instructions are provided.

  The naming scheme of the functions follows the following pattern (for example
  `plp_mat_mult_stride_i32s_xpulpv2`):

      `plp_<function name>_<data type><precision><method>_<isa_extension>`

  name          | description
  ------------- | ---------------------------------------------------------------------------
  function_name | `mat_mult`
  data type     | {`f`, `i`, `q`} respectively for floats, integers, fixed points
  precision     | {`32`, `16`, `8`} bits
  method        | {`s`, `v`, `p`} meaning scalar, vectorized (i.e. SIMD) and parallel, respectively
  isa_extension | {`rv32im`, `xpulpv2`} respectively for ibex and riscy

  The `strideX` argument tells how many elements are in between the start of each row of the matrix.
  In other words, it is the width of the original matrix. @ref groupMatrixStride
 */

/**
  @addtogroup BasicMatMultStrideKernels
  @{
 */

/**
  @brief Matrix multiplication of 32-bit integer matrices kernel for RV32IM extension.
  @param[in]  pSrcA     points to the first input matrix
  @param[in]  pSrcB     points to the second input matrix
  @param[in]  M         height of the first input matrix
  @param[in]  N         width of the first input matrix and hight of the second
  @param[in]  O         width of the second input matrix
  @param[in]  strideA   Stride of matrix A (elements between each row)
  @param[in]  strideB   Stride of matrix B (elements between each row)
  @param[in]  strideC   Stride of output matrix (elements between each row)
  @param[out] pDstC     points to the output matrix
  @return        none
 */

void plp_mat_mult_stride_i32s_rv32im(const int32_t *__restrict__ pSrcA,
                                     const int32_t *__restrict__ pSrcB,
                                     uint32_t M,
                                     uint32_t N,
                                     uint32_t O,
                                     uint32_t strideA,
                                     uint32_t strideB,
                                     uint32_t strideC,
                                     int32_t *__restrict__ pDstC) {

#define BASIC_VERSION // if used don't forget to also use the undefine at end of file
#ifdef BASIC_VERSION

    uint32_t m, n, o;

    for (m = 0; m < M; m++) {
        for (o = 0; o < O; o++) {
            int32_t sum = 0;
            for (n = 0; n < N; n++) {
                int32_t valA = (int32_t)pSrcA[m * strideA + n];
                int32_t valB = (int32_t)pSrcB[n * strideB + o];
                sum += valA * valB;
            }
            pDstC[m * strideC + o] = sum;
        }
    }

#else

    // TODO: Hackathon

#endif
#undef BASIC_VERSION
}

// #undef BASIC_VERSION

/**
   @} end of BasicDotProdKernels group
*/
