/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_mat_fill_stride_i16s_rv32im.c
 * Description:  16-bit strided matrix fill kernel for RV32IM
 *
 * $Date:        17. July 2020
 * $Revision:    V0
 *
 * Target Processor: PULP cores
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and Ubiversity of Bologna. All rights reserved.
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
  @ingroup MatFillStride
 */

/**
  @defgroup MatFillStrideKernels Strided Matrix Fill Kernels

  There are functions for integer 32- 16- and 8-bit data types, as well as for floating-point. The
  naming scheme of the functions follows the following pattern (for example
  `plp_mat_fill_stride_i32s_xpulpv2`):

      `plp_<function name>_<data type><precision><method>_<isa_extension>`

  name          | description
  ------------- | ---------------------------------------------------------
  function_name | `mat_fill_stride`
  data type     | {f, i, q} respectively for floats, integers, fixed points
  precision     | {32, 16, 8} bits
  method        | {`s`, `v`, `p`} meaning scalar, vectorized (i.e. SIMD) and parallel, respectively
  isa_extension | {`rv32im`, `xpulpv2`} respectively for ibex and riscy

  The `stride` argument tells how many elements are in between the start of each row of the matrix.
  In other words, it is the width of the original matrix. @ref groupMatrixStride
 */

/**
  @addtogroup MatFillStrideKernels
  @{
 */

/**
  @brief      Fill an MxN strided 16-bit integers matrix on RV32IM
  @param[in]  M      Height of the matrix
  @param[in]  N      Width of the matrix
  @param[in]  stride Stride of the matrix (elements between each row)
  @param[out] pSrc   Points to the output matrix
  @return     none
 */

void plp_mat_fill_stride_i16s_rv32im(
    uint32_t M, uint32_t N, uint32_t stride, int16_t value, int16_t *__restrict__ pDst) {

#define BASIC_VERSION // if used don' forget to also use undefine at end of file
#ifdef BASIC_VERSION

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            pDst[m * stride + n] = value;
        }
    }

#else

    // TODO: Hackathon

#endif
#undef BASIC_VERSION
}

/**
   @} end of MatFillStrideKernels group
*/
