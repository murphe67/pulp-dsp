/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_std_f32.c
 * Description:  Standard deviation value of a 32-bit floating point vector glue code
 *
 * $Date:        29.06.2020
 *
 * Target Processor: PULP cores
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Author: Moritz Scherer, ETH Zurich
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
   @ingroup groupStats
*/

/**
   @defgroup std Std
*/

/**
   @addtogroup std
   @{
*/

/**
   @brief         Glue code for standard deviation value of a 32-bit floating point vector.
   @param[in]     pSrc       points to the input vector
   @param[in]     blockSize  number of samples in input vector
   @param[out]    pRes    standard value returned here
   @return        none
 */

void plp_std_f32(const float *__restrict__ pSrc,
                 uint32_t blockSize,
                 float *__restrict__ pRes) {

    if (rt_cluster_id() == ARCHI_FC_CID) {
      *pRes = 0.0f;
    } else {
      plp_std_f32s_xpulpv2(pSrc, blockSize, pRes);
    }
}

/**
  @} end of mean group
 */
