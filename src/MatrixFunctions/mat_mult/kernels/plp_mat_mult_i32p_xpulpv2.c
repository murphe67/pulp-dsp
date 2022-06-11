/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_mat_mult_i32p_xpulpv2.c
 * Description:  3parallel 2-bit integer matrix multiplication for XPULPV2
 *
 * $Date:        18. July 2019
 * $Revision:    V0
 *
 * Target Processor: PULP cores
 * ===================================================================== */
/*
 * Copyright (C) 2019 ETH Zurich and University of Bologna.
 *
 * Author: Tom Kuchler, ETH Zurich
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
  @ingroup BasicMatMult
 */

/**
  @addtogroup BasicMatMultKernels
  @{
 */

/**
  @brief Parallel matrix multiplication of 32-bit integer matrices kernel for XPULPV2 extension.
  @param[in]  args  pointer to plp_mat_mult_instance_i32 struct initialized by
                    plp_mat_mult_i32_parallel
  @return     none
 */

// define BASIC_VERSION // if used don't forget to also use the undefine at end of file

#ifdef BASIC_VERSION

void plp_mat_mult_i32p_xpulpv2(void *args) {
    plp_mat_mult_instance_i32 *arguments = (plp_mat_mult_instance_i32 *)args;
    const int32_t *__restrict__ pSrcA = arguments->pSrcA;
    const int32_t *__restrict__ pSrcB = arguments->pSrcB;
    uint32_t M = arguments->M;
    uint32_t N = arguments->N;
    uint32_t O = arguments->O;
    uint32_t nPE = arguments->nPE;
    int32_t *__restrict__ pDstC = arguments->pDstC;

    uint32_t i; // loop counter
    uint32_t j; // loop counter
    uint32_t k; // loop counter

    int core_id = hal_core_id();
    int step = (M - 1 + nPE) / nPE;
    uint32_t START = step * core_id;
    uint32_t END = (core_id != hal_cl_nb_pe_cores() - 1) ? START + step : M;
    // printf("core id: %i, start: %i, end: %i\n", core_id, START, END);
    for (i = START; i < END; i++) {
        for (k = 0; k < O; k++) {
            int32_t sum = 0;
            for (j = 0; j < N; j++) {
                sum = sum + pSrcA[i * N + j] * pSrcB[j * O + k];
            }
            pDstC[i * O + k] = sum;
        }
    }

    hal_team_barrier();
}

#else

void plp_mat_mult_i32p_xpulpv2(void *args) {
    plp_mat_mult_instance_i32 *arguments = (plp_mat_mult_instance_i32 *)args;
    const int32_t *__restrict__ pSrcA = arguments->pSrcA;
    const int32_t *__restrict__ pSrcB = arguments->pSrcB;
    uint32_t M = arguments->M;
    uint32_t N = arguments->N;
    uint32_t O = arguments->O;
    uint32_t nPE = arguments->nPE;
    int32_t *__restrict__ pDstC = arguments->pDstC;

    uint32_t i = 0; // loop counter for M
    uint32_t j = 0; // loop counter for N
    uint32_t k = 0; // loop counter for O

    int core_id = hal_core_id();

    //printf("core id: %i, start: %i, end: %i\n", core_id, START, END);

    for (k = core_id; k < O / 2; k += nPE) {
        for (i = 0; i < M / 4; i++) {
            int32_t sum00 = 0;
            int32_t sum01 = 0;

            int32_t sum10 = 0;
            int32_t sum11 = 0;

            int32_t sum20 = 0;
            int32_t sum21 = 0;

            int32_t sum30 = 0;
            int32_t sum31 = 0;

            for (j = 0; j < N; j++) {
                int32_t AVal0 = pSrcA[i * 4 * N + (j)];
                int32_t AVal1 = pSrcA[i * 4 * N + N + (j)];

                int32_t AVal2 = pSrcA[i * 4 * N + (2 * N) + (j)];
                int32_t AVal3 = pSrcA[i * 4 * N + (3 * N) + (j)];

                int32_t BVal0 = pSrcB[j * O + (k * 2)];
                int32_t BVal1 = pSrcB[j * O + (k * 2 + 1)];

                sum00 = sum00 + AVal0 * BVal0;
                sum01 = sum01 + AVal0 * BVal1;

                sum10 = sum10 + AVal1 * BVal0;
                sum11 = sum11 + AVal1 * BVal1;

                sum20 = sum20 + AVal2 * BVal0;
                sum21 = sum21 + AVal2 * BVal1;

                sum30 = sum30 + AVal3 * BVal0;
                sum31 = sum31 + AVal3 * BVal1;
            }

            pDstC[(i * 4) * O + k * 2] = sum00;
            pDstC[(i * 4) * O + k * 2 + 1] = sum01;

            pDstC[(i * 4 + 1) * O + k * 2] = sum10;
            pDstC[(i * 4 + 1) * O + k * 2 + 1] = sum11;

            pDstC[(i * 4 + 2) * O + k * 2] = sum20;
            pDstC[(i * 4 + 2) * O + k * 2 + 1] = sum21;

            pDstC[(i * 4 + 3) * O + k * 2] = sum30;
            pDstC[(i * 4 + 3) * O + k * 2 + 1] = sum31;
        }

        i = i*4;
        for(i=i;i<M;i++){
            int32_t sum0 = 0;
            int32_t sum1 = 0;
            for(j = 0; j < N; j++){
                int32_t AVal = pSrcA[i * N + (j)];

                int32_t BVal0 = pSrcB[j * O + (k * 2)];
                int32_t BVal1 = pSrcB[j * O + (k * 2 + 1)];

                sum0 = sum0 + AVal * BVal0;
                sum1 = sum1 + AVal * BVal1; 
            }
            pDstC[i * O + k * 2] = sum0;
            pDstC[i * O + k * 2 + 1] = sum1;
        }
    }

    k = k * 2;
    if(k < O){
        for (i = 0; i < M / 2; i++) {
            int32_t sum00 = 0;
            int32_t sum10 = 0;

            for (j = 0; j < N; j++) {
                int32_t AVal0 = pSrcA[i * 2 * N + (j)];
                int32_t AVal1 = pSrcA[i * 2 * N + N + (j)];

                int32_t BVal = pSrcB[j * O + k];

                sum00 = sum00 + AVal0 * BVal;
                sum10 = sum10 + AVal1 * BVal;
            }

            pDstC[(i * 2) * O + k] = sum00;
            pDstC[(i * 2 + 1) * O + k] = sum10;
        }

        i = i*2;
        if(i < M){
            int32_t sum = 0;
            for(j = 0; j < N; j++){
                int32_t AVal = pSrcA[i * N + j];
                int32_t BVal = pSrcB[j * O + k];

                sum = sum + AVal * BVal;
            }

            pDstC[i * O + k] = sum;
        }
    }

    hal_team_barrier();
}

#endif

// undefine BASIC_VERSION
/**
   @} end of BasicMatMultKernels group
*/
