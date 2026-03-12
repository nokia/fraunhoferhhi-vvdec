// (c) 2026 Nokia
// Licensed under the BSD 3-Clause Clear License
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once

#define HAVE_OPENVINO

#include <stdint.h>

#ifdef __cplusplus
#define EXPORT extern "C" 
#else
#define EXPORT
#endif

EXPORT const char * nnpf_version();

EXPORT void *nnpf_create(int width, int height, const char* model_path, const char* weights_update_path=0);

EXPORT void nnpf_destroy(void *p);

EXPORT void nnpf_process_picture(void *p, uint16_t *in[3], int inStride[3], uint16_t *out[3], int outStride[3], float strength);
