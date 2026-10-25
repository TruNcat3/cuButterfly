/* SPDX-License-Identifier: BSD-3-Clause
 * CPU test interposer: abort if a test unexpectedly initializes CUDA.
 */
#include <stdlib.h>
int cuInit(unsigned int flags) { (void)flags; abort(); }
int cudaGetDevice(int* device) { (void)device; abort(); }
int cudaGetDeviceProperties(void* properties, int device) {
    (void)properties; (void)device; abort();
}
int cufftCreate(int* handle) { (void)handle; abort(); }
