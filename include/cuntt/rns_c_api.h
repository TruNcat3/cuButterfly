#ifndef CUNTT_RNS_C_API_H
#define CUNTT_RNS_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cubutterflyRnsPlanOpaque* cubutterflyRnsPlan;
/* Status: 0 success, 1 invalid argument/state, 2 runtime failure.
 * Error text is thread-local and valid until the next ABI operation.
 * All functions catch C++ exceptions. Handles must not be shared concurrently.
 */
const char* cubutterflyRnsLastError(void);
int cubutterflyRnsCreate(const char* config_json, cubutterflyRnsPlan* result);
/* placement: 0 separate, 1 in-place; policy: 0 prefer, 1 require, 2 refresh.
 * Prepare once, outside capture/timing. Empty cache_directory disables writes;
 * Require still requires a compatible cache entry. stream is a CUDA stream.
 */
int cubutterflyRnsPrepare(cubutterflyRnsPlan plan, int placement,
                         const char* cache_directory, int policy,
                         unsigned repetitions, void* stream);
/* Asynchronous and allocation-free on success. Canonical uint64 residues only.
 * Caller owns device buffers/stream and their lifetime. Keep plan alive until
 * all launches and captured graphs using it have completed and been released.
 * No implicit synchronization; use one stream at a time.
 */
int cubutterflyRnsExecute(cubutterflyRnsPlan plan, const uint64_t* input,
                         uint64_t* output, void* stream);
/* Two-call query: buffer=NULL returns required bytes including final NUL.
 * Insufficient buffers return status 1 and update *bytes to required size.
 */
int cubutterflyRnsInfoJson(cubutterflyRnsPlan plan, char* buffer, size_t* bytes);
int cubutterflyRnsDestroy(cubutterflyRnsPlan plan);

#ifdef __cplusplus
}
#endif
#endif
