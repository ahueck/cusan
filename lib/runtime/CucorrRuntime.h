// cucorr library
// Copyright (c) 2023 cucorr authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LIB_RUNTIME_CUCORR_H_
#define LIB_RUNTIME_CUCORR_H_
#include <cstddef>

#ifdef __cplusplus
namespace cucorr::runtime {
using TsanFiber = void*;
using Event     = const void*;
using RawStream = const void*;
}  // namespace cucorr::runtime
#else
#define TsanFiber void*
#define Event const void*
#define RawStream const void*
#endif

using cucorr::runtime::Event;
using cucorr::runtime::RawStream;
using cucorr::runtime::TsanFiber;

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cucorr_memcpy_kind_t : unsigned int {
  cucorr_MemcpyHostToHost     = 0,
  cucorr_MemcpyHostToDevice   = 1,
  cucorr_MemcpyDeviceToHost   = 2,
  cucorr_MemcpyDeviceToDevice = 3,
  cucorr_MemcpyDefault        = 4,
} cucorr_MemcpyKind;

void _cucorr_kernel_register(void** kernel_args, short* modes, int n, RawStream stream);
void _cucorr_sync_device();
void _cucorr_event_record(Event event, RawStream stream);
void _cucorr_sync_stream(RawStream stream);
void _cucorr_sync_event(Event event);
void _cucorr_stream_event(Event event);
void _cucorr_create_event(RawStream* event);
void _cucorr_create_stream(RawStream* stream);
void _cucorr_memcpy_async(void* target, const void* from, size_t count, cucorr_MemcpyKind kind, RawStream stream);
void _cucorr_memset_async(void* target, int, size_t count, RawStream stream);
void _cucorr_memcpy(void* target, const void* from, size_t count, cucorr_MemcpyKind);
void _cucorr_memset(void* target, int, size_t count);
void _cucorr_stream_wait_event(RawStream stream, Event event, unsigned int flags);
void _cucorr_stream_wait_event(RawStream stream, Event event, unsigned int flags);
void _cucorr_host_alloc(void** ptr, size_t size, unsigned int flags);
void _cucorr_host_free(void* ptr);
void _cucorr_host_register(void* ptr, size_t size, unsigned int flags);
void _cucorr_host_unregister(void* ptr);
#ifdef __cplusplus
}
#endif

#endif /* LIB_RUNTIME_CUCORR_H_ */
