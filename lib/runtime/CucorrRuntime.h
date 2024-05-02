// cucorr library
// Copyright (c) 2023 cucorr authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LIB_RUNTIME_CUCORR_H_
#define LIB_RUNTIME_CUCORR_H_

#ifdef __cplusplus
extern "C" {
#endif
void _cucorr_kernel_register(const void* ptr, short mode, const void* stream);
void _cucorr_kernel_register_n(void*** kernel_args, short* modes, int n, const void* stream);
void _cucorr_sync_device();
void _cucorr_event_record(const void* event, const void* stream);
void _cucorr_sync_stream(const void* stream);
void _cucorr_sync_event(const void* event);
void _cucorr_stream_event(const void* event);
void _cucorr_create_event(const void** event);
void _cucorr_create_stream(const void** stream);
#ifdef __cplusplus
}
#endif

#endif /* LIB_RUNTIME_CUCORR_H_ */
