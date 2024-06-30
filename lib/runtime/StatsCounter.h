#pragma once

// #include <AccessCounter.h>

#include <atomic>

namespace cucorr::runtime {

namespace softcounter {
using Counter       = long long int;
using AtomicCounter = std::atomic<Counter>;

#define cucorr_stat_handle(name) \
  inline void inc_##name() {     \
  }                              \
  inline Counter get_##name() {  \
    return 0;                    \
  }

class NoneRecorder {
 public:
  cucorr_stat_handle(event_query_calls);
  cucorr_stat_handle(stream_query_calls);
  cucorr_stat_handle(device_free_calls);
  cucorr_stat_handle(device_alloc_calls);
  cucorr_stat_handle(managed_alloc_calls);
  cucorr_stat_handle(host_unregister_calls);
  cucorr_stat_handle(host_register_calls);
  cucorr_stat_handle(stream_wait_event_calls);
  cucorr_stat_handle(memset_async_calls);
  cucorr_stat_handle(memcpy_async_calls);
  cucorr_stat_handle(memset_calls);
  cucorr_stat_handle(memcpy_calls);
  cucorr_stat_handle(create_event_calls);
  cucorr_stat_handle(create_stream_calls);
  cucorr_stat_handle(sync_event_calls);
  cucorr_stat_handle(sync_stream_calls);
  cucorr_stat_handle(sync_device_calls);
  cucorr_stat_handle(event_record_calls);
  cucorr_stat_handle(kernel_register_calls);
  cucorr_stat_handle(host_free_calls);
  cucorr_stat_handle(host_alloc_calls);
  cucorr_stat_handle(fiber_switches);
};

#undef cucorr_stat_handle
#define cucorr_stat_handle(name) \
  AtomicCounter name = 0;        \
  inline void inc_##name() {     \
    this->name++;                \
  }                              \
  inline Counter get_##name() {  \
    return this->name;           \
  }

struct AccessRecorder {
 public:
  cucorr_stat_handle(event_query_calls);
  cucorr_stat_handle(stream_query_calls);
  cucorr_stat_handle(device_free_calls);
  cucorr_stat_handle(device_alloc_calls);
  cucorr_stat_handle(managed_alloc_calls);
  cucorr_stat_handle(host_unregister_calls);
  cucorr_stat_handle(host_register_calls);
  cucorr_stat_handle(stream_wait_event_calls);
  cucorr_stat_handle(memset_async_calls);
  cucorr_stat_handle(memcpy_async_calls);
  cucorr_stat_handle(memset_calls);
  cucorr_stat_handle(memcpy_calls);
  cucorr_stat_handle(create_event_calls);
  cucorr_stat_handle(create_stream_calls);
  cucorr_stat_handle(sync_event_calls);
  cucorr_stat_handle(sync_stream_calls);
  cucorr_stat_handle(sync_device_calls);
  cucorr_stat_handle(event_record_calls);
  cucorr_stat_handle(kernel_register_calls);
  cucorr_stat_handle(host_free_calls);
  cucorr_stat_handle(host_alloc_calls);
  cucorr_stat_handle(fiber_switches);
};
#undef cucorr_stat_handle
}  // namespace softcounter

#ifdef CUCORR_SOFTCOUNTER
using Recorder = softcounter::AccessRecorder;
#else
using Recorder = softcounter::NoneRecorder;
#endif

}  // namespace cucorr::runtime