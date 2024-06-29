#pragma once

#include <AccessCounter.h>

namespace cucorr::runtime {

namespace softcounter {
using Counter       = long long int;
using AtomicCounter = std::atomic<Counter>;

#define basicIncrementerStat(name) \
  AtomicCounter name = 0;          \
  inline void inc_##name() {       \
    this->name++;                  \
  }                                \
  inline Counter get_##name() {    \
    return this->name;             \
  }

#define basicIncrementerStat_dummy(name) \
  inline void inc_##name() {             \
  }

class NoneRecorder {
 public:
  basicIncrementerStat_dummy(event_query_calls);
  basicIncrementerStat_dummy(stream_query_calls);
  basicIncrementerStat_dummy(device_free_calls);
  basicIncrementerStat_dummy(device_alloc_calls);
  basicIncrementerStat_dummy(managed_alloc_calls);
  basicIncrementerStat_dummy(host_unregister_calls);
  basicIncrementerStat_dummy(host_register_calls);
  basicIncrementerStat_dummy(stream_wait_event_calls);
  basicIncrementerStat_dummy(memset_async_calls);
  basicIncrementerStat_dummy(memcpy_async_calls);
  basicIncrementerStat_dummy(memset_calls);
  basicIncrementerStat_dummy(memcpy_calls);
  basicIncrementerStat_dummy(create_event_calls);
  basicIncrementerStat_dummy(create_stream_calls);
  basicIncrementerStat_dummy(sync_event_calls);
  basicIncrementerStat_dummy(sync_stream_calls);
  basicIncrementerStat_dummy(sync_device_calls);
  basicIncrementerStat_dummy(event_record_calls);
  basicIncrementerStat_dummy(kernel_register_calls);
  basicIncrementerStat_dummy(host_free_calls);
  basicIncrementerStat_dummy(host_alloc_calls);
  basicIncrementerStat_dummy(fiber_switches);
};

struct AccessRecorder {
 public:
  basicIncrementerStat(event_query_calls);
  basicIncrementerStat(stream_query_calls);
  basicIncrementerStat(device_free_calls);
  basicIncrementerStat(device_alloc_calls);
  basicIncrementerStat(managed_alloc_calls);
  basicIncrementerStat(host_unregister_calls);
  basicIncrementerStat(host_register_calls);
  basicIncrementerStat(stream_wait_event_calls);
  basicIncrementerStat(memset_async_calls);
  basicIncrementerStat(memcpy_async_calls);
  basicIncrementerStat(memset_calls);
  basicIncrementerStat(memcpy_calls);
  basicIncrementerStat(create_event_calls);
  basicIncrementerStat(create_stream_calls);
  basicIncrementerStat(sync_event_calls);
  basicIncrementerStat(sync_stream_calls);
  basicIncrementerStat(sync_device_calls);
  basicIncrementerStat(event_record_calls);
  basicIncrementerStat(kernel_register_calls);
  basicIncrementerStat(host_free_calls);
  basicIncrementerStat(host_alloc_calls);
  basicIncrementerStat(fiber_switches);
};
}  // namespace softcounter

// #if ENABLE_SOFTCOUNTER == 1
using Recorder = softcounter::AccessRecorder;
// #else
// using Recorder = softcounter::NoneRecorder;
// #endif

}  // namespace cucorr::runtime