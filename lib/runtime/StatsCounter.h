#ifndef LIB_STATSCOUNTER_CUCORR_H_
#define LIB_STATSCOUNTER_CUCORR_H_

#include "support/Logger.h"

#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits.h>
#include <map>
#include <numeric>
#include <vector>

#define CUCORR_CUDA_EVENT_LIST                 \
  cucorr_stat_handle(event_query_calls);       \
  cucorr_stat_handle(stream_query_calls);      \
  cucorr_stat_handle(device_free_calls);       \
  cucorr_stat_handle(device_alloc_calls);      \
  cucorr_stat_handle(managed_alloc_calls);     \
  cucorr_stat_handle(host_unregister_calls);   \
  cucorr_stat_handle(host_register_calls);     \
  cucorr_stat_handle(stream_wait_event_calls); \
  cucorr_stat_handle(memset_async_calls);      \
  cucorr_stat_handle(memcpy_async_calls);      \
  cucorr_stat_handle(memset_calls);            \
  cucorr_stat_handle(memcpy_calls);            \
  cucorr_stat_handle(create_event_calls);      \
  cucorr_stat_handle(create_stream_calls);     \
  cucorr_stat_handle(sync_event_calls);        \
  cucorr_stat_handle(sync_stream_calls);       \
  cucorr_stat_handle(sync_device_calls);       \
  cucorr_stat_handle(event_record_calls);      \
  cucorr_stat_handle(kernel_register_calls);   \
  cucorr_stat_handle(host_free_calls);         \
  cucorr_stat_handle(host_alloc_calls);

namespace cucorr::runtime {

namespace softcounter {

using Counter       = long long int;
using AtomicCounter = std::atomic<Counter>;

class Statistics {
 private:
  std::vector<unsigned> numbers{};
  std::map<unsigned, unsigned> histogram{};
  unsigned min{UINT_MAX};
  unsigned max{0};
  unsigned bucketSize{10};

  unsigned getBucket(unsigned number) const {
    if (number <= 8) {
      return number;
    }
    return (number / bucketSize) * bucketSize;
  }

 public:
  Statistics(unsigned bucketSize = 10) : bucketSize(bucketSize) {
  }

  void addNumber(unsigned number) {
    numbers.push_back(number);
    unsigned bucket = getBucket(number);
    histogram[bucket]++;

    if (number < min)
      min = number;
    if (number > max)
      max = number;
  }

  double getAverage() const {
    if (numbers.empty()) {
      return 0.0;
    }
    const double sum   = std::accumulate(numbers.begin(), numbers.end(), 0.0);
    const auto average = sum / double(numbers.size());
    return average;
  }

  unsigned getMin() const {
    return min;
  }

  unsigned getMax() const {
    return max;
  }

  void printHistogram(std::ostream& s) const {
    for (const auto& pair : histogram) {
      s << "[" << pair.first << " - " << (pair.first + bucketSize - 1) << "]: " << pair.second << ", ";
    }
    s << "\n";
  }
};

#define cucorr_stat_handle(name) \
  inline void inc_##name() {     \
  }                              \
  inline Counter get_##name() {  \
    return 0;                    \
  }

class NoneRecorder {
 public:
  Statistics stats_w;
  Statistics stats_r;
  CUCORR_CUDA_EVENT_LIST
#include "TsanEvents.inc"
  void TsanMemoryReadCount(unsigned count) {
  }
  void TsanMemoryWriteCount(unsigned count) {
  }
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
  Statistics stats_w;
  Statistics stats_r;

  CUCORR_CUDA_EVENT_LIST
#include "TsanEvents.inc"

  void inc_TsanMemoryReadCount(unsigned count) {
    this->TsanMemoryRead++;
    stats_r.addNumber(count);
  }
  void inc_TsanMemoryWriteCount(unsigned count) {
    this->TsanMemoryWrite++;
    stats_w.addNumber(count);
  }
};
#undef cucorr_stat_handle
}  // namespace softcounter

#ifdef CUCORR_SOFTCOUNTER
using Recorder = softcounter::AccessRecorder;
#else
using Recorder = softcounter::NoneRecorder;
#endif

}  // namespace cucorr::runtime

#endif