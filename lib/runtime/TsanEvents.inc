#ifndef cusan_stat_handle
#define cusan_stat_handle(event)
#endif

cusan_stat_handle(TsanMemoryRead);
cusan_stat_handle(TsanMemoryWrite);
cusan_stat_handle(TsanSwitchToFiber);
cusan_stat_handle(TsanHappensBefore);
cusan_stat_handle(TsanHappensAfter);
cusan_stat_handle(TsanCreateFiber);