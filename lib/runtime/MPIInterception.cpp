#include "StatsCounter.h"
#include "TSan_External.h"
#include "support/Table.h"

#include <mpi.h>
#include <threads.h>

#ifndef _EXTERN_C_
#ifdef __cplusplus
#define _EXTERN_C_ extern "C"
#else /* __cplusplus */
#define _EXTERN_C_
#endif /* __cplusplus */
#endif /* _EXTERN_C_ */

namespace cucorr::mpi::runtime {

using namespace cucorr::runtime::softcounter;

#define cucorr_stat_handle(name) \
  inline void inc_##name() {     \
  }                              \
  inline Counter get_##name() {  \
    return 0;                    \
  }
class MPINoneRecorder final {
 public:
  cucorr_stat_handle(TsanMemoryRead);
  cucorr_stat_handle(TsanMemoryWrite);
  cucorr_stat_handle(TsanSwitchToFiber);
  cucorr_stat_handle(TsanHappensBefore);
  cucorr_stat_handle(TsanHappensAfter);

  cucorr_stat_handle(Send);
  cucorr_stat_handle(Isend);
  cucorr_stat_handle(Recv);
  cucorr_stat_handle(Irecv);
  cucorr_stat_handle(Wait);
  cucorr_stat_handle(Waitall);
  cucorr_stat_handle(SendRecv);
  cucorr_stat_handle(Reduce);
  cucorr_stat_handle(AllReduce);
  cucorr_stat_handle(Barrier);
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

struct MPIAccessRecorder final {
 public:
  cucorr_stat_handle(TsanMemoryRead);
  cucorr_stat_handle(TsanMemoryWrite);
  cucorr_stat_handle(TsanSwitchToFiber);
  cucorr_stat_handle(TsanHappensBefore);
  cucorr_stat_handle(TsanHappensAfter);

  cucorr_stat_handle(Send);
  cucorr_stat_handle(Isend);
  cucorr_stat_handle(Recv);
  cucorr_stat_handle(Irecv);
  cucorr_stat_handle(Wait);
  cucorr_stat_handle(Waitall);
  cucorr_stat_handle(SendRecv);
  cucorr_stat_handle(Reduce);
  cucorr_stat_handle(AllReduce);
  cucorr_stat_handle(Barrier);
};

#ifdef CUCORR_SOFTCOUNTER
using MPIRecorder = MPIAccessRecorder;
#else
using MPIRecorder = MPINoneRecorder;
#endif

struct MPIRuntime final {
  MPIRecorder mpi_recorder;
  ~MPIRuntime() {
#undef cucorr_stat_handle
#define cucorr_stat_handle(name) table.put(Row::make(#name, mpi_recorder.get_##name()));
#if CUCORR_SOFTCOUNTER
    Table table{"Cucorr MPI runtime statistics"};
    cucorr_stat_handle(TsanMemoryRead);
    cucorr_stat_handle(TsanMemoryWrite);
    cucorr_stat_handle(TsanSwitchToFiber);
    cucorr_stat_handle(TsanHappensBefore);
    cucorr_stat_handle(TsanHappensAfter);

    cucorr_stat_handle(Send);
    cucorr_stat_handle(Isend);
    cucorr_stat_handle(Recv);
    cucorr_stat_handle(Irecv);
    cucorr_stat_handle(Wait);
    cucorr_stat_handle(Waitall);
    cucorr_stat_handle(SendRecv);
    cucorr_stat_handle(Reduce);
    cucorr_stat_handle(AllReduce);
    cucorr_stat_handle(Barrier);

    table.print(std::cout);
#endif
#undef cucorr_stat_handle
  }
};
}  // namespace cucorr::mpi::runtime

_EXTERN_C_ int PMPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);

_EXTERN_C_ int MPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
  int _wrap_py_return_val = PMPI_Send(buf, count, datatype, dest, tag, comm);

  int type_size;
  MPI_Type_size(datatype, &type_size);
  TsanMemoryReadPC(buf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));

  return _wrap_py_return_val;
}

_EXTERN_C_ int PMPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
                         MPI_Status* status);
_EXTERN_C_ int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
                        MPI_Status* status) {
  int _wrap_py_return_val = PMPI_Recv(buf, count, datatype, source, tag, comm, status);

  int type_size;
  MPI_Type_size(datatype, &type_size);
  TsanMemoryWritePC(buf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));

  return _wrap_py_return_val;
}

thread_local void* async_fiber = nullptr;

int PMPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request* request);
int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request* request) {
  int _wrap_py_return_val = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);

  int type_size;
  MPI_Type_size(datatype, &type_size);
  {
    if (!async_fiber) {
      async_fiber = TsanCreateFiber(0);
    }
    void* old_fiber = TsanGetCurrentFiber();
    TsanSwitchToFiber(async_fiber, 0);
    TsanMemoryWritePC(buf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));
    if (request) {
      TsanHappensBefore(request);
    }
    TsanSwitchToFiber(old_fiber, 1);
  }
  return _wrap_py_return_val;
}

_EXTERN_C_ int PMPI_Isend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
                          MPI_Request* request);
_EXTERN_C_ int MPI_Isend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
                         MPI_Request* request) {
  int _wrap_py_return_val = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);

  int type_size;
  MPI_Type_size(datatype, &type_size);
  {
    if (!async_fiber) {
      async_fiber = TsanCreateFiber(0);
    }
    void* old_fiber = TsanGetCurrentFiber();
    TsanSwitchToFiber(async_fiber, 0);
    TsanMemoryReadPC(buf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));
    if (request) {
      TsanHappensBefore(request);
    }
    TsanSwitchToFiber(old_fiber, 1);
  }

  return _wrap_py_return_val;
}

_EXTERN_C_ int PMPI_Wait(MPI_Request* request, MPI_Status* status);
_EXTERN_C_ int MPI_Wait(MPI_Request* request, MPI_Status* status) {
  int _wrap_py_return_val = PMPI_Wait(request, status);

  if (request) {
    TsanHappensAfter(request);
  }
  return _wrap_py_return_val;
}

_EXTERN_C_ int PMPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status* array_of_statuses);
_EXTERN_C_ int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status* array_of_statuses) {
  int _wrap_py_return_val = PMPI_Waitall(count, array_of_requests, array_of_statuses);

  for (int i = 0; i < count; ++i) {
    if (&(array_of_requests[i])) {
      TsanHappensAfter(&(array_of_requests[i]));
    }
  }

  return _wrap_py_return_val;
}

int PMPI_Sendrecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void* recvbuf,
                  int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status* status);
int MPI_Sendrecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void* recvbuf,
                 int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status* status) {
  int _wrap_py_return_val = PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype,
                                          source, recvtag, comm, status);

  {
    int send_type_size;
    MPI_Type_size(sendtype, &send_type_size);

    TsanMemoryReadPC(sendbuf, static_cast<uptr>(sendcount) * send_type_size, __builtin_return_address(0));

    int recv_type_size;
    MPI_Type_size(recvtype, &recv_type_size);
    TsanMemoryWritePC(recvbuf, static_cast<uptr>(recvcount) * recv_type_size, __builtin_return_address(0));
  }
  return _wrap_py_return_val;
}

int PMPI_Barrier(MPI_Comm comm);
int MPI_Barrier(MPI_Comm comm) {
  int _wrap_py_return_val = PMPI_Barrier(comm);
  if (!async_fiber) {
    async_fiber = TsanCreateFiber(0);
  }
  void* old_fiber = TsanGetCurrentFiber();
  TsanSwitchToFiber(async_fiber, 0);
  TsanSwitchToFiber(old_fiber, 0);
  return _wrap_py_return_val;
}

int PMPI_Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root,
                MPI_Comm comm);
int MPI_Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root,
               MPI_Comm comm) {
  int _wrap_py_return_val = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
  {
    int type_size;
    MPI_Type_size(datatype, &type_size);

    TsanMemoryReadPC(sendbuf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));
    TsanMemoryWritePC(recvbuf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));
  }
  return _wrap_py_return_val;
}

int PMPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  int _wrap_py_return_val = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  {
    int type_size;
    MPI_Type_size(datatype, &type_size);

    TsanMemoryReadPC(sendbuf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));
    TsanMemoryWritePC(recvbuf, static_cast<uptr>(count) * type_size, __builtin_return_address(0));
  }
  return _wrap_py_return_val;
}