#include "TSan_External.h"

#include <mpi.h>
#include <threads.h>

#ifndef _EXTERN_C_
#ifdef __cplusplus
#define _EXTERN_C_ extern "C"
#else /* __cplusplus */
#define _EXTERN_C_
#endif /* __cplusplus */
#endif /* _EXTERN_C_ */

_EXTERN_C_ int PMPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);

_EXTERN_C_ int MPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
  int _wrap_py_return_val = PMPI_Send(buf, count, datatype, dest, tag, comm);

  int type_size;
  MPI_Type_size(datatype, &type_size);
  TsanMemoryReadPC(buf, static_cast<uptr>(count * type_size), __builtin_return_address(0));

  return _wrap_py_return_val;
}

_EXTERN_C_ int PMPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
                         MPI_Status* status);
_EXTERN_C_ int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
                        MPI_Status* status) {
  int _wrap_py_return_val = PMPI_Recv(buf, count, datatype, source, tag, comm, status);

  int type_size;
  MPI_Type_size(datatype, &type_size);
  TsanMemoryWritePC(buf, static_cast<uptr>(count * type_size), __builtin_return_address(0));

  return _wrap_py_return_val;
}

thread_local void* async_fiber = nullptr;

int PMPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request* request);
int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request* request) {
  int _wrap_py_return_val = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);


  int type_size;
  MPI_Type_size(datatype, &type_size);
  {
    printf("mpiIrecv");
    if (!async_fiber) {
      async_fiber = TsanCreateFiber(0);
    }
    void* old_fiber = TsanGetCurrentFiber();
    TsanSwitchToFiber(async_fiber, 0);
    TsanMemoryWritePC(buf, static_cast<uptr>(count * type_size), __builtin_return_address(0));
    if (request) {
      TsanHappensBefore(request);
    }
    TsanSwitchToFiber(old_fiber, 1);
  }
  return _wrap_py_return_val;
}

int PMPI_Wait(MPI_Request* request, MPI_Status* status);
int MPI_Wait(MPI_Request* request, MPI_Status* status) {
  int _wrap_py_return_val = PMPI_Wait(request, status);

  if (request) {
    printf("mpiwait");
    TsanHappensAfter(request);
  }
  return _wrap_py_return_val;
}