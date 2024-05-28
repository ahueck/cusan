#pragma once
#include "FunctionDecl.h"
#include "analysis/KernelAnalysis.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>

using namespace llvm;
namespace cucorr {

namespace analysis {

using KernelArgInfo = cucorr::FunctionArg;

struct CudaKernelInvokeCollector {
  KernelModel model;
  struct KernelInvokeData {
    llvm::SmallVector<KernelArgInfo, 4> args{};
    llvm::Value* void_arg_array{nullptr};
    llvm::Value* cu_stream{nullptr};
  };
  using Data = KernelInvokeData;

  CudaKernelInvokeCollector(const KernelModel& current_stub_model) : model(current_stub_model) {
  }

  llvm::Optional<KernelInvokeData> match(llvm::CallBase& cb, Function& callee) const {
    if (callee.getName() == "cudaLaunchKernel") {
      auto* cu_stream_handle      = std::prev(cb.arg_end())->get();
      auto* void_kernel_arg_array = std::prev(cb.arg_end(), 3)->get();
      auto* cb_parent_function    = cb.getFunction();
      auto kernel_args            = extract_kernel_args_for(*cb_parent_function);
      return KernelInvokeData{kernel_args, void_kernel_arg_array, cu_stream_handle};
    }
    return llvm::NoneType();
  }

 private:
  llvm::SmallVector<KernelArgInfo, 4> extract_kernel_args_for(Function& stub) const {
    llvm::SmallVector<KernelArgInfo, 4> arg_info;
    for (auto arg : llvm::enumerate(stub.args())) {
      const auto index = arg.index();
      assert(index < model.args.size() && "More stub args than in model data.");
      auto model_arg = model.args[index];
      model_arg.arg  = &arg.value();
      arg_info.emplace_back(model_arg);
    }
    return arg_info;
  }
};

}  // namespace analysis

namespace transform {

struct KernelInvokeTransformer {
  callback::FunctionDecl* decls_;

  KernelInvokeTransformer(callback::FunctionDecl* decls) : decls_(decls) {
  }

  bool transform(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb) const {
    using namespace llvm;
    return generate_compound_cb(data, irb);
  }

 private:
  static short access_cast(AccessState access, bool is_ptr) {
    auto value = static_cast<short>(access);
    value <<= 1;
    if (is_ptr) {
      value |= 1;
    }
    return value;
  }

  static llvm::Value* get_cu_stream_ptr(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb) {
    auto cu_stream = data.cu_stream;
    assert(cu_stream != nullptr && "Require cuda stream!");
    auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(cu_stream, irb.getInt8PtrTy());
    return cu_stream_void_ptr;
  }

  bool generate_compound_cb(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb) const {
    const bool should_transform = llvm::count_if(data.args, [&](const auto& elem) { return elem.is_pointer; }) > 0;

    if (!should_transform) {
      return false;
    }

    auto target_callback = decls_->cucorr_register_access;

    auto i16_ty = Type::getInt16Ty(irb.getContext());
    auto i32_ty = Type::getInt32Ty(irb.getContext());

    auto cu_stream_void_ptr = get_cu_stream_ptr(data, irb);
    auto arg_size           = irb.getInt32(data.args.size());
    auto arg_access_array   = irb.CreateAlloca(i16_ty, arg_size);

    for (const auto& arg : llvm::enumerate(data.args)) {
      const auto access = access_cast(arg.value().state, arg.value().is_pointer);
      Value* Idx        = ConstantInt::get(i32_ty, arg.index());
      Value* acc        = ConstantInt::get(i16_ty, access);
      auto gep          = irb.CreateGEP(i16_ty, arg_access_array, Idx);
      irb.CreateStore(acc, gep);
    }

    auto ptr_ptr_ptr_ty           = PointerType::get(PointerType::get(Type::getInt8PtrTy(irb.getContext()), 0), 0);
    auto* void_ptr_array_cast     = irb.CreateBitOrPointerCast(data.void_arg_array, ptr_ptr_ptr_ty);
    Value* args_cucorr_register[] = {void_ptr_array_cast, arg_access_array, arg_size, cu_stream_void_ptr};
    irb.CreateCall(target_callback.f, args_cucorr_register);

    return true;
  }
};

template <class Collector, class Transformer>
class CallInstrumenter {
  Function& f_;
  Collector collector_;
  Transformer transformer_;
  struct InstrumentationData {
    typename Collector::Data user_data;
    CallBase* cb;
  };
  llvm::SmallVector<InstrumentationData, 4> data_vec_;

 public:
  CallInstrumenter(Collector c, Transformer t, Function& f) : f_(f), collector_(c), transformer_(t) {
  }

  bool instrument() {
    for (auto& I : instructions(f_)) {
      if (auto* cb = dyn_cast<CallBase>(&I)) {
        if (auto* f = cb->getCalledFunction()) {
          auto t = collector_.match(*cb, *f);
          if (t.hasValue()) {
            data_vec_.push_back({t.getValue(), cb});
          }
        }
      }
    }

    bool modified = false;
    if (data_vec_.size() > 0) {
      IRBuilder<> irb{data_vec_[0].cb};
      for (auto data : data_vec_) {
        irb.SetInsertPoint(data.cb);
        modified |= transformer_.transform(data.user_data, irb);
      }
    }
    return modified;
  }
};

template <class T>
class SimpleInstrumenter {
  enum class InsertLocation {
    // insert before or after the call that were instrumenting
    kBefore,
    kAfter
  };

  const FunctionCallee* callee_;
  StringRef func_name_;
  SmallVector<llvm::CallBase*, 4> target_callsites_;

 public:
  void setup(StringRef name, FunctionCallee* callee) {
    func_name_ = name;
    callee_    = callee;
  }

  bool instrument(Function& func, InsertLocation loc = InsertLocation::kAfter) {
    for (auto& I : instructions(func)) {
      if (auto* cb = dyn_cast<CallBase>(&I)) {
        if (auto* f = cb->getCalledFunction()) {
          if (func_name_ == f->getName()) {
            target_callsites_.push_back(cb);
          }
        }
      }
    }

    if (!target_callsites_.empty()) {
      IRBuilder<> irb{target_callsites_[0]};
      for (CallBase* cb : target_callsites_) {
        if (loc == InsertLocation::kBefore) {
          irb.SetInsertPoint(cb);
        } else {
          if (auto* invoke = dyn_cast<InvokeInst>(cb)) {
            irb.SetInsertPoint(invoke->getNormalDest()->getFirstNonPHI());
          } else {
            irb.SetInsertPoint(cb->getNextNonDebugInstruction());
          }
        }
        if (!cb->arg_empty()) {
          SmallVector<Value*> v;
          for (auto& arg : cb->args()) {
            v.push_back(arg.get());
          }
          auto args = T::map_arguments(irb, v);
          irb.CreateCall(*callee_, args);
        } else {
          irb.CreateCall(*callee_, {});
        }
      }
    }
    return !target_callsites_.empty();
  }
};

class DeviceSyncInstrumenter : public SimpleInstrumenter<DeviceSyncInstrumenter> {
 public:
  DeviceSyncInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaDeviceSynchronize", &decls->cucorr_sync_device.f);
  }
  static llvm::SmallVector<Value*, 4> map_arguments(IRBuilder<>&, llvm::ArrayRef<Value*>) {
    return {};
  }
};
class StreamSyncInstrumenter : public SimpleInstrumenter<StreamSyncInstrumenter> {
 public:
  StreamSyncInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaStreamSynchronize", &decls->cucorr_sync_stream.f);
  }
  static llvm::SmallVector<Value*, 1> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 1);
    Value* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {cu_stream_void_ptr};
  }
};
class EventSyncInstrumenter : public SimpleInstrumenter<EventSyncInstrumenter> {
 public:
  EventSyncInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaEventSynchronize", &decls->cucorr_sync_event.f);
  }
  static llvm::SmallVector<Value*, 1> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 1);
    auto* cu_event_void_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {cu_event_void_ptr};
  }
};
class EventRecordInstrumenter : public SimpleInstrumenter<EventRecordInstrumenter> {
 public:
  EventRecordInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaEventRecord", &decls->cucorr_event_record.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 2);
    auto* cu_event_void_ptr  = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
    return {cu_event_void_ptr, cu_stream_void_ptr};
  }
};
class EventRecordFlagsInstrumenter : public SimpleInstrumenter<EventRecordFlagsInstrumenter> {
 public:
  EventRecordFlagsInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaEventRecordWithFlags", &decls->cucorr_event_record.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 3);
    auto* cu_event_void_ptr  = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
    return {cu_event_void_ptr, cu_stream_void_ptr};
  }
};

class MemcpyAsyncInstrumenter : public SimpleInstrumenter<MemcpyAsyncInstrumenter> {
 public:
  MemcpyAsyncInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaMemcpyAsync", &decls->cucorr_memcpy_async.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    // void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0
    assert(args.size() == 5);
    auto* dst_ptr   = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* src_ptr   = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
    auto* count     = args[2];
    auto* kind      = args[3];
    auto* cu_stream = irb.CreateBitOrPointerCast(args[4], irb.getInt8PtrTy());
    return {dst_ptr, src_ptr, count, kind, cu_stream};
  }
};

class CudaMemcpyInstrumenter : public SimpleInstrumenter<CudaMemcpyInstrumenter> {
 public:
  CudaMemcpyInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaMemcpy", &decls->cucorr_memcpy.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    // void* dst, const void* src, size_t count, cudaMemcpyKind kind
    assert(args.size() == 4);
    auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* src_ptr = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
    auto* count   = args[2];
    auto* kind    = args[3];
    return {dst_ptr, src_ptr, count, kind};
  }
};

class MemsetAsyncInstrumenter : public SimpleInstrumenter<MemsetAsyncInstrumenter> {
 public:
  MemsetAsyncInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaMemsetAsync", &decls->cucorr_memset_async.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* devPtr, int  value, size_t count, cudaStream_t stream = 0 )
    assert(args.size() == 4);
    auto* dst_ptr   = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* value     = args[1];
    auto* count     = args[2];
    auto* cu_stream = irb.CreateBitOrPointerCast(args[3], irb.getInt8PtrTy());
    return {dst_ptr, value, count, cu_stream};
  }
};
class CudaMemsetInstrumenter : public SimpleInstrumenter<CudaMemsetInstrumenter> {
 public:
  CudaMemsetInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaMemset", &decls->cucorr_memset.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* devPtr, int  value, size_t count,)
    assert(args.size() == 3);
    auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* value   = args[1];
    auto* count   = args[2];
    return {dst_ptr, value, count};
  }
};

class CudaHostAlloc : public SimpleInstrumenter<CudaHostAlloc> {
 public:
  CudaHostAlloc(callback::FunctionDecl* decls) {
    setup("cudaHostAlloc", &decls->cucorr_memset.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void** ptr, size_t size, unsigned int flags )
    assert(args.size() == 3);
    auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* size    = args[1];
    auto* flags   = args[2];
    return {dst_ptr, size, flags};
  }
};

class CudaMallocHost : public SimpleInstrumenter<CudaMallocHost> {
 public:
  CudaMallocHost(callback::FunctionDecl* decls) {
    setup("cudaHostAlloc", &decls->cucorr_memset.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void** ptr, size_t size )
    assert(args.size() == 2);
    auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* size    = args[1];
    auto* flags   = llvm::ConstantInt::get(Type::getInt32Ty(irb.getContext()), 0, false);
    return {dst_ptr, size, flags};
  }
};

class EventCreateInstrumenter : public SimpleInstrumenter<EventCreateInstrumenter> {
 public:
  EventCreateInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaEventCreate", &decls->cucorr_event_create.f);
  }
  static llvm::SmallVector<Value*, 1> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 1);
    // auto* cu_event_void_ptr = irb.CreateLoad(irb.getInt8PtrTy(), args[0], "");
    auto* cu_event_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {cu_event_void_ptr_ptr};
  }
};

class StreamCreateInstrumenter : public SimpleInstrumenter<StreamCreateInstrumenter> {
 public:
  StreamCreateInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaStreamCreate", &decls->cucorr_stream_create.f);
  }
  static llvm::SmallVector<Value*, 1> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 1);
    // auto* cu_stream_void_ptr = irb.CreateLoad(irb.getInt8PtrTy(), args[0], "");
    auto* cu_stream_void_ptr_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {cu_stream_void_ptr_ptr};
  }
};

class StreamWaitEventInstrumenter : public SimpleInstrumenter<StreamWaitEventInstrumenter> {
 public:
  StreamWaitEventInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaStreamWaitEvent", &decls->cucorr_stream_wait_event.f);
  }
  static llvm::SmallVector<Value*, 1> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    assert(args.size() == 3);
    // auto* cu_stream_void_ptr = irb.CreateLoad(irb.getInt8PtrTy(), args[0], "");
    auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* cu_event_void_ptr  = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
    return {cu_stream_void_ptr, cu_event_void_ptr, args[2]};
  }
};

class CudaHostRegister : public SimpleInstrumenter<CudaHostRegister> {
 public:
  CudaHostRegister(callback::FunctionDecl* decls) {
    setup("cudaHostRegister", &decls->cucorr_memset.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* ptr)
    assert(args.size() == 3);
    auto* ptr   = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* size  = args[1];
    auto* flags = args[2];
    return {ptr, size, flags};
  }
};

class CudaHostUnregister : public SimpleInstrumenter<CudaHostUnregister> {
 public:
  CudaHostUnregister(callback::FunctionDecl* decls) {
    setup("cudaHostUnregister", &decls->cucorr_memset.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* ptr)
    assert(args.size() == 1);
    auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {ptr};
  }
};

class CudaHostFree : public SimpleInstrumenter<CudaHostFree> {
 public:
  CudaHostFree(callback::FunctionDecl* decls) {
    setup("cudaFreeHost", &decls->cucorr_memset.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* ptr)
    assert(args.size() == 1);
    auto* ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {ptr};
  }
};

}  // namespace transform
}  // namespace cucorr