#pragma once
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

using namespace llvm;
namespace cucorr {
namespace callback {
struct CucorrFunction {
  const std::string name;
  FunctionCallee f{nullptr};
  SmallVector<Type*, 4> arg_types{};
};

struct FunctionDecl {
  CucorrFunction cucorr_register_access{"_cucorr_kernel_register"};

  CucorrFunction cucorr_event_record{"_cucorr_event_record"};

  CucorrFunction cucorr_sync_device{"_cucorr_sync_device"};
  CucorrFunction cucorr_sync_stream{"_cucorr_sync_stream"};
  CucorrFunction cucorr_sync_event{"_cucorr_sync_event"};
  CucorrFunction cucorr_event_create{"_cucorr_create_event"};
  CucorrFunction cucorr_stream_create{"_cucorr_create_stream"};
  CucorrFunction cucorr_memset_async{"_cucorr_memset_async"};
  CucorrFunction cucorr_memcpy_async{"_cucorr_memcpy_async"};
  CucorrFunction cucorr_memset{"_cucorr_memset"};
  CucorrFunction cucorr_memcpy{"_cucorr_memcpy"};
  CucorrFunction cucorr_stream_wait_event{"_cucorr_stream_wait_event"};
  CucorrFunction cucorr_host_alloc{"_cucorr_host_alloc"};
  CucorrFunction cucorr_host_free{"_cucorr_host_free"};
  CucorrFunction cucorr_host_register{"_cucorr_host_register"};
  CucorrFunction cucorr_host_unregister{"_cucorr_host_unregister"};

  void initialize(Module& m) {
    using namespace llvm;
    auto& c = m.getContext();

    const auto add_optimizer_attributes = [&](auto& arg) {
      arg.addAttr(Attribute::NoCapture);
      arg.addAttr(Attribute::ReadOnly);
    };

    const auto make_function = [&](auto& f_struct, auto f_types) {
      auto func_type     = f_types.empty() ? FunctionType::get(Type::getVoidTy(c), false)
                                           : FunctionType::get(Type::getVoidTy(c), f_types, false);
      auto func_callee   = m.getOrInsertFunction(f_struct.name, func_type);
      f_struct.f         = func_callee;
      f_struct.arg_types = std::move(f_types);
      if (auto f = dyn_cast<Function>(f_struct.f.getCallee())) {
        f->setLinkage(GlobalValue::ExternalLinkage);
        if (f->arg_size() == 0) {
          return;
        }
        auto& first_param = *(f->arg_begin());
        if (first_param.getType()->isPointerTy()) {
          add_optimizer_attributes(first_param);
        }
      }
    };
    using ArgTypes = decltype(CucorrFunction::arg_types);
    // TODO address space?
    ArgTypes arg_types_cucorr_register = {PointerType::get(Type::getInt8PtrTy(c), 0),
                                          Type::getInt16PtrTy(c), Type::getInt32Ty(c), Type::getInt8PtrTy(c)};
    make_function(cucorr_register_access, arg_types_cucorr_register);

    ArgTypes arg_types_sync_device = {};
    make_function(cucorr_sync_device, arg_types_sync_device);

    ArgTypes arg_types_sync_stream = {Type::getInt8PtrTy(c)};
    make_function(cucorr_sync_stream, arg_types_sync_stream);

    ArgTypes arg_types_sync_event = {Type::getInt8PtrTy(c)};
    make_function(cucorr_sync_event, arg_types_sync_event);
    ArgTypes arg_types_event_record = {Type::getInt8PtrTy(c), Type::getInt8PtrTy(c)};
    make_function(cucorr_event_record, arg_types_event_record);

    ArgTypes arg_types_event_create = {Type::getInt8PtrTy(c)};
    make_function(cucorr_event_create, arg_types_event_create);

    ArgTypes arg_types_stream_create = {Type::getInt8PtrTy(c)};
    make_function(cucorr_stream_create, arg_types_stream_create);

    auto size_t_ty = m.getDataLayout().getIntPtrType(c);

    // void* devPtr, int  value, size_t count, RawStream* stream
    ArgTypes arg_types_memset_async = {Type::getInt8PtrTy(c), Type::getInt32Ty(c), size_t_ty, Type::getInt8PtrTy(c)};
    make_function(cucorr_memset_async, arg_types_memset_async);

    // void* dst, const void* src
    ArgTypes arg_types_memcpy_async = {Type::getInt8PtrTy(c), Type::getInt8PtrTy(c),
                                       // size_t count, MemcpyKind kind, RawStream stream
                                       size_t_ty, Type::getInt32Ty(c), Type::getInt8PtrTy(c)};
    make_function(cucorr_memcpy_async, arg_types_memcpy_async);

    // void* devPtr, int  value, size_t count
    ArgTypes arg_types_memset = {Type::getInt8PtrTy(c), Type::getInt32Ty(c), size_t_ty};
    make_function(cucorr_memset, arg_types_memset);

    // void* dst, const void* src
    ArgTypes arg_types_memcpy = {Type::getInt8PtrTy(c), Type::getInt8PtrTy(c),
                                 // size_t count, MemcpyKind kind
                                 size_t_ty, Type::getInt32Ty(c)};
    make_function(cucorr_memcpy, arg_types_memcpy);

    ArgTypes arg_types_stream_wait_event = {Type::getInt8PtrTy(c), Type::getInt8PtrTy(c), Type::getInt32Ty(c)};
    make_function(cucorr_stream_wait_event, arg_types_stream_wait_event);

    ArgTypes arg_types_host_alloc = {Type::getInt8PtrTy(c), size_t_ty, Type::getInt32Ty(c)};
    make_function(cucorr_host_alloc, arg_types_host_alloc);

    ArgTypes arg_types_host_register = {Type::getInt8PtrTy(c), size_t_ty, Type::getInt32Ty(c)};
    make_function(cucorr_host_register, arg_types_host_register);

    ArgTypes arg_types_host_unregister = {Type::getInt8PtrTy(c)};
    make_function(cucorr_host_unregister, arg_types_host_unregister);

    ArgTypes arg_types_host_free = {Type::getInt8PtrTy(c)};
    make_function(cucorr_host_free, arg_types_host_free);
  }
};

}  // namespace callback
}  // namespace cucorr