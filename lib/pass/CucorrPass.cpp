// cucorr library
// Copyright (c) 2023 cucorr authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#include "CucorrPass.h"

#include "CommandLine.h"
#include "analysis/KernelAnalysis.h"
#include "support/CudaUtil.h"
#include "support/Logger.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>

using namespace llvm;

namespace cucorr {

namespace callback {
struct CucorrFunction {
  const std::string name;
  llvm::FunctionCallee f{nullptr};
  llvm::SmallVector<llvm::Type*, 4> arg_types{};
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
    using ArgTypes                     = decltype(CucorrFunction::arg_types);
    // TODO address space?
    ArgTypes arg_types_cucorr_register = {PointerType::get(PointerType::get(Type::getInt8PtrTy(c), 0), 0),
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

    //void* devPtr, int  value, size_t count, RawStream* stream
    ArgTypes arg_types_memset_async = {Type::getInt8PtrTy(c), Type::getInt32Ty(c), size_t_ty, Type::getInt8PtrTy(c)};
    make_function(cucorr_memset_async, arg_types_memset_async);
    
    
    //void* dst, const void* src
    ArgTypes arg_types_memcpy_async = {Type::getInt8PtrTy(c), Type::getInt8PtrTy(c),
    //size_t count, MemcpyKind kind, RawStream stream
                                       size_t_ty, Type::getInt32Ty(c), Type::getInt8PtrTy(c)};
    make_function(cucorr_memcpy_async, arg_types_memcpy_async);
  }
};

}  // namespace callback

namespace analysis {

llvm::StringSet kCudaKernelInvokes{{"cudaLaunchKernel"}};

llvm::StringSet kCudaDeviceSyncInvokes{{"cudaDeviceSynchronize"}};

llvm::StringSet kCudaEventRecordInvokes{{"cudaEventRecord"}, {"cudaEventRecordWithFlags"}};

llvm::StringSet kCudaEventSyncInvokes{{"cudaEventSynchronize"}};

llvm::StringSet kCudaStreamSyncInvokes{{"cudaStreamSynchronize"}};

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
    if (kCudaKernelInvokes.contains(callee.getName())) {
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
  static short access_cast(AccessState access, bool is_ptr)  {
    auto value = static_cast<short>(access);
    value <<= 1;
    if (is_ptr) {
      value |= 1;
    }
    return value;
  }

  static llvm::Value* get_cu_stream_ptr(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb)  {
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

template <class T>
class SimpleInstrumenter {
  enum class InsertLocation {
    // insert before or after the call that were instrumenting
    kBefore,
    kAfter
  };

  const llvm::FunctionCallee* callee_;
  llvm::StringRef func_name_;
  llvm::SmallVector<llvm::CallBase*, 4> target_callsites_;

 public:
  void setup(llvm::StringRef name, FunctionCallee* callee) {
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
      for (llvm::CallBase* cb : target_callsites_) {
        if (loc == InsertLocation::kBefore) {
          irb.SetInsertPoint(cb);
        } else {
          //NOTE: this only needed if it can be the last instruction which in our case cant be the case since we only check callbases and assume corerct llvmir before?
          if (Instruction* insert_instruction = cb->getNextNonDebugInstruction()) {
            irb.SetInsertPoint(insert_instruction);
          } else {
            irb.SetInsertPoint(cb->getParent());
          }
        }
        if (!cb->arg_empty()) {
          llvm::SmallVector<llvm::Value*> v;
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
    //void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0 
    assert(args.size() == 5);
    auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* src_ptr = irb.CreateBitOrPointerCast(args[1], irb.getInt8PtrTy());
    auto* count = args[2];
    auto* kind = args[3];
    auto* cu_stream = irb.CreateBitOrPointerCast(args[4], irb.getInt8PtrTy());
    return {dst_ptr, src_ptr, count, kind, cu_stream};
  }
};

class MemsetAsyncInstrumenter : public SimpleInstrumenter<MemsetAsyncInstrumenter> {
 public:
  MemsetAsyncInstrumenter(callback::FunctionDecl* decls) {
    setup("cudaMemsetAsync", &decls->cucorr_memcpy_async.f);
  }
  static llvm::SmallVector<Value*, 2> map_arguments(IRBuilder<>& irb, llvm::ArrayRef<Value*> args) {
    //( void* devPtr, int  value, size_t count, cudaStream_t stream = 0 )
    assert(args.size() == 4);
    auto* dst_ptr = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    auto* value = args[1];
    auto* count = args[2];
    auto* cu_stream = irb.CreateBitOrPointerCast(args[3], irb.getInt8PtrTy());
    return {dst_ptr, value, count, cu_stream};
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
    auto* cu_event_void_ptr_ptr  = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
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
    auto* cu_stream_void_ptr_ptr  = irb.CreateBitOrPointerCast(args[0], irb.getInt8PtrTy());
    return {cu_stream_void_ptr_ptr};
  }
};

}  // namespace transform

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

class CucorrPass : public llvm::PassInfoMixin<CucorrPass> {
  cucorr::ModelHandler kernel_models_;
  callback::FunctionDecl cucorr_decls_;

 public:
  llvm::PreservedAnalyses run(llvm::Module&, llvm::ModuleAnalysisManager&);

  bool runOnModule(llvm::Module&);

  bool runOnFunc(llvm::Function&);

  bool runOnKernelFunc(llvm::Function&);
};

class LegacyCucorrPass : public llvm::ModulePass {
 private:
  CucorrPass pass_impl_;

 public:
  static char ID;  // NOLINT

  LegacyCucorrPass() : ModulePass(ID){};

  bool runOnModule(llvm::Module& module) override;

  ~LegacyCucorrPass() override = default;
};

bool LegacyCucorrPass::runOnModule(llvm::Module& module) {
  const auto modified = pass_impl_.runOnModule(module);
  return modified;
}

llvm::PreservedAnalyses CucorrPass::run(llvm::Module& module, llvm::ModuleAnalysisManager&) {
  const auto changed = runOnModule(module);
  return changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
}

bool CucorrPass::runOnModule(llvm::Module& module) {
  cucorr_decls_.initialize(module);
  const auto kernel_models_file = [&]() {
    if (cl_cucorr_kernel_file.getNumOccurrences()) {
      return cl_cucorr_kernel_file.getValue();
    }

    const auto* data_file = getenv("CUCORR_KERNEL_DATA_FILE");
    if (data_file) {
      return std::string{data_file};
    }

    for (llvm::DICompileUnit* cu : module.debug_compile_units()) {
      if (!cu->getFilename().empty()) {
        return std::string{cu->getFilename()} + "-data.yaml";
      }
    }
    return std::string{"cucorr-kernel.yaml"};
  }();

  LOG_DEBUG("Using model data file " << kernel_models_file)
  const auto result       = io::load(this->kernel_models_, kernel_models_file);
  const auto changed      = llvm::count_if(module.functions(), [&](auto& func) {
                         if (cuda::is_kernel(&func)) {
                           return runOnKernelFunc(func);
                         }
                         return runOnFunc(func);
                       }) > 1;
  const auto store_result = io::store(this->kernel_models_, kernel_models_file);
  return changed;
}

bool CucorrPass::runOnKernelFunc(llvm::Function& function) {
  if (function.isDeclaration()) {
    return false;
  }
  auto data = device::analyze_device_kernel(&function);
  if (data) {
    if (!cl_cucorr_quiet.getValue()) {
      LOG_DEBUG("[Device] Kernel data: " << data.value())
    }
    this->kernel_models_.insert(data.value());
  }

  return false;
}

bool CucorrPass::runOnFunc(llvm::Function& function) {
  bool modified = false;
  transform::DeviceSyncInstrumenter(&cucorr_decls_).instrument(function);
  transform::StreamSyncInstrumenter(&cucorr_decls_).instrument(function);
  transform::EventSyncInstrumenter(&cucorr_decls_).instrument(function);
  transform::EventRecordInstrumenter(&cucorr_decls_).instrument(function);
  transform::EventRecordFlagsInstrumenter(&cucorr_decls_).instrument(function);
  transform::EventCreateInstrumenter(&cucorr_decls_).instrument(function);
  transform::StreamCreateInstrumenter(&cucorr_decls_).instrument(function);
  transform::MemsetAsyncInstrumenter(&cucorr_decls_).instrument(function);
  transform::MemcpyAsyncInstrumenter(&cucorr_decls_).instrument(function);
  auto data_for_host = host::kernel_model_for_stub(&function, this->kernel_models_);
  if (data_for_host) {
    CallInstrumenter(analysis::CudaKernelInvokeCollector{data_for_host.value()},
                     transform::KernelInvokeTransformer{&cucorr_decls_}, function)
        .instrument();
  }

  return modified;
}

}  // namespace cucorr

#define DEBUG_TYPE "cucorr-pass"

//.....................
// New PM
//.....................
llvm::PassPluginLibraryInfo getCucorrPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "cucorr", LLVM_VERSION_STRING, [](PassBuilder& pass_builder) {
            pass_builder.registerPipelineParsingCallback(
                [](StringRef name, ModulePassManager& module_pm, ArrayRef<PassBuilder::PipelineElement>) {
                  if (name == "cucorr") {
                    module_pm.addPass(cucorr::CucorrPass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getCucorrPassPluginInfo();
}

//.....................
// Old PM
//.....................
char cucorr::LegacyCucorrPass::ID = 0;  // NOLINT

static RegisterPass<cucorr::LegacyCucorrPass> x("cucorr", "Cucorr Pass");  // NOLINT

ModulePass* createCucorrPass() {
  return new cucorr::LegacyCucorrPass();
}

extern "C" void AddCucorrPass(LLVMPassManagerRef pass_manager) {
  unwrap(pass_manager)->add(createCucorrPass());
}
