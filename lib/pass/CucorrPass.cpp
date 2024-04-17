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
  CucorrFunction cucorr_register_access_n{"_cucorr_kernel_register_n"};

  CucorrFunction cucorr_event_record{"_cucorr_event_record"};

  CucorrFunction cucorr_sync_device{"_cucorr_sync_device"};
  CucorrFunction cucorr_sync_stream{"_cucorr_sync_stream"};
  CucorrFunction cucorr_sync_event{"_cucorr_sync_event"};

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
    ArgTypes arg_types_cucorr_register = {Type::getInt8PtrTy(c), Type::getInt16Ty(c), Type::getInt8PtrTy(c)};
    make_function(cucorr_register_access, arg_types_cucorr_register);
    // TODO address space?
    ArgTypes arg_types_cucorr_register_n = {PointerType::get(PointerType::get(Type::getInt8PtrTy(c), 0), 0),
                                            Type::getInt16PtrTy(c), Type::getInt32Ty(c), Type::getInt8PtrTy(c)};
    make_function(cucorr_register_access_n, arg_types_cucorr_register_n);

    ArgTypes arg_types_sync_device = {};
    make_function(cucorr_sync_device, arg_types_sync_device);

    ArgTypes arg_types_sync_stream = {Type::getInt8PtrTy(c)};
    make_function(cucorr_sync_stream, arg_types_sync_stream);

    ArgTypes arg_types_sync_event = {Type::getInt8PtrTy(c)};
    make_function(cucorr_sync_event, arg_types_sync_event);
    ArgTypes arg_types_event_record = {Type::getInt8PtrTy(c), Type::getInt8PtrTy(c)};
    make_function(cucorr_event_record, arg_types_event_record);
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



struct CudaKernelInvokeCollector{
  KernelModel model;
  struct KernelInvokeData{
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

class DeviceSyncInvokeCollector {
 public:
  struct Data {};
  llvm::Optional<Data> match(llvm::CallBase& cb, Function& callee) const {
    if (kCudaDeviceSyncInvokes.contains(callee.getName())) {
      return Data{};
    }
    return llvm::NoneType();
  }
};

class EventRecordCollector {
 public:
  struct Data {
    llvm::Value* cu_event{nullptr};
    llvm::Value* cu_stream{nullptr};
    llvm::Optional<llvm::Value*> flags{nullptr};
  };
  llvm::Optional<Data> match(llvm::CallBase& cb, Function& callee) const {
    if (kCudaEventRecordInvokes.contains(callee.getName())) {
      llvm::Optional<llvm::Value*> flags = llvm::NoneType();
      if(callee.arg_size() == 3){
        flags = std::next(cb.arg_begin(), 2)->get();
      }
      
      return Data{
        cb.arg_begin()->get(),  
        std::next(cb.arg_begin())->get(),  
        flags
      };
    }
    return llvm::NoneType();
  }
};



class EventSynchronizeCollector {
 public:
  struct Data {
    llvm::Value* cu_event{nullptr};
  };
  llvm::Optional<Data> match(llvm::CallBase& cb, Function& callee) const {
    if (kCudaEventSyncInvokes.contains(callee.getName())) {
      return Data{
        cb.arg_begin()->get(),  
      };
    }
    return llvm::NoneType();
  }
};

class StreamSynchronizeCollector {
 public:
  struct Data {
    llvm::Value* cu_stream{nullptr};
  };
  llvm::Optional<Data> match(llvm::CallBase& cb, Function& callee) const {
    if (kCudaStreamSyncInvokes.contains(callee.getName())) {
      return Data{
        cb.arg_begin()->get(),  
      };
    }
    return llvm::NoneType();
  }
};

}  // namespace analysis

namespace transform {


class StreamSynchronizeTransformer {
 public:
  callback::FunctionDecl* decls;
  bool transform(const analysis::StreamSynchronizeCollector::Data& data, IRBuilder<>& irb) const {
    auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(data.cu_stream, irb.getInt8PtrTy());
    Value* args[] = {cu_stream_void_ptr};
    irb.CreateCall(decls->cucorr_sync_stream.f, args);
    return true;
  }
};

class EventSynchronizeTransformer {
 public:
  callback::FunctionDecl* decls;
  bool transform(const analysis::EventSynchronizeCollector::Data& data, IRBuilder<>& irb) const {
    auto* cu_event_void_ptr = irb.CreateBitOrPointerCast(data.cu_event, irb.getInt8PtrTy());
    Value* args[] = {cu_event_void_ptr};
    irb.CreateCall(decls->cucorr_sync_event.f, args);
    return true;
  }
};

class EventRecordTransformer {
 public:
  callback::FunctionDecl* decls;
  bool transform(const analysis::EventRecordCollector::Data& data, IRBuilder<>& irb) const {
    // TODO: do we care about the flags
    auto* cu_event_void_ptr = irb.CreateBitOrPointerCast(data.cu_event, irb.getInt8PtrTy());
    auto* cu_stream_void_ptr = irb.CreateBitOrPointerCast(data.cu_stream, irb.getInt8PtrTy());
    Value* args[] = {cu_event_void_ptr, cu_stream_void_ptr};
    irb.CreateCall(decls->cucorr_event_record.f, args);
    return true;
  }
};

class DeviceSyncInvokeTransformer {
 public:
  callback::FunctionDecl* decls;
  bool transform(const analysis::DeviceSyncInvokeCollector::Data&, IRBuilder<>& irb) const {
    irb.CreateCall(decls->cucorr_sync_device.f);
    return true;
  }
};

struct KernelInvokeTransformer {
  callback::FunctionDecl* decls_;

  KernelInvokeTransformer(callback::FunctionDecl* decls) : decls_(decls) {
  }

  bool transform(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb) const {
    using namespace llvm;
    generate_compound_cb(data, irb);
    return generate_single_cb(data, irb);
  }

 private:
  short access_cast(AccessState access, bool is_ptr) const {
    short value = static_cast<short>(access);
    value <<= 1;
    if (is_ptr) {
      value |= 1;
    }
    return value;
  }

  llvm::Value* get_cu_stream_ptr(const analysis::CudaKernelInvokeCollector::Data& data, IRBuilder<>& irb) const {
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

    auto target_callback = decls_->cucorr_register_access_n;

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

  bool generate_single_cb(const analysis::CudaKernelInvokeCollector::KernelInvokeData& data, IRBuilder<>& irb) const {
    const bool should_transform = llvm::count_if(data.args, [&](const auto& elem) { return elem.is_pointer; }) > 0;

    if (!should_transform) {
      return false;
    }

    auto target_callback    = decls_->cucorr_register_access;
    auto cu_stream_void_ptr = get_cu_stream_ptr(data, irb);
    for (const auto& arg : data.args) {
      if (arg.is_pointer) {
        auto* pointer     = const_cast<Value*>(dyn_cast<Value>(arg.arg.getValue()));
        auto* void_ptr    = irb.CreateBitOrPointerCast(pointer, irb.getInt8PtrTy());
        const auto access = access_cast(arg.state, arg.is_pointer);

        if (access == static_cast<short>(AccessState::kNone)) {
          continue;
        }
        auto const_access_val = irb.getInt16(access);
        //        LOG_DEBUG("Insert " << *void_ptr << " access " << *const_access_val << " and stream " <<
        //        *cu_stream_void_ptr)
        Value* args_cucorr_register[] = {void_ptr, const_access_val, cu_stream_void_ptr};
        irb.CreateCall(target_callback.f, args_cucorr_register);
      }
    }
    return true;
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
    for (inst_iterator I = inst_begin(f_), E = inst_end(f_); I != E; ++I) {
      if (auto* cb = dyn_cast<CallBase>(&*I)) {
        if (auto* f = cb->getCalledFunction()) {
          auto t = collector_.match(*cb, *f);
          if (t.hasValue()) {
            data_vec_.push_back({t.getValue(), cb});
          }
        }
      }
    }

    // iffy
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
  CallInstrumenter(analysis::DeviceSyncInvokeCollector{}, transform::DeviceSyncInvokeTransformer{&cucorr_decls_},
                   function)
      .instrument();
  CallInstrumenter(analysis::EventRecordCollector{}, transform::EventRecordTransformer{&cucorr_decls_},
                   function)
      .instrument();
  CallInstrumenter(analysis::EventSynchronizeCollector{}, transform::EventSynchronizeTransformer{&cucorr_decls_},
                   function)
      .instrument();
  CallInstrumenter(analysis::StreamSynchronizeCollector{}, transform::StreamSynchronizeTransformer{&cucorr_decls_},
                   function)
      .instrument();
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
