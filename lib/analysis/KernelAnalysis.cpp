//
// Created by ahueck on 05.07.23.
//

#include "KernelAnalysis.h"

#include "support/CudaUtil.h"
#include "support/Logger.h"
#include "support/Util.h"

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/IPO/Attributor.h>
#include <utility>

namespace cucorr {

namespace device {

// stolen and modified from clang19 https://llvm.org/doxygen/FunctionAttrs_8cpp_source.html#l00611
static llvm::Attribute::AttrKind determinePointerAccessAttrs(llvm::Value* A) {
  using namespace llvm;
  SmallVector<Use*, 32> worklist;
  SmallPtrSet<Use*, 32> visited;

  bool is_read  = false;
  bool is_write = false;

  for (Use& U : A->uses()) {
    visited.insert(&U);
    worklist.push_back(&U);
  }

  while (!worklist.empty()) {
    if (is_write && is_read)
      // No point in searching further..
      return Attribute::None;

    Use* U         = worklist.pop_back_val();
    Instruction* I = cast<Instruction>(U->getUser());

    switch (I->getOpcode()) {
      case Instruction::BitCast:
      case Instruction::GetElementPtr:
      case Instruction::PHI:
      case Instruction::Select:
      case Instruction::AddrSpaceCast:
        // The original value is not read/written via this if the new value isn't.
        for (Use& UU : I->uses())
          if (visited.insert(&UU).second)
            worklist.push_back(&UU);
        break;

      case Instruction::Call:
      case Instruction::Invoke: {
        auto& CB = cast<CallBase>(*I);
        if (CB.isCallee(U)) {
          is_read = true;
          // Note that indirect calls do not capture, see comment in
          // CaptureTracking for context
          continue;
        }

        // Given we've explictily handled the callee operand above, what's left
        // must be a data operand (e.g. argument or operand bundle)
        const unsigned use_index = CB.getDataOperandNo(U);

        // Some intrinsics (for instance ptrmask) do not capture their results,
        // but return results thas alias their pointer argument, and thus should
        // be handled like GEP or addrspacecast above.
        if (isIntrinsicReturningPointerAliasingArgumentWithoutCapturing(&CB, /*MustPreserveNullness=*/false)) {
          for (Use& UU : CB.uses())
            if (visited.insert(&UU).second)
              worklist.push_back(&UU);
        } else if (!CB.doesNotCapture(use_index)) {
          if (!CB.onlyReadsMemory())
            // If the callee can save a copy into other memory, then simply
            // scanning uses of the call is insufficient.  We have no way
            // of tracking copies of the pointer through memory to see
            // if a reloaded copy is written to, thus we must give up.
            return Attribute::None;
          // Push users for processing once we finish this one
          if (!I->getType()->isVoidTy())
            for (Use& UU : I->uses())
              if (visited.insert(&UU).second)
                worklist.push_back(&UU);
        }

        // The accessors used on call site here do the right thing for calls and
        // invokes with operand bundles.
        if (CB.doesNotAccessMemory(use_index)) {
          /* nop */
        } else if (CB.onlyReadsMemory(use_index)) {
          is_read = true;
        } else if (CB.dataOperandHasImpliedAttr(use_index, Attribute::WriteOnly)) {
          is_write = true;
        } else {
          return Attribute::None;
        }
        break;
      }

      case Instruction::Load:
        // A volatile load has side effects beyond what readonly can be relied
        // upon.
        if (cast<LoadInst>(I)->isVolatile())
          return Attribute::None;

        is_read = true;
        break;

      case Instruction::Store:
        if (cast<StoreInst>(I)->getValueOperand() == *U)
          // untrackable capture
          return Attribute::None;

        // A volatile store has side effects beyond what writeonly can be relied
        // upon.
        if (cast<StoreInst>(I)->isVolatile())
          return Attribute::None;

        is_write = true;
        break;

      case Instruction::ICmp:
      case Instruction::Ret:
        break;

      default:
        return Attribute::None;
    }
  }

  if (is_write && is_read)
    return Attribute::None;
  if (is_read)
    return Attribute::ReadOnly;
  if (is_write)
    return Attribute::WriteOnly;
  return Attribute::ReadNone;
}

inline AccessState state(const llvm::AAMemoryBehavior& mem) {
  if (mem.isAssumedReadNone()) {
    return AccessState::kNone;
  }
  if (mem.isAssumedReadOnly()) {
    return AccessState::kRead;
  }
  if (mem.isAssumedWriteOnly()) {
    return AccessState::kWritten;
  }
  return AccessState::kRW;
}

inline AccessState state(const llvm::Attribute::AttrKind mem) {
  using namespace llvm;
  if (mem == Attribute::ReadNone) {
    return AccessState::kNone;
  }
  if (mem == Attribute::ReadOnly) {
    return AccessState::kRead;
  }
  if (mem == Attribute::WriteOnly) {
    return AccessState::kWritten;
  }
  return AccessState::kRW;
}

struct ChildInfo {
  llvm::Value* val;
  llvm::SmallVector<int32_t> indicies;
};

void collect_children(llvm::SmallVector<FunctionArg, 4>& args, llvm::Value* init_val, unsigned arg_pos) {
  using namespace llvm;
  llvm::SmallVector<ChildInfo, 32> workList;
  workList.push_back({init_val, {}});

  while (!workList.empty()) {
    //not nice making copies of the stack all the time idk
    auto curr_info = workList.pop_back_val();
    auto* value = curr_info.val;
    auto index_stack = curr_info.indicies;

    Type* value_type = value->getType();
    if (auto* ptr_type = dyn_cast<PointerType>(value_type)) {
      auto* elem_type = ptr_type->getPointerElementType();
      if (elem_type->isStructTy()) {
        for (User* u : value->users()) {
          if (auto* gep = dyn_cast<GetElementPtrInst>(u)) {
            auto gep_indicies = gep->indices();
            for (unsigned i = 1; i < gep->getNumIndices(); i++) {
              auto* index       = gep_indicies.begin() + i;
              auto* index_value = dyn_cast<ConstantInt>(index->get());
              // TODO: handle gracefully if indexing into array and similar "dynamic" geps
              assert(index_value);
              index_stack.push_back((int32_t)index_value->getSExtValue());
            }

            workList.push_back({gep, index_stack});
            //collect_children(args, gep, index_stack, arg_pos);

            //for (unsigned i = 1; i < gep->getNumIndices(); i++) {
            //  index_stack.pop_back();
            //}
          }
        }
      }

      for (User* u : value->users()) {
        if (auto* load = dyn_cast<LoadInst>(u)) {
          index_stack.push_back(-1);
          workList.push_back({load, index_stack});
          //collect_children(args, load, index_stack, arg_pos);
          const auto res = determinePointerAccessAttrs(load);
          const FunctionArg kernel_arg{load, index_stack, arg_pos, true, state(res)};
          args.push_back(kernel_arg);
          //index_stack.pop_back();
        }
      }

    } else {
      return;
    }
  }
}

void attribute_value(llvm::SmallVector<FunctionArg, 4>& args, llvm::Value* value, llvm::Attributor& attrib,
                     unsigned arg_pos) {
  using namespace llvm;
  Type* value_type = value->getType();
  llvm::errs() << "Attributing Value: " << *value << " of type: " << *value_type << "\n";

  if (value_type->isPointerTy()) {
    IRPosition const val_pos = IRPosition::value(*value);
    const auto& mem_behavior = attrib.getOrCreateAAFor<AAMemoryBehavior>(val_pos);
    const auto res2          = determinePointerAccessAttrs(value);
    llvm::errs() << "   secRes: None:" << (res2 == Attribute::None) << " ReadNone" << (res2 == Attribute::ReadNone)
                 << " ReadOnly" << (res2 == Attribute::ReadOnly) << " WriteOnly" << (res2 == Attribute::WriteOnly)
                 << "\n";
    llvm::errs() << "     isValid:" << mem_behavior.isValidState() << "\n";
    llvm::errs() << "    Got ptr: KnownReadNone:" << mem_behavior.isKnownReadNone()
                 << " KnownReadOnly:" << mem_behavior.isKnownReadOnly()
                 << " KnownWriteOnly:" << mem_behavior.isKnownWriteOnly() << "\n";
    llvm::errs() << "    Got ptr: ReadNone:" << mem_behavior.isAssumedReadNone()
                 << " ReadOnly:" << mem_behavior.isAssumedReadOnly()
                 << " WriteOnly:" << mem_behavior.isAssumedWriteOnly() << "\n";
    const FunctionArg kernel_arg{value, {}, arg_pos, true, state(res2)};
    args.emplace_back(kernel_arg);
    collect_children(args, value, arg_pos);
  } else {
    const FunctionArg kernel_arg{value, {}, arg_pos, false, AccessState::kRW};
    args.emplace_back(kernel_arg);
  }
}

std::optional<KernelModel> info_with_attributor(llvm::Function* kernel) {
  using namespace llvm;

  auto* module = kernel->getParent();
  AnalysisGetter ag;
  SetVector<Function*> functions;
  for (auto& module_f : module->functions()) {
    functions.insert(&module_f);
  }
  CallGraphUpdater cg_updater;
  BumpPtrAllocator allocator;
  InformationCache info_cache(*module, ag, allocator, /* CGSCC */ nullptr);

  Attributor attrib(functions, info_cache, cg_updater);

  errs() << "Attributing " << kernel->getName() << "\n" << *kernel << "\n";

  llvm::SmallVector<FunctionArg, 4> args{};
  for (const auto& arg : llvm::enumerate(kernel->args())) {
    llvm::SmallVector<int32_t> index_stack = {};
    attribute_value(args, &arg.value(), attrib, arg.index());
  }

  KernelModel model{kernel, std::string{kernel->getName()}, (unsigned int)kernel->arg_size(), args};

  return model;
}

std::optional<KernelModel> analyze_device_kernel(llvm::Function* f) {
  if (!cuda::is_kernel(f)) {
    assert(f != nullptr && "Function should not be null here!");
    LOG_DEBUG("Function is not a kernel " << f->getName())
    return {};
  }
  using namespace llvm;
  const auto kernel_model = info_with_attributor(f);
  return kernel_model;
}

}  // namespace device

namespace host {

std::optional<KernelModel> kernel_model_for_stub(llvm::Function* f, const ModelHandler& models) {
  const auto stub_name = [&](const auto& name) {
    auto stub_name    = std::string{name};
    const auto prefix = std::string{"__device_stub__"};
    const auto pos    = stub_name.find(prefix);
    if (pos != std::string::npos) {
      stub_name.erase(pos, prefix.length());
    }
    return stub_name;
  }(util::try_demangle(*f));

  const auto result = llvm::find_if(models.models, [&stub_name](const auto& model_) {
    if (llvm::StringRef(util::demangle(model_.kernel_name)).startswith(stub_name)) {
      return true;
    }
    return false;
  });

  if (result != std::end(models.models)) {
    LOG_DEBUG("Found fitting kernel data " << *result)
    return *result;
  }

  LOG_DEBUG("Found no kernel data for stub: " << stub_name)
  return {};
}

}  // namespace host

}  // namespace cucorr
