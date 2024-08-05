#ifndef CUSAN_COMMANDLINE_H
#define CUSAN_COMMANDLINE_H

#include "llvm/Support/CommandLine.h"

#include <string>

static llvm::cl::OptionCategory cusan_category("CuCorr instrumentation pass", "These control the instrumentation.");

static llvm::cl::opt<bool> cl_cusan_quiet("cusan-quiet", llvm::cl::desc("Keep quiet during pass run"),
                                           llvm::cl::Hidden, llvm::cl::init(false), llvm::cl::cat(cusan_category));

static llvm::cl::opt<std::string> cl_cusan_kernel_file("cusan-kernel-data",
                                                        llvm::cl::desc("Kernel data model file for TU"),
                                                        llvm::cl::cat(cusan_category));

#endif  // CUSAN_COMMANDLINE_H
