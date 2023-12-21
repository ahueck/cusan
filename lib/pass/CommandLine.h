#ifndef CUCORR_COMMANDLINE_H
#define CUCORR_COMMANDLINE_H

#include "llvm/Support/CommandLine.h"

#include <string>

static llvm::cl::OptionCategory cucorr_category("CuCorr instrumentation pass", "These control the instrumentation.");

static llvm::cl::opt<bool> cl_cucorr_quiet("cucorr-quiet", llvm::cl::desc("Keep quiet during pass run"),
                                           llvm::cl::Hidden, llvm::cl::init(false), llvm::cl::cat(cucorr_category));

static llvm::cl::opt<std::string> cl_cucorr_kernel_file("cucorr-kernel-data",
                                                        llvm::cl::desc("Kernel data model file for TU"),
                                                        llvm::cl::cat(cucorr_category));

#endif  // CUCORR_COMMANDLINE_H
