include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(FeatureSummary)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE STRING "" FORCE)
include(FetchContent)

find_package(LLVM CONFIG HINTS "${LLVM_DIR}")
if(NOT LLVM_FOUND)
  message(STATUS "LLVM not found at: ${LLVM_DIR}.")
  find_package(LLVM REQUIRED CONFIG)
endif()

set_package_properties(LLVM PROPERTIES
  URL https://llvm.org/
  TYPE REQUIRED
  PURPOSE
  "LLVM framework installation required to compile (and apply) project cucorr."
)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}"
  PROJECT_IS_TOP_LEVEL
)

FetchContent_Declare(
  typeart
  GIT_REPOSITORY https://github.com/tudasc/TypeART.git
  GIT_TAG feat/cuda
  GIT_SHALLOW 1
)
FetchContent_MakeAvailable(typeart)

option(CUCORR_TEST_CONFIGURE_IDE "Add targets for tests to help the IDE with completion etc." ON)
mark_as_advanced(CUCORR_TEST_CONFIGURE_IDE)
option(CUCORR_CONFIG_DIR_IS_SHARE "Install to \"share/cmake/\" instead of \"lib/cmake/\"" OFF)
mark_as_advanced(CUCORR_CONFIG_DIR_IS_SHARE)

set(warning_guard "")
if(NOT PROJECT_IS_TOP_LEVEL)
  option(
      CUCORR_INCLUDES_WITH_SYSTEM
      "Use SYSTEM modifier for cucorr includes to disable warnings."
      ON
  )
  mark_as_advanced(CUCORR_INCLUDES_WITH_SYSTEM)

  if(CUCORR_INCLUDES_WITH_SYSTEM)
    set(warning_guard SYSTEM)
  endif()
endif()

include(modules/cucorr-llvm)
include(modules/cucorr-format)
include(modules/cucorr-target-util)

cucorr_find_llvm_progs(CUCORR_CLANG_EXEC "clang-${LLVM_VERSION_MAJOR};clang" DEFAULT_EXE "clang")
cucorr_find_llvm_progs(CUCORR_CLANGCXX_EXEC "clang-${LLVM_VERSION_MAJOR};clang++" DEFAULT_EXE "clang++")
cucorr_find_llvm_progs(CUCORR_LLC_EXEC "llc-${LLVM_VERSION_MAJOR};llc" DEFAULT_EXE "llc")
cucorr_find_llvm_progs(CUCORR_OPT_EXEC "opt-${LLVM_VERSION_MAJOR};opt" DEFAULT_EXE "opt")

if(PROJECT_IS_TOP_LEVEL)
  if(NOT CMAKE_BUILD_TYPE)
    # set default build type
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
    message(STATUS "Building as debug (default)")
  endif()

  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # set default install path
    set(CMAKE_INSTALL_PREFIX
        "${cucorr_SOURCE_DIR}/install/cucorr"
        CACHE PATH "Default install path" FORCE
    )
    message(STATUS "Installing to (default): ${CMAKE_INSTALL_PREFIX}")
  endif()

    # CUCORR_DEBUG_POSTFIX is only used for Config
    if(CMAKE_DEBUG_POSTFIX)
      set(CUCORR_DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    else()
      set(CUCORR_DEBUG_POSTFIX "-d")
    endif()

  if(NOT CMAKE_DEBUG_POSTFIX AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_DEBUG_POSTFIX ${CUCORR_DEBUG_POSTFIX})
  endif()
else()
  set(CUCORR_DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
endif()

include(GNUInstallDirs)

set(CUCORR_PREFIX ${PROJECT_NAME})
set(TARGETS_EXPORT_NAME ${CUCORR_PREFIX}Targets)

if(CUCORR_CONFIG_DIR_IS_SHARE)
  set(CUCORR_INSTALL_CONFIGDIR ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME})
else()
  set(CUCORR_INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
endif()
