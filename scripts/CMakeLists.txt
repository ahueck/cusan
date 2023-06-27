function(configure_cucorr_script input output)
  cmake_parse_arguments(
    ARG "" "" "INSTALL_MODE;COMPILER;WITH_FILTER;APPLY_MODE" ${ARGN}
  )
  set(TYPEART_SAN_FLAGS "")

  set(TYPEART_PROJECT_DIR ${typeart_SOURCE_DIR})
  set(TYPEART_BIN_DIR ${typeart_BINARY_DIR})
  set(TYPEART_SCRIPT_DIR ${TYPEART_PROJECT_DIR}/scripts)

  if(ARG_INSTALL_MODE)
    set(TYPEART_INCLUDE_DIRS
        "-I${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
    )
    set(TYPEART_RT_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set(TYPEART_PASS_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set(TYPEART_ANALYSIS_PASS_DIR
        ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
    )
    set(TYPEART_BINARY_DIR -I${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
    set(TYPEART_RELOCATABLE 1)
  else()
    set(TYPEART_INCLUDE_DIRS
        "-I${TYPEART_PROJECT_DIR}/lib/typelib -I${TYPEART_PROJECT_DIR}/lib/runtime -I${TYPEART_PROJECT_DIR}/lib/passes/typegen"
    )
    if(LIBRARY_OUTPUT_PATH)
      set(TYPEART_MPI_INTERCEPT_DIR ${LIBRARY_OUTPUT_PATH})
      set(TYPEART_RT_DIR ${LIBRARY_OUTPUT_PATH})
      set(TYPEART_PASS_DIR ${LIBRARY_OUTPUT_PATH})
      set(TYPEART_ANALYSIS_PASS_DIR ${LIBRARY_OUTPUT_PATH})
    else()
      set(TYPEART_MPI_INTERCEPT_DIR ${TYPEART_BIN_DIR}/lib/mpi_interceptor)
      set(TYPEART_RT_DIR ${TYPEART_BIN_DIR}/lib/runtime)
      set(TYPEART_PASS_DIR ${TYPEART_BIN_DIR}/lib/passes)
      set(TYPEART_ANALYSIS_PASS_DIR ${TYPEART_BIN_DIR}/analysis)

      set(CUCORR_RT_DIR ${CMAKE_BINARY_DIR}/lib/runtime)
      set(CUCORR_PASS_DIR ${CMAKE_BINARY_DIR}/lib/pass)
    endif()

    if(EXECUTABLE_OUTPUT_PATH)
        set(TYPEART_BINARY_DIR ${EXECUTABLE_OUTPUT_PATH})
    else()
        set(TYPEART_BINARY_DIR ${TYPEART_BIN_DIR})
        set(CUCORR_BINARY_DIR ${CMAKE_BINARY_DIR})
    endif()

    set(TYPEART_RELOCATABLE 0)
  endif()

  if(ARG_COMPILER)
    set(TYPEART_COMPILER ${ARG_COMPILER})
  endif()

  set(TYPEART_OPT "${TYPEART_OPT_EXEC}")
  if(${LLVM_VERSION_MAJOR} VERSION_GREATER_EQUAL "13")
    set(TYPEART_OPT "${TYPEART_OPT} -enable-new-pm=0")
  endif()

  set(TYPEART_LLC "${TYPEART_LLC_EXEC}")

  if(ARG_APPLY_MODE)
    set(TYPEART_RUN_SCRIPT 0)
  else()
    set(TYPEART_RUN_SCRIPT 1)
  endif()

  if(ARG_WITH_FILTER)
    set(TYPEART_CALLFILTER "--typeart-filter")
  endif()

  if(TYPEART_TSAN)
    typeart_target_tsan_flags(TYPEART_SAN_FLAGS)
  endif()

  if(TYPEART_ASAN)
    typeart_target_asan_flags(asan_flags)
    set(TYPEART_SAN_FLAGS ${TYPEART_SAN_FLAGS} ${asan_flags})
  endif()

  if(TYPEART_UBSAN)
    typeart_target_ubsan_flags(ubsan_flags)
    set(TYPEART_SAN_FLAGS ${TYPEART_SAN_FLAGS} ${ubsan_flags})
  endif()

  list(JOIN TYPEART_SAN_FLAGS " " TYPEART_SAN_FLAGS)

  set(CUCORR_INCLUDE_DIRS "-I${PROJECT_SOURCE_DIR}/lib/runtime")

  if(ARG_APPLY_MODE)
    set(CUCORR_RUN_SCRIPT 0)
  else()
    set(CUCORR_RUN_SCRIPT 1)
  endif()

  cucorr_target_generate_file(${input} ${output})
endfunction()

function(typeart_find_mpi_vendor_helper symbol ret_value)
  find_package(MPI)

  if(NOT MPI_FOUND)
    set(${ret_value} false)
    return()
  endif()

  set(MPI_TEST_CODE
   "#include <mpi.h>
    int main(void) {
      #if !defined(${symbol})
        an_illegal_stmt
      #endif
      return 0;
    }")

  set(CMAKE_REQUIRED_INCLUDES ${MPI_C_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_FLAGS "${MPI_C_COMPILE_FLAGS}")
  set(CMAKE_REQUIRED_DEFINITIONS "")
  set(CMAKE_REQUIRED_LIBRARIES "${MPI_C_LIBRARIES}")

  check_c_source_compiles("${MPI_TEST_CODE}" ${ret_value})
endfunction()

function(set_typeart_mpi_compiler)
  # Note on mpich & Intel: Intel may also define \"MPICH_NAME\"
  # in mpi.h, so check MPICH last
  list(APPEND l_vendor OPEN_MPI INTEL_MPI MPICH)
  list(APPEND l_symbol OPEN_MPI I_MPI_VERSION MPICH_NAME)
  list(APPEND l_env_c OMPI_CC I_MPI_CC MPICH_CC)
  list(APPEND l_env_cxx OMPI_CXX I_MPI_CXX MPICH_CXX)

  foreach(vendor symbol env_c env_cxx IN ZIP_LISTS l_vendor l_symbol l_env_c l_env_cxx)
    typeart_find_mpi_vendor_helper(${symbol} TYPEART_HAVE_${vendor})
    if(TYPEART_HAVE_${vendor})
      if(MPI_C_FOUND)
        set(TYPEART_MPICC
            "env ${env_c}=${TYPEART_CLANG_EXEC} ${MPI_C_COMPILER}"
            CACHE STRING "TypeART MPICC compiler command for scripts"
          )
        mark_as_advanced(TYPEART_MPICC)
      endif()
      if(MPI_CXX_FOUND)
        set(TYPEART_MPICXX
            "env ${env_cxx}=${TYPEART_CLANGCXX_EXEC} ${MPI_CXX_COMPILER}"
            CACHE STRING "TypeART MPICXX compiler command for scripts"
        )
        mark_as_advanced(TYPEART_MPICXX)
      endif()
      break()
    endif()
  endforeach()
endfunction()

if(MPI_FOUND)
  set_typeart_mpi_compiler()
endif()

find_package(MPI)
if(MPI_C_FOUND)
  configure_cucorr_script(
    cucorr-wrapper.in cucorr-mpicc${CMAKE_DEBUG_POSTFIX}
    INSTALL_MODE ON
    WITH_FILTER ON
    COMPILER "${TYPEART_MPICC}"
  )
  configure_cucorr_script(
    cucorr-wrapper.in cucorr-mpicc-test
    WITH_FILTER ON
    COMPILER "${TYPEART_MPICC}"
  )
endif()

if(MPI_CXX_FOUND)
  configure_cucorr_script(
    cucorr-wrapper.in cucorr-mpic++${CMAKE_DEBUG_POSTFIX}
    INSTALL_MODE ON
    WITH_FILTER ON
    COMPILER "${TYPEART_MPICXX}"
  )
  configure_cucorr_script(
    cucorr-wrapper.in cucorr-mpic++-test
    WITH_FILTER ON
    COMPILER "${TYPEART_MPICXX}"
  )
endif()

configure_cucorr_script(
  cucorr-wrapper.in cucorr-clang${CMAKE_DEBUG_POSTFIX}
  INSTALL_MODE ON
  COMPILER ${TYPEART_CLANG_EXEC}
)
configure_cucorr_script(
  cucorr-wrapper.in cucorr-clang++${CMAKE_DEBUG_POSTFIX}
  INSTALL_MODE ON
  COMPILER ${TYPEART_CLANGCXX_EXEC}
)

configure_cucorr_script(
  cucorr-wrapper.in cucorr-clang-test
  COMPILER ${TYPEART_CLANG_EXEC}
)
configure_cucorr_script(
  cucorr-wrapper.in cucorr-clang++-test
  COMPILER ${TYPEART_CLANGCXX_EXEC}
)

#configure_cucorr_script(cucorr-tmpl.sh.in run.sh)
configure_cucorr_script(
  cucorr-tmpl.sh.in apply.sh
  APPLY_MODE ON
)

configure_cucorr_script(cucorr-wrapper.in cucorr-clang.sh)
