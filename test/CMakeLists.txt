cusan_find_llvm_progs(CUSAN_FILECHECK_EXEC "FileCheck-${LLVM_VERSION_MAJOR};FileCheck" ABORT_IF_MISSING)

if(LLVM_EXTERNAL_LIT)
  cmake_path(GET LLVM_EXTERNAL_LIT PARENT_PATH LLVM_EXTERNAL_LIT_DIR)
endif()

cusan_find_llvm_progs(CUSAN_LIT_EXEC
  "llvm-lit;lit;lit.py"
  HINTS ${LLVM_EXTERNAL_LIT_DIR} /usr/lib/llvm-${LLVM_VERSION_MAJOR} /usr/lib/llvm /usr/bin /usr/local/bin /opt/local/bin
  ABORT_IF_MISSING
)

macro(pythonize_bool truth_var var)
  if(${truth_var})
    set(${var} True)
  else()
    set(${var} False)
  endif()
endmacro()

function(configure_cusan_lit_site_cfg input output)
  set(LIT_SITE_CFG_IN_HEADER
      "## Autogenerated for cusan from ${input}\n## Do not edit!"
  )
  
  set(CUSAN_PROJECT_DIR ${PROJECT_SOURCE_DIR})
  set(CUSAN_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)
  set(CUSAN_PASS_DIR ${CUSAN_LIBRARY_DIR}/pass)
  set(CUSAN_RUNTIME_DIR ${CUSAN_LIBRARY_DIR}/runtime)
  set(CUSAN_SCRIPT_DIR ${CMAKE_BINARY_DIR}/scripts)
  set(CUSAN_BUILD_DIR ${CMAKE_BINARY_DIR})

  if(${LLVM_VERSION_MAJOR} VERSION_GREATER_EQUAL "13")
    set(CUSAN_OPT_ARGS "-enable-new-pm=0")
  endif()

  cusan_target_generate_file(${input} ${output})
endfunction()

configure_cusan_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
)

function(cusan_add_lit_testsuite target comment)
  cmake_parse_arguments(ARG "" "" "PARAMS;DEPENDS;ARGS" ${ARGN})

  foreach(param ${ARG_PARAMS})
    list(APPEND TEST_PARAMS --param ${param})
  endforeach()

  add_custom_target(${target}
    COMMAND ${CUSAN_LIT_EXEC} ${ARG_ARGS} ${TEST_PARAMS} ${ARG_UNPARSED_ARGUMENTS}
    COMMENT "${comment}"
    USES_TERMINAL
  )

  if (ARG_DEPENDS)
    add_dependencies(${target} ${ARG_DEPENDS})
  endif()
endfunction()

function(cusan_add_lit_target)
  cmake_parse_arguments(ARG "" "" "SUITES" ${ARGN})

  foreach(suite IN LISTS ARG_SUITES)
    if("${suite}" STREQUAL "all")
      set(SUITE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
      set(TARGET_NAME check-cusan)
    else()
      set(SUITE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${suite})
      set(TARGET_NAME check-cusan-${suite})
    endif()

    if(NOT EXISTS ${SUITE_PATH})
      message(WARNING "Could not find suitable lit test target at ${SUITE_PATH}")
      continue()
    endif()

    cusan_add_lit_testsuite(${TARGET_NAME}
      "Running the lit suite cusan::${suite}"
      ${SUITE_PATH}
      ARGS -v -j 1
      PARAMS cusan_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
      DEPENDS ${CUSAN_TEST_DEPENDS}
    )
  endforeach()
endfunction()

set(CUSAN_TEST_DEPENDS
  cusan::TransformPass
  cusan::MPI_Interceptor
  cusan::Runtime
  cusan::Analysis
  typeart::Runtime
  typeart::TransformPass
)

set(CUSAN_SUITES
  all
  pass
  tsan
)

cusan_add_lit_target(SUITES ${CUSAN_SUITES})

add_test(
  NAME cusan-lit-suite
  COMMAND
    ${CUSAN_LIT_EXEC} -j 1 --param
    cusan_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    ${CMAKE_CURRENT_SOURCE_DIR}
)

cusan_target_format(
  format-cusan-tests "Formats project test files"
  TARGETS *.c *.cpp *.h
)

if(CUSAN_TEST_CONFIGURE_IDE)
  function(add_test_target_ide target header sources compile_std linker_language)
    if(NOT sources)
      return()
    endif()

    add_executable(${target} EXCLUDE_FROM_ALL ${header} ${sources})

    target_include_directories(
      ${target} PRIVATE ${PROJECT_SOURCE_DIR}/lib/pass
                        ${PROJECT_SOURCE_DIR}/lib
                        ${PROJECT_SOURCE_DIR}
    )
    target_compile_features(${target} PUBLIC ${compile_std})
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE ${linker_language})
  endfunction()

  file(GLOB_RECURSE CUSAN_CXX_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
  file(GLOB_RECURSE CUSAN_CC_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
  file(GLOB_RECURSE CUSAN_HEADER_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

  add_test_target_ide(cusan_cxx_test_objects
    "${CUSAN_HEADER_TESTS}" "${CUSAN_CXX_TESTS}"
    cxx_std_17 CXX
  )
  add_test_target_ide(cusan_cc_test_objects
    "${CUSAN_HEADER_TESTS}" "${CUSAN_CC_TESTS}"
    c_std_99 C
  )
endif()
