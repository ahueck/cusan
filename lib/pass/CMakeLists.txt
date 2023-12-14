set(PASS_SOURCES
  CucorrPass.cpp
  ../analysis/KernelMemory.cpp
  ../analysis/KernelMemory.h
  ../analysis/KernelDataManager.cpp
)

cucorr_llvm_module(cucorr_TransformPass
  "${PASS_SOURCES}"
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/lib
)
mark_as_advanced(LLVM_CUCORR_TRANSFORMPASS_LINK_INTO_TOOLS)

set_target_properties(
  cucorr_TransformPass
  PROPERTIES
  OUTPUT_NAME "${PROJECT_NAME}TransformPass"
  EXPORT_NAME "TransformPass"
)

add_library(cucorr::TransformPass ALIAS cucorr_TransformPass)

target_compile_features(cucorr_TransformPass PUBLIC cxx_std_17)

target_include_directories(cucorr_TransformPass ${warning_guard}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
          ${PROJECT_SOURCE_DIR}/lib/
)

set(CONFIG_NAME cucorrTransformPass)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(
  TARGETS cucorr_TransformPass
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  EXPORT ${TARGETS_EXPORT_NAME}
  NAMESPACE cucorr::
  DESTINATION ${CUCORR_INSTALL_CONFIGDIR}
)

export(
  EXPORT ${TARGETS_EXPORT_NAME}
  FILE ${CMAKE_BINARY_DIR}/${TARGETS_EXPORT_NAME}.cmake
  NAMESPACE cucorr::
)
