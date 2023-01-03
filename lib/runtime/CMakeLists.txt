add_library(cucorr_Runtime SHARED CucorrRuntime.cpp)
set_target_properties(
  cucorr_Runtime
  PROPERTIES
  OUTPUT_NAME "CucorrRuntime"
  EXPORT_NAME "Runtime"
)
add_library(cucorr::Runtime ALIAS cucorr_Runtime)

target_compile_features(cucorr_Runtime PUBLIC cxx_std_17)

target_link_libraries(cucorr_Runtime PRIVATE typeart::Runtime)

target_include_directories(cucorr_Runtime
  SYSTEM
  PRIVATE
  ${LLVM_INCLUDE_DIRS}
)

target_include_directories(cucorr_Runtime ${warning_guard}
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

cucorr_target_define_file_basename(cucorr_Runtime)

set(CONFIG_NAME cucorrRuntime)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(
  TARGETS cucorr_Runtime
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
