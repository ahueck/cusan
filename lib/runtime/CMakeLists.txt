find_package(MPI)
add_library(cucorr_mpi_interceptor SHARED MPIInterception.cpp)
target_link_libraries(cucorr_mpi_interceptor
  PRIVATE MPI::MPI_CXX
)
set_target_properties(
  cucorr_mpi_interceptor
  PROPERTIES
  OUTPUT_NAME "CucorrMPIInterceptor"
  EXPORT_NAME "MPIInterceptor"
)
add_library(cucorr::MPI_Interceptor ALIAS cucorr_mpi_interceptor)
target_compile_features(cucorr_mpi_interceptor PUBLIC cxx_std_17)
target_compile_definitions(
  cucorr_mpi_interceptor
  PRIVATE CUCORR_LOG_LEVEL=${CUCORR_LOG_LEVEL_RT}
   $<$<BOOL:${CUCORR_FIBERPOOL}>:CUCORR_FIBERPOOL=1>
   $<$<BOOL:${CUCORR_SOFTCOUNTER}>:CUCORR_SOFTCOUNTER>
)

target_include_directories(cucorr_mpi_interceptor ${warning_guard}
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

if(CUCORR_FIBERPOOL)
  target_link_libraries(cucorr_mpi_interceptor PRIVATE cucorr::fiberpool)
endif()

set(CONFIG_NAME cucorrMPIInterceptor)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(
  TARGETS cucorr_mpi_interceptor
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

add_library(cucorr_Runtime SHARED CucorrRuntime.cpp CucorrRuntime_cudaSpecific.cpp)
set_target_properties(
  cucorr_Runtime
  PROPERTIES
  OUTPUT_NAME "CucorrRuntime"
  EXPORT_NAME "Runtime"
)
add_library(cucorr::Runtime ALIAS cucorr_Runtime)

target_compile_features(cucorr_Runtime PUBLIC cxx_std_17)

target_link_libraries(cucorr_Runtime PRIVATE typeart::Runtime)

if(CUCORR_FIBERPOOL)
  target_link_libraries(cucorr_Runtime PRIVATE cucorr::fiberpool)
endif()


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

target_compile_definitions(
  cucorr_Runtime
  PRIVATE CUCORR_LOG_LEVEL=${CUCORR_LOG_LEVEL_RT}
   $<$<BOOL:${CUCORR_FIBERPOOL}>:CUCORR_FIBERPOOL=1>
   $<$<BOOL:${CUCORR_SOFTCOUNTER}>:CUCORR_SOFTCOUNTER>
)

set(CONFIG_NAME cucorrRuntime)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(
  TARGETS cucorr_Runtime
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(CUCORR_FIBERPOOL)
install(
  TARGETS cucorr_fiberpool
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
endif()

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
