add_library(cucorr_Analysis STATIC KernelModel.cpp KernelAnalysis.cpp ModelIO.cpp)
set_target_properties(
        cucorr_Analysis
        PROPERTIES
        OUTPUT_NAME "CucorrAnalysis"
        EXPORT_NAME "Analysis"
)
add_library(cucorr::Analysis ALIAS cucorr_Analysis)

target_compile_features(cucorr_Analysis PUBLIC cxx_std_17)

target_include_directories(cucorr_Analysis
        SYSTEM
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
)

target_include_directories(cucorr_Analysis ${warning_guard}
        PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
)

cucorr_target_define_file_basename(cucorr_Analysis)

set(CONFIG_NAME cucorrAnalysis)
set(TARGETS_EXPORT_NAME ${CONFIG_NAME}Targets)

install(
        TARGETS cucorr_Analysis
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
