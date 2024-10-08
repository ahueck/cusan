
if(CUSAN_FIBERPOOL)
    FetchContent_Declare(
    fiber_pool
    GIT_REPOSITORY https://git-ce.rwth-aachen.de/hpc-research/correctness/fiber-pool
    GIT_SHALLOW 1
    GIT_TAG main
    )

    FetchContent_GetProperties(fiber_pool)

    if(NOT fiber_pool_POPULATED)
        FetchContent_Populate(fiber_pool)

        add_library(cusan_fiberpool SHARED ${fiber_pool_SOURCE_DIR}/fiberpool.cpp)
        #target_include_directories(cusan_fiberpool SYSTEM INTERFACE ${fiber_pool_SOURCE_DIR})
        target_include_directories(cusan_fiberpool SYSTEM PUBLIC  
            PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
            PUBLIC $<BUILD_INTERFACE:${fiber_pool_SOURCE_DIR}>
        )
        add_library(cusan::fiberpool ALIAS cusan_fiberpool)
        set_target_properties(
            cusan_fiberpool
            PROPERTIES
            OUTPUT_NAME "CusanFiberpool"
            EXPORT_NAME "Fiberpool"
            )
    endif()

    mark_as_advanced(
    FETCHCONTENT_SOURCE_DIR_FIBER_POOL
    FETCHCONTENT_UPDATES_DISCONNECTED_FIBER_POOL
    )
endif()