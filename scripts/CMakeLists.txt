function(configure_cucorr_script input output)
  cmake_parse_arguments(
    ARG "" "" "APPLY_MODE" ${ARGN}
  )

  set(CUCORR_INCLUDE_DIRS "-I${PROJECT_SOURCE_DIR}/lib/runtime")

  if(ARG_APPLY_MODE)
    set(CUCORR_RUN_SCRIPT 0)
  else()
    set(CUCORR_RUN_SCRIPT 1)
  endif()

  cucorr_target_generate_file(${input} ${output})
endfunction()

configure_cucorr_script(cucorr-tmpl.sh.in run.sh)
configure_cucorr_script(
  cucorr-tmpl.sh.in apply.sh
  APPLY_MODE ON
)

