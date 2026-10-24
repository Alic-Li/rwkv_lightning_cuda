include_guard(GLOBAL)

# The VERSION file is the single source of the release version. Reconfigure
# when it changes so the generated header (and only the files including it)
# pick up the new value.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/VERSION")
file(STRINGS "${PROJECT_SOURCE_DIR}/VERSION" RWKV_LIGHTNING_VERSION LIMIT_COUNT 1)
string(STRIP "${RWKV_LIGHTNING_VERSION}" RWKV_LIGHTNING_VERSION)
if(NOT RWKV_LIGHTNING_VERSION)
  message(FATAL_ERROR "VERSION file is empty")
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/version.hpp.in"
  "${PROJECT_BINARY_DIR}/generated/rwkv/version.hpp"
  @ONLY
)
add_library(rwkv_version INTERFACE)
add_library(rwkv::version ALIAS rwkv_version)
target_include_directories(rwkv_version INTERFACE "${PROJECT_BINARY_DIR}/generated")
