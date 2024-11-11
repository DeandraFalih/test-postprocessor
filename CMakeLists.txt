cmake_minimum_required(VERSION 3.10.2)

project(sclbl-integration-examples)

# Add Scailable C Utilities for all subprojects
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/sclbl-utilities)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sclbl-utilities/include)

# Add Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-example)

# # Add installation option
install(PROGRAMS
    ${CMAKE_CURRENT_BINARY_DIR}/postprocessor-python-example/postprocessor-python-example
    DESTINATION /opt/networkoptix/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/
)