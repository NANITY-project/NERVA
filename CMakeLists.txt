cmake_minimum_required(VERSION 3.16)
project(nebula_ik CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(NEBULA_NATIVE "Tune for this CPU (-march=native)" ON)

# Header-only library.
add_library(nebula_ik INTERFACE)
target_include_directories(nebula_ik INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(NOT MSVC)
  target_compile_options(nebula_ik INTERFACE -fno-math-errno -fno-trapping-math)
  if(NEBULA_NATIVE)
    target_compile_options(nebula_ik INTERFACE -march=native)
  endif()
endif()

add_executable(ik_demo      src/main.cpp)
add_executable(ik_bench     src/bench.cpp)
add_executable(ik_quality   src/quality.cpp)
add_executable(ik_body_test src/body_test.cpp)
foreach(t ik_demo ik_bench ik_quality ik_body_test)
  target_link_libraries(${t} nebula_ik)
  if(NOT MSVC)
    target_compile_options(${t} PRIVATE -Wall -Wextra)
  endif()
endforeach()

enable_testing()
add_test(NAME ik_tests COMMAND ik_demo)
add_test(NAME ik_body_tests COMMAND ik_body_test)

# C API shared library for host apps (Studio, Blender add-on). Portable: no -march=native.
add_library(nebula SHARED capi/nebula_capi.cpp)
target_include_directories(nebula PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/capi)
set_target_properties(nebula PROPERTIES CXX_VISIBILITY_PRESET hidden)
if(NOT MSVC)
  target_compile_options(nebula PRIVATE -fno-math-errno -fno-trapping-math)
endif()

# The Vulkan renderer bootstrap is separate: see renderer/CMakeLists.txt
