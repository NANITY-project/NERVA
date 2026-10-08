cmake_minimum_required(VERSION 3.16)
project(nanity_renderer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

# Compile every GLSL shader in shaders/ to SPIR-V at build time, output
# alongside the executable so relative shader paths in code just work.
file(GLOB SHADER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag)
set(SHADER_OUTPUTS "")
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPIRV "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
        COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER} -o ${SPIRV}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader ${SHADER_NAME}"
    )
    list(APPEND SHADER_OUTPUTS ${SPIRV})
endforeach()
add_custom_target(shaders DEPENDS ${SHADER_OUTPUTS})

# NOTE: headless_triangle.cpp and shaders/ are not in this repo yet. Guarded so
# the rest of the project isn't blocked on them.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/headless_triangle.cpp)
  add_executable(headless_triangle src/headless_triangle.cpp)
  target_link_libraries(headless_triangle Vulkan::Vulkan)
  add_dependencies(headless_triangle shaders)
else()
  message(WARNING "renderer/src/headless_triangle.cpp missing -- nothing to build")
endif()

# headless_triangle reads shaders/ relative to its working directory --
# run it from the build dir, or copy build/shaders/ next to wherever
# you run it from.
