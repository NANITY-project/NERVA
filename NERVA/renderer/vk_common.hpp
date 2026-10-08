// vk_common.hpp -- shared helpers for the renderer bootstrap.
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <stdexcept>

#define VK_CHECK(expr)                                                       \
    do {                                                                     \
        VkResult _vk_result = (expr);                                       \
        if (_vk_result != VK_SUCCESS) {                                     \
            std::fprintf(stderr, "[vulkan] %s failed: VkResult=%d (%s:%d)\n",\
                         #expr, (int)_vk_result, __FILE__, __LINE__);        \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

inline std::vector<char> read_file_binary(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("failed to open file: " + path);
    size_t size = (size_t)f.tellg();
    std::vector<char> buf(size);
    f.seekg(0);
    f.read(buf.data(), (std::streamsize)size);
    return buf;
}

inline uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_filter,
                                  VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    std::fprintf(stderr, "[vulkan] no suitable memory type found\n");
    std::exit(1);
}
