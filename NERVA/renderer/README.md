# NANITY Renderer -- Vulkan bootstrap

This is the foundation of the embodiment renderer: instance -> physical
device -> logical device -> an offscreen render target -> a graphics
pipeline -> a real draw call -> readback. Deliberately separate from the
NEON inference runtime (which doesn't support Vulkan yet) -- this has no
dependency on it and never will need one; the model talks to the IK
engine via the text command protocol, the IK engine's joint rotations
feed the renderer, and none of that requires NEON to know Vulkan exists.

## What's actually verified here (not just "compiles")

I don't have a GPU or display in the environment I built this in, so this
is validated headlessly against Mesa's lavapipe (CPU software Vulkan
implementation) instead of a real windowed swapchain. That's a real,
non-trivial difference for windowing/presentation, but the actual
rendering code -- instance, device, pipeline, shader compilation,
rasterization, draw -- is bit-identical to what a windowed renderer runs;
only the swapchain-vs-offscreen-image part differs. Concretely verified,
not just eyeballed:

- Both shaders compile through the real SPIR-V toolchain (`glslangValidator`).
- The pipeline actually executes: a rendered pixel at the triangle's
  centroid differs from the cleared background by a wide margin (checked
  numerically in `headless_triangle.cpp`, not just "didn't crash").
- The output was pulled back through the GPU->CPU readback path and
  dumped to `output.ppm` -- it's a correctly barycentric-interpolated
  triangle (red/green/blue corners blending smoothly), which is strong
  evidence the whole coordinate/rasterization pipeline is right, not just
  "some pixels changed."

## Build & run

```
mkdir build && cd build
cmake .. && make
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json ./headless_triangle   # forces lavapipe; omit this on real hardware to use your GPU
```
Needs: a Vulkan loader + dev headers (`libvulkan-dev`), `glslangValidator`
(`glslang-tools`), and on real hardware, your GPU's Vulkan driver (`mesa-vulkan-drivers`
for AMD/Intel on Linux, or your vendor's driver). The `VK_ICD_FILENAMES`
override is only needed to force software rendering for testing --  drop
it once you're on a machine with a real GPU and it'll pick that up
automatically.

## Next steps toward Option A (avatar reaches, cursor follows the hand)

In rough dependency order:

1. **Windowed presentation** -- replace the offscreen image with a real
   `VkSurfaceKHR` + swapchain. Needs a windowing lib (GLFW is the
   pragmatic choice for Vulkan surface creation, or raw XCB/Wayland if
   staying dependency-free matters more here than it did for the math
   layer). This also needs to be a **transparent, click-through,
   always-on-top** surface for Option A -- that's compositor-level
   (Wayland: layer-shell protocol; X11: compositing WM + `_NET_WM_WINDOW_TYPE`
   hints), not just a plain Vulkan swapchain flag.
2. **glTF loading** -- `tinygltf` (single header) to parse `.glb`, extract
   bind-pose bone lengths/orientations to feed `IKEngine::add_chain`
   instead of hand-authored numbers, and load mesh/skin data.
3. **Skinned vertex shader** -- extend `triangle.vert`'s approach with
   real vertex buffers carrying joint indices + weights, plus a uniform
   buffer of joint matrices computed each frame from the IK engine's
   `Chain::world_positions()`/local rotations.
4. **Camera + projection** -- calibrated so the 3D scene's screen-space
   projection maps 1:1 to real desktop pixels (needed either way, but
   THE critical piece for Option A specifically).
5. **uinput bridge** -- project the animated fingertip's world position
   through the camera each frame, feed the resulting 2D pixel coordinate
   to the `/dev/uinput` virtual mouse device discussed earlier.

Step 1 is the biggest unknown since it can't be tested here -- it
depends on your actual compositor's transparency/overlay support, which
varies a lot between desktop environments. Worth checking what your
target machine runs (X11 vs Wayland, and which compositor) before
committing to an approach there.
