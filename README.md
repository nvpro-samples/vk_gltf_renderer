# Vulkan glTF Scene Renderer


|Pathtracer | Raster|
|:------------: | :------------: |
|![](doc/pathtrace.png) |![](doc/raster.png)|

## Overview

This application demonstrates a dual-mode renderer for glTF 2.0 scenes, implementing both ray tracing and rasterization pipelines. It showcases the utilization of shared Vulkan resources across rendering modes, including geometry, materials, and textures.

## What's New

This version brings significant improvements and modernization:

- **Modern Vulkan Framework**: Now using [Nvpro-Core2](https://github.com/nvpro-samples/nvpro_core2.git) which provides:
  - Vulkan 1.4 support
  - Volk for dynamic Vulkan loading
  - Modern C++ features and improved architecture
  - Enhanced debugging and validation layers
  - Better resource management

- **Slang Shading Language**: Replaced GLSL with [Slang](https://github.com/shader-slang/slang) for:
  - Enhanced shader development experience
  - Better cross-platform compatibility
  - Improved shader debugging capabilities
  - Hot-reloading support (F5)
  - Modern shader language features

- **DLSS-RR Denoiser**: Added support for NVIDIA's DLSS Ray Reconstruction denoiser (optional, enable with `USE_DLSS`).

## Key Features

- glTF 2.0 (.gltf/.glb) scene loading
- Pathtracing with global illumination
- PBR-based rasterization
- HDR environment mapping and Sun & Sky simulation
- Advanced tone mapping
- Camera control system
- Extensive debug visualization options

## Dependencies

 - Vulkan SDK ([latest version](https://vulkan.lunarg.com/sdk/home))
 - [Nvpro-Core2](https://github.com/nvpro-samples/nvpro_core2.git) framework
 - [Slang](https://github.com/shader-slang/slang) shading language (included with nvpro_core2)

## Build Instructions

1. Clone the repositories
```bash
git clone https://github.com/nvpro-samples/nvpro_core2.git
git clone https://github.com/nvpro-samples/vk_gltf_renderer.git
```

2. Build the project
```bash
cd vk_gltf_renderer
cmake -B build -S . -DUSE_DLSS=ON -DUSE_DRACO=ON
cmake --build build --config release
```

3. Run the application
```bash
.\_bin\Release\vk_gltf_renderer.exe
```

4. Install [optional] : if you want to package the application
``` bash
cmake --install .
```

### Draco Compression

To enable Draco mesh compression, you need to enable the option CMake. In the GUI interface, you will see the option `USE_DRACO`. If you are using the command line, you can add `-DUSE_DRACO=ON` to the cmake command. This will download the Draco library and it will be included in the project.

### DLSS Ray Reconstruction Denoiser

This release adds support for NVIDIA's [**DLSS Ray Reconstruction (DLSS-RR)**](https://developer.nvidia.com/rtx/dlss) denoiser. DLSS-RR provides state-of-the-art AI-based denoising for path-traced images, significantly improving image quality and temporal stability.

**How to enable:**

By default, DLSS-RR is **disabled**. To enable it, set the CMake option `USE_DLSS=ON` when configuring the project:

```bash
cmake -DUSE_DLSS=ON ..
```

This will automatically download and integrate the required DLSS SDK. The denoiser will then be available as an option in the renderer.

> **Note:** DLSS-RR requires a compatible NVIDIA GPU and drivers.


## glTF Core features

- [x] glTF 2.0 (.gltf/.glb)
- [x] images (HDR, PNG, JPEG, ...)
- [x] buffers (geometry, animation, skinning, ...)
- [x] textures (base color, normal, metallic, roughness, ...)
- [x] materials (PBR, ...)
- [x] animations
- [x] skins
- [x] morphs
- [x] cameras
- [x] lights
- [x] nodes
- [x] scenes
- [x] samplers
- [x] textures
- [x] extensions

## GLTF Extensions
 Here are the list of extensions that are supported by this application

- [ ] KHR_animation_pointer
- [x] KHR_draco_mesh_compression
- [x] KHR_lights_punctual
- [x] KHR_materials_anisotropy
- [x] KHR_materials_clearcoat
- [x] KHR_materials_diffuse_transmission
- [x] KHR_materials_dispersion
- [x] KHR_materials_emissive_strength
- [x] KHR_materials_ior
- [x] KHR_materials_iridescence
- [x] KHR_materials_sheen
- [x] KHR_materials_specular
- [x] KHR_materials_transmission
- [x] KHR_materials_unlit
- [x] KHR_materials_variants
- [x] KHR_materials_volume
- [ ] KHR_mesh_quantization
- [x] KHR_texture_basisu
- [x] KHR_texture_transform
- [ ] KHR_xmp_json_ld
- [x] EXT_mesh_gpu_instancing
- [x] KHR_node_visibility

## Pathtracer

Implements a path tracer with global illumination. 


![](doc/pathtracer_settings.png)

The options are:
* Max Depth : number of bounces the path can do
* Max Samples: how many samples per pixel at each frame iteration
* Aperture: depth-of-field
* Debug Method: shows information like base color, metallic, roughness, and some attributes
* Choice between indirect and RTX pipeline.
* Denoiser: A-trous denoiser 


## Raster

Utilizes shared Vulkan resources with the path tracer, including:

- Scene geometry
- Material data
- Textures
- Shading functions

The options are:
* Show wireframe: display wireframe on top of the geometry
* Super-Sampling: render the image 2x and blit it with linear filter.
* Debug Method: shows information like base color, metallic, roughness, and some attributes

![](doc/raster_settings.png)

Example with wireframe option turned on

![](doc/wireframe.png)


## Features

| | | 
|--|--|
| Showcase | ![](doc/ABeautifulGame.jpg) ![](doc/ToyCar.jpg) ![](doc/DamagedHelmet.jpg) ![](doc/DiffuseTransmissionPlant.jpg) <br> ![](doc/DiffuseTransmissionTeacup.jpg) ![](doc/AntiqueCamera.jpg)  ![](doc/BistroExterior.jpg) ![](doc/IridescentDishWithOlives.jpg) <br> ![](doc/SpecularSilkPouf.jpg) ![](doc/Sponza.jpg) ![](doc/SciFiHelmet.jpg) ![](doc/ChairDamaskPurplegold.jpg) <br> ![](doc/CarConcept.jpg) ![](doc/SunglassesKhronos.jpg)|
| Anisotropy | ![](doc/AnisotropyBarnLamp.jpg) ![](doc/AnisotropyDiscTest.jpg) ![](doc/AnisotropyRotationTest.jpg) ![](doc/AnisotropyStrengthTest.jpg) <br> ![](doc/CompareAnisotropy.jpg)|
| Attenuation | ![](doc/DragonAttenuation.jpg) ![](doc/AttenuationTest.jpg)|
| Alpha Blend | ![](doc/AlphaBlendModeTest.jpg) ![](doc/CompareAlphaCoverage.jpg) |
| Animation | ![](doc/BrainStem.jpg) ![](doc/CesiumMan.jpg) ![](doc/Fox.jpg) |
| Clear Coat | ![](doc/ClearCoatCarPaint.jpg) ![](doc/ClearCoatTest.jpg) ![](doc/ClearcoatWicker.jpg) ![](doc/CompareClearcoat.jpg)
| Dispersion | ![](doc/DispersionTest.jpg) ![](doc/DragonDispersion.jpg) ![](doc/CompareDispersion.jpg) |
| IOR | ![](doc/IORTestGrid.jpg) ![](doc/CompareIor.jpg) |
| Emissive |![](doc/EmissiveStrengthTest.jpg) ![](doc/CompareEmissiveStrength.jpg) |
| Iridescence | ![](doc/IridescenceAbalone.jpg) ![](doc/IridescenceDielectricSpheres.jpg) ![](doc/IridescenceLamp.jpg) ![](doc/IridescenceSuzanne.jpg) |
| Punctual | ![](doc/LightsPunctualLamp.jpg) ![](doc/light.jpg) |
| Sheen | ![](doc/SheenChair.jpg) ![](doc/SheenCloth.jpg) ![](doc/SheenTestGrid.jpg) ![](doc/CompareSheen.jpg) |
| Transmission | ![](doc/TransmissionRoughnessTest.jpg) ![](doc/TransmissionTest.jpg) ![](doc/TransmissionThinwallTestGrid.jpg) ![](doc/CompareTransmission.jpg) <br> ![](doc/CompareVolume.jpg) ![](doc/GlassBrokenWindow.jpg) ![](doc/MosquitoInAmber.jpg) |
| Variant | ![](doc/MaterialsVariantsShoe_1.jpg) ![](doc/MaterialsVariantsShoe_2.jpg) ![](doc/MaterialsVariantsShoe_3.jpg) |
| Others | ![](doc/BoxVertexColors.jpg) ![](doc/Duck.jpg) ![](doc/MandarinOrange.jpg) ![](doc/SpecularTest.jpg) ![](doc/OrientationTest.jpg) ![](doc/NegativeScaleTest.jpg) ![](doc/NormalTangentTest.jpg) ![](doc/TextureCoordinateTest.jpg) ![](doc/NormalTangentMirrorTest.jpg)  ![](doc/BarramundiFish.jpg) ![](doc/CarbonFibre.jpg) ![](doc/cornellBox.jpg) ![](doc/GlamVelvetSofa_1.jpg)  ![](doc/LightsPunctualLamp.jpg) ![](doc/MultiUVTest.jpg) ![](doc/SimpleInstancing.jpg) ![](doc/SpecGlossVsMetalRough.jpg) ![](doc/CompareBaseColor.jpg) ![](doc/CompareMetallic.jpg) ![](doc/CompareSpecular.jpg) |


## Debug

There is also the ability to debug various out channels, such as:

|metallic|roughness|normal|base|emissive|opacity|tangent|tex coord|
|---|---|---|---|---|---|---|---|
|![](doc/dbg_metallic.jpg)|![](doc/dbg_roughness.jpg)|![](doc/dbg_normal.jpg)|![](doc/dbg_base_color.jpg)|![](doc/dbg_emissive.jpg) |![](doc/dbg_opacity.jpg) |![](doc/dbg_tangent.jpg) | ![](doc/dbg_tex_coord.jpg) |


## Environment

### Sun & Sky

There is a built-in Sun & Sky physical shader module.

![](doc/sky_1.jpg) ![](doc/sky_2.jpg) ![](doc/sky_3.jpg)

### HDR 

Lighting of the scene can come from HDRi.

![](doc/hdr_1.jpg) ![](doc/hdr_2.jpg) ![](doc/hdr_3.jpg) ![](doc/hdr_4.jpg) <br> ![](doc/hdr_5.jpg) ![](doc/hdr_6.jpg) ![](doc/hdr_7.jpg) ![](doc/hdr_8.jpg)

It is possible to blur HDR to various level.

![](doc/hdr_1.jpg) ![](doc/hdr_blur_1.jpg) ![](doc/hdr_blur_2.jpg) ![](doc/hdr_blur_3.jpg)

The HDR can also be rotated to get the right illumination.

![](doc/hdr_1.jpg) ![](doc/hdr_rot_1.jpg)

### Background

Background can be also solid color and if saved as PNG, the alpha channel is taking into account. 

![](doc/background_1.jpg) ![](doc/background_2.jpg) ![](doc/background_3.png)



## Tonemapper

We could not get good results without a tone mapper. This is done with a compute shader and different settings can be made.

![](doc/tonemapper.png)

Multiple tonemapper are supported:
* [Filmic](http://filmicworlds.com/blog/filmic-tonemapping-operators/)
* Uncharted 2
* Clip : Simple Gamma correction (linear to sRGB)
* [ACES](https://www.oscars.org/science-technology/sci-tech-projects/aces): Academy Color Encoding System
* [AgX](https://github.com/EaryChow/AgX)
* [Khronos PBR](https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/README.md#pbr-neutral-specification) : PBR Neutral Specification


## Camera

The camera navigation follows the [Softimage](https://en.wikipedia.org/wiki/Softimage_(company)) default behavior. This means, the camera is always looking at a point of interest and orbit around it.

Here are the default navigations:

![](doc/cam_info.png)

The camera information can be fine tune by editing its values.

![](doc/cam_1.png)

Note: **copy** will copy in text the camera in the clipboard, and pressing the **paste** button will parse the clipboard to set the camera. 

Ex: `{0.47115, 0.32620, 0.52345}, {-0.02504, -0.12452, 0.03690}, {0.00000, 1.00000, 0.00000}`

### Save and Restore Cameras

It is also possible to save and restore multiple cameras in the second tab. Press the `+` button to save a camera, the middle button to delete it. By pressing one of the saved cameras, its position, interests, orientation and FOV will be changed smoothly. 

**Note**: If the glTF scene contains multiple cameras, they will be showing here. 

![](doc/cam_2.png)

### Other modes 

Other navigation modes also exist, like fly, where the `w`, `a`, `s`, `d` keys also moves the camera. 

![](doc/cam_3.png)

## Depth-of-Field

Depth of field works only for ray tracing and settings can be found under the `RendererPathtracer>Depth-of-Field`

![](doc/dof_1.jpg) ![](doc/dof_2.jpg)


----
## Schema of the Program

The nvvk::Application is a class that provides a framework for creating Vulkan applications. It encapsulates the Vulkan instance, device, and surface creation, as well as window management and event handling.

When using `nvvk::Application`, you can attach `nvvkhl::IAppElement` to it and each element will be called for the different state, allowing to customize the behavior of your application. The `nvvkhl::IAppElement` class provides default implementations for these functions, so you only need to override the ones you need.

Here is a brief overview of how `nvvk::Application` works:

### Initialization:
When you create an instance of `nvvk::Application`, it sets up the Vulkan instance, device, and surface. It also creates a window and sets up event handling.

### Attaching Elements
In `main()` we are attaching many elements, like:
* `ElementCamera` : this allow to control a singleton camera
* `ElementProfiler` : allow to time the execution on the GPU
* `ElementBenchmarkParameters` : command line arguments and test purpose
* `ElementLogger` : redirect log information in a window
* `ElementNvml` : shows the status of the GPU

But the main one that interest us, and which is the main of this application is `GltfRendererElement`. This is the one that will be controlling the scene and rendering.


### Main Loop: 
The `nvvk::Application` class provides a main loop that continuously processes events and updates the application state. Inside the main loop, it calls the following functions:

* **onAttach()**:<br> 
This function is called whenever the element is attached to the application. In `GltfRendererElement`, we are creating the resource needed internally. 

* **onDetach()**: <br>
This function is called when the user tries to close the window. You can override this function to handle window close events.

* **onRender(VkCommandBuffer)**: <br>
This function is called to render the frame using the current command buffer of the frame. You can override this function to perform rendering operations using Vulkan. In `GltfRendererElement` this is where the active renderer is called.

* **onResize()**: <br>
This function is called when the `viewport` is resized. You can override this function to handle window resize events. In `GltfRendererElement` the G-Buffer will be re-created

* **onUIRender()**: <br>
This function is called to allow the `IAppElement` to render the UI and to query any mouse or keyboard event. In `GltfRendererElement`, we render the UI, but also the final image. The rendered image is consider a UI element, and that image covers the entire `viewport` ImGui window. 

* **onUIMenu()** <br>
Will be modifying what we see in the the window title. It will also create the menu, like `File`, `Help` and deal with some key combinations.

* **onFileDrop()** <br>
Will receive the path of the file been dropped on. If it is a .gltf, .glb, .obj or .hdr, it will load that file. The file type is determined by its extension - 3D scene files (.gltf, .glb, .obj) are loaded as scenes, while HDR files (.hdr) are loaded as environment maps. 



----

## Scene Graph

The GLTF scene is loaded using tinygltf and then converted to a Vulkan version. The Vulkan version is a simplified version of the scene, where the geometry is stored in buffers, and the textures are uploaded to the GPU. The Vulkan version is used for both raster and ray tracing.

The scene is composed of nodes, where each node can have children and each node can have a mesh. The mesh is composed of primitives, where each primitive has a material. The material is composed of textures and parameters. However, none of this is directly used in the rendering, as we are using a simplified version of the scene.

![](doc/scene_graph.png)

Once the scene has been loaded, we proceed to parse it in order to collect the RenderNodes and RenderPrimitives. The RenderNode represents the flattened version of the tree of nodes, where the world transformation matrix and the material are stored. The RenderPrimitive, in contrast, represents the unique version of the primitive, where the index and vertex buffers are stored.

RenderNodes represent the elements to be rendered, while RenderPrimitives serve as references to the data utilized for rendering.

### Animation

If there is animation in the scene, a new section will appear under the Scene section.
It allows to play/pause, step and reset the animation, as well as changing its speed.

![](doc/animation_controls.png)


### Multiple Scene 

If there are multiple scenes,  a new section will appear under the Scene section.
It will show all the scenes and their name. Clicking on a scene name will switch to the scene.

![](doc/multiple_scenes.png)


### Material Variant

If there are multiple material variant, a new section will appear under the Scene section.
It will show all the material variant and their name. Clicking on a variant name will apply it on the models.

![](doc/material_variant.png)


### Scene Graph UI

![](doc/scene_graph_ui.png)

It is possible to visualize the scene hierarchy, to select node, to modify their transformation and their material, to some level.

Here's a shorter version of the text, tailored for developers on GitHub:

### Recompiling Shaders
For quick shader testing, use the `Recompile Shaders` button to hot-reload Slang shaders (F5). The shaders are located in the `shaders` folder and are automatically compiled during the build process.

Note: Hot-reloading won't work without the shared libraries and shaders, but the app will still run.

## Tools

The application comes with a few tools to help debug and visualize the scene.

### Profiler

The profiler is a tool that allows to measure the time spent on the GPU. It is possible to measure the time spent on the different stages of the rendering, like the path tracing, the rasterization, the tonemapping, etc.

![](doc/profiler.png)

### Logger

The logger is a tool that allows to see the log information. It is possible to filter the log information by selecting the level of the log.

![](doc/logger.png)

### Nvml

The Nvml is a tool that allows to see the status of the GPU. It is possible to see the temperature, the power, the memory usage, etc.

![](doc/nvml.png)

### Tangent Space

There is a tangent space tool that allows to fix or to recreate the tangent space of the model. This is useful when the normal map is not looking right or there are errors with the tangents in the scene.

## Utilities

### gltf-material-modifier.py

Modify materials in a GLTF file and optionally reorient the scene from Z-up to Y-up.

```
usage: gltf-material-modifier.py [-h] [--metallic METALLIC] [--roughness ROUGHNESS] [--override] [--reorient]
                                 input_file output_file
```                                 

positional arguments:
```
   input_file            Path to the input GLTF file.
   output_file           Path to save the modified GLTF file.
```

options:
```
  -h, --help             show this help message and exit
  --metallic METALLIC    Set the metallic factor (default: 0.1).
  --roughness ROUGHNESS  Set the roughness factor (default: 0.1).
  --override             Override existing material values if set.
  --reorient             Reorient the scene from Z-up to Y-up.
```