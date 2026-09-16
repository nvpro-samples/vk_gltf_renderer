# User Guide — Vulkan glTF Renderer (Ray Tracing & PBR)

> **For users.** How to run and use the app — RTX path tracer, PBR rasterizer, AI denoisers, scene editor, camera, environment, tone mapping, and the common CLI flags. For project overview and build instructions see the [README](../README.md); for internals and the authoritative flag/enum lists, see the [Developer Guide](developer.md) and the code.

## Quick Navigation

- [Renderer Modes](#renderer)
- [Environment](#environment)
- [Tone Mapping](#tone-mapping)
- [Camera](#camera)
- [Depth-of-Field](#depth-of-field)
- [Scene Asset Editor](#scene-asset-editor)
- [Animation](#animation)
- [Debug Visualization](#debug-visualization)
- [Tools](#tools)
- [Configuration and CLI](#configuration)
- [Troubleshooting](#troubleshooting)

## Recommended First Run

1. Launch the app and load a `.gltf`/`.glb` scene (`.obj` files are also accepted and converted to glTF on the fly).
2. Switch between Rasterizer and Path Tracer in the **Settings** panel.
3. Load an HDR environment or use Sun and Sky.
4. Enable a denoiser (DLSS-RR or OptiX) in path tracing mode.
5. Save a modified scene as `.gltf` or `.glb` from the **File** menu.

---

## Renderer

The application provides two Vulkan renderer modes that share GPU resources (geometry, materials, textures, and shading code). The **Path Tracer** is the primary quality reference for ray tracing and PBR materials; the **Rasterizer** is the fast fallback preview mode. Switch at any time from the **Settings** panel.

![](images/renderers.jpg)

### Path Tracer

A Monte Carlo path tracer with global illumination, progressive accumulation, and physically based light transport. This is the primary ray tracing mode and the reference implementation for glTF PBR material accuracy.

![](images/pathtracer_settings.jpg)

| Setting | Description |
|---|---|
| **Rendering Pipeline** | Choice between **Compute / Ray Query** (compute shader) and **Ray Tracing Pipeline** (hardware RT pipeline with SBT). Both produce identical results; Ray Query avoids pipeline overhead on some workloads. |
| **Use SER** | Enable [Shader Execution Reorder](https://developer.nvidia.com/blog/improving-ray-tracing-performance-with-shader-execution-reorder/) for the Ray Tracing pipeline. Can improve coherence on RTX 40-series GPUs. |
| **FireFly Clamp** | Clamps high-intensity samples to reduce firefly artifacts in early frames. |
| **Max Iterations** | Maximum number of frames accumulated before the renderer stops. |
| **Samples** | Number of samples per pixel per frame. Higher = cleaner but slower per frame. |
| **Auto SPP** | Adaptive sampling: automatically adjusts samples-per-pixel to maintain a target frame rate. Choose between Interactive, Balanced, Quality, and Max Quality presets. |
| **Aperture** | Depth-of-field lens aperture. Set to 0 for a pinhole camera (everything in focus). |
| **Auto Focus** | Automatically sets the focal distance to the camera's interest point (double-click an object to set). |
| **Infinite Plane** | Adds an infinite ground plane with optional **Shadow Catcher** mode. When enabled, the plane subtracts light from the environment and adds only shadows and reflections — ideal for product shots. Surface properties (color, roughness, metallic) are adjustable. |

### AI-Accelerated Denoisers

![](images/denoisers.jpg)

Two denoisers are available to reduce path tracing noise while preserving detail:

#### DLSS Ray Reconstruction (DLSS-RR)

[DLSS Ray Reconstruction](https://developer.nvidia.com/rtx/dlss) provides AI denoising with strong temporal stability for ray-traced content.

![](images/dlss.jpg)

- Enable or disable it from the denoiser activation row; the status appears next to the row label. **Loading** means the nonblocking NGX prewarm is running, **Ready** means it can be enabled without waiting for startup initialization, and **On** means it is actively denoising the current frame.
- Open the row's settings button to choose the input size (Min / Optimal / Max) — lower internal resolution means faster rendering, DLSS upscales to the viewport.
- Developer guide-buffer previews (albedo, normal, motion, depth, specular) live at the bottom of the panel under **Developer Guide Buffers**. Use **Rendered** to switch back to the main image.
- Transparency handling can be set to "Default (first hit)" or "Improved (blended guides)" for scenes with alpha-blended materials.

**How to enable:** Set `USE_DLSS=ON` in CMake (enabled by default). The DLSS SDK is downloaded automatically. Requires an NVIDIA RTX 20-series or newer GPU and up-to-date drivers.

#### DLSS Neural Rendering (DLSS-NR)

DLSS-NR is a display-resolution **image enhancer** that runs on top of DLSS-RR or DLSS-SR. It
operates on the tonemapped (LDR) image and applies learned local tone and structure adjustments.
Enable it in the DLSS panel once the main DLSS feature is available. In path tracing mode, DLSS-RR
must be enabled because NR needs a denoised input.

Open the row's settings button to edit:
- **Intensity** — overall NR effect strength.
- **Local Tone** — adjusts local luminance contrast.
- **Local Structure** — sharpens or smooths local detail; most visible on high-frequency surfaces.
- **Global Tone** — adjusts global tone mapping strength applied by NR.
- **Skin Structure** — NR-specific skin-detail preservation.
- **Style** — NR processing style preset (SDK-defined).
- **Auto Mask** — let NR derive its own per-pixel mask instead of using `EXT_DLSS_NR`.

**Per-material NR mask (`EXT_DLSS_NR`):** Add the `EXT_DLSS_NR` extension to any material in the
**Material Extensions** section of the Inspector to override NR strengths per object. Each channel
multiplies the corresponding global setting for pixels covered by that material:

| Channel | Global setting it scales |
|---|---|
| Intensity | Intensity |
| Local Tone | Local Tone |
| Local Structure | Local Structure |
| Global Tone | Global Tone |

Default `(1, 1, 1, 1)` leaves global values unchanged. Set a channel to `0` to fully suppress
that effect on the material's pixels. This enables per-character NR tuning — for example, disabling
local structure on a background prop while keeping it on a foreground character.

> The per-material mask is written by the path tracer only. It has no effect when the rasterizer
> is active (the rasterizer falls back to the global `Use Auto Mask` setting).

**How to enable:** Requires `USE_DLSSNR=ON` in CMake plus a compatible beta NGX SDK.

#### OptiX AI Denoiser

[OptiX AI Denoiser](https://developer.nvidia.com/optix-denoiser) uses albedo and normal guide buffers to preserve detail while removing Monte Carlo noise.

![OptiX AI Denoiser panel](images/optix.jpg)

- Enable or disable it from the denoiser activation row; the status appears next to the row label.
- Open the row's settings button, then click **Denoise Now** to denoise the current accumulation, or enable **Auto** to trigger automatically every N frames.
- Use the **Rendered** / **Denoised** viewport toggle in **OptiX Output Preview** to compare the current render with the OptiX result.

**How to enable:** Set `USE_OPTIX_DENOISER=ON` in CMake (enabled by default when CUDA Toolkit is found). OptiX headers are downloaded automatically — no separate SDK install needed. Requires the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (11.0+).

### Opacity Micromap (EXT_mesh_opacity_micromap)

[Opacity Micromaps (OMM)](https://developer.nvidia.com/blog/improve-ray-tracing-performance-with-opacity-micromaps/) pre-classify each micro-triangle in alpha-tested geometry as opaque, transparent, or unknown. The ray tracer skips the any-hit shader for opaque micro-triangles, eliminating per-ray alpha evaluation on most of the surface.

This renderer loads scenes that already contain `EXT_mesh_opacity_micromap` data and binds it when building the BLAS. Bake OMMs with [gltf_omm_baker](https://github.com/nvpro-samples/gltf_omm_baker), which writes the extension into the glTF file.

The **Opacity Micromap** visualization mode (Settings → Visualization) is a debug view for alpha-tested geometry that shows where the ray tracer still pays for alpha (any-hit) shading. Surfaces resolved by an opacity micromap as opaque are drawn green (no alpha work); "unknown" micro-triangles that still run the alpha shader are drawn yellow; transparent micro-triangles are culled, so those pixels show the environment behind. On a scene without an opacity micromap the whole alpha-tested surface reads yellow, illustrating the cost the OMM removes. The view is meaningful in the RayTracing (RT pipeline) technique, which consults the micromap.

<img src="images/omm_bake.png" alt="Left: scene with OMM (mostly green = OMM-resolved, some yellow = unknown micro-triangles). Right: same scene without OMM (all yellow = full any-hit evaluation on every ray)." width="480">

### Rasterizer

![](images/raster_settings.jpg)

The rasterizer provides a fast PBR preview using forward rendering. It shares the same Vulkan resources as the path tracer:

- Scene geometry (vertex/index buffers)
- Material data (PBR parameters, textures)
- Shading functions (Slang shader modules)

It does not implement the full glTF PBR model and is not intended as the main material-reference implementation. It is designed as a fast navigation and fallback mode with shared scene resources.

![](images/wireframe.png)

Wireframe mode can be toggled for mesh inspection.

---

## Environment

### Sun & Sky

A built-in physically based Sun & Sky shader module simulates atmospheric scattering. Adjust sun direction, turbidity, and ground albedo for different times of day and weather conditions.

![](images/sky_1.jpg) ![](images/sky_2.jpg) ![](images/sky_3.jpg)

### HDR Environment

Lighting can come from HDR environment maps (`.hdr` files). Drag and drop an HDR file onto the viewport, or load via **File > Load HDR Environment** (`Ctrl+Shift+O`).

![](images/hdr_1.jpg) ![](images/hdr_2.jpg) ![](images/hdr_3.jpg) ![](images/hdr_4.jpg) <br> ![](images/hdr_5.jpg) ![](images/hdr_6.jpg) ![](images/hdr_7.jpg) ![](images/hdr_8.jpg)

The environment can be **blurred** to soften reflections and lighting:

![](images/hdr_1.jpg) ![](images/hdr_blur_1.jpg) ![](images/hdr_blur_2.jpg) ![](images/hdr_blur_3.jpg)

And **rotated** to position the light source where you need it:

![](images/hdr_1.jpg) ![](images/hdr_rot_1.jpg)

### No Environment

Setting the **Environment Type** to **None** disables the sky and HDR entirely: the scene receives no environment lighting (only its own punctual and emissive lights contribute), and unless a **Solid Color** background is enabled the backdrop is black. Prefer this over dialing HDR intensity to zero — it also skips environment importance sampling and the dome pass.

If **None** is selected while the scene has neither punctual lights nor emissive materials, nothing is lit. With no **Solid Color** background the frame is then fully black, so the viewport shows a warning banner in that case so the empty result isn't mistaken for a bug.

### Background

The background can also be a solid color. When saving as PNG, the alpha channel is preserved — useful for compositing renders over custom backgrounds.

![](images/background_1.jpg) ![](images/background_2.jpg) ![](images/background_3.png)

---

## Tone Mapping

A tone mapper is essential for converting HDR rendering output to displayable LDR images. Tone mapping is performed with a compute shader, and settings like exposure, contrast, saturation, and vignette are adjustable in real time.

![](images/tonemapper.jpg)

Supported tone mappers:

| Tone Mapper | Description |
|---|---|
| [**Filmic**](http://filmicworlds.com/blog/filmic-tonemapping-operators/) | Classic film-like response curve |
| **Uncharted 2** | Popular game tone mapper with good highlight rolloff |
| **Clip** | Simple gamma correction (linear to sRGB), no compression |
| [**ACES**](https://www.oscars.org/science-technology/sci-tech-projects/aces) | Academy Color Encoding System — the film industry standard |
| [**AgX**](https://github.com/EaryChow/AgX) | Modern filmic with excellent highlight handling |
| [**Khronos PBR**](https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/README.md#pbr-neutral-specification) | PBR Neutral — designed for faithful material appearance |

---

## Camera

Camera navigation follows the [Softimage](https://en.wikipedia.org/wiki/Softimage_(company)) default behavior. The camera always looks at a **point of interest** and orbits around it.

### Controls

![](images/cam_info.png)

### Overview
![](images/cam_1.png)

### Copy / Restore / Save

![](images/cam_2.png)

- Click the **home** icon to restore the camera to its original position.
- Click the **camera+** icon to save the current view; saved cameras appear as #1, #2, etc.
- Click the **copy** icon to store camera parameters in the clipboard (JSON format).
- Click the **paste** icon to set the camera from clipboard data.
- Click a **camera number** to recall that saved view.

### Navigation Modes 

![](images/cam_3.png)

| Mode | Description |
|---|---|
| **Orbit** | Rotates around the point of interest. Double-click an object to re-center. |
| **Fly** | Free movement. Use `W` `A` `S` `D` to move, mouse to look. |
| **Walk** | Like Fly, but restricted to the horizontal (X-Z) plane. |

---

## Depth-of-Field

Depth of field is available in the path tracer under **Settings → Path Tracer** (Aperture, Auto Focus, Focal Distance). Adjust aperture and focal distance for cinematic bokeh.

![](images/dof_1.jpg) ![](images/dof_2.jpg)

Use **Auto Focus** to automatically set the focal distance to the camera's interest point.

---

## Scene Asset Editor

![](images/scene_graph_ui.png)

The application includes a full **glTF scene asset editor** that allows non-destructive modifications to the loaded scene. All changes operate directly on the in-memory glTF model and can be saved back to disk as `.gltf` or `.glb`.

### Scene Browser

The **Scene Browser** panel provides two complementary tabs:

- **Scene Graph** — an interactive tree showing the full node graph with children, meshes, primitives, cameras, lights, and skins. Supports selection, expansion, drag-and-drop, and right-click context menus.
- **Elements** — an editor-grade list with one icon tab per glTF collection (Nodes, Meshes, Materials, Cameras, Lights, Textures, Images, Samplers, Animations). Each row shows the element's **glTF index (`#`)**, its name, and a few browse columns that make the list an asset-review tool — mesh **triangles**/**instances**, material **used-by** with a color swatch, image **resolution** and **reference count**, animation **duration**, and so on. Columns are click-to-sort (e.g. heaviest mesh, largest texture), there is a name **filter**, and the footer shows aggregates (total triangles, texture memory). A uniform toolbar provides **Add / Duplicate / Delete / Rename** per category; selecting a row drives the **Inspector**. Nodes also carry an inline visibility (eye) toggle. Selection stays in sync with the Scene Graph tree and the viewport.

![The Elements tab's Nodes list with a node selected, showing the synced Inspector](images/elements_list.png)

Asset-level metadata (glTF asset info, generator, copyright, `KHR_xmp_json_ld`) is shown at the top.

### Node Operations

Right-click any node in the Scene Graph tree, use the Elements tab's Nodes toolbar, or press the shortcuts:

| Operation | Shortcut | Description |
|---|---|---|
| **Add Child** | — | Create an empty child node, or a Point / Directional / Spot light node under the selected node |
| **Duplicate** | `Ctrl+D` | Deep-copy the node and its entire subtree (geometry, materials, hierarchy) |
| **Delete** | `Del` | Delete the node and all descendants (undoable) |
| **Rename** | — | Inline rename for any node, mesh, or material |
| **Re-parent** | Drag & drop | Drag a node onto another to re-parent it. Drop on the Scene root to make it a root node. Cycle detection prevents invalid hierarchies. |

### Undo / Redo

All node editing operations support full undo/redo:

| Shortcut | Action |
|---|---|
| `Ctrl+Z` | Undo the last operation |
| `Ctrl+Y` | Redo the last undone operation |

The **Edit** menu also provides Undo and Redo items with a description of the action (e.g., "Undo Transform 'Cube'", "Redo Delete 'Light'").

**Supported operations:** Transform (gizmo + inspector), Material editing (PBR properties + all extensions), Light editing (type, color, intensity, range, spot angles), Rename node, Duplicate, Delete, Add Child, Add Light, Re-parent (drag & drop).

The undo history uses a linear model: performing a new action after an undo discards the redo stack. History is automatically cleared when loading, merging, or referencing a scene to prevent stale references.

**Current limitations:**
- Visibility toggles and scene-level transforms are not yet undoable.
- If an animation is playing, it may overwrite an undo-restored transform on the next frame — pause the animation first.

### Transform Editing

- **Inspector panel**: Edit Translation, Rotation, and Scale (TRS) numerically with precision.
- **Transform Gizmo**: Enable via toolbar or **View → Gizmo** (`T`) for interactive translate/rotate/scale directly in the viewport.
- **Scene Transform**: A popup on the scene root applies a global transform to all root nodes — useful for reorienting imported assets (e.g., Z-up to Y-up).

### Material Editing

![The material Inspector showing PBR fields, texture slots, and the extensions list](images/material_editor.png)

The **Inspector** panel provides full PBR material editing when a material or primitive is selected:

- Base color factor and texture, metallic, roughness, emissive (factor + strength)
- Normal map scale, occlusion strength, alpha mode and cutoff
- Double-sided toggle
- All glTF PBR material extensions: clearcoat, transmission, volume, volume scatter, sheen, specular, IOR, iridescence, anisotropy, diffuse transmission, dispersion, emissive strength, unlit, and retroreflection
- Material copy/paste via clipboard (right-click on materials)

Each extension gets its own collapsible section at the bottom of the material Inspector. A material that
doesn't carry the extension shows a single **Add** button, which attaches it with spec-default values and
expands the section; once present, the section instead shows its fields plus a **Remove** button, which
detaches the extension (and its data) from the material. Adding or removing an extension is a normal,
undoable material edit, like any field change above it.

`KHR_materials_pbrSpecularGlossiness` is a read/edit-only exception to this: a material that already
carries it shows its diffuse/specular/glossiness fields in place of the metallic-roughness ones above, but
the extension itself has no Add/Remove control — this renderer edits specular-glossiness materials it
loads, it does not author new ones.

Each texture slot shows a thumbnail of the assigned image — click it to open the full-size **image
viewer** — and offers **switch** to another existing texture (a filterable, thumbnailed picker), **load
from file** (import a new image — available even on a scene that starts with no textures), **clear**, and
a **UV transform** button. The transform button opens
a small popup for `KHR_texture_transform` (offset, rotation, scale): glTF stores the transform on the
material's texture *reference* — not the shared texture — so it is edited here per binding, and the same
texture used by two materials can carry two different transforms. The button offers **Add** when the
extension is absent, then the fields plus **Remove**. Imported images are referenced externally and copied
next to the file on save.

### Textures, Images, and Samplers

The **Elements** tab lists **Textures**, **Images**, and **Samplers** with thumbnails alongside the text
(text stays the fastest way to find one in large scenes), and selecting a row edits it in the **Inspector**.
The **Textures** list shows each texture's image resolution and sampler summary; its Inspector edits the
`{ source, sampler }` indices (matching the glTF texture object) and the referenced sampler's wrap/filter
inline. The **Samplers** list shows wrap/filter and how many textures use each; its Inspector edits the
wrap and filter modes. The per-binding UV transform is edited from the material Inspector (above). The
**Images** list reports each image's resolution and how many textures reference it — sort by resolution to
find the heaviest textures, and the footer totals the decoded texture memory. The Image Inspector opens the
full **image viewer** and can **replace from file** or **reload**; an image that no texture references can be
deleted from the toolbar (refcount-gated). Other unused resources are cleared by **Compact Scene** (`Ctrl+K`).

### Lights

The editor supports two kinds of lights, both visible in the unified **Lights** list in the Elements tab:

- **KHR_lights_punctual** — point, directional, and spot lights. Created from the node hierarchy context menu or the **Add ▾** button on the Lights element list.
- **EXT_lights_ies** — standalone photometric lights whose angular distribution comes from an IESNA LM-63 `.ies` file. These are loaded from the glTF; they do not require a `KHR_lights_punctual` companion.

**Creating KHR punctual lights:**

- Right-click any node in the hierarchy and select **Add Child → Light → Point / Directional / Spot Light**
- Right-click the Scene root and select **Add → Light → Point / Directional / Spot Light**
- Use **Add ▾** on the Elements tab's **Lights** (or **Nodes**) category

Each light is created as a new node in the scene graph. Position and orient the light by editing the node's transform (inspector or gizmo).

**Editing KHR light properties:**

When a KHR light is selected, the Inspector shows:

| Property | Applies to | Description |
|---|---|---|
| **Type** | All | Switch between point, directional, and spot |
| **Color** | All | Light color (edited in sRGB, stored as linear) |
| **Intensity** | All | Brightness multiplier |
| **Range** | Point, Spot | Maximum range of the light (0 = infinite) |
| **Inner Cone Angle** | Spot | Angle of full-intensity cone (radians) |
| **Outer Cone Angle** | Spot | Angle of light falloff cone (radians) |

All light property edits are undoable.

**Deleting lights:** Select the light's node in the hierarchy and delete it (`Del`). Use **Tools > Compact Scene** to remove orphaned light definitions from the file.

### IES Photometric Profiles

[EXT_lights_ies](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/EXT_lights_ies)
replaces a light's smooth falloff with the angular distribution measured from a real luminaire —
a ring downlight, a barn-door spot, a wide scatter fixture, and so on. The profile is an
IESNA LM-63 `.ies` file referenced by the glTF — either embedded as a bufferView or as an
external `.ies` URI alongside the `.gltf` file.

Different profiles produce dramatically different results on the same geometry:

| Ring + scatter | Tight beam + umbrella | X-arrow + jellyfish | Parallel beam + comet |
|---|---|---|---|
| ![](images/LightsIES_original.jpg) | ![](images/LightsIES_tightbeam_umbrella.jpg) | ![](images/LightsIES_xarrow_jellyfish.jpg) | ![](images/LightsIES_parallelbeam_comet.jpg) |

When an IES node is selected, the Inspector shows an **IES LIGHT** section with editable
**Multiplier** (brightness scale) and **Color** (linear RGB tint). Both are undoable. The
profile itself is read-only — it refers to the `.ies` file the scene was saved with.

![IES inspector panel — editable Multiplier and Color](images/LightsIES_inspector.png)

### Scene Merging

Multiple glTF files can be combined into a single scene:

- **File > Import/Merge Scene...** opens a file dialog to select a `.gltf` or `.glb` file (embeds it into the current scene, or opens it as a new scene if none is loaded)
- **Shift + Drag & Drop** a file onto the viewport to merge instead of replace
- Merged content is wrapped under a new root node, preserving both scenes' hierarchies
- Texture count is validated against the GPU descriptor limit before merging

To keep external files as **references** (glTF 2.1 external assets) instead of embedding them,
use **File > Reference Scene...** or **Ctrl+Shift + Drag & Drop**. Referenced assets stay
read-only, repeated references share geometry, and the links are preserved on save. Use the
context-menu **Make Editable** to break the lock on a referenced subtree, and
**File > Save Self-Contained As...** to bake references inline into a portable file. See
[External Assets](external_assets.md) for details.

### Save and Compact

- **File > Save / Save As** writes the modified scene to `.gltf` or `.glb`, including all edits (transforms, hierarchy changes, material tweaks, merged content, image copying)
- **Tools > Compact Scene** removes orphaned resources (unused meshes, materials, textures, images, accessors, buffer views, buffers) left behind by delete/merge operations and consolidates all geometry into a single buffer — reduces file size, cleans up the model, and collapses the extra buffers that editor operations (e.g. adding a primitive) leave behind. Pre-baked `EXT_mesh_opacity_micromap` data is preserved through compaction so OMM coverage survives the save/compact workflow.

### Visibility

Nodes with the `KHR_node_visibility` extension can be toggled visible/hidden directly from the hierarchy. Visibility propagates to children and is reflected in both renderers immediately.

### Selectability & Hoverability

The Inspector's node **Transform** panel exposes two interaction flags, next to **Visible**:

- **Selectable** (`KHR_node_selectability`) — when unchecked, clicking the node (or any of its children) in the viewport no longer selects it. Selection instead falls through to the nearest selectable ancestor, or is cleared if none exists. This lets an author mark decorative or grouping geometry as non-pickable while still selecting the meaningful parent object.
- **Hoverable** (`KHR_node_hoverability`) — a companion flag intended for hover interactions (primarily consumed by `KHR_interactivity`). It is parsed, editable, and preserved on save.

Both flags cascade to the whole subtree (a `false` on an ancestor disables its descendants) and default to enabled. Use **Add Selectability** / **Add Hoverability** to attach the extension to a node that does not yet carry it.

### Interactivity

Assets that carry a `KHR_interactivity` behavior graph run it automatically once the scene loads (lifecycle events, flow control, variables, math, `pointer/get`/`set` scene writes, and hover/select events — see [docs/interactivity.md](interactivity.md) for exact coverage). A small Play/Pause indicator appears in the viewport toolbar whenever the loaded scene actually has a graph — it's your at-a-glance sign that "this scene is interactive," and clicking it toggles the graph directly, no window required.

For live stats, variables, and a debug/log surface, open **View → Windows → Interactivity** (or press **F8**). This window is closed by default — most viewing doesn't need it — and gives you direct control over the graph:

- **Play / Pause** — stop or resume ticking the graph.
- **Reset** — discard all runtime state (variables, timers, pending delays) and restart it from scratch on the next tick.
- **Live stats** — node/variable/event counts, started/ticked state, and elapsed time.
- **Variables** — the current value of every graph variable, by index (the spec doesn't name them).
- **Send Event** — pick one of the graph's declared custom events and fire it manually, the same delivery path `event/send` nodes use internally.
- **Log** — the running history of `debug/log` output from the graph, useful for debugging authored content without a console.

Node support, including `animation/start`/`stop`/`stopAt` clip playback, is broad but not exhaustive; there is no visual node/wire graph debugger — see [docs/interactivity.md](interactivity.md) for the current coverage table. Loading a scene with a behavior graph disables the Animation Strip's autoplay by default (per spec, the graph is assumed to control all animations), though you can still scrub manually.


While the graph is playing, clicking a node in the viewport always re-selects it and re-fires `event/onSelect`, even if it was already selected — unlike normal editing, where clicking the selected node again deselects it. This makes click-driven behavior (a lever, a button) respond to every click instead of every other one. Clicks also register instantly while playing, skipping the brief single/double-click debounce normal editing uses to distinguish a click from a double-click-to-recenter-camera. Pause the graph to get ordinary click-to-deselect/double-click-recenter editing back.

Don't have a `KHR_interactivity` scene handy? [glTF-Test-Assets-Interactivity](https://github.com/KhronosGroup/glTF-Test-Assets-Interactivity) has the official Khronos conformance and showcase scenes, and the [Needle glTF Interactivity Editor](https://gltf-interactivity.needle.tools/) is a browser-based visual editor for authoring your own behavior graph and exporting it as glTF — see [docs/resources.md](resources.md) for both.

---

## Animation

If the loaded scene contains animations, an **Animation** control panel appears:

![](images/animation_controls.png)

- **Play / Pause** the active animation
- **Step** forward one frame at a time
- **Reset** to the beginning
- Adjust **playback speed** (0 to 100x, default 1x)
- **Timeline scrubbing** — drag the slider to any time position

Supported animation types: keyframe translation/rotation/scale, skeletal skinning, morph targets, and `KHR_animation_pointer` (animated material and light properties).

## Multiple Scenes

If the glTF file contains multiple scenes, a scene selector appears. Click a scene name to switch.

![](images/multiple_scenes.png)

## Material Variants

If the scene uses `KHR_materials_variants`, a variant selector shows all variant names. Click to apply a variant to all meshes that support it.

![](images/material_variant.png)

---

## Debug Visualization

Inspect individual material channels to diagnose shading issues:

|metallic|roughness|normal|base color|emissive|opacity|tangent|tex coord|
|---|---|---|---|---|---|---|---|
|![](images/dbg_metallic.jpg)|![](images/dbg_roughness.jpg)|![](images/dbg_normal.jpg)|![](images/dbg_base_color.jpg)|![](images/dbg_emissive.jpg)|![](images/dbg_opacity.jpg)|![](images/dbg_tangent.jpg)|![](images/dbg_tex_coord.jpg)|

Select the visualization mode from the **Visualization** combo in the **Settings** panel. The full set of modes is defined by `shaderio::Visualization` in `shaders/shaderio.h`. See also the [Opacity Micromap](#opacity-micromap-ext_mesh_opacity_micromap) section for the OMM coverage debug view.

---

## Material Feature Showcase

The ray tracing path tracer implements all glTF PBR material extensions with physically accurate results. Here is a selection of material features in action:

| | |
|--|--|
| Anisotropy | ![](images/AnisotropyBarnLamp.jpg) ![](images/AnisotropyDiscTest.jpg) ![](images/AnisotropyRotationTest.jpg) ![](images/AnisotropyStrengthTest.jpg) <br> ![](images/CompareAnisotropy.jpg)|
| Attenuation | ![](images/DragonAttenuation.jpg) ![](images/AttenuationTest.jpg)|
| Alpha Blend | ![](images/AlphaBlendModeTest.jpg) ![](images/CompareAlphaCoverage.jpg) |
| Clear Coat | ![](images/ClearCoatCarPaint.jpg) ![](images/ClearCoatTest.jpg) ![](images/ClearcoatWicker.jpg) ![](images/CompareClearcoat.jpg)|
| Dispersion | ![](images/DispersionTest.jpg) ![](images/DragonDispersion.jpg) ![](images/CompareDispersion.jpg) |
| IOR | ![](images/IORTestGrid.jpg) ![](images/CompareIor.jpg) |
| Emissive | ![](images/EmissiveStrengthTest.jpg) ![](images/CompareEmissiveStrength.jpg) |
| Iridescence | ![](images/IridescenceAbalone.jpg) ![](images/IridescenceDielectricSpheres.jpg) ![](images/IridescenceLamp.jpg) ![](images/IridescenceSuzanne.jpg) |
| Punctual Lights | ![](images/LightsPunctualLamp.jpg) ![](images/light.jpg) |
| Sheen | ![](images/SheenChair.jpg) ![](images/SheenCloth.jpg) ![](images/SheenTestGrid.jpg) ![](images/CompareSheen.jpg) |
| Transmission | ![](images/TransmissionRoughnessTest.jpg) ![](images/TransmissionTest.jpg) ![](images/TransmissionThinwallTestGrid.jpg) ![](images/CompareTransmission.jpg) <br> ![](images/CompareVolume.jpg) ![](images/GlassBrokenWindow.jpg) ![](images/MosquitoInAmber.jpg) |
| Volume | ![](images/volume.png) ![](images/volume_scatter.png) |
| Variant | ![](images/MaterialsVariantsShoe_1.jpg) ![](images/MaterialsVariantsShoe_2.jpg) ![](images/MaterialsVariantsShoe_3.jpg) |
| Others | ![](images/BoxVertexColors.jpg) ![](images/Duck.jpg) ![](images/MandarinOrange.jpg) ![](images/SpecularTest.jpg) ![](images/NormalTangentTest.jpg) ![](images/NormalTangentMirrorTest.jpg) <br> ![](images/BarramundiFish.jpg) ![](images/CarbonFibre.jpg) ![](images/cornellBox.jpg) ![](images/GlamVelvetSofa_1.jpg) ![](images/SimpleInstancing.jpg) ![](images/CompareSpecular.jpg) |

---

## Tools

### GPU Profiler

Measure time spent on each rendering stage (path tracing, rasterization, tone mapping, UI) with per-frame GPU timestamps.

![](images/profiler.png)

### Logger

A dockable log window showing all application messages. Filter by level (Info, Warning, Error) to focus on what matters.

![](images/logger.png)

### NVML GPU Monitor

Real-time GPU monitoring via NVML: temperature, power draw, memory usage, and clock speeds. Useful for identifying thermal throttling during long renders.

![](images/nvml.png)

### Tangent Space Repair

Repair or regenerate the model's tangent space when normal maps look incorrect or tangent-related validation errors appear.

| Method | Description |
|---|---|
| **Simple** | UV gradient method — fast, good for most cases. Based on [Foundations of Game Engine Development](https://foundationsofgameenginedev.com/FGED2-sample.pdf). |
| **MikkTSpace** | Industry-standard algorithm from [mikktspace.com](http://www.mikktspace.com/). Handles UV seams correctly with vertex splitting. |

### Shader Hot-Reload

Press **Ctrl+Shift+R** or use `Tools > Reload Shaders` to hot-reload all Slang shaders without restarting the application. Shader source files in the `shaders/` directory are recompiled on-the-fly, making it ideal for rapid shader development and debugging.

> **Note:** Hot-reload requires the Slang compiler and shader source files to be accessible at runtime (handled automatically by the build system's `copy_to_runtime_and_install`).

### Exporting Images

| Action | Shortcut | Description |
|---|---|---|
| **Save Image** | `Ctrl+Alt+I` | Save the current tonemapped render to a PNG file (alpha channel preserved for compositing). |
| **Save Screen Image** | `Ctrl+Alt+Shift+I` | Save a screenshot of the full application window including UI. |

Both are also available from **File > Save Image** and **File > Save Screen Image**.

### Memory Statistics

Open via **Windows > Memory Usage**. Displays GPU memory allocation broken down by category (textures, buffers, acceleration structures, etc.). Useful for tracking memory consumption on large scenes.

### Resetting the UI and Settings

Two entries at the bottom of the **Windows** menu put the application back to a known state without
having to quit and delete the `.ini`:

- **Reset UI Layout** — re-docks every panel where a fresh run puts it, leaving all settings alone.
  Use it after a panel has been dragged somewhere unhelpful or ended up floating off-screen.
- **Reset All to Default** — asks for confirmation, then returns every setting to its built-in
  default *and* restores the default layout: the state of a first launch with no `.ini`. The loaded
  scene, its edits, and the loaded environment image stay as they are -- but every *setting* resets,
  including the ones that drive the environment (sky vs. HDR, lighting, background), so the image
  can visibly change. This is not undoable with `Ctrl+Z`.

Both are also reachable from a script, a benchmark sequence, or MCP as `--resetUiLayout` and
`--resetAllToDefault`, which is how the behavior is exercised in a scripted UI run.

---

## Configuration

### Settings File

The application creates a `vk_gltf_renderer.ini` file next to the executable, which persists:

- UI layout and window positions (via ImGui)
- User preferences (axis visibility, grid display, gizmo state)
- Selected renderer type (path tracer or rasterizer)
- Path tracer settings (technique, adaptive sampling, performance target)
- DLSS settings (enable, size mode)
- Last used environment and rendering options

If you encounter UI issues or want to reset all settings to defaults, use **Windows > Reset All to
Default** (see above), or quit and delete this file — it is recreated with default values on the
next launch.

### Command-Line Reference

All settings can be overridden from the command line using `--paramName value` syntax.

> The tables below cover the commonly used flags. The **authoritative, complete** set is
> registered in code via `nvutils::ParameterRegistry` — see the `registerParameters()` /
> `parameterRegistry.add(...)` calls in `src/main.cpp`, `src/renderer.cpp`,
> `src/renderer_pathtracer.cpp`, `src/renderer_rasterizer.cpp`, and `src/benchmarking.cpp`.
> Names and value ranges there take precedence over this list.

**General**

| Parameter | Description |
|---|---|
| `--scenefile <path>` | Input scene file (.gltf, .glb) |
| `--hdrfile <path>` | Input HDR environment file (.hdr) |
| `--agenticBridgeInit` | Create the optional external generation bridge manifest/directories and exit |
| `--agenticBridgeRoot <path>` | Bridge folder for the optional generation bridge, used by `--agenticBridgeInit` and at runtime (default: `agentic_bridge` next to the executable) |
| `--size <W> <H>` | Window size |
| `--headless` | Run without UI (batch mode) |
| `--frames <N>` | Number of frames to render in headless mode |
| `--output <path>` | Output image file path for headless mode (default: `<exe_name>.jpg` next to executable) |
| `--vsync` | Enable vertical sync |
| `--vvl` | Activate Vulkan Validation Layers |
| `--logLevel <N>` | Log level (nvutils values): Stats (1), Info (3), Warning (4), Error (5) |
| `--logShow <N>` | Extra log info (bitset): None (0), Time (1), Level (2) |
| `--device <index>` | Force a specific Vulkan GPU by device index |
| `--vsyncOffMode <0-3>` | VSync-off present mode: Immediate (0), Mailbox (1), FIFO (2), FIFO Relaxed (3) |
| `--floatingWindows` | Allow dock windows to be separate OS windows |
| `--mcp` | Serve the shader-timing tools over MCP on `http://127.0.0.1:7671/mcp` (see [MCP shader timing](mcp.md)) |
| `--mcpPort <N>` | Port for the `--mcp` endpoint |

The bridge is also driven from the **Agentic** window (press F7, or open it from the Windows menu). From there, queue an HDRI prompt job or export the current render for image-to-image enhancement through an external adapter such as ComfyUI. See [ComfyUI Agentic Setup](comfyui-agentic-setup.md).

**Display**

| Parameter | Description |
|---|---|
| `--uiShowAxis` | Show the 3D axis widget in the viewport |
| `--uiShowMemStats` | Open the Memory Statistics window on launch |
| `--silhouetteColor <R> <G> <B>` | Selection silhouette color (0.0-1.0) |

**Rendering**

| Parameter | Description |
|---|---|
| `--renderSystem <0-1>` | Path tracer (0) or Rasterizer (1) |
| `--envSystem <0-2>` | Sky (0), HDR (1), None (2) |
| `--ptMaxFrames <N>` | Maximum path tracer iterations |
| `--dbgVisualization <N>` | Visualization mode (0 = Rendered). Values map to `shaderio::Visualization` in `shaders/shaderio.h` — see that enum for the current list. |
| `--useSolidBackground` | Use solid background color |
| `--solidBackgroundColor <R> <G> <B>` | Solid background color (0.0-1.0) |

**Path Tracer**

| Parameter | Description |
|---|---|
| `--ptTechnique <0-1>` | Ray Query (0) or Ray Tracing pipeline (1) |
| `--ptMaxDepth <N>` | Maximum ray bounce depth |
| `--ptSamples <N>` | Samples per pixel per frame |
| `--ptFireflyClamp <val>` | Firefly clamp threshold |
| `--ptAperture <val>` | Depth-of-field aperture |
| `--ptFocalDistance <val>` | Focal distance |
| `--ptAutoFocus` | Enable auto-focus |
| `--ptAdaptiveSampling` | Enable adaptive SPP to meet FPS target |
| `--ptPerformanceTarget <0-3>` | Interactive (0), Balanced (1), Quality (2), Max Quality (3) |

**Rasterizer**

| Parameter | Description |
|---|---|
| `--dbgWireframe` | Enable the wireframe overlay (global setting; both renderers honor it) |
| `--rasterUseRecordedCmd` | Use recorded (secondary) command buffers |

**Denoisers**

| Parameter | Description |
|---|---|
| `--dlssEnable` | Enable DLSS Ray Reconstruction |
| `--optixEnable` | Enable OptiX AI Denoiser |
| `--optixAutoDenoiseEnabled` | Auto-denoise every N frames |
| `--optixAutoDenoiseInterval <N>` | Auto-denoise interval (frames) |

**Tone Mapping**

| Parameter | Description |
|---|---|
| `--tmMethod <0-5>` | Filmic (0), Uncharted (1), Clip (2), ACES (3), AgX (4), Khronos PBR (5) |
| `--tmActive <0-1>` | Enable tone mapping |
| `--tmExposure <0.1-200>` | Exposure multiplier |
| `--tmContrast <0-2>` | Contrast |
| `--tmBrightness <0-2>` | Brightness (was `--tmGamma`) |
| `--tmSaturation <0-2>` | Saturation |
| `--tmVignette <-1..1>` | Vignette (was `--tmWhitePoint`) |
| `--tmDither <0-1>` | Dither |
| `--tmTemperature <2000-15000>` | White balance temperature, Kelvin |
| `--tmTint <-0.03..0.03>` | White balance tint (Duv) |
| `--tmVibrance <-1..1>` | Boosts muted colors only |
| `--tmShadowBias <-1..1>`, `--tmMidtoneBias`, `--tmHighlightBias` | Tonal range bias |
| `--tmCoolColor <R> <G> <B>`, `--tmWarmColor <R> <G> <B>` | Split-toning tints |
| `--tmSplitBalance <-0.5..0.5>` | Split-toning balance |
| `--tmAutoExposure <0-1>` | Auto-exposure (turn **off** for reproducible captures) |
| `--tmAutoExposureSpeed <0-100>` | Adaptation speed |
| `--tmEvMin <-24..24>`, `--tmEvMax <-24..24>` | Auto-exposure clamp, EV100 |

**Environment**

| Parameter | Description |
|---|---|
| `--envSystem <0-2>` | Sky (0), HDR (1), None (2) |
| `--hdrfile <path>` | HDR to load; loads immediately when set at runtime |
| `--hdrIntensity <0-100>` | HDR environment intensity |
| `--hdrRotation <-180..180>` | HDR environment rotation, **degrees** |
| `--hdrBlur <0-1>` | HDR environment blur |

**Sun & Sky** (used when `--envSystem 0`)

| Parameter | Description |
|---|---|
| `--skySunAzimuth <-180..180>` | Sun azimuth, degrees |
| `--skySunElevation <-90..90>` | Sun elevation, degrees — the "time of day" control |
| `--skyMultiplier <0-10>` | Overall sky brightness |
| `--skyHaze <0-15>` | Haze |
| `--skyRedBlueShift <-1..1>` | Red/blue shift |
| `--skySaturation <0-1>` | Saturation |
| `--skyHorizonHeight <-1..1>`, `--skyHorizonBlur <0-5>` | Horizon placement and softness |
| `--skyGroundColor <R> <G> <B>`, `--skyNightColor <R> <G> <B>` | Ground and night tints |
| `--skySunDiskScale <0-10>`, `--skySunDiskIntensity <0-5>`, `--skySunGlowIntensity <0-5>` | Sun disk and glow |

**Headless / Batch Rendering Example:**

```bash
./vk_gltf_renderer --headless --scenefile shader_ball.gltf --hdrfile daytime.hdr --envSystem 1 --frames 1000 --output render.jpg
```

**Benchmarking (scripted regression)**

| Parameter | Description |
|---|---|
| `--benchmark` | Enable scripted benchmark mode (requires `--sequencefile` or `--sequencestring`) |
| `--sequencefile <path>` | Benchmark script (`.cfg`) with `SEQUENCE` blocks |
| `--sequenceframes <N>` | Frames per sequence (script override) |
| `--sequenceaverages <N>` | Profiler averaging window |
| `--sequenceresetframes <N>` | Warmup frames after each sequence |
| `--gltfCamera <index>` | Apply glTF camera (benchmark script) |
| `--fitScene` | Fit camera to scene bounds (benchmark script) |
| `--resetFrame` / `--updateData` | Reset path-tracer accumulation |
| `--screenshot <path>` | Save tonemapped image (benchmark script) |

Single-scene manual run:

```bash
./vk_gltf_renderer --benchmark 1 --size 1920 1080 \
  --sequencefile utils/benchmark/quick.cfg \
  --scenefile shader_ball.gltf --hdrfile std_env.hdr
```

Batch runs, CSV export, and baseline vs candidate comparison:

```bash
python utils/benchmark/benchmark.py headless --scene resources/shader_ball.gltf --frames 500
python utils/benchmark/benchmark.py run quick.cfg --scene resources/shader_ball.gltf --hdr std_env.hdr
```

See [Benchmarking](benchmarking.md) for headless A/B timing, script format, and interpretation of results.

---

## Utilities

### gltf-material-modifier.py

Located in `utils/gltf-material-modifier.py`. A Python 3 script to batch-modify materials in a glTF file and optionally reorient the scene from Z-up to Y-up.

```
usage: gltf-material-modifier.py [-h] [--metallic METALLIC] [--roughness ROUGHNESS] [--override] [--reorient]
                                 input_file output_file
```

| Argument | Description |
|---|---|
| `input_file` | Path to the input glTF file |
| `output_file` | Path to save the modified glTF file |
| `--metallic` | Set metallic factor (default: 0.1) |
| `--roughness` | Set roughness factor (default: 0.1) |
| `--override` | Override existing material values |
| `--reorient` | Reorient the scene from Z-up to Y-up |

---

## Troubleshooting

**Application won't start / black screen**
- Ensure you have an NVIDIA RTX GPU with up-to-date drivers (535+).
- Verify the Vulkan SDK is installed: run `vulkaninfo` from a terminal.
- Try with validation layers: `--vvl` to get detailed Vulkan error messages.

**DLSS not available**
- DLSS requires an RTX 20-series or newer GPU and the `USE_DLSS=ON` CMake option.
- Check that the NGX runtime is present; it is downloaded automatically during the build.
- The application will still run with DLSS disabled if hardware support is missing.

**OptiX AI Denoiser not working**
- Ensure the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) is installed and found by CMake.
- On Windows, `cudart64_*.dll` is delay-loaded — the application starts without it, but the denoiser tab will show as unavailable.
- OptiX headers are auto-downloaded; no separate OptiX SDK install is needed.

**Shader hot-reload fails (Ctrl+Shift+R)**
- Hot-reload requires the Slang compiler and shader source files to be accessible at runtime.
- Verify the `shaders/` directory is present next to the executable (handled by `copy_to_runtime_and_install`).

**Resetting UI layout**
- Delete the `vk_gltf_renderer.ini` file next to the executable. It will be recreated with defaults on the next launch.
