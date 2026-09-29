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
| **Max Depth** | Maximum number of bounces per path. |
| **FireFly Clamp** | Clamps high-intensity samples to reduce firefly artifacts in early frames. |
| **Texture LOD** | Ray-footprint gradient scale for texture mip selection: 0 always samples mip 0, 1 uses the full physically derived LOD. |
| **Shadow Transmission** | Lets shadow rays pass straight through transmissive surfaces (`KHR_materials_transmission`), tinted by base color, Fresnel, and volume absorption. On by default: it brightens what lies behind glass and gives colored shadows, but it is a **biased approximation**, since it ignores refraction (no focused caustics) and adds to the light BSDF-sampled paths already carry through the surface. Turned off, every transmissive surface (thin-walled included) blocks shadow rays and light through it comes only from BSDF sampling. Nothing is approximated, but caustics from small or punctual lights (sun, point, spot, directional) are effectively missing: plain path tracing cannot sample light that reaches a diffuse surface through perfectly specular transmission. Alpha coverage is unaffected. |
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

**Environment Type** picks what lights the scene and what shows behind it. The list is built in
`UiEnvironment::render` (`src/ui_environment.cpp`) and each entry carries its own tooltip; the
sections below cover the ones with settings of their own.

Whatever the type, the renderer consumes it the same way: everything except **None** ends up as a
lat-long environment with an importance-sampling table behind it, so the path tracer, the
rasterizer's image-based lighting, and the background dome all share one code path. The analytic
types (**Plain**, **Gradient**, **Sky**) get there by being *baked* into that image — see
[Authored skies](#authored-skies) and, for the mechanics,
[developer.md § Environment Lighting](developer.md#environment-lighting).

**Reset** at the top of the panel restores the *selected* environment type to its defaults and
leaves everything else alone — the other sky types, the loaded HDR file, and the background color
all survive. Use **Windows > Reset All to Default** for the whole application.
The same action is available as `--envResetDefaults` for scripts.

**Orientation** turns the environment, and it is one control for every type that can be turned — the
HDR image, **Sky** and **Gradient** (a **Plain** sky looks the same from every direction, so it has
none). **Rotation** turns it about the Y axis, in degrees. It edits the quaternion the extension
stores, so what you see is exactly what a save writes. A tilt that a glTF's `OMI_environment_sky`
`rotation` carries is rendered and kept — the slider changes only the heading — and
`--envRotation x y z w` sets the full quaternion from the command line, in the same glTF order.

The rotation is also where north is: turning a sky places its sun again from Time of Day, on the
turned compass — see [Sun & Time of Day](#sun--time-of-day). A sun you aimed by hand is re-placed
the same way, as the old North Offset did. **Reset** puts the orientation back to none along with the
rest of the selected type.

### Authored skies

**Plain** and **Gradient** are the two sky types of the
[OMI_environment_sky](https://github.com/omigroup/gltf-extensions/tree/main/extensions/2.0/OMI_environment_sky)
glTF extension that this renderer evaluates. They are *scene* data, not viewer preferences: a scene
that carries the extension selects its own sky on load, and **Save with scene** writes the current
sky back out when the scene is saved.

- **Plain** — one solid color. It lights the scene and shows as the background. A plain sky at
  black renders like **None** but keeps the sky in the glTF; pick **None** when you actually want
  the no-environment fast path.
- **Gradient** — bottom / horizon / top colors with a curve on each half, plus a sun disk and glow.
  Each curve controls how sharply its band fades into the horizon color; 1.0 is a linear ramp and
  smaller values tighten the transition toward a hard horizon.

**The sun is never in the baked image.** The bake is the field that lights the scene, and a sun in
it would be counted twice — once when a ray happens to hit it, once when the renderer samples the
sun as a light. So the sky is baked without it and the sun is always a light, which also keeps the
disk sharp: a half-degree sun is about one texel of a 1024×512 lat-long.

The **Sky** type works the same way, for the same reason, though it gets there differently: its
baked image is a real atmosphere simulation and already contains the glow around the sun, so only
the disk itself is drawn on top. Expect it to redden as you lower the sun — that colour is the
sunlight's path through the atmosphere, not a tint.

Below the horizon the Sky type shows lit ground rather than black: a Lambertian surface at the
planet's own albedo, fading into haze toward the horizon. It lights the scene from below like any
other part of the environment, so expect warm bounce on downward-facing surfaces.

### Editing the atmosphere

The **Sky** and **Gradient** types split into two tabs — **Sun & Time** for aiming the sun over
the day, **Physical Sky** / **Gradient Sky** for what the sky itself looks like. They share the
same sun but nothing else, so scrolling past the group you weren't editing was the old panel's
main friction.

One collapsible group per physical concept, the same shape the Unreal Engine sky component uses.
Each atmospheric component owns *everything* about it — the strength coefficient, the tint, and
the shape of its density profile — rather than scattering those across a "simple" list and an
"advanced" fold. A scale height is not more advanced than a strength; it answers a different
question about the same substance.

**Preset** at the top applies a whole atmosphere at once — Earth, Mars, or a deliberately unearthly
one. Edit any slider afterwards and it reads **Custom**. Mars and Alien are plausible starting
points, not authoritative data.

The three atmospheric components, in the order light meets them on its way down:

- **Rayleigh** — scattering by the air molecules themselves. **Scattering** is the /km coefficient,
  **Tint** is which wavelengths it scatters hardest (blue on Earth), **Scale Height** is how fast
  the air thins with altitude (8 km on Earth).
- **Mie** — scattering by aerosols: haze, dust, smoke. Same **Scattering** + **Tint** pair, plus
  **Anisotropy** (how tight the glow around the sun is — 0.8 on Earth), **Albedo** (how much of
  what the aerosol removes it scatters vs absorbs), and **Scale Height** (1.2 km on Earth — haze
  hugs the ground far more closely than air does).
- **Ozone** — a high layer that *absorbs* without scattering, which is why its coefficient row is
  called **Absorption** rather than Scattering. **Center** is where the layer peaks (25 km on
  Earth), **Thickness** is how far it spreads.

Then the scene-scale bridge:

- **Aerial Perspective** — how much haze accumulates between the camera and what it is looking at;
  see the next section for how it scales with scene size.

And the world-shape facts most sessions leave alone:

- **Planet** — **Ground Radius**, **Ground Albedo** (what the ground reflects, and lights the
  lower half of the sky), **Atmosphere Thickness**, and **Observer Altitude** (how high the sky is
  baked from — fixed rather than following the camera, which is what makes one baked image valid
  for the whole scene).
- **Sun** — the star's own properties, not its position (that lives in the Sun & Time tab):
  **Angular Radius** (0.004675 rad ≈ half a degree on Earth; wider softens shadows) and
  **Irradiance** (what reaches the top of the atmosphere, per channel).

Most of these change only the scattering tables, which take about a tenth of a second to rebuild.
So a drag rebuilds them at reduced quality — enough to see the colour and shape move with the
slider — and the full rebuild happens once you let go. Expect the sky to settle slightly brighter
when you release. Ground albedo, sun angular radius and observer altitude are read directly by the
bake besides, so those respond immediately.

### Sky presets

**Preset → Save… / Load…** at the bottom of the Environment panel captures the current sky as a
`.sky.json`, or applies one. You can also drag a `.sky.json` straight onto the viewport.

It saves the whole sky -- type, colours, the full atmosphere -- plus the sun's angle, so a preset
restores the look and not merely the ingredients. A preset that was saved without a sun (a plain sky
has none) leaves your sun where it is rather than moving it.

The file is one entry of the `OMI_environment_sky` extension, which is the same thing a scene
carries, written on its own. It is plain indented JSON meant to be opened, diffed and hand-edited,
and a preset written by a newer build keeps whatever it knew that this one does not.

Scriptable as `--loadSkyPreset` / `--saveSkyPreset`:

```bash
vk_gltf_renderer --scenefile scene.gltf --loadSkyPreset presets/blue_hour.sky.json
```

A start-up preset is applied on the first frame, after the saved settings and the scene's own sky,
so it wins over both; `--saveSkyPreset` then writes the sky that is actually showing.

Loading a preset is **not** undoable -- like every other environment control. The exception is the
sun: if a light in your scene is marked as the sky's sun, aiming it is a scene edit and Ctrl+Z takes
it back.

### Aerial perspective

The air between the camera and what you are looking at — why distant ridges go pale and blue while
near ones stay crisp. **Sky → Aerial Perspective** says how many metres of air one scene unit is
worth; 1.0 is glTF's own answer, since the format specifies metres, and 0 turns it off.

It is a **distance**, not an opacity. Raising it says the scene is bigger, which keeps the result
inside the atmosphere model rather than tinting its output. A 10 km terrain authored 100 units
across wants 100.

Expect nothing on a small asset — and that is correct. Two metres of clear air does nothing, and a
shader ball has two metres of it. The effect needs a scene with real distance in it.

For a hazier day, raise **Mie Scattering**: that is the aerosol, and it is the knob with no
geometric side effect. Pushing Aerial Perspective far enough instead will eventually sink the scene
below the observer altitude in the Planet group, where there is no air left to model.

Both renderers show it. The path tracer's is the reference, and it is **shadowed**: air the scene
hides from the sun — inside a helmet, a room, a canyon — does not glow with sunlight, and shafts
form where the sun gets through a gap. Only the sun's own scattering is shadowed; the soft light the
rest of the sky adds stays. The rasterizer's haze is unshadowed, which is all it can be — it cannot
ask whether the sun reaches a point in mid-air — so at large scales the two can disagree inside
enclosed spaces.

**Exposure.** The Sky type is calibrated to sit where a loaded HDR sits, so it is usable at
exposure 1.0 without reaching for the tonemapper. Only ratios in an atmosphere model are physical;
the absolute scale is a unit choice, and the one this renderer uses is set by the HDR files people
load. A low sun is genuinely dimmer than midday, so expect the scene to darken as the sun sets —
that is the model, not a miscalibration.

**The sun is a light, and the sky says which one.** The extension describes an atmosphere, not a
light source, so a sky with a sun needs a directional light next to it — and the renderer marks
that light rather than guessing at one.

**Sun Source** under **Sun & Time of Day → Advanced** shows which light that is, and lets you
change it — pick *Renderer* for a sun the sky carries itself, or any of the scene's directional
lights to make that one the sun. The same switch is on a selected directional light in the
Inspector, as **Sky's Sun**. Choosing is undoable, and the choice travels with the file. Whichever
light is the sun draws with a **sun icon** in the Scene Browser instead of the usual lightbulb.

When a scene has exactly one directional light and nothing has said whether it is the sun, the
Advanced fold offers it — one click adopts it. With several lights it does not guess: pick the one
you mean.

Until you choose one, the renderer supplies the sun itself, so choosing an environment never edits
your scene. **Saving with Save with scene on writes it**, because a sky
saved without a sun renders unlit anywhere else. Save again and the same light is updated, not
duplicated; save a sky that has no sun and the light goes away with it.

Once the light exists it *is* the sun — moving it in the viewport moves the sky, and the panel's
angle sliders move it back, undoably. Because it is a real sun it has a diameter, so its shadows
carry a penumbra: widen **Angular Radius** under **Physical Sky → Sun** and the shadow edges
soften.

How much that sun actually *lights* depends on the sky. The **Sky** type's is a real sun and
dominates the scene the way daylight does. A gradient sky's is a drawn disk rather than a
measurement: its radiance is **Sun Color**, and spread over a half-degree disk that comes to a tiny
fraction of what the sky itself contributes — visible, but not a light you can work by. Add a
directional light if a gradient sky needs to cast real sunlight.

A scene's own directional lights are **ordinary lights and are left exactly as authored**. A
`KHR_lights_punctual` directional light is a delta light by specification — no angular extent, so
hard-edged shadows — and a scene may contain several; none of that makes one of them the sun. If
you want a soft-edged directional light of your own, give it a `radius` in its `extras` and the
renderer will sample it as an area light.

So a scene with a directional light under a sky that has a sun is lit by both: its own light, and
the sky's.

A scene authoring `physical` opens on the **Sky** type with that atmosphere applied: the
extension's Rayleigh and Mie coefficients, its Mie anisotropy and its ground colour all take
effect, and saving writes them back. Everything the extension does not describe — planet radii,
the ozone layer, the solar spectrum — stays at Earth values.

Sun angular radius and observer altitude are **not** written to the glTF. The extension has no
field for either, and both are viewer settings rather than descriptions of the scene's atmosphere,
so they persist in the .ini like blur does.

`panorama` is preserved through a load/save round trip and loads as an HDR environment when it
names an image.

#### Baking and refresh

Analytic skies are baked into a lat-long image of a fixed size (`kEnvBakeSize` in
`src/env_baker.hpp`). There is no setting for it, because nothing you would pick it for depends on
it: the sky is smooth and its sun is sampled as a light rather than looked up in the image, so the
sampling table is built at a coarser fixed size still, and the path tracer evaluates what the camera
sees per ray. The size only sets the sharpness of rough reflections, and of the physical sky's
horizon in the rasterizer.

Moving the sky previews with a cheap color-only re-bake and commits the full rebuild a frame after
the movement stops -- whether it came from a slider, from the transform gizmo on the sun light, or
from an animation driving that light. The rasterizer pays more for that preview than the path
tracer does, because it also has to refresh its prefiltered cubemaps; the path tracer reads the
lat-long image directly and updates essentially immediately.

### Sun & Time of Day

One sun serves every sky that has one, and this tab is where it is aimed. It sits next to the
look tab (**Physical Sky** or **Gradient Sky**) for the **Sky** and **Gradient** types; the other
three have no sun to point.

Simple first — the widgets are ordered by how often people reach for them.

**Time** is the fastest way to change what the sky looks like: drag it and the sun goes where it
really was at that moment, for the place and date under **Advanced → Location & Date**. The four
tick marks under the slider are that day's solar midnight, sunrise, noon and sunset, so a reading
of 06:00 has something to mean against.

**Jump To** offers the named moments — sunrise, both golden hours, noon, sunset, blue hour,
midnight. It *sets* Time, so it sits below the slider it drives rather than above. They are not
fixed clock times: each is an altitude the sun passes through, solved for your latitude and date,
which is why they move through the year and why most of them are greyed out inside the polar
circles. The sun does not rise at Tromsø in December, and the menu says so rather than picking a
plausible hour.

Which way north points is the environment's **Rotation**, at the top of the panel, rather than a
row of its own here. Everything in this tab is astronomy and speaks compass bearings; the rotation
is what ties them to your model's axes, so a building modelled facing any direction can still be lit
by the real sun — and the sky turns with the compass instead of disagreeing with it.

With no rotation, in a Y-up scene, north is **−Z** — the direction a glTF camera faces:

| Compass | Axis |
|---|---|
| North | −Z |
| East | +X |
| South | +Z |
| West | −X |

A Z-up scene rotates the same rule: north −Y, east +X, south +Y, west −X. Turning the environment
turns the compass around the scene, and the sun is placed again on it.

**Ctrl+Shift+L** in the viewport, held while moving the mouse, swings the sun directly: sideways
turns it, up and down raises and lowers it. A tooltip reports the angles while you drag, and the
whole drag is one undo step. (`Ctrl+L` alone is shader hot-reload.)

Under **Advanced** sit the widgets most sessions never touch, in order: **Sun Source** (which light
is the sun), **Azimuth** and **Elevation** in degrees for typing an exact angle (they are what the
command line carries and what a marked light stores), a **Gizmo on sun light** button that selects
the marked light and switches the transform gizmo on so it can be aimed in the viewport (needs a
marked light — with *Sun Source* on *Renderer* there is no node for a gizmo to hold, and the button
says so), and a nested **Location & Date** tree. That tree opens with a **Now** button — one click
fills the date, clock and UTC offset from this machine, the place left alone since it is the one
thing the computer cannot tell us — then **City**, **Latitude** and **Longitude** in degrees, then
**Date** as `yyyy-mm-dd`, and a **UTC Offset** in hours (a raw number rather than a named time
zone, since no zone database ships with the renderer). The panel opens on today and on your own
offset the first time it runs.

The city list runs north to south — the axis the sun cares about — and is chosen for spread rather
than for size: two inside the Arctic Circle, three near the equator, five in the southern
hemisphere, one on a half-hour offset. Picking one sets all three values. The name shown is
derived from the coordinates rather than remembered, so nudging either slider drops it to
*Custom*, and a scene already set at Tokyo's latitude reads as Tokyo.

**The offsets are standard time.** There is no daylight-saving rule, so a summer date at most of
these is an hour off until you nudge **UTC Offset** — a smaller lie than shipping a zone table
that goes stale. `--todCity` takes the name from the command line and forgives case and spacing:
`--todCity "new york"` and `--todCity NewYork` are the same place.

None of this is written to the glTF. `OMI_environment_sky` describes an atmosphere, not a moment,
and has no field for a place or a date; what travels with a saved scene is the sun light's
rotation, which captures the same thing. Location, date and time persist in the `.ini` so the
widget reopens where you left it, and every one of them is also a command-line flag — see
[benchmarking.md](benchmarking.md) for scripting a run:

```bash
vk_gltf_renderer --envSystem 0 --todCity Tokyo --todDate 2026-06-21 --todHour 16.5

# or spell the place out, which is what --todCity fills in
vk_gltf_renderer --envSystem 0 --todLatitude 47.37 --todLongitude 8.54 \
  --todDate 2026-06-21 --todUtcOffset 2 --todHour 19.5
```

![](images/sky_1.jpg) ![](images/sky_2.jpg) ![](images/sky_3.jpg)

### HDR Environment

Lighting can come from environment maps — Radiance `.hdr` or OpenEXR `.exr`. Drag and drop one onto
the viewport, or load via **File > Load HDR Environment** (`Ctrl+Shift+O`). The format is detected
from the file's contents, so a mis-spelled extension still loads.

With **Save with scene** enabled, the environment is written into the glTF as an
`OMI_environment_sky` `panorama` and reloads with the scene — the image itself stays an external
file, referenced by a URI relative to the glTF.

![](images/hdr_1.jpg) ![](images/hdr_2.jpg) ![](images/hdr_3.jpg) ![](images/hdr_4.jpg) <br> ![](images/hdr_5.jpg) ![](images/hdr_6.jpg) ![](images/hdr_7.jpg) ![](images/hdr_8.jpg)

The environment can be **blurred** to soften reflections and lighting:

![](images/hdr_1.jpg) ![](images/hdr_blur_1.jpg) ![](images/hdr_blur_2.jpg) ![](images/hdr_blur_3.jpg)

And **rotated** — with the panel's **Orientation**, shared by every environment — to position the light source where you need it:

![](images/hdr_1.jpg) ![](images/hdr_rot_1.jpg)

### No Environment

Setting the **Environment Type** to **None** disables every environment type entirely: the scene receives no environment lighting (only its own punctual and emissive lights contribute), and unless a **Solid Color** background is enabled the backdrop is black. Prefer this over dialing HDR intensity to zero — it also skips environment importance sampling and the dome pass.

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
| `--ptTexGradScale <val>` | Texture LOD ray-footprint scale (0-1) |
| `--ptShadowTransmission <0\|1>` | Shadow rays pass through transmissive surfaces (biased approximation) |
| `--ptAperture <val>` | Depth-of-field aperture |
| `--ptFocalDistance <val>` | Focal distance |
| `--ptAutoFocus <0\|1>` | Enable auto-focus |
| `--ptAdaptiveSampling <0\|1>` | Enable adaptive SPP to meet FPS target |
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
| `--tmExposure <0.01-200>` | Exposure multiplier |
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
| `--envSystem <n>` | Which environment source; the values are `shaderio::EnvSystem` and the flag's own `--help` text lists them |
| `--hdrfile <path>` | HDR to load; loads immediately when set at runtime |
| `--hdrIntensity <0-100>` | HDR environment intensity |
| `--envRotation <x> <y> <z> <w>` | Environment orientation as a unit quaternion, glTF order — `OMI_environment_sky`'s `rotation`. Turns the HDR, every sky, and the Time of Day compass |
| `--hdrBlur <0-1>` | HDR environment blur |

**The sun** (shared by every sky type that has one)

| Parameter | Description |
|---|---|
| `--sunAzimuth <-180..180>` | Sun azimuth, degrees |
| `--sunElevation <-90..90>` | Sun elevation, degrees — the "time of day" control |

**Physical sky atmosphere** — `--atmo*`; run `--help` for the current list and ranges. Everything
else about the atmosphere stays at Earth values for now.

**Physical sky** (the **Sky** environment type) is the `atmo*` group — one flag per row of the
**Physical Sky** tab, e.g. `--atmoPreset`, `--atmoMieScattering <R> <G> <B>` — described in
[Editing the atmosphere](#editing-the-atmosphere) above.

The authored skies add a `plain*`, `gradient*` and `env*` group of their own (colors, curves, bake
resolution, save-with-scene). Rather than repeat them here, run `--help`: every setting is declared
once and the command line is generated from those declarations, so `--help` cannot drift.

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
