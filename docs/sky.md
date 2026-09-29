# Sky

How this renderer's skies work: what each one models, what units they produce, where the sun comes
from, and what survives a glTF round trip.

This is about the **models**. The pipeline they feed — bake, alias table, prefiltered cubes, the
lighting/background split, preview vs commit — is described in
[developer.md § Environment Lighting](developer.md#environment-lighting), and is the same for every
sky type. Read that first if you are changing the pipeline rather than a sky.

## The sky types

`shaderio::EnvSystem` selects the source; the picker in `src/ui_renderer.cpp` is the user-facing
list. Three of them are analytic and are **baked** into the same lat-long image an HDR file would
produce:

- **Plain** — one colour. No sun.
- **Gradient** — nadir / horizon / zenith colours with a curve on each half, plus a drawn sun disk.
  Authored decoration: its sun is a colour you pick, not a measurement, so its brightness as a
  *light* is scaled against its own sky ([how bright it is](#how-bright-it-is)).
- **Sky** — a physical atmosphere (below). Its sun is a real sun.

**HDR** is a loaded image and skips the bake. **None** disables the environment.

Plain, gradient and the physical sky are all expressible as `OMI_environment_sky`, which is what a
scene saves and loads; see [glTF](#gltf-omi_environment_sky).

## The physical sky

A port of Eric Bruneton's *Precomputed Atmospheric Scattering* (2008), from the author's own
reference implementation. The maths is copied rather than redesigned — same formulae, same sample
counts, same LUT parameterisation, same function names and order — so the port can be read
side-by-side with upstream. Only the syntax layer moves: GLSL to Slang, fragment passes to compute
dispatches, and upstream's runtime shader-source generation to a uniform block, since Slang here is
compiled at CMake time.

- `shaders/sky_bruneton_functions.h.slang` — the maths, carrying upstream's BSD-3 notice.
- `shaders/sky_bruneton_io.h.slang` — parameters, LUT dimensions, bindings.
- `src/sky_bruneton.*` — the Vulkan half: LUT images, the precompute passes, the runtime
  descriptor set.

### The precompute

Six compute passes run in dependency order — transmittance, direct irradiance, single scattering,
then a loop of scattering density / indirect irradiance / multiple scattering for each order. Each
pass reads what the previous one wrote, so they serialise on a full barrier; there is no
parallelism to recover, the chain *is* the algorithm. The result is four lookup tables that the
bake samples.

The tables depend on the atmosphere, not on the sun or the camera, so this runs once at start-up
and again only when an atmosphere parameter changes — see [Cost](#cost).

The scattering-density pass is most of the bill: it integrates over every incident direction
for each of the ~1M texels of the 3D table, once per order; the multiple-scattering integration is
most of the rest. Along each zenith angle of that integral only `nu` varies, so every lookup lands
on the same row of `nu` slabs: the pass fetches that row once and blends it in registers, instead
of two filtered 3D fetches per sample. Its cost is quadratic in the sample count, which is why that count is a runtime value
rather than a constant — see `SkyBruneton::Quality`, and [Two quality tiers](#two-quality-tiers)
below for what that buys.

### Two quality tiers

Most atmosphere parameters — Rayleigh, Mie, ozone, the planet radii — change *nothing but* the
tables. Rebuilt only on a settle, they would be dead sliders: nothing moves until the mouse comes
up. So a drag rebuilds at preview quality, recorded into the frame rather than submitted and
waited on, and the settle that follows rebuilds at full quality.

`SkyBruneton::Quality` is the two dials that matter, both measured: the sample count of the density
integral and the number of orders. The LUT *dimensions* are deliberately not among them — they are
baked into the parameterisation the sampling functions invert, so changing them at runtime would
mean recompiling shaders, not rebinding a texture.

The preview is dimmer than the reference, because fewer orders means less accumulated light. That
is a real difference and it does step on release; it is the price of the slider moving at all. Both
tiers' numbers, and the measurement that chose them, are in [Cost](#cost).

### Fixed observer

The sky is baked from a fixed point above the planet surface, not from the camera. That is what
makes a single baked image valid for the whole scene: an observer that moved with the camera would
invalidate the image every frame, and with it the alias table and the prefiltered cubes. Altitude
is a settable parameter, and raising it visibly thins the horizon haze — the horizon moves farther
away, so there is more air between it and you.

### Aerial perspective, and the one function deliberately not ported

The air between the camera and a surface is the one thing a baked lat-long can never supply: it
depends on how far away the surface is. `shaders/aerial_perspective.h.slang` supplies it, and both
render paths call the same function so their haze cannot drift.

Upstream answers this with `GetSkyRadianceToPoint`, in two texture fetches, by subtracting
everything scattered beyond the surface from everything scattered beyond the eye. That function is
**not** ported — it was, and it was removed again. The scattering table is stored in two halves,
rays that meet the planet and rays that escape, and which half to read is decided by where the
*extended* ray goes. A view ray tipping past the horizon switches halves while the air being shaded
has not changed, and the halves do not agree: on a 10 km mountain it draws a hard line across the
rock. Upstream never saw it, because the only thing at its horizon was the ground.

So the segment is **marched** instead — no halves, no branch, and no difference of two large
nearly-equal numbers. Multiple scattering is the approximation that costs: Bruneton's tables
integrate it along whole rays and cannot return the local source term a march needs, so it comes
from the irradiance table as an isotropic term. Over a few kilometres that is a small part of a
small term, and it is smooth.

The march runs per primary hit. A camera-space froxel volume would amortise it across pixels, the
way Unreal and Hillaire do; measured cost is the number to decide on — see [Cost](#cost).

The atmosphere knows nothing of the scene, so the march treats every point of the segment as
sunlit. The path tracer corrects the sun's share: the march also returns the sun's single
scattering on its own and one point on the segment picked in proportion to it, and the path tracer
traces one shadow ray from there (`TraceShadow`, so glass and alpha-cut surfaces transmit as they do
for the sun's NEE). Visibility times that one-sample estimate replaces the unshadowed share — unbiased,
and converging with the rest of the image. The multiple-scattering term is light from the whole sky
and stays unshadowed; the rasterizer keeps the unshadowed total. The segment is marched in the
environment's frame, like the bake, so a tilted sky tilts its haze.

The bake answers the one case it can on its own — an observer looking at its own planet — with the
ground term in `evalSkyBruneton()` in `shaders/env_bake.slang`.

## Units, and why there is a calibration constant

**Only ratios in an atmosphere model are physical.** The absolute scale is a convention, and every
renderer picks one.

The model produces spectral radiance at three wavelengths. Turning that into a colour requires the
CIE colour-matching integral, and the three resulting factors differ widely — skipping it leaves
the sky not merely dark but the wrong colour. Those factors are physics, derived from upstream's
own tables, and are independently checkable.

What follows them is not physics. This renderer's scale is set by the HDR files people load: they
are authored to look right at exposure 1.0, and their environment integrals — the value
`nvvk::HdrIbl::getIntegral()` reports — cluster in a narrow band. An uncalibrated physical sky
lands about an order of magnitude above that band, which is why an uncalibrated build needs its
exposure pulled to a tenth before it looks like daylight.

`SKY_RADIANCE_TO_LUMINANCE` in `shaders/sky_bruneton_io.h.slang` therefore composes two constants:
the CIE factors, and a calibration that brings them into the renderer's band. **The header states
what to set the calibration to for physical output** — luminance in cd/m² and illuminance in lux —
for anyone who wants absolute units and will carry the exposure themselves.

Everything derived scales with that one constant: sky, ground, the solar disk, and the sun light
built from the disk. No ratio the model computes is affected.

> The sky this replaced did the same thing, with two unexplained constants in its parameter block.
> The difference is that this one says so.

## The sun

`OMI_environment_sky` describes a **medium**, not a light source — its own overview says to add
suns with `KHR_lights_punctual`. So a sky that has a sun needs a directional light beside it, and
the renderer has to know *which* light that is.

"The first directional light" is a guess, and a wrong one the moment a scene holds two. Instead the
sun light is **marked**, with `NV_sky_sun` in the node's `extras`:

- **No marked light** → the renderer supplies the sun itself. Nothing is added to the scene:
  choosing an environment is not an edit, so there is no node, no undo entry, and nothing written
  on save that nobody authored.
- **Marking is a user action**, from the Environment panel's *Sun Source* row or the Inspector's
  *Sky's Sun* checkbox on a directional light, through `SetSkySunCommand`. A light the user placed
  can therefore become the sun — before this existed the only sun that could be marked was one a
  save had invented, so an authored directional light was silently inert.
- **A marked light** → that light *is* the sun. The panel aims it through the undo stack like any
  other transform, and the sky follows it wherever it moves.
- **Every other light** → used exactly as authored. A `KHR_lights_punctual` directional light is a
  delta light by specification; nothing about being directional makes one a sun.

### How bright it is

Both skies with a sun end up publishing the same three things — a direction, a solid angle and an
illuminance — but only one of them measured any of it.

The **physical sky** measures its sun. The bake writes the disk's radiance in the renderer's own
units, and radiance times the solid angle the disk subtends is an illuminance. Nothing is invented.

The **gradient sky** measures nothing. `sunColor` is the colour the disk is *drawn* in, in the same
arbitrary linear units as the rest of that sky, where the dome sits near 1. Putting it through the
physical sky's conversion is a category error and a quiet one: a half-degree disk subtends 6.8e-5
sr, so a `sunColor` of 1 became 7e-5 lux beneath a dome delivering about 1.5 — a sun you could see
and could not light by. Its illuminance is **scaled against its own dome** instead, the one
quantity that sky does define, by `kGradientSunToSkyRatio` (`src/renderer.cpp`). `sunColor` keeps
its brightness as well as its hue.

That ratio is fitted to a measured clear day's direct-to-diffuse band rather than picked. It has to
be fitted *across* the band, because this sky cannot follow its shape: a real sky's diffuse half
dims as the sun drops, so direct:diffuse climbs faster than the sun's cosine, while the gradient's
dome does not move with the sun at all. Only a narrow window of the one free scale keeps every
reference elevation inside the band.

Below the horizon the gradient's sun light goes to zero, matching the disk: the gradient draws no
sun there, and a light that keeps shining after its disk has set would light the scene from
underneath. The physical sky needs no such rule — its transmittance takes the disk down on its own.

**Saving with the sky materialises the marked light.** A sky saved without one is half written:
another renderer builds the atmosphere, finds nothing to light it, and shows an unlit sky — a
failure invisible here, since this renderer supplies its own sun, which is why it is fixed where
the file is written. A re-save updates that light rather than adding another.

The marker records **who created the light**, because unmarking and deleting are not the same
thing. Saving a sky with no sun withdraws a light this renderer invented; a light that was already
in the scene merely stops being the sun. Switching between Sky and HDR never costs you a light you
authored.

### Where it points

Four things aim the sun — the panel's azimuth/elevation sliders, the transform gizmo on a marked
light, the viewport's **Ctrl+Shift+L** drag, and the **Time of Day** widget — and all of them go
through `SkySun::aim`, which answers what each would otherwise answer for itself: whether a marked
light owns the direction (then it is an undoable scene edit) or the renderer does, whether the
reported angles still match, and whether the image needs re-baking. Only the physical sky's bake
depends on the sun.

Time of Day is the only one that is not an angle. It takes a place, a date and a local clock
reading and computes where the sun actually was, by NOAA's algorithm in `src/sun_position.*` —
pure, self-contained, and tested against astronomy rather than recorded output. The named moments
(sunrise, the golden hours, the blue hour) are *solved* from the same function rather than
tabulated, so they follow the season and report honestly that most of them do not happen inside the
polar circles.

**None of it is serialized.** The extension describes a medium, not a moment, and the sun light's
rotation already captures the result exactly. Place, date and time persist in the `.ini`. When
comparing two runs, pin the sun explicitly — see [Verifying a change](#verifying-a-change).

The disk stops at the horizon, because the planet is in the way below it — the bake publishes where
that line falls for the observer it baked from. Without the cut the sun never sets, it slides down
the picture and sits on the ground.

Its radiance carries the transmittance toward the sun, so it reddens and dims as the sun drops,
sampled **once at the disk's centre** — correct to within the variation across half a degree, and
increasingly approximate the further `Sun Angular Radius` is pushed past the sun's true size.

The light's **intensity is the solar illuminance above the atmosphere**, not the attenuated value
shown here: the atmosphere travels in the same file, so a reader applies it rather than attenuating
twice.

Two consequences worth knowing:

- It has an **angular radius**, so its shadows carry a penumbra. `singleLightContribution` in
  nvpro_core2 already cone-samples any directional light with a non-zero angular size.
- The sun is **never in the baked image**. The bake is the field that lights the scene; a sun in it
  would be counted twice — once when a bounce ray hits it, once when next-event estimation samples
  it. The visible disk is composited by the background field instead. See
  `shaders/sky_background.h.slang`, which states the call rule at its definition because getting it
  wrong is silent and off by about a factor of two.

For the physical sky the disk's radiance is computed **by the bake**, on the GPU, because its colour
is the solar spectrum times the transmittance along the sun's own path through the atmosphere, and
that transmittance lives in a LUT the host never reads back. It is one value per sun and atmosphere,
and the bake already runs exactly when either changes, so it is computed there once rather than per
pixel by every background pass. It is published to a small device buffer and read by the background
passes through an address in `SceneFrameInfo` — no host readback, so a preview during a drag shows a
fresh disk rather than a stale one.

## What you can edit, and what travels with the scene

The whole atmosphere is editable, and every field is registered once through `SettingsRegistry`,
so the command line, benchmark scripts, MCP and the panel all reach the same value. The panel
itself is `src/ui_environment.cpp`; the field list is `atmo*` in `Settings` (`src/resources.hpp`).

The split that matters is **what the glTF carries**, not what can be edited:

- **Scene data** — the whole atmosphere. The scattering coefficients, Mie anisotropy and ground
  albedo go through `OMI_environment_sky`; everything else goes through
  [`NV_environment_sky_atmosphere`](#nv_environment_sky_atmosphere) beside it.
- **Viewer settings** — observer altitude. Where you stand is not a property of the world you are
  standing on, and no extension has a field for it.

All of them persist to the `.ini` regardless, because a scene cannot lose: loading one overwrites
the atmosphere outright, and that happens after the `.ini` is restored. Persistence therefore only
decides what you get when *no* scene carries an atmosphere.

Scattering is edited as a **strength and a tint** rather than three numbers near 0.01 — which is
also exactly how `OMI_environment_sky` stores it, so the panel and the file agree by construction.
Only the vector is stored. The panel splits it into the pair once and keeps the pair while you edit
-- re-deriving it every frame renormalised the tint to a peak of 1 and folded any greyer tint back
into the strength -- and splits it again whenever the vector changes underneath it.

### Atmosphere presets

`atmospherePresets()` in `src/sky_bruneton.cpp` is a table of whole atmospheres, and applying one
overwrites every field. The preset shown in the panel is *computed* from the current values rather
than remembered, so editing any slider drops it to Custom, and a scene that happens to carry
Earth's atmosphere reads as Earth. Resetting the environment applies the Earth preset rather than
copying fields one at a time — that is what stops the reset path drifting as the atmosphere grows.

Earth is upstream's apart from the aerosol. Upstream's Mie scattering is an optical depth of about
0.005 -- cleaner than the cleanest place on Earth -- and it showed: the sun outran the sky by 2-3x
at every elevation. Measured against a real clear day, direct-to-diffuse illuminance on a
horizontal surface:

| sun elevation | 60° | 45° | 30° | 15° |
|---|---|---|---|---|
| upstream's aerosol | 13.4 | 11.1 | 8.0 | 4.0 |
| here | 6.0 | 4.9 | 3.5 | 1.7 |
| measured, clear day | ~6–7 | ~5 | ~3 | ~1.3 |

The value used here is an optical depth of ~0.08, a clear continental day rather than a mountain
observatory. It barely changes how much light reaches the scene — total illuminance moves under 1%
— but it fixes the balance, and with it how contrasty and how blue the result is.

Mars and the alien world are plausible, not authoritative: Mars uses its real radius, solar
distance and scale height with a dust loading in the landers' range; the alien world exists to show
the parameters reach somewhere Earth cannot.

### What to set white balance to

The tonemapper's temperature names *the illuminant to neutralise* (its default, 6506 K, is the
identity). Measured CCT of what actually lights the scene, sun plus sky on a horizontal surface:

| sun elevation | 80° | 60° | 45° | 30° | 15° |
|---|---|---|---|---|---|
| sun only | 5341 K | 5246 K | 5096 K | 4818 K | 4078 K |
| sky only | 8837 K | 8712 K | 8501 K | 8112 K | 7315 K |
| **combined** | 5653 K | 5593 K | **5501 K** | 5342 K | 4957 K |

So **~5500 K** at the default 45° sun, and roughly 5000–5650 K across the day, which means one
setting stays close all day while a low sun still reads warm.

**D65 is not the target here.** 6504 K approximates *average* daylight with a large sky
contribution; a clear sky with the sun at 45° measures about 5500–6000 K, which is where this
lands. You approach D65 in open shade or under overcast, where the sky dominates — and the sky
alone is 8500 K here, if anything on the blue side.

The tonemapper's default stays at 6506 K, which is the identity. White balance is a photographic
decision, the same default has to serve HDR files with their own colour temperatures, and a
renderer that silently corrects its own output is harder to trust than one that does not. The
number above is the one to type when neutral is what you want.

## glTF: `OMI_environment_sky`

Load and save live in `src/gltf_environment_sky.*`. The parsed descriptor keeps the sky entry
**verbatim** alongside the typed fields, so keys this build does not model — a newer revision,
another vendor's exporter, a type not implemented here — survive a round trip instead of being
silently stripped. An unrecognized `type` is shown as a plain sky but saved as it was authored, until
the user picks a sky of their own (`SkyDescriptor::unknownType`).

The physical type needs one conversion. The extension states scattering as a **magnitude plus a
colour**, in m⁻¹; the model keeps a per-channel coefficient in km⁻¹. The split is on the largest
channel, which is exactly invertible — the property the round trip depends on — and reproduces the
extension's own default shape. Mie extinction is derived rather than read: the extension has no
absorption field, so a scene that sets Mie scattering keeps Earth's single-scattering albedo
instead of inheriting an extinction no longer related to it.

The extension defines no URI for its `panorama` type, so the image path is read from a vendor
sibling, `NV_environment_sky_panorama` — see
[developer.md § Panorama](developer.md#panorama-where-the-image-uri-lives).

### What is not implemented

Two properties are read and written back but never applied:

| Property | Behaviour |
|---|---|
| `ambientLightColor` | preserved, ignored |
| `ambientSkyContribution` | preserved, ignored |

They describe an ambient term separate from the sky, scaled against it. This renderer has no such
term: the scene is lit entirely by the environment the bake produces, plus the scene's own punctual
and emissive lights. There is nothing for these to scale, and a partial implementation — say,
multiplying the whole environment by `ambientSkyContribution` — would make the sky you see and the
sky that lights the scene disagree in a way the property does not actually ask for.

So they are deliberately not modelled. A file carrying them keeps them verbatim, the same way any
other unmodelled key survives, so authoring them in a tool that *does* implement them is safe.
`EnvironmentSky.UnsupportedAmbientTermsSurviveUntouched` pins that.

`rotation` *is* the renderer's orientation — `Settings::envRotation` stores the same quaternion,
tilt included, and every environment lookup goes through it (`SceneFrameInfo::envRotation`). So it
is applied exactly and written back exactly. There is no second rotation: the HDR image, every sky, and the Time of Day compass all
turn with this one. It is applied only when the file states it; a scene that omits `rotation` leaves
the viewer's current orientation alone rather than resetting it — the control is viewer state, and a
file that says nothing about orientation is not saying "identity".

The sun is world space (it is also a light), so loading an orientation does not move it — the
file's marked light already says where it is. The orientation is also where north is, so *editing*
it places the sun again from Time of Day on the turned compass (`GltfRenderer::onEnvironmentRotated`),
as the old North Offset did.

## Sharing presets

One sky per scene is the invariant. A `.sky.json` is how a sky gets to *another* scene, or to
somebody else.

The file is **exactly one entry of the OMI `skies[]` array** -- no wrapper, no version field, no
format of ours -- so anything that can read the extension can read a preset. That is not a
convenience: it is why there is one serializer. `gltf_environment_sky::toValue` / `fromValue` produce
and consume the entry, `write` / `parse` are the glTF document around them, and `src/sky_preset.*` is
the text layer and nothing else. A field added to the extension appears in presets the same day,
without anyone remembering to add it.

The format adds one key OMI has no place for: `sunRotation`, the sun's angle at capture. A sky and
the angle of its sun are one look, and a preset that restored the atmosphere while leaving the sun
where it was would be half the thing you saved. Absence is meaningful -- a file that omits it leaves
the sun alone rather than swinging it to a default nobody chose.

Keys this build does not model survive the trip, the same way they do through a glTF. A preset
written by a newer version, or by another vendor's exporter, comes back out intact.

Load and save from the Environment panel's **Preset** row, by dropping a `.sky.json` on the viewport,
or with `--loadSkyPreset` / `--saveSkyPreset` -- which also makes them reachable from a benchmark
sequence and from MCP, since all four come from one declaration.

A panorama preset names its image relative to the `.sky.json`, the same rule glTF external assets
follow ([external_assets.md](external_assets.md)), so a preset and its `.hdr` travel as a pair.

**Not undoable**, deliberately. Everything a preset changes is viewer state, and no environment
control in this renderer is on the undo stack; the exception is aiming a scene's marked sun light,
which is a scene edit and is recorded like any other transform.

## Cost

- **Per frame: next to nothing.** Every sky reaches the shaders as a lat-long image. That is the
  point of the design, and it is why the physical sky is *faster* than the analytic one it
  replaced. The one exception is what the path tracer's camera rays and mirror bounces see of the
  physical sky: that is evaluated per ray from the LUTs, so the horizon stays sharp at any bake
  resolution. Rough bounces and next-event estimation still read the image; see
  `evalPhysicalSkyBackground` in `shaders/gltf_pathtrace.slang`.
- **Per commit:** the bake, plus the alias table and PDF. The alias table is built on the CPU, but
  over a fixed coarse sampling grid rather than the bake's texels, so it costs about as much as the
  bake itself. See `nvvk::HdrIbl::updateFromGpuImage`.
- **Per atmosphere change:** the LUT precompute, on top of a commit. It is the largest single cost
  in the system and does not run at any other time.
- **Per primary hit, when aerial perspective is on:** a sixteen-step march through the segment
  between the eye and the surface, plus, in the path tracer, one shadow ray toward the sun. Measured on a 10 km terrain at 1920x1080: **5.6 ms**, against a
  715 ms frame — under 1% there, and the number to weigh if a froxel volume is ever added, since
  that is what a volume would replace.

Edits are previewed and committed on a settle rather than per frame — see
[developer.md § Environment Lighting](developer.md#environment-lighting). For the physical sky the
preview also rebuilds the tables, at the reduced quality described in
[Two quality tiers](#two-quality-tiers); ground albedo and sun angular radius are read directly by
the bake besides, so those two move even before the tables catch up.

Measured on an RTX PRO 4500 Blackwell at 1024×512 bake resolution, sun 5° above the horizon — the hardest
case, since that is where multiple scattering contributes most. Error is against the reference
tier, over the baked lighting field:

| Tier | Precompute | Mean error | Total energy |
|---|---|---|---|
| Reference (settle) | ~34 ms | — | — |
| Preview (drag) | ~6 ms | 8.9% | −8.6% |

Spending more on the preview buys little: doubling the density samples costs ~2.5 ms, doubling
the orders ~8.5 ms, and either still lands at −6%. The remaining gap is inherent to coarse tables, not to the choice between the two dials, which
is why the preview tier is the cheap one. Re-measure and compare the baked
`.hdr` — see [Verifying a change](#verifying-a-change).

## `NV_environment_sky_atmosphere`

OMI's `physical` type says what the air scatters, but nothing about the world it surrounds: there
is no field for the star's spectrum, the planet's size, how fast either gas thins with altitude,
how much the aerosol absorbs, or whether there is an ozone layer. Every one of those changes the
sky visibly, so without them a saved scene reopens as Earth whatever it was.

This block carries exactly those, as a sibling inside the sky entry's `extensions` — the same shape
`NV_environment_sky_panorama` already uses, and for the same reason: a reader that knows only OMI
still gets a coherent sky. It sees the scattering it understands and falls back to its own defaults
for the rest, rather than meeting keys the OMI schema does not allow.

Notes worth knowing:

- **Metres and m⁻¹**, matching OMI's own units so the two blocks agree inside one file. The model
  works in kilometres; the conversion happens at the file boundary. Converting back divides by
  1000 rather than scaling by 0.001 — the reciprocal is not representable, and scaling by it costs
  a few ulps per load, so a scene saved and reopened repeatedly would creep.
- **Written whole, or not at all.** The block is always emitted beside a physical sky, never
  partially. A partial block would make "the file said nothing" and "the file said Earth"
  indistinguishable on load, and that distinction is the point: a scene carrying only the OMI block
  has not said what its planet is, so loading it leaves the current atmosphere alone instead of
  forcing Earth onto it.
- **The two are disjoint**, so there is no precedence to resolve — the overlay is read after the
  OMI block purely for ordering, not to win.
- The sun's angular radius lives here rather than being treated as a viewer setting, because a
  preset has to survive a round trip: Mars' sun is smaller than Earth's, and a file that dropped
  that would reopen as Custom.

Round-tripping is exact, not merely close: a saved scene reloaded and re-saved reproduces every
value bit-for-bit, and stays there across repeated cycles.

### The sun is missing from anything that measures the image

The bake deliberately excludes the sun disk, which means **any quantity derived from the baked
image alone describes the sky and not the scene's dominant light**. For a daylit atmosphere the sky
is the smaller half by a wide margin -- at a 15 degree sun, the sky integrates to ~6 while the sun
delivers ~22 in the same units.

The firefly clamp is the case where that bit: it defaults to the environment's luminance integral,
which is exactly such a quantity. An `.hdr` file has its sun baked in and needs no correction; an
analytic sky with a sun of its own does. `GltfRenderer::defaultFireflyClamp` adds the sun back, in
the integral's own units -- a disk of radiance L covering solid angle W contributes max(L) x W to
that sum, which is its illuminance.

Two things about *which* sun, both learned the hard way:

- **The sun above the atmosphere, not the one currently shining.** The attenuated sun is the
  honest answer to "how much light is there now" and the wrong answer here: it goes to zero at
  dusk and takes the clamp with it. Measured, the attenuated version gave 0.34 at 2 degrees below
  the horizon, 0.03 at 5, and exactly 0 at 10 -- which the shader reads as *disabled*, so it
  flipped from crushing every sample to not clamping at all. The unattenuated value is a property
  of the star rather than the hour, so it holds steady through a time-of-day scrub while still
  following the atmosphere: Mars' sun is dimmer, and editing the solar irradiance moves it.
- **Calibrated once per environment, not per bake.** A commit happens every time the sun moves or
  a slider settles, and the clamp is a starting point rather than a derived quantity -- it has to
  accommodate emissive materials and punctual lights that the environment integral knows nothing
  about. Recomputing it on a gesture as ordinary as dragging the sun would throw away whatever the
  user had set. `GltfRenderer::calibrateFireflyClamp` is the single place that sets it.

Resulting range for Earth: ~27.6 with the sun down, ~39.4 at zenith.

Note what this does *not* fix. The clamp scales with the sun's *illuminance*, while a specular path
carries its *radiance*, and a half-degree disk at physical brightness has an enormous radiance
(~3e5 here) for a modest illuminance. Scenes with sharp speculars may still want it raised by hand
-- which is now something that sticks.

## Both renderers must agree

The rasterizer draws the sky through a compute pass and the path tracer through escaped rays, so
the two can only agree about *where* the sky is if they reconstruct the camera ray the same way.
They call the same function to do it — `getRay` in `shaders/camera_ray.h.slang`.

That is deliberate rather than tidy. The background pass once had its own copy, which normalised a
world-space point on the far plane instead of a direction from the eye: identical wherever the
camera sat at the origin, and wrong in proportion to how far away it was. A half-degree sun disk
landed about 70 pixels from where the path tracer drew it at 500 units out, while the smooth sky
gradient hid the same error completely.

A sharp feature is the test worth running: put the sun in frame, render both paths, and compare
where the disk lands.

## Verifying a change

`utils/sky_luminance_factors.py` recomputes the sky and sun spectral constants in
`shaders/sky_bruneton_io.h.slang` from upstream's own CIE and solar-spectrum tables (fetched from
GitHub, or `--upstream` for a local clone) and reports how far the shipped values are from them.
Run it if those constants are ever revisited.

**Reach outside the renderer at least once.** Every gate through phase 4b tested
self-consistency -- bit-exact round trips, bit-identical re-bakes -- and all of them passed
while three separate calibration defects were live, because none of them compared the sky to a
number from the physical world. Comparing the bake's direct-to-diffuse illuminance and colour
temperature against a measured clear day is what found and verified the aerosol and spectral
fixes.

**Pin every viewer-side input before comparing two runs.** Sun angle, observer altitude and bake
resolution all come from the `.ini` unless overridden, so two runs of "the same" command need not
be the same. The sun is the one to watch: `--todHour` and friends compute it, `--sunElevation` sets
it directly, and a scene carrying a marked sun light overrides both.

Two things to know when reading the output:

- The **horizon is a real discontinuity**, not a seam. At a low observer altitude the horizon is
  only a few kilometres away and the air barely attenuates the ground, so the line is genuinely
  sharp. It softens as the observer rises — which is the test that distinguishes it from a LUT
  artifact, since a LUT artifact would not care where the observer stands.
- A **seam** is coherent: it runs the length of a row or column. Compare the wrap column against
  any other adjacent pair; if they are the same magnitude, there is no seam.

## Extending: volumetric clouds

Clouds are out of scope, but the architecture leaves a seam for them. The background pass writes an
alpha channel that is currently constant and is reserved for a cloud transmittance term, and the
lighting field is already a baked image that a cloud pass could modulate before the alias table is
built. A cloud system would attach at those two points rather than inside the atmosphere model.

## References

- Eric Bruneton and Fabrice Neyret, *Precomputed Atmospheric Scattering*, EGSR 2008, and the
  author's reference implementation at
  <https://github.com/ebruneton/precomputed_atmospheric_scattering>. The port tracks the
  implementation, which differs from the paper in its LUT parameterisation.
- [`OMI_environment_sky`](https://github.com/omigroup/gltf-extensions/tree/main/extensions/2.0/OMI_environment_sky)
