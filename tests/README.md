# Testing Framework

This directory contains the unit tests and benchmarks for the glTF renderer.

## Building Tests

Tests are **disabled by default** to keep the build clean. To enable them:

```bash
# Configure with testing enabled
cmake -B build -DBUILD_TESTING=ON

# Build everything including tests
cmake --build build

# Or build just the tests
cmake --build build --target vk_gltf_renderer_tests
cmake --build build --target vk_gltf_renderer_benchmarks
```

## Running Tests

### Via CTest (recommended)
```bash
cd build
ctest -C Release --output-on-failure
```

> CTest runs the **unit tests** only (`vk_gltf_renderer_tests`, registered via
> `gtest_discover_tests`). The Google Benchmark microbenchmarks
> (`vk_gltf_renderer_benchmarks`) are not registered with CTest — run that executable directly
> (see [Running Benchmarks](#running-benchmarks)).

### Direct Execution

> Paths below assume a multi-config generator (Visual Studio). On single-config generators
> (Ninja/Makefiles, typical on Linux) the binaries are in `_bin/<Config>/` without the `.exe`
> suffix, e.g. `_bin/Release/vk_gltf_renderer_tests`.

```bash
# Run all tests
_bin/Release/vk_gltf_renderer_tests.exe

# Run with color output
_bin/Release/vk_gltf_renderer_tests.exe --gtest_color=yes

# Run specific test
_bin/Release/vk_gltf_renderer_tests.exe --gtest_filter=BasicTests.RoundTrip

# Run with verbose output
_bin/Release/vk_gltf_renderer_tests.exe --gtest_color=yes --gtest_print_time=1
```

### Interactivity Tests

`KHR_interactivity` behavior-graph engine tests (graph parsing/validation, flow scheduling,
node evaluation — see [docs/interactivity.md](../docs/interactivity.md)) are self-contained:
they build graphs directly as `tinygltf::Value` trees in-process, no GPU or scene file needed.

```bash
# Via ctest
ctest -C Release -R Interactivity --output-on-failure

# Directly
_bin/Release/vk_gltf_renderer_tests.exe --gtest_filter="Interactivity*"
```

The `ready` gate waits for a render pass to actually complete (`Resources::renderPassCount`, a
monotonic counter unaffected by dirty-flag resets) rather than the accumulation frame counter -
a scene whose graph writes a pointer every tick (e.g. `BowShooting.glb`'s permanent idle sway
animation) resets the accumulation counter every single frame, which used to make `ready` hang
forever.

### UI Scenario Scripts (`tests/ui/`) — currently dormant

Hover/select/UI interaction used to be scriptable end-to-end via a scripted UI-automation
integration, driving real ImGui widgets with `click`/`open`/`close`/`check`/`visible` and
screenshot capture. That integration has been pulled out of the app for now — the glue code and
its CLI parameters are gone — and is tracked as future work to reinstate. The scenario scripts
below remain in `tests/ui/` as reference for what the coverage was and should be again; they are
not runnable against the current build.

| Script | Verifies | Scene |
|---|---|---|
| `interactivity_panel.txt` | KHR_interactivity Interactivity window: Play/Pause/Reset, Variables/Send Event/Log sub-sections, Reset actually restarting the graph | `Tests/Interactivity/event/send_and_receive/glTF-Binary/send_and_receive.glb` (Khronos `glTF-Test-Assets-Interactivity`) |
| `animation_playback.txt` | `animation/start`/`stop`/`stopAt` clip playback driven end-to-end (real conformance scene's self-verifying `debug/log` checks), combined with the Interactivity window's Play/Pause/Reset | `Tests/Interactivity/animation/{start,stop,stopAt}/glTF-Binary/*.glb` (Khronos `glTF-Test-Assets-Interactivity`) |
| `select_ray_info.txt` | `event/onSelect`'s `selectionPoint`/`selectionRayOrigin` output sockets via a simulated ray-pick click (`pickrendernode`) - regression test for the fix where a real click's ray data never reached `notifyNodeSelected()` | `Tests/Interactivity/UserInteractions/eventOnSelect/glTF-Binary/eventOnSelect.glb` (Khronos `glTF-Test-Assets-Interactivity`) |
| `calculator.txt` | Clicking a digit button fires `event/onSelect` and correctly re-renders its digit-strip display - regression test for the fix where a `pointer/set`-only write to a texture-transform's `offset` dropped the baked-in `scale` | `Models/Calculator/glTF-Binary/Calculator.glb` (Khronos `glTF-Test-Assets-Interactivity`) |
| `bow_shooting.txt` | Aim -> draw -> release via three clicks on the same handle node, verifying the arrow actually launches - regression test for the fix where `pointer/set` writing a bool to any non-`/visible` path (e.g. `KHR_node_selectability/selectable`) silently failed and killed the whole shoot chain | `Models/BowShooting/glTF-Binary/BowShooting.glb` (Khronos `glTF-Test-Assets-Interactivity`) |
| `reclick_selection_during_play.txt` | Three consecutive clicks on the same node all re-fire `event/onSelect` while a graph is playing, instead of every other click being absorbed as a deselect - regression test for the editor's normal click-to-deselect toggle making click-driven interactivity feel like it needs "two clicks" | `Models/BowShooting/glTF-Binary/BowShooting.glb` (Khronos `glTF-Test-Assets-Interactivity`) |
| `editor_operations.txt` | Scene Browser Elements tab + Inspector: category tabs, Materials add/duplicate/rename/delete/undo/redo, material value edits, Nodes add/visibility/delete, sort/filter, Textures/Samplers/Images editing, primitive split/merge | A texture-rich scene, e.g. `DamagedHelmet.gltf` |

## Running Benchmarks

```bash
# Run all benchmarks
_bin/Release/vk_gltf_renderer_benchmarks.exe

# Run specific benchmark (substring match on the registered name, e.g. BM_SceneLoad_Simple)
_bin/Release/vk_gltf_renderer_benchmarks.exe --benchmark_filter=BM_SceneLoad

# Run with repetitions for statistical accuracy
_bin/Release/vk_gltf_renderer_benchmarks.exe --benchmark_repetitions=10
```

## Test Structure

```
tests/
├── CMakeLists.txt              # Test build configuration (authoritative list of test sources)
├── README.md                   # This file
├── test_main.cpp               # Test entry point
├── benchmark_main.cpp          # Benchmark entry point
├── test_basic.cpp              # Basic scene loading
├── test_roundtrip.cpp          # Load → save → reload fidelity
├── test_features_preserved.cpp # Features/extensions survive round-trip
├── test_model_primary.cpp      # Primary-model selection
├── test_basic_editing.cpp      # Node add/delete/duplicate editing
├── test_index_remapping_basic.cpp / _advanced.cpp # Index remapping after edits
├── test_dirty_flags_and_render_nodes.cpp # Dirty-flag → render-node sync
├── test_node_hierarchy_operations.cpp    # Reparent / hierarchy ops
├── test_children_order_preservation.cpp  # Child ordering stability
├── test_animation_and_validation.cpp     # Animation + scene validation
├── test_animation_update.cpp / test_compute_animation.cpp # CPU / GPU animation
├── test_error_paths.cpp        # Error handling
├── test_material_cache.cpp     # Material cache
├── test_extensions_metadata.cpp # Extension metadata
├── test_primitives.cpp         # Procedural primitives
├── test_interactivity_engine.cpp # KHR_interactivity graph parsing, flow scheduling, node eval
└── common/
    ├── test_utils.hpp          # Test utilities header
    └── test_utils.cpp          # Test utilities implementation
```

> The list above is a snapshot; `tests/CMakeLists.txt` is the source of truth for
> which test files are built.

## Adding New Tests

### Unit Tests

1. Create a new test file (e.g., `test_mynewfeature.cpp`)
2. Add to `CMakeLists.txt`:
   ```cmake
   set(TEST_SOURCES
       ${TEST_COMMON_SOURCES}
       test_basic.cpp
       test_mynewfeature.cpp  # Add here
   )
   ```
3. Write tests using GoogleTest:
   ```cpp
   #include <gtest/gtest.h>
   #include "common/test_utils.hpp"
   
   TEST(MyFeature, BasicTest) {
       ASSERT_TRUE(true);
   }
   ```

### Benchmarks

Add to `benchmark_main.cpp`:
```cpp
static void BM_MyOperation(benchmark::State& state) {
    for (auto _ : state) {
        // Your code to benchmark
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MyOperation);
```

## Visual Studio Organization

Test dependencies (GoogleTest, Benchmark) are organized into the `External/` folder in the Solution Explorer to keep the main project clean.

## Dependencies

- **GoogleTest v1.14.0** - Unit testing framework
- **Google Benchmark v1.8.3** - Performance benchmarking

Both are automatically downloaded via CMake FetchContent when `BUILD_TESTING=ON`.

## CI/CD Integration

For continuous integration, add to your workflow:

```yaml
- name: Configure with testing
  run: cmake -B build -DBUILD_TESTING=ON

- name: Build tests
  run: cmake --build build --config Release

- name: Run tests
  run: ctest --test-dir build -C Release --output-on-failure
```
