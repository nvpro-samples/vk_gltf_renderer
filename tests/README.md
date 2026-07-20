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
