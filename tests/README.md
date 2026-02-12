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

### Direct Execution
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

# Run specific benchmark
_bin/Release/vk_gltf_renderer_benchmarks.exe --benchmark_filter=SceneLoad

# Run with repetitions for statistical accuracy
_bin/Release/vk_gltf_renderer_benchmarks.exe --benchmark_repetitions=10
```

## Test Structure

```
tests/
├── CMakeLists.txt              # Test build configuration
├── README.md                   # This file
├── test_main.cpp               # Test entry point
├── test_basic.cpp              # Basic scene loading tests
├── benchmark_main.cpp          # Benchmark entry point
└── common/
    ├── test_utils.hpp          # Test utilities header
    └── test_utils.cpp          # Test utilities implementation
```

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
