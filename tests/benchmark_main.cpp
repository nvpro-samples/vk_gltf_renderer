#include <benchmark/benchmark.h>
#include <gltf_scene.hpp>
#include "common/test_utils.hpp"

// Benchmark scene loading
static void BM_SceneLoad_Simple(benchmark::State& state)
{
  try
  {
    auto path = gltf_test::TestResources::getResourcePath("cube.gltf");
    for(auto _ : state)
    {
      nvvkgltf::Scene scene;
      if(!scene.load(path))
        state.SkipWithError("load failed");
      benchmark::DoNotOptimize(scene.getRenderNodes().size());
    }
  }
  catch(const std::runtime_error& e)
  {
    state.SkipWithError(e.what());
  }
}
BENCHMARK(BM_SceneLoad_Simple);

// Benchmark complex scene loading
static void BM_SceneLoad_Complex(benchmark::State& state)
{
  try
  {
    auto path = gltf_test::TestResources::getResourcePath("shader_ball.gltf");
    for(auto _ : state)
    {
      nvvkgltf::Scene scene;
      if(!scene.load(path))
        state.SkipWithError("load failed");
      benchmark::DoNotOptimize(scene.getRenderNodes().size());
    }
  }
  catch(const std::runtime_error& e)
  {
    state.SkipWithError(e.what());
  }
}
BENCHMARK(BM_SceneLoad_Complex);

// Benchmark scene saving
static void BM_SceneSave(benchmark::State& state)
{
  try
  {
    // Load once
    auto            loadPath = gltf_test::TestResources::getResourcePath("shader_ball.gltf");
    nvvkgltf::Scene scene;
    if(!scene.load(loadPath))
      state.SkipWithError("load failed");

    auto savePath = gltf_test::TestResources::getTempPath("benchmark_save.gltf");

    for(auto _ : state)
    {
      bool saved = scene.save(savePath);
      benchmark::DoNotOptimize(saved);
    }

    // Cleanup
    if(std::filesystem::exists(savePath))
    {
      std::filesystem::remove(savePath);
    }
  }
  catch(const std::runtime_error& e)
  {
    state.SkipWithError(e.what());
  }
}
BENCHMARK(BM_SceneSave);

// Benchmark round-trip (load + save + load)
static void BM_SceneRoundTrip(benchmark::State& state)
{
  try
  {
    auto loadPath = gltf_test::TestResources::getResourcePath("shader_ball.gltf");
    auto savePath = gltf_test::TestResources::getTempPath("benchmark_roundtrip.gltf");

    for(auto _ : state)
    {
      // Load original
      nvvkgltf::Scene scene1;
      if(!scene1.load(loadPath))
        state.SkipWithError("load failed");

      // Save
      if(!scene1.save(savePath))
        state.SkipWithError("save failed");

      // Load saved
      nvvkgltf::Scene scene2;
      if(!scene2.load(savePath))
        state.SkipWithError("load failed");

      benchmark::DoNotOptimize(scene2.getRenderNodes().size());
    }

    // Cleanup
    if(std::filesystem::exists(savePath))
    {
      std::filesystem::remove(savePath);
    }
  }
  catch(const std::runtime_error& e)
  {
    state.SkipWithError(e.what());
  }
}
BENCHMARK(BM_SceneRoundTrip);

// Benchmark node matrix updates
static void BM_UpdateNodeWorldMatrices(benchmark::State& state)
{
  try
  {
    auto            path = gltf_test::TestResources::getResourcePath("shader_ball.gltf");
    nvvkgltf::Scene scene;
    if(!scene.load(path))
      state.SkipWithError("load failed");

    for(auto _ : state)
    {
      scene.updateNodeWorldMatrices();
      benchmark::DoNotOptimize(scene.getRenderNodes().size());
    }
  }
  catch(const std::runtime_error& e)
  {
    state.SkipWithError(e.what());
  }
}
BENCHMARK(BM_UpdateNodeWorldMatrices);

BENCHMARK_MAIN();
