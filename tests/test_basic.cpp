#include <gtest/gtest.h>
#include "common/test_utils.hpp"
#include <nvvkgltf/scene.hpp>

using namespace gltf_test;

// Test that we can instantiate a Scene object
TEST(BasicTests, SceneConstruction)
{
  nvvkgltf::Scene scene;
  EXPECT_FALSE(scene.valid()) << "New scene should not be valid before loading";
}

// Test loading a simple scene
TEST(BasicTests, CanLoadScene)
{
  nvvkgltf::Scene scene;

  try
  {
    auto path   = TestResources::getResourcePath("Box.glb");
    bool loaded = scene.load(path);

    ASSERT_TRUE(loaded) << "Failed to load Box.glb";
    ASSERT_TRUE(scene.valid()) << "Scene should be valid after successful load";
    ASSERT_GT(scene.getRenderNodes().size(), 0) << "Should have render nodes";
  }
  catch(const std::runtime_error& e)
  {
    GTEST_SKIP() << "Test resource not found: " << e.what();
  }
}

// Test loading shader ball (more complex scene)
TEST(BasicTests, CanLoadComplexScene)
{
  nvvkgltf::Scene scene;

  try
  {
    auto path   = TestResources::getResourcePath("shader_ball.gltf");
    bool loaded = scene.load(path);

    ASSERT_TRUE(loaded) << "Failed to load shader_ball.gltf";
    ASSERT_TRUE(scene.valid()) << "Scene should be valid after successful load";

    // Verify we have nodes and data
    ASSERT_GT(scene.getModel().nodes.size(), 0) << "Should have nodes in model";
    ASSERT_GT(scene.getRenderNodes().size(), 0) << "Should have render nodes";
  }
  catch(const std::runtime_error& e)
  {
    GTEST_SKIP() << "Test resource not found: " << e.what();
  }
}

// Test round-trip: load, save, reload
TEST(BasicTests, RoundTrip)
{
  nvvkgltf::Scene scene;

  try
  {
    auto path = TestResources::getResourcePath("shader_ball.gltf");
    ASSERT_TRUE(scene.load(path)) << "Failed to load original scene";

    size_t origNodeCount       = scene.getModel().nodes.size();
    size_t origRenderNodeCount = scene.getRenderNodes().size();

    // Save to temporary file
    auto tempFile = TestResources::getTempPath("roundtrip_test.gltf");
    ASSERT_TRUE(scene.save(tempFile)) << "Failed to save scene";
    ASSERT_TRUE(std::filesystem::exists(tempFile)) << "Saved file should exist";

    // Reload from saved file
    nvvkgltf::Scene scene2;
    ASSERT_TRUE(scene2.load(tempFile)) << "Failed to reload saved scene";
    ASSERT_TRUE(scene2.valid()) << "Reloaded scene should be valid";

    // Verify structure preserved
    EXPECT_EQ(scene2.getModel().nodes.size(), origNodeCount) << "Node count should match after round-trip";
    EXPECT_EQ(scene2.getRenderNodes().size(), origRenderNodeCount) << "Render node count should match after round-trip";
  }
  catch(const std::runtime_error& e)
  {
    GTEST_SKIP() << "Test resource not found: " << e.what();
  }
}

// Test that loading invalid file fails gracefully
TEST(BasicTests, LoadInvalidFileFails)
{
  nvvkgltf::Scene scene;

  auto fakePath = std::filesystem::path("nonexistent_file_12345.gltf");
  bool loaded   = scene.load(fakePath);

  EXPECT_FALSE(loaded) << "Loading nonexistent file should fail";
  EXPECT_FALSE(scene.valid()) << "Scene should not be valid after failed load";
}

// Test performance of scene loading
TEST(BasicTests, LoadPerformance)
{
  nvvkgltf::Scene scene;

  try
  {
    auto path = TestResources::getResourcePath("Box.glb");

    PerformanceTimer timer;
    timer.start();

    bool loaded = scene.load(path);

    double elapsed = timer.stop();

    ASSERT_TRUE(loaded) << "Failed to load scene";

    // Simple scene should load quickly (less than 1 second)
    EXPECT_LT(elapsed, 1000.0) << "Simple scene took too long to load: " << elapsed << " ms";

    std::cout << "Scene load time: " << elapsed << " ms" << std::endl;
  }
  catch(const std::runtime_error& e)
  {
    GTEST_SKIP() << "Test resource not found: " << e.what();
  }
}
