#pragma once
#include <string>
#include <filesystem>
#include <chrono>

#ifdef GTEST_API_
#include <gtest/gtest.h>
#endif

namespace gltf_test {

// Test resource management
class TestResources
{
public:
  static std::filesystem::path getResourcePath(const std::string& filename);
  static std::filesystem::path getTempPath(const std::string& filename);
  static void                  cleanupTempFiles();
};

#ifdef GTEST_API_
// Test fixture for scene tests (only available when gtest is included)
class SceneTestFixture : public ::testing::Test
{
protected:
  void SetUp() override;
  void TearDown() override;

  std::filesystem::path m_tempDir;
};
#endif

// Visual comparison utilities (future)
class VisualValidator
{
public:
  static bool compareScreenshots(const std::filesystem::path& expected, const std::filesystem::path& actual, float tolerance = 0.01f);
};

// Performance measurement
class PerformanceTimer
{
public:
  void   start();
  double stop();  // Returns milliseconds
private:
  std::chrono::high_resolution_clock::time_point m_start;
};

}  // namespace gltf_test
