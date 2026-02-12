#include "test_utils.hpp"
#include <chrono>
#include <ctime>

namespace gltf_test {

std::filesystem::path TestResources::getResourcePath(const std::string& filename)
{
  // Try several locations
  std::vector<std::filesystem::path> searchPaths = {std::filesystem::current_path() / "resources",
                                                    std::filesystem::current_path() / ".." / "resources",
                                                    std::filesystem::current_path() / ".." / ".." / "resources"};

  for(const auto& base : searchPaths)
  {
    auto fullPath = base / filename;
    if(std::filesystem::exists(fullPath))
    {
      return fullPath;
    }
  }

  throw std::runtime_error("Test resource not found: " + filename);
}

std::filesystem::path TestResources::getTempPath(const std::string& filename)
{
  auto tempDir = std::filesystem::temp_directory_path() / "gltf_renderer_tests";
  std::filesystem::create_directories(tempDir);
  return tempDir / filename;
}

void TestResources::cleanupTempFiles()
{
  auto tempDir = std::filesystem::temp_directory_path() / "gltf_renderer_tests";
  if(std::filesystem::exists(tempDir))
  {
    std::filesystem::remove_all(tempDir);
  }
}

#ifdef GTEST_API_
void SceneTestFixture::SetUp()
{
  m_tempDir = std::filesystem::temp_directory_path() / ("gltf_test_" + std::to_string(std::time(nullptr)));
  std::filesystem::create_directories(m_tempDir);
}

void SceneTestFixture::TearDown()
{
  if(std::filesystem::exists(m_tempDir))
  {
    std::filesystem::remove_all(m_tempDir);
  }
}
#endif

bool VisualValidator::compareScreenshots(const std::filesystem::path& expected, const std::filesystem::path& actual, float tolerance)
{
  // TODO: Implement visual comparison using image library
  // For now, just check that files exist
  return std::filesystem::exists(expected) && std::filesystem::exists(actual);
}

void PerformanceTimer::start()
{
  m_start = std::chrono::high_resolution_clock::now();
}

double PerformanceTimer::stop()
{
  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
  return duration.count() / 1000.0;  // Convert to milliseconds
}

}  // namespace gltf_test
