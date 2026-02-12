#include <gtest/gtest.h>
#include "common/test_utils.hpp"
#include "nvutils/logger.hpp"

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  nvutils::Logger::getInstance().breakOnError(false);

  // Run tests
  int result = RUN_ALL_TESTS();

  // Cleanup
  gltf_test::TestResources::cleanupTempFiles();

  return result;
}
