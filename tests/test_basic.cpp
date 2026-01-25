// Basic test to verify testing framework is working
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Testing framework is operational", "[basic]") {
    REQUIRE(1 + 1 == 2);
}

TEST_CASE("Basic arithmetic", "[basic]") {
    REQUIRE(2 * 3 == 6);
    REQUIRE(10 / 2 == 5);
}
