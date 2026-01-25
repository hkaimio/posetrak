#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/camera.hpp"

using namespace posetrak;

TEST_CASE("Camera construction and accessors", "[camera]") {
    Camera cam(800.0, 800.0, 320.0, 240.0, 640, 480);

    REQUIRE(cam.fx() == 800.0);
    REQUIRE(cam.fy() == 800.0);
    REQUIRE(cam.cx() == 320.0);
    REQUIRE(cam.cy() == 240.0);
    REQUIRE(cam.width() == 640);
    REQUIRE(cam.height() == 480);

    REQUIRE(cam.position().isApprox(Eigen::Vector3d::Zero()));
    REQUIRE(cam.orientation().coeffs().isApprox(Eigen::Quaterniond::Identity().coeffs()));
}

TEST_CASE("Camera extrinsics", "[camera]") {
    Camera cam(800.0, 800.0, 320.0, 240.0, 640, 480);

    Eigen::Vector3d const pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond const quat(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()));

    cam.set_extrinsics(pos, quat);

    REQUIRE(cam.position().isApprox(pos));
    REQUIRE(cam.orientation().coeffs().isApprox(quat.normalized().coeffs()));
}

TEST_CASE("Camera projection without distortion", "[camera]") {
    Camera cam(800.0, 800.0, 320.0, 240.0, 640, 480);

    SECTION("Point at origin with camera at (0,0,2) looking at origin") {
        Eigen::Vector3d const pos(0.0, 0.0, 2.0);
        Eigen::Quaterniond const quat(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()));
        cam.set_extrinsics(pos, quat);

        Eigen::Vector3d const point(0.0, 0.0, 0.0);
        Eigen::Vector2d const pixel = cam.project(point);

        // Should project to principal point
        REQUIRE_THAT(pixel.x(), Catch::Matchers::WithinAbs(320.0, 1e-6));
        REQUIRE_THAT(pixel.y(), Catch::Matchers::WithinAbs(240.0, 1e-6));
    }

    SECTION("Point offset from center") {
        Eigen::Vector3d const pos(0.0, 0.0, 0.0);
        cam.set_extrinsics(pos, Eigen::Quaterniond::Identity());

        Eigen::Vector3d const point(1.0, 0.5, 2.0);
        Eigen::Vector2d const pixel = cam.project(point);

        // x_pixel = fx * (x/z) + cx = 800 * (1/2) + 320 = 720
        // y_pixel = fy * (y/z) + cy = 800 * (0.5/2) + 240 = 440
        REQUIRE_THAT(pixel.x(), Catch::Matchers::WithinAbs(720.0, 1e-6));
        REQUIRE_THAT(pixel.y(), Catch::Matchers::WithinAbs(440.0, 1e-6));
    }

    SECTION("Point behind camera returns invalid") {
        Eigen::Vector3d const pos(0.0, 0.0, 0.0);
        cam.set_extrinsics(pos, Eigen::Quaterniond::Identity());

        Eigen::Vector3d const point(0.0, 0.0, -1.0);
        Eigen::Vector2d const pixel = cam.project(point);

        REQUIRE(pixel.x() < 0.0);
        REQUIRE(pixel.y() < 0.0);
    }
}

TEST_CASE("Camera unprojection without distortion", "[camera]") {
    Camera cam(800.0, 800.0, 320.0, 240.0, 640, 480);
    cam.set_extrinsics(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());

    SECTION("Unproject ray from principal point") {
        Eigen::Vector2d const pixel(320.0, 240.0);
        Eigen::Vector3d const ray = cam.unproject_ray(pixel);

        // Ray should point along +Z axis
        REQUIRE_THAT(ray.x(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(ray.y(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(ray.z(), Catch::Matchers::WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(ray.norm(), Catch::Matchers::WithinAbs(1.0, 1e-6));
    }

    SECTION("Unproject with known depth") {
        Eigen::Vector2d const pixel(720.0, 440.0);
        double const depth = 2.0;
        Eigen::Vector3d const point = cam.unproject(pixel, depth);

        // x = (720 - 320) / 800 * 2 = 1.0
        // y = (440 - 240) / 800 * 2 = 0.5
        // z = 2.0
        REQUIRE_THAT(point.x(), Catch::Matchers::WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(point.y(), Catch::Matchers::WithinAbs(0.5, 1e-6));
        REQUIRE_THAT(point.z(), Catch::Matchers::WithinAbs(2.0, 1e-6));
    }

    SECTION("Project then unproject roundtrip") {
        Eigen::Vector3d const original(1.5, -0.8, 3.0);
        Eigen::Vector2d const pixel = cam.project(original);
        Eigen::Vector3d const reconstructed = cam.unproject(pixel, 3.0);

        REQUIRE(reconstructed.isApprox(original, 1e-6));
    }
}

TEST_CASE("Camera distortion", "[camera]") {
    SECTION("No distortion is identity") {
        Camera cam(800.0, 800.0, 320.0, 240.0, 640, 480);

        Eigen::Vector2d const point(0.5, 0.3);
        Eigen::Vector2d const distorted = cam.distort(point);

        REQUIRE(distorted.isApprox(point, 1e-10));
    }

    SECTION("Radial distortion (barrel)") {
        Camera cam(800.0, 800.0, 320.0, 240.0, 640, 480, -0.2, 0.0, 0.0);

        Eigen::Vector2d const point(0.5, 0.3);
        Eigen::Vector2d const distorted = cam.distort(point);

        // Negative k1 causes barrel distortion (points move outward)
        double const r2 = 0.5 * 0.5 + 0.3 * 0.3;
        double const expected_radial = 1.0 - 0.2 * r2;

        REQUIRE_THAT(distorted.x(), Catch::Matchers::WithinAbs(0.5 * expected_radial, 1e-6));
        REQUIRE_THAT(distorted.y(), Catch::Matchers::WithinAbs(0.3 * expected_radial, 1e-6));
    }

    SECTION("Tangential distortion") {
        Camera cam(800.0, 800.0, 320.0, 240.0, 640, 480, 0.0, 0.0, 0.0, 0.01, 0.02);

        Eigen::Vector2d const point(0.5, 0.3);
        Eigen::Vector2d const distorted = cam.distort(point);

        // Should have tangential distortion component
        REQUIRE(std::abs(distorted.x() - point.x()) > 1e-6);
        REQUIRE(std::abs(distorted.y() - point.y()) > 1e-6);
    }
}

TEST_CASE("Camera undistortion", "[camera]") {
    SECTION("Undistort inverts distortion") {
        Camera cam(800.0, 800.0, 320.0, 240.0, 640, 480, -0.1, 0.05, 0.0, 0.01, 0.02);

        Eigen::Vector2d const original(0.5, 0.3);
        Eigen::Vector2d const distorted = cam.distort(original);
        Eigen::Vector2d const undistorted = cam.undistort(distorted);

        REQUIRE(undistorted.isApprox(original, 1e-6));
    }

    SECTION("Multiple points roundtrip") {
        Camera cam(800.0, 800.0, 320.0, 240.0, 640, 480, 0.15, -0.08, 0.03);

        std::vector<Eigen::Vector2d> const points = {
            Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(0.5, 0.5), Eigen::Vector2d(-0.3, 0.4),
            Eigen::Vector2d(0.8, -0.2)};

        for (auto const& pt : points) {
            Eigen::Vector2d const distorted = cam.distort(pt);
            Eigen::Vector2d const undistorted = cam.undistort(distorted);
            REQUIRE(undistorted.isApprox(pt, 1e-6));
        }
    }
}

TEST_CASE("Camera projection with distortion", "[camera]") {
    Camera cam(800.0, 800.0, 320.0, 240.0, 640, 480, 0.1, -0.05, 0.0);
    cam.set_extrinsics(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());

    SECTION("Project and unproject with distortion") {
        Eigen::Vector3d const original(1.0, 0.5, 2.0);
        Eigen::Vector2d const pixel = cam.project(original);
        Eigen::Vector3d const reconstructed = cam.unproject(pixel, 2.0);

        REQUIRE(reconstructed.isApprox(original, 1e-5));
    }
}

TEST_CASE("Camera bounds checking", "[camera]") {
    Camera cam(800.0, 800.0, 320.0, 240.0, 640, 480);

    REQUIRE(cam.is_in_bounds(Eigen::Vector2d(0.0, 0.0)));
    REQUIRE(cam.is_in_bounds(Eigen::Vector2d(639.9, 479.9)));
    REQUIRE(cam.is_in_bounds(Eigen::Vector2d(320.0, 240.0)));

    REQUIRE(!cam.is_in_bounds(Eigen::Vector2d(-1.0, 240.0)));
    REQUIRE(!cam.is_in_bounds(Eigen::Vector2d(320.0, -1.0)));
    REQUIRE(!cam.is_in_bounds(Eigen::Vector2d(640.0, 240.0)));
    REQUIRE(!cam.is_in_bounds(Eigen::Vector2d(320.0, 480.0)));
}

TEST_CASE("Camera JSON serialization", "[camera]") {
    Camera cam(800.0, 800.0, 320.0, 240.0, 640, 480, 0.1, -0.05, 0.02, 0.01, 0.03);
    Eigen::Vector3d const pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond const quat(Eigen::AngleAxisd(M_PI / 6, Eigen::Vector3d::UnitZ()));
    cam.set_extrinsics(pos, quat);

    nlohmann::json const json = cam.to_json();
    Camera const cam2 = Camera::from_json(json);

    REQUIRE_THAT(cam2.fx(), Catch::Matchers::WithinAbs(cam.fx(), 1e-10));
    REQUIRE_THAT(cam2.fy(), Catch::Matchers::WithinAbs(cam.fy(), 1e-10));
    REQUIRE_THAT(cam2.cx(), Catch::Matchers::WithinAbs(cam.cx(), 1e-10));
    REQUIRE_THAT(cam2.cy(), Catch::Matchers::WithinAbs(cam.cy(), 1e-10));
    REQUIRE(cam2.width() == cam.width());
    REQUIRE(cam2.height() == cam.height());

    REQUIRE(cam2.position().isApprox(cam.position(), 1e-10));
    REQUIRE(cam2.orientation().coeffs().isApprox(cam.orientation().coeffs(), 1e-10));

    // Test projection consistency
    Eigen::Vector3d const test_point(2.0, 1.0, 5.0);
    REQUIRE(cam2.project(test_point).isApprox(cam.project(test_point), 1e-10));
}

TEST_CASE("Camera JSON without extrinsics", "[camera]") {
    Camera cam(800.0, 800.0, 320.0, 240.0, 640, 480);

    nlohmann::json json = cam.to_json();

    // Remove extrinsics to test optional loading
    json.erase("extrinsics");

    Camera const cam2 = Camera::from_json(json);

    REQUIRE(cam2.position().isApprox(Eigen::Vector3d::Zero()));
    REQUIRE(cam2.orientation().coeffs().isApprox(Eigen::Quaterniond::Identity().coeffs()));
}
