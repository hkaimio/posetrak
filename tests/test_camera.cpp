#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/camera.hpp"

using namespace posetrak;

// Helper functions to create test objects
Intrinsics make_test_intrinsics(double fx = 800.0, double fy = 800.0, double cx = 320.0,
                                double cy = 240.0, int width = 640, int height = 480,
                                std::vector<double> distortion = {}) {
    Intrinsics intr;
    intr.fx = fx;
    intr.fy = fy;
    intr.cx = cx;
    intr.cy = cy;
    intr.width = width;
    intr.height = height;
    intr.model = Intrinsics::DistortionModel::BrownConrady;
    intr.distortion_coeffs = distortion.empty() ? std::vector<double>{0, 0, 0, 0, 0} : distortion;
    return intr;
}

Extrinsics make_test_extrinsics(Eigen::Vector3d const& pos = Eigen::Vector3d::Zero(),
                                Eigen::Quaterniond const& quat = Eigen::Quaterniond::Identity()) {
    Extrinsics extr;
    extr.position = pos;
    extr.orientation = quat.normalized();
    return extr;
}

Camera make_test_camera(std::string const& name = "test_camera",
                        Intrinsics const& intr = make_test_intrinsics(),
                        Extrinsics const& extr = make_test_extrinsics(), double fps = 30.0,
                        uint32_t start_frame = 0) {
    return Camera(0, name, intr, extr, fps, start_frame);  // Use default ID of 0
}

TEST_CASE("Camera construction and accessors", "[camera]") {
    auto const intr = make_test_intrinsics();
    auto const extr = make_test_extrinsics();
    Camera const cam = make_test_camera("cam1", intr, extr, 60.0, 100);

    REQUIRE(cam.name() == "cam1");
    REQUIRE(cam.intrinsics().fx == 800.0);
    REQUIRE(cam.intrinsics().fy == 800.0);
    REQUIRE(cam.intrinsics().cx == 320.0);
    REQUIRE(cam.intrinsics().cy == 240.0);
    REQUIRE(cam.intrinsics().width == 640);
    REQUIRE(cam.intrinsics().height == 480);
    REQUIRE(cam.fps() == 60.0);
    REQUIRE(cam.start_frame() == 100);

    REQUIRE(cam.position().isApprox(Eigen::Vector3d::Zero()));
    REQUIRE(cam.orientation().coeffs().isApprox(Eigen::Quaterniond::Identity().coeffs()));
}

TEST_CASE("Camera extrinsics", "[camera]") {
    Eigen::Vector3d const pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond const quat(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()));

    auto const extr = make_test_extrinsics(pos, quat);
    Camera const cam = make_test_camera("cam1", make_test_intrinsics(), extr);

    REQUIRE(cam.position().isApprox(pos));
    REQUIRE(cam.orientation().coeffs().isApprox(quat.normalized().coeffs()));
}

TEST_CASE("Camera projection without distortion", "[camera]") {
    Camera cam = make_test_camera();

    SECTION("Point at origin with camera at (0,0,2) looking at origin") {
        Eigen::Vector3d const pos(0.0, 0.0, 2.0);
        Eigen::Quaterniond const quat(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()));

        auto const extr = make_test_extrinsics(pos, quat);
        cam = make_test_camera("cam1", make_test_intrinsics(), extr);

        Eigen::Vector3d const point(0.0, 0.0, 0.0);
        auto pixel_opt = cam.project(point);
        REQUIRE(pixel_opt.has_value());
        Eigen::Vector2d const pixel = *pixel_opt;

        // Should project to principal point
        REQUIRE_THAT(pixel.x(), Catch::Matchers::WithinAbs(320.0, 1e-6));
        REQUIRE_THAT(pixel.y(), Catch::Matchers::WithinAbs(240.0, 1e-6));
    }

    SECTION("Point offset from center") {
        // Use z=5 so the projected pixel stays within the 640×480 image.
        Eigen::Vector3d const point(1.0, 0.5, 5.0);
        auto pixel_opt = cam.project(point);
        REQUIRE(pixel_opt.has_value());
        Eigen::Vector2d const pixel = *pixel_opt;

        // x_pixel = fx * (x/z) + cx = 800 * (1/5) + 320 = 480
        // y_pixel = fy * (y/z) + cy = 800 * (0.5/5) + 240 = 320
        REQUIRE_THAT(pixel.x(), Catch::Matchers::WithinAbs(480.0, 1e-6));
        REQUIRE_THAT(pixel.y(), Catch::Matchers::WithinAbs(320.0, 1e-6));
    }

    SECTION("Point behind camera returns invalid") {
        Eigen::Vector3d const point(0.0, 0.0, -1.0);
        auto pixel_opt = cam.project(point);

        REQUIRE(!pixel_opt.has_value());
    }
}

TEST_CASE("Camera unprojection without distortion", "[camera]") {
    Camera const cam = make_test_camera();

    SECTION("Unproject ray from principal point") {
        Eigen::Vector2d const pixel(320.0, 240.0);
        Eigen::Vector3d const ray = cam.unproject_ray(pixel);

        // Ray should point in +Z direction
        REQUIRE_THAT(ray.x(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(ray.y(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(ray.z(), Catch::Matchers::WithinAbs(1.0, 1e-6));
    }

    SECTION("Unproject with depth") {
        Eigen::Vector2d const pixel(720.0, 440.0);
        double const depth = 2.0;
        Eigen::Vector3d const point = cam.unproject(pixel, depth);

        // x = ((720 - 320) / 800) * 2 = 1.0
        // y = ((440 - 240) / 800) * 2 = 0.5
        // z = 2.0
        REQUIRE_THAT(point.x(), Catch::Matchers::WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(point.y(), Catch::Matchers::WithinAbs(0.5, 1e-6));
        REQUIRE_THAT(point.z(), Catch::Matchers::WithinAbs(2.0, 1e-6));
    }

    SECTION("Project-unproject roundtrip") {
        // Use z=5 so the projected pixel stays within the 640×480 image.
        Eigen::Vector3d const original(1.0, 0.5, 5.0);
        auto pixel_opt = cam.project(original);
        REQUIRE(pixel_opt.has_value());
        Eigen::Vector2d const pixel = *pixel_opt;
        Eigen::Vector3d const unprojected = cam.unproject(pixel, original.z());

        REQUIRE(unprojected.isApprox(original, 1e-6));
    }
}

TEST_CASE("Camera distortion", "[camera]") {
    SECTION("No distortion at principal point") {
        Camera const cam = make_test_camera();
        Eigen::Vector2d const point_norm(0.0, 0.0);
        Eigen::Vector2d const distorted = cam.distort(point_norm);

        REQUIRE_THAT(distorted.x(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(distorted.y(), Catch::Matchers::WithinAbs(0.0, 1e-6));
    }

    SECTION("Radial distortion (barrel)") {
        auto const intr =
            make_test_intrinsics(800.0, 800.0, 320.0, 240.0, 640, 480, {-0.2, 0, 0, 0, 0});
        Camera const cam = make_test_camera("cam1", intr);

        Eigen::Vector2d const point_norm(0.5, 0.5);
        Eigen::Vector2d const distorted = cam.distort(point_norm);

        // With negative k1, points move inward (barrel distortion)
        double const r2 = 0.5 * 0.5 + 0.5 * 0.5;
        double const expected_radial = 1.0 - 0.2 * r2;

        REQUIRE_THAT(distorted.x(), Catch::Matchers::WithinAbs(0.5 * expected_radial, 1e-6));
        REQUIRE_THAT(distorted.y(), Catch::Matchers::WithinAbs(0.5 * expected_radial, 1e-6));
    }

    SECTION("Tangential distortion") {
        auto const intr =
            make_test_intrinsics(800.0, 800.0, 320.0, 240.0, 640, 480, {0, 0, 0.01, 0.02, 0});
        Camera const cam = make_test_camera("cam1", intr);

        Eigen::Vector2d const point_norm(0.5, 0.5);
        Eigen::Vector2d const distorted = cam.distort(point_norm);

        // Tangential should add asymmetric offset
        REQUIRE(distorted.x() != point_norm.x());
        REQUIRE(distorted.y() != point_norm.y());
    }

    SECTION("Combined radial and tangential distortion") {
        auto const intr =
            make_test_intrinsics(800.0, 800.0, 320.0, 240.0, 640, 480, {-0.1, 0.05, 0.01, 0.02, 0});
        Camera const cam = make_test_camera("cam1", intr);

        Eigen::Vector2d const point_norm(0.3, 0.4);
        Eigen::Vector2d const distorted = cam.distort(point_norm);

        // Just verify distortion is applied (complex formula)
        REQUIRE(distorted.x() != point_norm.x());
        REQUIRE(distorted.y() != point_norm.y());
    }
}

TEST_CASE("Camera distortion rational model", "[camera]") {
    SECTION("Rational model with k4/k5/k6 matches expected formula") {
        // Coefficients: k1,k2,p1,p2,k3,k4,k5,k6
        std::vector<double> const coeffs = {0.3, -0.1, 0.0, 0.0, 0.05, 0.2, -0.05, 0.01};
        auto const intr = make_test_intrinsics(800.0, 800.0, 320.0, 240.0, 640, 480, coeffs);
        Camera const cam = make_test_camera("cam1", intr);

        Eigen::Vector2d const p(0.3, 0.2);
        Eigen::Vector2d const got = cam.distort(p);

        double const r2 = 0.3 * 0.3 + 0.2 * 0.2;
        double const r4 = r2 * r2;
        double const r6 = r4 * r2;
        double const numer = 1.0 + 0.3 * r2 + (-0.1) * r4 + 0.05 * r6;
        double const denom = 1.0 + 0.2 * r2 + (-0.05) * r4 + 0.01 * r6;
        double const radial = numer / denom;
        double const expected_x = p.x() * radial;
        double const expected_y = p.y() * radial;

        REQUIRE_THAT(got.x(), Catch::Matchers::WithinAbs(expected_x, 1e-10));
        REQUIRE_THAT(got.y(), Catch::Matchers::WithinAbs(expected_y, 1e-10));
    }

    SECTION("Rational model degenerates to polynomial when k4=k5=k6=0") {
        std::vector<double> const poly = {0.3, -0.1, 0.01, 0.02, 0.05};
        std::vector<double> const ratio = {0.3, -0.1, 0.01, 0.02, 0.05, 0.0, 0.0, 0.0};
        auto const intr_poly = make_test_intrinsics(800.0, 800.0, 320.0, 240.0, 640, 480, poly);
        auto const intr_ratio = make_test_intrinsics(800.0, 800.0, 320.0, 240.0, 640, 480, ratio);
        Camera const cam_poly = make_test_camera("p", intr_poly);
        Camera const cam_ratio = make_test_camera("r", intr_ratio);

        Eigen::Vector2d const p(0.3, 0.2);
        REQUIRE(cam_poly.distort(p).isApprox(cam_ratio.distort(p), 1e-12));
    }

    SECTION("Realistic GoPro 11 mini rational coefficients — distort/undistort roundtrip") {
        // Coefficients from actual camera calibration (rational model, nearly-cancelling terms)
        std::vector<double> const coeffs = {-2.49055232, -0.52732380, 1.96701588e-3, 1.63309190e-3,
                                            4.56096122,  -2.48739746, -0.53610768,   4.56835348};
        // K_orig: fx=1999.89, cx=1928.68, cy=1093.89  K_new: fx=2002.34, cx=1938.60, cy=1100.48
        auto intr = make_test_intrinsics(2002.34, 2005.89, 1938.60, 1100.48, 3840, 2160, coeffs);
        Camera cam = make_test_camera("gopro", intr);
        // Set K_orig so undistort() normalises with the right matrix
        Eigen::Matrix3d K_orig;
        K_orig << 1999.89, 0.0, 1928.68, 0.0, 2000.72, 1093.89, 0.0, 0.0, 1.0;
        cam.set_K_original(K_orig);

        // A pixel in the distorted frame (leg region from real data)
        Eigen::Vector2d const distorted_px(2165.2, 1424.8);

        // Undistort then distort back must recover the original pixel
        Eigen::Vector2d const undistorted_px = cam.undistort(distorted_px);
        // Re-distort: normalise with K_new, apply distortion, apply K_orig
        double const x_n = (undistorted_px.x() - 1938.60) / 2002.34;
        double const y_n = (undistorted_px.y() - 1100.48) / 2005.89;
        Eigen::Vector2d const dist_norm = cam.distort(Eigen::Vector2d(x_n, y_n));
        Eigen::Vector2d const recovered(1999.89 * dist_norm.x() + 1928.68,
                                        2000.72 * dist_norm.y() + 1093.89);

        // Should recover the original distorted pixel to sub-pixel accuracy
        REQUIRE_THAT(recovered.x(), Catch::Matchers::WithinAbs(distorted_px.x(), 0.1));
        REQUIRE_THAT(recovered.y(), Catch::Matchers::WithinAbs(distorted_px.y(), 0.1));

        // The undistorted pixel must differ noticeably from the distorted one
        // (confirms the rational model actually shifts, not just ≈identity)
        REQUIRE((undistorted_px - distorted_px).norm() > 5.0);
    }
}

TEST_CASE("Camera undistortion", "[camera]") {
    SECTION("Undistort-distort roundtrip (polynomial model)") {
        auto const intr = make_test_intrinsics(800.0, 800.0, 320.0, 240.0, 640, 480,
                                               {0.15, -0.08, 0.01, 0.02, 0.03});
        Camera const cam = make_test_camera("cam1", intr);

        Eigen::Vector2d const original_pixel(500.0, 350.0);
        Eigen::Vector2d const undistorted_pixel = cam.undistort(original_pixel);

        // Distort back by extracting normalized coordinates and applying distortion manually
        Eigen::Vector2d norm((undistorted_pixel.x() - 320.0) / 800.0,
                             (undistorted_pixel.y() - 240.0) / 800.0);
        Eigen::Vector2d dist_norm = cam.distort(norm);
        Eigen::Vector2d reconstructed(800.0 * dist_norm.x() + 320.0, 800.0 * dist_norm.y() + 240.0);

        REQUIRE(reconstructed.isApprox(original_pixel, 0.01));  // 0.01 pixel tolerance
    }

    SECTION("Undistort-distort roundtrip (rational model)") {
        std::vector<double> const coeffs = {0.3, -0.1, 0.005, 0.003, 0.05, 0.2, -0.05, 0.01};
        auto const intr = make_test_intrinsics(800.0, 800.0, 320.0, 240.0, 640, 480, coeffs);
        Camera const cam = make_test_camera("cam1", intr);

        for (auto const& px : {Eigen::Vector2d(500.0, 350.0), Eigen::Vector2d(100.0, 400.0),
                               Eigen::Vector2d(600.0, 100.0)}) {
            Eigen::Vector2d const undistorted = cam.undistort(px);
            Eigen::Vector2d norm((undistorted.x() - 320.0) / 800.0,
                                 (undistorted.y() - 240.0) / 800.0);
            Eigen::Vector2d dist_norm = cam.distort(norm);
            Eigen::Vector2d recovered(800.0 * dist_norm.x() + 320.0, 800.0 * dist_norm.y() + 240.0);
            REQUIRE(recovered.isApprox(px, 0.01));
        }
    }
}

TEST_CASE("Camera projection with distortion", "[camera]") {
    auto const intr =
        make_test_intrinsics(800.0, 800.0, 320.0, 240.0, 640, 480, {0.1, -0.05, 0, 0, 0});
    Camera const cam = make_test_camera("cam1", intr);

    // Use z=5 so the projected pixel stays within the 640×480 image.
    Eigen::Vector3d const point(1.0, 0.5, 5.0);
    auto pixel_distorted_opt = cam.project(point);
    auto pixel_undistorted_opt = cam.project_undistorted(point);
    REQUIRE(pixel_distorted_opt.has_value());
    REQUIRE(pixel_undistorted_opt.has_value());
    Eigen::Vector2d const pixel_distorted = *pixel_distorted_opt;
    Eigen::Vector2d const pixel_undistorted = *pixel_undistorted_opt;

    // Distorted and undistorted should be different
    REQUIRE((pixel_distorted - pixel_undistorted).norm() > 0.01);
}

TEST_CASE("Camera bounds checking", "[camera]") {
    Camera const cam = make_test_camera();

    REQUIRE(cam.is_in_bounds(Eigen::Vector2d(0.0, 0.0)));
    REQUIRE(cam.is_in_bounds(Eigen::Vector2d(639.0, 479.0)));
    REQUIRE_FALSE(cam.is_in_bounds(Eigen::Vector2d(-1.0, 0.0)));
    REQUIRE_FALSE(cam.is_in_bounds(Eigen::Vector2d(0.0, -1.0)));
    REQUIRE_FALSE(cam.is_in_bounds(Eigen::Vector2d(640.0, 0.0)));
    REQUIRE_FALSE(cam.is_in_bounds(Eigen::Vector2d(0.0, 480.0)));
}

TEST_CASE("Camera frame-to-timestamp conversion", "[camera]") {
    SECTION("Uniform FPS without sync points") {
        Camera const cam =
            make_test_camera("cam1", make_test_intrinsics(), make_test_extrinsics(), 30.0, 0);

        REQUIRE_THAT(cam.get_timestamp(0), Catch::Matchers::WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(cam.get_timestamp(30), Catch::Matchers::WithinAbs(1.0, 1e-9));
        REQUIRE_THAT(cam.get_timestamp(60), Catch::Matchers::WithinAbs(2.0, 1e-9));
    }

    SECTION("With start_frame offset") {
        Camera const cam =
            make_test_camera("cam1", make_test_intrinsics(), make_test_extrinsics(), 30.0, 100);

        REQUIRE_THAT(cam.get_timestamp(100), Catch::Matchers::WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(cam.get_timestamp(130), Catch::Matchers::WithinAbs(1.0, 1e-9));
    }

    SECTION("With sync points") {
        Camera cam =
            make_test_camera("cam1", make_test_intrinsics(), make_test_extrinsics(), 30.0, 0);

        std::vector<SyncPoint> sync_points;
        sync_points.push_back({0, 0.0});
        sync_points.push_back({100, 3.5});  // Non-uniform: 100 frames in 3.5 seconds
        sync_points.push_back({200, 7.0});  // Linear: next 100 frames in 3.5 seconds
        cam.set_sync_points(sync_points);

        REQUIRE_THAT(cam.get_timestamp(0), Catch::Matchers::WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(cam.get_timestamp(100), Catch::Matchers::WithinAbs(3.5, 1e-9));
        REQUIRE_THAT(cam.get_timestamp(200), Catch::Matchers::WithinAbs(7.0, 1e-9));

        // Interpolated value: halfway between frame 0 and 100
        REQUIRE_THAT(cam.get_timestamp(50), Catch::Matchers::WithinAbs(1.75, 1e-9));

        // Extrapolate beyond last sync point using FPS
        double expected = 7.0 + (250 - 200) / 30.0;
        REQUIRE_THAT(cam.get_timestamp(250), Catch::Matchers::WithinAbs(expected, 1e-9));
    }

    SECTION("Sync points starting after frame 0 - wraparound bug test") {
        // Test case for bug: first sync point is at frame 100, but we request frame 50
        // This previously caused wraparound: (50 - 100) as uint32_t wraps to huge number
        Camera cam =
            make_test_camera("cam1", make_test_intrinsics(), make_test_extrinsics(), 30.0, 0);

        std::vector<SyncPoint> sync_points;
        sync_points.push_back({100, 0.0});  // First sync point at frame 100, time 0.0
        sync_points.push_back({200, 3.0});  // Second at frame 200, time 3.0
        cam.set_sync_points(sync_points);

        // Request timestamp for frame BEFORE first sync point
        // Should extrapolate backward: frame 50 is 50 frames before frame 100
        // At 30 fps, that's 50/30 = -1.667 seconds
        double expected = 0.0 - (100 - 50) / 30.0;  // -1.6667 seconds
        REQUIRE_THAT(cam.get_timestamp(50), Catch::Matchers::WithinAbs(expected, 1e-6));

        // Another test: frame 0 should be 100 frames before first sync point
        double expected_frame0 = 0.0 - 100 / 30.0;  // -3.333 seconds
        REQUIRE_THAT(cam.get_timestamp(0), Catch::Matchers::WithinAbs(expected_frame0, 1e-6));
    }
}

TEST_CASE("Camera timestamp-to-frame conversion", "[camera]") {
    SECTION("Uniform FPS - floor semantics") {
        Camera const cam =
            make_test_camera("cam1", make_test_intrinsics(), make_test_extrinsics(), 30.0, 0);

        // Floor: last frame at or before timestamp
        auto f0 = cam.get_frame_at_time(0.0);
        REQUIRE(f0.has_value());
        REQUIRE(*f0 == 0);

        auto f1 = cam.get_frame_at_time(1.0);  // Exactly 30 frames at 30 fps
        REQUIRE(f1.has_value());
        REQUIRE(*f1 == 30);

        auto f2 = cam.get_frame_at_time(1.5);  // 45 frames at 30 fps
        REQUIRE(f2.has_value());
        REQUIRE(*f2 == 45);

        // Between frames: should return floor
        auto f3 = cam.get_frame_at_time(0.02);  // Between frame 0 (0.0s) and 1 (0.0333s)
        REQUIRE(f3.has_value());
        REQUIRE(*f3 == 0);  // Floor

        auto f4 = cam.get_frame_at_time(0.04);  // Just past frame 1
        REQUIRE(f4.has_value());
        REQUIRE(*f4 == 1);  // Floor
    }

    SECTION("Before first frame returns nullopt") {
        Camera const cam =
            make_test_camera("cam1", make_test_intrinsics(), make_test_extrinsics(), 30.0, 100);

        auto f = cam.get_frame_at_time(-1.0);  // Before start
        REQUIRE_FALSE(f.has_value());

        auto f2 = cam.get_frame_at_time(0.0);  // Exactly at first frame
        REQUIRE(f2.has_value());
        REQUIRE(*f2 == 100);
    }

    SECTION("With sync points") {
        Camera cam =
            make_test_camera("cam1", make_test_intrinsics(), make_test_extrinsics(), 30.0, 0);

        std::vector<SyncPoint> sync_points;
        sync_points.push_back({0, 0.0});
        sync_points.push_back({100, 3.5});
        sync_points.push_back({200, 7.0});
        cam.set_sync_points(sync_points);

        auto f0 = cam.get_frame_at_time(0.0);
        REQUIRE(f0.has_value());
        REQUIRE(*f0 == 0);

        auto f1 = cam.get_frame_at_time(3.5);
        REQUIRE(f1.has_value());
        REQUIRE(*f1 == 100);

        // Interpolated: halfway between (should be floor of 50)
        auto f2 = cam.get_frame_at_time(1.75);
        REQUIRE(f2.has_value());
        REQUIRE(*f2 == 50);

        // Between frames with interpolation
        auto f3 = cam.get_frame_at_time(1.76);  // Slightly past 1.75
        REQUIRE(f3.has_value());
        REQUIRE(*f3 == 50);  // Still floor of ~50.x

        // Extrapolate forward using rate from last two sync points
        // Rate = (200-100)/(7.0-3.5) = 100/3.5 ≈ 28.57 fps
        auto f4 = cam.get_frame_at_time(8.0);  // 1 second past last sync
        REQUIRE(f4.has_value());
        REQUIRE(*f4 == 228);  // 200 + floor(1.0 * 28.57) = 228
    }

    SECTION("Before first sync point extrapolates backward") {
        Camera cam =
            make_test_camera("cam1", make_test_intrinsics(), make_test_extrinsics(), 30.0, 0);

        std::vector<SyncPoint> sync_points;
        sync_points.push_back({100, 1.0});  // First frame starts at t=1.0
        cam.set_sync_points(sync_points);

        // Before first sync point: extrapolate backward using FPS
        // t=0.5, sp0={100, 1.0}, dt=-0.5, frame_offset = -0.5 * 30 = -15, frame = 100 - 15 = 85
        auto f = cam.get_frame_at_time(0.5);
        REQUIRE(f.has_value());
        REQUIRE(*f == 85);

        auto f2 = cam.get_frame_at_time(1.0);  // At first sync point
        REQUIRE(f2.has_value());
        REQUIRE(*f2 == 100);
    }
}

TEST_CASE("Camera JSON serialization", "[camera]") {
    auto const intr =
        make_test_intrinsics(800.0, 800.0, 320.0, 240.0, 640, 480, {0.1, -0.05, 0.01, 0.02, 0.03});

    Eigen::Vector3d const pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond const quat(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()));
    auto const extr = make_test_extrinsics(pos, quat);

    Camera cam = make_test_camera("cam1", intr, extr, 60.0, 100);

    // Add sync points
    std::vector<SyncPoint> sync_points;
    sync_points.push_back({100, 0.0});
    sync_points.push_back({200, 1.5});
    cam.set_sync_points(sync_points);

    // Serialize
    nlohmann::json const j = cam.to_json();

    // Deserialize
    Camera const cam2 = Camera::from_json(j);

    // Verify
    REQUIRE(cam2.name() == cam.name());
    REQUIRE_THAT(cam2.intrinsics().fx, Catch::Matchers::WithinAbs(cam.intrinsics().fx, 1e-10));
    REQUIRE_THAT(cam2.intrinsics().fy, Catch::Matchers::WithinAbs(cam.intrinsics().fy, 1e-10));
    REQUIRE_THAT(cam2.intrinsics().cx, Catch::Matchers::WithinAbs(cam.intrinsics().cx, 1e-10));
    REQUIRE_THAT(cam2.intrinsics().cy, Catch::Matchers::WithinAbs(cam.intrinsics().cy, 1e-10));
    REQUIRE(cam2.intrinsics().width == cam.intrinsics().width);
    REQUIRE(cam2.intrinsics().height == cam.intrinsics().height);
    REQUIRE(cam2.fps() == cam.fps());
    REQUIRE(cam2.start_frame() == cam.start_frame());

    REQUIRE(cam2.position().isApprox(cam.position(), 1e-10));
    REQUIRE(cam2.orientation().coeffs().isApprox(cam.orientation().coeffs(), 1e-10));

    // Verify sync points were preserved
    REQUIRE_THAT(cam2.get_timestamp(100), Catch::Matchers::WithinAbs(cam.get_timestamp(100), 1e-9));
    REQUIRE_THAT(cam2.get_timestamp(200), Catch::Matchers::WithinAbs(cam.get_timestamp(200), 1e-9));
}
