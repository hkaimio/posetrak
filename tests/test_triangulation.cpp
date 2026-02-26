#include <posetrak/io/skeleton_loader.hpp>
#include <posetrak/kinematics/forward_kinematics.hpp>
#include <posetrak/kinematics/pinocchio_model_builder.hpp>
#include <posetrak/kinematics/triangulation.hpp>

#include <fmt/core.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/skeleton_layout.hpp"
#include <random>

using namespace posetrak;

namespace {

/// @brief Test fixture for creating synthetic triangulation scenarios
class TriangulationTestFixture {
   public:
    TriangulationTestFixture() : rng_(42) {}  // Fixed seed for reproducibility

    /// @brief Create cameras in a semi-circle around origin
    /// @param num_cameras Number of cameras to create
    /// @param radius Distance from origin
    /// @param height Height above ground plane
    void setup_cameras(int num_cameras, double radius = 5.0, double height = 2.0) {
        cameras.clear();

        for (int i = 0; i < num_cameras; ++i) {
            // Position cameras in semi-circle
            double angle = M_PI * static_cast<double>(i) / static_cast<double>(num_cameras - 1);
            Eigen::Vector3d pos(radius * std::cos(angle), radius * std::sin(angle), height);

            // Look at target point at camera height (horizontal look)
            Eigen::Vector3d target(0, 0, height);  // Look horizontally
            Eigen::Vector3d look_dir = (target - pos).normalized();
            Eigen::Vector3d up(0, 0, 1);
            Eigen::Vector3d right = look_dir.cross(up).normalized();
            up = right.cross(look_dir).normalized();

            Eigen::Matrix3d R_cam_to_world;
            R_cam_to_world.col(0) = right;
            R_cam_to_world.col(1) = -up;       // Camera y points down
            R_cam_to_world.col(2) = look_dir;  // Camera z points forward

            // Transpose to get world-to-camera rotation (Camera stores R such that p_cam = R *
            // (p_world - C))
            Eigen::Matrix3d R = R_cam_to_world.transpose();

            // Create intrinsics (wider FOV for better coverage)
            Intrinsics intr;
            intr.fx = 400.0;  // Reduced from 800 for wider FOV
            intr.fy = 400.0;
            intr.cx = 640.0;
            intr.cy = 360.0;
            intr.width = 1280;
            intr.height = 720;
            intr.model = Intrinsics::DistortionModel::BrownConrady;
            intr.distortion_coeffs = {0, 0, 0, 0, 0};  // No distortion for tests

            Extrinsics extr;
            extr.position = pos;
            extr.orientation = Eigen::Quaterniond(R);

            cameras.emplace_back(i, "camera_" + std::to_string(i), intr, extr);
        }
    }

    /// @brief Project 3D point to all cameras
    /// @param point_3d Point in world frame
    /// @param confidences Output confidences (optional, defaults to 1.0)
    /// @return 2D observations (undistorted)
    std::vector<Eigen::Vector2d> project_point(Eigen::Vector3d const& point_3d,
                                               std::vector<double>* confidences = nullptr) {
        std::vector<Eigen::Vector2d> observations;

        if (confidences) {
            confidences->clear();
        }

        for (size_t i = 0; i < cameras.size(); ++i) {
            auto const& cam = cameras[i];
            auto pixel_opt = cam.project_undistorted(point_3d);

            // Only include if projection succeeded (in bounds and in front of camera)
            if (pixel_opt.has_value()) {
                observations.push_back(*pixel_opt);
                if (confidences) {
                    confidences->push_back(1.0);  // Perfect confidence
                }
            }
        }

        return observations;
    }

    /// @brief Add Gaussian noise to observations
    /// @param observations Observations to modify
    /// @param std_dev_pixels Standard deviation in pixels
    void add_noise(std::vector<Eigen::Vector2d>& observations, double std_dev_pixels = 2.0) {
        std::normal_distribution<double> dist(0.0, std_dev_pixels);

        for (auto& obs : observations) {
            obs.x() += dist(rng_);
            obs.y() += dist(rng_);
        }
    }

    /// @brief Create ObservationSet for full skeleton
    /// @param skeleton Skeleton with markers
    /// @param joint_angles Joint configuration
    /// @param timestamp Timestamp for observations
    /// @return Observation set across all cameras
    ObservationSet create_skeleton_observations(Skeleton const& skeleton,
                                                Eigen::VectorXd const& joint_angles,
                                                double timestamp) {
        // Compute FK to get marker positions
        pinocchio::Model model;
        pinocchio::Data data;
        PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
        auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);

        Eigen::Vector3d root_pos = Eigen::Vector3d::Zero();
        Eigen::Quaterniond root_quat = Eigen::Quaterniond::Identity();
        Eigen::Vector3d root_vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d root_angvel = Eigen::Vector3d::Zero();
        Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(joint_angles.size());

        State state(root_pos, root_quat, joint_angles, root_vel, root_angvel, joint_vels);
        auto fk_layout =
            SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));

        ForwardKinematics fk(model, data, marker_map, fk_layout);
        Eigen::VectorXd q = ForwardKinematics::state_to_config(state, *fk_layout);
        auto marker_positions = fk.compute(q);

        // Create observations
        ObservationSet obs_set(0);

        for (size_t cam_idx = 0; cam_idx < cameras.size(); ++cam_idx) {
            ObservationSequence seq;
            seq.camera_id = cameras[cam_idx].id();
            seq.camera_name = cameras[cam_idx].name();

            for (auto const& marker : skeleton.markers()) {
                auto pos_it = marker_positions.find(marker.name);
                if (pos_it == marker_positions.end()) {
                    continue;
                }

                Eigen::Vector3d const& pos_3d = pos_it->second;
                auto pixel_opt = cameras[cam_idx].project_undistorted(pos_3d);

                if (pixel_opt.has_value()) {
                    Eigen::Vector2d pixel = *pixel_opt;

                    Observation obs;
                    obs.camera_id = cameras[cam_idx].id();
                    obs.marker_id = static_cast<int>(
                        &marker - &skeleton.markers()[0]);  // Use marker index as ID
                    obs.frame_idx = 0;
                    obs.timestamp = timestamp;
                    obs.position = pixel;
                    obs.position_distorted = pixel;
                    obs.confidence = 1.0;

                    seq.observations.push_back(obs);
                }
            }

            obs_set.add_sequence(seq);
        }

        return obs_set;
    }

    std::vector<Camera> cameras;

   private:
    std::mt19937 rng_;
};

}  // namespace

TEST_CASE("Triangulation with 2 cameras - perfect observations", "[triangulation]") {
    TriangulationTestFixture fixture;
    fixture.setup_cameras(2);

    // Known 3D point at camera height, slightly off-center to avoid degenerate geometry
    // (rays from opposite cameras would be anti-parallel at exact center)
    Eigen::Vector3d point_3d(0.0, 0.5, 2.0);

    // Project to both cameras
    auto observations = fixture.project_point(point_3d);
    REQUIRE(observations.size() == 2);

    // Get camera pointers
    std::vector<Camera const*> cams = {&fixture.cameras[0], &fixture.cameras[1]};

    SECTION("Mid-point method") {
        Triangulator triangulator(Triangulator::Method::MidPoint);
        auto result = triangulator.triangulate(observations, cams);

        REQUIRE(result.success);
        REQUIRE(result.num_cameras == 2);

        // Should recover position within numerical precision
        REQUIRE_THAT(result.position.x(), Catch::Matchers::WithinAbs(point_3d.x(), 1e-6));
        REQUIRE_THAT(result.position.y(), Catch::Matchers::WithinAbs(point_3d.y(), 1e-6));
        REQUIRE_THAT(result.position.z(), Catch::Matchers::WithinAbs(point_3d.z(), 1e-6));

        // Reprojection error should be near zero
        REQUIRE_THAT(result.reprojection_error, Catch::Matchers::WithinAbs(0.0, 1e-3));
    }

    SECTION("DLT method") {
        Triangulator triangulator(Triangulator::Method::DLT);
        auto result = triangulator.triangulate(observations, cams);

        REQUIRE(result.success);
        REQUIRE(result.num_cameras == 2);

        // Should recover position within numerical precision
        REQUIRE_THAT(result.position.x(), Catch::Matchers::WithinAbs(point_3d.x(), 1e-6));
        REQUIRE_THAT(result.position.y(), Catch::Matchers::WithinAbs(point_3d.y(), 1e-6));
        REQUIRE_THAT(result.position.z(), Catch::Matchers::WithinAbs(point_3d.z(), 1e-6));

        REQUIRE_THAT(result.reprojection_error, Catch::Matchers::WithinAbs(0.0, 1e-3));
    }
}

TEST_CASE("Triangulation with 4 cameras - overdetermined system", "[triangulation]") {
    TriangulationTestFixture fixture;
    fixture.setup_cameras(4);

    // Point at camera height (should project to center)
    Eigen::Vector3d point_3d(0.0, 0.0, 2.0);

    std::vector<double> confidences;
    auto observations = fixture.project_point(point_3d, &confidences);
    REQUIRE(observations.size() == 4);

    // Add small noise
    fixture.add_noise(observations, 0.5);  // 0.5 pixel noise

    std::vector<Camera const*> cams;
    for (auto const& cam : fixture.cameras) {
        cams.push_back(&cam);
    }

    Triangulator triangulator(Triangulator::Method::DLT);
    auto result = triangulator.triangulate(observations, cams, confidences);

    REQUIRE(result.success);
    REQUIRE(result.num_cameras == 4);

    // With small noise, should still be very accurate
    REQUIRE_THAT(result.position.x(), Catch::Matchers::WithinAbs(point_3d.x(), 0.01));
    REQUIRE_THAT(result.position.y(), Catch::Matchers::WithinAbs(point_3d.y(), 0.01));
    REQUIRE_THAT(result.position.z(), Catch::Matchers::WithinAbs(point_3d.z(), 0.01));

    // Reprojection error should be small
    REQUIRE(result.reprojection_error < 1.0);
}

TEST_CASE("Triangulation respects confidence weights", "[triangulation]") {
    TriangulationTestFixture fixture;
    fixture.setup_cameras(3);

    // Point at camera height (should project to center)
    Eigen::Vector3d point_3d(0.0, 0.0, 2.0);

    std::vector<double> confidences;
    auto observations = fixture.project_point(point_3d, &confidences);
    REQUIRE(observations.size() == 3);

    // Add large noise to first observation, small noise to others
    std::normal_distribution<double> dist_large(0.0, 10.0);
    std::normal_distribution<double> dist_small(0.0, 0.5);
    std::mt19937 rng(123);

    observations[0].x() += dist_large(rng);
    observations[0].y() += dist_large(rng);
    observations[1].x() += dist_small(rng);
    observations[1].y() += dist_small(rng);
    observations[2].x() += dist_small(rng);
    observations[2].y() += dist_small(rng);

    std::vector<Camera const*> cams;
    for (auto const& cam : fixture.cameras) {
        cams.push_back(&cam);
    }

    // Triangulate without confidence weighting
    Triangulator triangulator(Triangulator::Method::DLT);
    auto result_unweighted = triangulator.triangulate(observations, cams);

    // Triangulate with confidence weighting (low confidence for noisy observation)
    confidences[0] = 0.1;  // Low confidence
    confidences[1] = 1.0;  // High confidence
    confidences[2] = 1.0;  // High confidence
    auto result_weighted = triangulator.triangulate(observations, cams, confidences);

    REQUIRE(result_weighted.success);

    // Weighted result should be closer to ground truth
    double error_weighted = (result_weighted.position - point_3d).norm();
    double error_unweighted = (result_unweighted.position - point_3d).norm();

    REQUIRE(error_weighted < error_unweighted);
}

TEST_CASE("Triangulation handles degenerate configurations", "[triangulation]") {
    TriangulationTestFixture fixture;

    SECTION("Single camera fails") {
        fixture.setup_cameras(1);
        Eigen::Vector3d point_3d(0.0, 0.0, 1.0);
        auto observations = fixture.project_point(point_3d);

        std::vector<Camera const*> cams = {&fixture.cameras[0]};

        Triangulator triangulator;
        auto result = triangulator.triangulate(observations, cams);

        REQUIRE_FALSE(result.success);
    }

    SECTION("No observations fails") {
        fixture.setup_cameras(2);

        std::vector<Eigen::Vector2d> observations;
        std::vector<Camera const*> cams;

        Triangulator triangulator;
        auto result = triangulator.triangulate(observations, cams);

        REQUIRE_FALSE(result.success);
    }

    SECTION("Mismatched observations and cameras fails") {
        fixture.setup_cameras(2);
        Eigen::Vector3d point_3d(0.0, 0.0, 1.0);
        auto observations = fixture.project_point(point_3d);

        std::vector<Camera const*> cams = {&fixture.cameras[0]};  // Only one camera

        Triangulator triangulator;
        auto result = triangulator.triangulate(observations, cams);

        REQUIRE_FALSE(result.success);
    }
}

TEST_CASE("Triangulate full frame with multiple markers", "[triangulation]") {
    TriangulationTestFixture fixture;
    fixture.setup_cameras(3);

    // Create 10 markers at different positions near camera height
    std::vector<Eigen::Vector3d> marker_positions;
    for (int i = 0; i < 10; ++i) {
        double x = (i % 3) * 0.2 - 0.2;  // -0.2 to 0.2
        double y = (i / 3) * 0.2 - 0.2;  // -0.2 to 0.2
        double z = 1.8 + (i % 2) * 0.4;  // 1.8 to 2.2 (around camera height 2.0)
        marker_positions.emplace_back(x, y, z);
    }

    // Create ObservationSet
    ObservationSet obs_set(0);
    double timestamp = 1.0;

    for (size_t cam_idx = 0; cam_idx < fixture.cameras.size(); ++cam_idx) {
        ObservationSequence seq;
        seq.camera_id = fixture.cameras[cam_idx].id();
        seq.camera_name = fixture.cameras[cam_idx].name();

        for (size_t marker_idx = 0; marker_idx < marker_positions.size(); ++marker_idx) {
            auto pixel_opt =
                fixture.cameras[cam_idx].project_undistorted(marker_positions[marker_idx]);

            if (pixel_opt.has_value()) {
                Eigen::Vector2d pixel = *pixel_opt;

                Observation obs;
                obs.camera_id = fixture.cameras[cam_idx].id();
                obs.marker_id = static_cast<int>(marker_idx);
                obs.frame_idx = 0;
                obs.timestamp = timestamp;
                obs.position = pixel;
                obs.position_distorted = pixel;
                obs.confidence = 1.0;

                seq.observations.push_back(obs);
            }
        }

        obs_set.add_sequence(seq);
    }

    // Triangulate frame
    Triangulator triangulator;
    auto results = triangulator.triangulate_frame(timestamp, obs_set, fixture.cameras);

    // All markers should be triangulated
    REQUIRE(results.size() == marker_positions.size());

    // Check accuracy
    for (size_t i = 0; i < marker_positions.size(); ++i) {
        REQUIRE(results.count(i) > 0);
        auto const& result = results.at(i);

        REQUIRE(result.success);
        REQUIRE_THAT(result.position.x(),
                     Catch::Matchers::WithinAbs(marker_positions[i].x(), 0.01));
        REQUIRE_THAT(result.position.y(),
                     Catch::Matchers::WithinAbs(marker_positions[i].y(), 0.01));
        REQUIRE_THAT(result.position.z(),
                     Catch::Matchers::WithinAbs(marker_positions[i].z(), 0.01));
    }
}

TEST_CASE("Triangulate markers for full skeleton pose", "[triangulation]") {
    // Load real skeleton
    Skeleton skeleton = load_skeleton_from_yaml("tests/data/simple_humanoid.yaml");

    TriangulationTestFixture fixture;
    fixture.setup_cameras(4);

    // Create synthetic pose (all zeros for simplicity)
    int num_dof = 0;
    for (auto const& joint : skeleton.joints()) {
        num_dof += joint.dof;
    }
    Eigen::VectorXd joint_angles = Eigen::VectorXd::Zero(num_dof);

    // Create observations
    double timestamp = 0.0;
    ObservationSet obs_set =
        fixture.create_skeleton_observations(skeleton, joint_angles, timestamp);

    // Add realistic noise
    // (Note: This modifies the internal observations in a real implementation,
    //  but for this test we'll triangulate with perfect observations)

    // Triangulate
    Triangulator triangulator;
    auto results = triangulator.triangulate_frame(timestamp, obs_set, fixture.cameras);

    // Should triangulate most markers (some may be out of view)
    REQUIRE(results.size() >= static_cast<size_t>(skeleton.markers().size() / 2));

    // All triangulated markers should have reasonable error
    for (auto const& [marker_id, result] : results) {
        REQUIRE(result.success);
        REQUIRE(result.num_cameras >= 2);
        REQUIRE(result.reprojection_error < 2.0);  // Should be very accurate with perfect obs

        // Position should be reasonable (within skeleton bounds)
        REQUIRE(result.position.norm() < 5.0);
    }
}
