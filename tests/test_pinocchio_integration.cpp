#include <nlohmann/json.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/io/skeleton_loader.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include "posetrak/kinematics/pinocchio_model_builder.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

using namespace posetrak;
using json = nlohmann::json;

TEST_CASE("PinocchioModelBuilder builds model from skeleton", "[pinocchio]") {
    // Create a simple skeleton: root + 1 revolute joint + 1 marker
    Skeleton skeleton;

    // Root joint (pelvis)
    uint32_t pelvis_idx = skeleton.add_joint("pelvis", std::nullopt, JointType::REVOLUTE,
                                             Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    // Child joint (spine)
    uint32_t spine_idx = skeleton.add_joint("spine", pelvis_idx, JointType::REVOLUTE,
                                            Eigen::Vector3d(0, 0, 0.1), Eigen::Vector3d::Zero());

    // Marker on spine
    skeleton.add_marker("head_marker", spine_idx, Eigen::Vector3d(0, 0, 0.2), std::nullopt);

    SECTION("Model builds successfully") {
        pinocchio::Model model;
        REQUIRE_NOTHROW(PinocchioModelBuilder::build_model(skeleton, model));

        // Check model dimensions
        // Root: 7 DOF (3 pos + 4 quat), Spine: 1 DOF (revolute)
        REQUIRE(model.nq == 8);  // 7 + 1
        REQUIRE(model.nv == 7);  // 6 (root velocity) + 1

        // Should have 2 joints + universe
        REQUIRE(model.njoints == 3);
    }

    SECTION("Marker frames are created") {
        pinocchio::Model model;
        PinocchioModelBuilder::build_model(skeleton, model);

        // Check marker frame exists
        REQUIRE(model.existFrame("head_marker"));
        auto frame_id = model.getFrameId("head_marker");
        REQUIRE(frame_id >= 0);

        // Check frame type
        REQUIRE(model.frames[frame_id].type == pinocchio::OP_FRAME);
    }

    SECTION("Marker frame map builds correctly") {
        pinocchio::Model model;
        PinocchioModelBuilder::build_model(skeleton, model);

        auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
        REQUIRE(marker_map.size() == 1);
        REQUIRE(marker_map.count("head_marker") > 0);
    }
}

TEST_CASE("ForwardKinematics computes marker positions", "[forward_kinematics]") {
    // Create a simple skeleton
    Skeleton skeleton;
    uint32_t pelvis_idx = skeleton.add_joint("pelvis", std::nullopt, JointType::REVOLUTE,
                                             Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    uint32_t spine_idx = skeleton.add_joint("spine", pelvis_idx, JointType::REVOLUTE,
                                            Eigen::Vector3d(0, 0, 0.1), Eigen::Vector3d::Zero());

    skeleton.add_marker("head_marker", spine_idx, Eigen::Vector3d(0, 0, 0.2), std::nullopt);

    // Build Pinocchio model
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);

    auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);

    // Create FK computer
    auto fk_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    ForwardKinematics fk(model, data, marker_map, fk_layout);

    SECTION("FK with zero configuration") {
        // Configuration: root at origin, no rotation, spine at 0
        Eigen::VectorXd q = Eigen::VectorXd::Zero(model.nq);
        q[3] = 0.0;  // quat x
        q[4] = 0.0;  // quat y
        q[5] = 0.0;  // quat z
        q[6] = 1.0;  // quat w (identity)
        // q[7] = 0.0;  // spine angle (already zero)

        auto positions = fk.compute(q);

        REQUIRE(positions.size() == 1);
        REQUIRE(positions.count("head_marker") > 0);

        // Marker should be at [0, 0, 0.3] (0.1 from root to spine + 0.2 from spine to marker)
        auto const& pos = positions["head_marker"];
        REQUIRE_THAT(pos.x(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(pos.y(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(pos.z(), Catch::Matchers::WithinAbs(0.3, 1e-6));
    }

    SECTION("state_to_config converts State correctly") {
        // Create a state
        Eigen::Vector3d root_pos(1.0, 2.0, 3.0);
        Eigen::Quaterniond root_quat = Eigen::Quaterniond::Identity();
        Eigen::VectorXd joint_angles(1);
        joint_angles << 0.5;  // spine angle
        Eigen::Vector3d root_vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d root_angvel = Eigen::Vector3d::Zero();
        Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(1);

        State state(root_pos, root_quat, joint_angles, root_vel, root_angvel, joint_vels);

        auto q = ForwardKinematics::state_to_config(state, skeleton);

        REQUIRE(q.size() == model.nq);

        // Check root position
        REQUIRE_THAT(q[0], Catch::Matchers::WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(q[1], Catch::Matchers::WithinAbs(2.0, 1e-6));
        REQUIRE_THAT(q[2], Catch::Matchers::WithinAbs(3.0, 1e-6));

        // Check root quaternion (identity: x=0, y=0, z=0, w=1)
        REQUIRE_THAT(q[3], Catch::Matchers::WithinAbs(0.0, 1e-6));  // x
        REQUIRE_THAT(q[4], Catch::Matchers::WithinAbs(0.0, 1e-6));  // y
        REQUIRE_THAT(q[5], Catch::Matchers::WithinAbs(0.0, 1e-6));  // z
        REQUIRE_THAT(q[6], Catch::Matchers::WithinAbs(1.0, 1e-6));  // w

        // Check spine angle
        REQUIRE_THAT(q[7], Catch::Matchers::WithinAbs(0.5, 1e-6));
    }
}

TEST_CASE("ForwardKinematics handles spherical joints", "[forward_kinematics]") {
    // Create skeleton with spherical joint
    Skeleton skeleton;
    uint32_t pelvis_idx = skeleton.add_joint("pelvis", std::nullopt, JointType::REVOLUTE,
                                             Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    uint32_t shoulder_idx = skeleton.add_joint("shoulder", pelvis_idx, JointType::SPHERICAL,
                                               Eigen::Vector3d(0.2, 0, 0), Eigen::Vector3d::Zero());

    skeleton.add_marker("hand", shoulder_idx, Eigen::Vector3d(0.3, 0, 0), std::nullopt);
    skeleton.add_marker("shoulder_marker", shoulder_idx, Eigen::Vector3d(0, 0, 0.1), std::nullopt);

    // Build model
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);

    auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
    auto fk_layout2 =
        SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    ForwardKinematics fk(model, data, marker_map, fk_layout2);

    SECTION("state_to_config converts spherical joint correctly") {
        // State with spherical joint angles
        Eigen::Vector3d root_pos = Eigen::Vector3d::Zero();
        Eigen::Quaterniond root_quat = Eigen::Quaterniond::Identity();
        Eigen::VectorXd joint_angles(3);  // 3 DOF for spherical
        joint_angles << 0.1, 0.2, 0.3;    // rotation angles

        Eigen::Vector3d root_vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d root_angvel = Eigen::Vector3d::Zero();
        Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(3);

        State state(root_pos, root_quat, joint_angles, root_vel, root_angvel, joint_vels);

        auto q = ForwardKinematics::state_to_config(state, skeleton);

        // Root: 7 DOF, Shoulder (spherical): 4 DOF (quaternion)
        REQUIRE(q.size() == 11);

        // Verify quaternion conversion (angle-axis)
        Eigen::Vector3d angles(0.1, 0.2, 0.3);
        double angle = angles.norm();
        Eigen::Vector3d axis = angles.normalized();
        Eigen::Quaterniond expected_quat(Eigen::AngleAxisd(angle, axis));

        // Check shoulder quaternion [x,y,z,w] at indices 7-10
        REQUIRE_THAT(q[7], Catch::Matchers::WithinAbs(expected_quat.x(), 1e-6));
        REQUIRE_THAT(q[8], Catch::Matchers::WithinAbs(expected_quat.y(), 1e-6));
        REQUIRE_THAT(q[9], Catch::Matchers::WithinAbs(expected_quat.z(), 1e-6));
        REQUIRE_THAT(q[10], Catch::Matchers::WithinAbs(expected_quat.w(), 1e-6));
    }
}

TEST_CASE("ForwardKinematics validates against Python ground truth",
          "[forward_kinematics][integration]") {
    // Load full skeleton from YAML
    std::string test_data_dir = "tests/data";
    Skeleton skeleton =
        load_skeleton_from_yaml(test_data_dir + "/Harri_skeleton-shouldery-rot.yaml");

    INFO("Loaded skeleton: " << skeleton.joints().size() << " joints, " << skeleton.markers().size()
                             << " markers");

    // Build Pinocchio model
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
    auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
    auto fk_layout_gt =
        SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    ForwardKinematics fk(model, data, marker_map, fk_layout_gt);

    std::cout << "\n=== Pinocchio Model Joint Order ===" << std::endl;
    for (size_t i = 1; i < model.names.size(); ++i) {  // Skip universe (index 0)
        std::cout << "  " << i << ": " << model.names[i] << " (nq=" << model.nqs[i]
                  << ", nv=" << model.nvs[i] << ")" << std::endl;
        if (i >= 35) {
            std::cout << "  ... (showing first 35 joints)" << std::endl;
            break;
        }
    }
    std::cout << std::endl;

    INFO("Built model: nq=" << model.nq << ", nv=" << model.nv);

    // Load states from JSON
    std::ifstream states_file(test_data_dir + "/states_0.json");
    REQUIRE(states_file.is_open());
    json states_json;
    states_file >> states_json;
    states_file.close();

    // Load ground truth marker positions from JSON
    std::ifstream markers_file(test_data_dir + "/marker_positions_0.json");
    REQUIRE(markers_file.is_open());
    json markers_json;
    markers_file >> markers_json;
    markers_file.close();

    // Open CSV file for debug output
    std::ofstream csv_file(test_data_dir + "/marker_positions_comparison.csv");
    csv_file << "frame,marker_name,computed_x,computed_y,computed_z,gt_x,gt_y,gt_z,error\n";
    csv_file << std::fixed << std::setprecision(6);

    auto const& frames = states_json["frames"];
    auto const& marker_frames = markers_json["frames"];

    REQUIRE(frames.size() == marker_frames.size());
    int num_frames = std::min(static_cast<int>(frames.size()), 3);  // Test first 3 frames

    INFO("Testing " << num_frames << " frames");

    double total_rmse = 0.0;
    int total_markers = 0;

    for (int i = 0; i < num_frames; ++i) {
        auto const& frame = frames[i];
        auto const& marker_frame = marker_frames[i];

        INFO("Processing frame " << i);

        // Extract joint angles from state
        auto const& joint_angles_map = frame["joint_angles"];

        // Build configuration vector directly from JSON
        // Now that we use vector storage, joints are in YAML file order
        Eigen::VectorXd q = Eigen::VectorXd::Zero(model.nq);
        int idx = 0;

        // Root position and orientation
        q[idx++] = joint_angles_map["root_position_x"].get<double>();
        q[idx++] = joint_angles_map["root_position_y"].get<double>();
        q[idx++] = joint_angles_map["root_position_z"].get<double>();
        q[idx++] = joint_angles_map["root_quaternion_x"].get<double>();  // Pinocchio order: x,y,z,w
        q[idx++] = joint_angles_map["root_quaternion_y"].get<double>();
        q[idx++] = joint_angles_map["root_quaternion_z"].get<double>();
        q[idx++] = joint_angles_map["root_quaternion_w"].get<double>();

        // Other joints - iterate through skeleton in YAML file order
        for (auto const& joint : skeleton.joints()) {
            if (!joint.parent_index.has_value())
                continue;  // Skip root

            if (joint.type == JointType::SPHERICAL) {
                // Get 3 angles and convert to quaternion via angle-axis
                std::string key0 = "joint_" + joint.name + "_angle_0";
                std::string key1 = "joint_" + joint.name + "_angle_1";
                std::string key2 = "joint_" + joint.name + "_angle_2";

                double rx = joint_angles_map.value(key0, 0.0);
                double ry = joint_angles_map.value(key1, 0.0);
                double rz = joint_angles_map.value(key2, 0.0);

                Eigen::Vector3d v(rx, ry, rz);
                double angle = v.norm();
                Eigen::Quaterniond quat;
                if (angle == 0.0) {
                    quat = Eigen::Quaterniond::Identity();
                } else {
                    Eigen::Vector3d axis = v / angle;
                    quat = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
                }

                // Debug output for forearm joints
                if (i == 0 && joint.name.find("forearm") != std::string::npos) {
                    std::cout << "Joint " << joint.name << " axis-angle: [" << rx << ", " << ry
                              << ", " << rz << "]" << std::endl;
                    std::cout << "  -> quaternion: [" << quat.x() << ", " << quat.y() << ", "
                              << quat.z() << ", " << quat.w() << "]" << std::endl;
                }

                q[idx++] = quat.x();
                q[idx++] = quat.y();
                q[idx++] = quat.z();
                q[idx++] = quat.w();

            } else if (joint.type == JointType::REVOLUTE) {
                std::string key = "joint_" + joint.name + "_angle_0";
                q[idx++] = joint_angles_map.value(key, 0.0);
            }
        }

        if (i == 0) {
            WARN("Built config vector with " << idx << " elements (expected " << model.nq << ")");
            WARN("First 15 elements of q: " << q.head(15).transpose());
        }

        // Compute FK
        auto computed_positions = fk.compute(q);

        // Compare with ground truth
        auto const& gt_markers = marker_frame["markers"];
        double frame_squared_error = 0.0;
        int frame_marker_count = 0;

        if (i == 0) {
            std::cout << "Frame 0: comparing " << gt_markers.size() << " markers" << std::endl;
        }

        for (auto it = gt_markers.begin(); it != gt_markers.end(); ++it) {
            std::string marker_name = it.key();
            CAPTURE(marker_name);
            std::cout << "Checking marker: " << marker_name << std::endl;
            auto const& marker_data = it.value();

            if (!marker_data["is_valid"].get<bool>())
                continue;

            auto const& gt_pos_arr = marker_data["position_3d"];
            Eigen::Vector3d gt_pos(gt_pos_arr[0].get<double>(), gt_pos_arr[1].get<double>(),
                                   gt_pos_arr[2].get<double>());

            auto comp_it = computed_positions.find(marker_name);
            if (comp_it == computed_positions.end()) {
                WARN("Marker " << marker_name << " not found in computed positions");
                continue;
            }

            Eigen::Vector3d diff = comp_it->second - gt_pos;
            double error = diff.norm();
            frame_squared_error += diff.squaredNorm();
            frame_marker_count++;

            // Write to CSV file
            csv_file << i << "," << marker_name << "," << comp_it->second.x() << ","
                     << comp_it->second.y() << "," << comp_it->second.z() << "," << gt_pos.x()
                     << "," << gt_pos.y() << "," << gt_pos.z() << "," << error << "\n";

            // Debug: print marker errors
            if (error > 0.01) {
                std::cout << "Large error for marker " << marker_name << ": " << error << " m"
                          << std::endl;
                std::cout << "  Computed: [" << comp_it->second.transpose() << "]" << std::endl;
                std::cout << "  GT:       [" << gt_pos.transpose() << "]" << std::endl;
            } else {
                std::cout << "Marker " << marker_name << " ok (error: " << error << " m)"
                          << std::endl;
            }

            // Check individual marker error is small (skip first 3 to see more)
            // if (frame_marker_count > 3) {
            //     REQUIRE(error < 0.01);  // Less than 1cm error per marker
            // }
        }

        if (frame_marker_count > 0) {
            double frame_rmse = std::sqrt(frame_squared_error / frame_marker_count);
            total_rmse += frame_squared_error;
            total_markers += frame_marker_count;

            INFO("Frame " << i << " RMSE: " << frame_rmse << " m (" << frame_marker_count
                          << " markers)");
            // REQUIRE(frame_rmse < 0.005);  // Less than 5mm RMSE per frame
        }
    }

    // Close CSV file
    csv_file.close();
    INFO("Wrote comparison data to " << test_data_dir << "/marker_positions_comparison.csv");

    // Check overall RMSE
    double overall_rmse = std::sqrt(total_rmse / total_markers);
    INFO("Overall RMSE across " << num_frames << " frames: " << overall_rmse << " m");
    REQUIRE(overall_rmse < 0.005);  // Less than 5mm overall RMSE
}

// ===========================================================================
// Phase 3d: Subtree Model Tests
// ===========================================================================

/// Build the shared hand skeleton fixture:
///
///   pelvis (SPHERICAL, group="main")    ← root (no parent)
///     └── upper_arm.R (SPHERICAL, group="main")
///          └── forearm.R (REVOLUTE, group="main")
///               └── wrist.R (FIXED, group="main")   ← freeflyer anchor for HandR
///                    └── palm.R (SPHERICAL, group="HandR")
///                         ├── finger1.R (REVOLUTE, group="HandR")
///                         └── finger2.R (REVOLUTE, group="HandR")
///
/// Markers:
///   MRK-body   on pelvis    (group main)
///   MRK-palm   on palm.R    (group HandR)
///   MRK-tip1   on finger1.R (group HandR)
static Skeleton make_hand_skeleton() {
    Skeleton skel;

    uint32_t pelvis =
        skel.add_joint("pelvis", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero());
    uint32_t upper_arm =
        skel.add_joint("upper_arm.R", pelvis, JointType::SPHERICAL, Eigen::Vector3d(0.2, 0, 0));
    uint32_t forearm =
        skel.add_joint("forearm.R", upper_arm, JointType::REVOLUTE, Eigen::Vector3d(0.3, 0, 0));
    uint32_t wrist =
        skel.add_joint("wrist.R", forearm, JointType::FIXED, Eigen::Vector3d(0.25, 0, 0));
    uint32_t palm =
        skel.add_joint("palm.R", wrist, JointType::SPHERICAL, Eigen::Vector3d(0.05, 0, 0));
    uint32_t f1 =
        skel.add_joint("finger1.R", palm, JointType::REVOLUTE, Eigen::Vector3d(0.04, 0.01, 0));
    skel.add_joint("finger2.R", palm, JointType::REVOLUTE, Eigen::Vector3d(0.04, -0.01, 0));

    skel.register_group("main", {"pelvis", "upper_arm.R", "forearm.R", "wrist.R"}, {});
    skel.register_group("HandR", {"palm.R", "finger1.R", "finger2.R"}, {});

    skel.add_marker("MRK-body", pelvis, Eigen::Vector3d(0, 0, 0.1));
    skel.add_marker("MRK-palm", palm, Eigen::Vector3d(0, 0, 0.01));
    skel.add_marker("MRK-tip1", f1, Eigen::Vector3d(0.03, 0, 0));

    return skel;
}

TEST_CASE("Subtree model: joint count and velocity DOF", "[subtree_model]") {
    Skeleton skel = make_hand_skeleton();
    pinocchio::Model model;

    REQUIRE_NOTHROW(PinocchioModelBuilder::build_subtree_model(skel, "wrist.R", {"HandR"}, model));

    // Universe (0) + wrist.R freeflyer (1) + palm.R (2) + finger1.R (3) + finger2.R (4)
    REQUIRE(model.njoints == 5);
    // nv: 6 (freeflyer) + 3 (palm spherical) + 1 (finger1) + 1 (finger2) = 11
    REQUIRE(model.nv == 11);
    // nq: 7 (freeflyer) + 4 (palm spherical) + 1 + 1 = 13
    REQUIRE(model.nq == 13);
}

TEST_CASE("Subtree model: freeflyer placed at identity", "[subtree_model]") {
    Skeleton skel = make_hand_skeleton();
    pinocchio::Model model;
    PinocchioModelBuilder::build_subtree_model(skel, "wrist.R", {"HandR"}, model);

    // Joint index 1 = wrist.R (first after universe)
    // jointPlacements[1] is the placement of the joint relative to its parent (universe)
    auto const& placement = model.jointPlacements[1];
    REQUIRE(placement.translation().isZero(1e-10));
    REQUIRE(placement.rotation().isApprox(Eigen::Matrix3d::Identity(), 1e-10));
}

TEST_CASE("Subtree model: child joint offsets preserved", "[subtree_model]") {
    Skeleton skel = make_hand_skeleton();
    pinocchio::Model model;
    PinocchioModelBuilder::build_subtree_model(skel, "wrist.R", {"HandR"}, model);

    // palm.R should be joint index 2; its offset in skeleton is (0.05, 0, 0)
    // In the subtree model palm.R is a direct child of the wrist.R freeflyer,
    // but wrist.R is FIXED so palm.R's placement carries wrist.R's offset (0.25, 0, 0)
    // plus palm.R's own offset (0.05, 0, 0) = (0.30, 0, 0) -- wait,
    // actually FIXED wrist.R maps to freeflyer (parent_pin_id), and palm.R's placement
    // is palm.R's own offset (0.05, 0, 0) relative to wrist.R (the pinocchio freeflyer).
    // palm.R is added with parent = freeflyer pin id (wrist.R) and placement = palm.R.offset
    REQUIRE((model.existFrame("wrist.R") || model.names[1] == "wrist.R"));
    pinocchio::JointIndex palm_id = model.getJointId("palm.R");
    auto const& palm_placement = model.jointPlacements[palm_id];
    // palm.R.offset = (0.05, 0, 0)
    CHECK_THAT(palm_placement.translation()[0], Catch::Matchers::WithinAbs(0.05, 1e-10));
    CHECK_THAT(palm_placement.translation()[1], Catch::Matchers::WithinAbs(0.0, 1e-10));
    CHECK_THAT(palm_placement.translation()[2], Catch::Matchers::WithinAbs(0.0, 1e-10));
}

TEST_CASE("Subtree model: only subtree markers included", "[subtree_model]") {
    Skeleton skel = make_hand_skeleton();
    pinocchio::Model model;
    PinocchioModelBuilder::build_subtree_model(skel, "wrist.R", {"HandR"}, model);
    auto layout = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"HandR"});

    auto marker_map = PinocchioModelBuilder::build_subtree_marker_frame_map(model, *layout);

    // MRK-palm and MRK-tip1 are in HandR group; MRK-body is in main group → excluded
    REQUIRE(marker_map.size() == 2);
    REQUIRE(marker_map.count("MRK-palm") > 0);
    REQUIRE(marker_map.count("MRK-tip1") > 0);
    REQUIRE(marker_map.count("MRK-body") == 0);
}

TEST_CASE("Subtree model: FK at identity state gives correct marker position", "[subtree_model]") {
    Skeleton skel = make_hand_skeleton();
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_subtree_model(skel, "wrist.R", {"HandR"}, model);
    data = pinocchio::Data(model);
    auto layout = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"HandR"});

    auto marker_map = PinocchioModelBuilder::build_subtree_marker_frame_map(model, *layout);

    // Identity state: root at origin, all joint angles zero
    int n_angles = static_cast<int>(layout->total_storage_dof_count());
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(),
                Eigen::VectorXd::Zero(n_angles), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                Eigen::VectorXd::Zero(n_angles));

    ForwardKinematics fk(model, data, marker_map, layout);
    auto positions = fk.compute(state);

    REQUIRE(positions.count("MRK-palm") > 0);
    REQUIRE(positions.count("MRK-tip1") > 0);

    // At identity (root at origin, zero angles):
    // palm.R is at offset (0.05, 0, 0) from wrist.R, plus marker local_pos (0,0,0.01)
    // wrist.R is the freeflyer at origin → palm.R world pos = (0.05, 0, 0)
    // MRK-palm = (0.05, 0, 0) + (0, 0, 0.01) = (0.05, 0, 0.01)
    auto const& palm_pos = positions.at("MRK-palm");
    CHECK_THAT(palm_pos[0], Catch::Matchers::WithinAbs(0.05, 1e-6));
    CHECK_THAT(palm_pos[1], Catch::Matchers::WithinAbs(0.0, 1e-6));
    CHECK_THAT(palm_pos[2], Catch::Matchers::WithinAbs(0.01, 1e-6));
}

TEST_CASE("Subtree model: FK with injected root transform", "[subtree_model]") {
    Skeleton skel = make_hand_skeleton();
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_subtree_model(skel, "wrist.R", {"HandR"}, model);
    data = pinocchio::Data(model);
    auto layout = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"HandR"});

    auto marker_map = PinocchioModelBuilder::build_subtree_marker_frame_map(model, *layout);

    // Inject root offset: wrist.R world position = (1, 2, 3), identity orientation
    Eigen::Vector3d root_pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond root_ori = Eigen::Quaterniond::Identity();
    int n_angles = static_cast<int>(layout->total_storage_dof_count());
    State state(root_pos, root_ori, Eigen::VectorXd::Zero(n_angles), Eigen::Vector3d::Zero(),
                Eigen::Vector3d::Zero(), Eigen::VectorXd::Zero(n_angles));

    ForwardKinematics fk(model, data, marker_map, layout);
    auto positions = fk.compute(state);
    auto const& palm_pos = positions.at("MRK-palm");
    CHECK_THAT(palm_pos[0], Catch::Matchers::WithinAbs(1.05, 1e-6));
    CHECK_THAT(palm_pos[1], Catch::Matchers::WithinAbs(2.0, 1e-6));
    CHECK_THAT(palm_pos[2], Catch::Matchers::WithinAbs(3.01, 1e-6));
}

TEST_CASE("Subtree model: connectivity assertion throws for non-descendant group",
          "[subtree_model]") {
    Skeleton skel = make_hand_skeleton();
    pinocchio::Model model;

    // "upper_arm.R" is not a descendant of "wrist.R" — should throw
    REQUIRE_THROWS_AS(PinocchioModelBuilder::build_subtree_model(skel, "wrist.R", {"main"}, model),
                      std::invalid_argument);
}

TEST_CASE("Subtree model: freeflyer not in skeleton throws", "[subtree_model]") {
    Skeleton skel = make_hand_skeleton();
    pinocchio::Model model;
    REQUIRE_THROWS_AS(
        PinocchioModelBuilder::build_subtree_model(skel, "nonexistent_joint", {"HandR"}, model),
        std::invalid_argument);
}

TEST_CASE("Subtree model: state_to_config layout overload", "[subtree_model]") {
    Skeleton skel = make_hand_skeleton();
    auto layout = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"HandR"});

    // palm.R(3) + finger1.R(1) + finger2.R(1) = 5 storage DOFs
    REQUIRE(layout->total_storage_dof_count() == 5);

    Eigen::Vector3d root_pos(1.0, 0.0, 0.0);
    Eigen::Quaterniond root_ori = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles = Eigen::VectorXd::Zero(5);
    angles[3] = 0.5;  // finger1.R angle

    State state(root_pos, root_ori, angles, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                Eigen::VectorXd::Zero(5));

    Eigen::VectorXd q = ForwardKinematics::state_to_config(state, *layout);

    // nq: 7 (freeflyer) + 4 (palm spherical) + 1 (finger1) + 1 (finger2) = 13
    REQUIRE(q.size() == 13);

    // Root position
    CHECK_THAT(q[0], Catch::Matchers::WithinAbs(1.0, 1e-10));
    CHECK_THAT(q[1], Catch::Matchers::WithinAbs(0.0, 1e-10));
    CHECK_THAT(q[2], Catch::Matchers::WithinAbs(0.0, 1e-10));
    // Root quaternion (identity → x=0, y=0, z=0, w=1)
    CHECK_THAT(q[3], Catch::Matchers::WithinAbs(0.0, 1e-10));
    CHECK_THAT(q[4], Catch::Matchers::WithinAbs(0.0, 1e-10));
    CHECK_THAT(q[5], Catch::Matchers::WithinAbs(0.0, 1e-10));
    CHECK_THAT(q[6], Catch::Matchers::WithinAbs(1.0, 1e-10));
    // palm.R zero angles → identity quaternion (q[7..10])
    CHECK_THAT(q[7], Catch::Matchers::WithinAbs(0.0, 1e-10));   // x
    CHECK_THAT(q[8], Catch::Matchers::WithinAbs(0.0, 1e-10));   // y
    CHECK_THAT(q[9], Catch::Matchers::WithinAbs(0.0, 1e-10));   // z
    CHECK_THAT(q[10], Catch::Matchers::WithinAbs(1.0, 1e-10));  // w
    // finger1.R angle
    CHECK_THAT(q[11], Catch::Matchers::WithinAbs(0.5, 1e-10));
    // finger2.R angle = 0
    CHECK_THAT(q[12], Catch::Matchers::WithinAbs(0.0, 1e-10));
}
// ===========================================================================
// Phase 3f: world_transform() Tests
// ===========================================================================

/// Helper: build full-skeleton FK from make_hand_skeleton() and compute at
/// the given root position with zero joint angles.  Caller owns model/data.
///
/// Uses from_groups(skel, {"main"}) — not from_full_skeleton — because the full
/// pinocchio model stops at the FIXED wrist.R joint (add_joint_recursive returns
/// immediately on FIXED joints), so HandR joints (palm.R, fingers) are absent from
/// the model.  The "main" group layout covers exactly the joints that ARE in the
/// model: upper_arm.R (SPHERICAL) and forearm.R (REVOLUTE).
static ForwardKinematics make_full_fk(Skeleton const& skel, pinocchio::Model& model,
                                      pinocchio::Data& data,
                                      std::map<std::string, pinocchio::FrameIndex>& marker_map,
                                      std::shared_ptr<const SkeletonLayout>& layout) {
    PinocchioModelBuilder::build_model_and_data(skel, model, data);
    marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skel);
    layout = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"main"});
    return ForwardKinematics(model, data, marker_map, layout);
}

TEST_CASE("world_transform: non-fixed joints in full-skeleton FK", "[world_transform]") {
    Skeleton skel = make_hand_skeleton();
    pinocchio::Model model;
    pinocchio::Data data;
    std::map<std::string, pinocchio::FrameIndex> marker_map;
    std::shared_ptr<const SkeletonLayout> layout;
    ForwardKinematics fk = make_full_fk(skel, model, data, marker_map, layout);

    // zero-angle state
    int n = static_cast<int>(layout->total_storage_dof_count());
    State state_id(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(),
                   Eigen::VectorXd::Zero(n), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                   Eigen::VectorXd::Zero(n));
    fk.compute(state_id);

    // Offsets (all along X at identity orientation):
    //   pelvis root at (0,0,0)
    //   upper_arm.R: offset (0.2, 0, 0)
    //   forearm.R: offset (0.3, 0, 0) from upper_arm.R → world (0.5, 0, 0)

    SECTION("upper_arm.R position at identity") {
        auto [pos, ori] = fk.world_transform("upper_arm.R");
        CHECK_THAT(pos[0], Catch::Matchers::WithinAbs(0.2, 1e-6));
        CHECK_THAT(pos[1], Catch::Matchers::WithinAbs(0.0, 1e-6));
        CHECK_THAT(pos[2], Catch::Matchers::WithinAbs(0.0, 1e-6));
    }

    SECTION("forearm.R position at identity") {
        auto [pos, ori] = fk.world_transform("forearm.R");
        CHECK_THAT(pos[0], Catch::Matchers::WithinAbs(0.5, 1e-6));
        CHECK_THAT(pos[1], Catch::Matchers::WithinAbs(0.0, 1e-6));
        CHECK_THAT(pos[2], Catch::Matchers::WithinAbs(0.0, 1e-6));
    }

    SECTION("non-zero root shifts all joints") {
        State state_shifted(Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Quaterniond::Identity(),
                            Eigen::VectorXd::Zero(n), Eigen::Vector3d::Zero(),
                            Eigen::Vector3d::Zero(), Eigen::VectorXd::Zero(n));
        fk.compute(state_shifted);
        auto [pos, ori] = fk.world_transform("forearm.R");
        CHECK_THAT(pos[0], Catch::Matchers::WithinAbs(1.5, 1e-6));
        CHECK_THAT(pos[1], Catch::Matchers::WithinAbs(0.0, 1e-6));
        CHECK_THAT(pos[2], Catch::Matchers::WithinAbs(0.0, 1e-6));
    }

    SECTION("FIXED joint (wrist.R) throws out_of_range") {
        // wrist.R is FIXED in the full-skeleton model so it has no pinocchio joint slot
        REQUIRE_THROWS_AS(fk.world_transform("wrist.R"), std::out_of_range);
    }

    SECTION("unknown joint name throws out_of_range") {
        REQUIRE_THROWS_AS(fk.world_transform("nonexistent"), std::out_of_range);
    }
}

TEST_CASE("world_transform: subtree FK freeflyer and child joint", "[world_transform]") {
    Skeleton skel = make_hand_skeleton();
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_subtree_model(skel, "wrist.R", {"HandR"}, model);
    data = pinocchio::Data(model);
    auto layout = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"HandR"});

    auto marker_map = PinocchioModelBuilder::build_subtree_marker_frame_map(model, *layout);
    ForwardKinematics fk(model, data, marker_map, layout);

    // Inject freeflyer root at (1, 2, 3)
    Eigen::Vector3d root_pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond root_ori = Eigen::Quaterniond::Identity();
    int n = static_cast<int>(layout->total_storage_dof_count());
    State state(root_pos, root_ori, Eigen::VectorXd::Zero(n), Eigen::Vector3d::Zero(),
                Eigen::Vector3d::Zero(), Eigen::VectorXd::Zero(n));
    fk.compute(state);

    SECTION("wrist.R freeflyer returns injected root transform") {
        // In the subtree model wrist.R IS the pinocchio freeflyer (joint index 1)
        auto [pos, ori] = fk.world_transform("wrist.R");
        CHECK_THAT(pos[0], Catch::Matchers::WithinAbs(1.0, 1e-6));
        CHECK_THAT(pos[1], Catch::Matchers::WithinAbs(2.0, 1e-6));
        CHECK_THAT(pos[2], Catch::Matchers::WithinAbs(3.0, 1e-6));
    }

    SECTION("palm.R world position = root + palm offset") {
        // palm.R is placed at (0.05, 0, 0) from the wrist.R freeflyer
        auto [pos, ori] = fk.world_transform("palm.R");
        CHECK_THAT(pos[0], Catch::Matchers::WithinAbs(1.05, 1e-6));
        CHECK_THAT(pos[1], Catch::Matchers::WithinAbs(2.0, 1e-6));
        CHECK_THAT(pos[2], Catch::Matchers::WithinAbs(3.0, 1e-6));
    }

    SECTION("orientation matches injected root orientation") {
        auto [pos, ori] = fk.world_transform("wrist.R");
        Eigen::Quaterniond expected = Eigen::Quaterniond::Identity();
        CHECK_THAT(ori.angularDistance(expected), Catch::Matchers::WithinAbs(0.0, 1e-6));
    }
}
