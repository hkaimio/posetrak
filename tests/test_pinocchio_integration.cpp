#include <nlohmann/json.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
                                             Eigen::Vector3d::Zero(), "", Eigen::Vector3d::Zero());

    // Child joint (spine)
    uint32_t spine_idx =
        skeleton.add_joint("spine", pelvis_idx, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 0.1), "",
                           Eigen::Vector3d::Zero());

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
                                             Eigen::Vector3d::Zero(), "", Eigen::Vector3d::Zero());

    uint32_t spine_idx =
        skeleton.add_joint("spine", pelvis_idx, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 0.1), "",
                           Eigen::Vector3d::Zero());

    skeleton.add_marker("head_marker", spine_idx, Eigen::Vector3d(0, 0, 0.2), std::nullopt);

    // Build Pinocchio model
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);

    auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);

    // Create FK computer
    ForwardKinematics fk(model, data, marker_map);

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
        joint_angles << 0.5;                                  // spine angle
        Eigen::VectorXd root_vel = Eigen::VectorXd::Zero(3);  // 3D linear velocity
        Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(1);

        State state(root_pos, root_quat, joint_angles, root_vel, joint_vels);

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
                                             Eigen::Vector3d::Zero(), "", Eigen::Vector3d::Zero());

    uint32_t shoulder_idx =
        skeleton.add_joint("shoulder", pelvis_idx, JointType::SPHERICAL, Eigen::Vector3d(0.2, 0, 0),
                           "", Eigen::Vector3d::Zero());

    skeleton.add_marker("hand", shoulder_idx, Eigen::Vector3d(0.3, 0, 0), std::nullopt);
    skeleton.add_marker("shoulder_marker", shoulder_idx, Eigen::Vector3d(0, 0, 0.1), std::nullopt);

    // Build model
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);

    auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
    ForwardKinematics fk(model, data, marker_map);

    SECTION("state_to_config converts spherical joint correctly") {
        // State with spherical joint angles
        Eigen::Vector3d root_pos = Eigen::Vector3d::Zero();
        Eigen::Quaterniond root_quat = Eigen::Quaterniond::Identity();
        Eigen::VectorXd joint_angles(3);  // 3 DOF for spherical
        joint_angles << 0.1, 0.2, 0.3;    // rotation angles

        Eigen::VectorXd root_vel = Eigen::VectorXd::Zero(3);  // 3D linear velocity
        Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(3);

        State state(root_pos, root_quat, joint_angles, root_vel, joint_vels);

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
    ForwardKinematics fk(model, data, marker_map);

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
