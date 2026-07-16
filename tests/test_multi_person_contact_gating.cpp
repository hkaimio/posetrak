/**
 * Stage 2 unit tests for the cross-person relative observations plan
 * (docs/roadmap/features/error-improvements/phase5-cross-person-plan.md):
 * contact gating (update_contact_pairs) and anchor-observation construction
 * (build_cross_person_anchors). Both are pure functions -- no Tracker/FK/DB
 * access -- so they're tested directly against synthetic marker positions
 * and observations, per the plan's Stage 2 verification list.
 */
#include <posetrak/core/skeleton.hpp>
#include <posetrak/tracking/multi_person_tracker.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace posetrak;

namespace {

Skeleton make_skeleton_with_markers(std::vector<std::string> const& marker_names) {
    Skeleton s;
    s.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    for (auto const& name : marker_names) {
        s.add_marker(name, 0, Eigen::Vector3d::Zero());
    }
    return s;
}

Camera make_camera(int id, Eigen::Vector3d const& position) {
    Intrinsics intr;
    intr.fx = 1000.0;
    intr.fy = 1000.0;
    intr.cx = 640.0;
    intr.cy = 360.0;
    intr.width = 1280;
    intr.height = 720;
    intr.model = Intrinsics::DistortionModel::BrownConrady;
    intr.distortion_coeffs = {0, 0, 0, 0, 0};

    // Identity orientation: camera-frame == world-frame, so depth is world Z.
    // *position* should be behind (smaller Z than) whatever it's meant to see.
    Extrinsics extr;
    extr.position = position;
    extr.orientation = Eigen::Quaterniond::Identity();
    return Camera(id, "cam" + std::to_string(id), intr, extr);
}

Observation make_detection(int camera_id, int marker_id, Eigen::Vector2d const& pos,
                           double confidence, double crop_scale = 1.0) {
    Observation o;
    o.camera_id = camera_id;
    o.marker_id = marker_id;
    o.frame_idx = 1;
    o.timestamp = 1.0;
    o.position = pos;
    o.position_distorted = pos;
    o.confidence = confidence;
    o.crop_scale = crop_scale;
    return o;
}

}  // namespace

// ---------------------------------------------------------------------------
// update_contact_pairs (gate levels 1-2 + hysteresis)
// ---------------------------------------------------------------------------

TEST_CASE("update_contact_pairs: out-of-range persons produce no active pairs",
          "[multi_person][gating]") {
    PersonGatingInput a;
    a.cross_person_max_world_mm = 100.0;  // 10 cm
    a.marker_world_positions["m0"] = Eigen::Vector3d(0, 0, 0);
    a.marker_name_to_id["m0"] = 0;

    PersonGatingInput b;
    b.cross_person_max_world_mm = 100.0;
    b.marker_world_positions["m0"] = Eigen::Vector3d(5, 0, 0);  // 5 m away
    b.marker_name_to_id["m0"] = 0;

    std::map<ContactMarkerPair, double> active;
    update_contact_pairs({a, b}, active);

    REQUIRE(active.empty());
}

TEST_CASE("update_contact_pairs: within-threshold persons produce an active pair",
          "[multi_person][gating]") {
    PersonGatingInput a;
    a.cross_person_max_world_mm = 100.0;  // 10 cm
    a.marker_world_positions["m0"] = Eigen::Vector3d(0, 0, 0);
    a.marker_name_to_id["m0"] = 0;

    PersonGatingInput b;
    b.cross_person_max_world_mm = 100.0;
    b.marker_world_positions["m0"] = Eigen::Vector3d(0.05, 0, 0);  // 5 cm away
    b.marker_name_to_id["m0"] = 0;

    std::map<ContactMarkerPair, double> active;
    update_contact_pairs({a, b}, active);

    REQUIRE(active.size() == 1);
    ContactMarkerPair key{0, 0, 1, 0};
    auto it = active.find(key);
    REQUIRE(it != active.end());
    REQUIRE(it->second == Catch::Approx(0.05));
}

TEST_CASE("update_contact_pairs: disabled for a person (0 threshold) never gates",
          "[multi_person][gating]") {
    PersonGatingInput a;
    a.cross_person_max_world_mm = 0.0;  // disabled
    a.marker_world_positions["m0"] = Eigen::Vector3d(0, 0, 0);
    a.marker_name_to_id["m0"] = 0;

    PersonGatingInput b;
    b.cross_person_max_world_mm = 100.0;
    b.marker_world_positions["m0"] = Eigen::Vector3d(0.01, 0, 0);
    b.marker_name_to_id["m0"] = 0;

    std::map<ContactMarkerPair, double> active;
    update_contact_pairs({a, b}, active);

    REQUIRE(active.empty());
}

TEST_CASE(
    "update_contact_pairs: three persons -- all gated pairs produce entries, no "
    "hardcoded two-person assumption",
    "[multi_person][gating]") {
    // Person 0 and 1 are close; person 2 is far from both.
    PersonGatingInput p0, p1, p2;
    p0.cross_person_max_world_mm = p1.cross_person_max_world_mm = p2.cross_person_max_world_mm =
        100.0;
    p0.marker_world_positions["m0"] = Eigen::Vector3d(0, 0, 0);
    p0.marker_name_to_id["m0"] = 0;
    p1.marker_world_positions["m0"] = Eigen::Vector3d(0.05, 0, 0);
    p1.marker_name_to_id["m0"] = 0;
    p2.marker_world_positions["m0"] = Eigen::Vector3d(10, 0, 0);
    p2.marker_name_to_id["m0"] = 0;

    std::map<ContactMarkerPair, double> active;
    update_contact_pairs({p0, p1, p2}, active);

    REQUIRE(active.size() == 1);
    REQUIRE(active.count(ContactMarkerPair{0, 0, 1, 0}) == 1);
    REQUIRE(active.count(ContactMarkerPair{0, 0, 2, 0}) == 0);
    REQUIRE(active.count(ContactMarkerPair{1, 0, 2, 0}) == 0);
}

TEST_CASE("update_contact_pairs: hysteresis keeps a pair active between T and 1.2*T",
          "[multi_person][gating]") {
    PersonGatingInput a;
    a.cross_person_max_world_mm = 100.0;  // T = 0.1 m
    a.marker_name_to_id["m0"] = 0;
    PersonGatingInput b = a;

    std::map<ContactMarkerPair, double> active;

    // Frame 1: well inside T -> becomes active.
    a.marker_world_positions["m0"] = Eigen::Vector3d(0, 0, 0);
    b.marker_world_positions["m0"] = Eigen::Vector3d(0.05, 0, 0);
    update_contact_pairs({a, b}, active);
    REQUIRE(active.size() == 1);

    // Frame 2: between T (0.10) and 1.2*T (0.12) -- must NOT flicker off.
    b.marker_world_positions["m0"] = Eigen::Vector3d(0.11, 0, 0);
    update_contact_pairs({a, b}, active);
    REQUIRE(active.size() == 1);

    // Frame 3: back under T -- still active, distance updates.
    b.marker_world_positions["m0"] = Eigen::Vector3d(0.06, 0, 0);
    update_contact_pairs({a, b}, active);
    REQUIRE(active.size() == 1);
    REQUIRE(active.begin()->second == Catch::Approx(0.06));

    // Frame 4: beyond 1.2*T -- now exits.
    b.marker_world_positions["m0"] = Eigen::Vector3d(0.13, 0, 0);
    update_contact_pairs({a, b}, active);
    REQUIRE(active.empty());
}

TEST_CASE(
    "update_contact_pairs: a distance just over T alone (not over 1.2*T) would never "
    "have entered -- confirms enter/exit thresholds actually differ",
    "[multi_person][gating]") {
    PersonGatingInput a;
    a.cross_person_max_world_mm = 100.0;
    a.marker_name_to_id["m0"] = 0;
    a.marker_world_positions["m0"] = Eigen::Vector3d(0, 0, 0);
    PersonGatingInput b = a;
    b.marker_world_positions["m0"] = Eigen::Vector3d(0.105, 0, 0);  // just over T=0.10

    std::map<ContactMarkerPair, double> active;
    update_contact_pairs({a, b}, active);
    REQUIRE(active.empty());  // never entered -- enter threshold is strict T, not 1.2*T
}

// ---------------------------------------------------------------------------
// build_cross_person_anchors (gate level 3 + anchor construction)
// ---------------------------------------------------------------------------

TEST_CASE("build_cross_person_anchors: produces a correctly composed anchor observation",
          "[multi_person][anchors]") {
    Skeleton other_skeleton = make_skeleton_with_markers({"wrist"});

    std::map<ContactMarkerPair, double> active_pairs;
    active_pairs[ContactMarkerPair{0, 0, 1, 0}] = 0.05;  // person 0 marker 0 <-> person 1 marker 0

    std::vector<Observation> my_frame_obs = {
        make_detection(/*camera=*/0, /*marker=*/0, Eigen::Vector2d(700, 400), 0.9, 1.2)};
    std::vector<Observation> other_frame_obs = {
        make_detection(/*camera=*/0, /*marker=*/0, Eigen::Vector2d(680, 390), 0.8, 1.5)};

    // Other's anchor marker sits directly in front of the camera so projection succeeds.
    std::map<std::string, Eigen::Vector3d> other_anchor_positions = {
        {"wrist", Eigen::Vector3d(1.0, 0.0, 0.0)}};

    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, make_camera(0, Eigen::Vector3d(0.0, 0.0, -2.0)));

    auto anchors = build_cross_person_anchors(
        /*my_idx=*/0, /*other_idx=*/1, active_pairs, my_frame_obs, other_frame_obs,
        other_anchor_positions, other_skeleton, cameras,
        /*my_min_confidence=*/0.5, /*other_min_confidence=*/0.5, /*max_n=*/10,
        /*my_pose_noise_std=*/2.0, /*other_pose_noise_std=*/3.0,
        /*anchor_noise_std_floor=*/5.0, /*frame_idx=*/7, /*timestamp=*/1.5);

    REQUIRE(anchors.size() == 1);
    auto const& obs = anchors[0];
    REQUIRE(obs.camera_id == 0);
    REQUIRE(obs.marker_id == 0);  // my_frame_obs's marker id
    REQUIRE(obs.frame_idx == 7);
    REQUIRE(obs.timestamp == Catch::Approx(1.5));
    REQUIRE(obs.mode == MeasurementMode::PAIR_DIFF);
    REQUIRE(obs.position.x() == Catch::Approx(700.0 - 680.0));
    REQUIRE(obs.position.y() == Catch::Approx(400.0 - 390.0));
    REQUIRE(obs.confidence == Catch::Approx(0.8));  // min(0.9, 0.8)
    REQUIRE(obs.crop_scale == Catch::Approx(1.2));  // mine's crop_scale
    REQUIRE_FALSE(obs.force_inlier);
    REQUIRE(obs.anchor_position.has_value());

    // Noise composition: sqrt((2.0*1.2)^2 + (3.0*1.5)^2 + 5.0^2)
    double const expected =
        std::sqrt((2.0 * 1.2) * (2.0 * 1.2) + (3.0 * 1.5) * (3.0 * 1.5) + 5.0 * 5.0);
    REQUIRE(obs.noise_std_override == Catch::Approx(expected));
}

TEST_CASE("build_cross_person_anchors: no active pairs -> no observations",
          "[multi_person][anchors]") {
    Skeleton other_skeleton = make_skeleton_with_markers({"wrist"});
    std::map<ContactMarkerPair, double> active_pairs;  // empty
    std::vector<Observation> my_frame_obs = {make_detection(0, 0, {700, 400}, 0.9)};
    std::vector<Observation> other_frame_obs = {make_detection(0, 0, {680, 390}, 0.8)};
    std::map<std::string, Eigen::Vector3d> other_anchor_positions = {
        {"wrist", Eigen::Vector3d(1.0, 0.0, 0.0)}};
    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, make_camera(0, Eigen::Vector3d(0.0, 0.0, -2.0)));

    auto anchors = build_cross_person_anchors(0, 1, active_pairs, my_frame_obs, other_frame_obs,
                                              other_anchor_positions, other_skeleton, cameras, 0.5,
                                              0.5, 10, 2.0, 3.0, 5.0, 1, 1.0);
    REQUIRE(anchors.empty());
}

TEST_CASE("build_cross_person_anchors: low-confidence detections are excluded",
          "[multi_person][anchors]") {
    Skeleton other_skeleton = make_skeleton_with_markers({"wrist"});
    std::map<ContactMarkerPair, double> active_pairs;
    active_pairs[ContactMarkerPair{0, 0, 1, 0}] = 0.05;

    std::vector<Observation> my_frame_obs = {make_detection(0, 0, {700, 400}, 0.3)};  // below floor
    std::vector<Observation> other_frame_obs = {make_detection(0, 0, {680, 390}, 0.8)};
    std::map<std::string, Eigen::Vector3d> other_anchor_positions = {
        {"wrist", Eigen::Vector3d(1.0, 0.0, 0.0)}};
    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, make_camera(0, Eigen::Vector3d(0.0, 0.0, -2.0)));

    auto anchors = build_cross_person_anchors(0, 1, active_pairs, my_frame_obs, other_frame_obs,
                                              other_anchor_positions, other_skeleton, cameras, 0.5,
                                              0.5, 10, 2.0, 3.0, 5.0, 1, 1.0);
    REQUIRE(anchors.empty());
}

TEST_CASE("build_cross_person_anchors: caps candidates per camera, closest-first",
          "[multi_person][anchors]") {
    // 5 marker pairs between person 0 and person 1, all in camera 0, at distinct
    // cached distances; cap at 2 must keep only the two closest.
    Skeleton other_skeleton = make_skeleton_with_markers({"m0", "m1", "m2", "m3", "m4"});

    std::map<ContactMarkerPair, double> active_pairs;
    std::vector<Observation> my_frame_obs;
    std::vector<Observation> other_frame_obs;
    std::map<std::string, Eigen::Vector3d> other_anchor_positions;

    std::vector<double> distances = {0.09, 0.02, 0.07, 0.01, 0.05};  // marker index i -> distance
    for (int i = 0; i < 5; ++i) {
        active_pairs[ContactMarkerPair{0, i, 1, i}] = distances[static_cast<size_t>(i)];
        my_frame_obs.push_back(make_detection(0, i, {700.0 + i, 400.0}, 0.9));
        other_frame_obs.push_back(make_detection(0, i, {680.0 + i, 390.0}, 0.8));
        other_anchor_positions["m" + std::to_string(i)] = Eigen::Vector3d(1.0, 0.01 * i, 0.0);
    }

    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, make_camera(0, Eigen::Vector3d(0.0, 0.0, -2.0)));

    auto anchors = build_cross_person_anchors(0, 1, active_pairs, my_frame_obs, other_frame_obs,
                                              other_anchor_positions, other_skeleton, cameras, 0.5,
                                              0.5, /*max_n=*/2, 2.0, 3.0, 5.0, 1, 1.0);

    REQUIRE(anchors.size() == 2);
    // Closest two distances are marker 3 (0.01) and marker 1 (0.02) -> my marker_id 3 and 1.
    std::vector<int> got_marker_ids;
    for (auto const& o : anchors)
        got_marker_ids.push_back(o.marker_id);
    std::sort(got_marker_ids.begin(), got_marker_ids.end());
    REQUIRE(got_marker_ids == std::vector<int>{1, 3});
}

TEST_CASE(
    "build_cross_person_anchors: mismatched cameras between the two detections are not paired",
    "[multi_person][anchors]") {
    Skeleton other_skeleton = make_skeleton_with_markers({"wrist"});
    std::map<ContactMarkerPair, double> active_pairs;
    active_pairs[ContactMarkerPair{0, 0, 1, 0}] = 0.05;

    std::vector<Observation> my_frame_obs = {make_detection(/*camera=*/0, 0, {700, 400}, 0.9)};
    std::vector<Observation> other_frame_obs = {
        make_detection(/*camera=*/1, 0, {680, 390}, 0.8)};  // different camera
    std::map<std::string, Eigen::Vector3d> other_anchor_positions = {
        {"wrist", Eigen::Vector3d(1.0, 0.0, 0.0)}};
    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, make_camera(0, Eigen::Vector3d(0.0, 0.0, -2.0)));
    cameras.emplace(1, make_camera(1, Eigen::Vector3d(0.0, 1.0, -2.0)));

    auto anchors = build_cross_person_anchors(0, 1, active_pairs, my_frame_obs, other_frame_obs,
                                              other_anchor_positions, other_skeleton, cameras, 0.5,
                                              0.5, 10, 2.0, 3.0, 5.0, 1, 1.0);
    REQUIRE(anchors.empty());
}
