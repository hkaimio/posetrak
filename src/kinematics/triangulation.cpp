#include <posetrak/kinematics/triangulation.hpp>

#include <Eigen/SVD>

#include <cmath>
#include <stdexcept>

namespace posetrak {

Triangulator::Triangulator(Method method) : method_(method) {}

TriangulationResult Triangulator::triangulate(std::vector<Eigen::Vector2d> const& observations,
                                              std::vector<Camera const*> const& cameras,
                                              std::vector<double> const& confidences) const {
    // Validate inputs
    if (observations.empty() || cameras.empty()) {
        return TriangulationResult::failure();
    }

    if (observations.size() != cameras.size()) {
        return TriangulationResult::failure();
    }

    if (!confidences.empty() && confidences.size() != observations.size()) {
        return TriangulationResult::failure();
    }

    // Need at least 2 cameras
    if (observations.size() < 2) {
        return TriangulationResult::failure();
    }

    // Dispatch to appropriate method
    switch (method_) {
        case Method::MidPoint:
            return triangulate_midpoint(observations, cameras, confidences);
        case Method::DLT:
            return triangulate_dlt(observations, cameras, confidences);
        case Method::LeastSquares:
            return triangulate_least_squares(observations, cameras, confidences);
        default:
            return TriangulationResult::failure();
    }
}

TriangulationResult
Triangulator::triangulate_midpoint(std::vector<Eigen::Vector2d> const& observations,
                                   std::vector<Camera const*> const& cameras,
                                   std::vector<double> const& confidences) const {
    // Mid-point method works best for exactly 2 cameras
    // For more cameras, fall back to simple pairwise average

    if (observations.size() < 2) {
        return TriangulationResult::failure();
    }

    // For 2 cameras: compute mid-point of closest approach
    if (observations.size() == 2) {
        // Get camera extrinsics
        auto const& cam0 = cameras[0];
        auto const& cam1 = cameras[1];

        Eigen::Vector3d const& C0 = cam0->extrinsics().position;
        Eigen::Vector3d const& C1 = cam1->extrinsics().position;

        // Compute ray directions (from camera to point)
        // pixel -> normalized camera coords -> world direction
        Eigen::Vector3d ray0 = cam0->unproject_ray(observations[0]);
        Eigen::Vector3d ray1 = cam1->unproject_ray(observations[1]);

        // Find closest point between two rays
        // Ray 0: P = C0 + s * ray0
        // Ray 1: Q = C1 + t * ray1
        // Minimize |P - Q|^2

        Eigen::Vector3d w = C0 - C1;
        double a = ray0.dot(ray0);
        double b = ray0.dot(ray1);
        double c = ray1.dot(ray1);
        double d = ray0.dot(w);
        double e = ray1.dot(w);

        double denom = a * c - b * b;

        // Check for parallel rays (degenerate case)
        if (std::abs(denom) < 1e-10) {
            return TriangulationResult::failure();
        }

        double s = (b * e - c * d) / denom;
        double t = (a * e - b * d) / denom;

        // Closest points on each ray
        Eigen::Vector3d P0 = C0 + s * ray0;
        Eigen::Vector3d P1 = C1 + t * ray1;

        // Mid-point
        Eigen::Vector3d position = 0.5 * (P0 + P1);

        // Compute reprojection error
        double error = compute_reprojection_error(position, observations, cameras);

        return TriangulationResult{position, error, 2, true};
    }

    // For > 2 cameras: compute all pairs and average
    // (Simple fallback, not optimal)
    std::vector<Eigen::Vector3d> points;

    for (size_t i = 0; i < observations.size(); ++i) {
        for (size_t j = i + 1; j < observations.size(); ++j) {
            std::vector<Eigen::Vector2d> pair_obs = {observations[i], observations[j]};
            std::vector<Camera const*> pair_cams = {cameras[i], cameras[j]};

            auto result = triangulate_midpoint(pair_obs, pair_cams, {});
            if (result.success) {
                points.push_back(result.position);
            }
        }
    }

    if (points.empty()) {
        return TriangulationResult::failure();
    }

    // Average all pairwise results
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    for (auto const& p : points) {
        position += p;
    }
    position /= static_cast<double>(points.size());

    double error = compute_reprojection_error(position, observations, cameras);

    return TriangulationResult{position, error, static_cast<int>(observations.size()), true};
}

TriangulationResult Triangulator::triangulate_dlt(std::vector<Eigen::Vector2d> const& observations,
                                                  std::vector<Camera const*> const& cameras,
                                                  std::vector<double> const& confidences) const {
    if (observations.size() < 2) {
        return TriangulationResult::failure();
    }

    size_t const n = observations.size();

    // Build DLT matrix A
    // Each observation contributes 2 rows:
    // x * P(3,:) - P(1,:) = 0
    // y * P(3,:) - P(2,:) = 0
    // where P is the camera projection matrix

    Eigen::MatrixXd A(2 * n, 4);

    for (size_t i = 0; i < n; ++i) {
        // Get projection matrix for this camera
        Eigen::Matrix<double, 3, 4> P = cameras[i]->get_projection_matrix();

        double const x = observations[i].x();
        double const y = observations[i].y();

        // Confidence weight (default to 1.0 if not provided)
        double weight = 1.0;
        if (!confidences.empty()) {
            weight = std::sqrt(confidences[i]);  // sqrt for weighting rows
        }

        // First row: x * P(3,:) - P(1,:)
        A.row(2 * i) = weight * (x * P.row(2) - P.row(0));

        // Second row: y * P(3,:) - P(2,:)
        A.row(2 * i + 1) = weight * (y * P.row(2) - P.row(1));
    }

    // Solve A*X = 0 via SVD
    // Solution is the right singular vector corresponding to smallest singular value
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X_homogeneous = svd.matrixV().col(3);

    // Convert from homogeneous to 3D
    if (std::abs(X_homogeneous(3)) < 1e-10) {
        // Point at infinity
        return TriangulationResult::failure();
    }

    Eigen::Vector3d position = X_homogeneous.head<3>() / X_homogeneous(3);

    // Compute reprojection error
    double error = compute_reprojection_error(position, observations, cameras);

    return TriangulationResult{position, error, static_cast<int>(n), true};
}

TriangulationResult
Triangulator::triangulate_least_squares(std::vector<Eigen::Vector2d> const& observations,
                                        std::vector<Camera const*> const& cameras,
                                        std::vector<double> const& confidences) const {
    // TODO: Implement non-linear least squares refinement
    // For now, throw error
    throw std::runtime_error("Least squares triangulation not yet implemented");
}

double Triangulator::compute_reprojection_error(Eigen::Vector3d const& point_3d,
                                                std::vector<Eigen::Vector2d> const& observations,
                                                std::vector<Camera const*> const& cameras) const {
    if (observations.empty()) {
        return -1.0;
    }

    double sum_squared_error = 0.0;

    for (size_t i = 0; i < observations.size(); ++i) {
        // Project 3D point to camera
        Eigen::Vector2d projected = cameras[i]->project(point_3d);

        // Compute error
        Eigen::Vector2d error = projected - observations[i];
        sum_squared_error += error.squaredNorm();
    }

    // Return RMS error
    return std::sqrt(sum_squared_error / static_cast<double>(observations.size()));
}

std::map<int, TriangulationResult>
Triangulator::triangulate_frame(double timestamp, ObservationSet const& observations,
                                std::vector<Camera> const& cameras, double tolerance) const {
    std::map<int, TriangulationResult> results;

    // Get all observations at this timestamp
    std::vector<Observation> frame_obs = observations.get_all_at_time(timestamp, tolerance);

    if (frame_obs.empty()) {
        return results;  // No observations at this time
    }

    // Group observations by marker_id
    std::map<int, std::vector<Observation>> obs_by_marker;
    for (auto const& obs : frame_obs) {
        obs_by_marker[obs.marker_id].push_back(obs);
    }

    // Build camera lookup map
    std::map<int, Camera const*> camera_map;
    for (auto const& cam : cameras) {
        camera_map[cam.id()] = &cam;
    }

    // Triangulate each marker
    for (auto const& [marker_id, marker_obs] : obs_by_marker) {
        // Need at least 2 observations
        if (marker_obs.size() < 2) {
            continue;
        }

        // Collect observations, cameras, and confidences for this marker
        std::vector<Eigen::Vector2d> obs_2d;
        std::vector<Camera const*> cams;
        std::vector<double> confs;

        for (auto const& obs : marker_obs) {
            // Find camera
            auto cam_it = camera_map.find(obs.camera_id);
            if (cam_it == camera_map.end()) {
                continue;  // Camera not found, skip
            }

            obs_2d.push_back(obs.position);
            cams.push_back(cam_it->second);
            confs.push_back(obs.confidence);
        }

        // Skip if insufficient observations after filtering
        if (obs_2d.size() < 2) {
            continue;
        }

        // Triangulate
        TriangulationResult result = triangulate(obs_2d, cams, confs);

        // Only store successful results
        if (result.success) {
            results[marker_id] = result;
        }
    }

    return results;
}

}  // namespace posetrak
