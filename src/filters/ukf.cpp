/**
 * @file ukf.cpp
 * @brief Implementation of Unscented Kalman Filter
 */

#include "posetrak/filters/ukf.hpp"

#include <cmath>
#include <stdexcept>

namespace posetrak {

UnscentedKalmanFilter::UnscentedKalmanFilter(Skeleton const& skeleton, double process_noise_std,
                                             double alpha, double beta, double kappa)
    : skeleton_(skeleton),
      state_(skeleton.total_dof_count()),
      covariance_(Eigen::MatrixXd::Identity(2 * (6 + skeleton.total_dof_count()),
                                            2 * (6 + skeleton.total_dof_count()))),
      process_noise_(Eigen::MatrixXd::Identity(2 * (6 + skeleton.total_dof_count()),
                                               2 * (6 + skeleton.total_dof_count()))),
      sigma_gen_(skeleton, alpha, beta, kappa),
      process_model_(skeleton) {
    // Initialize process noise with given standard deviation
    double const variance = process_noise_std * process_noise_std;
    process_noise_ *= variance;
}

void UnscentedKalmanFilter::set_covariance(Eigen::MatrixXd const& covariance) {
    int const expected_dim = error_dim();
    if (covariance.rows() != expected_dim || covariance.cols() != expected_dim) {
        throw std::invalid_argument("Covariance size must match error dimension");
    }
    covariance_ = covariance;
}

void UnscentedKalmanFilter::predict(double dt) {
    // Generate sigma points
    auto sigma_points = sigma_gen_.generate_sigma_points(state_, covariance_);

    // Propagate sigma points through process model
    std::vector<State> propagated_points;
    propagated_points.reserve(sigma_points.size());

    for (auto const& sigma_state : sigma_points) {
        propagated_points.push_back(process_model_.propagate(sigma_state, dt));
    }

    // Compute predicted mean
    state_ = compute_state_mean(propagated_points, sigma_gen_.get_mean_weights());

    // Compute predicted covariance
    covariance_ =
        compute_state_covariance(propagated_points, state_, sigma_gen_.get_covariance_weights());

    // Add process noise
    covariance_ += process_noise_ * dt;
}

State UnscentedKalmanFilter::compute_state_mean(std::vector<State> const& states,
                                                Eigen::VectorXd const& weights) const {
    // Create mean state
    State mean_state(skeleton_.total_dof_count());

    // Mean position (simple weighted average)
    Eigen::Vector3d pos_mean = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < states.size(); ++i) {
        pos_mean += weights(i) * states[i].root_position();
    }
    mean_state.set_root_position(pos_mean);

    // Mean quaternion (iterative on manifold)
    Eigen::Quaterniond q_mean = states[0].root_orientation();
    for (int iter = 0; iter < 5; ++iter) {
        Eigen::Vector3d error_sum = Eigen::Vector3d::Zero();

        for (size_t i = 0; i < states.size(); ++i) {
            Eigen::Quaterniond const& q_i = states[i].root_orientation();
            // Compute quaternion difference: q_mean^-1 * q_i
            Eigen::Quaterniond q_diff = q_mean.conjugate() * q_i;

            // Convert to axis-angle (error space)
            double const angle = 2.0 * std::atan2(q_diff.vec().norm(), q_diff.w());
            if (angle > 1e-8) {
                Eigen::Vector3d const axis = q_diff.vec().normalized();
                error_sum += weights(i) * angle * axis;
            }
        }

        // Update mean quaternion
        Eigen::Quaterniond q_error = State::axis_angle_to_quaternion(error_sum);
        q_mean = (q_mean * q_error).normalized();

        if (error_sum.norm() < 1e-6) {
            break;
        }
    }
    mean_state.set_root_orientation(q_mean);

    // Mean velocities (simple weighted average)
    Eigen::Vector3d vel_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel_mean = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < states.size(); ++i) {
        vel_mean += weights(i) * states[i].root_velocity();
        angvel_mean += weights(i) * states[i].root_angular_velocity();
    }
    mean_state.set_root_velocity(vel_mean);
    mean_state.set_root_angular_velocity(angvel_mean);

    // Mean joint angles and velocities
    auto const joints_ordered = skeleton_.get_joints_ordered();
    Eigen::VectorXd angles_mean = Eigen::VectorXd::Zero(skeleton_.total_dof_count());
    Eigen::VectorXd velocities_mean = Eigen::VectorXd::Zero(skeleton_.total_dof_count());

    int dof_idx = 0;
    for (auto const& joint : joints_ordered) {
        if (!joint.parent_index.has_value()) {
            continue;  // Skip root
        }

        if (joint.type == JointType::REVOLUTE) {
            // Simple weighted average
            for (size_t i = 0; i < states.size(); ++i) {
                angles_mean(dof_idx) += weights(i) * states[i].joint_angles()(dof_idx);
                velocities_mean(dof_idx) += weights(i) * states[i].joint_velocities()(dof_idx);
            }
            dof_idx += 1;

        } else if (joint.type == JointType::SPHERICAL) {
            // Always 3 DOFs: iterative mean on SO(3) manifold
            Eigen::Vector3d const initial_aa = states[0].joint_angles().segment<3>(dof_idx);
            Eigen::Matrix3d R_mean = State::axis_angle_to_quaternion(initial_aa).toRotationMatrix();

            for (int iter = 0; iter < 10; ++iter) {
                Eigen::Vector3d error_sum = Eigen::Vector3d::Zero();

                for (size_t i = 0; i < states.size(); ++i) {
                    Eigen::Vector3d const aa_i = states[i].joint_angles().segment<3>(dof_idx);
                    Eigen::Matrix3d const R_i =
                        State::axis_angle_to_quaternion(aa_i).toRotationMatrix();

                    // Relative rotation: R_mean^T * R_i
                    Eigen::Matrix3d const R_rel = R_mean.transpose() * R_i;
                    Eigen::Quaterniond const q_rel(R_rel);
                    Eigen::Vector3d const error_i = State::quaternion_to_axis_angle(q_rel);

                    error_sum += weights(i) * error_i;
                }

                // Update mean
                Eigen::Matrix3d const R_delta =
                    State::axis_angle_to_quaternion(error_sum).toRotationMatrix();
                R_mean = R_mean * R_delta;

                if (error_sum.norm() < 1e-6) {
                    break;
                }
            }

            // Convert back to axis-angle
            Eigen::Quaterniond const q_mean(R_mean);
            angles_mean.segment<3>(dof_idx) = State::quaternion_to_axis_angle(q_mean);

            // Velocities: simple weighted average
            for (size_t i = 0; i < states.size(); ++i) {
                velocities_mean.segment<3>(dof_idx) +=
                    weights(i) * states[i].joint_velocities().segment<3>(dof_idx);
            }

            dof_idx += 3;
        }
    }

    mean_state.set_joint_angles(angles_mean);
    mean_state.set_joint_velocities(velocities_mean);

    return mean_state;
}

Eigen::MatrixXd
UnscentedKalmanFilter::compute_state_covariance(std::vector<State> const& states,
                                                State const& mean_state,
                                                Eigen::VectorXd const& weights) const {
    int const n = error_dim();
    int const n_sigma = states.size();

    // Compute all error vectors
    Eigen::MatrixXd error_vectors(n_sigma, n);
    for (int i = 0; i < n_sigma; ++i) {
        error_vectors.row(i) = compute_state_error(states[i], mean_state);
    }

    // Compute weighted covariance: Σ wc[i] * error[i] * error[i]^T
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n_sigma; ++i) {
        cov += weights(i) * error_vectors.row(i).transpose() * error_vectors.row(i);
    }

    return cov;
}

Eigen::VectorXd UnscentedKalmanFilter::compute_state_error(State const& state,
                                                           State const& reference) const {
    int const dof = skeleton_.total_dof_count();
    Eigen::VectorXd error = Eigen::VectorXd::Zero(error_dim());

    // Position error
    error.segment<3>(0) = state.root_position() - reference.root_position();

    // Rotation error (in tangent space)
    Eigen::Quaterniond const& q_ref = reference.root_orientation();
    Eigen::Quaterniond const& q_state = state.root_orientation();
    Eigen::Quaterniond const q_diff = q_ref.conjugate() * q_state;

    // Convert to axis-angle
    double const angle = 2.0 * std::atan2(q_diff.vec().norm(), q_diff.w());
    if (angle > 1e-8) {
        Eigen::Vector3d const axis = q_diff.vec().normalized();
        error.segment<3>(3) = angle * axis;
    }

    // Velocity errors
    error.segment<3>(6 + dof) = state.root_velocity() - reference.root_velocity();
    error.segment<3>(9 + dof) = state.root_angular_velocity() - reference.root_angular_velocity();

    // Joint angle and velocity errors
    auto const joints_ordered = skeleton_.get_joints_ordered();
    int error_pos_idx = 6;     // Start after root position/rotation in error vector
    int joint_angles_idx = 0;  // Index in joint_angles/joint_velocities vectors

    for (auto const& joint : joints_ordered) {
        if (!joint.parent_index.has_value()) {
            continue;  // Skip root
        }

        if (joint.type == JointType::REVOLUTE) {
            // Position (angle) error
            error(error_pos_idx) =
                state.joint_angles()(joint_angles_idx) - reference.joint_angles()(joint_angles_idx);

            // Velocity error
            error(6 + dof + joint_angles_idx) = state.joint_velocities()(joint_angles_idx) -
                                                reference.joint_velocities()(joint_angles_idx);

            error_pos_idx += 1;
            joint_angles_idx += 1;

        } else if (joint.type == JointType::SPHERICAL) {
            // Always 3 DOFs: error on SO(3) manifold
            Eigen::Vector3d const aa_ref = reference.joint_angles().segment<3>(joint_angles_idx);
            Eigen::Vector3d const aa_state = state.joint_angles().segment<3>(joint_angles_idx);

            Eigen::Matrix3d const R_ref =
                State::axis_angle_to_quaternion(aa_ref).toRotationMatrix();
            Eigen::Matrix3d const R_state =
                State::axis_angle_to_quaternion(aa_state).toRotationMatrix();

            // Relative rotation: R_ref^T * R_state
            Eigen::Matrix3d const R_rel = R_ref.transpose() * R_state;
            Eigen::Quaterniond const q_rel(R_rel);
            error.segment<3>(error_pos_idx) = State::quaternion_to_axis_angle(q_rel);

            // Velocity error
            error.segment<3>(6 + dof + joint_angles_idx) =
                state.joint_velocities().segment<3>(joint_angles_idx) -
                reference.joint_velocities().segment<3>(joint_angles_idx);

            error_pos_idx += 3;
            joint_angles_idx += 3;
        }
    }

    return error;
}

UpdateResult UnscentedKalmanFilter::update(std::vector<Observation> const& observations,
                                           std::unordered_map<int, Camera> const& cameras,
                                           ForwardKinematics& fk, double measurement_noise_std,
                                           double outlier_threshold_mahalanobis) {
    UpdateResult result;

    if (observations.empty()) {
        return result;  // No observations to process
    }

    int const n_obs = static_cast<int>(observations.size());
    int const measurement_dim = 2 * n_obs;  // 2D pixel per observation

    // These will be updated after outlier rejection
    int effective_n_obs = n_obs;
    int effective_measurement_dim = measurement_dim;

    // Step 1: Generate sigma points from current state and covariance
    auto sigma_points = sigma_gen_.generate_sigma_points(state_, covariance_);
    int const n_sigma = static_cast<int>(sigma_points.size());

    // Step 2: Predict measurements for each sigma point
    Eigen::MatrixXd predicted_measurements(measurement_dim, n_sigma);
    for (int i = 0; i < n_sigma; ++i) {
        predicted_measurements.col(i) =
            predict_measurements(sigma_points[i], observations, cameras, fk);
    }

    // Step 3: Compute mean predicted measurement
    Eigen::VectorXd const weights_mean = sigma_gen_.get_mean_weights();
    Eigen::VectorXd measurement_mean = Eigen::VectorXd::Zero(measurement_dim);
    for (int i = 0; i < n_sigma; ++i) {
        measurement_mean += weights_mean(i) * predicted_measurements.col(i);
    }

    // Step 4: Compute innovation covariance S = Pyy + R
    Eigen::VectorXd const weights_cov = sigma_gen_.get_covariance_weights();
    Eigen::MatrixXd innovation_cov = Eigen::MatrixXd::Zero(measurement_dim, measurement_dim);

    for (int i = 0; i < n_sigma; ++i) {
        Eigen::VectorXd innovation = predicted_measurements.col(i) - measurement_mean;
        innovation_cov += weights_cov(i) * (innovation * innovation.transpose());
    }

    // Add measurement noise R (diagonal, same noise for all observations)
    for (int i = 0; i < n_obs; ++i) {
        double noise_std = observations[i].measurement_noise_std(measurement_noise_std);
        double variance = noise_std * noise_std;
        innovation_cov(2 * i, 2 * i) += variance;          // x coordinate
        innovation_cov(2 * i + 1, 2 * i + 1) += variance;  // y coordinate
    }

    // Step 4.5: Perform outlier rejection if enabled
    std::vector<Observation> inlier_observations;
    std::vector<ObservationResult> observation_results;
    Eigen::MatrixXd cross_cov;
    Eigen::VectorXd observed;

    if (outlier_threshold_mahalanobis > 0.0) {
        // Perform outlier rejection
        auto [inliers, results] =
            reject_outliers(observations, predicted_measurements, measurement_mean, innovation_cov,
                            outlier_threshold_mahalanobis);
        inlier_observations = inliers;
        observation_results = results;

        // If all observations rejected, return early
        if (inlier_observations.empty()) {
            result.num_observations = static_cast<int>(observations.size());
            result.num_outliers = static_cast<int>(observations.size());
            result.num_inliers = 0;
            result.observations = observation_results;
            return result;
        }

        // Recompute predictions with only inliers
        int const n_inliers = static_cast<int>(inlier_observations.size());
        int const inlier_dim = 2 * n_inliers;

        Eigen::MatrixXd inlier_predictions(inlier_dim, n_sigma);
        for (int i = 0; i < n_sigma; ++i) {
            inlier_predictions.col(i) =
                predict_measurements(sigma_points[i], inlier_observations, cameras, fk);
        }

        // Recompute mean predicted measurement for inliers
        measurement_mean = Eigen::VectorXd::Zero(inlier_dim);
        for (int i = 0; i < n_sigma; ++i) {
            measurement_mean += weights_mean(i) * inlier_predictions.col(i);
        }

        // Recompute innovation covariance for inliers
        innovation_cov = Eigen::MatrixXd::Zero(inlier_dim, inlier_dim);
        for (int i = 0; i < n_sigma; ++i) {
            Eigen::VectorXd innovation = inlier_predictions.col(i) - measurement_mean;
            innovation_cov += weights_cov(i) * (innovation * innovation.transpose());
        }

        // Add measurement noise for inliers
        for (int i = 0; i < n_inliers; ++i) {
            double noise_std = inlier_observations[i].measurement_noise_std(measurement_noise_std);
            double variance = noise_std * noise_std;
            innovation_cov(2 * i, 2 * i) += variance;
            innovation_cov(2 * i + 1, 2 * i + 1) += variance;
        }

        // Recompute cross-covariance with inliers
        cross_cov = Eigen::MatrixXd::Zero(error_dim(), inlier_dim);
        for (int i = 0; i < n_sigma; ++i) {
            Eigen::VectorXd state_error = compute_state_error(sigma_points[i], state_);
            Eigen::VectorXd measurement_error = inlier_predictions.col(i) - measurement_mean;
            cross_cov += weights_cov(i) * (state_error * measurement_error.transpose());
        }

        // Update observed vector to use inliers
        observed = observations_to_vector(inlier_observations);

        // Update effective dimensions for inliers
        effective_n_obs = n_inliers;
        effective_measurement_dim = inlier_dim;
    } else {
        // No outlier rejection - compute diagnostics for all observations
        observation_results =
            compute_observation_diagnostics(observations, measurement_mean, innovation_cov);
        inlier_observations = observations;
        observed = observations_to_vector(observations);
    }

    // Step 5: Compute cross-covariance Pxy (already computed if outlier rejection enabled)
    if (outlier_threshold_mahalanobis <= 0.0) {
        // Cross-covariance not yet computed (no outlier rejection)
        cross_cov = Eigen::MatrixXd::Zero(error_dim(), measurement_dim);
        for (int i = 0; i < n_sigma; ++i) {
            Eigen::VectorXd state_error = compute_state_error(sigma_points[i], state_);
            Eigen::VectorXd measurement_error = predicted_measurements.col(i) - measurement_mean;
            cross_cov += weights_cov(i) * (state_error * measurement_error.transpose());
        }
    }

    // Step 6: Compute Kalman gain K = Pxy * S^-1
    Eigen::MatrixXd kalman_gain = cross_cov * innovation_cov.inverse();

    // Step 7: Compute innovation (observed - predicted)
    Eigen::VectorXd innovation = observed - measurement_mean;

    // Step 8: Update state in error space
    Eigen::VectorXd state_correction = kalman_gain * innovation;
    state_.apply_error_update(state_correction);

    // Step 9: Update covariance using Joseph form for numerical stability
    // P' = (I - K*H)*P*(I - K*H)^T + K*R*K^T
    // In UKF, we compute H implicitly through the unscented transform
    // We need to extract R (measurement noise) separately from S = H*P*H^T + R

    // Build measurement noise covariance R (use inlier observations if outlier rejection was done)
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(effective_measurement_dim, effective_measurement_dim);
    for (int i = 0; i < effective_n_obs; ++i) {
        double noise_std = inlier_observations[i].measurement_noise_std(measurement_noise_std);
        double variance = noise_std * noise_std;
        R(2 * i, 2 * i) = variance;          // x coordinate
        R(2 * i + 1, 2 * i + 1) = variance;  // y coordinate
    }

    // Compute H*P*H^T implicitly as S - R
    Eigen::MatrixXd HPH = innovation_cov - R;

    // Compute (I - K*H) where H is represented through the cross-covariance
    // K*H ≈ K * (cross_cov^T * P^-1) in the error space
    // Simplified approach: Use standard covariance update
    // P' = P - K*S*K^T (this is the simplified Joseph form)
    covariance_ = covariance_ - kalman_gain * innovation_cov * kalman_gain.transpose();

    // Step 10: Condition covariance for numerical stability
    // Ensure symmetry (numerical errors can cause asymmetry)
    covariance_ = 0.5 * (covariance_ + covariance_.transpose());

    // Ensure positive definiteness by checking eigenvalues
    // Use self-adjoint eigenvalue solver (faster for symmetric matrices)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(covariance_);
    if (eigen_solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to compute eigenvalues for covariance conditioning");
    }

    Eigen::VectorXd eigenvalues = eigen_solver.eigenvalues();
    double min_eigenvalue = eigenvalues.minCoeff();

    if (min_eigenvalue < 0.0) {
        // Add small positive value to diagonal to ensure positive definiteness
        double epsilon = std::abs(min_eigenvalue) + 1e-6;
        covariance_ += epsilon * Eigen::MatrixXd::Identity(error_dim(), error_dim());
    }

    // Step 11: Damp velocity covariance for joints near limits
    damp_velocity_covariance_at_limits();

    // Step 12: Compute Normalized Innovation Squared (NIS) for filter validation
    // NIS = innovation^T * S^-1 * innovation (should follow chi-squared distribution)
    double nis = 0.0;
    try {
        Eigen::MatrixXd innovation_cov_inv = innovation_cov.inverse();
        nis = innovation.transpose() * innovation_cov_inv * innovation;
    } catch (...) {
        // If inversion fails, use pseudo-inverse
        Eigen::MatrixXd innovation_cov_pinv =
            innovation_cov.completeOrthogonalDecomposition().pseudoInverse();
        nis = innovation.transpose() * innovation_cov_pinv * innovation;
    }

    // Fill in result
    result.num_observations = static_cast<int>(observations.size());
    result.num_inliers = static_cast<int>(inlier_observations.size());
    result.num_outliers = result.num_observations - result.num_inliers;
    result.observations = observation_results;
    result.nis = nis;
    result.nis_dof = static_cast<int>(innovation.size());

    return result;
}

Eigen::VectorXd UnscentedKalmanFilter::predict_measurements(
    State const& state, std::vector<Observation> const& observations,
    std::unordered_map<int, Camera> const& cameras, ForwardKinematics& fk) const {
    // Compute forward kinematics to get marker positions
    auto marker_positions = fk.compute(state);

    // Project each marker to its camera
    int const n_obs = static_cast<int>(observations.size());
    Eigen::VectorXd predictions(2 * n_obs);

    for (int i = 0; i < n_obs; ++i) {
        Observation const& obs = observations[i];

        // Get marker position (3D world)
        auto const& marker = skeleton_.markers()[obs.marker_id];
        std::string const& marker_name = marker.name;

        auto it = marker_positions.find(marker_name);
        if (it == marker_positions.end()) {
            // Marker not found in FK result - use fallback (project to image center)
            predictions(2 * i) = cameras.at(obs.camera_id).intrinsics().cx;
            predictions(2 * i + 1) = cameras.at(obs.camera_id).intrinsics().cy;
            continue;
        }

        Eigen::Vector3d const& marker_pos_world = it->second;

        // Project to camera (undistorted coordinates for UKF)
        Camera const& camera = cameras.at(obs.camera_id);
        Eigen::Vector2d projected = camera.project_undistorted(marker_pos_world);

        // Check for invalid projections (markers behind camera produce NaN/inf)
        // This can happen when state estimate is poor or during initialization
        if (!std::isfinite(projected.x()) || !std::isfinite(projected.y())) {
            // Use image center as fallback for failed projections
            predictions(2 * i) = camera.intrinsics().cx;
            predictions(2 * i + 1) = camera.intrinsics().cy;
        } else {
            predictions(2 * i) = projected.x();
            predictions(2 * i + 1) = projected.y();
        }
    }

    return predictions;
}

Eigen::VectorXd
UnscentedKalmanFilter::observations_to_vector(std::vector<Observation> const& observations) const {
    int const n_obs = static_cast<int>(observations.size());
    Eigen::VectorXd measurements(2 * n_obs);

    for (int i = 0; i < n_obs; ++i) {
        measurements(2 * i) = observations[i].position.x();
        measurements(2 * i + 1) = observations[i].position.y();
    }

    return measurements;
}

double
UnscentedKalmanFilter::compute_mahalanobis_distance(Eigen::Vector2d const& innovation,
                                                    Eigen::Matrix2d const& covariance) const {
    // Mahalanobis distance: sqrt(innovation^T * cov^-1 * innovation)
    Eigen::Matrix2d cov_inv = covariance.inverse();
    double distance_squared = innovation.transpose() * cov_inv * innovation;
    return std::sqrt(distance_squared);
}

std::pair<std::vector<Observation>, std::vector<ObservationResult>>
UnscentedKalmanFilter::reject_outliers(std::vector<Observation> const& observations,
                                       Eigen::MatrixXd const& predicted_measurements,
                                       Eigen::VectorXd const& measurement_mean,
                                       Eigen::MatrixXd const& innovation_cov,
                                       double threshold) const {
    std::vector<Observation> inliers;
    std::vector<ObservationResult> results;

    Eigen::VectorXd observed = observations_to_vector(observations);

    for (size_t i = 0; i < observations.size(); ++i) {
        Observation const& obs = observations[i];

        // Extract predicted and actual measurements for this observation
        Eigen::Vector2d predicted = measurement_mean.segment<2>(2 * i);
        Eigen::Vector2d actual = observed.segment<2>(2 * i);

        // Check for NaN in predicted (failed projection)
        if (!std::isfinite(predicted.x()) || !std::isfinite(predicted.y())) {
            // Projection failed - reject as outlier
            ObservationResult obs_result;
            obs_result.marker_name = skeleton_.markers()[obs.marker_id].name;
            obs_result.camera_id = obs.camera_id;
            obs_result.is_outlier = true;
            obs_result.mahalanobis_distance = 0.0;
            obs_result.innovation = Eigen::Vector2d::Zero();
            obs_result.predicted = predicted;
            obs_result.actual = actual;
            results.push_back(obs_result);
            continue;
        }

        // Extract 2x2 covariance for this observation
        Eigen::Matrix2d cov_2x2 = innovation_cov.block<2, 2>(2 * i, 2 * i);

        // Compute innovation
        Eigen::Vector2d innovation = actual - predicted;

        // Compute Mahalanobis distance
        double mahal_dist = compute_mahalanobis_distance(innovation, cov_2x2);

        // Check if outlier
        bool is_outlier = mahal_dist > threshold;

        // Create result
        ObservationResult obs_result;
        obs_result.marker_name = skeleton_.markers()[obs.marker_id].name;
        obs_result.camera_id = obs.camera_id;
        obs_result.is_outlier = is_outlier;
        obs_result.mahalanobis_distance = mahal_dist;
        obs_result.innovation = innovation;
        obs_result.predicted = predicted;
        obs_result.actual = actual;
        results.push_back(obs_result);

        // Keep observation if inlier
        if (!is_outlier) {
            inliers.push_back(obs);
        }
    }

    return {inliers, results};
}

std::vector<ObservationResult> UnscentedKalmanFilter::compute_observation_diagnostics(
    std::vector<Observation> const& observations, Eigen::VectorXd const& measurement_mean,
    Eigen::MatrixXd const& innovation_cov) const {
    std::vector<ObservationResult> results;

    Eigen::VectorXd observed = observations_to_vector(observations);

    for (size_t i = 0; i < observations.size(); ++i) {
        Observation const& obs = observations[i];

        // Extract predicted and actual measurements
        Eigen::Vector2d predicted = measurement_mean.segment<2>(2 * i);
        Eigen::Vector2d actual = observed.segment<2>(2 * i);

        // Check for NaN in predicted (failed projection)
        if (!std::isfinite(predicted.x()) || !std::isfinite(predicted.y())) {
            ObservationResult obs_result;
            obs_result.marker_name = skeleton_.markers()[obs.marker_id].name;
            obs_result.camera_id = obs.camera_id;
            obs_result.is_outlier = true;  // Mark as outlier for diagnostics
            obs_result.mahalanobis_distance = 0.0;
            obs_result.innovation = Eigen::Vector2d::Zero();
            obs_result.predicted = predicted;
            obs_result.actual = actual;
            results.push_back(obs_result);
            continue;
        }

        // Extract 2x2 covariance for this observation
        Eigen::Matrix2d cov_2x2 = innovation_cov.block<2, 2>(2 * i, 2 * i);

        // Compute innovation
        Eigen::Vector2d innovation = actual - predicted;

        // Compute Mahalanobis distance
        double mahal_dist = compute_mahalanobis_distance(innovation, cov_2x2);

        // Create result (not an outlier - just diagnostics)
        ObservationResult obs_result;
        obs_result.marker_name = skeleton_.markers()[obs.marker_id].name;
        obs_result.camera_id = obs.camera_id;
        obs_result.is_outlier = false;
        obs_result.mahalanobis_distance = mahal_dist;
        obs_result.innovation = innovation;
        obs_result.predicted = predicted;
        obs_result.actual = actual;
        results.push_back(obs_result);
    }

    return results;
}

void UnscentedKalmanFilter::damp_velocity_covariance_at_limits(double damping_factor,
                                                               double limit_margin) {
    // Check each joint to see if it's near its limits
    // We damp the velocity covariance for joints that are close to their limits
    // to prevent the filter from trying to push through the limit.

    int const error_pos_dim = error_dim() / 2;  // Position error dimension
    int pos_idx = 3;                            // Start after root position (3 DOF)

    // Get joint angles from state
    Eigen::VectorXd const& joint_angles = state_.joint_angles();

    // Iterate through joints (skip root which has no limits)
    auto const& joints = skeleton_.joints();
    int joint_angle_offset = 0;  // Offset into joint_angles vector

    for (size_t joint_idx = 1; joint_idx < joints.size(); ++joint_idx) {
        Joint const& joint = joints[joint_idx];

        // Only process joints with limits
        if (joint.num_limits == 0 || joint.type == JointType::FIXED) {
            pos_idx += joint.active_dof();
            joint_angle_offset += joint.dof;
            continue;
        }

        // Check each DOF of this joint
        auto active_mask = joint.get_active_dof_mask();
        int error_dof_idx = 0;  // Index within the active DOFs of this joint

        for (size_t dof = 0; dof < joint.num_limits && dof < 3; ++dof) {
            // Skip if this DOF is locked
            if (!active_mask[dof]) {
                continue;
            }

            // Get current angle and limits
            int angle_idx = joint_angle_offset + static_cast<int>(dof);
            if (angle_idx >= joint_angles.size()) {
                error_dof_idx++;
                continue;
            }

            double angle = joint_angles(angle_idx);
            double min_limit = joint.limits[dof].x();
            double max_limit = joint.limits[dof].y();

            // Check if near limit
            bool near_limit =
                (angle < min_limit + limit_margin) || (angle > max_limit - limit_margin);

            if (near_limit) {
                // Damp velocity covariance for this DOF
                int vel_idx = error_pos_dim + pos_idx + error_dof_idx;

                if (vel_idx < error_dim()) {
                    // Damp row and column
                    covariance_.row(vel_idx) *= damping_factor;
                    covariance_.col(vel_idx) *= damping_factor;

                    // Ensure minimum diagonal value for numerical stability
                    covariance_(vel_idx, vel_idx) = std::max(covariance_(vel_idx, vel_idx), 1e-8);
                }
            }

            error_dof_idx++;
        }

        pos_idx += joint.active_dof();
        joint_angle_offset += joint.dof;
    }
}

}  // namespace posetrak
