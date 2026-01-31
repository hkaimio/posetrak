/**
 * @file inverse_kinematics.hpp
 * @brief Basic inverse kinematics solver for pose initialization
 *
 * Implements damped least squares IK using Pinocchio's Jacobian computation.
 * Designed for tracker initialization - prioritizes convergence over precision.
 */

#pragma once

#include <Eigen/Dense>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include <map>
#include <string>

namespace posetrak {

/**
 * @brief Result of IK solve
 */
struct IKResult {
    State state;      ///< Computed state (joint angles, root pose)
    double residual;  ///< Final position error (RMS in meters)
    int iterations;   ///< Number of iterations taken
    bool converged;   ///< True if converged within tolerance

    static IKResult failure() {
        return IKResult{State(0), std::numeric_limits<double>::infinity(), 0, false};
    }
};

/**
 * @brief Inverse kinematics solver using damped least squares
 *
 * Uses Pinocchio's computeFrameJacobian to compute marker Jacobians,
 * then solves: Δq = J^T(JJ^T + λI)^(-1) * e
 * where e is the position error vector.
 *
 * This is a simple solver suitable for initialization:
 * - No secondary objectives (joint limit centering, etc.)
 * - Fixed damping parameter
 * - Stops after max iterations or when error < tolerance
 */
class InverseKinematics {
   public:
    /**
     * @brief Construct IK solver
     * @param model Pinocchio model
     * @param data Pinocchio data structure
     * @param fk Forward kinematics computer
     * @param marker_frame_map Map from marker name to frame index
     */
    InverseKinematics(pinocchio::Model const& model, pinocchio::Data& data,
                      ForwardKinematics const& fk,
                      std::map<std::string, pinocchio::FrameIndex> const& marker_frame_map);

    /**
     * @brief Solve IK for target marker positions
     *
     * @param target_markers Map of marker name → 3D world position
     * @param skeleton Skeleton structure (for joint types and limits)
     * @param initial_guess Initial state (defaults to zero pose)
     * @param max_iterations Maximum iterations (default 20)
     * @param tolerance Position error tolerance in meters (default 0.01 = 10mm)
     * @param damping Damping parameter λ for DLS (default 1e-6)
     * @return IK result with final state and convergence info
     */
    IKResult solve(std::map<std::string, Eigen::Vector3d> const& target_markers,
                   Skeleton const& skeleton,
                   std::optional<State> const& initial_guess = std::nullopt,
                   int max_iterations = 20, double tolerance = 0.01, double damping = 1e-4);

   private:
    /**
     * @brief Compute position error for all markers
     * @param q Current configuration
     * @param target_markers Target positions
     * @return Error vector (3 * num_markers)
     */
    Eigen::VectorXd compute_error(Eigen::VectorXd const& q,
                                  std::map<std::string, Eigen::Vector3d> const& target_markers);

    /**
     * @brief Compute stacked Jacobian for all markers
     * @param q Current configuration
     * @param marker_names Markers to include
     * @return Jacobian matrix (3*num_markers × nv)
     */
    Eigen::MatrixXd compute_jacobian(Eigen::VectorXd const& q,
                                     std::vector<std::string> const& marker_names);

    /**
     * @brief Apply joint limits to configuration
     * @param q Configuration to clamp
     * @param skeleton Skeleton with joint limits
     */
    void enforce_joint_limits(Eigen::VectorXd& q, Skeleton const& skeleton);

    /**
     * @brief Convert Pinocchio configuration to State
     * @param q Configuration vector
     * @param skeleton Skeleton structure
     * @return State with root pose and joint angles
     */
    State config_to_state(Eigen::VectorXd const& q, Skeleton const& skeleton);

    pinocchio::Model const& model_;
    pinocchio::Data& data_;
    ForwardKinematics const& fk_;
    std::map<std::string, pinocchio::FrameIndex> const& marker_frame_map_;
};

}  // namespace posetrak
