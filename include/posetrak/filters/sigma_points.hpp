/**
 * @file sigma_points.hpp
 * @brief Sigma point generation for Unscented Kalman Filter
 */

#pragma once

#include <posetrak/core/skeleton.hpp>
#include <posetrak/core/skeleton_layout.hpp>
#include <posetrak/core/state.hpp>

#include <Eigen/Core>

#include <vector>

namespace posetrak {

/**
 * @brief Generates sigma points for Unscented transform in error-state space
 *
 * Uses the scaled unscented transform to generate 2n+1 sigma points around
 * a nominal state, where n is the error-state dimension. Sigma points are
 * generated in error-state space to properly handle the quaternion manifold.
 *
 * Parameters:
 * - alpha: Spread of sigma points (typically 1e-3)
 * - beta: Distribution parameter (2 for Gaussian)
 * - kappa: Secondary scaling (typically 0)
 */
class SigmaPointGenerator {
   public:
    /**
     * @brief Construct sigma point generator with UKF parameters
     * @param skeleton Skeleton model for DOF information
     * @param alpha Spread parameter (default: 0.001)
     * @param beta Distribution parameter (default: 2.0)
     * @param kappa Secondary scaling parameter (default: 0.0)
     */
    SigmaPointGenerator(std::shared_ptr<const SkeletonLayout> layout, double alpha = 0.001,
                        double beta = 2.0, double kappa = 0.0);

    /**
     * @brief Generate sigma points in error-state space
     * @param nominal_state Nominal state around which to generate sigma points
     * @param covariance Error-state covariance matrix (n x n)
     * @return Vector of 2n+1 sigma points as State objects
     *
     * Algorithm:
     * 1. Compute Cholesky decomposition: L = chol(P)
     * 2. Generate error vectors: 0, ±gamma*L[:,i] for i=1..n
     * 3. Convert to state space: x_i = x_nominal ⊕ error_i
     */
    std::vector<State> generate_sigma_points(State const& nominal_state,
                                             Eigen::MatrixXd const& covariance) const;

    /**
     * @brief Get mean weights
     * @return Vector of weights for mean computation (size 2n+1)
     */
    Eigen::VectorXd const& get_mean_weights() const { return wm_; }

    /**
     * @brief Get covariance weights
     * @return Vector of weights for covariance computation (size 2n+1)
     */
    Eigen::VectorXd const& get_covariance_weights() const { return wc_; }

    /**
     * @brief Get sigma point scaling factor
     * @return Gamma value used to scale Cholesky factors
     */
    double get_gamma() const { return gamma_; }

    /**
     * @brief Get error-state dimension
     * @return Dimension of error state (2 * active_dof for pos + vel)
     */
    int error_dim() const { return error_dim_; }

    /**
     * @brief Apply error-state vector to nominal state
     * @param nominal_state Nominal state
     * @param error_vec Error vector in tangent space
     * @return New state with error applied
     *
     * For quaternion: q_new = q_nominal ⊗ exp(error_rotation)
     * For other states: x_new = x_nominal + error
     *
     * Handles locked DOFs and joint group filtering correctly.
     */
    State apply_error_to_state(State const& nominal_state, Eigen::VectorXd const& error_vec) const;

   private:
    std::shared_ptr<const SkeletonLayout> layout_;  ///< Precomputed DOF index table
    int error_dim_;                                 ///< Error-state dimension (2 * active_dof)
    double alpha_;                                  ///< Spread parameter
    double beta_;                                   ///< Distribution parameter
    double kappa_;                                  ///< Secondary scaling
    double gamma_;                                  ///< Sigma point scaling factor
    Eigen::VectorXd wm_;                            ///< Weights for mean
    Eigen::VectorXd wc_;                            ///< Weights for covariance
};

}  // namespace posetrak
