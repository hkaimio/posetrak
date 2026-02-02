#pragma once

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <sstream>
#include <string>

namespace posetrak::test_helpers {

/// @brief Check if two matrices are equal within tolerance
/// @param a First matrix
/// @param b Second matrix
/// @param tol Tolerance
/// @return True if matrices are equal within tolerance
inline bool matrices_equal(Eigen::MatrixXd const& a, Eigen::MatrixXd const& b, double tol = 1e-10) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return false;
    }

    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            double diff = std::abs(a(i, j) - b(i, j));
            if (diff > tol) {
                return false;
            }
        }
    }

    return true;
}

/// @brief Find and print first differing element in matrices
/// @param cpp C++ matrix
/// @param python Python matrix
/// @param name Name for display
/// @return String describing the difference
inline std::string matrix_diff_string(Eigen::MatrixXd const& cpp, Eigen::MatrixXd const& python,
                                      double tol = 1e-10) {
    std::ostringstream ss;

    if (cpp.rows() != python.rows() || cpp.cols() != python.cols()) {
        ss << "Size mismatch: C++=[" << cpp.rows() << "×" << cpp.cols() << "], Python=["
           << python.rows() << "×" << python.cols() << "]";
        return ss.str();
    }

    for (int i = 0; i < cpp.rows(); ++i) {
        for (int j = 0; j < cpp.cols(); ++j) {
            double diff = std::abs(cpp(i, j) - python(i, j));
            if (diff > tol) {
                ss << "First difference at (" << i << "," << j << "): C++=" << cpp(i, j)
                   << ", Python=" << python(i, j) << ", diff=" << diff;
                return ss.str();
            }
        }
    }

    return "Matrices are equal";
}

/// @brief Custom Catch2 matcher for Eigen vectors
class VectorEquals : public Catch::Matchers::MatcherBase<Eigen::VectorXd> {
   public:
    VectorEquals(Eigen::VectorXd const& expected, double tolerance)
        : expected_(expected), tolerance_(tolerance) {}

    bool match(Eigen::VectorXd const& actual) const override {
        if (actual.size() != expected_.size()) {
            return false;
        }

        for (int i = 0; i < actual.size(); ++i) {
            if (std::abs(actual(i) - expected_(i)) > tolerance_) {
                return false;
            }
        }
        return true;
    }

    std::string describe() const override {
        std::ostringstream ss;
        ss << "equals vector within tolerance " << tolerance_;
        return ss.str();
    }

   private:
    Eigen::VectorXd expected_;
    double tolerance_;
};

/// @brief Custom Catch2 matcher for Eigen matrices
class MatrixEquals : public Catch::Matchers::MatcherBase<Eigen::MatrixXd> {
   public:
    MatrixEquals(Eigen::MatrixXd const& expected, double tolerance)
        : expected_(expected), tolerance_(tolerance) {}

    bool match(Eigen::MatrixXd const& actual) const override {
        return matrices_equal(actual, expected_, tolerance_);
    }

    std::string describe() const override {
        std::ostringstream ss;
        ss << "equals matrix within tolerance " << tolerance_;
        if (last_actual_.size() > 0) {
            ss << "\n" << matrix_diff_string(last_actual_, expected_, tolerance_);
        }
        return ss.str();
    }

   private:
    Eigen::MatrixXd expected_;
    double tolerance_;
    mutable Eigen::MatrixXd last_actual_;
};

}  // namespace posetrak::test_helpers
