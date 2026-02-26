#include "posetrak/core/skeleton_layout.hpp"

#include <fmt/format.h>

#include <stdexcept>
#include <unordered_set>

namespace posetrak {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

/// @brief True if this joint should be skipped (contributes no DOFs to state).
bool is_fixed(Joint const& joint) {
    return joint.type == JointType::FIXED;
}

/// @brief True if this joint is the kinematic root (no parent).
bool is_root(Joint const& joint) {
    return !joint.parent_index.has_value();
}

}  // namespace

// ---------------------------------------------------------------------------
// Core builder
// ---------------------------------------------------------------------------

std::shared_ptr<const SkeletonLayout>
SkeletonLayout::build(std::shared_ptr<const Skeleton> skeleton, bool include_all,
                      std::vector<std::string> const& group_names) {
    if (!include_all && group_names.empty()) {
        throw std::invalid_argument("SkeletonLayout::from_groups: group_names must not be empty");
    }

    std::unordered_set<std::string> group_set(group_names.begin(), group_names.end());

    // Use new, not make_shared, because constructor is private.
    auto layout = std::shared_ptr<SkeletonLayout>(new SkeletonLayout());
    layout->skeleton_ = skeleton;  // shared ownership — no copy

    // ------------------------------------------------------------------
    // Pass 1: walk joints in state-vector order and record the full-skeleton
    //         state_index for every non-root, non-fixed joint.
    //         This allows Pass 2 (subset) to look up a joint's position in the
    //         full skeleton — needed for build_index_map_from().
    //
    //         We store these in a temporary map; the layout itself only stores
    //         layout-relative indices.
    // ------------------------------------------------------------------
    std::unordered_map<std::string, int> full_state_idx;
    {
        int fs_idx = 0;
        for (auto const& joint : skeleton->get_joints_ordered()) {
            if (is_root(joint) || is_fixed(joint))
                continue;
            full_state_idx[joint.name] = fs_idx;
            fs_idx += static_cast<int>(joint.dof);
        }
    }

    // ------------------------------------------------------------------
    // Pass 2: filter joints to the requested groups and build layout-relative
    //         JointDesc entries.
    // ------------------------------------------------------------------
    int layout_state_idx = 0;  // layout-relative position in joint_angles
    int layout_error_idx = 0;  // layout-relative position in error-state joint block

    for (auto const& joint : skeleton->get_joints_ordered()) {
        bool const in_layout = include_all || (group_set.count(joint.group) > 0);
        if (!in_layout)
            continue;

        if (is_root(joint)) {
            // Root is a floating body: contributes 6 DOFs to the error-state
            // (3 pos + 3 ori) but is NOT added to joints_ list.
            layout->has_floating_root_ = true;
            continue;
        }

        if (is_fixed(joint)) {
            // FIXED joints contribute no DOFs; skip them entirely.
            continue;
        }

        JointDesc desc;
        desc.name = joint.name;
        desc.type = joint.type;
        desc.storage_dof_count = static_cast<int>(joint.dof);
        desc.active_dof_count = static_cast<int>(joint.active_dof());
        desc.state_index = layout_state_idx;
        desc.error_index = layout_error_idx;
        desc.is_floating_root = false;
        desc.limits = joint.limits;
        desc.limit_count = static_cast<int>(joint.num_limits);
        desc.active_dof_mask = joint.get_active_dof_mask();

        layout->name_to_idx_[desc.name] = static_cast<int>(layout->joints_.size());
        layout->joints_.push_back(std::move(desc));

        layout_state_idx += static_cast<int>(joint.dof);
        layout_error_idx += static_cast<int>(joint.active_dof());
        layout->total_storage_dof_count_ += static_cast<int>(joint.dof);
        layout->joint_active_dof_count_ += static_cast<int>(joint.active_dof());
    }

    if (!include_all && layout->joints_.empty() && !layout->has_floating_root_) {
        throw std::invalid_argument(
            fmt::format("SkeletonLayout::from_groups: no joints matched the provided groups"));
    }

    return layout;
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

std::shared_ptr<const SkeletonLayout>
SkeletonLayout::from_full_skeleton(std::shared_ptr<const Skeleton> skeleton) {
    return build(std::move(skeleton), /*include_all=*/true, /*group_names=*/{});
}

std::shared_ptr<const SkeletonLayout>
SkeletonLayout::from_groups(std::shared_ptr<const Skeleton> skeleton,
                            std::vector<std::string> const& group_names) {
    return build(std::move(skeleton), /*include_all=*/false, group_names);
}

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

JointDesc const* SkeletonLayout::get_joint(std::string const& name) const {
    auto it = name_to_idx_.find(name);
    if (it == name_to_idx_.end())
        return nullptr;
    return &joints_[it->second];
}

// ---------------------------------------------------------------------------
// Index mapping
// ---------------------------------------------------------------------------

std::vector<int> SkeletonLayout::build_index_map_from(SkeletonLayout const& subset) const {
    // Joints are matched by name only. Caller must ensure both layouts were
    // derived from the same Skeleton — this function cannot verify that.
    std::vector<int> map;
    map.reserve(static_cast<size_t>(subset.total_storage_dof_count_));

    for (auto const& subdesc : subset.joints_) {
        JointDesc const* desc = get_joint(subdesc.name);
        if (!desc) {
            throw std::invalid_argument(
                fmt::format("SkeletonLayout::build_index_map_from: joint '{}' exists in "
                            "subset but not in this layout",
                            subdesc.name));
        }
        // One entry per storage DOF of this joint
        for (int i = 0; i < subdesc.storage_dof_count; ++i) {
            map.push_back(desc->state_index + i);
        }
    }
    return map;
}

State SkeletonLayout::slice_state(State const& full_state) const {
    // If already the right size, return a copy
    if (full_state.num_dof() == total_storage_dof_count_) {
        return full_state;
    }

    // Build full-skeleton layout for the index map
    auto full_layout = SkeletonLayout::from_full_skeleton(skeleton_);

    // Build index map: this layout (subset) -> full layout
    std::vector<int> map = full_layout->build_index_map_from(*this);

    // Extract joint angles and velocities
    Eigen::VectorXd sliced_angles(total_storage_dof_count_);
    Eigen::VectorXd sliced_velocities(total_storage_dof_count_);

    for (size_t i = 0; i < map.size(); ++i) {
        sliced_angles(i) = full_state.joint_angles()(map[i]);
        sliced_velocities(i) = full_state.joint_velocities()(map[i]);
    }

    // Root pose/velocity pass through unchanged
    return State(full_state.root_position(), full_state.root_orientation(), sliced_angles,
                 full_state.root_velocity(), full_state.root_angular_velocity(), sliced_velocities);
}

}  // namespace posetrak
