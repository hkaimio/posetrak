#include "posetrak/calibration/scale_calibration.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace posetrak {

// ---------------------------------------------------------------------------
// CSV helpers
// ---------------------------------------------------------------------------

namespace {

/// Split a comma-separated string into fields (no quote handling needed here).
std::vector<std::string> split_csv(std::string const& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

/// Return true for scale-group position columns (not velocity columns).
/// Header column name matches "scale_group_<name>" but NOT "scale_group_<name>_velocity".
bool is_scale_pos_column(std::string const& col_name) {
    static std::string const prefix = "scale_group_";
    static std::string const vel_suffix = "_velocity";
    if (col_name.rfind(prefix, 0) != 0) {
        return false;
    }
    // Reject velocity columns
    if (col_name.size() >= vel_suffix.size() &&
        col_name.compare(col_name.size() - vel_suffix.size(), vel_suffix.size(), vel_suffix) == 0) {
        return false;
    }
    return true;
}

/// Extract scale group name from a column header like "scale_group_femur".
std::string group_name_from_column(std::string const& col_name) {
    static std::string const prefix = "scale_group_";
    return col_name.substr(prefix.size());
}

}  // namespace

// ---------------------------------------------------------------------------
// check_scale_convergence
// ---------------------------------------------------------------------------

std::vector<ScaleGroupResult> check_scale_convergence(std::string const& state_vectors_csv_path,
                                                      ScaleCalibrationOptions const& opts) {
    std::ifstream f(state_vectors_csv_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open state_vectors.csv: " + state_vectors_csv_path);
    }

    // --- Parse header ---
    std::string header_line;
    if (!std::getline(f, header_line)) {
        throw std::runtime_error("state_vectors.csv is empty");
    }
    auto header = split_csv(header_line);

    // Find column indices for scale position DOFs
    std::vector<int> scale_col_indices;
    std::vector<std::string> group_names;
    for (int i = 0; i < static_cast<int>(header.size()); ++i) {
        if (is_scale_pos_column(header[i])) {
            scale_col_indices.push_back(i);
            group_names.push_back(group_name_from_column(header[i]));
        }
    }

    if (scale_col_indices.empty()) {
        throw std::runtime_error(
            "No scale_group_* columns found in state_vectors.csv — "
            "was the tracker run with calibration enabled?");
    }

    int const n_groups = static_cast<int>(scale_col_indices.size());
    int const min_col_needed =
        *std::max_element(scale_col_indices.begin(), scale_col_indices.end());

    // Rolling window per group: keep last opts.window_frames values
    std::vector<std::deque<double>> windows(n_groups);

    // --- Read data rows ---
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        auto fields = split_csv(line);
        if (static_cast<int>(fields.size()) <= min_col_needed)
            continue;

        for (int g = 0; g < n_groups; ++g) {
            double val = std::stod(fields[scale_col_indices[g]]);
            windows[g].push_back(val);
            if (static_cast<int>(windows[g].size()) > opts.window_frames) {
                windows[g].pop_front();
            }
        }
    }

    // --- Compute statistics ---
    std::vector<ScaleGroupResult> results;
    results.reserve(n_groups);

    for (int g = 0; g < n_groups; ++g) {
        auto const& w = windows[g];
        if (w.empty()) {
            results.push_back({group_names[g], 1.0, 0.0, false});
            continue;
        }

        double sum = 0.0;
        for (double v : w)
            sum += v;
        double mean = sum / static_cast<double>(w.size());

        double sq_sum = 0.0;
        for (double v : w)
            sq_sum += (v - mean) * (v - mean);
        double std_dev = std::sqrt(sq_sum / static_cast<double>(w.size()));

        bool converged = std_dev < opts.converge_std;

        results.push_back({group_names[g], mean, std_dev, converged});
    }

    // Sort for stable output ordering
    std::sort(results.begin(), results.end(),
              [](ScaleGroupResult const& a, ScaleGroupResult const& b) { return a.name < b.name; });

    return results;
}

// ---------------------------------------------------------------------------
// write_calibrated_yaml
// ---------------------------------------------------------------------------

void write_calibrated_yaml(std::string const& input_yaml_path, std::string const& output_yaml_path,
                           std::vector<ScaleGroupResult> const& results) {
    // Build: group_name → final_scale
    std::unordered_map<std::string, double> scale_by_group;
    for (auto const& r : results) {
        scale_by_group[r.name] = r.final_scale;
    }

    // Load skeleton YAML
    YAML::Node root = YAML::LoadFile(input_yaml_path);

    // Build: joint_name → scale_factor, from the scale_groups section
    std::unordered_map<std::string, double> scale_by_joint;
    if (root["scale_groups"] && root["scale_groups"].IsSequence()) {
        for (auto const& sg : root["scale_groups"]) {
            if (!sg["name"] || !sg["joints"])
                continue;
            std::string const gname = sg["name"].as<std::string>();
            auto it = scale_by_group.find(gname);
            if (it == scale_by_group.end())
                continue;
            double const s = it->second;
            for (auto const& jnode : sg["joints"]) {
                scale_by_joint[jnode.as<std::string>()] = s;
            }
        }
    }

    // Pre-pass: collect original offset vectors and parent names before any mutation.
    // These are used to identify which child's offset a bone_tip_offset matches.
    struct ChildScaleEntry {
        std::array<double, 3> original_offset;
        double scale;
    };
    // parent_name → list of (original child offset, child scale factor) for scaled children only
    std::unordered_map<std::string, std::vector<ChildScaleEntry>> children_of;

    if (root["joints"] && root["joints"].IsSequence()) {
        for (auto const& joint_node : root["joints"]) {
            if (!joint_node["name"])
                continue;
            std::string const jname = joint_node["name"].as<std::string>();
            auto sit = scale_by_joint.find(jname);
            if (sit == scale_by_joint.end())
                continue;  // unscaled joint — no entry needed
            if (!joint_node["parent"] || joint_node["parent"].IsNull())
                continue;
            std::string const pname = joint_node["parent"].as<std::string>();
            if (!joint_node["offset"] || !joint_node["offset"].IsSequence() ||
                joint_node["offset"].size() != 3)
                continue;
            ChildScaleEntry entry;
            entry.scale = sit->second;
            for (std::size_t i = 0; i < 3; ++i) {
                entry.original_offset[i] = joint_node["offset"][i].as<double>();
            }
            children_of[pname].push_back(entry);
        }
    }

    // Modify joint offsets and bone_tip_offsets in-place
    if (root["joints"] && root["joints"].IsSequence()) {
        for (auto joint_node : root["joints"]) {
            if (!joint_node["name"])
                continue;
            std::string const jname = joint_node["name"].as<std::string>();

            // Scale this joint's own offset if it belongs to a scale group
            auto sit = scale_by_joint.find(jname);
            if (sit != scale_by_joint.end()) {
                double const s = sit->second;
                if (joint_node["offset"] && joint_node["offset"].IsSequence() &&
                    joint_node["offset"].size() == 3) {
                    for (std::size_t i = 0; i < 3; ++i) {
                        double v = joint_node["offset"][i].as<double>();
                        joint_node["offset"][i] = v * s;
                    }
                }
            }

            // Scale bone_tip_offset if it closely matches one of this joint's scaled children.
            // A match means the Euclidean distance to a child's original offset is < 5mm.
            // This correctly handles multi-child joints (e.g. spine2 → shoulder.L/R + neck1):
            // only the child whose offset direction/magnitude matches the bone_tip_offset
            // contributes its scale factor.
            if (joint_node["bone_tip_offset"] && joint_node["bone_tip_offset"].IsSequence() &&
                joint_node["bone_tip_offset"].size() == 3) {
                auto cit = children_of.find(jname);
                if (cit != children_of.end()) {
                    double const bx = joint_node["bone_tip_offset"][0].as<double>();
                    double const by = joint_node["bone_tip_offset"][1].as<double>();
                    double const bz = joint_node["bone_tip_offset"][2].as<double>();

                    constexpr double kTolerance = 0.005;  // 5 mm
                    double best_dist = kTolerance;
                    double best_scale = -1.0;
                    for (auto const& entry : cit->second) {
                        double const dx = bx - entry.original_offset[0];
                        double const dy = by - entry.original_offset[1];
                        double const dz = bz - entry.original_offset[2];
                        double const dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_scale = entry.scale;
                        }
                    }
                    if (best_scale > 0.0) {
                        joint_node["bone_tip_offset"][0] = bx * best_scale;
                        joint_node["bone_tip_offset"][1] = by * best_scale;
                        joint_node["bone_tip_offset"][2] = bz * best_scale;
                    }
                }
            }
        }
    }

    // Emit without scale_groups (iterate top-level map, skip that key)
    YAML::Emitter out;
    out << YAML::BeginMap;
    for (YAML::const_iterator it = root.begin(); it != root.end(); ++it) {
        std::string const key = it->first.as<std::string>();
        if (key == "scale_groups")
            continue;
        out << YAML::Key << it->first << YAML::Value << it->second;
    }
    out << YAML::EndMap;

    std::ofstream ofs(output_yaml_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot write calibrated YAML to: " + output_yaml_path);
    }
    ofs << out.c_str();
    if (!ofs) {
        throw std::runtime_error("Write error for calibrated YAML: " + output_yaml_path);
    }
}

}  // namespace posetrak
