// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/tracking/tracklet_assignment.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace posetrak {

void TrackletAssignments::add(int camera_id, int tracklet_id, TrackletRule const& rule) {
    rules_[camera_id][tracklet_id].push_back(rule);
    ++n_rules_;
}

TrackletVerdict TrackletAssignments::verdict(int camera_id, int tracklet_id, int frame) const {
    TrackletVerdict out;
    auto const camera_it = rules_.find(camera_id);
    if (camera_it == rules_.end())
        return out;
    auto const tracklet_it = camera_it->second.find(tracklet_id);
    if (tracklet_it == camera_it->second.end())
        return out;
    auto const& rules = tracklet_it->second;
    // Latest rule first, so that a later rule overrides an earlier one: an ASSIGN or SUGGEST of a
    // marker that a later FORBID ruled out does not count, and a later REJECT or OTHER ends the
    // search.
    auto const forbidden = [&](int marker) {
        return std::find(out.forbidden_markers.begin(), out.forbidden_markers.end(), marker) !=
               out.forbidden_markers.end();
    };
    for (auto it = rules.rbegin(); it != rules.rend(); ++it) {
        if (!it->covers(frame))
            continue;
        switch (it->kind) {
            case TrackletRuleKind::ASSIGN:
                if (!forbidden(it->marker_id)) {
                    out.kind = TrackletVerdict::Kind::ASSIGNED;
                    out.assigned_marker = it->marker_id;
                    return out;
                }
                break;
            case TrackletRuleKind::REJECT:
            case TrackletRuleKind::OTHER:
                out.kind = TrackletVerdict::Kind::EXCLUDED;
                return out;
            case TrackletRuleKind::FORBID:
                out.forbidden_markers.push_back(it->marker_id);
                break;
            case TrackletRuleKind::SUGGEST:
                if (out.suggested_marker < 0 && !forbidden(it->marker_id))
                    out.suggested_marker = it->marker_id;
                break;
        }
    }
    return out;
}

namespace {

std::vector<std::string> split_csv(std::string const& line) {
    std::vector<std::string> fields;
    std::string field;
    std::stringstream stream(line);
    while (std::getline(stream, field, ','))
        fields.push_back(field);
    if (!line.empty() && line.back() == ',')
        fields.emplace_back();
    return fields;
}

std::string trimmed(std::string s) {
    auto const not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

bool parse_kind(std::string const& name, TrackletRuleKind& kind) {
    if (name == "assign")
        kind = TrackletRuleKind::ASSIGN;
    else if (name == "forbid")
        kind = TrackletRuleKind::FORBID;
    else if (name == "reject")
        kind = TrackletRuleKind::REJECT;
    else if (name == "other")
        kind = TrackletRuleKind::OTHER;
    else if (name == "suggest")
        kind = TrackletRuleKind::SUGGEST;
    else
        return false;
    return true;
}

}  // namespace

TrackletAssignmentFile
load_tracklet_assignments(std::string const& path,
                          std::unordered_map<std::string, int> const& camera_ids,
                          std::unordered_map<std::string, int> const& marker_ids) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("cannot open tracklet assignment file: " + path);
    TrackletAssignmentFile out;
    std::string line;
    while (std::getline(file, line)) {
        line = trimmed(line);
        if (line.empty())
            continue;
        auto fields = split_csv(line);
        if (fields.size() < 6 || fields[0] == "camera_label") {
            if (fields[0] != "camera_label")
                ++out.rows_skipped;
            continue;
        }
        ++out.rows_read;
        for (auto& f : fields)
            f = trimmed(f);
        TrackletRule rule;
        auto const camera = camera_ids.find(fields[0]);
        if (camera == camera_ids.end() || !parse_kind(fields[4], rule.kind)) {
            ++out.rows_skipped;
            continue;
        }
        if (rule.kind != TrackletRuleKind::REJECT && rule.kind != TrackletRuleKind::OTHER) {
            auto const marker = marker_ids.find(fields[5]);
            if (marker == marker_ids.end()) {
                ++out.rows_skipped;
                continue;
            }
            rule.marker_id = marker->second;
        }
        try {
            int const tracklet_id = std::stoi(fields[1]);
            if (!fields[2].empty())
                rule.first_frame = std::stoi(fields[2]);
            if (!fields[3].empty())
                rule.last_frame = std::stoi(fields[3]);
            out.assignments.add(camera->second, tracklet_id, rule);
        } catch (std::exception const&) {
            ++out.rows_skipped;
        }
    }
    return out;
}

}  // namespace posetrak
