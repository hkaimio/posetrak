// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/tracking/keypoint_choice_phase.hpp"

#include <fmt/core.h>

#include "posetrak/tracking/keypoint_window_choice.hpp"
#include "posetrak/tracking/seg_mask.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define PSAPI_VERSION 2
#include <psapi.h>
#include <windows.h>
#elif defined(__linux__)
#include <sys/resource.h>
#endif

namespace posetrak {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Box height in pixels per unit of crop scale (the model input covers 0.75 of a crop that is 1.6
// times the box): the same relation the duplicate merge of the per-step loader uses.
constexpr double kBoxHeightPerCropScale = 160.0;

std::vector<std::string> split(std::string const& text, char sep) {
    std::vector<std::string> out;
    std::string item;
    std::stringstream ss(text);
    while (std::getline(ss, item, sep))
        out.push_back(item);
    if (!text.empty() && text.back() == sep)
        out.emplace_back();
    return out;
}

}  // namespace

double peak_memory_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
        return static_cast<double>(counters.PeakWorkingSetSize) / (1024.0 * 1024.0);
    return 0.0;
#elif defined(__linux__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
        return static_cast<double>(usage.ru_maxrss) / 1024.0;  // kilobytes
    return 0.0;
#else
    return 0.0;
#endif
}

KeypointChoiceModel load_keypoint_choice_model(std::string const& path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot open keypoint choice model: " + path);
    KeypointChoiceModel model;
    std::string line;
    int number = 0;
    while (std::getline(file, line)) {
        ++number;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty() || line[0] == '#')
            continue;
        auto const fields = split(line, ',');
        if (fields.size() >= 4 && fields[0] == "pair") {
            model.pairs.push_back({fields[1], fields[2], std::stod(fields[3])});
        } else if (fields.size() >= 3 && fields[0] == "class") {
            model.keypoint_class[fields[1]] = fields[2];
        } else {
            throw std::runtime_error(fmt::format(
                "{}:{}: expected 'pair,a,b,limit' or 'class,landmark,class'", path, number));
        }
    }
    return model;
}

KeypointChoiceReport run_keypoint_choice_phase(PersonContext& ctx, KeypointChoiceInputs const& in) {
    auto const t_phase = Clock::now();
    KeypointChoiceReport report;
    auto const& cfg = ctx.tracker_config.kp_choice;
    bool const needs_masks = cfg.strict_filter || cfg.mask_term_weight != 0.0;
    if (needs_masks && (in.seg_run_id.empty() || in.seg_mask_label <= 0)) {
        throw std::runtime_error(
            "The strict mask filter and the mask term need the person's masks: pass --seg-run "
            "(and --seg-label if the subject registry does not name the label).");
    }

    // The window holds the tracked range and, on both sides, what the support averages look at.
    double margin_s = cfg.support_window_s;
    if (cfg.mask_term_weight != 0.0)
        margin_s = std::max(margin_s, cfg.mask_term_window_s);
    double const window_start = ctx.start_time - margin_s;
    double const window_end = ctx.end_time + margin_s;

    // ---- 1. load -------------------------------------------------------------------------
    auto t0 = Clock::now();
    SessionReader reader(in.db_path);
    KeypointChoiceData data = reader.load_keypoint_choice_data(
        in.sequence_id, ctx.cameras_by_name, ctx.skeleton, in.person_id, window_start, window_end);
    if (data.candidates.empty() || data.marker_keypoint.empty()) {
        throw std::runtime_error(
            "The tracker config asks for keypoint candidates, but sequence '" + in.sequence_id +
            "' holds none for this skeleton's markers. Run detection with --candidates and "
            "finalise it, or track without candidates.");
    }
    report.load_s = seconds_since(t0);
    report.candidates = static_cast<long>(data.candidates.size());

    int const n_keypoints = static_cast<int>(data.keypoint_landmark.size());
    int const n_cameras = static_cast<int>(data.cameras.size());
    if (n_keypoints > 255 || n_cameras > 255)
        throw std::runtime_error("Keypoint choice: more than 255 keypoints or cameras");
    std::vector<int> keypoint_marker(static_cast<size_t>(n_keypoints), -1);
    for (auto const& [marker, keypoint] : data.marker_keypoint)
        keypoint_marker[static_cast<size_t>(keypoint)] = marker;

    // ---- 2. input of the choice, tracklet decisions --------------------------------------
    kpchoice::Input input;
    input.n_keypoints = n_keypoints;
    input.own_mask_label = in.seg_mask_label;
    input.frames.resize(static_cast<size_t>(n_cameras));
    input.active.assign(static_cast<size_t>(n_cameras), 1);
    for (int c = 0; c < n_cameras; ++c) {
        auto const& cam = data.cameras[static_cast<size_t>(c)];
        for (auto const& f : cam.frames)
            input.frames[static_cast<size_t>(c)].push_back(
                {f.video_frame, f.timestamp, f.crop_scale * kBoxHeightPerCropScale});
        if (std::find(cfg.skip_camera_ids.begin(), cfg.skip_camera_ids.end(), cam.camera_id) !=
            cfg.skip_camera_ids.end())
            input.active[static_cast<size_t>(c)] = 0;
        input.projection.push_back(ctx.cameras_by_id.at(cam.camera_id).get_projection_matrix());
    }
    std::unordered_map<std::string, int> keypoint_of;
    for (int k = 0; k < n_keypoints; ++k)
        if (!data.keypoint_landmark[static_cast<size_t>(k)].empty())
            keypoint_of[data.keypoint_landmark[static_cast<size_t>(k)]] = k;
    input.neighbours.assign(static_cast<size_t>(n_keypoints), {});
    for (auto const& pair : in.model.pairs) {
        auto a = keypoint_of.find(pair.a);
        auto b = keypoint_of.find(pair.b);
        if (a == keypoint_of.end() || b == keypoint_of.end())
            continue;
        input.neighbours[static_cast<size_t>(a->second)].emplace_back(b->second, pair.limit);
        input.neighbours[static_cast<size_t>(b->second)].emplace_back(a->second, pair.limit);
    }
    if (cfg.mask_term_weight != 0.0) {
        std::set<std::string> classes;
        for (auto const& name : split(cfg.mask_term_classes, ','))
            classes.insert(name);
        input.mask_keypoint.assign(static_cast<size_t>(n_keypoints), 0);
        for (int k = 0; k < n_keypoints; ++k) {
            auto it = in.model.keypoint_class.find(data.keypoint_landmark[static_cast<size_t>(k)]);
            if (it != in.model.keypoint_class.end() && classes.count(it->second))
                input.mask_keypoint[static_cast<size_t>(k)] = 1;
        }
    }

    bool const has_rules = in.rules != nullptr && !in.rules->empty();
    input.candidates.reserve(data.candidates.size());
    std::set<std::tuple<int, int, int, int>> fixed_seen;
    for (size_t i = 0; i < data.candidates.size(); ++i) {
        auto const& row = data.candidates[i];
        auto const& cam = data.cameras[static_cast<size_t>(row.camera)];
        kpchoice::Candidate c;
        c.x = row.x;
        c.y = row.y;
        c.ux = row.ux;
        c.uy = row.uy;
        c.peak = static_cast<float>(row.peak / in.confidence_scale);
        c.tracklet = row.tracklet_id;
        c.frame = row.frame;
        c.source = static_cast<std::int32_t>(i);
        c.rank = static_cast<std::int16_t>(row.rank);
        c.camera = static_cast<std::uint8_t>(row.camera);
        c.keypoint = static_cast<std::uint8_t>(row.keypoint);
        if (has_rules && row.tracklet_id >= 0) {
            int const video_frame = cam.frames[static_cast<size_t>(row.frame)].video_frame;
            auto const verdict = in.rules->verdict(cam.camera_id, row.tracklet_id, video_frame);
            int const marker = keypoint_marker[static_cast<size_t>(row.keypoint)];
            if (verdict.kind == TrackletVerdict::Kind::EXCLUDED) {
                c.removed = true;
            } else if (verdict.kind == TrackletVerdict::Kind::ASSIGNED) {
                if (verdict.assigned_marker == marker) {
                    if (fixed_seen.emplace(row.camera, row.keypoint, row.frame, row.tracklet_id)
                            .second)
                        input.fixed.push_back(
                            {row.camera, row.keypoint, row.frame, row.tracklet_id});
                } else {
                    c.removed = true;  // the tracklet was decided to be another marker
                }
            }
            if (!c.removed &&
                std::find(verdict.forbidden_markers.begin(), verdict.forbidden_markers.end(),
                          marker) != verdict.forbidden_markers.end())
                c.removed = true;
            report.removed_by_decisions += c.removed;
        }
        input.candidates.push_back(c);
    }

    // ---- 3. mask labels ------------------------------------------------------------------
    if (needs_masks) {
        t0 = Clock::now();
        SessionReader mask_reader(in.db_path);
        size_t i = 0;
        size_t const n = input.candidates.size();
        // Rows come grouped by camera and frame, in that order.
        while (i < n) {
            auto const& first = data.candidates[i];
            size_t j = i;
            while (j < n && data.candidates[j].camera == first.camera &&
                   data.candidates[j].frame == first.frame)
                ++j;
            auto const& cam = data.cameras[static_cast<size_t>(first.camera)];
            int const video_frame = cam.frames[static_cast<size_t>(first.frame)].video_frame;
            DecodedSegMask mask;
            bool have = false;
            auto const svid = in.camera_shot_video_ids.find(cam.camera_id);
            auto const camera = ctx.cameras_by_id.find(cam.camera_id);
            if (svid != in.camera_shot_video_ids.end() && !svid->second.empty() &&
                camera != ctx.cameras_by_id.end()) {
                auto blob = mask_reader.load_seg_mask_blob(in.seg_run_id, svid->second, video_frame,
                                                           in.seg_revision);
                if (blob.has_value()) {
                    mask.native_w = camera->second.intrinsics().width;
                    mask.native_h = camera->second.intrinsics().height;
                    have = decode_seg_mask(*blob, mask);
                }
            }
            for (size_t r = i; r < j; ++r) {
                input.candidates[r].mask_label =
                    have ? static_cast<std::int16_t>(mask_label_at(
                               mask, Eigen::Vector2d(data.candidates[r].x, data.candidates[r].y)))
                         : kpchoice::kMaskUnknown;
            }
            i = j;
        }
        report.masks_s = seconds_since(t0);
    }

    // ---- 4. choice -----------------------------------------------------------------------
    kpchoice::Settings settings;
    settings.support_window_s = cfg.support_window_s;
    settings.mask_window_s = cfg.mask_term_window_s;
    settings.delta = cfg.delta;
    settings.beta = cfg.beta;
    settings.gamma = cfg.gamma;
    settings.mask_weight = cfg.mask_term_weight;
    settings.rounds = cfg.rounds;
    kpchoice::WindowChoice choice(std::move(input), settings);
    t0 = Clock::now();
    choice.flag_consistent();
    report.consensus_s = seconds_since(t0);
    t0 = Clock::now();
    if (cfg.strict_filter)
        choice.apply_strict_filter(cfg.strict_filter_window_frames, cfg.strict_filter_min_fraction);
    else
        choice.count_keypoint_frames();
    choice.run();
    report.choice_s = seconds_since(t0);

    // ---- 5. observations and stored picks ------------------------------------------------
    std::unordered_map<int, std::string> camera_name;
    for (auto const& [name, camera] : ctx.cameras_by_name)
        camera_name[camera.id()] = name;

    // Hand-placed keypoint edits stay: they are fixed observations the picks must not replace.
    std::set<std::tuple<int, int, int>> edited;
    std::set<int> bound_markers;
    for (auto const& [marker, keypoint] : data.marker_keypoint)
        bound_markers.insert(marker);
    for (auto const& [name, sequence] : ctx.observations.sequences())
        for (auto const& o : sequence.observations)
            if (o.force_inlier && o.mode == MeasurementMode::POSITION)
                edited.emplace(o.camera_id, o.marker_id, o.frame_idx);

    for (int c = 0; c < n_cameras; ++c) {
        auto const& cam = data.cameras[static_cast<size_t>(c)];
        int const n_frames = static_cast<int>(cam.frames.size());
        KeypointPicks picks;
        picks.camera_instance_id = cam.camera_instance_id;
        picks.n_frames = n_frames;
        picks.n_keypoints = n_keypoints;
        picks.video_frames.reserve(cam.frames.size());
        for (auto const& f : cam.frames)
            picks.video_frames.push_back(f.video_frame);
        picks.tracklet_ids.assign(static_cast<size_t>(n_frames) * n_keypoints, -1);
        picks.ranks.assign(static_cast<size_t>(n_frames) * n_keypoints, -1);
        std::vector<Observation> fresh;
        if (!choice.input().active[static_cast<size_t>(c)])
            continue;
        for (int f = 0; f < n_frames; ++f) {
            for (int k = 0; k < n_keypoints; ++k) {
                int const r = choice.pick(c, k, f);
                if (r < 0)
                    continue;
                auto const& cand = choice.candidates()[static_cast<size_t>(r)];
                size_t const at = static_cast<size_t>(f) * n_keypoints + k;
                picks.tracklet_ids[at] = cand.tracklet >= 0 ? cand.tracklet : -2;
                picks.ranks[at] = static_cast<std::int8_t>(std::min<int>(cand.rank, 127));
                ++picks.n_picked;
                int const marker = keypoint_marker[static_cast<size_t>(k)];
                if (marker < 0)
                    continue;
                auto const& row = data.candidates[static_cast<size_t>(cand.source)];
                if (row.peak < in.min_confidence)
                    continue;
                auto const& frame = cam.frames[static_cast<size_t>(f)];
                if (edited.count({cam.camera_id, marker, frame.video_frame}))
                    continue;
                Observation obs;
                obs.camera_id = cam.camera_id;
                obs.marker_id = marker;
                obs.frame_idx = frame.video_frame;
                obs.timestamp = frame.timestamp;
                obs.position_distorted = Eigen::Vector2d(row.x, row.y);
                obs.position = Eigen::Vector2d(row.ux, row.uy);
                obs.confidence = row.peak;
                obs.crop_scale = frame.crop_scale;
                fresh.push_back(obs);
            }
        }
        // The camera's candidate-bearing statistics.
        auto const& stats = choice.camera_stats()[static_cast<size_t>(c)];
        picks.n_candidate_keypoint_frames = stats.keypoint_frames;
        picks.n_filter_emptied = stats.emptied;
        report.keypoint_frames += stats.keypoint_frames;
        report.emptied_by_filter += stats.emptied;
        report.picked += picks.n_picked;
        report.observations += static_cast<long>(fresh.size());
        if (ctx.result_writer)
            ctx.result_writer->write_kp_picks(picks);

        // Replace the camera's fixed POSITION observations of the bound markers.
        auto name_it = camera_name.find(cam.camera_id);
        if (name_it == camera_name.end())
            continue;
        auto& sequences = ctx.observations.sequences();
        auto seq_it = sequences.find(name_it->second);
        if (seq_it == sequences.end()) {
            ObservationSequence created;
            created.camera_id = cam.camera_id;
            created.camera_name = name_it->second;
            seq_it = sequences.emplace(name_it->second, std::move(created)).first;
        }
        auto& existing = seq_it->second.observations;
        existing.erase(std::remove_if(existing.begin(), existing.end(),
                                      [&](Observation const& o) {
                                          return o.mode == MeasurementMode::POSITION &&
                                                 !o.force_inlier &&
                                                 bound_markers.count(o.marker_id);
                                      }),
                       existing.end());
        std::vector<Observation> merged;
        merged.reserve(existing.size() + fresh.size());
        std::merge(existing.begin(), existing.end(), fresh.begin(), fresh.end(),
                   std::back_inserter(merged), [](Observation const& a, Observation const& b) {
                       return a.timestamp < b.timestamp;
                   });
        existing = std::move(merged);
    }

    report.total_s = seconds_since(t_phase);
    report.peak_memory_mb = peak_memory_mb();
    if (!in.quiet) {
        double const share = report.keypoint_frames
                                 ? 100.0 * static_cast<double>(report.emptied_by_filter) /
                                       static_cast<double>(report.keypoint_frames)
                                 : 0.0;
        fmt::print(
            "Keypoint choice (window): {} candidates, {} keypoint-frames, {} picked, {} "
            "observations; {} removed by tracklet decisions\n",
            report.candidates, report.keypoint_frames, report.picked, report.observations,
            report.removed_by_decisions);
        if (cfg.strict_filter) {
            fmt::print("  Strict mask filter emptied {} of {} keypoint-frames ({:.1f}%)\n",
                       report.emptied_by_filter, report.keypoint_frames, share);
        }
        fmt::print(
            "  Phase time {:.1f}s (load {:.1f}, masks {:.1f}, consensus {:.1f}, choice {:.1f}); "
            "peak memory {:.0f} MB\n",
            report.total_s, report.load_s, report.masks_s, report.consensus_s, report.choice_s,
            report.peak_memory_mb);
    }
    return report;
}

}  // namespace posetrak
