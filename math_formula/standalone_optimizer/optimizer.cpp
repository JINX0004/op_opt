#include "optimizer.h"
#include "event_builder.h"
#include "plan_finalizer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace mmopt {
namespace {

std::uint64_t ParseU64(const std::string& text, const std::string& what) {
    if (text.empty() || text[0] == '-') throw std::runtime_error("invalid " + what + ": " + text);
    std::size_t used = 0;
    std::uint64_t value = 0;
    try {
        value = std::stoull(text, &used, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid " + what + ": " + text);
    }
    if (used != text.size()) throw std::runtime_error("invalid " + what + ": " + text);
    return value;
}

double ParseNonnegative(const std::string& text, const std::string& what) {
    std::size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid " + what + ": " + text);
    }
    if (used != text.size() || !std::isfinite(value) || value < 0.0)
        throw std::runtime_error("invalid " + what + ": " + text);
    return value;
}

std::vector<std::string> SplitComma(const std::string& text) {
    if (text == "-") return {};
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, ',')) {
        if (part.empty()) throw std::runtime_error("empty event dependency");
        parts.push_back(part);
    }
    if (parts.empty() || text.back() == ',') throw std::runtime_error("empty event dependency");
    return parts;
}

void RequireCount(const std::vector<std::string>& words, std::size_t count,
                  std::size_t line_number) {
    if (words.size() != count) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                 ": expected " + std::to_string(count - 1) +
                                 " arguments for " + words.front());
    }
}

std::uint64_t RequiredTile(const Candidate& candidate, const std::string& field) {
    auto it = candidate.tile.find(field);
    if (it == candidate.tile.end() || it->second == 0)
        throw std::runtime_error("missing or zero tile field " + field);
    return it->second;
}

void Validate(const Workload& workload, const Hardware& hardware, const Candidate& candidate) {
    if (workload.m == 0 || workload.n == 0 || workload.k == 0 || workload.dtype.empty())
        throw std::runtime_error("invalid workload");
    if (hardware.aic_count == 0) throw std::runtime_error("zero available AICs");
    if (candidate.id.empty() || candidate.family.empty() || candidate.tiling_key.empty())
        throw std::runtime_error("missing candidate identity");
    if (!candidate.backend_verified)
        throw std::runtime_error("candidate not marked backend-verified");
    if (candidate.workspace_bytes > hardware.workspace_limit_bytes)
        throw std::runtime_error("workspace budget exceeded");
    const auto cores = RequiredTile(candidate, "usedCoreNum");
    if (cores > hardware.aic_count) throw std::runtime_error("AIC count exceeded");
    for (const auto* field : {"singleCoreM", "singleCoreN", "singleCoreK",
                              "baseM", "baseN", "baseK"})
        RequiredTile(candidate, field);
    if (candidate.events.empty()) throw std::runtime_error("empty event graph");
}

Score Evaluate(const Hardware& hardware, const Candidate& candidate) {
    const auto& events = candidate.events;
    const std::size_t count = events.size();
    std::unordered_map<std::string, std::size_t> by_id;
    std::vector<double> duration(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& event = events[i];
        if (event.id.empty() || event.resource.empty() || event.primitive.empty() ||
            event.unit.empty() || !std::isfinite(event.work_units) || event.work_units < 0.0)
            throw std::runtime_error("invalid event at index " + std::to_string(i));
        const auto primitive = hardware.primitives.find(event.primitive);
        if (primitive == hardware.primitives.end())
            throw std::runtime_error("unknown hardware primitive " + event.primitive);
        const auto& timing = primitive->second;
        if (event.unit != timing.unit)
            throw std::runtime_error("unit mismatch for primitive " + event.primitive);
        if (!std::isfinite(timing.fixed_latency_us) || timing.fixed_latency_us < 0.0 ||
            !std::isfinite(timing.rate_per_us) || timing.rate_per_us <= 0.0)
            throw std::runtime_error("invalid hardware primitive " + event.primitive);
        duration[i] = timing.fixed_latency_us + event.work_units / timing.rate_per_us;
        if (!std::isfinite(duration[i])) throw std::runtime_error("event time overflow");
        if (!by_id.emplace(event.id, i).second)
            throw std::runtime_error("duplicate event id " + event.id);
    }

    std::vector<std::vector<std::size_t>> edges(count);
    std::vector<std::size_t> indegree(count, 0);
    std::unordered_map<std::string, std::size_t> last_resource;
    for (std::size_t i = 0; i < count; ++i) {
        std::set<std::size_t> predecessors;
        for (const auto& dependency : events[i].after) {
            const auto found = by_id.find(dependency);
            if (found == by_id.end())
                throw std::runtime_error("unknown dependency " + dependency);
            predecessors.insert(found->second);
        }
        const auto prior = last_resource.find(events[i].resource);
        if (prior != last_resource.end()) predecessors.insert(prior->second);
        last_resource[events[i].resource] = i;
        for (const auto predecessor : predecessors) {
            edges[predecessor].push_back(i);
            ++indegree[i];
        }
    }

    std::priority_queue<std::size_t, std::vector<std::size_t>, std::greater<std::size_t>> ready;
    for (std::size_t i = 0; i < count; ++i)
        if (indegree[i] == 0) ready.push(i);

    std::vector<double> start(count, 0.0), finish(count, 0.0);
    std::vector<std::size_t> critical_predecessor(count, count);
    std::size_t visited = 0;
    while (!ready.empty()) {
        const auto i = ready.top();
        ready.pop();
        ++visited;
        finish[i] = start[i] + duration[i];
        if (!std::isfinite(finish[i])) throw std::runtime_error("event time overflow");
        for (const auto next : edges[i]) {
            if (finish[i] > start[next]) {
                start[next] = finish[i];
                critical_predecessor[next] = i;
            }
            if (--indegree[next] == 0) ready.push(next);
        }
    }
    if (visited != count) throw std::runtime_error("cyclic event or resource dependencies");

    Score score;
    std::size_t terminal = 0;
    for (std::size_t i = 0; i < count; ++i) {
        score.timeline.push_back({events[i].id, start[i], finish[i]});
        if (finish[i] > score.runtime_us) {
            score.runtime_us = finish[i];
            terminal = i;
        }
    }
    std::vector<std::string> reverse_path;
    for (std::size_t i = terminal; i != count; i = critical_predecessor[i])
        reverse_path.push_back(events[i].id);
    score.critical_path.assign(reverse_path.rbegin(), reverse_path.rend());
    return score;
}

}  // namespace

SolveResult Optimize(const Workload& workload, const Hardware& hardware,
                     const std::vector<Candidate>& candidates) {
    if (workload.m == 0 || workload.n == 0 || workload.k == 0 || workload.dtype.empty() ||
        hardware.aic_count == 0)
        throw std::invalid_argument("workload dimensions, dtype, and AIC count are required");
    SolveResult result;
    result.supplied = candidates.size();
    std::set<std::string> candidate_ids;
    for (const auto& candidate : candidates) {
        if (!candidate_ids.insert(candidate.id).second) {
            result.rejected.push_back({candidate.id, "duplicate candidate id"});
            continue;
        }
        try {
            Validate(workload, hardware, candidate);
            result.ranking.push_back({candidate.id, candidate.family, candidate.tiling_key,
                                      Evaluate(hardware, candidate)});
        } catch (const std::exception& error) {
            result.rejected.push_back({candidate.id, error.what()});
        }
    }
    std::stable_sort(result.ranking.begin(), result.ranking.end(),
                     [](const RankedCandidate& a, const RankedCandidate& b) {
                         return a.score.runtime_us < b.score.runtime_us;
                     });
    return result;
}

HierarchicalResult OptimizeImplementations(
    const Workload& workload, const Hardware& hardware,
    const std::vector<Implementation>& implementations) {
    if (workload.m == 0 || workload.n == 0 || workload.k == 0 || workload.dtype.empty() ||
        hardware.aic_count == 0)
        throw std::invalid_argument("workload dimensions, dtype, and AIC count are required");

    HierarchicalResult result;
    std::set<std::string> implementation_ids;
    for (const auto& implementation : implementations) {
        if (implementation.id.empty() || !implementation_ids.insert(implementation.id).second ||
            !implementation.build)
            throw std::invalid_argument("implementation needs unique id and candidate builder");
        std::set<std::string> variable_names;
        for (const auto& variable : implementation.variables) {
            if (variable.name.empty() || !variable_names.insert(variable.name).second ||
                variable.values.empty() ||
                std::set<std::uint64_t>(variable.values.begin(), variable.values.end()).size() !=
                    variable.values.size())
                throw std::invalid_argument("tiling variables need unique names and nonempty, unique domains");
        }

        ImplementationResult family;
        family.id = implementation.id;
        TileAssignment assignment;
        std::set<std::string> candidate_ids;
        std::function<void(std::size_t)> search = [&](std::size_t depth) {
            if (family.best && implementation.lower_bound_us) {
                const double bound = implementation.lower_bound_us(assignment);
                if (!std::isfinite(bound) || bound < 0.0)
                    throw std::invalid_argument("invalid tiling lower bound");
                if (bound >= family.best->score.runtime_us) {
                    ++family.pruned_subtrees;
                    return;
                }
            }
            if (depth == implementation.variables.size()) {
                try {
                    const auto candidate = implementation.build(assignment);
                    if (!candidate) {
                        ++family.infeasible;
                        return;
                    }
                    if (candidate->family != implementation.id ||
                        !candidate_ids.insert(candidate->id).second)
                        throw std::runtime_error("candidate family/id does not match implementation");
                    for (const auto& [name, value] : assignment) {
                        const auto tile = candidate->tile.find(name);
                        if (tile == candidate->tile.end() || tile->second != value)
                            throw std::runtime_error("candidate does not contain assigned tiling value " + name);
                    }
                    Validate(workload, hardware, *candidate);
                    auto score = Evaluate(hardware, *candidate);
                    ++family.scored;
                    if (!family.best || score.runtime_us < family.best->score.runtime_us) {
                        family.best = RankedCandidate{candidate->id, candidate->family,
                                                      candidate->tiling_key, std::move(score)};
                        family.best_plan = *candidate;
                        family.best_tiling = assignment;
                    }
                } catch (const std::exception&) {
                    ++family.rejected;
                }
                return;
            }
            const auto& variable = implementation.variables[depth];
            for (const auto value : variable.values) {
                assignment[variable.name] = value;
                search(depth + 1);
            }
            assignment.erase(variable.name);
        };
        search(0);
        if (family.best && (!result.best ||
                            family.best->score.runtime_us < result.best->score.runtime_us)) {
            result.best = family.best;
            result.best_plan = family.best_plan;
            result.best_tiling = family.best_tiling;
        }
        result.implementations.push_back(std::move(family));
    }
    return result;
}

Manifest ReadManifest(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open manifest: " + path);
    Manifest manifest;
    Candidate* current = nullptr;
    bool generate_events = false;
    bool preflight = false;
    bool saw_workload = false, saw_hardware = false;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        std::istringstream stream(line);
        std::vector<std::string> words;
        for (std::string word; stream >> word;) words.push_back(word);
        if (words.empty()) continue;
        const auto& command = words[0];
        if (command == "workload" && !current && !saw_workload) {
            RequireCount(words, 5, line_number);
            manifest.workload = {ParseU64(words[1], "M"), ParseU64(words[2], "N"),
                                 ParseU64(words[3], "K"), words[4]};
            saw_workload = true;
        } else if (command == "hardware" && !current && !saw_hardware) {
            RequireCount(words, 3, line_number);
            manifest.hardware.aic_count = ParseU64(words[1], "AIC count");
            manifest.hardware.workspace_limit_bytes = ParseU64(words[2], "workspace limit");
            saw_hardware = true;
        } else if (command == "primitive" && !current) {
            RequireCount(words, 5, line_number);
            const double fixed = ParseNonnegative(words[3], "primitive latency");
            const double rate = ParseNonnegative(words[4], "primitive rate");
            if (rate == 0.0 || !manifest.hardware.primitives.emplace(
                    words[1], PrimitiveTiming{words[2], fixed, rate}).second)
                throw std::runtime_error("line " + std::to_string(line_number) +
                                         ": zero rate or duplicate primitive " + words[1]);
        } else if (command == "candidate" && !current) {
            if (words.size() != 4 && words.size() != 6)
                throw std::runtime_error("line " + std::to_string(line_number) +
                                         ": candidate needs id family preflight or the legacy five arguments");
            manifest.candidates.push_back({});
            current = &manifest.candidates.back();
            generate_events = false;
            preflight = words.size() == 4;
            current->id = words[1];
            current->family = words[2];
            if (preflight) {
                if (words[3] != "preflight")
                    throw std::runtime_error("line " + std::to_string(line_number) +
                                             ": expected preflight");
            } else {
                current->tiling_key = words[3];
                if (words[4] != "verified" && words[4] != "unverified")
                    throw std::runtime_error("line " + std::to_string(line_number) +
                                             ": expected verified or unverified");
                current->backend_verified = words[4] == "verified";
                current->workspace_bytes = ParseU64(words[5], "workspace bytes");
            }
        } else if (command == "tile" && current) {
            RequireCount(words, 3, line_number);
            if (!current->tile.emplace(words[1], ParseU64(words[2], words[1])).second)
                throw std::runtime_error("line " + std::to_string(line_number) +
                                         ": duplicate tile field " + words[1]);
        } else if (command == "event" && current) {
            RequireCount(words, 7, line_number);
            if (generate_events)
                throw std::runtime_error("line " + std::to_string(line_number) +
                                         ": event records cannot follow generate_events");
            current->events.push_back({words[1], words[2], words[3],
                                       ParseNonnegative(words[4], "event work"), words[5],
                                       SplitComma(words[6])});
        } else if (command == "generate_events" && current) {
            RequireCount(words, 1, line_number);
            if (generate_events || !current->events.empty())
                throw std::runtime_error("line " + std::to_string(line_number) +
                                         ": generate_events cannot be repeated or mixed with event records");
            generate_events = true;
        } else if (command == "end" && current) {
            RequireCount(words, 1, line_number);
            if (preflight) {
                if (!saw_workload || !saw_hardware)
                    throw std::runtime_error("line " + std::to_string(line_number) +
                                             ": workload and hardware required before preflight");
                if (!current->events.empty() || !generate_events)
                    throw std::runtime_error("line " + std::to_string(line_number) +
                                             ": preflight candidate requires generate_events only");
                if (current->family != "BASE" && current->family != "DETERMINISTIC_SPLIT_K")
                    throw std::runtime_error("unsupported preflight family " + current->family);
                static const std::set<std::string> fields = {
                    "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
                    "baseM", "baseN", "baseK"};
                for (const auto& [name, value] : current->tile) {
                    (void)value;
                    if (!fields.count(name))
                        throw std::runtime_error("unmodeled preflight tile field " + name);
                }
                PlanDecision decision;
                decision.family = current->family == "BASE" ? KernelFamily::Base
                                                               : KernelFamily::DeterministicSplitK;
                decision.used_cores = RequiredTile(*current, "usedCoreNum");
                decision.single_core_m = RequiredTile(*current, "singleCoreM");
                decision.single_core_n = RequiredTile(*current, "singleCoreN");
                decision.single_core_k = RequiredTile(*current, "singleCoreK");
                decision.base_m = RequiredTile(*current, "baseM");
                decision.base_n = RequiredTile(*current, "baseN");
                decision.base_k = RequiredTile(*current, "baseK");
                if (decision.family == KernelFamily::DeterministicSplitK)
                    decision.step_k = decision.single_core_k / decision.base_k;
                const auto finalized = FinalizePlan(manifest.workload, manifest.hardware,
                                                     decision, current->id);
                *current = std::move(finalized.candidate);
            }
            if (generate_events) {
                if (!saw_workload)
                    throw std::runtime_error("line " + std::to_string(line_number) +
                                             ": workload required before generate_events");
                *current = BuildMatmulEvents(manifest.workload, std::move(*current));
            }
            current = nullptr;
        } else {
            throw std::runtime_error("line " + std::to_string(line_number) +
                                     ": unexpected command " + command);
        }
    }
    if (current) throw std::runtime_error("unterminated candidate");
    if (!saw_workload || !saw_hardware)
        throw std::runtime_error("manifest requires workload and hardware records");
    return manifest;
}

}  // namespace mmopt
