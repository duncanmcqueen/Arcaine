#pragma once
// Validate structured read input before GPU work.
#include "model.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

inline void validate_decision_read_request(int vocab_size, int kv_cache_max_seq,
                                           int canvas_capacity,
                                           const std::vector<int>& prompt_ids,
                                           const CompiledDecisionTemplate& compiled,
                                           const DecisionReadOptions& options)
{
    auto require = [](bool ok, const std::string& msg) {
        if (!ok) throw std::runtime_error("read_decisions: " + msg);
    };

    const int V = vocab_size;
    const int C = (int)compiled.canvas.size();
    const int K = (int)compiled.label_ids.size();
    const int64_t prompt_n = (int64_t)prompt_ids.size();
    const int64_t max_seq  = (int64_t)kv_cache_max_seq;

    // Check sizes in 64-bit before conversion to device sizes.
    require(V > 0, "model vocabulary size is not configured");
    require(max_seq > 0, "KV cache capacity is not configured");
    require(!prompt_ids.empty(), "prompt is empty");
    for (size_t i = 0; i < prompt_ids.size(); ++i)
        require(prompt_ids[i] >= 0 && prompt_ids[i] < V,
                "prompt token id " + std::to_string(prompt_ids[i]) +
                " at position " + std::to_string(i) +
                " outside the vocabulary [0," + std::to_string(V) + ")");
    require(prompt_n <= max_seq,
            "prompt exceeds the KV cache capacity (" +
            std::to_string(max_seq) + " tokens, a local limit)");
    require(C >= 1 && C <= canvas_capacity,
            "canvas width " + std::to_string(C) +
            " outside the supported range 1.." + std::to_string(canvas_capacity) +
            " (local limit; the activation arena is planned for the configured canvas)");
    require(prompt_n + (int64_t)C <= max_seq,
            "prompt (" + std::to_string(prompt_n) + ") + canvas (" +
            std::to_string(C) + ") exceeds the KV cache capacity (" +
            std::to_string(max_seq) + ")");
    require(compiled.slot_position >= 0 && compiled.slot_position < C,
            "slot position outside the canvas");
    require(K >= 2 && K <= 256, "label count outside 2..256");
    for (int id : compiled.canvas)
        require(id >= 0 && id < V, "canvas token id outside the vocabulary");
    {
        std::vector<int32_t> sorted = compiled.label_ids;
        std::sort(sorted.begin(), sorted.end());
        for (int j = 0; j < K; ++j) {
            require(sorted[j] >= 0 && sorted[j] < V, "label id outside the vocabulary");
            require(j == 0 || sorted[j] != sorted[j - 1], "duplicate label id");
        }
    }
    require(options.reads >= 1 && options.reads <= 32, "reads outside 1..32");
    require(options.auto_max >= 1 && options.auto_max <= 32, "auto_max outside 1..32");
    require(options.auto_entropy_threshold >= 0.0f && options.auto_entropy_threshold <= 1.0f,
            "auto entropy threshold outside [0,1]");
    require(options.auto_min_label_mass >= 0.0f && options.auto_min_label_mass <= 1.0f,
            "auto minimum label mass outside [0,1]");
    require(options.watchdog_s >= 0.0f && options.watchdog_s <= 3600.0f,
            "GPU watchdog timeout outside [0,3600] seconds");
}
