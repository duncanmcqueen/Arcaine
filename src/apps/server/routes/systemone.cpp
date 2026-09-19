#include "apps/server/routes/systemone.hpp"

#include "apps/server/app_state.hpp"
#include "apps/server/openai/error_encoder.hpp"
#include "apps/server/openai/schemas.hpp"
#include "modeling/diffusion_gemma/model.hpp"
#include "modeling/diffusion_gemma/service.hpp"
#include "runtime/gpu/nvfp4_session_stats.hpp"
#include "utils/chat.hpp"
#include "utils/decision_schema.hpp"

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace arcaine::server {
namespace {

using json = nlohmann::ordered_json;
namespace dg = arcaine::diffusion_gemma;

void set_systemone_headers(httplib::Response& res) {
    // Mark the local confidence method.
    res.set_header("X-Arcaine-Confidence-Method", "normalized-entropy-v1");
    res.set_header("X-Arcaine-Compatibility", "jev-format-approximate-confidence");
}

[[noreturn]] void config_error(const std::string& msg) {
    throw std::runtime_error("invalid SystemOne deployment configuration: " + msg);
}

std::string ascii_lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// Parse an explicit boolean value.
bool parse_bool_env(const char* name, bool fallback) {
    const char* e = std::getenv(name);
    if (!e) return fallback;
    const std::string v = ascii_lower(e);
    if (v == "1" || v == "true" || v == "on" || v == "yes") return true;
    if (v == "0" || v == "false" || v == "off" || v == "no") return false;
    config_error(std::string(name) + " must be one of 0/1/true/false/on/off/yes/no, got '" +
                 std::string(e) + "'");
}

int parse_int_env(const char* name, int fallback, int lo, int hi) {
    const char* e = std::getenv(name);
    if (!e || !*e) return fallback;
    errno = 0;
    char* end = nullptr;
    long v = std::strtol(e, &end, 10);
    if (errno != 0 || end == e || *end != '\0')
        config_error(std::string(name) + " must be an integer, got '" + std::string(e) + "'");
    if (v < lo || v > hi)
        config_error(std::string(name) + " must be in [" + std::to_string(lo) + "," +
                     std::to_string(hi) + "], got " + std::to_string(v));
    return (int)v;
}

float parse_float_env(const char* name, float fallback, float lo, float hi) {
    const char* e = std::getenv(name);
    if (!e || !*e) return fallback;
    errno = 0;
    char* end = nullptr;
    float v = std::strtof(e, &end);
    if (errno != 0 || end == e || *end != '\0')
        config_error(std::string(name) + " must be a number, got '" + std::string(e) + "'");
    if (!(v >= lo && v <= hi))
        config_error(std::string(name) + " must be in [" + std::to_string(lo) + "," +
                     std::to_string(hi) + "], got " + std::to_string(v));
    return v;
}

// Hold local read settings.
struct SystemoneConfig {
    int   fixed_reads = 1;
    bool  auto_mode = false;
    int   auto_max = 4;
    float auto_threshold = 0.5f;
    float auto_min_label_mass = 0.0f;
    float watchdog_s = 0.0f;   // 0 = off
};

SystemoneConfig load_systemone_config() {
    SystemoneConfig c;
    c.fixed_reads = parse_int_env("ARCAINE_SYSTEMONE_READS", 1, 1, 32);
    c.auto_mode   = parse_bool_env("ARCAINE_SYSTEMONE_READS_AUTO", false);
    c.auto_max    = parse_int_env("ARCAINE_SYSTEMONE_READS_AUTO_MAX", 4, 1, 32);
    c.auto_threshold = parse_float_env("ARCAINE_SYSTEMONE_AUTO_ENTROPY_THRESHOLD", 0.5f, 0.0f, 1.0f);
    c.auto_min_label_mass = parse_float_env("ARCAINE_SYSTEMONE_AUTO_MIN_LABEL_MASS", 0.0f, 0.0f, 1.0f);
    c.watchdog_s = parse_float_env("ARCAINE_GPU_WATCHDOG_S", 0.0f, 0.0f, 3600.0f);
    return c;
}

const SystemoneConfig& systemone_config() {
    static const SystemoneConfig cfg = load_systemone_config();
    return cfg;
}

DecisionReadOptions read_options(uint64_t stream_seed) {
    const SystemoneConfig& c = systemone_config();
    DecisionReadOptions o;
    o.reads = c.fixed_reads;
    o.auto_mode = c.auto_mode;
    o.auto_max = c.auto_max;
    o.auto_entropy_threshold = c.auto_threshold;
    o.auto_min_label_mass = c.auto_min_label_mass;
    o.watchdog_s = c.watchdog_s;
    o.stream_seed = stream_seed;
    return o;
}

std::string model_alias() {
    // Return the configured local alias.
    const char* e = std::getenv("ARCAINE_SYSTEMONE_MODEL_ALIAS");
    return e ? std::string(e) : std::string();
}

}  // namespace

void validate_systemone_config() {
    // Validate settings at startup.
    (void)systemone_config();
}

void handle_systemone(const httplib::Request& req, httplib::Response& res,
                      AppState& app) {
    set_systemone_headers(res);
    try {
        auto* service = dynamic_cast<dg::DiffusionGemmaService*>(app.model.service.get());
        if (!service || app.model.descriptor.implementation_id != "diffusion_gemma") {
            res.status = 422;
            res.set_content(arcaine::openai::error_body(
                "/v1/systemone requires the diffusion_gemma backend; the loaded model '" +
                    app.opts.served_model_name + "' uses '" +
                    app.model.descriptor.implementation_id + "'",
                "invalid_request_error", "unsupported_backend").dump(),
                "application/json");
            return;
        }

        json body = dg::parse_json_no_duplicates(req.body);

        dg::DecisionRequestSettings settings;
        settings.base_seed     = app.opts.seed;
        settings.max_questions = parse_int_env("ARCAINE_SYSTEMONE_MAX_QUESTIONS", 64, 1, 4096);
        settings.max_seq       = app.opts.max_seq;
        settings.canvas_max    = service->model().canvas_capacity();
        settings.vocab_size    = service->model().config().text.vocab_size;
        settings.pad_token_id  = service->model().config().gen.pad_token_id;
        TokenizerBridge& tokenizer = service->tokenizer();
        if (tokenizer.has_token("<turn|>"))
            settings.turn_close_id = tokenizer.token_id("<turn|>");
        else if (tokenizer.has_token("<end_of_turn>"))
            settings.turn_close_id = tokenizer.token_id("<end_of_turn>");
        else if (!service->model().config().gen.eos_token_ids.empty())
            settings.turn_close_id = service->model().config().gen.eos_token_ids[0];
        else
            throw dg::DecisionSchemaError(
                "no turn-closing token found in the tokenizer (need <turn|> or <end_of_turn>)");

        dg::DecisionRequest dreq = dg::parse_decision_request(body, settings);

        if (dreq.model != app.opts.served_model_name &&
            (model_alias().empty() || dreq.model != model_alias())) {
            res.status = 404;
            res.set_content(arcaine::openai::error_body(
                "unknown model '" + dreq.model + "'; this server serves '" +
                    app.opts.served_model_name + "'",
                "invalid_request_error", "model_not_found").dump(),
                "application/json");
            return;
        }

        dg::compile_decision_questions(dreq, tokenizer, settings);

        json answers = json::object();
        json diagnostics = json::object();
        long long input_tokens = 0, output_tokens = 0;
        Nvfp4GraphCounts graph_before = dreq.want_diagnostics
            ? nvfp4_graph_capture_counts() : Nvfp4GraphCounts{};
        {
            // Run questions in order under the model lock.
            std::lock_guard<std::mutex> lock(app.generate_mu);
            for (auto& q : dreq.questions) {
                DecisionReadResult r = service->model().read_decisions(
                    q.prompt_ids, q.templ, read_options(q.stream_seed));
                input_tokens += r.prompt_tokens;
                output_tokens += (long long)r.reads.size();
                if (dreq.want_diagnostics) {
                    json d;
                    answers[q.external_id] = dg::map_decision_answer(q, r, &d);
                    diagnostics[q.external_id] = std::move(d);
                } else {
                    answers[q.external_id] = dg::map_decision_answer(q, r, nullptr);
                }
            }
        }

        json out;
        out["model"] = app.opts.served_model_name;
        out["answers"] = std::move(answers);
        out["usage"] = {{"input_tokens", input_tokens},
                        {"output_tokens", output_tokens}};
        if (dreq.want_diagnostics) {
            Nvfp4GraphCounts graph_after = nvfp4_graph_capture_counts();
            json ext;
            ext["diagnostics"] = std::move(diagnostics);
            ext["backend"] = "arcaine diffusion_gemma (approximate Jev format; "
                             "confidence = normalized-entropy-v1)";
            // These values must be zero for a structured read.
            ext["graph_captures_delta"] = graph_after.captures - graph_before.captures;
            ext["graph_replays_delta"] = graph_after.replays - graph_before.replays;
            out["arcaine"] = std::move(ext);
        }
        res.status = 200;
        res.set_content(out.dump(), "application/json");
    } catch (const dg::DecisionSchemaError& e) {
        res.status = 422;
        res.set_content(arcaine::openai::error_body(e.what(), "validation_error",
                                                    "invalid_request_error").dump(),
                        "application/json");
    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content(arcaine::openai::error_body(e.what(), "server_error",
                                                    "internal_error").dump(),
                        "application/json");
    }
}

}  // namespace arcaine::server
