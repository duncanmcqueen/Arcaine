#pragma once

namespace httplib { struct Request; struct Response; }

namespace arcaine::server {

struct AppState;

// Validates ARCAINE_SYSTEMONE_* deployment settings once at startup; throws
// std::runtime_error with a clear message on malformed values.
void validate_systemone_config();

// POST /v1/systemone — Jev-format structured decisions on the DiffusionGemma
// backend.  See docs/systemone.md.
void handle_systemone(const httplib::Request& req, httplib::Response& res,
                      AppState& app);

}  // namespace arcaine::server
