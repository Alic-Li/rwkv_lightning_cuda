#include "rwkv/server/router/route_helpers.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace rwkv7_server::router {

using namespace drogon;

Json::Value make_error(const std::string& message) {
  Json::Value out;
  out["error"] = message;
  return out;
}

Json::Value prefill_limit_error(const PrefillBatchLimitExceeded& error) {
  Json::Value out;
  out["error"] = error.what();
  out["request_bsz"] = error.request_batch_size();
  out["max_bsz"] = error.max_batch_size();
  return out;
}

HttpResponsePtr json_response(Json::Value payload, HttpStatusCode code) {
  Json::StreamWriterBuilder builder;
  builder["emitUTF8"] = true;
  builder["indentation"] = "";
  auto resp = HttpResponse::newHttpResponse();
  resp->setContentTypeCode(CT_APPLICATION_JSON);
  resp->setBody(Json::writeString(builder, payload));
  resp->setStatusCode(code);
  return resp;
}

std::optional<std::string> bearer_token(const HttpRequestPtr& req) {
  auto auth = req->getHeader("authorization");
  if (auth.empty()) {
    auth = req->getHeader("Authorization");
  }
  if (auth.rfind("Bearer ", 0) != 0) {
    return std::nullopt;
  }
  return auth.substr(7);
}

bool check_password(
    const HttpRequestPtr& req,
    const Json::Value& body,
    const std::optional<std::string>& password,
    HttpResponsePtr& out) {
  if (!password.has_value()) {
    return true;
  }
  const auto token = bearer_token(req);
  if (body.get("password", "").asString() == *password ||
      (token.has_value() && *token == *password)) {
    return true;
  }
  out = json_response(make_error("Unauthorized: invalid or missing password"), k401Unauthorized);
  return false;
}


std::string trim(const std::string& input) {
  const auto begin = input.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return {};
  }
  const auto end = input.find_last_not_of(" \t\r\n");
  return input.substr(begin, end - begin + 1);
}

// Reads a request header under both the documented and the all-lowercase wire
// name, mirroring bearer_token(); older drogon builds look headers up
// verbatim instead of case-insensitively.
std::string request_header_value(const HttpRequestPtr& req, const char* name) {
  auto value = req->getHeader(name);
  if (!value.empty()) {
    return value;
  }
  std::string lower(name);
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return req->getHeader(lower);
}

std::string state_id_from_body(const Json::Value& body) {
  if (!body.isMember("state_id") || body["state_id"].isNull()) {
    return {};
  }
  if (!body["state_id"].isString()) {
    throw std::runtime_error("state_id must be a string");
  }
  return body["state_id"].asString();
}

std::string session_id_from_body(const Json::Value& body) {
  return body.get("session_id", "").asString();
}

// Resolves an identifier that may arrive through several channels (JSON body,
// request header, query string). Repeating the same value across channels is
// harmless; conflicting values are ambiguous, so the request fails instead of
// silently picking one.
std::string resolve_identifier(
    std::initializer_list<std::pair<std::string, std::string>> channels,
    const std::string& field) {
  std::string value;
  std::string used;
  for (const auto& channel : channels) {
    const std::string current = trim(channel.second);
    if (current.empty()) {
      continue;
    }
    if (!used.empty()) {
      if (current == value) {
        continue;
      }
      throw std::runtime_error(
          field + " has conflicting values in " + used + " and " +
          channel.first + "; use only one of them");
    }
    used = channel.first;
    value = current;
  }
  return value;
}

std::string session_id_from_request(const HttpRequestPtr& req, const Json::Value& body) {
  return resolve_identifier(
      {{"request body", session_id_from_body(body)},
       {std::string(kSessionIdHeader) + " header",
        request_header_value(req, kSessionIdHeader)}},
      "session_id");
}

std::string state_id_from_request(const HttpRequestPtr& req, const Json::Value& body) {
  return resolve_identifier(
      {{"request body", state_id_from_body(body)},
       {std::string(kStateIdHeader) + " header",
        request_header_value(req, kStateIdHeader)}},
      "state_id");
}

}  // namespace rwkv7_server::router
