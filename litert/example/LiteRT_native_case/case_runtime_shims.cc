// Symbols the C++ headers call directly, but libLiteRt.so does not export.
//
// Logging APIs are hidden by export_litert_runtime_only_linux.lds.
// LrtAmlOptions* is not linked into libLiteRt.so. This is the same
// LrtAmlOptionsT layout as litert/c/options/litert_aml_options.cc. The
// compiler plugin reads the opaque payload as a TOML string.

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_aml_options.h"

class LrtAmlOptionsT {
 public:
  std::optional<LrtAmlOptionsLogLevel> log_level;
  std::optional<LrtAmlOptionsProfiling> profiling;
  std::optional<LrtAmlOptionsPerfMode> perf_mode;
  std::optional<LrtAmlOptionsMemoryPolicy> memory_policy;
  std::optional<std::string> model_path;
  std::optional<std::string> model_name;
};

namespace {

void DeleteTomlPayload(void* payload) { delete[] static_cast<char*>(payload); }

void AppendTomlInt(std::string* toml, const char* key, int value) {
  *toml += key;
  *toml += " = ";
  *toml += std::to_string(value);
  *toml += "\n";
}

void AppendTomlString(std::string* toml, const char* key,
                      const std::string& value) {
  *toml += key;
  *toml += " = \"";
  *toml += value;
  *toml += "\"\n";
}

}  // namespace

extern "C" {

LiteRtStatus LiteRtGetMinLoggerSeverity(LiteRtLogger /*logger*/,
                                        LiteRtLogSeverity* min_severity) {
  if (min_severity == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *min_severity = kLiteRtLogSeverityInfo;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetMinLoggerSeverity(LiteRtLogger /*logger*/,
                                        LiteRtLogSeverity /*min_severity*/) {
  return kLiteRtStatusOk;
}

LiteRtLogger LiteRtGetDefaultLogger() {
  // The real default logger stays inside libLiteRt.so. Callers only pass this
  // pointer into the severity and log shims above, which ignore it.
  static int dummy_logger;
  return reinterpret_cast<LiteRtLogger>(&dummy_logger);
}

const char* LiteRtGetLogSeverityName(LiteRtLogSeverity severity) {
  switch (severity) {
    case kLiteRtLogSeverityDebug:
      return "DEBUG";
    case kLiteRtLogSeverityVerbose:
      return "VERBOSE";
    case kLiteRtLogSeverityInfo:
      return "INFO";
    case kLiteRtLogSeverityWarning:
      return "WARNING";
    case kLiteRtLogSeverityError:
      return "ERROR";
    case kLiteRtLogSeveritySilent:
      return "SILENT";
    default:
      return "UNKNOWN";
  }
}

LiteRtStatus LiteRtLoggerLog(LiteRtLogger /*logger*/,
                             LiteRtLogSeverity /*severity*/,
                             const char* format, ...) {
  if (format == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fputc('\n', stderr);
  return kLiteRtStatusOk;
}

const char* LrtAmlOptionsGetIdentifier() { return "amlogic"; }

LiteRtStatus LrtCreateAmlOptions(LrtAmlOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LrtAmlOptionsT();
  return kLiteRtStatusOk;
}

void LrtDestroyAmlOptions(LrtAmlOptions options) { delete options; }

LiteRtStatus LrtAmlOptionsSetLogLevel(LrtAmlOptions options,
                                      LrtAmlOptionsLogLevel log_level) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->log_level = log_level;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsGetLogLevel(LrtAmlOptions options,
                                      LrtAmlOptionsLogLevel* log_level) {
  if (options == nullptr || log_level == nullptr ||
      !options->log_level.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *log_level = *options->log_level;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsSetPerfMode(LrtAmlOptions options,
                                      LrtAmlOptionsPerfMode mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->perf_mode = mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsGetPerfMode(LrtAmlOptions options,
                                      LrtAmlOptionsPerfMode* mode) {
  if (options == nullptr || mode == nullptr || !options->perf_mode.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *mode = *options->perf_mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsSetProfiling(LrtAmlOptions options,
                                       LrtAmlOptionsProfiling profiling) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->profiling = profiling;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsGetProfiling(LrtAmlOptions options,
                                       LrtAmlOptionsProfiling* profiling) {
  if (options == nullptr || profiling == nullptr ||
      !options->profiling.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *profiling = *options->profiling;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsSetMemoryPolicy(LrtAmlOptions options,
                                          LrtAmlOptionsMemoryPolicy policy) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->memory_policy = policy;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsGetMemoryPolicy(LrtAmlOptions options,
                                          LrtAmlOptionsMemoryPolicy* policy) {
  if (options == nullptr || policy == nullptr ||
      !options->memory_policy.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *policy = *options->memory_policy;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsSetModelPath(LrtAmlOptions options,
                                       const char* model_path) {
  if (options == nullptr || model_path == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->model_path = model_path;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsGetModelPath(LrtAmlOptions options,
                                       const char** model_path) {
  if (options == nullptr || model_path == nullptr ||
      !options->model_path.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *model_path = options->model_path->c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsSetModelName(LrtAmlOptions options,
                                       const char* model_name) {
  if (options == nullptr || model_name == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->model_name = model_name;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsGetModelName(LrtAmlOptions options,
                                       const char** model_name) {
  if (options == nullptr || model_name == nullptr ||
      !options->model_name.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *model_name = options->model_name->c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetOpaqueAmlOptionsData(LrtAmlOptions options,
                                        const char** identifier, void** payload,
                                        void (**payload_deleter)(void*)) {
  if (options == nullptr || identifier == nullptr || payload == nullptr ||
      payload_deleter == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *identifier = LrtAmlOptionsGetIdentifier();
  std::string toml;
  if (options->log_level.has_value()) {
    AppendTomlInt(&toml, "log_level", static_cast<int>(*options->log_level));
  }
  if (options->profiling.has_value()) {
    AppendTomlInt(&toml, "profiling", static_cast<int>(*options->profiling));
  }
  if (options->perf_mode.has_value()) {
    AppendTomlInt(&toml, "perf_mode", static_cast<int>(*options->perf_mode));
  }
  if (options->memory_policy.has_value()) {
    AppendTomlInt(&toml, "memory_policy",
                  static_cast<int>(*options->memory_policy));
  }
  if (options->model_path.has_value()) {
    AppendTomlString(&toml, "model_path", *options->model_path);
  }
  if (options->model_name.has_value()) {
    AppendTomlString(&toml, "model_name", *options->model_name);
  }
  char* bytes = new char[toml.size() + 1];
  std::memcpy(bytes, toml.c_str(), toml.size() + 1);
  *payload = bytes;
  *payload_deleter = DeleteTomlPayload;
  return kLiteRtStatusOk;
}

}  // extern "C"
