#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace splash::engine {

class MemoryGovernor;
class NativeRuntime;

// Settings an operator can change while the server runs. They arrive as a
// small `key=value` file the HTTP front end replaces atomically.
struct RuntimeProfile final {
  uint32_t dutyPercent = 100;
  bool paused = false;
  // Zero selects the startup ceiling.
  uint64_t memoryLimitBytes = 0;
  // A change asks for one release of empty KV backing.
  uint64_t reclaimSeq = 0;

  bool operator==(const RuntimeProfile &) const = default;
};

// Applies the recognized keys onto `profile`. Returns false, leaving
// `profile` untouched, for an unknown key or an invalid value.
[[nodiscard]] bool parseRuntimeProfile(std::string_view text,
                                       RuntimeProfile &profile);

// Reads the requested profile and applies it from the control handler.
// Duty and pause take effect at once. A new memory limit and a reclaim
// request wait until no request is active, so the next request is the first
// to see them. The applied state is published for the front end to read.
class RuntimeProfileApplier final {
public:
  RuntimeProfileApplier(std::filesystem::path directory,
                        RuntimeProfile startup, uint64_t ceilingBytes);

  // Owner thread only. Returns true while a follow-up pass is still useful.
  [[nodiscard]] bool update(NativeRuntime &runtime, MemoryGovernor &governor);

private:
  void publish(const MemoryGovernor &governor, uint64_t pendingBytes);

  std::filesystem::path requestPath_;
  std::filesystem::path appliedPath_;
  RuntimeProfile requested_;
  RuntimeProfile applied_;
  uint64_t ceilingBytes_ = 0;
  std::string lastRequestText_;
  std::string lastPublished_;
};

} // namespace splash::engine
