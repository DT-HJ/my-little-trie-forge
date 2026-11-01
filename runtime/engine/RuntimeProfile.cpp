#include "engine/RuntimeProfile.hpp"

#include "engine/MemoryGovernor.hpp"
#include "engine/NativeRuntime.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <iostream>
#include <sstream>
#include <system_error>

namespace splash::engine {
namespace {

bool parseUnsigned(std::string_view text, uint64_t &value) {
  if (text.empty())
    return false;
  const char *end = text.data() + text.size();
  const auto parsed = std::from_chars(text.data(), end, value);
  return parsed.ec == std::errc{} && parsed.ptr == end;
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return {};
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}

} // namespace

bool parseRuntimeProfile(std::string_view text, RuntimeProfile &profile) {
  RuntimeProfile parsed = profile;
  while (!text.empty()) {
    const size_t newline = text.find('\n');
    std::string_view line = text.substr(0, newline);
    text = newline == std::string_view::npos ? std::string_view{}
                                             : text.substr(newline + 1);
    if (line.empty())
      continue;
    const size_t equals = line.find('=');
    if (equals == std::string_view::npos)
      return false;
    const std::string_view key = line.substr(0, equals);
    uint64_t value = 0;
    if (!parseUnsigned(line.substr(equals + 1), value))
      return false;
    if (key == "duty_percent") {
      if (value < 1 || value > 100)
        return false;
      parsed.dutyPercent = static_cast<uint32_t>(value);
    } else if (key == "paused") {
      if (value > 1)
        return false;
      parsed.paused = value == 1;
    } else if (key == "memory_limit_bytes") {
      parsed.memoryLimitBytes = value;
    } else if (key == "reclaim_seq") {
      parsed.reclaimSeq = value;
    } else {
      return false;
    }
  }
  profile = parsed;
  return true;
}

RuntimeProfileApplier::RuntimeProfileApplier(std::filesystem::path directory,
                                             RuntimeProfile startup,
                                             uint64_t ceilingBytes)
    : requestPath_(directory / "profile"), appliedPath_(directory / "applied"),
      requested_(startup), applied_(startup), ceilingBytes_(ceilingBytes) {}

bool RuntimeProfileApplier::update(NativeRuntime &runtime,
                                   MemoryGovernor &governor) {
  const std::string text = readFile(requestPath_);
  if (!text.empty() && text != lastRequestText_) {
    lastRequestText_ = text;
    if (!parseRuntimeProfile(text, requested_))
      std::cerr << "warning: ignoring an invalid runtime profile\n";
  }

  if (requested_.dutyPercent != applied_.dutyPercent) {
    runtime.setDutyPercent(requested_.dutyPercent);
    applied_.dutyPercent = requested_.dutyPercent;
  }
  if (requested_.paused != applied_.paused) {
    runtime.setDispatchHeld(requested_.paused);
    applied_.paused = requested_.paused;
  }

  const bool idle = runtime.idle();
  const uint64_t target = requested_.memoryLimitBytes
                              ? std::min(requested_.memoryLimitBytes,
                                         ceilingBytes_)
                              : ceilingBytes_;
  bool followUp = false;
  uint64_t pending = 0;
  if (target != governor.snapshot().limitBytes) {
    if (idle)
      governor.setLimitBytes(target);
    else
      pending = target;
  }
  if (idle) {
    const MemoryGovernorSnapshot memory = governor.snapshot();
    if (memory.observedResidentBytes > memory.limitBytes) {
      const uint64_t released = runtime.reclaimMemory(MemoryReclaimDirective{
          .reclaimEmptyKvExtents = true,
          .targetBytes = memory.observedResidentBytes - memory.limitBytes});
      followUp = released > 0;
    }
    if (requested_.reclaimSeq != applied_.reclaimSeq) {
      static_cast<void>(runtime.reclaimMemory(
          MemoryReclaimDirective{.reclaimEmptyKvExtents = true}));
      applied_.reclaimSeq = requested_.reclaimSeq;
    }
  }
  publish(governor, pending);
  return followUp;
}

void RuntimeProfileApplier::publish(const MemoryGovernor &governor,
                                    uint64_t pendingBytes) {
  std::ostringstream text;
  text << "duty_percent=" << applied_.dutyPercent
       << "\npaused=" << (applied_.paused ? 1 : 0)
       << "\nmemory_limit_bytes=" << governor.snapshot().limitBytes
       << "\nmemory_limit_pending_bytes=" << pendingBytes
       << "\nmemory_ceiling_bytes=" << ceilingBytes_
       << "\nreclaim_seq=" << applied_.reclaimSeq << '\n';
  const std::string content = text.str();
  if (content == lastPublished_)
    return;
  const std::filesystem::path temporary = appliedPath_.string() + ".tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output << content;
    if (!output)
      return;
  }
  std::error_code error;
  std::filesystem::rename(temporary, appliedPath_, error);
  if (!error)
    lastPublished_ = content;
}

} // namespace splash::engine
