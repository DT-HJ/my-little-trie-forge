#include "engine/Engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace splash::engine {
namespace {

constexpr double kResourceRetryBackoffMilliseconds = 100.0;
constexpr double kHealthCheckIntervalMilliseconds = 1000.0;

// Development-only trace, enabled by SPLASH_DEBUG_STATE_TRACE=FILE: one line
// per cache restore, draft plan and completed step, with the request's
// path-independent state content, so a cached and a cold run of the same
// prompt can be compared command by command. Off, it costs one null check.
std::FILE *debugTrace() {
  static std::FILE *const file = []() -> std::FILE * {
    const char *path = std::getenv("SPLASH_DEBUG_STATE_TRACE");
    return path && *path ? std::fopen(path, "a") : nullptr;
  }();
  return file;
}

std::FILE *decodeReadyTrace() {
  static std::FILE *const file = []() -> std::FILE * {
    const char *path = std::getenv("SPLASH_DEV_DECODE_READY_TRACE");
    return path && *path ? std::fopen(path, "a") : nullptr;
  }();
  return file;
}

double traceMonotonicMilliseconds() {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

void tracePlan(std::FILE *file, const char *event, const Scheduler &scheduler,
               const BatchPlan *plan, double at,
               double commandWallMilliseconds = 0.0,
               uint32_t confirmedTokens = 0) {
  if (!file) return;
  std::fprintf(file, "{\"event\":\"%s\",\"steady_ms\":%.3f,\"command_wall_ms\":%.3f,\"confirmed_tokens\":%u,\"kind\":%d,\"width\":%zu,\"selected\":[",
               event, at, commandWallMilliseconds, confirmedTokens,
               plan ? static_cast<int>(plan->kind) : -1,
               plan ? plan->items.size() : 0);
  if (plan)
    for (size_t i = 0; i < plan->items.size(); ++i)
      std::fprintf(file, "%s%llu", i ? "," : "",
                   static_cast<unsigned long long>(plan->items[i].requestId));
  std::fprintf(file, "],\"requests\":[");
  const auto requests = scheduler.decodeReadyView();
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto &r = requests[i];
    std::fprintf(file, "%s{\"id\":%llu,\"generation\":%llu,\"position\":%u,\"priority\":%d,\"cohort\":%d,\"stage\":%d,\"phase\":%d}",
                 i ? "," : "", static_cast<unsigned long long>(r.requestId),
                 static_cast<unsigned long long>(r.generation),
                 r.logicalPosition, static_cast<int>(r.priority),
                 static_cast<int>(r.cohort), static_cast<int>(r.stage),
                 static_cast<int>(r.phase));
  }
  std::fprintf(file, "]}\n");
  std::fflush(file);
}

void traceContent(std::FILE *file, const model::Model &model, uint64_t id) {
  const auto digest = model.debugStateDigest(id);
  if (!digest) {
    std::fprintf(file, " content unavailable\n");
    return;
  }
  const uint32_t parity = digest->activeParity;
  std::fprintf(file,
               " tokens %llu pending %u gdn_conv %016llx gdn_rec %016llx "
               "draft_k %016llx draft_v %016llx draft_base %llu "
               "draft_length %u gdn_layers",
               static_cast<unsigned long long>(digest->targetTokens),
               digest->pendingTokenValid ? digest->pendingToken : 0,
               static_cast<unsigned long long>(digest->gdnConvolution[parity]),
               static_cast<unsigned long long>(digest->gdnRecurrent[parity]),
               static_cast<unsigned long long>(digest->draftKeysWindow),
               static_cast<unsigned long long>(digest->draftValuesWindow),
               static_cast<unsigned long long>(digest->draftBase),
               digest->draftLength);
  for (uint32_t layer = 0; layer < digest->gdnLayerCount; ++layer)
    std::fprintf(file, " %016llx",
                 static_cast<unsigned long long>(digest->gdnActiveLayer[layer]));
  std::fprintf(file, "\n");
}

uint32_t replayStateBoundary(uint32_t tokens) noexcept {
  return tokens > 1 ? (tokens - 1) / KvCache::pageTokens * KvCache::pageTokens
                    : 0;
}

} // namespace

Engine::Engine(EngineConfig config, Cache &cache, model::Model &model,
         EngineEventSink &events)
    : config_(config), cache_(cache), model_(model), events_(events),
      scheduler_([] {
        const char *value = std::getenv("SPLASH_DEV_MAX_CONTEXT_PAIRED_PREFILL");
        // The supported R=2 policy is the default. The development switch
        // retains the old scheduler for matched diagnostics and fallback.
        return !(value && value[0] == '0' && value[1] == '\0');
      }(), [] {
        const char *value = std::getenv("SPLASH_DEV_CANONICAL_LONG_PREFILL");
        return value && value[0] == '1' && value[1] == '\0';
      }(), [] {
        const char *value = std::getenv("SPLASH_DEV_CANONICAL_MIN_TOKENS");
        if (!value)
          return 131072U;
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (end == value || *end != '\0' || parsed < 4096 || parsed > 131072)
          throw std::invalid_argument("invalid canonical prefill minimum");
        return static_cast<uint32_t>(parsed);
      }(), [] {
        const char *value = std::getenv("SPLASH_DEV_CANONICAL_DECODE_BURST");
        return value && value[0] == '1' && value[1] == '\0';
      }()) {
  if (!config_.maxContext || !config_.vocabularySize) {
    throw std::invalid_argument("context and vocabulary sizes must be positive");
  }
  if (!std::isfinite(config_.resourceWaitTimeoutMilliseconds) ||
      config_.resourceWaitTimeoutMilliseconds <= 0.0)
    throw std::invalid_argument("resource wait timeout must be positive and finite");
  if (config_.prefillCheckpointTokens &&
      (config_.prefillCheckpointTokens <
           model::ExecutionLimits::draftContextTokens ||
       config_.prefillCheckpointTokens % KvCache::pageTokens)) {
    throw std::invalid_argument(
        "prefill checkpoint interval must span a draft window and whole KV pages");
  }
}

void Engine::submit(EngineRequest value) {
  const bool scoring = !value.scoreTokens.empty();
  if (!value.id || value.prompt.empty() ||
      (scoring ? value.maxNewTokens != 0 : !value.maxNewTokens) ||
      value.prompt.size() + value.maxNewTokens > config_.maxContext ||
      !std::isfinite(value.deadlineMilliseconds) ||
      value.deadlineMilliseconds <= 0.0) {
    throw std::invalid_argument("invalid backend request");
  }
  if (std::any_of(value.prompt.begin(), value.prompt.end(), [&](uint32_t token) {
        return token >= config_.vocabularySize;
      })) {
    throw std::invalid_argument("prompt token is out of vocabulary");
  }
  uint64_t previousImageEnd = 0;
  uint64_t pixelBytes = 0;
  for (const ImageSpan &image : value.images) {
    const uint64_t patches = uint64_t{image.gridHeight} * image.gridWidth;
    if (image.gridHeight < 2 || image.gridWidth < 2 || image.gridHeight % 2 ||
        image.gridWidth % 2 || patches > config_.maxImagePatches ||
        image.tokens != (image.gridHeight / 2) * (image.gridWidth / 2) ||
        image.offset < previousImageEnd ||
        uint64_t{image.offset} + image.tokens > value.prompt.size()) {
      throw std::invalid_argument("invalid backend request image span");
    }
    previousImageEnd = image.end();
    pixelBytes += image.pixelBytes();
  }
  if (value.imagePixels.size() != pixelBytes) {
    throw std::invalid_argument("invalid backend request image pixels");
  }
  const uint64_t id = value.id;
  if (scoring) {
    if (value.cohort != BatchCohort::Greedy ||
        value.constraint != ConstraintMode::None || !value.images.empty() ||
        value.sampling.temperature != 0.0f || value.sampling.topP != 1.0f ||
        value.sampling.topK != 0 ||
        value.scoreTokens.size() < model::ExecutionLimits::minimumScoreOptions ||
        value.scoreTokens.size() > model::ExecutionLimits::maximumScoreOptions) {
      throw std::invalid_argument("invalid score request");
    }
    std::vector<uint32_t> distinct(value.scoreTokens.begin(),
                                   value.scoreTokens.end());
    std::sort(distinct.begin(), distinct.end());
    if (std::adjacent_find(distinct.begin(), distinct.end()) !=
            distinct.end() ||
        std::any_of(distinct.begin(), distinct.end(), [&](uint32_t token) {
          return token >= config_.vocabularySize;
        })) {
      throw std::invalid_argument("score token is out of vocabulary");
    }
  }
  Request requestState;
  requestState.storageGeneration = nextStorageGeneration_++;
  if (!requestState.storageGeneration)
    requestState.storageGeneration = nextStorageGeneration_++;
  requestState.promptTokens = static_cast<uint32_t>(value.prompt.size());
  requestState.replayTokens = requestState.promptTokens;
  requestState.request = std::move(value);
  auto [entry, inserted] = requests_.emplace(id, std::move(requestState));
  if (!inserted)
    throw std::invalid_argument("duplicate backend request id");
  const Request &stored = entry->second;
  scheduler_.submit(
      {.id = id,
       .priority = stored.request.priority,
       .cohort = stored.request.cohort,
       .promptTokens = stored.promptTokens,
       .deadlineMilliseconds = stored.request.deadlineMilliseconds});
  ++counters_.submitted;
}

void Engine::cancel(uint64_t id) {
  auto found = requests_.find(id);
  if (found == requests_.end() || found->second.finalized)
    return;
  if (pending_) {
    for (const BatchItem &item : pending_->plan.items) {
      if (item.requestId == id) {
        // An earlier deadline failure of the same in-flight lane stands.
        if (!found->second.failure) {
          found->second.failure = Failure{"cancelled", "request cancelled"};
        }
        pending_->ticket->abandonMask(id);
        return;
      }
    }
  }
  finish(found->second, EngineFinishReason::Cancelled, {});
}

bool Engine::storageReady(uint64_t id, uint64_t generation) {
  auto found = requests_.find(id);
  if (found == requests_.end() || found->second.finalized ||
      found->second.storageGeneration != generation ||
      scheduler_.phase(id) != Phase::WaitingStorage)
    return false;
  scheduler_.storageReady(id);
  signalResourceProgress();
  if (completionNotifier_)
    completionNotifier_();
  return true;
}

void Engine::failRequest(uint64_t id, std::string code, std::string message) {
  Request &active = request(id);
  if (active.finalized || active.failure)
    return;
  Failure failure{std::move(code), std::move(message)};
  if (pending_) {
    for (const BatchItem &item : pending_->plan.items) {
      if (item.requestId == id) {
        active.failure = std::move(failure);
        pending_->ticket->abandonMask(id);
        return;
      }
    }
  }
  finishFailure(active, std::move(failure));
}

void Engine::provideMask(uint64_t id, std::span<const uint32_t> words) {
  Request &active = request(id);
  const bool ownedByActiveBatch =
      pending_ && pending_->ticket->ownsMaskWait(id);
  // A request that already left its mask wait (cancellation, deadline, or a
  // failure raced the frontend) treats the response as stale.
  if (active.finalized || active.failure ||
      (!ownedByActiveBatch && scheduler_.phase(id) != Phase::WaitingMask)) {
    return;
  }
  model_.provideMask(id, words);
  if (!ownedByActiveBatch)
    scheduler_.maskReady(id);
}

void Engine::setCompletionNotifier(std::function<void()> notifier) {
  completionNotifier_ = std::move(notifier);
}

namespace {
// A long idle gap lets the GPU drop its clocks, which inflates the next
// command's wall time and, through the duty formula, the next pause.
constexpr double kMaxDutyPauseMilliseconds = 250.0;
} // namespace

bool Engine::tick(double now) {
  model_.checkHealth();
  nextHealthCheckMilliseconds_ = now + kHealthCheckIntervalMilliseconds;
  bool progressed = scheduler_.expireDeadlines(now);
  const bool draining = drainingForRecovery();
  for (auto &[_, active] : requests_) {
    // Admission is deliberately paused while resident peers finish. Start a
    // fresh resource wait only if admission still fails after that drain.
    if (draining) {
      active.resourceWait.deadlineMilliseconds = 0.0;
      continue;
    }
    if (!active.finalized && active.resourceWait.deadlineMilliseconds > 0.0 &&
        now >= active.resourceWait.deadlineMilliseconds) {
      finishFailure(active, {"resource_timeout", "memory did not become available within the resource wait limit", true});
      progressed = true;
    }
  }
  // Finalize expired requests now, even while a command is in flight, so a
  // late mask or cancel for them is a no-op rather than a scheduler error.
  if (progressed)
    sweepTerminal();
  if (pending_) {
    auto forwardMaskRequests = [&] {
      for (ModelMaskRequest &request : pending_->ticket->takeMaskRequests()) {
        events_.maskRequested(request.requestId, request.simulationTokens);
        progressed = true;
      }
    };
    forwardMaskRequests();
    for (const BatchItem &item : pending_->plan.items) {
      Request &active = request(item.requestId);
      if (!active.failure &&
          active.request.deadlineMilliseconds <= now) {
        active.failure =
            Failure{"deadline_exceeded", "request deadline exceeded"};
        pending_->ticket->abandonMask(item.requestId);
        progressed = true;
      }
    }
    // A mask response, cancellation, or deadline can make the commit tail
    // runnable without another Metal completion wake.
    forwardMaskRequests();
    if (!pending_->ticket->ready())
      return progressed;
    Pending command = std::move(*pending_);
    pending_.reset();
    std::vector<ModelStepResult> results = command.ticket->wait();
    apply(command.plan, results, command.ticket->wallMilliseconds(),
          command.ticket->prefillTimingIsRepresentative());
    if (config_.dutyPercent < 100) {
      nextDispatchMilliseconds_ =
          now + std::min(kMaxDutyPauseMilliseconds,
                         command.ticket->wallMilliseconds() *
                             (100.0 - config_.dutyPercent) /
                             config_.dutyPercent);
    }
    sweepTerminal();
    return true;
  }

  progressed = admitQueued(now) || progressed;
  sweepTerminal();
  // Metal may need a transient placement even for a resident decode lane.
  // At critical host pressure keep the committed state ready and resume it
  // when the governor allows commands again, instead of turning an optional
  // pressure episode into a process-wide Metal execution failure.
  if (config_.dispatchPaused && config_.dispatchPaused())
    return progressed;
  if (dispatchHeld_)
    return progressed;
  if (config_.dutyPercent < 100 && now < nextDispatchMilliseconds_)
    return progressed;
  auto plan = scheduler_.next();
  if (std::FILE *trace = decodeReadyTrace())
    tracePlan(trace, "plan", scheduler_, plan ? &*plan : nullptr,
              traceMonotonicMilliseconds());
  if (!plan)
    return progressed;
  std::vector<ModelBatchItem> items;
  if (!prepare(*plan, items, now)) {
    sweepTerminal();
    return true;
  }

  std::unique_ptr<ModelBatchTicket> ticket =
      model_.submit(*plan, items, completionNotifier_);
  if (!ticket) {
    throw std::logic_error("model returned an empty command ticket");
  }
  scheduler_.commit(*plan);
  if (std::FILE *trace = decodeReadyTrace())
    tracePlan(trace, "submit", scheduler_, &*plan,
              traceMonotonicMilliseconds());
  pending_ = Pending{std::move(*plan), std::move(ticket)};
  return true;
}

bool Engine::idle() const noexcept { return requests_.empty() && !pending_; }

bool Engine::drainingForRecovery() const {
  return recoveringResources_ &&
         std::any_of(requests_.begin(), requests_.end(), [](const auto &entry) {
           return entry.second.stateCell.has_value();
         });
}

std::optional<double> Engine::nextWakeupMilliseconds() const {
  std::optional<double> result;
  if (pending_ || model_.needsHealthCheck())
    result = nextHealthCheckMilliseconds_;
  const bool draining = drainingForRecovery();
  for (const auto &[_, active] : requests_) {
    if (active.finalized || active.failure)
      continue;
    if (!result || active.request.deadlineMilliseconds < *result)
      result = active.request.deadlineMilliseconds;
    if (!draining && active.resourceWait.deadlineMilliseconds > 0.0 &&
        (!result || active.resourceWait.deadlineMilliseconds < *result))
      result = active.resourceWait.deadlineMilliseconds;
    if (draining || active.resourceWait.retryMilliseconds <= 0.0)
      continue;
    const double wakeup = active.resourceWait.epoch == resourceEpoch_
                              ? active.resourceWait.retryMilliseconds
                              : 0.0;
    if (!result || wakeup < *result)
      result = wakeup;
  }
  // Without a command in flight, a throttled engine with work waits for the
  // duty-cycle window instead of sleeping until the next request event.
  if (!pending_ && config_.dutyPercent < 100 && !requests_.empty() &&
      (!result || nextDispatchMilliseconds_ < *result))
    result = nextDispatchMilliseconds_;
  return result;
}

EngineSnapshot Engine::snapshot() const {
  EngineSnapshot result = counters_;
  result.maximumContextTokens = config_.maxContext;
  result.scheduler = scheduler_.snapshot();
  result.resources = cache_.snapshot();
  return result;
}

ResourceWaitSnapshot Engine::resourceWaitSnapshot(double now) const {
  ResourceWaitSnapshot result;
  result.draining = drainingForRecovery();
  for (const auto &[id, active] : requests_) {
    if (active.finalized || scheduler_.phase(id) != Phase::WaitingResources)
      continue;
    if (active.resourceWait.reason == StateFailure::ConcurrencyLimit)
      ++result.concurrency;
    else
      ++result.memory;
    if (active.suspended)
      ++result.suspended;
    if (active.resourceWait.startedMilliseconds)
      result.oldestWaitMilliseconds = std::max(
          result.oldestWaitMilliseconds, now - *active.resourceWait.startedMilliseconds);
  }
  return result;
}

bool Engine::admitQueued(double now) {
  const bool recovering = std::any_of(
      requests_.begin(), requests_.end(),
      [](const auto &entry) { return entry.second.suspended; });
  // Once pressure has preempted work, let resident lanes finish before
  // spending their released headroom on a retry or a new request. Recover
  // one lane at a time; this prevents repeated B4 admission/preemption churn.
  if (!recovering)
    recoveringResources_ = false;
  if (drainingForRecovery())
    return false;
  const std::vector<uint64_t> order = scheduler_.admissionOrder();
  if (recovering) {
    for (uint64_t id : order) {
      Request &active = request(id);
      if (active.suspended && resourceRetryReady(active, now) && admit(active, now))
        return true;
    }
    return false;
  }

  std::vector<PrefillAdmission> candidates;
  for (uint64_t id : order) {
    Request &active = request(id);
    if (!resourceRetryReady(active, now))
      continue;
    // A live coherent RAM prefix is already cheaper to resume than opening
    // its durable copy. Probe first so optional storage never stalls a hot
    // follow-up behind a read or an in-flight sidecar writer.
    active.admissionProbe =
        cache_.probe(active.request.prompt, active.request.images);
    const uint32_t cached = active.admissionProbe->cachedTokens();
    if (config_.beginStorageLookup && !active.storageLookupStarted && !cached) {
      active.storageLookupStarted = true;
      bool waiting = false;
      try {
        waiting = config_.beginStorageLookup(
            id, active.storageGeneration, active.request.prompt,
            active.request.images);
      } catch (const std::exception &) {
        // An optional cache failure is a cold-path miss.
      }
      if (waiting) {
        active.admissionProbe.reset();
        scheduler_.waitForStorage(id);
        continue;
      }
    }
    if (pendingSharedPrefill(active, cached)) {
      active.admissionProbe.reset();
      active.resourceWait = {};
      scheduler_.waitForPrefix(id);
      continue;
    }
    candidates.push_back({id, cached});
  }
  bool progressed = false;
  while (!candidates.empty()) {
    const auto selected = scheduler_.prefillAdmissionOrder(candidates);
    if (selected.empty())
      break;
    for (uint64_t id : selected) {
      progressed = admit(request(id), now) || progressed;
      std::erase_if(candidates, [id](const auto &value) {
        return value.requestId == id;
      });
    }
    // Failed admissions must not prevent other eligible work from running.
    if (progressed)
      break;
  }
  // Waiting for scheduling does not consume the memory-retry deadline.
  for (const auto &candidate : candidates) {
    Request &active = request(candidate.requestId);
    active.admissionProbe.reset();
    active.resourceWait = {};
    scheduler_.deferAdmission(candidate.requestId);
  }
  return progressed;
}

uint32_t Engine::sharedPrefillBoundary(const Request &left,
                                       const Request &right) {
  const auto prompt = [](const Request &value) -> std::span<const uint32_t> {
    return value.exactTokens.empty()
               ? std::span<const uint32_t>(value.request.prompt)
               : std::span<const uint32_t>(value.exactTokens)
                     .first(value.promptTokens);
  };
  const auto a = prompt(left);
  const auto b = prompt(right);
  const auto end = std::mismatch(a.begin(), a.end(), b.begin(), b.end()).first;
  uint32_t boundary = std::min<uint32_t>(
      static_cast<uint32_t>(end - a.begin()),
      std::min(replayStateBoundary(left.promptTokens),
               replayStateBoundary(right.promptTokens)));
  boundary -= boundary % KvCache::pageTokens;
  if (!left.request.images.empty() || !right.request.images.empty()) {
    for (uint32_t offset = 0; offset < boundary; offset += KvCache::pageTokens) {
      if (blockImageIdentity(offset, KvCache::pageTokens, left.request.images) !=
          blockImageIdentity(offset, KvCache::pageTokens, right.request.images))
        return offset;
    }
  }
  return boundary;
}

bool Engine::pendingSharedPrefill(const Request &active,
                                  uint32_t resumeBoundary) const {
  for (const auto &[id, peer] : requests_) {
    if (!peer.stateCell || peer.finalized || peer.failure ||
        peer.request.priority > active.request.priority ||
        scheduler_.phase(id) != Phase::Prefill)
      continue;
    const uint32_t shared = sharedPrefillBoundary(active, peer);
    for (size_t i = peer.stateBoundaryCursor; i < peer.stateBoundaries.size(); ++i) {
      const uint32_t boundary = peer.stateBoundaries[i].tokens;
      if (boundary > resumeBoundary && boundary <= shared)
        return true;
    }
  }
  return false;
}

bool Engine::admit(Request &active, double now) {
  const bool resuming = active.suspended;
  ModelRequest modelRequest = active.request.modelView();
  if (resuming)
    modelRequest.prompt = active.exactTokens;
  CacheLookup lookup = cache_.lookup(
      modelRequest.prompt, active.request.images,
      active.admissionProbe ? &*active.admissionProbe : nullptr);
  active.admissionProbe.reset();
  // Only unstarted requests wait for a resident producer. Recheck planned
  // boundaries each step so producer loss leaves no stale dependency or lease.
  if (!resuming && pendingSharedPrefill(active, lookup.resumeBoundary())) {
    active.resourceWait = {};
    scheduler_.waitForPrefix(active.request.id);
    return false;
  }
  bool executorStarted = false;
  bool resourcesStarted = false;
  try {
    const auto activate = [&] {
      return resuming ? model_.resume(modelRequest) : model_.begin(modelRequest);
    };
    const uint64_t releaseGeneration = cache_.releaseGeneration();
    StateAdmission admission = activate();
    bool reclaimedForAdmission = false;
    while (!admission.granted() &&
           admission.failure == StateFailure::MemoryPressure) {
      const bool hostPressure =
          admission.allocationFailure == metal::AllocationFailure::HostPressure;
      if (hostPressure ? reclaimIdleState() : reclaimForGrowth()) {
        reclaimedForAdmission = true;
        admission = activate();
        continue;
      }
      if (hostPressure)
        break;
      // A useful restore remains pinned throughout ordinary eviction. If
      // that pin is the last obstacle to admitting even one lane, prefer
      // cold recomputation over waiting forever for our own cache lease.
      if (!growthPaused() && lookup.state) {
        lookup = {};
        continue;
      }
      break;
    }
    if (!admission.granted()) {
      const bool terminalAllocation =
          admission.allocationFailure == metal::AllocationFailure::EngineBudget ||
          admission.allocationFailure == metal::AllocationFailure::DriverRejected;
      const bool anotherResident = std::any_of(
          requests_.begin(), requests_.end(), [&](const auto &entry) {
            return entry.first != active.request.id && entry.second.stateCell;
          });
      const bool releasingBudget = budgetMayRecover(
          admission.allocationFailure, releaseGeneration, reclaimedForAdmission);
      if (terminalAllocation && !growthPaused() && !anotherResident &&
          !releasingBudget) {
        finishFailure(active,
                      {"capacity_exhausted",
                       std::string("could not allocate request state: ") +
                           metal::allocationFailureName(
                               admission.allocationFailure),
                       true});
        return true;
      }
      scheduler_.waitForResources(active.request.id);
      deferResourceRetry(active, now, admission.failure);
      return false;
    }
    executorStarted = true;
    cache_.beginRequest(active.request.id);
    resourcesStarted = true;
    active.stateCell = *admission.cell;
    const uint32_t resumeBoundary = lookup.resumeBoundary();
    if (lookup.state)
      cache_.restoreRequest(active.request.id, lookup);
    if (resuming) {
      const auto [kv, retryableBudget] =
          admitKv(active, active.resumeKvTargetTokens);
      if (!kv.granted()) {
        // The host continuation survives this failed admission. No recurrent
        // state restore or replay has run, and all temporary leases are freed.
        model_.suspend(active.request.id);
        cache_.endRequest(active.request.id);
        active.stateCell.reset();
        executorStarted = resourcesStarted = false;
        if (!growthPaused() && !retryableBudget &&
            kv.allocationFailure != metal::AllocationFailure::HostPressure) {
          finishCapacity(active, kv);
          return true;
        }
        deferResourceRetry(active, now);
        return false;
      }
      active.resumeKvTargetTokens = 0;
    }
    active.resourceWait = {};
    if (resuming)
      scheduler_.resumeFromResources(active.request.id, resumeBoundary,
                                     active.replayTokens);
    DraftContextPlan draft = configureDraftStatePlan(
        active, resumeBoundary, lookup.junctionBoundary());
    if (std::FILE *trace = debugTrace()) {
      std::fprintf(trace,
                   "admit id %llu prompt %u replay %u resume %u kv %u "
                   "junction %u draft_restore %d captures",
                   static_cast<unsigned long long>(active.request.id),
                   active.promptTokens, active.replayTokens, resumeBoundary,
                   lookup.kvBoundary, lookup.junctionBoundary(),
                   lookup.state ? int(!draft.draftStateRestoreSkipped) : -1);
      for (const DraftCaptureSpan &span : draft.captures())
        std::fprintf(trace, " [%u,%u)%s", span.begin, span.end,
                     span.resetDraftState ? "r" : "");
      std::fprintf(trace, " boundaries");
      for (const DraftBoundaryPlan &boundary : draft.plannedBoundaries())
        std::fprintf(trace, " %u", boundary.boundary);
      std::fprintf(trace, "\n");
      std::fflush(trace);
    }
    active.latestCheckpoint = {};
    if (lookup.state) {
      model_.restore(active.request.id, resumeBoundary, lookup.state->state(),
                     !draft.draftStateRestoreSkipped);
      if (config_.publishStorage &&
          modelRequest.prompt.size() >= resumeBoundary) {
        try {
          config_.publishStorage(
              lookup.state->kvBlock(), resumeBoundary,
              modelRequest.prompt.first(resumeBoundary),
              active.request.images, lookup.state->state());
        } catch (const std::exception &) {
        }
      }
      active.latestCheckpoint = cache_.checkpointState(lookup.state->kvBlock());
      // A restored endpoint already has the ordinary replay state we need.
      // Other restored progress points retain their rolling lifetime.
      if (active.latestCheckpoint &&
          resumeBoundary == replayStateBoundary(active.replayTokens)) {
        static_cast<void>(
            cache_.reuseCompositeState(active.latestCheckpoint.kvBlock));
        ++counters_.deduplicatedStatePublications;
        active.latestCheckpoint = {};
      }
    }
    model_.setDraftContextPlan(active.request.id, std::move(draft));
    if (resuming) {
      active.suspended = false;
      active.replaying = true;
      ++counters_.resourceResumptions;
      return true;
    }
    active.exactTokens = std::move(active.request.prompt);
    scheduler_.resourcesReady(active.request.id, resumeBoundary);
    cache_.recordLookup(lookup);
    events_.started(active.request.id,
                    resumeBoundary ? EngineCacheStatus::PrefixHit
                                   : EngineCacheStatus::Miss,
                    resumeBoundary, *admission.cell);
    if (active.request.returnProgress) {
      active.reportedPromptTokens = resumeBoundary;
      events_.promptProgress(active.request.id, resumeBoundary);
    }
    if (resumeBoundary) {
      ++counters_.cacheHits;
      counters_.reusedTokens += resumeBoundary;
    } else {
      ++counters_.coldMisses;
    }
    return true;
  } catch (...) {
    discardPendingStateBoundaries(active);
    if (executorStarted)
      model_.end(active.request.id);
    if (resourcesStarted)
      cache_.endRequest(active.request.id);
    active.stateCell.reset();
    throw;
  }
}

bool Engine::resourceRetryReady(const Request &active,
                                double now) const noexcept {
  return active.resourceWait.retryMilliseconds <= 0.0 ||
         active.resourceWait.epoch != resourceEpoch_ ||
         now >= active.resourceWait.retryMilliseconds;
}

void Engine::deferResourceRetry(Request &active, double now,
                                StateFailure reason) noexcept {
  auto &wait = active.resourceWait;
  if (!wait.startedMilliseconds)
    wait.startedMilliseconds = now;
  wait.reason = reason;
  if (reason == StateFailure::ConcurrencyLimit)
    wait.deadlineMilliseconds = 0.0;
  else if (wait.deadlineMilliseconds <= 0.0)
    wait.deadlineMilliseconds = now + config_.resourceWaitTimeoutMilliseconds;
  wait.epoch = resourceEpoch_;
  wait.retryMilliseconds = now + kResourceRetryBackoffMilliseconds;
}

void Engine::signalResourceProgress() noexcept {
  if (resourceEpoch_ != std::numeric_limits<uint64_t>::max())
    ++resourceEpoch_;
}

DraftContextPlan Engine::configureDraftStatePlan(Request &active,
                                                 uint32_t stateBoundary,
                                                 uint32_t junctionBoundary) {
  if (!active.stateBoundaries.empty() || active.stateBoundaryCursor != 0) {
    throw std::logic_error("request already has a composite-state plan");
  }

  const auto addCandidate = [&](uint32_t tokens,
                                Request::StateBoundary::Purpose purpose) {
    if (!tokens || tokens <= stateBoundary)
      return;
    for (size_t index = 0; index < active.stateBoundaries.size(); ++index) {
      if (active.stateBoundaries[index].tokens != tokens)
        continue;
      if (purpose > active.stateBoundaries[index].purpose)
        active.stateBoundaries[index].purpose = purpose;
      return;
    }
    active.stateBoundaries.push_back({tokens, purpose});
  };

  // Plan draft windows before prefill; arbitrary chunk ends do not carry a
  // complete draft state. Progress points remain disposable after restoration.
  const uint32_t latestReplayBoundary = replayStateBoundary(active.replayTokens);
  if (const uint32_t interval = config_.prefillCheckpointTokens) {
    for (uint64_t boundary = (uint64_t{stateBoundary} / interval + 1) * interval;
         boundary < latestReplayBoundary; boundary += interval) {
      addCandidate(static_cast<uint32_t>(boundary),
                   Request::StateBoundary::Purpose::Checkpoint);
    }
  }
  addCandidate(junctionBoundary, Request::StateBoundary::Purpose::Junction);
  addCandidate(latestReplayBoundary, Request::StateBoundary::Purpose::Replay);
  std::sort(active.stateBoundaries.begin(), active.stateBoundaries.end(),
            [](const Request::StateBoundary &left,
               const Request::StateBoundary &right) {
              return left.tokens < right.tokens;
            });

  static_cast<void>(addSharedPrefillBoundaries(active, stateBoundary));

  try {
    DraftContextPlan draft = pendingDraftStatePlan(active, stateBoundary);
    armNextStateBoundary(active);
    return draft;
  } catch (...) {
    discardPendingStateBoundaries(active);
    throw;
  }
}

bool Engine::addSharedPrefillBoundaries(Request &active, uint32_t after) {
  if (active.suspended || active.replaying)
    return false;
  bool changed = false;
  const uint32_t replay = replayStateBoundary(active.replayTokens);
  for (const auto &[id, peer] : requests_) {
    if (id == active.request.id || peer.stateCell || peer.suspended ||
        peer.finalized || peer.failure ||
        peer.request.priority < active.request.priority)
      continue;
    const uint32_t shared = sharedPrefillBoundary(active, peer);
    if (shared <= after || shared >= replay)
      continue;
    auto found = std::lower_bound(
        active.stateBoundaries.begin(), active.stateBoundaries.end(), shared,
        [](const auto &point, uint32_t tokens) { return point.tokens < tokens; });
    if (found == active.stateBoundaries.end() || found->tokens != shared) {
      active.stateBoundaries.insert(
          found, {shared, Request::StateBoundary::Purpose::Junction});
      changed = true;
    } else if (found->purpose == Request::StateBoundary::Purpose::Checkpoint) {
      found->purpose = Request::StateBoundary::Purpose::Junction;
    }
  }
  return changed;
}

DraftContextPlan Engine::pendingDraftStatePlan(const Request &active,
                                               uint32_t stateBoundary) const {
  std::vector<uint32_t> boundaries;
  boundaries.reserve(active.stateBoundaries.size() - active.stateBoundaryCursor);
  for (size_t i = active.stateBoundaryCursor; i < active.stateBoundaries.size(); ++i)
    boundaries.push_back(active.stateBoundaries[i].tokens);
  return planDraftContext(
      stateBoundary, active.replayTokens,
      stateBoundary ? std::optional<uint32_t>(stateBoundary) : std::nullopt,
      boundaries);
}

void Engine::armNextStateBoundary(Request &active) {
  const std::optional<uint32_t> next =
      active.stateBoundaryCursor < active.stateBoundaries.size()
          ? std::optional<uint32_t>(
                active.stateBoundaries[active.stateBoundaryCursor].tokens)
          : std::nullopt;
  scheduler_.setPrefillBoundary(active.request.id, next);
}

void Engine::discardPendingStateBoundaries(Request &active) noexcept {
  active.stateBoundaries.clear();
  active.stateBoundaryCursor = 0;
}

bool Engine::retireCheckpoint(Request &active) {
  // Shared progress points remain disposable under memory pressure, but a
  // lane's normal rolling replacement must not retire its peer's recovery point.
  const auto point = active.latestCheckpoint;
  if (point && std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
        const auto &peer = entry.second;
        return &peer != &active && !peer.finalized &&
               peer.latestCheckpoint.kvBlock == point.kvBlock &&
               peer.latestCheckpoint.publication == point.publication;
      })) {
    active.latestCheckpoint = {};
    return true;
  }
  if (!cache_.retireCheckpointState(active.latestCheckpoint))
    return false;
  active.latestCheckpoint = {};
  return true;
}

void Engine::publishReachedStateBoundaries(Request &active,
                                           uint32_t promptProcessed) {
  bool materialized = false;
  while (active.stateBoundaryCursor < active.stateBoundaries.size() &&
         active.stateBoundaries[active.stateBoundaryCursor].tokens <=
             promptProcessed) {
    const Request::StateBoundary objective =
        active.stateBoundaries[active.stateBoundaryCursor++];
    const bool checkpoint =
        objective.purpose == Request::StateBoundary::Purpose::Checkpoint;
    const bool junction =
        objective.purpose == Request::StateBoundary::Purpose::Junction;
    uint64_t &failures = checkpoint ? counters_.checkpointPublicationFailures
                         : junction ? counters_.junctionMaterializationFailures
                                    : counters_.replayStatePublicationFailures;
    uint64_t &publications = checkpoint ? counters_.checkpointPublications
                             : junction ? counters_.junctionMaterializations
                                        : counters_.replayStatePublications;
    // The scheduler ends a command exactly at an armed boundary; a boundary
    // passed inside a command has no materialized state to copy.
    if (objective.tokens != promptProcessed) {
      ++failures;
      continue;
    }
    materialized = true;
    try {
      const uint64_t block = cache_.blockAt(active.request.id, objective.tokens);
      if (cache_.reuseCompositeState(block, checkpoint)) {
        ++counters_.deduplicatedStatePublications;
        if (!checkpoint && config_.publishStorage &&
            active.exactTokens.size() >= objective.tokens) {
          auto lease = cache_.acquireCompositeState(block);
          if (lease) {
            try {
              config_.publishStorage(
                  block, objective.tokens,
                  std::span<const uint32_t>(active.exactTokens).first(
                      objective.tokens),
                  active.request.images, lease->state());
            } catch (const std::exception &) {
            }
          }
        }
      } else {
        // Recycle the previous recovery point before allocating its replacement.
        // A restore lease can delay this optional publication.
        if (!retireCheckpoint(active) && checkpoint) {
          ++failures;
          continue;
        }
        std::shared_ptr<const CompositeState> state =
            model_.snapshot(active.request.id);
        if (!state && cache_.reclaimOneState(checkpoint)) {
          state = model_.snapshot(active.request.id);
          if (state)
            ++counters_.recycledStatePublications;
        }
        if (!state) {
          ++failures;
          continue;
        }
        const auto durableState = state;
        cache_.publishCompositeState(block, std::move(state), checkpoint);
        ++publications;
        if (!checkpoint && config_.publishStorage &&
            active.exactTokens.size() >= objective.tokens) {
          try {
            config_.publishStorage(
                block, objective.tokens,
                std::span<const uint32_t>(active.exactTokens).first(
                    objective.tokens),
                active.request.images, durableState);
          } catch (const std::exception &) {
            // Optional persistence cannot invalidate a committed RAM state.
          }
        }
      }
      if (active.latestCheckpoint.kvBlock != block)
        static_cast<void>(retireCheckpoint(active));
      active.latestCheckpoint = checkpoint ? cache_.checkpointState(block)
                                           : StateCheckpoint{};
    } catch (const std::exception &) {
      ++failures;
    }
  }
  // Late siblings can extend the remaining plan only where both target and
  // draft states are complete, never at an arbitrary in-flight chunk boundary.
  if (materialized && addSharedPrefillBoundaries(active, promptProcessed))
    model_.setDraftContextPlan(
        active.request.id, pendingDraftStatePlan(active, promptProcessed));
  if (active.stateBoundaryCursor == active.stateBoundaries.size()) {
    active.stateBoundaries.clear();
    active.stateBoundaryCursor = 0;
  }
  armNextStateBoundary(active);
}

bool Engine::prepare(BatchPlan &plan, std::vector<ModelBatchItem> &items,
                     double now) {
  items.reserve(plan.items.size());
  std::vector<BatchItem> admitted;
  admitted.reserve(plan.items.size());
  struct Denied final {
    uint64_t requestId = 0;
    TokenAdmission admission;
    uint64_t workEnd;
    bool retryableBudget;
  };
  std::vector<Denied> denied;
  denied.reserve(plan.items.size());
  for (const BatchItem &scheduled : plan.items) {
    Request &active = request(scheduled.requestId);
    if (!active.stateCell)
      throw std::logic_error("scheduled request is not resident");
    const uint64_t position = plan.kind == WorkKind::Prefill
                                  ? scheduled.promptOffset
                                  : active.exactTokens.size();
    const uint64_t workEnd =
        plan.kind == WorkKind::Prefill
            ? position + scheduled.tokenCount
            : position + model::ExecutionLimits::targetVerifyRows;
    const auto [admission, retryableBudget] = admitKv(active, workEnd);
    if (!admission.granted()) {
      denied.push_back(
          Denied{active.request.id, admission, workEnd, retryableBudget});
      continue;
    }
    admitted.push_back(scheduled);
    ModelBatchItem item;
    item.requestId = active.request.id;
    item.stateSlot = *active.stateCell;
    item.logicalPosition = position;
    item.promptOffset =
        plan.kind == WorkKind::Prefill ? scheduled.promptOffset : 0;
    item.tokenCount = scheduled.tokenCount;
    const PageTableView pageTable = cache_.pageTable(active.request.id);
    item.pageTable = pageTable.pages;
    item.pageTableRevision = pageTable.revision;
    if (plan.kind == WorkKind::Decode &&
        config_.decodeBoundaryRetention && config_.publishDecodeStorage &&
        active.request.cohort == BatchCohort::Greedy &&
        !active.decodeStorageCandidateSeen &&
        active.decodeStorageAttempts < 4 &&
        !active.kvPageReplacedByReuse) {
      const uint64_t nextPage =
          (position / KvCache::pageTokens + 1) *
          KvCache::pageTokens;
      if (position >= active.promptTokens && nextPage <= config_.maxContext &&
          nextPage - position <= model::ExecutionLimits::targetVerifyRows &&
          (!config_.decodeBoundaryReadyOnly ||
           (config_.decodeStorageReady && config_.decodeStorageReady()))) {
        item.decodeRetentionBoundary = static_cast<uint32_t>(nextPage);
        ++counters_.decodeRetentionDispatches;
      }
    }
    if (plan.kind == WorkKind::Prefill) {
      item.inputTokens =
          std::span<const uint32_t>(active.exactTokens)
              .subspan(scheduled.promptOffset, scheduled.tokenCount);
    }
    items.push_back(std::move(item));
  }
  if (!admitted.empty()) {
    plan.items = std::move(admitted);
    return true;
  }

  // Partial admissions execute at their actual width. If no lane fits, choose
  // among all runnable residents: an unstarted peer can release its state
  // cell before completed prefill is discarded.
  if (denied.empty())
    throw std::logic_error("empty resource admission result");
  const auto completedTokens = [&](const Request &active) -> uint64_t {
    return scheduler_.phase(active.request.id) == Phase::Prefill
               ? scheduler_.promptProcessed(active.request.id)
               : active.exactTokens.size();
  };
  const auto yieldsBefore = [&](const Request &a, const Request &b) {
    if (a.request.priority != b.request.priority)
      return a.request.priority > b.request.priority;
    const Phase aPhase = scheduler_.phase(a.request.id);
    const Phase bPhase = scheduler_.phase(b.request.id);
    // At equal priority, prefer uninterrupted streaming over less replay work.
    if (aPhase != bPhase)
      return aPhase == Phase::Prefill;
    return completedTokens(a) < completedTokens(b);
  };
  const Denied &victim = *std::min_element(
      denied.begin(), denied.end(),
      [&](const Denied &left, const Denied &right) {
        return yieldsBefore(request(left.requestId), request(right.requestId));
      });
  Request *selected = &request(victim.requestId);
  uint64_t resumeTarget = victim.workEnd;
  for (auto &[id, candidate] : requests_) {
    if (!candidate.stateCell)
      continue;
    // Requests enter the scheduler before they can acquire a resident cell.
    const Phase phase = scheduler_.phase(id);
    if ((phase == Phase::Prefill || phase == Phase::Decode) &&
        yieldsBefore(candidate, *selected)) {
      selected = &candidate;
      // This peer has not failed a growth attempt. Retain its current KV
      // capacity as the resume target, not the blocked lane's requirement.
      resumeTarget =
          uint64_t{cache_.pageTable(id).pages.size()} * KvCache::pageTokens;
    }
  }
  Request &active = *selected;
  const bool anotherResident =
      std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
        return entry.first != active.request.id && entry.second.stateCell;
      });
  if (growthPaused() || anotherResident || victim.retryableBudget ||
      victim.admission.allocationFailure ==
          metal::AllocationFailure::HostPressure) {
    suspendForGrowth(active, resumeTarget, now);
    return false;
  }
  finishCapacity(request(victim.requestId), victim.admission);
  return false;
}

Engine::KvAdmission Engine::admitKv(Request &active, uint64_t workEnd) {
  const uint64_t releaseGeneration = cache_.releaseGeneration();
  bool reclaimed = false;
  TokenAdmission admission = cache_.ensureTokens(active.request.id, workEnd);
  while (!admission.granted() &&
         admission.failure == KvPageAcquireFailure::PhysicalCapacity) {
    const bool paused = growthPaused() ||
        admission.allocationFailure == metal::AllocationFailure::HostPressure;
    const bool progressed = paused
        ? reuseIdleBackingWhilePaused(admission)
        : reclaimForGrowth(CacheReclaimMode::ReuseBacking);
    if (!progressed)
      break;
    reclaimed = true;
    admission = cache_.ensureTokens(active.request.id, workEnd);
  }
  return {admission,
          !admission.granted() &&
              budgetMayRecover(admission.allocationFailure, releaseGeneration,
                               reclaimed)};
}

bool Engine::budgetMayRecover(metal::AllocationFailure failure,
                             uint64_t generation, bool reclaimed) const {
  if (failure != metal::AllocationFailure::EngineBudget)
    return false;
  const bool pending = cache_.releasePending();
  return pending || reclaimed || cache_.releaseGeneration() != generation;
}

bool Engine::growthPaused() const {
  return config_.growthPaused && config_.growthPaused();
}

bool Engine::reclaimForGrowth(CacheReclaimMode mode) {
  if (reclaimIdleState())
    return true;
  // The background pressure controller owns physical shrink. Retrying a
  // paused allocator here would drain the cache before macOS can acknowledge
  // any reclaimed bytes.
  if (growthPaused())
    return false;
  const CacheReclaimResult reclaimed = cache_.reclaimOne(mode);
  if (reclaimed.madeProgress)
    signalResourceProgress();
  return reclaimed.madeProgress;
}

bool Engine::reclaimIdleState() noexcept {
  if (!model_.reclaimIdleState())
    return false;
  signalResourceProgress();
  return true;
}

// Host pressure pauses growth, and the pressure controller owns physical
// shrink. Backing that stays resident is outside that accounting: a request
// short of pages may take idle cached pages instead of being suspended and
// replaying its whole prefix once the pause lifts. Cache is only evicted when
// the resident idle pages can actually cover the shortfall; otherwise the
// request yields as before and the cache survives for later hits.
bool Engine::reuseIdleBackingWhilePaused(const TokenAdmission &admission) {
  if (reclaimIdleState())
    return true;
  const KvPoolSnapshot pool = cache_.snapshot().pool;
  // Cached prefixes can also have active owners; those pages cannot be reused.
  const uint32_t reusable = pool.pagesResident - pool.pagesActive;
  if (reusable < admission.additionalPages)
    return false;
  const CacheReclaimResult reused =
      cache_.reclaimOne(CacheReclaimMode::ReuseBacking);
  if (reused.madeProgress) {
    // An evicted state parks its buffers in the model's pool; under pressure
    // that memory goes back to the host now rather than waiting for the
    // next background pass.
    while (model_.reclaimIdleState()) {
    }
    signalResourceProgress();
  }
  return reused.madeProgress;
}

void Engine::suspendForGrowth(Request &active, uint64_t workEnd, double now) {
  if (!active.stateCell || active.suspended) {
    throw std::logic_error("request cannot be suspended for growth");
  }
  if (!active.stateBoundaries.empty()) {
    discardPendingStateBoundaries(active);
    scheduler_.setPrefillBoundary(active.request.id, std::nullopt);
  }
  model_.suspend(active.request.id);
  cache_.endRequest(active.request.id);
  active.stateCell.reset();
  active.suspended = true;
  active.resumeKvTargetTokens = workEnd;
  recoveringResources_ = true;
  active.replayTokens = static_cast<uint32_t>(active.exactTokens.size());
  scheduler_.suspendForResources(active.request.id);
  deferResourceRetry(active, now);
  ++counters_.resourceSuspensions;
}

uint64_t Engine::reclaimMemory(const MemoryReclaimDirective &directive) {
  if (!directive.reclaimEmptyKvExtents)
    return 0;

  uint64_t released = 0;
  while (const uint64_t idle = model_.reclaimIdleState())
    released += idle;
  const uint64_t remaining =
      released >= directive.targetBytes ? 0 : directive.targetBytes - released;
  // Even a zero-byte directive may release completely empty KV extents.
  released += cache_.reclaimCache(remaining, directive.evictAllUnpinnedPrefixes,
                                 directive.keepResumePoint);
  // Evicted states park their buffers in the model's pool; a pressure pass
  // returns that memory to the host now rather than keeping it warm.
  while (model_.reclaimIdleState()) {
  }
  if (released)
    signalResourceProgress();
  return released;
}

void Engine::apply(const BatchPlan &plan,
                   std::span<const ModelStepResult> results,
                   double wallMilliseconds, bool representativePrefillTiming) {
  if (results.size() != plan.items.size()) {
    throw std::logic_error("model result count changed");
  }
  std::vector<StepResult> schedulerResults;
  schedulerResults.reserve(results.size());
  uint32_t inputTokens = 0;
  uint32_t outputTokens = 0;
  uint32_t draftedTokens = 0;
  uint32_t acceptedDraftTokens = 0;
  for (size_t index = 0; index < results.size(); ++index) {
    const ModelStepResult &result = results[index];
    const BatchItem &item = plan.items[index];
    if (result.requestId != item.requestId) {
      throw std::logic_error("model result order changed");
    }
    Request &active = request(result.requestId);
    if (!result.failure.empty() && !active.failure) {
      // The model rejected this lane's own numerical result. An earlier
      // cancellation or deadline failure of the same lane still stands.
      active.failure = Failure{"model_result_invalid", result.failure};
    }
    if (active.failure) {
      // An in-flight Metal command cannot be revoked safely. Its provisional
      // writes remain invisible, but a cancelled, deadline-expired or
      // model-rejected request must not publish cache state or emit output
      // when that command drains.
      schedulerResults.push_back({active.request.id,
                                  result.consumedPromptTokens, true,
                                  result.nextDecodeStage});
      continue;
    }
    if (result.outputTokensWithoutKv > result.outputTokens.size()) {
      throw std::logic_error("model reported more uncommitted tokens than output");
    }
    if (plan.kind == WorkKind::Prefill) {
      const uint32_t promptProcessed =
          item.promptOffset + result.consumedPromptTokens;
      inputTokens += result.consumedPromptTokens;
      if (active.replaying)
        counters_.resourceReplayTokens += result.consumedPromptTokens;
      if (promptProcessed == active.replayTokens)
        active.replaying = false;
      bool reusedPage = false;
      static_cast<void>(cache_.publishCommittedBlocks(
          active.request.id, active.exactTokens, promptProcessed,
          active.request.images, &reusedPage));
      active.kvPageReplacedByReuse |= reusedPage;
      publishReachedStateBoundaries(active, promptProcessed);
      // Recovery may replay an already reported prefix, including generated
      // history.
      const uint32_t processed = std::min(promptProcessed, active.promptTokens);
      if (active.request.returnProgress &&
          processed > active.reportedPromptTokens) {
        active.reportedPromptTokens = processed;
        events_.promptProgress(active.request.id, processed);
      }
    }
    if (!result.outputTokens.empty()) {
      active.exactTokens.insert(active.exactTokens.end(),
                                result.outputTokens.begin(),
                                result.outputTokens.end());
      outputTokens += static_cast<uint32_t>(result.outputTokens.size());
      events_.tokens(active.request.id, result.outputTokens);
    }
    if (plan.kind == WorkKind::Decode) {
      draftedTokens += result.draftedTokens;
      acceptedDraftTokens += result.acceptedDraftTokens;
      // A terminal anchor is emitted without a target KV row; it never enters
      // a cached block.
      const uint32_t storedTokens =
          static_cast<uint32_t>(active.exactTokens.size()) -
          result.outputTokensWithoutKv;
      const uint32_t priorTokens = static_cast<uint32_t>(
          active.exactTokens.size() - result.outputTokens.size());
      if (storedTokens / KvCache::pageTokens >
          priorTokens / KvCache::pageTokens)
        counters_.decodePageCrossings +=
            storedTokens / KvCache::pageTokens -
            priorTokens / KvCache::pageTokens;
      if (storedTokens > active.promptTokens &&
          storedTokens % KvCache::pageTokens == 0)
        ++counters_.decodeExactPageLandings;
      bool reusedPage = false;
      const uint64_t committedBlock = cache_.publishCommittedBlocks(
          active.request.id, active.exactTokens, storedTokens,
          active.request.images, &reusedPage);
      active.kvPageReplacedByReuse |= reusedPage;
      if (config_.publishDecodeStorage && !active.decodeStorageCandidateSeen &&
          storedTokens > active.promptTokens &&
          storedTokens % KvCache::pageTokens == 0) {
        if (active.kvPageReplacedByReuse)
          ++counters_.decodeCaptureSkippedLineage;
        else if (storedTokens != active.exactTokens.size())
          ++counters_.decodeCaptureSkippedTerminal;
      }
      // A speculative step may jump over a Page32 boundary. Never label its
      // later recurrent state as the earlier KV page. Capture only an exact,
      // fully committed decode boundary, once per request, and let storage
      // reserve before the model copies its state.
      if (!active.decodeStorageCandidateSeen &&
          active.decodeStorageAttempts < 4 &&
          !active.kvPageReplacedByReuse &&
          config_.publishDecodeStorage &&
          storedTokens > active.promptTokens &&
          storedTokens == active.exactTokens.size() &&
          storedTokens % KvCache::pageTokens == 0) {
        ++active.decodeStorageAttempts;
        ++counters_.decodeCaptureAttempts;
        try {
          active.decodeStorageCandidateSeen = config_.publishDecodeStorage(
              committedBlock, storedTokens,
              std::span<const uint32_t>(active.exactTokens).first(storedTokens),
              active.request.images,
              [&] { return model_.snapshot(active.request.id); });
        } catch (const std::exception &) {
          // Optional durable publication cannot fail committed decode.
        }
      }
    }
    const uint64_t completionTokens =
        active.exactTokens.size() - active.promptTokens;
    const bool scoring = !active.request.scoreTokens.empty();
    if (scoring && !result.outputTokens.empty()) {
      throw std::logic_error("score request produced output tokens");
    }
    if (!result.scoreLogits.empty()) {
      if (!scoring ||
          result.scoreLogits.size() != active.request.scoreTokens.size()) {
        throw std::logic_error("model returned mismatched score logits");
      }
      active.scoreLogits = result.scoreLogits;
    }
    // Score requests carry maxNewTokens == 0; only the model's finished flag
    // on the final prompt chunk completes them.
    const bool complete =
        result.finished ||
        (!scoring && completionTokens >= active.request.maxNewTokens);
    if (scoring && complete &&
        item.promptOffset + result.consumedPromptTokens !=
            active.promptTokens) {
      throw std::logic_error("score request finished before the prompt ended");
    }
    if (result.outputTokensWithoutKv && !complete) {
      throw std::logic_error("model emitted an uncommitted token and continued");
    }
    schedulerResults.push_back({active.request.id, result.consumedPromptTokens,
                                complete, result.nextDecodeStage,
                                static_cast<uint32_t>(result.outputTokens.size())});
    if (std::FILE *trace = debugTrace()) {
      std::fprintf(trace, "%s id %llu offset %u rows %u width %zu out",
                   plan.kind == WorkKind::Prefill ? "prefill" : "decode",
                   static_cast<unsigned long long>(active.request.id),
                   item.promptOffset, result.consumedPromptTokens,
                   plan.items.size());
      for (uint32_t token : result.outputTokens)
        std::fprintf(trace, " %u", token);
      std::fprintf(trace, " drafted %u accepted %u;", result.draftedTokens,
                   result.acceptedDraftTokens);
      traceContent(trace, model_, active.request.id);
      std::fflush(trace);
    }
    if (plan.kind == WorkKind::Decode && waitsForMask(result.nextDecodeStage)) {
      events_.maskRequested(active.request.id, {});
    }
  }
  scheduler_.complete(plan, schedulerResults, wallMilliseconds,
                      representativePrefillTiming);
  if (std::FILE *trace = decodeReadyTrace())
    tracePlan(trace, "complete", scheduler_, &plan,
              traceMonotonicMilliseconds(), wallMilliseconds, outputTokens);
  events_.batchCompleted(plan.kind, plan.width(), inputTokens, outputTokens,
                         draftedTokens, acceptedDraftTokens, wallMilliseconds);

  for (size_t index = 0; index < results.size(); ++index) {
    const ModelStepResult &result = results[index];
    Request &active = request(result.requestId);
    if (active.failure) {
      Failure failure = std::move(*active.failure);
      active.failure.reset();
      if (failure.code == "cancelled") {
        finish(active, EngineFinishReason::Cancelled, {});
      } else {
        finishFailure(active, std::move(failure));
      }
    } else if (schedulerResults[index].finished) {
      finish(active, result.finished ? EngineFinishReason::Stop
                                     : EngineFinishReason::Length,
             active.scoreLogits);
    }
  }
}

void Engine::finish(Request &active, EngineFinishReason reason,
                    std::span<const float> optionLogits) {
  if (active.finalized)
    return;
  if (reason == EngineFinishReason::Cancelled) {
    scheduler_.cancel(active.request.id);
  }
  active.finalized = true;
  const uint32_t completionTokens =
      active.exactTokens.size() > active.promptTokens
          ? static_cast<uint32_t>(active.exactTokens.size() -
                                  active.promptTokens)
          : 0;
  events_.completed(active.request.id, reason, active.promptTokens,
                    completionTokens, optionLogits);
  if (reason == EngineFinishReason::Cancelled) {
    ++counters_.cancelled;
  } else {
    static_cast<void>(retireCheckpoint(active));
    ++counters_.completed;
  }
  release(active);
}

void Engine::finishFailure(Request &active, Failure failure) {
  if (active.finalized)
    return;
  scheduler_.fail(active.request.id);
  active.finalized = true;
  events_.failed(active.request.id, std::move(failure.code),
                 std::move(failure.message), failure.retryable);
  ++counters_.failed;
  release(active);
}

void Engine::finishCapacity(Request &active, const TokenAdmission &admission) {
  if (admission.allocationFailure != metal::AllocationFailure::None &&
      admission.allocationFailure != metal::AllocationFailure::Capacity) {
    finishFailure(active,
                  {"capacity_exhausted",
                   std::string("could not allocate KV target: ") +
                       metal::allocationFailureName(admission.allocationFailure) +
                       " (additional_pages=" +
                       std::to_string(admission.additionalPages) +
                       ", logical_pages_free=" +
                       std::to_string(admission.availablePages) + ")",
                   true});
    return;
  }
  if (active.finalized)
    return;
  scheduler_.fail(active.request.id);
  active.finalized = true;
  events_.capacityExhausted(active.request.id, admission.additionalPages,
                            admission.availablePages, 0);
  ++counters_.failed;
  release(active);
}

void Engine::release(Request &active) {
  active.resourceWait = {};
  if (active.stateCell || active.suspended) {
    discardPendingStateBoundaries(active);
    model_.end(active.request.id);
    cache_.endRequest(active.request.id);
    active.stateCell.reset();
    active.suspended = false;
    signalResourceProgress();
  }
}

void Engine::sweepTerminal() {
  for (auto iterator = requests_.begin(); iterator != requests_.end();) {
    const uint64_t id = iterator->first;
    Request &active = iterator->second;
    const Phase phase = scheduler_.phase(id);
    if (phase == Phase::Failed && !active.finalized) {
      finishFailure(active, {"deadline_exceeded", "request deadline elapsed"});
    }
    if (!active.finalized) {
      ++iterator;
      continue;
    }
    scheduler_.remove(id);
    iterator = requests_.erase(iterator);
  }
}

Engine::Request &Engine::request(uint64_t id) {
  auto found = requests_.find(id);
  if (found == requests_.end())
    throw std::out_of_range("unknown request");
  return found->second;
}

} // namespace splash::engine
