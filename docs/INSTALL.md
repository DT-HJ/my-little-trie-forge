# Installation

[한국어](INSTALL.ko.md) · [中文](INSTALL.zh.md) · [README](../README.md)

Reference validation: M5 Max40GPU/128GiB, macOS27.0(26A428). Native minimum: macOS26.4 with required GPU features; Python3.13. Other GPU/RAM configurations are not performance-certified.

From reviewed source, use `make install-environment`, `make -j4 MAX_BATCH_WIDTH=4`, and `make check`. The README serves a locally prepared q8c model and separates model data from installed program files. Default port8000, loopback bind; Ctrl+C drains and exits.

Code packages contain no weights. Read [MODEL_ASSETS.md](../MODEL_ASSETS.md) and explicitly accept model terms before preparation. Fresh source preparation checks110GiB, or65GiB with verified target-blob reuse: preparation disk space, not runtime RAM. This campaign reused an existing verified model and does not claim a fresh network weight download. Raw MLX8bit repositories do not substitute for prepared q8c packages.

## Install and download the model

```sh
brew install hojin12312/mltf/mltf
# Alternatively: pip install https://github.com/hojin12312/my-little-trie-forge/releases/download/v0.1.2/my_little_trie_forge-0.1.2-py3-none-macosx_26_0_arm64.whl
mltf download --model Ho-Jin-93/Qwen3.8-27B-MLTF-q8c --accept-model-licenses
mltf serve --model Ho-Jin-93/Qwen3.8-27B-MLTF-q8c --max-context 256K --max-memory 96G
```

The downloader verifies the manifest and SHA-256. Review the model card and terms before consenting. Source conversion uses `prepare-q8c` from [MODEL_ASSETS.md](../MODEL_ASSETS.md).

### Qwen3.6-35B-A3B (MoE)

The runtime also contains the Qwen3.6-35B-A3B layout (`qwen3_5_moe`: 40 layers, 256 experts, 8 per token, about 3B active parameters) and accepts `splash-packed-q4-moe` packages. It is listed in the official model catalog and downloads through the same command:

```sh
mltf download --model incoai/Qwen3.6-35B-A3B-Splash --accept-model-licenses
mltf serve --model incoai/Qwen3.6-35B-A3B-Splash --max-context 256K --max-memory 96G
```

This package is not produced by `prepare-q8c`; there is no source-conversion tool for it in this repository. It has not been benchmarked or validated on the M3 Max profile here, and the M5-only prefill kernels do not apply on M3.

README256K/96G matches the validated target-machine profile. 32K/48G is a separate conservative profile and the prior soak scope. Signing is ad-hoc; Developer ID/notarization is not included.

## Run at login (macOS)

`mltf service` installs `mltf serve` as a per-user LaunchAgent, so the server starts at login and keeps running without a terminal. The engine needs your graphical session for Metal, so this is a LaunchAgent, not a system daemon.

```sh
mltf service install -- --model Ho-Jin-93/Qwen3.8-27B-MLTF-q8c --max-context 256K --max-memory 96G
mltf service status
mltf service stop        # stops now; it still starts at the next login
mltf service start
mltf service restart
mltf service uninstall   # also removes the login start
```

- Everything after `--` is passed to `mltf serve` unchanged and checked by the same parser, so a mistake fails at install time. The port is always recorded explicitly, because a login service does not see `SPLASH_PORT`. Each port is a separate service. The other actions find the installed service by themselves, so you do not need to remember the port: `mltf service status` lists every installed service, and `start`, `stop`, `restart` and `uninstall` act on the only one, or ask for `--port` when there are several (`--port`, or `SPLASH_PORT`, always wins).
- The service launches `mltf` from `PATH` (`./mltf` in a source checkout) and keeps the path as given, so Homebrew's `bin/mltf` stays valid across upgrades. Choose another executable with `--executable PATH`. As with any running server, stop the service before upgrading.
- A relative `--model-path` is rejected because a login service does not start in your current directory. `MLTF_DATA_ROOT` is carried over; `SPLASH_API_KEY` is not. Pass `--api-key` explicitly if you need one: it is then stored in the plist, which is readable only by you.
- There is no automatic restart after a crash, so a server that fails under memory pressure does not loop. `mltf service start` brings it back.
- Prepare the model first (`mltf download`): a login service has no terminal for model preparation.
- The label is `io.github.hojin12312.mltf.serve-<port>`, the plist is `~/Library/LaunchAgents/<label>.plist`, and output goes to `~/Library/Logs/mltf-<port>.log`, which is not rotated. `mltf service status` prints the state and these paths. A listening port does not mean Ready: check the log.

## Update notification

MLTF performs an automatic update check (notification only; it is not an automatic updater). After the server is Ready, a background daemon thread asks the public GitHub Releases API (`hojin12312/my-little-trie-forge`, latest published, non-draft, non-prerelease release) whether a newer stable version exists. The request is one unauthenticated HTTPS GET with a fixed `User-Agent`; it carries no model name, prompt, path, hostname, hardware or usage data, and no telemetry is sent.

- Non-blocking: the server never waits for GitHub. DNS, TLS, rate-limit (403/429), 404, 5xx, invalid JSON, timeout and offline conditions are all non-fatal, do not change `/status` health, and leave model loading, readiness, the HTTP API and the Web UI unaffected. The state reads `checking` for at most 8 seconds, then `unavailable` with a short `reason`.
- Cached: the result is stored in `update/update-check.json` under the MLTF data directory (source checkouts: `build/runtime/update/`) and reused for 24 hours. Failed attempts back off for one hour. A missing, corrupt, old-schema or future-dated cache is ignored and rewritten atomically. A stale cache is never shown as a fresh result.
- Version comparison is numeric (`0.1.10` is newer than `0.1.9`). An installed version newer than the latest release (a development or local build) is reported as `up_to_date`; no downgrade is suggested. The installed version comes from the packaged `release.json`, or from `pyproject.toml` in a source checkout.
- Only a newer version prints one startup line, for example `A new MLTF version is available: 0.1.3 (installed: 0.1.2). See: <release URL>`. Installations inside a Homebrew Cellar keg name `brew upgrade hojin12312/mltf/mltf`; other installations get only the release link because the installation method cannot be identified reliably. The link is always built from the validated release tag.
- `/status` gains an `update` object with `status` (`unknown`, `checking`, `up_to_date`, `update_available`, `unavailable`, `disabled`), `installed_version`, `latest_version`, `update_available`, `checked_at` (Unix seconds), `release_url` and, when relevant, `reason`. Existing fields are unchanged. The Web UI reads `/status` and shows a small dismissible notice with a "View release" link (new tab) only for `update_available`; the browser never contacts GitHub.
- Opt out with `mltf serve --no-update-check` or `MLTF_NO_UPDATE_CHECK=1` (`1`, `true`, `yes` or `on`). Either one disables the check: no network call is made, the cache is not read, `/status` reports `disabled`, and nothing is printed. Direct `server/server.py` runs check only when started by the launcher with the installed version.

Upgrading remains manual: stop running MLTF servers, then upgrade through the channel you installed from (for Homebrew, `brew upgrade hojin12312/mltf/mltf`; otherwise follow the release page). Model data and caches are kept.


## Memory and SSD cache

`--max-memory` defaults to `auto`: Metal's recommended working set minus `max(1GiB, 2%)`, capped further by a smaller user limit. A larger user value cannot exceed the automatic ceiling. This machine reports a 120GiB recommended working set, giving an automatic ceiling of about 117.6GiB; measurements and the README command explicitly use 96GiB.

The budget covers engine Metal allocations and governor-managed memory. It is not an OS limit on all processes' RAM. Host availability and system memory pressure are also checked; low headroom or critical pressure denies new allocations. The host reserve is `min(10% of physical RAM, 2GiB)`, with another 1GiB margin checked for growth admission. Inspect effective values in `/status` under `memory_plan` and `memory_governor`.

| Option | Default | Controls |
|---|---|---|
| `--max-memory` | `auto` | MLTF memory budget, e.g. `48G` or `96G` (`G` means GiB) |
| `--ssd-root` | unset, SSD cache OFF | Opt-in SSD cache at an absolute private 0700 directory |
| `--ssd-quota-bytes` | when enabled: 1,073,741,824 (1GiB) | Cache quota, positive integer bytes |
| `--ssd-free-floor-bytes` | when enabled: 8,589,934,592 (8GiB) | Filesystem free-space floor, nonnegative integer bytes |
| `--idle-reclaim-percent` | `60` | After 30 seconds with no active request, return empty KV backing and idle state buffers to the OS once host memory use reaches this percent of physical RAM, integer 1–100. Cached prefixes are kept |
| `--gpu-duty-percent` | `100` | Share of wall time the GPU may be busy, integer 1–100. Below 100 the engine pauses after each command for `command time × (100 − N) / N`, so a background workload runs slower and leaves the GPU idle in between. See [Quiet mode](QUIET_MODE.md) |

Default cache disk usage is zero while SSD caching is disabled; model weights/downloads use separate disk space. SSD byte-limit flags require `--ssd-root`. For a 20GiB cache quota with an 8GiB free-space floor:

```sh
mltf serve --model local/Qwen3.8-27B-q8c --model-path "$HOME/mltf-models/q8c" --max-context 256K --max-memory 96G --ssd-root "$HOME/mltf-cache" --ssd-quota-bytes 21474836480 --ssd-free-floor-bytes 8589934592
```

The directory must satisfy the private permission checks. `/status` reports storage, restore and quota state under `storage`.
