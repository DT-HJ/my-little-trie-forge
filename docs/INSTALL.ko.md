# 설치

[English](INSTALL.md) · [中文](INSTALL.zh.md) · [README](../README.ko.md)

기준 검증 기기는 M5 Max GPU40코어/128GiB, macOS27.0(26A428)입니다. Native 최소 환경은 필요한 GPU 기능을 갖춘 macOS26.4이며 Python3.13을 사용합니다. 다른 GPU/RAM 구성의 성능을 인증하지 않습니다.

검토 소스에서 `make install-environment`, `make -j4 MAX_BATCH_WIDTH=4`, `make check`를 사용합니다. 로컬 prepared q8c 모델로 서버를 실행하는 README 명령은 모델 데이터와 프로그램 설치를 분리합니다. 기본 포트는8000, bind는 loopback이며 Ctrl+C로 정상 종료합니다.

코드 패키지에는 가중치가 없습니다. 모델 준비는 [MODEL_ASSETS.md](../MODEL_ASSETS.md)의 이용 조건에 명시적으로 동의한 뒤 실행합니다. Fresh source 준비의110GiB 또는 검증된 target blob reuse의65GiB 조건은 준비 중 디스크 공간입니다. Runtime RAM 요구량이 아닙니다. 이번 측정에서는 기존 검증 모델을 재사용했으며 새 network weight 다운로드로 표현하지 않습니다. Raw MLX8bit repository는 prepared q8c의 대체물이 아닙니다.

## 설치와 모델 다운로드

```sh
brew install hojin12312/mltf/mltf
# 또는: pip install https://github.com/hojin12312/my-little-trie-forge/releases/download/v0.1.2/my_little_trie_forge-0.1.2-py3-none-macosx_26_0_arm64.whl
mltf download --model Ho-Jin-93/Qwen3.8-27B-MLTF-q8c --accept-model-licenses
mltf serve --model Ho-Jin-93/Qwen3.8-27B-MLTF-q8c --max-context 256K --max-memory 96G
```

모델 다운로드는 manifest와 SHA-256을 검증합니다. 모델 카드를 먼저 읽고 이용 조건에 동의하세요. 직접 변환은 [MODEL_ASSETS.md](../MODEL_ASSETS.md)의 `prepare-q8c`를 사용합니다.

README의 256K/96G는 검증한 목표 머신의 프로필입니다. 32K/48G는 별도의 보수적인 프로필이며 이전 soak 검증 범위입니다. 서명은 ad-hoc이며 Developer ID와 notarization은 포함하지 않습니다.

## 로그인 시 자동 실행 (macOS)

`mltf service`는 `mltf serve`를 사용자별 LaunchAgent로 등록합니다. 터미널을 열어 두지 않아도 로그인할 때 서버가 시작되어 계속 실행됩니다. 엔진이 Metal을 쓰려면 그래픽 로그인 세션이 필요하므로 시스템 데몬이 아니라 LaunchAgent를 사용합니다.

```sh
mltf service install -- --model Ho-Jin-93/Qwen3.8-27B-MLTF-q8c --max-context 256K --max-memory 96G
mltf service status
mltf service stop        # 지금 중지합니다. 다음 로그인 때는 다시 시작됩니다
mltf service start
mltf service restart
mltf service uninstall   # 로그인 시 시작도 함께 제거합니다
```

- `--` 뒤의 인자는 `mltf serve`에 그대로 전달되고 같은 파서로 검증되므로, 잘못된 인자는 설치 시점에 오류가 납니다. 로그인 서비스는 `SPLASH_PORT`를 볼 수 없으므로 포트는 항상 명시적으로 기록됩니다. 포트마다 별도의 서비스입니다. 나머지 동작은 설치된 서비스를 스스로 찾으므로 포트를 기억할 필요가 없습니다. `mltf service status`는 설치된 모든 서비스를 나열하고, `start`, `stop`, `restart`, `uninstall`은 하나뿐이면 그 서비스를 대상으로 하며 여러 개이면 `--port`를 요구합니다(`--port`나 `SPLASH_PORT`가 항상 우선합니다).
- 서비스는 `PATH`의 `mltf`(소스 checkout에서는 `./mltf`)를 실행하며 경로를 준 그대로 유지합니다. 그래서 Homebrew의 `bin/mltf`는 업그레이드 후에도 유효합니다. 다른 실행 파일은 `--executable PATH`로 지정합니다. 실행 중인 다른 서버와 마찬가지로 업그레이드 전에 서비스를 중지하세요.
- 로그인 서비스는 현재 디렉터리에서 시작하지 않으므로 상대 경로 `--model-path`는 거부됩니다. `MLTF_DATA_ROOT`는 전달되지만 `SPLASH_API_KEY`는 전달되지 않습니다. API key가 필요하면 `--api-key`를 직접 지정하세요. 이 경우 plist에 저장되며 본인만 읽을 수 있습니다.
- 충돌 후 자동 재시작은 없으므로, 메모리 압박으로 실패하는 서버가 반복해서 재시작되지 않습니다. `mltf service start`로 다시 시작합니다.
- 모델은 먼저 준비하세요(`mltf download`). 로그인 서비스에는 모델 준비 과정을 보여 줄 터미널이 없습니다.
- 라벨은 `io.github.hojin12312.mltf.serve-<port>`, plist는 `~/Library/LaunchAgents/<label>.plist`, 출력은 `~/Library/Logs/mltf-<port>.log`이며 로그는 순환(rotate)되지 않습니다. `mltf service status`가 상태와 이 경로를 출력합니다. 포트가 열려 있다고 Ready인 것은 아닙니다. 로그를 확인하세요.

## 업데이트 알림

MLTF는 자동 업데이트 확인(알림 전용, 자동 업데이터 아님)을 수행합니다. 서버가 Ready가 된 뒤 백그라운드 데몬 스레드가 공개 GitHub Releases API(`hojin12312/my-little-trie-forge`의 게시된 최신 non-draft, non-prerelease 릴리스)에서 더 새로운 안정 버전이 있는지 확인합니다. 요청은 고정된 `User-Agent`를 사용하는 인증 없는 HTTPS GET 한 번이며 모델 이름, 프롬프트, 경로, 호스트명, 하드웨어, 사용 데이터를 포함하지 않고 텔레메트리도 보내지 않습니다.

- 비차단: 서버는 GitHub를 기다리지 않습니다. DNS, TLS, rate limit(403/429), 404, 5xx, 잘못된 JSON, timeout, 오프라인은 모두 치명적이지 않으며 `/status` 건강 상태를 바꾸지 않고 모델 로드, readiness, HTTP API, Web UI에 영향을 주지 않습니다. 상태는 최대 8초 동안 `checking`이다가 짧은 `reason`과 함께 `unavailable`이 됩니다.
- 캐시: 결과는 MLTF 데이터 디렉터리의 `update/update-check.json`(소스 체크아웃은 `build/runtime/update/`)에 저장되어 24시간 재사용됩니다. 실패한 시도는 1시간 동안 재시도하지 않습니다. 없거나 손상됐거나 이전 스키마이거나 미래 시각인 캐시는 무시하고 원자적으로 다시 씁니다. 오래된 캐시를 방금 확인한 결과처럼 표시하지 않습니다.
- 버전 비교는 숫자 기준입니다(`0.1.10`은 `0.1.9`보다 새 버전). 설치 버전이 최신 릴리스보다 높으면(개발·로컬 빌드) `up_to_date`로 보고하며 다운그레이드를 안내하지 않습니다. 설치 버전은 패키지의 `release.json`, 소스 체크아웃에서는 `pyproject.toml`에서 읽습니다.
- 더 새로운 버전이 있을 때만 시작 시 한 줄을 출력합니다. 예: `A new MLTF version is available: 0.1.3 (installed: 0.1.2). See: <release URL>`. Homebrew Cellar keg 안의 설치는 `brew upgrade hojin12312/mltf/mltf`를 안내하고, 그 밖의 설치는 설치 방식을 확실히 식별할 수 없으므로 릴리스 링크만 안내합니다. 링크는 항상 검증된 릴리스 태그로 직접 생성합니다.
- `/status`에 `update` 객체가 추가됩니다: `status`(`unknown`, `checking`, `up_to_date`, `update_available`, `unavailable`, `disabled`), `installed_version`, `latest_version`, `update_available`, `checked_at`(Unix 초), `release_url`, 필요하면 `reason`. 기존 필드는 바뀌지 않습니다. Web UI는 `/status`를 읽어 `update_available`일 때만 닫을 수 있는 작은 알림과 "View release" 링크(새 탭)를 표시하며 브라우저가 GitHub에 직접 접속하지 않습니다.
- `mltf serve --no-update-check` 또는 `MLTF_NO_UPDATE_CHECK=1`(`1`, `true`, `yes`, `on`)로 끌 수 있습니다. 둘 중 하나만 있어도 확인하지 않으며, 네트워크 호출이 없고 캐시를 읽지 않으며 `/status`는 `disabled`를 보고하고 아무것도 출력하지 않습니다. `server/server.py`를 직접 실행하면 런처가 설치 버전을 전달한 경우에만 확인합니다.

업그레이드는 계속 수동입니다. 실행 중인 MLTF 서버를 중지한 뒤 설치한 경로로 업그레이드합니다(Homebrew는 `brew upgrade hojin12312/mltf/mltf`, 그 외에는 릴리스 페이지를 따릅니다). 모델 데이터와 캐시는 유지됩니다.


## 메모리와 SSD 캐시

`--max-memory` 기본값은 `auto`입니다. Metal 권장 working set에서 `max(1GiB, 2%)`의 여유를 빼고, 사용자가 지정한 상한이 더 작으면 그 값을 사용합니다. 더 큰 값을 지정해도 이 자동 상한을 넘지 않습니다. 이번 머신이 보고한 권장 working set은 120GiB이므로 자동 상한 계산값은 약 117.6GiB이며, 실제 측정과 README 명령은 96GiB를 명시합니다.

이 예산은 MLTF 엔진의 Metal 할당과 메모리 관리자가 추적하는 메모리에 적용됩니다. 시스템 전체 프로세스의 RAM 사용량을 강제로 제한하는 OS 상한은 아닙니다. 호스트 가용 메모리와 시스템 메모리 압력도 검사하며, 가용 공간이 부족하거나 메모리 압력이 critical 상태이면 새 할당을 거부합니다. 시스템에 남겨 두는 여유 메모리는 `min(물리 RAM의 10%, 2GiB)`이며, 추가 메모리를 할당하기 전에 1GiB의 여유도 확인합니다. 실제 값과 승인 상태는 `/status`의 `memory_plan` 및 `memory_governor`에서 확인할 수 있습니다.

| 옵션 | 기본값 | 제어 범위 |
|---|---|---|
| `--max-memory` | `auto` | MLTF 메모리 예산. 예: `48G`, `96G` (`G`는 GiB) |
| `--ssd-root` | 미지정, SSD 캐시 OFF | SSD 캐시를 켤 때 지정하는 절대 경로. 권한 0700의 전용 디렉토리 |
| `--ssd-quota-bytes` | 켠 경우 1,073,741,824 (1GiB) | SSD 캐시 용량 한도. 양의 정수(바이트) |
| `--ssd-free-floor-bytes` | 켠 경우 8,589,934,592 (8GiB) | 남겨 둘 파일시스템 여유 공간. 0 이상의 정수(바이트) |
| `--idle-reclaim-percent` | `60` | 처리 중인 요청 없이 30초가 지나고 호스트 메모리 사용량이 물리 RAM의 이 비율 이상이면, 비어 있는 KV 백킹과 유휴 상태 버퍼를 OS에 반환합니다. 1–100 정수. 캐시된 prefix는 유지됩니다 |
| `--gpu-duty-percent` | `100` | GPU가 바쁠 수 있는 실제 시간 비율, 1–100 정수. 100 미만이면 엔진이 명령마다 `명령 시간 × (100 − N) / N`만큼 쉬어 백그라운드 작업을 느리게 실행합니다. [Quiet mode](QUIET_MODE.md) 참고 |

SSD 캐시를 사용하지 않는 기본 실행의 캐시 디스크 사용량은 0입니다. 모델 가중치·다운로드 파일은 별도의 디스크 사용량입니다. SSD 한도 옵션에는 `--ssd-root`가 필요합니다. 예를 들어 캐시 한도 20GiB, 여유 공간 8GiB로 지정합니다:

```sh
mltf serve --model local/Qwen3.8-27B-q8c --model-path "$HOME/mltf-models/q8c" --max-context 256K --max-memory 96G --ssd-root "$HOME/mltf-cache" --ssd-quota-bytes 21474836480 --ssd-free-floor-bytes 8589934592
```

이 디렉토리는 권한 0700을 사용해야 합니다. 캐시 저장·복원과 용량의 상세 상태는 `/status`의 `storage`에서 확인하세요.
