# 安静模式（外部控制器 + 启动参数）

安静模式 = 在后台以较低资源运行不急于出结果的长期任务；用户有其他高负载工作时立即让出资源，负载结束后再恢复。

MLTF 本身**不监测负载、不管理任务**。它只提供“按参数运行”的能力，由外部控制器检测整机负载并切换参数。任务进度（哪些请求已完成）由调用方保存，MLTF 不做断点。

## 可用于限制资源的参数

这些参数都在**启动时**设定，运行中不能改。

| 参数 | 限制的资源 | 安静模式建议值 |
|---|---|---|
| `--gpu-duty-percent N` | GPU 占空比（1–100）。每条命令结束后暂停 `命令耗时 × (100−N)/N` | `20`–`30` |
| `--max-concurrent-requests` | 并发请求数（1–4） | `1` |
| `--max-context` | 最大上下文，决定 KV 占用 | 按任务需要缩小 |
| `--max-memory` | 引擎内存上限（不能低于权重，约 31 GiB） | 视情况 |
| `--idle-reclaim-percent` | 空闲 30 秒后，整机内存占用达到该比例就释放空 KV | `30` |

`--gpu-duty-percent` 是为安静模式新增的：nice/QoS 对 Metal 计算基本无效，只有让引擎自己停顿，GPU 才会真正降速。每条命令之后的暂停是 `命令耗时 × (100−N)/N`，但**单次暂停上限 250 ms**：实测发现，暂停过长时 GPU 会降频，下一条命令耗时被放大约 9 倍，公式又据此算出更长的暂停，形成恶性循环（N=25 时 256 个 token 用了约 4 分钟，仅 0.9 tok/s）。加上限后 decode 命令耗时恢复到约 95 ms。所以 decode 阶段的有效占空比下限约 28%（95 ms 命令配 250 ms 暂停），N 小于 28 时不会再慢下去。

## 控制器的做法（切换即重启；不想重启请用下面的 `/admin/profile`）

1. 控制器定期读取整机 CPU、GPU（`ioreg` 的 AGXAccelerator `Device Utilization %`）和内存压力（`memory_pressure -Q`）。
2. **进入高负载**：停止 MLTF 服务（`mltf service stop` 或 `launchctl bootout`），约 30 GiB 权重和 GPU 全部释放。
3. **负载低于阈值并持续 N 秒**（带滞回，避免抖动）：用安静参数启动服务，例如
   `mltf serve --gpu-duty-percent 25 --max-concurrent-requests 1 --idle-reclaim-percent 30`，
   然后让客户端继续发送未完成的请求（请求可用 `priority: "background"`）。
4. 被打断的请求在重启后由客户端重发；已完成的由客户端跳过。
5. 判定时要排除 MLTF 自己造成的 GPU/CPU 占用，否则安静模式会把自己当成高负载。例如只在服务停止或请求间隙采样，或使用进程级 CPU 占用。

代价：每次切换都要重新加载模型（数十秒），内存中的前缀缓存丢失（配置 `--ssd-root` 可保留部分）。

## 运行时调节：`/admin/profile`

不重启、不重新加载模型，随时改资源占用。外部应用（测温度、内存占用的脚本）直接调用：

```sh
# 查看：requested 是你设过的值，applied 是引擎实际生效的值
curl -s http://127.0.0.1:8000/admin/profile

# 限速到 30%，同时把请求并发降到 1
curl -s -X POST http://127.0.0.1:8000/admin/profile \
  -d '{"gpu_duty_percent": 30, "max_concurrent_requests": 1}'

# 暂停派发新命令（进行中的命令跑完，请求不会被取消）；再恢复
curl -s -X POST http://127.0.0.1:8000/admin/profile -d '{"paused": true}'
curl -s -X POST http://127.0.0.1:8000/admin/profile -d '{"paused": false}'

# 下一条请求起，内存限额改为 48G；null 或 "auto" 回到启动上限
curl -s -X POST http://127.0.0.1:8000/admin/profile -d '{"memory_limit": "48G"}'

# 立即释放空的 KV 内存
curl -s -X POST http://127.0.0.1:8000/admin/profile -d '{"reclaim_now": true}'
```

| 设置 | 取值 | 生效时机 |
|---|---|---|
| `gpu_duty_percent` | 1–100 | 立即 |
| `paused` | true / false | 立即；暂停期间请求照常排队，超时仍按 `--request-timeout` 计 |
| `max_concurrent_requests` | 1–4 | 立即；已在跑的请求不受影响，需要启动时带了 `--max-concurrent-requests`，否则返回 409 |
| `memory_limit` | 字节数、`"48G"` 这样的大小、`null` | **所有请求结束、引擎空闲时**才生效，下一条请求使用新值；降低时会同时驱逐超出限额的缓存前缀 |
| `reclaim_now` | `true` | 同样等引擎空闲后执行一次 |

约束：

- `memory_limit` 只能在“权重所需下限～启动时 `--max-memory`”之间调。启动值是上限（内存池按它规划），调高超过它会被拒绝（400），需要重启。想要最大灵活度，启动值就设成能接受的最大值。
- 并发大于 1 时，“空闲”指所有请求都结束；持续有请求时，`memory_limit` 和 `reclaim_now` 会一直排队。`applied.memory_limit_pending_bytes` 能看到排队中的值。
- `memory_limit` 设得比权重加一个请求所需的 KV 还小时，请求会等内存直到超时失败。接口不会替你判断这个下限，请用 `/status` 里的内存数据确认。
- 运行中改的值在重启后失效，回到启动参数。
- 接口走 HTTP：设置了 `--api-key` 时需要 key；没设置时只接受本机访问。所有数值都会校验，非法值返回 400，不会改变任何状态。
- 引擎侧通过服务进程创建的私有临时目录交换设置（目录权限 0700，退出时删除），不新增原生协议帧。

## 实测（M3 Max 128 GB，Qwen3.8-27B q8c，上下文 8K，并发 1）

同一请求（256 token，temperature 0），生成速度取 3 次中位数：

| `--gpu-duty-percent` | 生成速度 | 相对 100% |
|---:|---:|---:|
| 100（不限速） | 26.0 tok/s | 100% |
| 50 | 15.4 tok/s | 59% |
| 25（带 250 ms 暂停上限） | 8.7 tok/s | 33% |

注意：`ioreg` 的 `Device Utilization %` 在三档下都在 88–99%，不能用来衡量限速效果，GPU 功耗和温度未测量（需要 `powermetrics`，要 sudo）。已验证的只有“同一任务变慢、GPU 有空闲间隙”，不能据此宣称省电。

## 验证

- `python3 -m unittest tests.test_gpu_duty_percent`：参数解析和透传。
- 实测：同一请求在 `--gpu-duty-percent` 100/50/25 下比较生成速度和 GPU 利用率，见 PR 描述。
