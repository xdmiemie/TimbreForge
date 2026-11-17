# 实施验收报告

测试日期 2026-11-16（香港时间）。测试数据位于本机 `data/`。计划中每个编号对应下表。

| 编号 | 结果 | 输入、操作与证据 |
| --- | --- | --- |
| A01、A02、A03、A04 | PASS | `tests/test_datasets.py`。中文空格路径、WAV/FLAC/损坏 MP3/静音/重复，整理前后 SHA256 一致，保留时长 3.5 秒，训练验证按原录音分组。 |
| A05 | PASS | 真实缓存补跑任务 `d8c03a0948c0492f9424d321a68ddcbf` 修复缺失 F0，其他 39 段的 156 个特征文件时间戳不变。 |
| A06 | PASS | RTX 5080 实际 RIFT 1024-16 常规微调，事件记录每次物理 batch 16，默认 LoRA 关闭，BF16；最终累计 optimizer step 8。 |
| A07 | PASS | `tests/test_training_config.py`。15 条训练片段、物理 batch 16 拒绝；16 条得到一个批次。 |
| A08、A09 | PASS | 真实 step 2 完整恢复点保存成功并释放 GPU，完整重启后达到原 step 8 目标。优化器更新计数与进度延续。 |
| A10、A11、A12、A15 | PASS | 训练中 step 1 发布后达到 step 2，独立提前任务 1/4 发布，自然完成 8/8 自动发布。均经 CPU 严格加载后出现在模型库。 |
| A13、A14 | PASS | 实际恢复点导出注入 ENOSPC，未列出半成品，使用同一快照重试只发布一次。`tests/test_checkpoint_index.py` 覆盖损坏索引目标回退、忽略未提交文件。 |
| A16、A17 | PASS | 作者成品读出 dit-512-8 和 speaker「七海」，非 RIFT 夹具被拒。自训 step 1 模型按 64 步、seed 42 转换 5 秒音频，Range 返回 206。 |
| A18 | PASS | 并发提交真实特征任务与模型加载，模型加载在特征任务结束约 63 毫秒后进入 RUNNING。整曲分离进程完成并释放 GPU 后才启动 RIFT。 |
| A19 | BLOCKED | 独立 CUDA 探针使预检可用显存从 14845 MiB 降至 14036 MiB，训练物理 batch 保持 16。Windows WDDM 未枚举探针 PID；显存耗尽情形待实测。 |
| A20、A21 | PASS | 相邻 DPAPI 账号校验，搜索「稻香」并选定歌 ID 1945895725。搜索 196.094 秒，实际下载完整解码 196.0945 秒。 |
| A22、A23 | PASS | `tests/test_music_bridge.py` 注入第一次 CDN 403 后重新解析同一 ID。两个 CDN 请求均无 Cookie。实际本机搜索 API 返回 10 条候选，结果无 Cookie 与 CDN 地址。 |
| A24 | PASS | `tests/test_cover_audio.py` 及实际 190.98 秒整曲，转换音轨在前奏及间奏原坐标为零，输入及输出总样本数一致，立体声伴奏保留。 |
| A25 | PASS | `tests/test_cover_audio.py` 验证伴奏 440 Hz 升高八度，主频 880 Hz，长度逐样本不变，保留立体声。 |
| A26 | PASS | 实际整曲重混音，成品 SHA256 改变，两个处理收据与两条输入轨道时间戳不变。 |
| A27 | PASS | 真实发起下一次搜索，上一首临时播放仍返回 206。 |
| A28、A29 | PASS | 真实无效提交保留旧结果，新任务受理后旧 URL 返回 410。保存的短音频与整曲在服务重启后均返回 Range 206。 |
| A30、A31 | PASS | 线程并发保存和提交下一首，数据库事务串行；注入 Windows 共享冲突后保留待清理状态，重试释放临时目录并保留永久作品。 |
| A32 | PASS | `tests/test_jobs.py`。真实短任务进程，完整 SSE 与断点事件窗口均符合预期。 |
| A33 | PASS | `tests/test_supervisor_recovery.py` 注入 PID 相同而进程创建时间不同的旧任务，标记为 INTERRUPTED；实机暂停后服务重启恢复成功。 |
| A34 | PASS | 网易云《稻香 (女声版)》song ID 1947584923，190.975625 秒，12 段各 64 步，输出 44.1 kHz 立体声 WAV。完整结果 Range 206。硬件指标见下表。 |
| A35 | PASS | 独立 Python 3.12 API、RIFT、分离环境安装完成，安装脚本可用校验缓存重复运行。正常停止、启动后页面返回 HTTP 200，源录音与保存整曲均返回 Range 206。 |

## 在线源音频证据

- 音质 standard/mp3，来源为相邻 Windows DPAPI 会话。结果中只输出账号状态、选定 ID、时长与文件哈希。
- 音频验收 SHA256 为 `c1d8ba1968a2c090d0e3c8e98079d5617178a3d843c00af5f056ccdd0acb78c9`。
- 实机整曲任务 `3505484c612245ff94012f65f9ad44eb` 的选择、下载收据和结果清单都为 song ID `1947584923`。源文件 SHA256 为 `d7733c800f8cba7586df8d83f5eda6d5e7eef0f4c39f7be10fd7fda83231f0e4`，首次混音 SHA256 为 `79544192f3112c37e85e3ef268474aa850b6363eba2fafe0667482e141b760ee`。成品已在本机作品库保存。

## RTX 5080 实测

GPU RTX 5080 16303 MiB，驱动 621.01，Python 3.12.10，PyTorch 2.9.0+cu128，Lightning 2.5.5。GPU峰值是进程的 `torch.cuda.max_memory_allocated`。主存为任务进程树分次采样的 RSS 总和。

| 阶段 | 实际配置 | 峰值 CUDA 分配 | 用时 |
| --- | --- | ---: | ---: |
| 训练 | 1024-16 常规微调，物理 batch 16，BF16 | 10.02 GiB | 重启续训优化器更新至 step 8，细节见任务日志 |
| 验证 | batch 1 | 2.73 GiB | 与训练分开记录 |
| 特征 | 40 段 | 0.78 GiB | 使用真实四类特征 |
| 整曲分离 | BS-RoFormer 317，batch 1，overlap 4，FP32 | 1.58 GiB | 56.625 秒 |
| 整曲 RIFT | FP16，12 段，64 步 | 1.54 GiB | 41.859 秒 |
| 整条翻唱 | 含在线下载、解码、混音、检查 | 采样主存 3.32 GiB | 121.5 秒 |

训练数据来自本机生成的短合成语音。训练、验证、导出和声音转换的工程结果已测，目标角色音质需要用户的角色录音再验收。

## 构建与本地测试

`pnpm run build` 成功生成生产页面，Ruff 检查通过。十项 Python 测试通过。Windows 工具沙箱中 Proactor 回环套接字无法完成初始化，HTTP 测试在正常回环环境通过。测试有一条第三方 Starlette 关于 HTTPX 客户端的弃用提示。
