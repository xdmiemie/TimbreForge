# 音色工坊 · TimbreForge

本机 AI 歌声转换工作台。**微调训练**整理角色录音、准备特征并运行 RIFT-SVC V3 1024-16 常规微调，默认物理 batch 为 16。**歌曲翻唱**选择已验证的模型，通过同级 `astrbot-chat` 的网易云 WeAPI 选定歌曲版本，或读取本地歌曲。翻唱与效果试听默认使用 64 个推理采样步。

## Windows 启动

安装完成后，双击仓库根目录的 `启动TimbreForge.cmd`。服务就绪后会自动在默认浏览器打开 [本地 WebUI](http://127.0.0.1:7866/)。启动窗口显示服务日志。再次双击会打开当前运行的服务。

首次安装需要 Python 3.11 或 3.12、Node.js、pnpm、uv、FFmpeg/FFprobe 和支持 CUDA 12.8 的 NVIDIA 驱动。在 PowerShell 中运行以下命令，将示例的 Python 路径改为本机路径。

```powershell
.\scripts\assets.ps1
.\scripts\setup.ps1 -Python 'C:\Python312\python.exe' -RiftPython 'C:\Python312\python.exe' -Rift -Separator
.\scripts\doctor.ps1
```

也可在 PowerShell 中运行 `.\scripts\start.ps1` 手动启动服务。停止服务可从另一个终端运行 `.\scripts\stop.ps1`。训练活跃时先在页面暂停并等待恢复点保存。

本机目标 GPU 为 RTX 5080 16GB。资产和验收结果见 [锁定清单](configs/upstreams.lock.json)、[实施进度](docs/CODEX_PROGRESS.md) 与 [验收报告](docs/ACCEPTANCE_REPORT.md)。角色效果取决于录音和实际训练。项目的数据和模型留在本机。

[使用手册](docs/USER_GUIDE.md) · [故障排查](docs/TROUBLESHOOTING.md) · [架构](docs/ARCHITECTURE.md) · [上游审计](docs/UPSTREAM_AUDIT.md)
