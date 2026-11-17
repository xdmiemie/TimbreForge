# 上游审计

## 相邻项目

`E:\astrbot-chat` 当前 HEAD 指向 `astrbot-chat`，commit 为 `1c1cce46e0d14efaf1ea415c226ff8c8326d2a2c`。只读 `git status --porcelain` 显示 40 个工作树变更条目，媒体服务和相邻脚本范围有 3 个。审计按磁盘现行代码进行。

- `services.media_api.song_source.NetEaseSource` 提供 `search(query, page=1, *, limit=10)`、`detail(song_id)`、`playable_url(song_id)`、`download(song_id, destination, *, max_bytes=30*1024*1024)`。搜索最多 5 页，每页最多 10 条，返回专辑、别名和通过播放权限判断后的可用标志。
- `netease_auth.resolve_cookie()` 优先使用 `ASTRBOT_CHAT_NETEASE_COOKIE`，其次 `ALYSIA_NETEASE_COOKIE`，最后读取 `ASTRBOT_CHAT_STATE_DIR` 下的 `auth/netease.dpapi`。未设置状态目录时使用仓库的 `storage/qq_companion`。本次检测到 DPAPI 文件存在，环境变量未设置。联网有效性仍待验证。
- 播放器查询使用 standard/mp3，保留上游播放器专有请求头。过滤试听，验证歌曲 ID 和允许的 CDN 主机。CDN 通过独立 `requests.get` 下载，禁用重定向，验证音频 Content-Type 与大小。上游会在下载目录短时保存签名地址用于续传。
- 相邻调用方另用 FFprobe 检查完整时长。TimbreForge 的桥接进程只导入独立模块，无需启动 AstrBot；桥接边界不返回 Cookie 或播放器 URL。
- 文件 SHA256 记录于锁定清单，作为本次实际审计的指纹。

## RIFT 与分离器

- 选定真实 commit `c03fa5940fb8edc32bd43be41eb2655fd8a93deb`。其项目声明 Python 3.11+，Windows 的 Torch/Torchaudio 使用 CUDA 12.8 仓库。[固定源码](https://github.com/Pur1zumu/RIFT-SVC/tree/c03fa5940fb8edc32bd43be41eb2655fd8a93deb)
- 上游 finetune 配置使用 `adamwsf`、256 帧裁剪。训练 Trainer、验证 DataLoader 和完整状态保存需要显式控制。Lightning 模块的验证钩子已调用优化器 eval/train，训练保存钩子还需按固定源码核对。[微调配置](https://github.com/Pur1zumu/RIFT-SVC/blob/c03fa5940fb8edc32bd43be41eb2655fd8a93deb/config/finetune.yaml)
- 1024-16 来自官方底模记录，其公开 SHA256 已锁定。本机尚未下载和加载。[官方底模与校验值](https://huggingface.co/Pur1zumu/RIFT-SVC-pretrained/blob/main/pretrain-v3_dit-1024-16.ckpt)
- BS-RoFormer 权重和 YAML 已从分离器官方 Release 取得并锁定 SHA256。后端版本固定为 `audio-separator 0.47.0`。[分离器模型 Release](https://github.com/nomadkaraoke/python-audio-separator/releases/tag/model-configs)
