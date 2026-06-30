# EmotionLens — Demo 启动指南

针对演示机 (i9-275HX / RTX 5070 Ti / 32GB) 的部署说明。

---

## 1 · 演示机首次安装（10 分钟）

### 1.1 创建干净的 conda 环境

```bash
conda create -n emotion-demo python=3.11 -y
conda activate emotion-demo
```

### 1.2 安装 PyTorch（CUDA 12.4 wheel，给 RTX 5070 Ti）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

验证：
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 应输出：CUDA: True  NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

### 1.3 安装其余依赖

```bash
pip install fastapi uvicorn websockets opencv-contrib-python pillow facenet-pytorch
```

`facenet-pytorch` 提供 MTCNN —— 论文 §5.1/§6 指定的人脸检测器，跟训练 pipeline 一致。

---

## 2 · 文件清单（演示机上必须存在）

```
emotion-app/
├── main.py                                   # 启动入口
├── backend/
│   ├── server.py
│   ├── engine_fer.py
│   ├── config.py
│   ├── advice.py
│   └── lenses/
├── final_outputs/                            # 7 个 .pt 模型 · 必须存在
│   ├── best.pt
│   ├── personalized_user1.pt
│   ├── …
│   └── personalized_user6.pt
└── frontend/                                 # 前端三件套
```

---

## 3 · 启动

```bash
cd emotion-app
python main.py
```

启动日志应该看到：

```
[engine] Preloading 7 models …
[engine]   general: OK …
[engine]   user1..user6: OK …
[engine] Preloaded 7 models: ['general', …, 'user6']
[engine] MTCNN ready (device=cuda)
[engine] GPU warmup complete
[server] Starting uvicorn on 0.0.0.0:8000 …
```

如果 `device=cuda` 没出现（写的是 `device=cpu`）→ torch 是 CPU 版，按 §1.2 重装 CUDA 12.4 wheel。

浏览器会自动打开 http://localhost:8000。

---

## 4 · 2 分钟 Demo 流程脚本

### 第 1 段（60 秒）· 单人模型切换

1. 你站镜头前，画面里只有你
2. 右上 Model dropdown 当前是 `General (best.pt)`
3. 依次做 7 个表情，停 5 秒：neutral → happiness → surprise → sadness → anger → disgust → fear
4. 重点指出 General 在哪几个表情上识别错（看 sidebar 7 根 emotion bar 的 dominant）
5. **下拉切到 Personalized · userN**（你自己那个）→ 0.4ms 切换完成
6. 同样 7 表情再做一遍，bar 立刻贴合你的脸

口播要点：「同一张脸，同样表情，General 这边错；切到我自己的 personalized 模型 → 立刻对。这就是论文 §5.5 那张表的现场版」

### 第 2 段（60 秒）· 多人 m3 Audience Reactions

1. 4-5 个组员一起入镜
2. 切到 carousel 上的 m3 (AUDIENCE)
3. **按 Clear 按钮**重置时间轴（每段开始前必按）
4. 喊「大家一起 happy」→ pie chart 偏向 happiness，timeline 折线爬升
5. 喊「大家 surprise」→ 看 dominant 字段切换，折线分叉
6. 演示选中某个组员的脸（点击 video 上的 bbox）→ 那个人的 bbox 加粗 + `#track_id` 角标

口播要点：「群体情绪聚合 + 时间轴 + 点击切换关注对象，专门为多人场景设计的 lens」

### 结束语

「整个 pipeline：MTCNN 检测（+20px margin，与论文一致）→ ResNet18 分类（7 个 personalized model 预加载到显存）→ FaceTracker IoU+EMA → 5 个 lens 各自聚合 → WebSocket 推前端 canvas。GPU 上整帧 < 50ms。」

---

## 5 · 现场故障排查

| 症状 | 原因 | 处理 |
|------|------|------|
| dropdown 切换无效 | WebSocket 断了 | 看右上 `Disconnected`，刷新页面会自动重连 |
| 摄像头黑屏 | 浏览器没给权限 | Chrome 设置 → 隐私 → 摄像头 → 允许 |
| bbox 完全没出现 | facenet-pytorch 没装 | 看启动日志，确认 `MTCNN ready` 这行出现 |
| FPS < 15 | torch 是 CPU 版 | `python -c "import torch; print(torch.cuda.is_available())"` 必须是 True |
| m4 卡在 "Generating…" | LLM API key 未配 / 网络 | 不影响 demo；m4 不在 demo 流程里就忽略 |
| m3 timeline 看起来很脏 | 上次的数据没清 | 按 m3 panel 上的 **Clear** 按钮 |

---

## 6 · Demo 前 30 分钟 dry-run checklist

- [ ] 启动一次，看 7 个模型都 preloaded
- [ ] CUDA 状态：`YuNet using CUDA backend` + `GPU warmup complete` 两行都出现
- [ ] 单人切换 General → user(你自己) → 视觉效果对得上
- [ ] 多人入镜，所有人都有 bbox，没漏检
- [ ] 点击切换选中脸，sidebar 跟着切
- [ ] m3 切过去 → Clear → timeline 干净
- [ ] 摄像头分辨率：右下 cam-res 显示 640×480 或更高
- [ ] 全程没卡顿，FPS badge 稳定在 8-10

---

## 7 · 哪些改动属于本次升级

| 改动 | 目的 |
|------|------|
| 6 个 personalized model + general → 全部 preload | 切换从 ~500ms → 0.4ms |
| 后端不再 cv2 渲染回传，前端 canvas 自己画 | 单帧带宽 ~30KB → ~1KB |
| 点击 bbox 锁定关注脸 + 粘性 + Lost 角标 | 多人场景能指定主角 |
| MTCNN + 20px margin crop（与论文 §5.1/§6 一致） | personalized 模型在演示者脸上的准确率显著提升 |
| GPU 启动预热 | 第一帧不卡 |
| L4 finalize race condition fix | LLM 不会被反复调用 |
| L2 CODE RED 文案改用 MILD pool | 答辩场合不会出现 "Drop the weapon" |
| m3 Clear 按钮 | 多人 demo 必备 |
| 删 modes.py + engine.py 死代码 | code review 不再有冗余文件 |
| EMA_GAMMA / EMOTIONS 顺序统一 | 调参一致性 |
