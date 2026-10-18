# 人声分离的 note / contour 热图对比

入口 `compare_separation.py` 放在 `vocal/` 第一级。默认输入仍是外层
`input/莫非定律乐团 - Lidocaine.mp3`，所有新结果都保存在 `vocal/output/`。
既有主旋律扒谱仍使用 pYIN/CREPE；这个工具额外调用与小提琴项目相同的
Basic Pitch 0.4.0，专门用来观察分离前后的音高响应，不改变现有扒谱算法。

## 已有环境下运行

在 ToNote 根目录执行（从其他目录运行也不会改变默认输入/输出位置）：

```powershell
.\.venv\Scripts\python.exe vocal/compare_separation.py
```

默认结果在 `vocal/output/莫非定律乐团 - Lidocaine_separation_compare/`，
打开 `index.html` 可以播放三路音频、查看整曲连续热图，并随播放位置自动横向滚动。
动态页前三行依次是原曲、HTDemucs 人声、BS-RoFormer 人声的 note，后三行是相同顺序的 contour，方便同类热图直接对比。
点击热图跳转时间；播放另一条音频时沿用当前时间并暂停上一条。可调整横向缩放、只显示 note/contour、关闭跟随自由浏览。图的时间都是原曲时间。
旧 segments 文件保留，但页面不再展示，新渲染不生成分段图；静态整曲对照图仍可单独打开。

已经生成部分结果或希望重画图时：

```powershell
.\.venv\Scripts\python.exe vocal/compare_separation.py --resume
.\.venv\Scripts\python.exe vocal/compare_separation.py --resume --stage render
```

其他歌曲：

```powershell
.\.venv\Scripts\python.exe vocal/compare_separation.py "input/其他歌曲.mp3"
.\.venv\Scripts\python.exe vocal/compare_separation.py "input/其他歌曲.mp3" --demucs-vocals "vocal/output/其他歌曲/vocals.wav"
```

显式复用的人声必须来自同一首完整音频，不允许裁切或移动起点。脚本核对采样率、采样数和声道数；
不能仅靠这些信息证明声音内容相同。Lidocaine 会自动寻找
`vocal/output/revised/莫非定律乐团 - Lidocaine/vocals.wav` 并检查历史输入路径。
历史记录没有原始 MP3 哈希，所以复用来源会明确记入报告。找不到可复用文件时才重新跑 HTDemucs。
已有结果默认拒绝覆盖；`--resume` 只允许同一输入文件哈希及模型设置继续。
显式设置过 `--demucs-vocals` 的任务，继续时要带相同参数。

## 环境

三个子进程环境分别负责：

- `../.venv`：已有 FFmpeg 解码和 HTDemucs。
- `../violin/.venv`：已有 Basic Pitch、TensorFlow 和 Matplotlib。
- `.venv-separator`：新增 audio-separator、BS-RoFormer。

隔离是因为新 audio-separator 使用 NumPy 2，而现有 TensorFlow 环境固定 NumPy 1.26。
初次准备在 ToNote 根目录运行：

```powershell
.\.tools\bin\uv.exe venv vocal/.venv-separator --python .python/cpython-3.11-windows-x86_64-none/python.exe
.\.tools\bin\uv.exe pip install --python vocal/.venv-separator/Scripts/python.exe -r vocal/requirements-separator.txt
```

也可以用系统 Python 3.11 的 `python -m venv` 和新环境的 `python -m pip install`。
环境位置可以通过 `--vocal-python`、`--pitch-python`、`--separator-python` 显式指定。
BS 权重与配套配置下载到 `vocal/.cache/audio-separator/`；FFmpeg 使用已安装的 imageio-ffmpeg，
不要求系统另装。默认最多使用 12 个 CPU 计算线程，可通过 `--threads` 调整。
本机 Core Ultra 5 225H 上实测：4 线程约 75 秒/分块，12 线程约 39 秒/分块；
这里只是本曲、本模型的运行耗时，不代表其他电脑或音频的表现。

## 文件与解释

```text
<歌曲>_separation_compare/
├── index.html                 离线查看入口
├── comparison_full.png        三路整曲对照
├── segments/                  三路按相同时间窗放大对照
├── original_mix/
├── demucs/
└── bs_roformer/
    每路包含 audio.wav、pitch_debug.png、model_outputs.npz、
    notes_raw.json、basic_pitch_raw.mid、analysis.json
```

- note：88 个半音格，A0–C8；contour：264 个格，每半音 3 格。
- 三路使用同一 Basic Pitch 模型、相同参数和固定 0–1 色标，完整保留低音、高音及同时发声。
- 热图是模型激活值，不是分贝频谱，也不是校准后的概率或分离准确率。
- note/contour 热图不受音符事件的最短时长门限裁切。下方音符块使用与小提琴默认相同的
  onset=0.5、frame=0.3、minimum_note_length=127.7 ms、multiple_pitch_bends=True；
  很短的热图亮点可能没有对应音符块。未套用小提琴音域筛选，也未做八度纠错或节拍量化。
- contour 原始输出和第三幅图上的离散弯音，不是高精度连续 F0。
- 整曲图仅在显示时按时间取最大值压缩；完整每帧矩阵保留在 NPZ，15 秒放大图不合并帧。
- 保持 Basic Pitch 0.4.0 的模型、音频窗口及重叠裁边方式，按每个窗口的实际采样起点记录时间。
  上游默认用整数 86 FPS 截断拼接结果，会在这首约 301 秒的歌曲末尾少保留约 2 秒；
  本工具保留所有有效窗口，直到原曲末尾。NPZ 另外保留上游时间标签用于复查。
  `notes_raw.json` 同时保存上游和映射后的起止时间，图采用映射后的原曲时间。
  附带的 `basic_pitch_raw.mid` 保留上游时间，仅供诊断；本任务不生成正式修订乐谱。
- 原曲统一解码为 44.1 kHz 双声道浮点 WAV。Basic Pitch 按其默认方式转单声道/22.05 kHz。
  不裁切、不移动时间，不对三路分别做峰值/响度匹配；BS 包设 amplification=0、normalization=1，
  如果输入或输出峰值超出 1，包仍可能做限幅前的归一化。这首 MP3 解码后峰值约 1.127，
  因此 BS 包会先降低约 1 dB 的输入增益；这属于实际分离流程，不做额外响度匹配补偿。
  颜色差异不能直接解释为声音响度差异。
- BS 使用 `model_bs_roformer_ep_317_sdr_12.9755.ckpt`，模型默认 8 秒窗口，overlap=2、batch_size=1，
  不使用半精度、音高移调或集成。模型版本、权重 SHA256、配置和耗时写入 `separation.json`。
- `validation.json` 检查三路时间网格、矩阵形状和音频时长一致；不会虚构准确率。
- 本轮不替换正式扒谱的分离器，不重新生成乐谱或 PDF。
- 仅刷新连续播放页（复用缓存，不分离、不识别）：`violin/.venv/Scripts/python.exe vocal/vocal_transcriber/heatmap_player.py "vocal/output/莫非定律乐团 - Lidocaine_separation_compare"`。页面需要同目录的 `player_assets` 和各音频文件夹。
- BS 人声另存为 `bs_roformer/vocals.mp3`（320 kbps），可在对照页直接试听、下载；无损 `audio.wav` 保留，热图仍使用 WAV。恢复运行会复用已有分离结果并补齐 MP3。
