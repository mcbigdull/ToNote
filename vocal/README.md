# 本地歌曲人声自动扒谱

本项目现位于 `ToNote/vocal/`。以下命令均在此目录执行，共用上一级的 Python 环境与缓存。原输入素材和历史结果仍位于 `../input/`、`../output/`，新运行默认写入本目录的 `output/`。通用 PDF 工具已移至 `../tools/`。

面向 **单主唱 + 乐器伴奏** 的 Python CLI。完整实现：FFmpeg 解码 → Demucs 预训练人声分离 → pYIN/CREPE 连续 F0 → 平滑、稳定音高切分 → 原歌曲节拍检测 → 节奏量化 → 单轨 MIDI、MusicXML、CSV、调试图。音频在本地处理，不上传；首次 Demucs 运行会下载预训练模型。

## Windows 安装

推荐 Python **3.11 x64**，也支持 3.10。请勿用 3.14 安装这套固定版本依赖。PowerShell 中执行：

```powershell
winget install -e --id Python.Python.3.11
# 安装后重新打开终端，进入项目目录
cd D:\myAPP\ToNote\vocal
py -3.11 -m venv ../.venv
..\.venv\Scripts\python.exe -m pip install --upgrade pip
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

无需激活虚拟环境，直接用 `..\.venv\Scripts\python.exe`，避免 PowerShell 执行策略问题。
依赖中的 `imageio-ffmpeg` 提供本地 FFmpeg 可执行程序；程序优先使用 PATH 中的 FFmpeg，否则检测该依赖提供的程序。如果两者都不可用，会明确报错。也可手动安装：

```powershell
winget install -e --id Gyan.FFmpeg
```

无 NVIDIA GPU 时使用 CPU；pYIN 本身只使用 CPU。想启用 NVIDIA 加速，安装与驱动兼容的 CUDA 版 PyTorch，例如以下与本项目固定版本配套的 wheel：

```powershell
..\.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
..\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Demucs、CREPE 自动使用可用 CUDA；`--device cpu` 强制 CPU。Demucs 遇到显存不足会尝试 CPU，CREPE 显存不足则提示重试 CPU。首次安装依赖和模型需要网络及数 GB 磁盘空间；模型缓存在工作区 `../.cache/torch`（尊重已有 `TORCH_HOME`）。网络失败会给出明确错误，重试可复用缓存。

## 运行

```powershell
..\.venv\Scripts\python.exe transcribe_vocal.py "D:/music/song.mp3"
..\.venv\Scripts\python.exe transcribe_vocal.py "../input/莫非定律乐团 - Lidocaine.mp3"
```

完整参数示例（PowerShell 一行即可）：

```powershell
..\.venv\Scripts\python.exe transcribe_vocal.py "song.mp3" --output output --min-note-duration 0.08 --pitch-tolerance-cents 50 --min-confidence 0.5 --time-signature 4/4 --keep-intermediate
..\.venv\Scripts\python.exe transcribe_vocal.py "vocals.wav" --skip-separation --bpm 120 --key "F#m"
..\.venv\Scripts\python.exe transcribe_vocal.py "song.mp3" --pitch-backend crepe --max-note B6
```

pYIN 是默认后端：无需额外 F0 模型，支持默认 C2–C7，长音频按带重叠的 20 秒区块计算以限制内存。CREPE 使用预训练 full 模型及 Viterbi 解码；其模型上限约 2006 Hz，**C7 超过上限**，所以选择 CREPE 时要显式将 `--max-note` 设为 B6 或更低。两种后端的置信度含义不同，阈值需要按录音调整。

## 输出

### 不安装 MuseScore，直接看 PDF

已安装可选 PDF 依赖时，可以将现有 MusicXML 排版为 A4 多页矢量 PDF：

```powershell
..\.venv\Scripts\python.exe -m pip install -r ../tools/requirements-pdf.txt
..\.venv\Scripts\python.exe ../tools/export_score_pdf.py "output/歌曲名/melody.musicxml" --title "Vocal Melody"
```

默认在 MusicXML 旁生成 `melody.pdf` 和可离线浏览的 `melody.html`，也可用 `--output "output/score.pdf"` 指定位置。PDF 可直接用 Edge、Chrome 或 PDF 阅读器打开、缩放、打印，无需 MuseScore。此步骤使用 [Verovio](https://book.verovio.org/) 排版，通过本机 Chrome/Edge 无界面打印为矢量 PDF，保留原谱音符与节奏；需要浏览器安装在标准 Windows 目录。

默认生成 `output/<输入文件名>/`；已有同名目录时另建带随机后缀的目录，避免覆盖历史结果或混入旧文件。

| 文件 | 用途 |
| --- | --- |
| `vocals.wav` | 分离人声；跳过分离时为标准化的原输入 |
| `accompaniment.wav` | 其余分离声部之和；跳过分离时不生成 |
| `pitch_raw.csv` | time、frequency（Hz）、confidence、midi_pitch；无声值可为 NaN |
| `pitch_smoothed.csv` | 门控和平滑后的同格式数据 |
| `notes_raw.csv` | 稳定音高切分后的原始秒级音符 |
| `notes_quantized.csv` | 音高、音名、秒级起止、时长、谱面拍位置、拍时值、置信度 |
| `beats.json` | BPM、节拍来源及节拍网格；可能含音频开始前的外推拍 |
| `melody.mid` | type 0 单轨、Voice Oohs、统一 velocity=85、节拍速度映射 |
| `melody.musicxml` | 调号、拍号、小节、休止符、跨小节连音及 BPM；用 MuseScore 打开 |
| `debug_pitch.png` | 原始 F0、平滑 F0、最终 MIDI 音符块 |
| `run.json` | 配置、执行状态、错误或结果摘要 |

默认保留中间文件。`--no-keep-intermediate` **仅在全部成功后删除两个 WAV**；CSV、节拍、调试图始终保留用于复查。失败时保留已经生成的数据。MusicXML 失败返回非零退出码，但保留有效 MIDI。程序不会将“没有音符”作为成功。

## 量化与记谱约定

所有 `beat` 都是四分音符单位，6/8 一小节为 3 个单位。自动节拍使用原始歌曲，网格前后按局部间隔外推；不是仅根据纯人声猜 BPM。手动 `--bpm 120 --beat-offset 0.2` 可覆盖自动节拍与相位。

十六分音符网格支持全、二分、四分、八分、十六分、附点二分、附点四分、附点八分；复杂时值用连音表示。落在同一位置的音符选置信度较高者并警告，避免重叠或零长度。第一版不自动识别三连音。

`--quantization-strength 0..1` 控制 **MIDI 的秒级演奏时间** 向网格靠拢的强度，默认 1；0 保持原始时序（网格冲突仍会清理）。MusicXML 始终采用完整网格时值，避免弱量化产生难读的任意分数。CSV 的 `start_sec/end_sec` 对应 MIDI，`start_beat/duration_beats` 对应谱面，弱量化时两者有意不同。

MIDI 保留自动检测的局部速度变化与音频绝对起始时间。MusicXML 使用估计的总体 BPM，谱面从首个外推网格拍开始；不记录逐拍微小速度抖动。自动算法不识别强拍、小节起点或弱起，默认按给定拍号划分；可在 MuseScore 校正。自动调号只是单旋律估计，也可 `--key G` / `--key F#m` 覆盖；不会把音高强制吸附到调内。

## 调参

### 八度交叉校验与还原号修复

默认启用 `--octave-check`：在主检测结果之外运行另一种 F0 模型，检查相差一个或两个八度的片段。只有参考模型置信度、候选基频与谐波的频谱证据、持续时间及邻近旋律支持时才修正；不会统一升八度，也不会把低音全部删除。这是保守的自动判断，不能保证区分叠唱中的主唱。

新增 `pitch_reference.csv`、`pitch_corrected.csv` 和 `octave_review.json`，分别保存第二模型、修正后曲线和逐段修正/待复核记录。原始 `pitch_raw.csv` 保持不变。校验按完整稳定音符汇总参考模型覆盖率与八度一致率，避免置信度短暂下降导致一个长音被拆成高低两段。非八度分歧也会记录，但不会自动改成另一条旋律。使用 `--no-octave-check` 可关闭额外检测以缩短耗时。CREPE 的覆盖范围约 50–2006 Hz，超出它音域的主模型结果不应视作已获交叉验证。

已有带两份 F0 曲线、`vocals.wav`、`beats.json` 和 `run.json` 的完整运行目录，可以快速重生成，无需再次运行模型：

```powershell
..\.venv\Scripts\python.exe reprocess_pitch.py "output/已有运行目录" --output "output/新结果目录"
```

目标目录必须尚不存在。重生成结果的 `run.json` 会记录来源运行和人声路径，音频本身不重复复制。

已有同长度、同起点的人声文件时，可加 `--vocals-file "已有目录/vocals.wav"` 避免重复分离；输入仍指定原歌曲，使节拍分析使用完整伴奏。长度差超过 0.1 秒会报错；同长度但不同起点仍需用户保证对齐。

MusicXML 导出现在先按音名建立音符，再由 music21 按调号和小节上下文整理变音记号，最后拆分跨小节连音。普通自然音不再强制显示还原号，取消临时升降号所需的还原号仍保留。

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--min-note / --max-note` | C2 / C7 | F0 搜索范围 |
| `--min-confidence` | 0.5 | 丢弃低置信度帧 |
| `--min-note-duration` | 0.08 秒 | 最短保留音符 |
| `--pitch-tolerance-cents` | 50 | 稳定窗口音高跨度与中心音高变化阈值 |
| `--smooth-ms` | 70 | 中值滤波窗口 |
| `--stable-ms` | 100 | 新音高必须保持稳定的时间 |
| `--max-gap-ms` | 30 | 可合并的同音短间隙 |
| `--silence-db` | -55 dBFS | 静音门控 |

若碎音多，提高稳定时长/平滑窗口；若快音被漏掉，降低稳定时长及最短音符时长。检测失败先听 `vocals.wav`，再看调试图；不要仅靠降低置信度把伴奏残留变成音符。直接相连且没有音高变化或无声间隙的重复同音，第一版可能合并；尚未使用歌词或精细起音检测。

## 验证

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
..\.venv\Scripts\python.exe -m tests.smoke_pipeline
```

单元测试覆盖颤音、瞬时八度异常、滑音、半音变化、无声间隙、低置信度、量化冲突、变速节拍与 MIDI 秒级时间、MusicXML 跨小节连音。合成音频集成测试真正经过 FFmpeg、22050Hz 双声道重采样、pYIN/CREPE、MIDI/XML/PNG，并检查中文/空格路径、保留策略与空白音频失败。

实际歌曲的完整流程验证（默认截取第 45–65 秒，保留结果；不会改动原音频）：

```powershell
..\.venv\Scripts\python.exe -m tests.real_audio_smoke "../input/莫非定律乐团 - Lidocaine.mp3"
```

本机已通过 16 项单元测试，以及该歌曲 20 秒片段的 Demucs → pYIN → 自动节拍 → MIDI/MusicXML 完整流程。该片段输出 37 个音符，估计 117.19 BPM；这是链路验证，不是人工标注的准确率评测。没有对整首歌或真实 CUDA 硬件作验证。详细记录见 [VALIDATION.md](VALIDATION.md)。

## 项目结构与边界

`vocal_transcriber/` 内的 `audio`、`separation`、`pitch`、`slides`、`segmentation`、`beat`、`quantization`、`midi_export`、`musicxml_export`、`visualization` 分别负责各处理阶段，`config` 集中数据结构和参数，`main` 负责 CLI 与流水线。

这是可编辑旋律草稿工具。Demucs 提取的是人声声部，不能保证把主唱与和声分开；合唱、叠唱、rap、自由速度、气声和大幅滑音可能不可靠。八度误判和复杂节奏仍需人工校正，不承诺直接生成出版级乐谱。

实现参考：[Demucs](https://github.com/facebookresearch/demucs)、[torchcrepe](https://github.com/maxrmorrison/torchcrepe)、[librosa pYIN](https://librosa.org/doc/0.10.2/generated/librosa.pyin.html)、[music21](https://www.music21.org/music21docs/)。Demucs 上游已归档，因此本项目固定依赖版本。
