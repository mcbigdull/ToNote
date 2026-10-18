# Basic Pitch 小提琴转 MIDI MVP

## 音符修整（2026-10-18）

已有 `output/model_outputs.npz` 时，在 `violin/` 中运行下面的命令，不需要再次跑模型：

```powershell
.\.venv\Scripts\python.exe refine_violin.py "input/天鹅-小提琴.mp3"
```

新结果放在 `output_refined/`；目录已有结果时自动创建带后缀的目录，不覆盖旧结果。优先试听 `violin_refined_plain.mid`，并查看 `refinement_comparison.png` 和 `refinement_110_end.png`。

新音频可以一次执行原始转录和修整：

```powershell
.\.venv\Scripts\python.exe transcribe_violin.py "input/新录音.wav" --refine --output output_new
```

原始结果保存在 `output_new/`，修整结果另放 `output_new/refined/`。修整默认起音阈值0.7、持续阈值0.3、最短127.7毫秒、Melodia开启。`librosa` 从原音频提取经过频率方向最大值滤波的起音强度（减轻揉弦影响），用音频起音作为“保留再次起音”的辅助证据。

仅尝试合并同音高、不重叠、间隙最多40毫秒且边界处模型激活持续存在的事件；若60毫秒内有音频起音候选，或模型起音强度达到0.9，则不合并。同音重叠、不同音高和缺乏持续激活的间隙均保留。合并后的 expressive MIDI 使用各源音符原有的弯音时间戳，不把弯音拉伸到合并后的整段。

参数集中在 `violin_transcriber/config.py` 的 `RefinementConfig`。例如：

```powershell
.\.venv\Scripts\python.exe refine_violin.py "input/天鹅-小提琴.mp3" --onset-threshold 0.7 --merge-gap-ms 40 --onset-tolerance-ms 60 --strong-model-onset 0.9 --audio-onset-delta 0.12 --no-melodia-trick --output output_refined_trial
```

`--melodia-trick` / `--no-melodia-trick` 同时适用于原始转录入口；在修整入口会另外记录0.5/当前阈值、Melodia开/关的音符数对照。默认继续开启，因为它也可能找回柔和起音。

| 输出 | 用途 |
| --- | --- |
| `violin_decoded_plain.mid` / `violin_decoded_expressive.mid` | 调参数后、合并前 |
| `violin_refined_plain.mid` / `violin_refined_expressive.mid` | 保守合并后 |
| `notes_baseline.csv` / `notes_decoded.csv` / `notes_refined.csv` | 默认解码、调整解码、修整三个阶段；ID在不同阶段间不能直接对应 |
| `merge_decisions.json` | 同音相邻事件的合并或保留原因、模型/音频证据 |
| `pitch_bends_refined.json` | 源ID（对应decoded CSV）、原始弯音bin及不经拉伸的秒时间 |
| `audio_onsets.npz` | 原音频的归一化起音强度及起音候选秒数 |
| `suspected_errors_refined.csv` | 疑似谐波及重叠等候选；不自动删音 |
| `refinement_report.json` / `report.txt` | 参数、输入哈希、事件数、MIDI回读验证和限制 |

《天鹅》实测为417 → 165 → 144个事件，21个边界合并。116.7秒附近B4由13次变为1次，末尾G5由25次变为1次；这不能证明整曲准确率。不同音高不参与合并，但提高解码阈值仍可能漏掉真实音符，弱起音也可能被误合并，必须回听。模型帧时间沿用原缓存，尚未修正潜在的模型/音频时间差。

本阶段仍没有节奏量化或MusicXML，不把谱面当作修整算法的输入，也不自动删除谐波候选。

面向已分离、相对干净的单把小提琴音频。使用 Spotify Basic Pitch 0.4.0 预训练模型，保留复音和原始弯音。不做人声分离、单音化、节奏量化或技法分类，不接入 VioPTT。

## 本机运行

已在 `violin/.venv` 安装 Python 3.11 对应的独立依赖，不改动人声项目环境。PowerShell：

```powershell
cd D:\myAPP\ToNote\violin
.\.venv\Scripts\python.exe transcribe_violin.py "input/天鹅-小提琴.mp3"
```

激活环境后也可使用普通命令：

```powershell
.\.venv\Scripts\Activate.ps1
python transcribe_violin.py "D:/music/violin.wav" --output output --multiple-pitch-bends --min-frequency 196 --max-frequency 2637
```

若激活脚本受 PowerShell 执行策略限制，使用第一种方式。相对路径基于当前工作目录，支持中文和空格。从本目录执行时默认写入 `violin/output/`；非空输出目录会另建带随机后缀的同级目录，保留旧结果。

## 安装与依赖

在其他 Windows x64 / Python 3.11 电脑安装：

```powershell
cd D:\myAPP\ToNote\violin
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

本工作区也可使用已有 uv（在 ToNote 根目录执行）：

```powershell
.\.tools\bin\uv.exe venv violin/.venv --python .venv/Scripts/python.exe --cache-dir .cache/uv
.\.tools\bin\uv.exe pip install --python violin/.venv/Scripts/python.exe --cache-dir .cache/uv -r violin/requirements.txt
```

固定 NumPy 1.26.4 以兼容 TensorFlow 2.14，固定 setuptools 80.9.0 以保留 resampy 所用的 `pkg_resources`。预训练权重随 Basic Pitch 包安装；安装完成后本地推理不需联网。未安装 CoreML/TFLite/ONNX 的提示不影响 TensorFlow 后端。

## 参数

参数集中在 [config.py](violin_transcriber/config.py)。

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--onset-threshold` | 0.5 | 起音激活阈值 |
| `--frame-threshold` | 0.3 | 持续激活阈值 |
| `--minimum-note-length` | 127.7 | **毫秒**，传入上游 API，由其转换为帧数 |
| `--min-frequency` / `--minimum-frequency` | 196 | 筛选下界 Hz |
| `--max-frequency` / `--maximum-frequency` | 2637 | 筛选上界 Hz |
| `--multiple-pitch-bends` | 始终开启 | 不加此开关也为 True，不能关闭 |
| `--short-note-ms` | 150 | 仅标记短音，不删除 |
| `--output` | output | 输出目录 |
| `--verbose` | 关闭 | 错误堆栈与详细日志 |

```powershell
.\.venv\Scripts\python.exe transcribe_violin.py "input/天鹅-小提琴.mp3" --onset-threshold 0.5 --frame-threshold 0.3 --minimum-note-length 127.7 --output output_trial
```

音域转换为最近的 MIDI 半音中心后包含两端，196–2637 Hz 对应 G3–E7（55–100），避免 Hz 四舍五入误删边界音。只筛选音符中心，不裁剪弯音；可以扩大音域保留更高泛音。模型自身范围 A0–C8，不能靠参数扩展模型能力。

## 原始结果与保守处理

只运行一次 `predict()`，显式 `multiple_pitch_bends=True`，频率上下限传 None。使用 Basic Pitch 自带的读取、重采样、分窗与重叠拼接，不自行切片或移动音频起点。

1. **Raw Baseline**：官方解码结果，未应用小提琴音域规则。raw MIDI 保留官方 Electric Piano 音色、按音高划分的 instrument 和 MIDI 弯音。这仍包含模型自带的阈值、Melodia 解码及 MIDI 转换，完整原始激活另存 NPZ。
2. **Violin Filtered**：仅按配置筛选音域，不改音高、起止、score 或弯音，不合并重复音、不删疑似谐波。
3. **Suspected Errors**：八度或整数倍频率关系还需高度时间重合、近似起止对齐才标记，并记录 score 比及激活均值。真实八度双音也可能被标记，不能据此判错。单独出现的错误八度无法仅靠音符关系确定。

三音以上区间、同音高重叠、音域外事件和短音也只标记。双音漏检、技法和真实声学起止仍需参考谱或人工标注验证。

## MIDI 与弯音

官方 `multiple_pitch_bends=True` 按 pitch 建立 instrument，它们都代表同一把小提琴。官方 MIDI 写出库超过 15 个 instrument 时循环使用旋律通道，仍可能冲突。

raw 原样保留。expressive 按同时发声分配独立通道（包括同音高重叠），使用与上游相同的弯音转换，加入 ±2 半音 RPN 和结束归零；每个源弯音值、时间都回读核对。plain 不带弯音，常规双音保留在一个 Violin 轨道，同音高重叠时另分通道保护起止。两份筛选 MIDI 均使用零基 Violin program=40。

超过 15 个真正同时发声的 expressive 音符会使用额外 MIDI port 保留事件，报告显示端口数；有些播放器忽略 port，需检查兼容性。固定 120 BPM / 9600 ticks per beat 仅用于秒与 ticks 转换，不是音乐节拍估计或量化。

contour 为每半音 3 个 bin，约 33.3 音分，帧步长约 11.61 ms。上游弯音导出时间用音符起止之间的 `linspace`，与源帧时间不同，MIDI 弯音还会在 ±2 半音饱和。JSON 保留两种时间和原始 bin，不能将这些弯音当成高精度连续 F0。大幅滑音需结合原始 contour 激活研究。

## 输出

| 文件 | 内容 |
| --- | --- |
| `violin_raw.mid` | 官方未筛音域结果和原始 MIDI 弯音 |
| `violin_plain.mid` | 音域内复音，不含弯音，便于制谱导入 |
| `violin_expressive.mid` | 音域内复音和弯音，独立通道保护 |
| `notes_raw.csv` / `notes_filtered.csv` | note_id、起止秒、pitch_midi、note_name、velocity、amplitude_or_score |
| `suspected_errors.csv` | 候选类型、区间、关联 ID、音高、倍频、重合、score/激活证据 |
| `pitch_bends.json` | 全部 raw 音符（含筛掉的）、原始 bin、源帧/导出时间、MIDI 弯音值、保留状态和路由 |
| `model_outputs.npz` | 完整 note/onset/contour，官方帧时间、音高坐标和原音频时长，无 pickle 对象 |
| `piano_roll.png` | 灰色 raw、蓝色音域内矩形，可显示复音 |
| `pitch_debug.png` | 原始 note/contour 激活、音符叠加和导出弯音；仅显示可能池化，NPZ 不降采样 |
| `report.txt` | 数量、重叠、疑点、音频/MIDI 时长、验证和限制 |
| `run.json` | 机器可读统计、参数、API 签名、版本、输入 SHA-256 和状态 |

score 是模型激活强度，不是校准后的正确概率。CSV 保留原始浮点值。ID 在 raw、filtered、JSON 和疑点文件中一致，筛选后不重新编号。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m tests.smoke_basic_pitch
```

单元测试覆盖合法双音、只标记谐波、同音高重叠、弯音和时间回读、通道重用、15 通道限制、音域边界及旧结果保护。集成测试实际运行模型，检测带独立揉弦的 G3+D4 合成双音，使用中文空格路径，保存完整输出，验证静音返回失败。

真实《天鹅》已整段运行，见 [VALIDATION.md](VALIDATION.md)。无参考谱时不报告准确率或 Precision / Recall / F1。MIDI 序列化时间误差不等于声学起止检测误差。

未复用现有人声项目的单音切分和节拍量化。格式读取失败时复用 `../vocal/vocal_transcriber/audio.py` 的 FFmpeg 解码；通用 PDF 工具保留于 `../tools/`。当前不输出 MusicXML 或排版乐谱。

官方参考：[Basic Pitch](https://github.com/spotify/basic-pitch)、[predict API](https://github.com/spotify/basic-pitch/blob/main/basic_pitch/inference.py)、[复音与弯音导出](https://github.com/spotify/basic-pitch/blob/main/basic_pitch/note_creation.py)。

## 参考谱热图对照

用户提供的《天鹅》图片谱已人工录入为 `reference/the_swan_score.json`（28小节，初稿待逐音校对）。在 `violin/` 中运行：

```powershell
.\.venv\Scripts\python.exe compare_reference.py
```

查看 `output/reference_comparison/reference_vs_model_heatmap.png`：参考谱音符占用、模型 note 激活及 contour 激活共用相同时间轴和音高轴，并有四张分段放大图。参考时间由原音频特征自动估计，不是人工标注的起止真值；图片谱不能提供真实频谱、揉弦或谐波能量。此命令不重新推理，也不修改原始模型输出及 MIDI。详情见 [对照说明](output/reference_comparison/README.md)。
