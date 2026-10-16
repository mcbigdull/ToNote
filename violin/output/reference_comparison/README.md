# 《天鹅》参考谱与模型热图

先打开 `reference_vs_model_heatmap.png`。三行从上到下：

1. 用户参考谱的人工录入音高，经音频特征对齐后的音符占用图。
2. 原始 Basic Pitch `note` 激活，与此前 `pitch_debug.png` 的第一行同源。
3. 原始 Basic Pitch `contour` 激活，与此前调试图的第二行同源。

三行采用完全相同的横轴秒数、纵轴 MIDI 音高和绘图区宽度，便于垂直比较同一时间。参考行的 m2、m3 等是原谱小节。颜色沿用 magma；参考亮度1表示谱面有这个音，模型亮度是激活强度，两者不代表相同的声学能量或概率。

## 图片

- `reference_vs_model_heatmap.png`：整曲三行对照。
- `reference_heatmap.png`：单独的参考谱热图。
- `comparison_000_040.png`：0–40秒放大。
- `comparison_040_080.png`：40–80秒放大。
- `comparison_080_110.png`：80–110秒放大。
- `comparison_110_end.png`：110秒至末尾放大，能对照B4和末尾G5长音。

## 参考与时间轴

人工录入来源是 `../../reference/the_swan_original.png`，结构数据在 `../../reference/the_swan_score.json`。全谱28小节，每小节时值和均为6个四分单位，按实际同音延音合并持续发音；指法和连弓不自动当作新的音高。

这是谱面的音高热图，**不是频谱图**。图片谱无法提供演奏音色、真实谐波强弱、揉弦或滑音细节，因此没有给参考行虚构这些内容。

谱子没有数字BPM，并有自由速度变化。为让它与音频共用秒轴，使用原录音的CQT色度特征与谱面音高做受约束的动态时间规整（DTW）；未使用Basic Pitch音符事件或其激活来生成参考音高或估计对齐。首尾整小节休止按录音的实际可用头尾分配，不能从该图反推完整伴奏前奏的演出时长。

**这是为了看旋律而估计的时间对齐，不是人工标注的音符起止真值。** 色度特征有八度模糊性，弱音、休止及连弓处的时间边界可能偏差。模型行保留此前NPZ中保存的原始帧时间，没有偷偷校正模型时间轴。故局部横向偏移可能包含对齐误差或模型原有时间误差，不能直接当成漏音/错音。

本MP3的容器报告约140.411秒，librosa实际解码约140.199秒。与Basic Pitch相同，按解码采样零点起算；图内所有行采用同一秒轴。尾端未覆盖的模型时间段保持空白。

## 可复核数据

- `reference_symbols.csv`：逐小节的谱面音符、休止、时值和延音线。
- `reference_notes_aligned.csv`：合并同音延音后的参考事件，同时保留拍位置和估计秒位置。
- `alignment_path.csv`：谱面四分单位到录音秒数的映射。
- `alignment_report.json`：来源哈希、配置、范围及局限。
- `reference_heatmap.npz`：参考热图及坐标。
- `audio_alignment_features.npz`：用于对齐的音频与参考色度特征。

原有模型激活、MIDI和检测阈值均未改动。这组图本身不宣称全曲准确率，手工读谱也仍应接受逐音校对。

重新生成（在 `violin/` 运行）：

```powershell
.\.venv\Scripts\python.exe compare_reference.py
```

DTW方法使用 [librosa官方实现](https://librosa.org/doc/0.10.2/generated/librosa.sequence.dtw.html)。
