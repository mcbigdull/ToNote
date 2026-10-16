# ToNote 转谱工作区

按处理对象分项目，当前目录如下：

```text
ToNote/
├── vocal/           现有人声转谱项目：源码、测试、依赖和文档
│   └── output/      人声项目后续运行结果
├── violin/          Basic Pitch 小提琴复音转 MIDI 项目（独立 .venv）
│   ├── input/       小提琴输入音频
│   └── output/      小提琴转谱结果
├── tools/           通用 MusicXML 转 PDF 工具及其可选依赖
├── input/           整理前已有的输入素材，保留原位置
├── output/          整理前已有的人声结果及验证记录，保留原位置
├── .venv/           已安装的 Python 环境
├── .python/         本地 Python 运行时
├── .cache/          已下载模型和计算缓存
└── .tools/          本地环境工具
```

现有素材和历史结果保留原路径，以保持历史记录中的绝对路径有效。新项目的素材请放入对应项目目录。

## 人声项目

在 PowerShell 中进入人声目录后，使用根目录现有 Python 环境：

```powershell
cd D:\myAPP\ToNote\vocal
..\.venv\Scripts\python.exe transcribe_vocal.py "../input/莫非定律乐团 - Lidocaine.mp3"
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

使用方法见 [人声项目说明](vocal/README.md)，原验证记录见 [VALIDATION.md](vocal/VALIDATION.md)。

## 通用 PDF 导出工具

在工作区根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r tools/requirements-pdf.txt
.\.venv\Scripts\python.exe tools/export_score_pdf.py "output/歌曲名/melody.musicxml"
```

## 小提琴项目

已实现 Basic Pitch 本地转录，保留原始结果、双音与弯音，并输出音域筛选及疑点检查文件。使用独立环境运行：

```powershell
cd D:\myAPP\ToNote\violin
.\.venv\Scripts\python.exe transcribe_violin.py "input/天鹅-小提琴.mp3"
```

详见 [小提琴项目说明](violin/README.md)。

## 获取项目与协作

仓库包含人声与小提琴源码、测试、输入素材、参考谱及已有分析结果。Python 环境、已下载模型缓存和临时文件在本地重新生成，不纳入版本管理。音频素材随项目保存不表示获得了再次分发的许可；新增录音前请确认其来源与使用范围。

WAV 使用 Git LFS 保存（部分人声分离结果超过 GitHub 普通文件大小上限）。克隆前安装 Git 和 Git LFS，执行：

```powershell
git lfs install
git clone <仓库地址>
cd ToNote
git lfs pull
```

小提琴项目建议使用 Python 3.11：

```powershell
py -3.11 -m venv violin/.venv
violin/.venv/Scripts/python.exe -m pip install -r violin/requirements.txt
```

多人协作时各自克隆仓库，在自己的分支修改，通过 Pull Request 合并；不要让两个 Codex 同时编辑同一份本地工作目录。
