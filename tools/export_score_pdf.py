"""Render an existing MusicXML score to a vector A4 PDF without MuseScore."""
import argparse
from pathlib import Path
import html
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET


def export_pdf(source: Path, target: Path, title: str | None = None) -> int:
    import verovio
    from pypdf import PdfReader

    xml = source.read_text(encoding="utf-8")
    if title:
        tree = ET.fromstring(xml)
        for element in tree.findall("./work/work-title") + tree.findall("./movement-title"):
            element.text = title
        xml = ET.tostring(tree, encoding="unicode")
    toolkit = verovio.toolkit()
    toolkit.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 38,
                        "pageMarginTop": 100, "pageMarginBottom": 100,
                        "pageMarginLeft": 100, "pageMarginRight": 100,
                        "adjustPageHeight": False, "breaks": "auto",
                        "svgViewBox": True, "footer": "none"})
    if not toolkit.loadData(xml):
        raise RuntimeError("MusicXML loading failed: " + toolkit.getLog())
    pages = toolkit.getPageCount()
    if pages < 1:
        raise RuntimeError("The score has no printable pages")
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = []
    for page in range(1, pages + 1):
        svg = toolkit.renderToSVG(page)
        rendered.append(f'<section>{svg}<div class="page-number">{page} / {pages}</div></section>')
        print(f"Rendered page {page}/{pages}", flush=True)
    preview = target.with_suffix(".html")
    preview.write_text('<!doctype html><html><head><meta charset="utf-8"><title>'
                       + html.escape(title or source.stem) + '</title><style>'
                       '@page { size: A4; margin: 0; }'
                       'body { margin:0; background:#ddd; }'
                       'section { width:210mm; height:297mm; position:relative; background:white;'
                       ' overflow:hidden; break-after:page; } section:last-child { break-after:auto; }'
                       'section > svg { display:block; width:210mm; height:297mm; }'
                       'section:not(:first-of-type) .pgHead { display:none; }'
                       '.page-number { position:absolute; bottom:7mm; width:100%; text-align:center;'
                       'font:10pt sans-serif; color:#555; }'
                       '</style></head><body>' + ''.join(rendered) + '</body></html>', encoding="utf-8")
    candidates = [Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
                  Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe"]
    browser = next((p for p in candidates if p.is_file()), None)
    if browser is None:
        raise RuntimeError(f"Chrome/Edge not found. Open {preview} in a browser and print to PDF.")
    cache = Path(__file__).resolve().parent.parent / ".cache"
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdf_browser_", dir=cache) as profile:
        result = subprocess.run([str(browser), "--headless", "--disable-gpu", "--no-first-run",
                                 "--no-default-browser-check", "--no-pdf-header-footer",
                                 "--disable-extensions", "--disable-background-networking",
                                 f"--user-data-dir={profile}", f"--print-to-pdf={target.resolve()}",
                                 preview.resolve().as_uri()], capture_output=True, timeout=90)
        if result.returncode or not target.is_file():
            raise RuntimeError("Browser PDF rendering failed: " + result.stderr.decode(errors="replace")[-2000:])
    reader = PdfReader(target)
    if len(reader.pages) != pages or any(not p.get_contents().get_data() for p in reader.pages):
        raise RuntimeError("PDF page validation failed")
    return pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title")
    args = parser.parse_args()
    output = args.output or args.source.with_suffix(".pdf")
    count = export_pdf(args.source, output, args.title)
    print(f"PDF ready: {output.resolve()} ({count} pages)")
