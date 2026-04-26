from pathlib import Path
import re
import textwrap

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "proposal" / "group_project_proposal.md"
OUTPUT = ROOT / "proposal" / "group_project_proposal.pdf"


def clean_inline_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    return text.strip()


def wrap_to_width(text: str, font_name: str, font_size: float, max_width: float):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def parse_markdown(path: Path):
    blocks = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            blocks.append(("space", ""))
            continue
        if line.startswith("## "):
            blocks.append(("heading", clean_inline_markdown(line[3:])))
        else:
            blocks.append(("body", clean_inline_markdown(line.rstrip("  "))))
    return blocks


def layout_blocks(blocks, body_size, page_width):
    margin_x = 34
    max_width = page_width - margin_x * 2
    line_gap = body_size + 1.15
    heading_size = body_size + 1.65
    rendered = []
    for kind, text in blocks:
        if kind == "space":
            rendered.append(("space", "", body_size, body_size * 0.25))
            continue
        if kind == "heading":
            rendered.append(("heading", text, heading_size, heading_size + 1.2))
            continue
        for wrapped in wrap_to_width(text, "Helvetica", body_size, max_width):
            rendered.append(("body", wrapped, body_size, line_gap))
    return rendered


def render_pdf():
    blocks = parse_markdown(SOURCE)
    page_width, page_height = letter
    margin_top = 30
    margin_bottom = 28

    for body_size in [8.4, 8.2, 8.0, 7.8, 7.6, 7.4, 7.2]:
        rendered = layout_blocks(blocks, body_size, page_width)
        total_height = sum(item[3] for item in rendered)
        if total_height <= page_height - margin_top - margin_bottom:
            break
    else:
        raise RuntimeError("Proposal is too long to fit one letter page.")

    pdf = canvas.Canvas(str(OUTPUT), pagesize=letter)
    y = page_height - margin_top
    for kind, text, size, advance in rendered:
        if kind == "space":
            y -= advance
            continue
        if kind == "heading":
            pdf.setFont("Helvetica-Bold", size)
        else:
            pdf.setFont("Helvetica", size)
        pdf.drawString(34, y, text)
        y -= advance

    pdf.setTitle("Group Project Proposal - Adaptive Inventory & Pricing")
    pdf.setAuthor("BU.520.750.51.SP26 Group [TBD]")
    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    render_pdf()
    print(f"Wrote {OUTPUT}")
