import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from typing import TYPE_CHECKING, List, Dict, Optional
from .visualizer import create_neutralization_heatmap

if TYPE_CHECKING:
    from biovoice.core.grant_config import GrantOutput

class PPTGenerator:
    def __init__(self, template_path: Optional[str] = None):
        if template_path and os.path.exists(template_path):
            self.prs = Presentation(template_path)
        else:
            self.prs = Presentation()
            if template_path:
                print(f"Warning: Template file '{template_path}' not found. Using default blank presentation.")

    def add_title_slide(self, title: str, subtitle: str = ""):
        slide_layout = self.prs.slide_layouts[0]  # title slide
        slide = self.prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = title
        if subtitle:
            slide.placeholders[1].text = subtitle

    def add_content_slide(self, title: str, bullets: List[str]):
        slide_layout = self.prs.slide_layouts[1]  # title and content
        slide = self.prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = title
        content = slide.placeholders[1]
        text_frame = content.text_frame
        text_frame.clear()
        for bullet in bullets:
            p = text_frame.add_paragraph()
            p.text = bullet
            p.level = 0

    def add_table_slide(self, title: str, headers: List[str], rows: List[List[str]]):
        slide_layout = self.prs.slide_layouts[5]  # title only
        slide = self.prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = title
        rows_count = len(rows)
        cols_count = len(headers)
        left = Inches(1)
        top = Inches(1.5)
        width = Inches(8)
        height = Inches(0.5 * (rows_count + 1))
        table = slide.shapes.add_table(rows_count + 1, cols_count, left, top, width, height).table
        # header row
        for col, header in enumerate(headers):
            table.cell(0, col).text = header
        # data rows
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                table.cell(i+1, j).text = str(cell)

    def add_neutralization_heatmap(self, data: Dict[str, List[float]], title: str = "Neutralization Breadth"):
        """添加热图幻灯片"""
        img_bytes = create_neutralization_heatmap(data, title)
        slide_layout = self.prs.slide_layouts[6]  # blank
        slide = self.prs.slides.add_slide(slide_layout)
        left = Inches(1)
        top = Inches(1.5)
        slide.shapes.add_picture(img_bytes, left, top, height=Inches(5))
        # 添加标题（可选）
        title_box = slide.shapes.add_textbox(left, Inches(0.5), Inches(8), Inches(1))
        title_box.text = title

    def save(self, path: str):
        self.prs.save(path)

    # ── Grant mode methods ─────────────────────────────────────────────────────

    def add_grant_section_slide(self, section_title: str, text: str, citation_count: int = 0):
        """Add a slide for a single NIH grant section (Significance, Innovation, etc.)."""
        slide_layout = self.prs.slide_layouts[1]  # title + content
        slide = self.prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = section_title
        content = slide.placeholders[1]
        tf = content.text_frame
        tf.word_wrap = True
        tf.clear()
        # Trim to ~800 chars so it fits on one slide
        display = text[:800] + ("..." if len(text) > 800 else "")
        p = tf.add_paragraph()
        p.text = display
        p.level = 0
        if citation_count:
            note_p = tf.add_paragraph()
            note_p.text = f"({citation_count} source(s) cited)"
            note_p.level = 1

    def add_grant_key_findings_slide(self, papers):
        """
        Slide 5: key findings table.
        papers: list of Citation objects (title, authors, year, pmid)
        """
        if not papers:
            return
        headers = ["#", "Authors", "Year", "Title (truncated)", "PMID"]
        rows = [
            [
                str(i + 1),
                (c.authors or "")[:30],
                c.year or "",
                (c.title or "")[:60],
                c.pmid or "",
            ]
            for i, c in enumerate(papers[:12])
        ]
        self.add_table_slide("Key Literature", headers, rows)

    def add_grant_references_slide(self, citations):
        """Slide 6: full reference list."""
        slide_layout = self.prs.slide_layouts[1]
        slide = self.prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = "References"
        tf = slide.placeholders[1].text_frame
        tf.word_wrap = True
        tf.clear()
        for i, c in enumerate(citations[:15], start=1):
            p = tf.add_paragraph()
            p.text = (
                f"[{i}] {c.authors or 'Unknown'}. {(c.title or '')[:60]}. "
                f"{c.journal or ''} {c.year or ''}. PMID:{c.pmid}"
            )
            p.level = 0
            p.font.size = Pt(10) if hasattr(p, 'font') else None


def render_grant_ppt(grant_output, output_path: str) -> str:
    """
    Render a GrantOutput into a 6-slide PPT.

    Slide structure (fixed in v1):
      1. Title slide (research question + date)
      2. Specific Aims
      3. Significance
      4. Innovation
      5. Key findings table (top cited papers)
      6. References

    Returns output_path.
    """
    from datetime import date

    gen = PPTGenerator()

    # Slide 1: Title
    gen.add_title_slide(
        "Virology Grant Copilot",
        f"Research question: {grant_output.research_question[:80]}\n"
        f"Generated: {date.today().isoformat()}",
    )

    # Slides 2–4: one per grant section in output order
    section_order = ["specific_aims", "significance", "innovation", "background"]
    for key in section_order:
        gs = grant_output.section(key)
        if gs is None:
            continue
        gen.add_grant_section_slide(gs.title, gs.text, len(gs.citations))

    # Slide 5: key findings
    gen.add_grant_key_findings_slide(grant_output.all_citations)

    # Slide 6: references
    gen.add_grant_references_slide(grant_output.all_citations)

    gen.save(output_path)
    print(f"[PPT] Saved: {output_path}")
    return output_path