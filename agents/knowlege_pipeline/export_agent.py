import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem, Flowable, PageBreak
from crewai import Agent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExportPDFAgent(Agent):
    def __init__(self, name="ExportPDFAgent", role="PDF Report Generator",
                 goal="Generate professional research reports in PDF format",
                 backstory="You turn research findings into structured, professional-looking papers for decision-makers."):
        super().__init__(name=name, role=role, goal=goal, backstory=backstory)

    # --- Helper functions ---
    
    def _get(self, d: dict, *keys, default=None):
        """Safely get value from dict with fallback keys"""
        for key in keys:
            if key in d:
                return d[key]
        return default

    def _split_llm_explanation(self, text: str) -> List[Tuple[str, str]]:
        """
        Split LLM explanation into logical sections.
        Returns list of (section_title, section_content) pairs.
        """
        if not text or not isinstance(text, str):
            return []
        
        sections = []
        lines = text.split('\n')
        current_section = ""
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this looks like a section header (ends with colon, shorter line)
            if ':' in line and len(line) < 80 and not line.startswith('-') and not line.startswith('*'):
                # Save previous section
                if current_section and current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                # Start new section
                current_section = line.replace(':', '').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_section and current_content:
            sections.append((current_section, '\n'.join(current_content)))
        elif current_content:  # No sections detected, treat as single content
            sections.append(("Analysis", '\n'.join(current_content)))
        logger.info(f"Split explanation into {len(sections)} logical sections.")
        return sections

    def _md_to_paragraphs(self, text: str, style: ParagraphStyle) -> List[Paragraph]:
        """Convert text with basic markdown to ReportLab Paragraphs"""
        if not text:
            return []
        
        # Clean the text first
        text = self.clean_text_for_pdf(text)
        
        # Basic markdown processing with proper tag matching
        # Handle bold markdown
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # Handle italic markdown  
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        
        # Clean up any malformed tags
        text = re.sub(r'<b>([^<]*)<b>', r'<b>\1</b>', text)  # Fix unclosed bold tags
        text = re.sub(r'<i>([^<]*)<i>', r'<i>\1</i>', text)  # Fix unclosed italic tags
        
        paragraphs = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                # Remove any remaining problematic characters
                line = re.sub(r'[^\w\s<>/\-:.,!?()&;]', '', line)
                try:
                    paragraphs.append(Paragraph(line, style))
                except Exception as e:
                    # Fallback to plain text if parsing fails
                    clean_line = re.sub(r'<[^>]*>', '', line)  # Remove all HTML tags
                    paragraphs.append(Paragraph(clean_line, style))
        
        return paragraphs if paragraphs else [Paragraph(text, style)]

    def _kv_table(self, items: List[Tuple[str, str]]) -> Table:
        """Create a simple key-value table"""
        table = Table(items, colWidths=[140, 300])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB'))
        ]))
        return table

    def _list(self, items: List[str], style: ParagraphStyle) -> ListFlowable:
        """Create a bulleted list"""
        list_items = [ListItem(Paragraph(item, style)) for item in items]
        return ListFlowable(list_items, bulletType='bullet')

    def _header_footer(self, canvas, doc):
        """Draw header and footer on each page"""
        # Header
        canvas.saveState()
        canvas.setFont('Helvetica-Bold', 12)
        canvas.setFillColor(colors.HexColor('#2E3440'))
        canvas.drawString(50, A4[1] - 50, "Research & Fact-Check Report")
        
        # Footer
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.HexColor('#6B7280'))
        canvas.drawString(50, 30, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        canvas.drawRightString(A4[0] - 50, 30, f"Page {doc.page}")
        canvas.restoreState()

    def clean_text_for_pdf(self, text):
        """Clean text for PDF rendering by replacing problematic Unicode characters"""
        if not isinstance(text, str):
            text = str(text)
        
        replacements = {
            '\u2019': "'", # Right single quotation mark
            '\u2018': "'", # Left single quotation mark  
            '\u201c': '"', # Left double quotation mark
            '\u201d': '"', # Right double quotation mark
            '\u2013': '-', # En dash
            '\u2014': '--', # Em dash
        }
        
        for unicode_char, replacement in replacements.items():
            text = text.replace(unicode_char, replacement)
        
        return text

    def export_pdf(self, research_data: Any, fact_check: Any, output_path: str = "exported_research.pdf") -> str:
        """
        Render a well-structured PDF from research_data and fact_check JSON.
        - Handles new schema with: statement, relevance_score, llm_explanation
        - Falls back to older structures if necessary
        """
        #print("-"*100)
        #print("Research data keys:", list(fact_check.keys()) if isinstance(fact_check, dict) else type(fact_check))
        #print("-"*100)

        # Prepare data
        logger.info("Starting PDF export...")
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # Custom styles
        H1 = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=16,
            textColor=colors.HexColor('#2E3440')
        )
        
        H2 = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=12,
            spaceAfter=8,
            textColor=colors.HexColor('#5E81AC')
        )
        
        Body = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            leftIndent=0,
            spaceAfter=6,
            alignment=0  # Left alignment
        )

        story = []

        # Title
        story.append(Paragraph("Research & Fact-Checking Report", H1))
        story.append(Spacer(1, 12))

        # Metadata table
        meta_data = [
            ("Report Generated:", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("Document Type:", "Research Analysis with Fact-Check")
        ]
        story.append(self._kv_table(meta_data))
        story.append(Spacer(1, 16))

        # --- Extract and process fact_check data ---
        if isinstance(fact_check, dict):
            # New schema: extract statement, relevance_score, llm_explanation
            statement = self._get(fact_check, "statement", "content", "response", default="No statement provided")
            score = self._get(fact_check, "relevance_score", "score", default=0.0)
            explanation = self._get(fact_check, "llm_explanation", "explanation", default="No explanation provided")
            
            # Parse explanation
            explanation_parts = self._split_llm_explanation(explanation)
            
            # Extract follow-up questions
            followups = self._get(fact_check, "follow_up_questions", "follow_ups", default=[])
            
        elif isinstance(fact_check, str):
            # Simple string content
            try:
                parsed_fact_check = json.loads(fact_check)
                statement = self._get(parsed_fact_check, "statement", "content", "response", default=fact_check)
                score = self._get(parsed_fact_check, "relevance_score", "score", default=0.0)
                explanation = self._get(parsed_fact_check, "llm_explanation", "explanation", default="")
                followups = self._get(parsed_fact_check, "follow_up_questions", "follow_ups", default=[])
                explanation_parts = self._split_llm_explanation(explanation)
            except:
                statement = fact_check
                score = 0.0
                explanation = ""
                followups = []
                explanation_parts = []
        else:
            statement = "Unknown content format"
            score = 0.0
            explanation = ""
            followups = []
            explanation_parts = []

        # Executive Summary
        story.append(Paragraph("Executive Summary", H2))
        summary_text = f"This report presents research findings with a relevance score of {score:.1f}/1.0. " \
                      f"The analysis {"meets" if score >= 0.7 else "partially meets" if score >= 0.4 else "does not meet"} " \
                      f"standard relevance thresholds."
        story.append(Paragraph(summary_text, Body))
        story.append(Spacer(1, 12))

        # Research Findings
        story.append(Paragraph("Research Findings", H2))
        story.extend(self._md_to_paragraphs(self.clean_text_for_pdf(str(statement)), Body))
        story.append(Spacer(1, 12))

        # Fact-Check Analysis
        story.append(Paragraph("Fact-Check Analysis", H2))

        # Relevance Score with visual bar
        story.append(Paragraph(f"<b>Relevance Score:</b> {score:.2f}/1.0", Body))
        story.append(ScoreBar(score, label=f"{score:.1f}"))
        story.append(Spacer(1, 12))

        # Detailed Analysis
        if explanation_parts:
            story.append(Paragraph("Detailed Analysis", H2))
            
            for section_title, section_content in explanation_parts:
                if section_title.strip() and section_content.strip():
                    story.append(Paragraph(f"<b>{section_title.strip()}:</b>", Body))
                    story.extend(self._md_to_paragraphs(self.clean_text_for_pdf(section_content), Body))
                    story.append(Spacer(1, 6))
            story.append(Spacer(1, 6))

        # Follow-up suggestions
        if followups:
            story.append(Paragraph("Suggested Follow-up Questions", H2))
            story.append(self._list(followups, Body))
            story.append(Spacer(1, 6))

        # Notes & Limitations (static section)
        story.append(Paragraph("Notes & Limitations", H2))
        story.append(Paragraph(
            "The above content is based on synthesized and fact-checked information from publicly available sources. "
            "While care has been taken to ensure accuracy, readers should evaluate the sources and context for their specific use cases.",
            Body
        ))

        # Build PDF
        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        logger.info(f"PDF report generated: {output_path}")
        return output_path


class ScoreBar(Flowable):
    """
    Simple relevance score bar (0..1) with label.
    """
    def __init__(self, score: float, width: float = 160, height: float = 10, label: Optional[str] = None):
        super().__init__()
        self.score = max(0.0, min(1.0, float(score or 0.0)))
        self.width = width
        self.height = height
        self.label = label or f"{self.score:.1f}"

    def draw(self):
        logger.info("Drawing score bar. Relevance score: %s", self.score)
        c = self.canv
        x, y = 0, 0
        # border
        c.setStrokeColor(colors.black)
        c.rect(x, y, self.width, self.height, stroke=1, fill=0)
        # fill
        c.setFillColor(colors.green if self.score >= 0.7 else (colors.orange if self.score >= 0.4 else colors.red))
        c.rect(x, y, self.width * self.score, self.height, stroke=0, fill=1)
        # label
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 9)
        c.drawString(x + self.width + 6, y - 1 + self.height, self.label)
