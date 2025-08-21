import json
from fpdf import FPDF
from crewai import Agent

class ExportPDFAgent(Agent):
    def __init__(self, name="ExportPDFAgent", role="PDF Report Generator",
                 goal="Generate professional research reports in PDF format",
                 backstory="You turn research findings into structured, professional-looking papers for decision-makers."):
        super().__init__(name=name, role=role, goal=goal, backstory=backstory)

    def clean_text_for_pdf(self, text):
        """Clean text of problematic Unicode characters for PDF export"""
        # Replace smart quotes and other Unicode characters
        replacements = {
            '\u2019': "'",  # Right single quotation mark
            '\u2018': "'",  # Left single quotation mark
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
        }
        
        for unicode_char, replacement in replacements.items():
            text = text.replace(unicode_char, replacement)
        
        return text

    def export_pdf(self, research_data, fact_check_results, filename="research_report.pdf"):
        research_data = [self.clean_text_for_pdf(item) for item in research_data]
        fact_check_results = [self.clean_text_for_pdf(item) for item in fact_check_results]
        # If input is a stringified JSON, parse it
        if isinstance(fact_check_results, str):
            try:
                fact_check_results = json.loads(fact_check_results)
            except json.JSONDecodeError:
                fact_check_results = {"statement": fact_check_results, "relevance_score": 0, "llm_explanation": ""}

        # Ensure fact_check_results is a list
        if isinstance(fact_check_results, dict):
            fact_check_results = [fact_check_results]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)

        # Title
        pdf.cell(200, 10, "Research Report", ln=True, align="C")
        pdf.ln(10)

        # Abstract
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Abstract", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 10, "This report presents the findings of an AI-powered research "
                              "process combining Tavily fact-checking and LLM synthesis. "
                              "The purpose is to provide reliable, verified information "
                              "with relevance scoring for decision-making.")
        pdf.ln(5)

        # Introduction
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Introduction", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 10, "In today's fast-paced digital environment, misinformation "
                              "is widespread. This report leverages advanced research "
                              "agents and fact-checking tools to evaluate information "
                              "credibility. The following sections summarize the research "
                              "findings and assess their reliability.")
        pdf.ln(5)

        # Research Findings
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Research Findings", ln=True)
        pdf.set_font("Arial", "", 11)
        for i, item in enumerate(research_data, 1):
            pdf.multi_cell(0, 10, f"{i}. {item}")
            pdf.ln(3)

        # Fact-Checking Results
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Fact-Checking Results", ln=True)
        pdf.set_font("Arial", "", 11)
        for result in fact_check_results:
            # If result is still a string, convert to dict
            if isinstance(result, str):
                result = {"statement": result, "relevance_score": 0, "llm_explanation": ""}

            pdf.multi_cell(
                0, 10,
                f"- Statement: {result.get('statement', 'N/A')}\n"
                f"  Relevance Score: {result.get('relevance_score', 0)*100:.1f}/100\n"
                f"  Notes: {result.get('llm_explanation', 'No explanation available')}"
            )
            pdf.ln(3)

        # Conclusion
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Conclusion", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 10, "The combined use of Tavily's fact-checking and LLM-based "
                              "analysis provides a structured, transparent, and trustworthy "
                              "overview of the topic. The relevance scores help determine "
                              "which statements are most reliable. This approach ensures "
                              "decisions can be informed by validated knowledge.")
        pdf.ln(5)

        # References
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "References", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 10, "This report was generated using the Tavily Fact Checker API, "
                              "an LLM for synthesis, and automated reporting tools.")

        pdf.output(filename)
        return filename
