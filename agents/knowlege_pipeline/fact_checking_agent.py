# pip install openai tavily

import os
from tavily import TavilyClient
from openai import AzureOpenAI
from .research_agent import ResearchAgent
from agents.knowlege_pipeline.export_agent import ExportPDFAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactCheckAgent:
    def __init__(self):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.llm = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-01"
        )

    def summarize_content(self, content, max_length: int = 400):
        """
        Summarize a content string to fit within the max_length.
        Returns a single string summary.
        """
        try:
            if not content:
                return ""
            if len(content) < max_length:
                return content
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Extract 3-5 key factual claims from this content that can be fact-checked.
                        Each claim should be under 300 characters and be a specific, verifiable statement.
                        
                        Content: {content[:1000]}...
                        
                        Return as a JSON array of strings:
                        ["claim 1", "claim 2", "claim 3"]"""}, 
                    {"role": "user", "content": f"Please summarize this research:\n{content}"}
                ],
                max_tokens=60,
                temperature=0.3,
                timeout=30
            )
            summary = response.choices[0].message.content.strip()
            if summary and len(summary) > 10:
                logger.info("Content summarized successfully. Summary length: %d", len(summary))
                return summary
            return content[:max_length] + "..."
        except Exception as e:
            #print(f"❌ Summary error: {e}")
            return content[:max_length] + "..." 

    def fact_check_with_tavily_llm(self, research_data: dict):
        """
        Combines Tavily retrieval with LLM reasoning (strict fact-checking).
        Returns JSON with statement, relevance_score, and explanation.
        """
        all_results = research_data.get("results", [])
        tavily_results = []
        logger.info(f"Fact-checking {len(all_results)} research results with Tavily...")
        
        for r in all_results:
            content = r.get("content", "")
            if content:
                # Truncate content to 400 chars for Tavily search
                search_query = self.summarize_content(content)
                try:
                    tavily_result = self.client.search(search_query)
                    tavily_results.append(tavily_result)
                except Exception as e:
                    #print(f"❌ Tavily search error: {e}")
                    continue
        # Strict LLM reasoning prompt
        logger.info("Generating LLM prompt for fact-checking...")
        llm_prompt = f"""
        You are a strict fact-checking assistant.

        Statement to check:
        "{research_data.get('query')}"

        Evidence retrieved from Tavily:
        {tavily_results}

        - Be strict: if the evidence is about replicas or unrelated facts, the statement is FALSE.
        - If the evidence is inconclusive, return "Undefined".
        Write a long, structured report with:
            - An executive summary (5 to 7 sentences)
            - A detailed breakdown of findings (at least 3 to 5 paragraphs)
            - A section explaining fact-checking relevance scores
            - A concluding analysis with recommendations

        The report should read like a professional research paper,
        not just a short answer. Provide depth, context, and reasoning.
        Task:
        1. Decide if the statement is factually correct.
        2. Calculate a relevance score: a number between 0.00 (completely false) and 1.00 (completely true).
        3. Explain your reasoning.

        Return JSON ONLY in this format:
        {{
          "statement": "{research_data.get('query')}",
          "relevance_score": <number>,
          "llm_explanation": "<your report, reasoning, and conclusion>"
        }}
        """

        llm_response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": llm_prompt}],
        )

        return llm_response.choices[0].message.content


# Example usage
if __name__ == "__main__":
    research_agent = ResearchAgent()
    research_data = research_agent.research_agent("The Eiffel Tower is located in Berlin.")
    fact_checking_agent = FactCheckAgent()
    result = fact_checking_agent.fact_check_with_tavily_llm(research_data)
    export_agent = ExportPDFAgent()
    export_agent.export_pdf(research_data, result, filename="fact_check_report.pdf")
    #print("Fact-checking report generated successfully.")
    #print(result)
