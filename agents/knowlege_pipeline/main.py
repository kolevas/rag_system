from .export_agent import ExportPDFAgent
from .research_agent import ResearchAgent
import os
from .fact_checking_agent import FactCheckAgent

def research_pipeline(query: str):
    """Complete research pipeline that returns structured results"""
    print(f"🔬 Starting research pipeline for: {query}")
    
    try:
        research_agent = ResearchAgent()
        fact_checking_agent = FactCheckAgent()
        export_agent = ExportPDFAgent()
        
        # Step 1: Research
        print("📚 Step 1: Conducting research...")
        research_data = research_agent.research_agent(query)
        
        # Step 2: Fact check
        print("✅ Step 2: Fact checking...")
        result = fact_checking_agent.fact_check_with_tavily_llm(research_data)
        
        # Step 3: Export to PDF
        print("📄 Step 3: Exporting to PDF...")
        export_agent.export_pdf(research_data, result, filename="fact_check_report.pdf")
        
        print("🎉 Research pipeline completed successfully!")
        
        # Return structured result
        return {
            "query": query,
            "research_data": research_data,
            "fact_check_results": result,
            "pdf_exported": "fact_check_report.pdf",
            "status": "completed"
        }
        
    except Exception as e:
        print(f"❌ Research pipeline failed: {e}")
        return {
            "query": query,
            "error": str(e),
            "status": "failed",
            "fallback_message": f"I apologize, but I couldn't research '{query}' at the moment. Please try again later or rephrase your question."
        }
    