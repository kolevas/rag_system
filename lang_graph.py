from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Optional
import json
from agents.billing_agent import BillingAgent
from agents.cli_agent import CLIAgent
from agents.knowlege_pipeline.fact_checking_agent import FactCheckAgent
from agents.knowlege_pipeline.research_agent import ResearchAgent
from agents.knowlege_pipeline.export_agent import ExportPDFAgent
from rag_system.llamaindex.llamaindex_engine import LlamaIndexEngine
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def _normalize_relevance_score(val) -> Optional[float]:
    """Normalize relevance score to 0.0-1.0 range from various formats"""
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            f = float(val)
        else:
            s = str(val).strip().replace("%", "")
            if "/" in s:  # e.g., "87/100" or "0.87/1"
                num, den = s.split("/", 1)
                num = float(num.strip())
                den = float(den.strip())
                f = (num / den) if den else 0.0
            else:
                f = float(s)
        if f > 1.0:
            f = f / 100.0
        if f < 0.0:
            f = 0.0
        if f > 1.0:
            f = 1.0
        return f
    except Exception:
        return None

# create a single shared LLM client for use in the pipeline
LLM_CLIENT = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

class QueryState(TypedDict):
    query: str
    classification: str
    result: Any
    relevance: str
    research_data: Any
    fact_check_results: Any
    user_choice: str
    llm: Any
    non_interactive: bool

# --- Classifier ---
class QueryClassifier:
    def __init__(self):
        self.billing_keywords = ["cost", "price", "pricing", "billing", "estimate", "aws pricing", "ec2 cost"]
        self.cli_keywords = ["command", "cli", "terminal", "bash", "kubectl", "docker", "git"]

    def classify(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in self.billing_keywords):
            return "billing"
        if any(k in q for k in self.cli_keywords):
            return "cli"
        return "knowledge"

# --- Agents ---
def classify_query(state: QueryState) -> QueryState:
    state["classification"] = QueryClassifier().classify(state["query"])
    return state

def billing_agent(state: QueryState) -> QueryState:
    if not BillingAgent:
        state["result"] = {"error": "BillingAgent not available"}
        return state
    
    try:
        state["result"] = BillingAgent().estimate_cost(state["query"])
    except Exception as e:
        state["result"] = {"error": f"Billing agent failed: {str(e)}"}
    return state

def cli_agent(state: QueryState) -> QueryState:
    if not CLIAgent:
        state["result"] = {"error": "CLIAgent not available"}
        return state
    
    try:
        state["result"] = CLIAgent().generate_cli_command(state["query"])
    except Exception as e:
        state["result"] = {"error": f"CLI agent failed: {str(e)}"}
    return state

def retrieval_agent(state: QueryState) -> QueryState:
    if not LlamaIndexEngine:
        state["result"] = {"error": "LlamaIndexEngine not available"}
        state["relevance"] = "low"
        return state
    
    try:
        manager = LlamaIndexEngine(
            persist_directory="./rag_system/chroma_db",
            collection_name="multimodal_downloaded_data_with_embedding",
            user_id="default_user"
        )
        nodes = manager.get_document_context(state["query"])
        state["result"] = nodes
        state["relevance"] = calculate_relevance(nodes)
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        state["result"] = {"error": str(e)}
        state["relevance"] = "low"
    return state

def rag_agent(state: QueryState) -> QueryState:
    if not LlamaIndexEngine:
        state["result"] = {"error": "LlamaIndexEngine not available"}
        return state
    
    try:
        manager = LlamaIndexEngine(
            persist_directory="./rag_system/chroma_db",
            collection_name="multimodal_downloaded_data_with_embedding",
            user_id="default_user"
        )
        result = manager.query(state["query"], use_chat_history=True)
        state["result"] = result
    except Exception as e:
        print(f"❌ RAG error: {e}")
        state["result"] = {"error": str(e)}
    return state

def calculate_relevance(result: Any) -> str:
    # print("RESULT: ", result)
    if not result:
        return "low"
    
    try:
        max_relevance = 0
        for node in result:
            if hasattr(node, "score"):
                max_relevance = max(max_relevance, node.score)
            elif hasattr(node, "score"):
                max_relevance = max(max_relevance, node.score)
        return "high" if max_relevance > 0.5 else "low"
    except Exception as e:
        print(f"Error calculating relevance: {e}")
        return "low"

def user_choice_node(state: QueryState) -> QueryState:
    # Non-interactive (web/UI) mode: do NOT block
    if state.get("non_interactive"):
        if not state.get("user_choice") or state.get("user_choice") == "":
            state["result"] = {
                "status": "need_user_choice",
                "options": ["rag", "research"],
                "message": "Low relevance documents. Choose 'rag' to answer with current retrieval or 'research' for deeper research + fact check + PDF export.",
                "query": state["query"]
            }
            state["user_choice"] = "pending"
            return state
        # validate provided choice
        if state["user_choice"] not in ["rag", "research", "pending"]:
            state["user_choice"] = "research"
        return state
    
    # CLI / terminal mode (blocking)
    print("⚠️ Retrieved docs have low relevance.")
    choice = input("Do you want to continue with [rag] results or run [research]? ").strip().lower()
    if choice not in ["rag", "research"]:
        choice = "research"
    state["user_choice"] = choice
    return state

def pending_node(state: QueryState) -> QueryState:
    """Placeholder terminal node when waiting for frontend user choice."""
    return state

def research_agent(state: QueryState) -> QueryState:
    if not ResearchAgent:
        state["result"] = {"error": "ResearchAgent not available"}
        return state
    
    try:
        research_result = ResearchAgent().research_agent(state["query"])
        # Ensure research_data includes the original query for PDF export
        if isinstance(research_result, dict):
            research_result["original_query"] = state["query"]
        state["research_data"] = research_result
        state["result"] = state["research_data"]
    except Exception as e:
        print(f"❌ Research error: {e}")
        state["result"] = {"error": f"Research failed: {str(e)}"}
    return state

def fact_checker(state: QueryState) -> QueryState:
    if not FactCheckAgent:
        state["result"] = {"error": "FactCheckAgent not available"}
        return state
    
    try:
        research = state.get("research_data", state["result"])
        
        # The FactCheckAgent expects research_data to have 'query' and 'results' keys
        # Make sure the research data has the expected format
        if isinstance(research, dict) and "query" in research and "results" in research:
            # Pass research data as-is
            fact_check_data = research
        else:
            # Build compatible format for FactCheckAgent
            fact_check_data = {
                "query": state["query"],
                "results": research.get("results", []) if isinstance(research, dict) else [],
                "answer": research.get("answer", "") if isinstance(research, dict) else str(research)
            }

        raw_result = FactCheckAgent().fact_check_with_tavily_llm(fact_check_data)

        # Parse JSON string response if needed
        if isinstance(raw_result, str):
            try:
                raw_result = json.loads(raw_result)
            except json.JSONDecodeError:
                raw_result = {"llm_explanation": raw_result}

        # Ensure dict result and normalize
        if not isinstance(raw_result, dict):
            raw_result = {"llm_explanation": str(raw_result)}

        if not raw_result.get("statement") or raw_result.get("statement") == "None":
            raw_result["statement"] = state["query"]

        # Normalize score from various possible keys
        raw_score = raw_result.get("relevance_score") or raw_result.get("relevance") or raw_result.get("score")
        norm = _normalize_relevance_score(raw_score)
        if norm is not None:
            raw_result["relevance_score"] = norm
        else:
            # Do not force 0 if missing; leave it absent to avoid showing 0.00
            if "relevance_score" in raw_result and raw_result["relevance_score"] is None:
                del raw_result["relevance_score"]

        # Debug aid when zero score is suspicious
        if raw_result.get("relevance_score") == 0.0:
            try:
                keys = list(research.keys()) if isinstance(research, dict) else str(type(research))
                print(f"⚠️ Fact-check score is 0.0. research keys/type: {keys}, raw_score={raw_score}")
                print(f"   Query in fact_check_data: '{fact_check_data.get('query')}'")
                print(f"   Number of results: {len(fact_check_data.get('results', []))}")
            except Exception:
                pass

        state["fact_check_results"] = raw_result
        state["result"] = state["fact_check_results"]
    except Exception as e:
        print(f"❌ Fact checking error: {e}")
        state["result"] = {"error": f"Fact checking failed: {str(e)}"}
    return state

def export_agent(state: QueryState) -> QueryState:
    try:
        filename = "exported_research.pdf"
        agent = ExportPDFAgent()
        agent.export_pdf(state.get("research_data", {}), state["result"], filename)
        state["result"] = {
            "content": state["result"], 
            "pdf_exported": filename,
            "pdf_status": "Successfully exported research findings to PDF"
        }
        print(f"✅ PDF exported: {filename}")
    except Exception as e:
        print(f"❌ Export error: {e}")
        # Don't fail the entire pipeline if export fails
        if isinstance(state["result"], dict):
            state["result"]["export_error"] = str(e)
        else:
            state["result"] = {"content": state["result"], "export_error": str(e)}
    return state

def response_agent(state: QueryState) -> QueryState:
    """
    Final response agent to format the output.
    This can be customized to return JSON, text, or any other format.
    """
    client = state["llm"]

    # Preserve metadata from previous step (e.g., PDF export info)
    meta_pdf = {}
    payload_for_llm = state["result"]
    if isinstance(state["result"], dict):
        if "content" in state["result"]:
            payload_for_llm = state["result"]["content"]
        for k in ("pdf_exported", "pdf_status"):
            if state["result"].get(k):
                meta_pdf[k] = state["result"][k]

    system_prompt = f"""You are a knowledgeable AI assistant that can restructure and summarize structured data from AI agents.
        You will receive the user query and data from either a research agent, a billing agent or a cli agent.
        Your task is to format that input into the following JSON structure:
        ```json
        {{
            "response": "Explain what the input is about in an understanding and concise manner",
            "follow_up_questions": [
                "Follow-up question 1",
                "Follow-up question 2",
                "Follow-up question 3"
            ]
        }}
        ```
        When answering:
        - Give accurate, detailed responses. Be specific and informative in your responses. Write the response as if you were explaining the content of the given input.
        - Generate follow-up question suggestions after each response to keep the conversation going
        - Stick to the provided guideline and format
        - ALWAYS include the cli command in the response if it was generated
        - ALWAYS include the billing estimate in the response if it was generated
        - ALWAYS state that a research was conducted and the results were fact-checked when applicable
        - ALWAYS mention PDF export when pdf_exported field is present in the result
        - use bullet points for structured data, if applicable
        - NEVER mention the input or the agents since they are a part of the process and should not be included in the final output

        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User input: {state['query']} Agent: {state['classification']} result: {payload_for_llm}"}
        ],
        max_tokens=1500,
        temperature=0.3
    )
    raw = response.choices[0].message.content.strip()

    # Try to parse model output as JSON; fallback to plain text
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            parsed = {"response": raw, "follow_up_questions": []}
    except Exception:
        parsed = {"response": raw, "follow_up_questions": []}

    # Reattach metadata so UI can render a download button
    parsed.update(meta_pdf)
    state["result"] = parsed
    return state

# --- Routing ---
def route_after_classification(state: QueryState) -> str:
    return state["classification"]

def route_after_retrieval(state: QueryState) -> str:
    return "rag" if state["relevance"] == "high" else "user_choice"

def route_after_user_choice(state: QueryState) -> str:
    choice = state.get("user_choice", "")
    if choice in ["rag", "research"]:
        return choice
    return "pending"

# --- Graph ---
graph = StateGraph(QueryState)

graph.add_node("classify", classify_query)
graph.add_node("billing", billing_agent)
graph.add_node("cli", cli_agent)
graph.add_node("retrieval", retrieval_agent)
graph.add_node("rag", rag_agent)
graph.add_node("user_choice", user_choice_node)
graph.add_node("pending", pending_node)
graph.add_node("research", research_agent)
graph.add_node("fact_checker", fact_checker)
graph.add_node("export", export_agent)
# add the response/structurize node and attach it to appropriate branches
graph.add_node("response", response_agent)

graph.set_entry_point("classify")

# classification → billing/cli/knowledge
graph.add_conditional_edges("classify", route_after_classification, {
    "billing": "billing",
    "cli": "cli",
    "knowledge": "retrieval"
})

# retrieval → high relevance → rag / low relevance → ask user
graph.add_conditional_edges("retrieval", route_after_retrieval, {
    "rag": "rag",
    "user_choice": "user_choice"
})

# user choice → rag or research or pending
graph.add_conditional_edges("user_choice", route_after_user_choice, {
    "rag": "rag",
    "research": "research",
    "pending": "pending"
})

# pending is a terminal stop
graph.add_edge("pending", END)

# terminal paths
# route billing and cli to the response node, then to END (leave rag as direct END)
graph.add_edge("billing", "response")
graph.add_edge("cli", "response")
graph.add_edge("rag", END)

# research flow
graph.add_edge("research", "fact_checker")
graph.add_edge("fact_checker", "export")
# after export, route to response node to structurize/exported data
graph.add_edge("export", "response")

# final structurize node leads to END
graph.add_edge("response", END)

app = graph.compile()

# --- Non-blocking runner for UI ---
def process_query_step(user_query: str, user_choice: Optional[str] = None) -> dict:
    """Non-blocking version of the pipeline for web UI.
    Call 1: process_query_step(query) -> may return {status: need_user_choice}
    Call 2: process_query_step(query, user_choice='rag'|'research') -> final result
    (If relevance high or classification != knowledge, returns final result in one call.)
    """
    state: QueryState = {
        "query": user_query,
        "classification": "",
        "result": {},
        "relevance": "",
        "research_data": None,
        "fact_check_results": None,
        "user_choice": (user_choice or "").lower(),
        "llm": LLM_CLIENT,
        "non_interactive": True
    }
    try:
        out_state = app.invoke(state)
        return out_state["result"]
    except Exception as e:
        return {"error": str(e)}

# --- Runner ---
def process_query(user_query: str) -> dict:
    state: QueryState = {
        "query": user_query,
        "classification": "",
        "result": {},
        "relevance": "",
        "research_data": None,
        "fact_check_results": None,
        "user_choice": "",
        "llm": LLM_CLIENT,
        "non_interactive": False
    }
    try:
        return app.invoke(state)["result"]
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    tests = [
        # "How much does it cost to run an EC2 t3.medium instance?",
        # "What command can I use to create a new S3 bucket?",
        # "What is the latest information about AWS Lambda pricing?",
        "Research the benefits of microservices architecture"
    ]
    for q in tests:
        print(f"\nQuery: {q}")
        print(json.dumps(process_query(q), indent=2))
