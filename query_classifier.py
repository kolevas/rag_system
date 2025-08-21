from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
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
    if not result:
        return "low"
    
    try:
        max_relevance = 0
        for node in result:
            if hasattr(node, "relevance"):
                max_relevance = max(max_relevance, node.relevance)
            elif hasattr(node, "score"):
                max_relevance = max(max_relevance, node.score)
        return "high" if max_relevance > 0.5 else "low"
    except Exception as e:
        print(f"Error calculating relevance: {e}")
        return "low"

def user_choice_node(state: QueryState) -> QueryState:
    print("⚠️ Retrieved docs have low relevance.")
    choice = input("Do you want to continue with [rag] results or run [research]? ").strip().lower()
    if choice not in ["rag", "research"]:
        choice = "research"
    state["user_choice"] = choice
    return state

def research_agent(state: QueryState) -> QueryState:
    if not ResearchAgent:
        state["result"] = {"error": "ResearchAgent not available"}
        return state
    
    try:
        state["research_data"] = ResearchAgent().research_agent(state["query"])
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
        data = state.get("research_data", state["result"])
        state["fact_check_results"] = FactCheckAgent().fact_check_with_tavily_llm(data)
        state["result"] = state["fact_check_results"]
    except Exception as e:
        print(f"❌ Fact checking error: {e}")
        state["result"] = {"error": f"Fact checking failed: {str(e)}"}
    return state

def export_agent(state: QueryState) -> QueryState:
    if not ExportPDFAgent:
        state["result"] = {"error": "ExportPDFAgent not available"}
        return state
    
    try:
        filename = "exported_research.pdf"
        ExportPDFAgent().export_pdf(state.get("research_data", {}), state["result"], filename)
        state["result"] = {"content": state["result"], "pdf_exported": filename}
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
        - use bullet points for structured data, if applicable
        - NEVER mention the input or the agents since they are a part of the process and should not be included in the final output

        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User input: {state['query']} Agent: {state['classification']} result: {state['result']}"}
        ],
        max_tokens=1500,
        temperature=0.3
    )
    state["result"] = response.choices[0].message.content.strip()
    return state

# --- Routing ---
def route_after_classification(state: QueryState) -> str:
    return state["classification"]

def route_after_retrieval(state: QueryState) -> str:
    return "rag" if state["relevance"] == "high" else "user_choice"

def route_after_user_choice(state: QueryState) -> str:
    return state["user_choice"]

# --- Graph ---
graph = StateGraph(QueryState)

graph.add_node("classify", classify_query)
graph.add_node("billing", billing_agent)
graph.add_node("cli", cli_agent)
graph.add_node("retrieval", retrieval_agent)
graph.add_node("rag", rag_agent)
graph.add_node("user_choice", user_choice_node)
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

# user choice → rag or research
graph.add_conditional_edges("user_choice", route_after_user_choice, {
    "rag": "rag",
    "research": "research"
})

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
        "llm": LLM_CLIENT
    }
    try:
        return app.invoke(state)["result"]
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    tests = [
        "How much does it cost to run an EC2 t3.medium instance?",
        "What command can I use to create a new S3 bucket?",
        "What is the latest information about AWS Lambda pricing?",
        "Research the benefits of microservices architecture"
    ]
    for q in tests:
        print(f"\nQuery: {q}")
        print(json.dumps(process_query(q), indent=2))
