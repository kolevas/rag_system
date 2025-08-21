import os
import requests
from dotenv import load_dotenv
import asyncio
from openai import AzureOpenAI
from tavily import TavilyClient
from typing import List, Dict, Any, Optional, TypedDict, Literal
# --- Configuration ---



class ResearchAgent:
    def __init__(self):
        load_dotenv()
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.azure_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-01"
        )

    # --- Function to query Tavily ---
    def query_tavily(self, search_query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Queries Tavily's API and returns a list of search results.
        """
        response = self.tavily_client.search(query=search_query, limit=max_results)
        # print("Response from Tavily:", response)
        return  response

    def stringify_tavily_response(self, response) -> str:
        """
        Convert Tavily response to a readable string format for Azure AI processing
        """
        if not response or 'results' not in response:
            return "No search results available."
        
        # Option 2: Custom formatted string
        stringified = f"""
            SEARCH QUERY: {response.get('query', 'Unknown')}
            RESPONSE TIME: {response.get('response_time', 'Unknown')} seconds
            RESULTS FOUND: {len(response.get('results', []))}

            SEARCH RESULTS:
            """
                    
        for i, result in enumerate(response.get('results', []), 1):
                        stringified += f"""
            {i}. TITLE: {result.get('title', 'No title')}
            URL: {result.get('url', 'No URL')}
            CONTENT: {result.get('content', 'No content')}
            RELEVANCE SCORE: {result.get('score', 'Unknown')}
            ---
            """
        return stringified.strip()
    
    # --- Function to summarize research using LLM ---
    def summarize_results(self, results, topic):

        combined_text = self.stringify_tavily_response(results)
        response = self.azure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system", "content": "You are a helpful assistant that summarizes research findings. Summarize the findings in a concise and informative way. Provide a clear summary highlighting the main points."
            },
            {"role": "user", "content": combined_text}],
            temperature=0.3
        )
        summary = response.choices[0].message.content.strip()
        print(f"Summary for topic '{topic}':\n{summary}")
        return summary     

    # --- Main function ---
    def research_agent(self, topic):
        print(f"🔎 Researching: {topic}")
        
        # Step 1: Fetch data from Tavily
        results = self.query_tavily(topic)

        # print(f"Found results:", results)

        # Step 2: Summarize using LLM or stringify
        # summary = self.summarize_results(results, topic)
        # summary = self.stringify_tavily_response(results)
        # return summary
        return results

# --- Example usage ---
if __name__ == "__main__":
    topic = "Latest AI trends in healthcare"
    agent = ResearchAgent()
    summary = agent.research_agent(topic)
    print("\n📝 Summary:\n")
    print(summary)
