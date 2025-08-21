"""
Main Application Package

This is the root package for the application containing:
- RAG System for document retrieval and question answering
- Various AI agents (billing, CLI, research)
- Query classification and routing system
"""

import sys
import os

# Add the project root to Python path for absolute imports
def setup_project_path():
    """Add project root to Python path"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root

# Setup path on import
PROJECT_ROOT = setup_project_path()

# Import main components with error handling
try:
    from .agents import BillingAgent, CLIAgent, research_pipeline
except ImportError as e:
    print(f"⚠️  Could not import agents: {e}")
    BillingAgent = CLIAgent = research_pipeline = None

try:
    from .rag_system import LlamaIndexEngine, get_available_rag_components
except ImportError as e:
    print(f"⚠️  Could not import RAG system: {e}")
    LlamaIndexEngine = get_available_rag_components = None

__all__ = [
    'BillingAgent',
    'CLIAgent', 
    'research_pipeline',
    'LlamaIndexEngine',
    'get_available_rag_components',
    'PROJECT_ROOT'
]

def check_system_status():
    """Check the status of all system components"""
    print("🔍 System Component Status:")
    print("=" * 40)
    
    # Check agents
    if BillingAgent:
        print("✅ Billing Agent available")
    else:
        print("❌ Billing Agent not available")
    
    if CLIAgent:
        print("✅ CLI Agent available")
    else:
        print("❌ CLI Agent not available")
    
    if research_pipeline:
        print("✅ Research Pipeline available")
    else:
        print("❌ Research Pipeline not available")
    
    # Check RAG system
    if LlamaIndexEngine:
        print("✅ RAG System available")
        if get_available_rag_components:
            components = get_available_rag_components()
            print(f"   Available components: {', '.join(components)}")
    else:
        print("❌ RAG System not available")
    
    print("=" * 40)

if __name__ == "__main__":
    check_system_status()
