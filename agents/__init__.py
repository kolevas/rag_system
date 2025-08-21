"""
Agents Package

This package contains various AI agents for different tasks including:
- Billing estimation
- CLI command generation  
- Knowledge pipeline (research, fact-checking, export)
"""

# Import main agents with error handling
try:
    from .billing_agent import BillingAgent
except ImportError as e:
    print(f"⚠️  Could not import BillingAgent: {e}")
    BillingAgent = None

try:
    from .cli_agent import CLIAgent
except ImportError as e:
    print(f"⚠️  Could not import CLIAgent: {e}")
    CLIAgent = None

try:
    from .knowlege_pipeline import research_pipeline, ResearchAgent, FactCheckAgent, ExportPDFAgent
except ImportError as e:
    print(f"⚠️  Could not import knowledge pipeline: {e}")
    research_pipeline = ResearchAgent = FactCheckAgent = ExportPDFAgent = None

# Export available components
__all__ = [
    'BillingAgent',
    'CLIAgent', 
    'research_pipeline',
    'ResearchAgent',
    'FactCheckAgent', 
    'ExportPDFAgent'
]

def get_available_agents():
    """Return a list of available agent classes"""
    available = []
    if BillingAgent:
        available.append('BillingAgent')
    if CLIAgent:
        available.append('CLIAgent')
    if ResearchAgent:
        available.append('ResearchAgent')
    if FactCheckAgent:
        available.append('FactCheckAgent')
    if ExportPDFAgent:
        available.append('ExportPDFAgent')
    return available