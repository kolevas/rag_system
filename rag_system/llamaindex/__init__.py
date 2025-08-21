"""
LlamaIndex Integration Package

Contains LlamaIndex-specific components for RAG functionality.
"""

# Import LlamaIndex components with error handling
try:
    from .llamaindex_engine import LlamaIndexEngine
except ImportError as e:
    print(f"⚠️  Could not import LlamaIndexEngine: {e}")
    LlamaIndexEngine = None

try:
    from .llamaindex_preprocessing import LlamaIndexPreprocessor, Config
except ImportError as e:
    print(f"⚠️  Could not import LlamaIndexPreprocessor: {e}")
    LlamaIndexPreprocessor = Config = None

__all__ = [
    'LlamaIndexEngine',
    'LlamaIndexPreprocessor',
    'Config'
]