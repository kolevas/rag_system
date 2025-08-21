import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from openai import AzureOpenAI
import time
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

# Import the preprocessor with relative import
from rag_system.llamaindex.llamaindex_preprocessing import LlamaIndexPreprocessor, Config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LlamaIndexEngine:
    """Handles querying and chat functionality for LlamaIndex"""

    def __init__(self, persist_directory="./chroma_db", collection_name="multimodal_downloaded_data_with_embedding", 
                 user_id="default_user", project_id="1"):
        self.config = Config()
        self.user_id, self.project_id = user_id, project_id
        
        # Initialize preprocessor
        self.preprocessor = LlamaIndexPreprocessor(persist_directory, collection_name)
        
        # Initialize query components
        self._setup_llm()
        self._setup_chat_history()
        
        self.query_engine = None
        self._initialize_query_engine()

    def _log_setup(self, component: str, success: bool, error: Exception = None):
        """Unified logging for component setup"""
        if success:
            print(f"✅ {component} configured")
        else:
            print(f"⚠️  {component} setup failed: {error}")

    def _setup_llm(self):
        """Setup Azure OpenAI LLM with error handling"""
        if not self.config.azure_available:
            self._log_setup("Azure OpenAI", False, "Credentials not found")
            Settings.llm = self.azure_client = None
            return

        try:
            self.azure_client = AzureOpenAI(
                azure_endpoint=self.config.azure_endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version
            )
            Settings.llm = OpenAI(
                model=self.config.deployment_name,
                openai_client=self.azure_client,
                temperature=0.1, max_tokens=4000
            )
            self._log_setup("Azure OpenAI", True)
        except Exception as e:
            self._log_setup("Azure OpenAI", False, e)
            Settings.llm = self.azure_client = None

    def _setup_chat_history(self):
        """Setup chat history with graceful fallback"""
        try:
            # Use relative import since we're in the rag_system package
            from rag_system.chat_history.mongo_chat_history import MongoDBChatHistoryManager
            self.chat_manager = MongoDBChatHistoryManager(db_name="chat_history_db", collection_name="conversations")
            self.session_id = self._initialize_session()
            self._log_setup("MongoDB chat history", True)
        except ImportError as e:
            print(f"⚠️  MongoDB chat history not available: {e}, using in-memory fallback")
            self.chat_manager = None
            self.session_id = None
            self.chat_history = []  # Simple in-memory storage
        except Exception as e:
            self._log_setup("Chat history", False, e)
            self.chat_manager = self.session_id = None
            self.chat_history = []  # Fallback to in-memory

    def _initialize_query_engine(self):
        """Initialize query engine if index is available"""
        # Try to build/load index first
        if not self.preprocessor.index:
            self.preprocessor.build_index()
        
        if self.preprocessor.index:
            self.query_engine = self.preprocessor.index.as_query_engine(similarity_top_k=3, llm=Settings.llm)
            print("✅ Query engine initialized")
        else:
            print("⚠️  No index available for query engine")

    def build_index(self, doc_directory=None) -> bool:
        """Build index using preprocessor and initialize query engine"""
        result = self.preprocessor.build_index(doc_directory)
        if result:
            self._initialize_query_engine()
        return result

    def has_existing_index(self) -> bool:
        """Check if ChromaDB collection has any documents"""
        try:
            return self.chroma_collection.count() > 0
        except Exception:
            return False

    # Querying methods
    
    def query(self, question: str, use_chat_history: bool = True, save_conversation: bool = True) -> str:
        start_time = time.time()
        """Enhanced query with chat history and system prompts"""
        if not self.query_engine and not self.build_index():
            return "❌ No index available and unable to create one. Please ensure documents are available."

        print(f"\n🔍 Processing: '{question}'")

        try:
            # Get conversation history and document context
            relevant_history = self._get_conversation_history(question) if use_chat_history else None
            nodes = self.get_document_context(question) if self.preprocessor.index else []
            #if the relevance is low call agents to do research
            context_chunks = [node.text for node in nodes[:10]]
            context_string = "\n".join(context_chunks) if context_chunks else "No relevant documents found."
            # Build system message and generate response
            history_text = f"\n\nRelevant History:\n{relevant_history}" if relevant_history else ""
            system_message = self._build_system_message(context_string, history_text)
            response = self._generate_response(question, system_message)
            
            if save_conversation:
                self._save_conversation(question, response)
            
            print("✅ Response generated successfully")
            print(f"Response time: {time.time() - start_time:.2f} seconds")
            return response
        except Exception as e:
            print(f"❌ Error during query: {e}")
            # Fallback to simple LlamaIndex query
            try:
                response = self.query_engine.query(question)
                return str(response)
            except Exception as fallback_error:
                return f"❌ Error: {fallback_error}"

    def get_document_context(self, query: str) -> str:
        """Retrieve and process document context using LlamaIndex retriever"""
        print(f"🔍 Retrieving documents...")
        retriever = self.preprocessor.index.as_retriever(similarity_top_k=10)
        nodes = retriever.retrieve(query)
        return nodes

    def _improve_query(self, query: str, is_interactive: bool = False) -> str:
        """Improve query for better retrieval"""
        if not self.azure_client:
            return query

        prompt = (f"Rewrite the following question to be as clear, specific, and context-rich as possible for a document retrieval system. "
                 f"Do not answer the question, just rewrite it.\nOriginal: {query}\nImproved:")

        try:
            completion = self.azure_client.chat.completions.create(
                model=self.config.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an assistant that is a part of a RAG system. Your task is to rewrite vague or ambiguous questions to be more specific and helpful for document search."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3, max_tokens=60,
            )
            improved = completion.choices[0].message.content.strip()
            if improved and improved.lower() != query.lower():
                print(f"💡 Improved query: {improved}")
                if is_interactive:
                    print("Would you like to use this improved query? (yes/no)")
                    user_choice = input().strip().lower()
                    return improved if user_choice in ['yes', 'y'] else query
                return improved
        except Exception as e:
            print(f"⚠️  Query improvement failed: {e}")
        return query

    def _generate_response(self, query: str, system_message: str) -> str:
        """Generate response using Azure OpenAI"""
        if not self.azure_client:
            raise ValueError("Azure OpenAI client not available")

        print("🤖 Generating response with Azure OpenAI...")
        improved_query = self._improve_query(query)
        
        completion = self.azure_client.chat.completions.create(
            model=self.config.deployment_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": improved_query}
            ],
            temperature=0.6, max_tokens=4000,
        )
        return completion.choices[0].message.content

    def _build_system_message(self, context_string: str, history_text: str = "") -> str:
        """Build the system message for the LLM"""
        return f"""You are a knowledgeable AI assistant that can answer questions about literature, science, and various other topics. 
        When answering:
        - Use the provided context to give accurate, detailed responses. Be specific and informative in your responses
        - If previous conversation history is available, reference it when relevant
        - Use ONLY the provided context and conversation history to answer questions. If you don't know the answer state that.
        - Generate follow-up question suggestions after each response to keep the conversation going
        - Your responses should look like this:
        ```json
        {{
            "response": "Your answer here",
            "follow_up_questions": [
                "Follow-up question 1",
                "Follow-up question 2",
                "Follow-up question 3"
            ]
        }}
        ```
        - You MUST ALWAYS stick to the provided guideline and format for your responses
        - If the question is vague or ambiguous, ask for clarification
        
        Context:
        {context_string}
        
        {history_text}
        
        If by any chance you find conflicting information in the context, use the most recent information to answer the question."""

    def _initialize_session(self) -> Optional[str]:
        """Initialize or get existing session for chat history"""
        if not self.chat_manager:
            return None
        sessions = self.chat_manager.get_sessions(self.user_id, self.project_id)
        if sessions:
            return sessions[-1]["session_id"]
        session = self.chat_manager.create_session(self.user_id, self.project_id)
        return session["session_id"]

    def _get_conversation_history(self, query: str) -> Optional[str]:
        """Get relevant conversation history"""
        if self.chat_manager and self.session_id:
            print("🔍 Searching for relevant conversation history...")
            try:
                relevant_history = self.chat_manager.find_most_relevant_conversation(
                    query, session_id=self.session_id, n_results=5, max_tokens=500
                )
                if relevant_history:
                    conversation_count = len(relevant_history) if isinstance(relevant_history, list) else relevant_history.count("Question:") or 1
                    print(f"📊 Retrieved {conversation_count} relevant conversation(s)")
                    return relevant_history
            except Exception as e:
                print(f"⚠️  Error retrieving chat history: {e}")
        
        # Fallback to in-memory chat history
        if hasattr(self, 'chat_history') and self.chat_history:
            recent_history = self.chat_history[-3:]  # Get last 3 conversations
            return "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in recent_history])
        
        print("📊 No conversation history available")
        return None

    def _save_conversation(self, query: str, response: str):
        """Save conversation to chat history"""
        if self.chat_manager and self.session_id:
            try:
                print("💾 Saving conversation...")
                self.chat_manager.add_conversation(query=query, response=response, session_id=self.session_id)
            except Exception as e:
                print(f"⚠️  Error saving to MongoDB: {e}")
        
        # Always save to in-memory backup
        if hasattr(self, 'chat_history'):
            self.chat_history.append({"question": query, "answer": response})
            # Keep only last 10 conversations
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]

    def get_sessions(self) -> List[Dict[str, Any]]:
        """Return all sessions for the current user as a list of dicts"""
        if not self.chat_manager:
            return []
        try:
            return self.chat_manager.get_sessions(self.user_id, self.project_id) or []
        except Exception as e:
            print(f"Error retrieving sessions: {e}")
            return []

    def simple_query(self, question: str) -> str:
        """Simple query without chat history for quick testing"""
        if not self.query_engine:
            raise ValueError("Query engine not initialized. Call build_index first.")
        response = self.query_engine.query(question)
        return str(response)

    def chat_loop(self):
        """Main chat interaction loop"""
        print(f"LlamaIndex Document Chat Bot initialized!")
        print(f"Storage: ChromaDB + MongoDB")
        print(f"Collection: {self.collection_name}")
        print("Commands: 'quit'")
        print("-" * 60)

        try:
            while True:
                
                query = input("\nEnter your question (or command): ").strip()
                if not query:
                    continue
                if query.lower() in ["quit", "exit"]:
                    break
                try:
                    start_time = time.time()
                    response = self.query(query)
                    print("\nResponse:")
                    print(response)
                    print("-" * 60)
                    print(f"Response time: {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    print(f"Error: {e}")
                    print("Please try again.")
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended")

if __name__ == "__main__":
    manager = LlamaIndexEngine(
        persist_directory="./chroma_db",
        collection_name="multimodal_downloaded_data_with_embedding",
        user_id="default_user"
    )
    manager.chat_loop()