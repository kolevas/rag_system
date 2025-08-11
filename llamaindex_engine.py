import chromadb
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LlamaIndexManager:
    def __init__(self, persist_directory="./chroma_db", collection_name="document_collection", 
                 user_id="default_user", project_id="1"):
        load_dotenv()
        
        # User and session management
        self.user_id = user_id
        self.project_id = project_id
        
        # Initialize chat history manager
        try:
            from chat_history.mongo_chat_history import MongoDBChatHistoryManager
            self.chat_manager = MongoDBChatHistoryManager(
                db_name="chat_history_db", 
                collection_name="conversations"
            )
            self.session_id = self._initialize_session()
            print(f"✅ MongoDB chat history initialized (session: {self.session_id})")
        except Exception as e:
            print(f"⚠️  Chat history disabled: {e}")
            self.chat_manager = None
            self.session_id = None
        
        # Setup embeddings - use local HuggingFace embeddings by default
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_length=512,
            device="cpu",
            trust_remote_code=False,
            normalize=True
        )
        Settings.embed_model = embed_model
    
        
        # Setup LLM - try Azure OpenAI, fall back to None if not available
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        
        print(f"🔧 Azure Config: endpoint={azure_endpoint}, deployment={deployment_name}, api_version={api_version}")
        print(f"🔧 API Key present: {bool(api_key)}")
        
        if api_key and azure_endpoint:
            try:
                # Create Azure OpenAI client manually (like your working example)
                self.azure_client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version
                )
                
                # Pass the Azure client to LlamaIndex
                Settings.llm = OpenAI(
                    model=deployment_name,  # Azure deployment name
                    openai_client=self.azure_client,  # Use our configured Azure client
                    temperature=0.1,
                    max_tokens=4000
                )
                print(f"✅ Azure OpenAI LLM configured: {deployment_name}")
            except Exception as e:
                print(f"⚠️  Could not configure Azure OpenAI LLM: {e}")
                print(f"⚠️  Error details: {type(e).__name__}: {str(e)}")
                Settings.llm = None
                self.azure_client = None
        else:
            print("⚠️  Azure OpenAI credentials not found. LLM features disabled.")
            Settings.llm = None
            self.azure_client = None

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self.chroma_collection = self.chroma_client.get_or_create_collection(self.collection_name)

        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        self.index = None
        self.query_engine = None

    def build_index(self, docs=None):
        """Build vector index and query engine, or load existing index"""
        
        # Try to load existing index first
        try:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context
            )
            print(f"✅ Loaded existing index from ChromaDB (collection: {self.collection_name})")
            
        except Exception as e:
            # If loading fails, create new index from documents
            if docs is None:
                raise ValueError("No existing index found and no documents provided to create new index")
            
            print(f"📝 No existing index found, creating new index from {len(docs)} documents...")
            self.index = VectorStoreIndex.from_documents(docs, storage_context=self.storage_context)
            print(f"✅ Created new index and saved to ChromaDB")
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,
            llm=Settings.llm
        )
  
    def query(self, question: str, use_chat_history: bool = True, save_conversation: bool = True):
        """Enhanced query with chat history and system prompts"""
        if not self.query_engine:
            try:
                self.build_index()  # Ensure index is built
            except ValueError as ve:
                print(f"❌ Error: {ve}")
        
        print(f"\n🔍 Processing: '{question}'")
        
        try:
            # Get conversation history if enabled
            relevant_history = None
            if use_chat_history:
                relevant_history = self._get_conversation_history(question)
            
            # Get retriever to access document context
            retriever = self.index.as_retriever(similarity_top_k=10)
            nodes = retriever.retrieve(question)
            
            print(f"📚 Found {len(nodes)} relevant document chunks")
            
            # Prepare context for LLM
            history_text = ""
            if relevant_history:
                history_text = f"\n\nRelevant History:\n{relevant_history}"
            
            # Build system message with retrieved context
            system_message = self._build_system_message(nodes, history_text)
            
            # Use the stored azure_client instead of trying to access it from Settings.llm
            if not self.azure_client:
                raise ValueError("Azure OpenAI client not available")
            
            print("🤖 Generating response with Azure OpenAI...")
            completion = self.azure_client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=4000,
            )
            response = completion.choices[0].message.content
            
            self._save_conversation(question, response)
            
            print("✅ Response generated successfully")
            return response
            
        except Exception as e:
            print(f"❌ Error during query: {e}")
            # Fallback to simple query
            response = self.query_engine.query(question)
            return str(response)

    def has_existing_index(self):
        """Check if ChromaDB collection has any documents"""
        try:
            count = self.chroma_collection.count()
            return count > 0
        except Exception:
            return False
    
    def _initialize_session(self):
        """Initialize or get existing session for chat history"""
        if not self.chat_manager:
            return None
            
        sessions = self.chat_manager.get_sessions(self.user_id, self.project_id)
        if sessions:
            return sessions[-1]["session_id"]  # Use the most recent session
        else:
            session = self.chat_manager.create_session(self.user_id, self.project_id)
            return session["session_id"]
    
    def _get_conversation_history(self, query: str):
        """Get relevant conversation history"""
        if not self.chat_manager or not self.session_id:
            return None
            
        print("🔍 Searching for relevant conversation history...")
        relevant_history = self.chat_manager.find_most_relevant_conversation(
            query, session_id=self.session_id, n_results=5, max_tokens=500
        )
        
        if relevant_history:
            if isinstance(relevant_history, list):
                conversation_count = len(relevant_history)
            else:
                conversation_count = relevant_history.count("Question:") or 1
            
            print(f"📊 Retrieved {conversation_count} relevant conversation(s)")
            return relevant_history
        else:
            print("📊 No relevant conversation history found")
            return None
    
    def _build_system_message(self, context_nodes: list, history_text: str = "") -> str:
        """Build the system message for the LLM"""
        
        # Extract content from nodes
        context_string = "\n".join([node.text for node in context_nodes])
        
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

    def _save_conversation(self, query: str, response: str):
        """Save conversation to chat history"""
        if self.chat_manager and self.session_id:
            print("💾 Saving conversation...")
            self.chat_manager.add_conversation(
                query=query, response=response, session_id=self.session_id
            )

    def get_sessions(self):
        """Return all sessions for the current user as a list of dicts"""
        if not self.chat_manager:
            return []
            
        try:
            sessions = self.chat_manager.get_sessions(self.user_id, self.project_id)
            return sessions if sessions else []
        except Exception as e:
            print(f"Error retrieving sessions: {e}")
            return []
    
    def simple_query(self, question: str):
        """Simple query without chat history for quick testing"""
        if not self.query_engine:
            raise ValueError("Query engine not initialized. Call build_index first.")
        
        response = self.query_engine.query(question)
        return str(response)

if __name__ == "__main__":
    
    manager = LlamaIndexManager(
        persist_directory="./chroma_db",  # Use your main ChromaDB
        collection_name="my_documents"   # Use existing collection
    )
    manager.build_index()  # Load existing index or create new one
    
    while True:
        question = input("Ask a question: ")
        if question.lower() in ["exit", "quit"]:
            break
        response = manager.query(question)
        print("\nResponse:")
        print(response)