import os
import uuid
import hashlib
from typing import Optional
from openai import AzureOpenAI
from preprocessing.document_reader import DocumentReader 
from chat_history.mongo_chat_history import MongoDBChatHistoryManager
from token_utils import count_tokens
from dotenv import load_dotenv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

class DocumentChatBot:
    """Simplified document chat bot with MongoDB storage backend."""
    
    def __init__(self, entra_oid="default_user", mongo_uri=None, db_name="chat_history_db", 
                 collection_name="conversations"):
        self.entra_oid = entra_oid
        
        # Model configuration
        self.model_name = "gpt-4o"
        self.max_context_tokens = 20000    # GPT-4o context window decreased for demonstration purposes
        self.max_response_tokens = 4000    # Reserve tokens for response
        self.safety_buffer = 500           # Safety buffer for token counting variations
        
        # Generate session ID
        session_base = f"{entra_oid}_{uuid.uuid4().hex[:8]}"
        self.session_id = hashlib.md5(session_base.encode()).hexdigest()[:24]
        
        # Initialize components
        print("Using MONGODB for chat history storage")
        self.reader = DocumentReader(chroma_db_path="./chroma_db")
        self.chat_manager = MongoDBChatHistoryManager(
            mongo_uri=mongo_uri,
            db_name=db_name,
            collection_name=collection_name
        )
        
        self.client = self._initialize_openai_client()
    
    def _initialize_openai_client(self):
        """Initialize Azure OpenAI client."""
        return AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
    
    def _get_document_context(self, query: str, max_chunks: int = 10) -> str:
        """Retrieve document context with a reasonable chunk limit."""
        retrieved_content = self.reader.get_document_content(
            query=query,
            collection_name="multimodal_downloaded_data_with_embedding",
            n_results=13
        ) 

        context_parts = retrieved_content[:max_chunks]
        context_string = "\n".join(context_parts)
        return context_string
    
    def _calculate_remaining_tokens_for_history(self, query: str, context_string: str) -> int:
        """Calculate how many tokens remain for conversation history after allocating for documents."""
        
        query_tokens = count_tokens(query)
        base_system_with_context = self._build_system_message(context_string, "")
        system_tokens = count_tokens(base_system_with_context)
        used_tokens = query_tokens + system_tokens + self.max_response_tokens + self.safety_buffer
        remaining_tokens = self.max_context_tokens - used_tokens
        history_tokens = max(remaining_tokens, 200)  
        
        print(f"🔢 Used={used_tokens}, Remaining_for_history={history_tokens}")
        
        return history_tokens

    def _get_conversation_history(self, query: str, available_history_tokens: int) -> Optional[str]:
        """Get relevant conversation history from MongoDB within token limit."""
        relevant_history = self.chat_manager.find_most_relevant_conversation(
            query, session_id=self.session_id, n_results=5, max_tokens=available_history_tokens
        )
        if relevant_history:
            actual_tokens = count_tokens(relevant_history)
            print(f"📊 Retrieved conversation history ({actual_tokens} tokens)")
            return relevant_history
        else:
            print("📊 No conversation history found")
            return None

    def _build_system_message(self, context_string: str, history_text: str = "") -> str:
        """Build the system message for the LLM."""
        base_prompt = """You are a knowledgeable AI assistant that can answer questions about literature and economics. 
                When answering:
                - Use ONLY the provided context to give accurate, detailed responses
                - If previous conversation history is available, reference it when relevant
                - Be specific and informative in your responses
                - Use ONLY the provided context and conversation history to answer questions. If you don't know the answer state that."""
        
        full_context = context_string + history_text
        
        return f"""{base_prompt}
                
                Context and Previous Conversations:
                {full_context}"""
    
    def _generate_response(self, query: str, system_message: str) -> str:
        """Generate response using Azure OpenAI with calculated token limits."""
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ],
            temperature=0.6,
            max_tokens=self.max_response_tokens,
        )
        return completion.choices[0].message.content
    
    def _save_conversation(self, query: str, response: str):
        """Save conversation to MongoDB chat history."""
        self.chat_manager.add_conversation(
            query=query, response=response, session_id=self.session_id
        )
    
    def process_query(self, query: str) -> str:
        """
        Process a single query with intelligent token management.
        
        Flow:
        1. Retrieve document context first (top N chunks)
        2. Calculate remaining tokens for history based on actual document usage
        3. Retrieve conversation history with the calculated token limit
        4. Generate response with all context
        """
        try:
            print(f"🔍 Processing: '{query}'")
            
            context_string = self._get_document_context(query, max_chunks=10)
            history_tokens = self._calculate_remaining_tokens_for_history(query, context_string)
            relevant_history = self._get_conversation_history(query, history_tokens)
            history_text = ""
            if relevant_history:
                history_text = f"\n\nRelevant Previous Conversations:\n{relevant_history}"
            system_message = self._build_system_message(context_string, history_text)
            response = self._generate_response(query, system_message)
            self._save_conversation(query, response)
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            print(error_msg)
            return error_msg

    def cleanup_session(self):
        """Clean up the current session."""
        if hasattr(self.chat_manager, 'close'):
            self.chat_manager.close()
        print("Session cleaned up")
    
    def chat_loop(self):
        """Main chat interaction loop."""
        print(f"Document Chat Bot initialized!")
        print(f"Storage: MONGODB")
        print(f"Model: {self.model_name} (Context: {self.max_context_tokens:,} tokens)")
        print("Commands: 'quit'")
        print("-" * 60)
        
        try:
            while True:
                query = input("\nEnter your question (or command): ").strip()
                if not query:
                    continue
                
                try:
                    response = self.process_query(query)
                    print("\nResponse:")
                    print(response)
                    print("-" * 60)
                    
                except Exception as e:
                    print(f"Error: {e}")
                    print("Please try again.")
        
        except KeyboardInterrupt:
            print("\n\nSession interrupted by user")
            self.cleanup_session()
        except EOFError:
            print("\n\nSession ended")
            self.cleanup_session()


if __name__ == "__main__":
    # Simple usage examples:
    # Default: chat()
    # Custom user: chat("user_123")
    # Custom MongoDB: chat("user_123", "mongodb://localhost:27017", "my_chat_db", "my_conversations")
    
    bot = DocumentChatBot(
        entra_oid="test_user_12345",
        mongo_uri=None,  
        db_name="chat_history_db",
        collection_name="conversations"
    )
    bot.chat_loop()
