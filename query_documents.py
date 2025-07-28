from openai import AzureOpenAI
from preprocessing.document_reader import DocumentReader
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DocumentChatBot:

    def __init__(self, user_id="default_user"):
        self.user_id = user_id
        self.project_id = "1"
        self.reader = DocumentReader(chroma_db_path="./chroma_db")
        from chat_history.mongo_chat_history import MongoDBChatHistoryManager
        self.chat_manager = MongoDBChatHistoryManager(db_name="chat_history_db", collection_name="conversations")
        self.session_id = self._initialize_session()
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )

    def _initialize_session(self):
        # Use chat_manager to create or get a session (MongoDB only)
        sessions = self.chat_manager.get_sessions(self.user_id, self.project_id)
        if sessions:
            return sessions[-1]["session_id"]  # Use the most recent session
        else:
            session = self.chat_manager.create_session(self.user_id, self.project_id)
            return session["session_id"]

    def _initialize_chat_manager(self):
        """Initialize the MongoDB chat history manager only."""
        from chat_history.mongo_chat_history import MongoDBChatHistoryManager
        return MongoDBChatHistoryManager(
            db_name="chat_history_db", collection_name="conversations"
        )
    
    def _initialize_openai_client(self):
        """Initialize Azure OpenAI client."""
        return AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
    
    def get_sessions(self):
        """Return all sessions for the current user as a list of dicts."""
        try:
            sessions = self.chat_manager.get_sessions(self.user_id, self.project_id)
            return sessions if sessions else []
        except Exception as e:
            print(f"Error retrieving sessions: {e}")
            return []

    def _get_conversation_history(self, query: str):
        """Get relevant conversation history based on storage type."""
        print("Searching for relevant conversation history...")
        relevant_history = self.chat_manager.find_most_relevant_conversation(
            query, session_id=self.session_id, n_results=5, max_tokens=500
        )
        if relevant_history:
            if isinstance(relevant_history, list):
                conversation_count = len(relevant_history)
                return relevant_history[0] if relevant_history else None
            else:
                conversation_count = relevant_history.count("Question:")
                if conversation_count == 0:
                    conversation_count = 1  # Assume at least one conversation if content exists
            
            print(f"📊 Retrieved {conversation_count} relevant conversation(s)")
            return relevant_history
        else:
            print("📊 No relevant conversation history found")
            return None
    
    def _get_document_context(self, query: str) -> str:
        """Retrieve and process document context."""
        print(f"Retrieving documents...")
        retrieved_content = self.reader.query_documents(
            query=query,
            collection_name="multimodal_downloaded_data_with_embedding",
            n_results=20
        )
        
        context_string = "\n".join(retrieved_content[:10])  # Use top 10 chunks
        print(f"Using {len(retrieved_content[:10])} document chunks")
        return context_string
    
    def _build_system_message(self, context_string: str, history_text: str = "") -> str:
        """Build the system message for the LLM."""
        
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
                - Stick to the provided guideline and format
                - If the question is vague or ambiguous, ask for clarification
                Context:
                {context_string}
                History:
                {history_text}
                If by any chance you find conflicting information in the context, use the most recent information to answer the question."""

    def _improve_query(self, query: str, is_interactive: bool = False) -> str:
        prompt = (
            "Rewrite the following question to be as clear, specific, and context-rich as possible for a document retrieval system. "
            "Do not answer the question, just rewrite it.\n"
            f"Original: {query}\nImproved:"
        )
        # user da izbere dali saka da go prasa negovoto prsanje ili podobrenoto
        try:
            completion = self.client.chat.completions.create(
                model="gpt-35-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that is a part of a RAG system. Your task is to rewrite vague or ambiguous questions to be more specific and helpful for document search."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=60,
            )
            improved = completion.choices[0].message.content.strip()
            if improved and improved.lower() != query.lower():
                print(f"Improved query: {improved}")
                if is_interactive:
                    print("Would you like to use this improved query? (yes/no)")
                    user_choice = input().strip().lower()
                    if user_choice in ['yes', 'y']: 
                        return improved
                    return query  # Use original query if user declines
                else:
                    return improved
        except Exception:
            pass
        return query

    def _generate_response(self, query: str, system_message: str) -> str:
        """Generate response using Azure OpenAI."""
        print("Generating response...")
        improved_query = self._improve_query(query)  
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": improved_query} #improved_query
            ],
            temperature=0.6,
            max_tokens=4000,
        )
        return completion.choices[0].message.content
    
    def _save_conversation(self, query: str, response: str):
        """Save conversation to chat history."""
        print("Saving conversation...")
        self.chat_manager.add_conversation(
            query=query, response=response, session_id=self.session_id
        )
    
    def chat_loop(self):
        """Main chat interaction loop."""
        print(f"Document Chat Bot initialized!")
        print(f"Storage: MongoDB")
        print("Commands: 'quit'")
        print("-" * 60)
        try:
            while True:
                query = input("\nEnter your question (or command): ").strip()
                if not query:
                    continue
                try:
                    print(f"\n🔍 Processing: '{query}'")
                    # Get conversation history and document context
                    relevant_history = self._get_conversation_history(query)
                    context_string = self._get_document_context(query)
                    # Prepare context for LLM
                    history_text = ""
                    if relevant_history:
                        history_text = f"\n\nRelevant Context:\n{relevant_history}"
                    # Generate and display response
                    system_message = self._build_system_message(context_string, history_text)
                    response = self._generate_response(query, system_message)
                    print("\nResponse:")
                    print(response)
                    # Save conversation
                    self._save_conversation(query, response)
                    print("-" * 60)
                except Exception as e:
                    print(f"Error: {e}")
                    print("Please try again.")
        except KeyboardInterrupt:
            print("\n\nSession interrupted by user")
        except EOFError:
            print("\n\nSession ended")

if __name__ == "__main__":
    
    bot = DocumentChatBot(user_id="6175")
    bot.chat_loop()
