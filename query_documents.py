import re 
import os
from openai import AzureOpenAI
from preprocessing.document_reader import DocumentReader 
from chat_history.chroma_chat_history import ChatHistoryManager
from user_preferences import UserPreferences
import time
import uuid
from token_utils import count_tokens

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def chat(history_type="chroma"):
   
    reader = DocumentReader(chroma_db_path="./chroma_db")
    user_prefs = UserPreferences()

    session_id = str(uuid.uuid4())
    print(f"Generated session ID")

    if history_type.lower() == "postgres":
        print("Using PostgreSQL for chat history storage")
        from chat_history.postgres_chat_history import PostgreSQLChatHistoryManager
        chat_manager = PostgreSQLChatHistoryManager(
            db_name="chat_history_db",
            db_user="postgres",
            db_password="postgres"
        )
    elif history_type.lower() == "mongo":
        print("Using MongoDB for chat history storage")
        from chat_history.mongo_chat_history import MongoDBChatHistoryManager
        chat_manager = MongoDBChatHistoryManager(
            db_name="chat_history_db",
            collection_name="conversations"
        )
    else:
        print("Using ChromaDB for chat history storage")
        chat_manager = ChatHistoryManager(chroma_db_path="./chroma_db")
    
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") 
    )

    print(f"RAG System with User Preferences ({history_type.upper()} History)")
    print("Commands: 'prefs' = setup preferences, 'show' = show current preferences, 'history' = recent history,  'exit' = quit")
    user_prefs.show_current_preferences()

    while True:
        query = input("Enter your question (or command): ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Clearing chat history...")
            chat_manager.clear_history(session_id=session_id)
            print("Chat history cleared. Goodbye!")
            break
        elif query.lower() == 'prefs':
            user_prefs.interactive_setup()
            continue
        elif query.lower() == 'show':
            user_prefs.show_current_preferences()
            continue
        elif query.lower() == 'history':
            print("\nRecent Conversations:")
            recent = chat_manager.get_recent_conversations(session_id=session_id, n_recent=5)
            for i, conv in enumerate(recent, 1):
                print(f"{i}. Q: {conv['query'][:60]}...")
                print(f"   A: {conv['response'][:80]}...")
                print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(conv['timestamp']))}")
                print()
            continue
        
        if history_type.lower() == "mongo":
            print(f"Searching for relevant conversation history...")
            conversation_summary = chat_manager.find_most_relevant_conversation(query, session_id=session_id, n_results=5, max_tokens=500)
            if conversation_summary and conversation_summary != "No conversations found for this session.":
                print(f"Generated conversation summary: {conversation_summary[:100]}...")
                relevant_history = conversation_summary  # Store as string for later use
            else:
                print("No conversation history found for summarization")
                relevant_history = None
        else:
            print(f"Searching for relevant conversation history...")
            relevant_history = chat_manager.find_most_relevant_conversation(query, session_id=session_id, n_results=5, max_tokens=500)
            if relevant_history:
                print(f"Found {len(relevant_history)} relevant conversation(s)")
                print(f"   Best match similarity: {relevant_history[0].get('similarity_score', 0):.3f}")
            else:
                print("No relevant conversation history found")
        
        print(f"Retrieving documents for query: '{query}'...")
        retrieved_content = reader.query_documents(
            query=query,
            collection_name="multimodal_downloaded_data_with_embedding",
            n_results=30
        )
        
        max_total_tokens = 128000  
        max_response_tokens = 40000
        prompt_overhead_tokens = 500  
        available_tokens = max_total_tokens - max_response_tokens - prompt_overhead_tokens
        context_token_budget = int(available_tokens * 0.8)  
        history_token_budget = available_tokens - context_token_budget 
        
        context_chunks = []
        current_context_tokens = 0
        for chunk in retrieved_content:
            chunk_tokens = count_tokens(chunk)
            if current_context_tokens + chunk_tokens > context_token_budget:
                break
            context_chunks.append(chunk)
            current_context_tokens += chunk_tokens
        context_string = "\n".join(context_chunks)
        print(f"Document retrieval complete. Using {len(context_chunks)} chunks, {current_context_tokens} tokens for context.")

        if relevant_history:
            if history_type.lower() == "mongo":
                history_text = f"\n\nConversation Summary:\n{relevant_history}"
            else:
                history_text = f"\n\nRelevant previous conversation:\nQ: {relevant_history[0]['query']}\nA: {relevant_history[0]['response']}"
            
            history_tokens = count_tokens(history_text)
            if history_tokens <= history_token_budget:
                context_string += history_text
                print(f"Added conversation history to context ({history_tokens} tokens)")
            else:
                print(f"Skipped adding history: {history_tokens} tokens exceeds history budget of {history_token_budget}")
        
        preference_instructions = user_prefs.get_system_prompt_addition()
        
        if not context_string:
            print(" No relevant documents available for context. LLM will answer based on its general knowledge.")
            system_message_content = (
                "You are a helpful assistant. Answer the user's question to the best of your ability. "
                "If you cannot find relevant information, state that.\n\n"
                f"User Preferences: {preference_instructions}"
            )
        else:
            system_message_content = (
                "You are a helpful AI assistant that answers questions based on provided document context. "
                "Instructions:"
                "- Use only the information from the provided context to answer questions"
                "- If the context doesn't contain relevant information, say 'I don't have enough information in the provided context to answer that question'"
                "- Be concise and factual"
                "- Include specific details from the context when relevant"
                "- Do not make up information beyond what's provided"
                "Format your responses clearly and cite relevant parts of the context when possible."
                f"User Preferences: {preference_instructions}\n\n"
                f"Context:\n{context_string}"
            )

        user_message_content = f"Question: {query}"

        print("Generating response...")
        completion = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_message_content}
            ],
            temperature=0.6, 
            max_tokens=1000, 
        )

        print("\nLLM Response:")
        response = completion.choices[0].message.content
        print(response)
        
        print(f"Saving conversation to {history_type.upper()} history...")
        chat_manager.add_conversation(query=query, response=response, session_id=session_id)
        
        print(f"Storage: {history_type.upper()}, Context: Retrieved from ChromaDB")
        print("-" * 80)

if __name__ == "__main__":
    chat(history_type="mongo") # or chat(history_type="postgres") or chat(history_type="chroma")