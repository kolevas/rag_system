import chromadb
from chromadb.utils import embedding_functions
import time
from typing import List, Tuple, Dict

class ChatHistoryManager:
    def __init__(self, chroma_db_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.chat_collection = self.client.get_or_create_collection(
            name="chat_history",
            embedding_function=self.embedding_function
        )
    
    def add_conversation(self, query: str, response: str, session_id: str, timestamp: float = None):
        """Add a conversation pair to history for a specific session"""
        if timestamp is None:
            timestamp = time.time()
        
        combined_text = f"Q: {query}\nA: {response}"
        
        self.chat_collection.add(
            documents=[combined_text],
            metadatas=[{
                "query": query,
                "response": response,
                "timestamp": timestamp,
                "session_id": session_id
            }],
            ids=[f"{session_id}_conv_{int(timestamp * 1000)}"]
        )

    def find_most_relevant_conversation(self, current_query: str, session_id: str, n_results: int = 1, 
                                        min_relevance: float = 0.7, max_tokens: int = 2000) -> List[Dict]:
        """Find most relevant conversations using ChromaDB semantic search with fallback"""
        
        if self.chat_collection.count() == 0:
            return []
        
        search_results = max(n_results * 3, 15) 
        
        # First try: Semantic similarity search
        results = self.chat_collection.query(
            query_texts=[current_query],
            n_results=search_results,
            where={"session_id": session_id}
        )
        
        relevant_conversations = []
        if results and results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                distance = results.get('distances', [[1.0]])[0][i] if results.get('distances') else 1.0
                
                if distance <= min_relevance:
                    metadata = results['metadatas'][0][i]
                    relevant_conversations.append({
                        'query': metadata['query'],
                        'response': metadata['response'],
                        'timestamp': metadata['timestamp'],
                        'distance': distance,
                        'similarity_score': 1.0 - distance,  
                        'combined_text': doc
                    })
                else:
                    print(f"💡 Found conversation but relevance too low (distance: {distance:.3f}, threshold: {min_relevance})")
        
        # If no relevant conversations found, fall back to recent conversations
        if not relevant_conversations:
            print("No semantic search matches found. Checking recent conversations...")
            try:
                # Get recent conversations from the collection
                all_results = self.chat_collection.get(where={"session_id": session_id})
                
                if all_results and all_results['documents']:
                    # Sort by timestamp (most recent first)
                    recent_data = list(zip(all_results['documents'], all_results['metadatas']))
                    recent_data.sort(key=lambda x: x[1]['timestamp'], reverse=True)
                    
                    # Take up to 3 most recent
                    for doc, metadata in recent_data[:min(3, n_results)]:
                        relevant_conversations.append({
                            'query': metadata['query'],
                            'response': metadata['response'],
                            'timestamp': metadata['timestamp'],
                            'distance': 0.7,  # Neutral distance for fallback
                            'similarity_score': 0.3,  # Lower score for fallback
                            'combined_text': doc
                        })
                    
                    print(f"Using {len(relevant_conversations)} recent conversations as fallback context")
                else:
                    print("No conversations found for this session")
                    return []
            except Exception as e:
                print(f"Error retrieving recent conversations: {e}")
                return []
        else:
            print(f"Found {len(relevant_conversations)} conversations via semantic search")
        
        # Filter by token budget
        from token_utils import count_tokens
        filtered_conversations = []
        total_tokens = 0
        
        for conv in relevant_conversations:
            conv_tokens = count_tokens(conv['query']) + count_tokens(conv['response'])
            
            if total_tokens + conv_tokens <= max_tokens:
                filtered_conversations.append(conv)
                total_tokens += conv_tokens
            else:
                break
                
            if len(filtered_conversations) >= n_results:
                break
        
        # Sort by similarity score and recency
        filtered_conversations.sort(key=lambda x: (x['similarity_score'], x['timestamp']), reverse=True)
        
        print(f"Returning {len(filtered_conversations)} conversations ({total_tokens} tokens)")
        return filtered_conversations
    
    def get_recent_conversations(self, session_id: str, n_recent: int = 5) -> List[Dict]:
        """Get recent conversations for a session ordered by timestamp"""
        results = self.chat_collection.get(where={"session_id": session_id}, limit=n_recent, sort=[("timestamp", -1)])
        conversations = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents']):
                metadata = results['metadatas'][i]
                conversations.append({
                    'query': metadata['query'],
                    'response': metadata['response'],
                    'timestamp': metadata['timestamp'],
                    'combined_text': doc
                })
        return conversations

    def clear_history(self, session_id: str = None):
        """Clear chat history for a session or all if session_id is None"""
        if session_id:
            self.chat_collection.delete(where={"session_id": session_id})
            print(f"✅ Chat history cleared for session '{session_id}'")
        else:
            self.chat_collection.delete()
            print(f"✅ Chat history cleared for all sessions")
    
