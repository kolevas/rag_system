import pymongo
from pymongo import MongoClient
import time
from typing import List, Dict, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
from token_utils import count_tokens
load_dotenv()

class MongoDBChatHistoryManager:
    def __init__(self, mongo_uri=None, db_name="chat_history_db", collection_name="conversations"):
        
        self.mongo_uri = mongo_uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
        self.db_name = db_name
        self.collection_name = collection_name
        
        print(f"Connecting to MongoDB: {self.mongo_uri}")
        
        # Initialize MongoDB client
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        
        # Test connection and setup
        self._test_connection()
        self._setup_collection()
    
    def _test_connection(self):
        """Test MongoDB connection"""
        try:
            # Ping the database
            self.client.admin.command('ping')
            print("MongoDB connection successful")
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            raise
    
    def _setup_collection(self):
        """Setup collection with proper indexes"""
        try:
            # Create indexes for better performance
            self.collection.create_index([("session_id", 1), ("timestamp", -1)])
            self.collection.create_index([("session_id", 1), ("type", 1)])
            self.collection.create_index([("timestamp", -1)])
            
            print("MongoDB collection indexes created/verified")
        except Exception as e:
            print(f"Warning: Could not create indexes: {e}")
    
    def add_conversation(self, query: str, response: str, session_id: str, timestamp: float = None):
        """Add a conversation pair to history for a specific session"""
        if timestamp is None:
            timestamp = time.time()
        
        # Create document
        document = {
            "_id": f"{session_id}_conv_{int(timestamp * 1000)}",
            "query": query,
            "response": response,
            "session_id": session_id,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp),
            "type": "conversation" 
        }
        
        try:
            # Insert or replace if exists
            self.collection.replace_one(
                {"_id": document["_id"]}, 
                document, 
                upsert=True
            )
            print(f"Added conversation to session '{session_id}'")
        except Exception as e:
            print(f"Error adding conversation: {e}")
            raise
    
    def find_most_relevant_conversation(self, current_query: str, session_id: str, n_results: int = 1, 
                                        min_relevance: float = 0.7, max_tokens: int = 2000) -> List[Dict]:
        print(f"Using recent conversations for context (max_tokens={max_tokens})...")
        return self.get_recent_conversations(session_id, n_results, max_tokens)
    
    def get_recent_conversations(self, session_id: str, n_recent: int = 5, max_tokens = 3000) -> List[Dict]:
        try:
            conversations = list(self.collection.find(
                {"session_id": session_id, "type": "conversation"}
            ).sort("timestamp", -1).limit(n_recent * 4))  
            
            print(f"Found {len(conversations)} total conversations for session {session_id}")
            if not conversations:
                print("No conversations found in database")
                return []
            
            result = []
            total_tokens = 0
            excluded_conversations = []
            token_limit_70_percent = int(max_tokens * 0.7)
            
            for conv in conversations:
                conv_tokens = count_tokens(conv['query']) + count_tokens(conv['response'])
                
                if total_tokens + conv_tokens <= max_tokens and len(result) < n_recent:
                    result.append({
                        'query': conv['query'],
                        'response': conv['response'],
                        'timestamp': conv['timestamp']
                    })
                    total_tokens += conv_tokens
                else:
                    excluded_conversations.append(conv)
            
            if total_tokens >= token_limit_70_percent and excluded_conversations:
                summary = self._summarize_old_conversations(excluded_conversations)
                summary_tokens = count_tokens(summary)
                
                if total_tokens + summary_tokens <= max_tokens:
                    result.append({
                        'query': '[SUMMARY]',
                        'response': summary,
                        'timestamp': min(conv['timestamp'] for conv in excluded_conversations),
                        'is_summary': True
                    })
                    total_tokens += summary_tokens
                    print(f"Added summary of {len(excluded_conversations)} older conversations ({summary_tokens} tokens)")
            
            print(f"Retrieved {len(result)} conversations ({total_tokens} tokens) for session {session_id}")
            return result
        
        except Exception as e:
            print(f"Error getting recent conversations: {e}")
            return []
    
    def _summarize_old_conversations(self, conversations: List[Dict]) -> str:
        """Create a summary of older conversations that couldn't fit in the token window"""
        if not conversations:
            return ""
        
        conversations.sort(key=lambda x: x['timestamp'])
        
        topics = []
        key_responses = []
        user_questions = []
        
        for conv in conversations:
            query_words = conv['query'].lower().split()
            meaningful_words = [
                w for w in query_words 
                if len(w) > 3 and w not in ['what', 'when', 'where', 'why', 'how', 'can', 'will', 'would', 'could', 'should', 'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']
            ]
            if meaningful_words:
                topics.extend(meaningful_words[:2])  
            
            # Collect short user questions
            if len(conv['query']) < 100:
                user_questions.append(conv['query'])
            
            # Collect short responses that might contain key information
            if len(conv['response']) < 150:
                key_responses.append(conv['response'])
        
        # Create summary
        unique_topics = list(dict.fromkeys(topics))[:10]  # Top 10 unique topics, preserving order
        
        summary_parts = []
        summary_parts.append(f"Earlier conversation summary ({len(conversations)} exchanges):")
        
        if unique_topics:
            summary_parts.append(f"Key topics discussed: {', '.join(unique_topics)}")
        
        if user_questions:
            # Include a few representative questions
            representative_questions = user_questions[:3]
            summary_parts.append(f"Example questions: {'; '.join(representative_questions)}")
        
        if key_responses:
            # Include a few key short responses
            key_short_responses = [r for r in key_responses[:2]]
            if key_short_responses:
                summary_parts.append(f"Key responses: {'; '.join(key_short_responses)}")
        
        # Add time context
        start_time = datetime.fromtimestamp(conversations[0]['timestamp']).strftime("%H:%M")
        end_time = datetime.fromtimestamp(conversations[-1]['timestamp']).strftime("%H:%M")
        summary_parts.append(f"Time range: {start_time} - {end_time}")
        
        return " | ".join(summary_parts)
    
    def clear_history(self, session_id: str = None):
        """Clear chat history for a session or all if session_id is None"""
        try:
            if session_id:
                result = self.collection.delete_many({"session_id": session_id})
                print(f"Chat history cleared for session '{session_id}' - {result.deleted_count} documents deleted")
            else:
                result = self.collection.delete_many({})
                print(f"Chat history cleared for all sessions - {result.deleted_count} documents deleted")
        except Exception as e:
            print(f"Error clearing history: {e}")

    
    def close(self):
        """Close MongoDB connection"""
        try:
            self.client.close()
            print("MongoDB connection closed")
        except Exception as e:
            print(f"Warning: Error closing MongoDB connection: {e}")
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        try:
            self.close()
        except:
            pass

# Alias for compatibility with existing code
ChatHistoryManager = MongoDBChatHistoryManager
