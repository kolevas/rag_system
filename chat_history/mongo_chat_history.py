import pymongo
from pymongo import MongoClient
import time
from typing import List, Dict, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
from token_utils import count_tokens
from openai import AzureOpenAI
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables once at module level
load_dotenv()

class MongoDBChatHistoryManager:
    def __init__(self, mongo_uri=None, db_name="chat_history_db", collection_name="conversations"):
        self.mongo_uri = mongo_uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
        self.db_name = db_name
        self.collection_name = collection_name
        
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        
        try:
            self.openai_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION")
            )
        except Exception as e:
            self.openai_client = None
        
        # Test connection and setup
        self._test_connection()
        self._setup_collection()
    
    def _test_connection(self):
        """Test MongoDB connection"""
        try:
            # Ping the database
            self.client.admin.command('ping')
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
            
        except Exception as e:
            pass  # Silently ignore index creation errors
   
    def _normalize_conversation(query: str, response: str) -> str:
        normalized_response = re.sub(r"\[\[\d+\]\],?\s*", "", response)
        normalized_response = re.sub(r"\n\s*#{1,6}\s*([^\n]+)", r". \1:", normalized_response)
        normalized_response = re.sub(r"\n\s*-\s*\*\*([^*]+)\*\*:\s*", r". \1: ", normalized_response)
        normalized_response = re.sub(r"\n\s*-\s*", ". ", normalized_response)
        normalized_response = re.sub(r"\*\*([^*]+)\*\*", r"\1", normalized_response)
        normalized_response = re.sub(r"\*([^*]+)\*", r"\1", normalized_response)
        normalized_response = re.sub(r"\n{3,}", "\n\n", normalized_response)
        normalized_response = re.sub(r"\n\n", ". ", normalized_response)
        normalized_response = re.sub(r"\n", " ", normalized_response)
        normalized_response = re.sub(r"\.+", ".", normalized_response)
        normalized_response = re.sub(r"\s*\.\s*\.", ".", normalized_response)
        normalized_response = re.sub(r":\s*\.", ":", normalized_response)
        normalized_response = re.sub(r"\.\s*:", ":", normalized_response)
        normalized_response = re.sub(r"\s+", " ", normalized_response)
        normalized_response = re.sub(r"\s*([.,:;!?])", r"\1", normalized_response)
        normalized_response = re.sub(r"([.,:;!?])\s*", r"\1 ", normalized_response)
        normalized_response = re.sub(r"\.\s*([a-z])", lambda m: ". " + m.group(1).upper(), normalized_response)
        normalized_response = re.sub(r"(\d+)\.\s+(\d+)", r"\1.\2", normalized_response)
        normalized_response = normalized_response.strip()
        if normalized_response and not normalized_response[0].isupper():
            normalized_response = normalized_response[0].upper() + normalized_response[1:]
        if normalized_response and normalized_response[-1] not in '.!?':
            normalized_response += "."
        return normalized_response
    
    def _summarize_conversation(self, query: str, response: str) -> str:
        """Summarize conversation using OpenAI model"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are an expert at creating concise summaries while preserving ALL important information. Create a summary that: 
                        1) Retains all key facts, numbers, data points, and specific details 
                        2) Maintains the logical flow of the conversation 
                        3) Targets 60% of original length while preserving 100% of factual content 
                        4) Returns clean text without markdown and new lines (don't add \n)
                        5) Prioritizes factual accuracy over brevity"""}, 
                    {"role": "user", "content": f"Please summarize this conversation:\nUser: {query}\nAssistant: {response}"}
                ],
                max_tokens=500,
                temperature=0.3,
                timeout=30
            )
            summary = response.choices[0].message.content.strip()
            if summary and len(summary) > 10:
                return summary
        except Exception as e:
            return self._normalize_conversation(query, response)

    def add_conversation(self, query: str, response: str, session_id: str, timestamp: float = None):
        """Add a conversation pair to history for a specific session"""
        if timestamp is None:
            timestamp = time.time()
        
        document = {
            "_id": f"{session_id}_conv_{int(timestamp * 1000)}",
            "query": query,
            "response": response,
            "session_id": session_id,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp),
            "type": "conversation" ,
            "summary": self._summarize_conversation(query, response) if self.openai_client else self._normalize_conversation(query, response)
        }
        
        try:
            self.collection.replace_one(
                {"_id": document["_id"]}, 
                document, 
                upsert=True
            )
        except Exception as e:
            print(f"Error adding conversation: {e}")
            raise
    
    def find_most_relevant_conversation(self, current_query: str, session_id: str, n_results: int = 5, 
                                        min_relevance: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """Find the most relevant conversations using hybrid approach (30% recent, 70% similarity)"""
        conversations = self.get_hybrid_conversations(current_query, session_id, n_results, max_tokens)
        
        if not conversations:
            return None
        
        # Format conversations for context
        context_parts = []
        for conv in conversations:
            if conv.get('is_summary'):
                context_parts.append(f"[SUMMARY] {conv.get('response', conv.get('summary', ''))}")
            else:
                # Use summary since that's what we have in the hybrid conversations
                context_parts.append(f"[CONVERSATION] {conv.get('summary', '')}")
        
        return "\n".join(context_parts)

    def get_hybrid_conversations(self, current_query: str, session_id: str, n_results: int = 5, max_tokens: int = 3000) -> List[Dict]: 
        """
        Simplified hybrid approach: 30% recent, 70% similarity-based retrieval
        """
        try:
            # Check if conversations exist
            total_conversations = self.collection.count_documents({"session_id": session_id, "type": "conversation"})
            if total_conversations == 0:
                return []
            
            # Simple 30/70 split
            recent_tokens = int(max_tokens * 0.3)
            similarity_tokens = int(max_tokens * 0.7)
            
            # Get recent and similar conversations
            recent_conversations = self._get_recent_conversations_for_hybrid(session_id, recent_tokens)
            similarity_conversations = self._get_similarity_conversations(current_query, session_id, similarity_tokens, recent_conversations)
            
            # Combine without duplicates
            combined_conversations = recent_conversations.copy()
            recent_ids = {conv.get('_id', '') for conv in recent_conversations}
            
            for conv in similarity_conversations:
                if conv.get('_id', '') not in recent_ids:
                    combined_conversations.append(conv)
            
            return combined_conversations
            
        except Exception as e:
            print(f"Error in hybrid conversation retrieval: {e}")
            return []
    
    def _get_recent_conversations_for_hybrid(self, session_id: str, max_tokens: int) -> List[Dict]:
        """Get recent conversations within token limit"""
        try:
            conversations = list(self.collection.find(
                {"session_id": session_id, "type": "conversation"}
            ).sort("timestamp", -1).limit(10))
            
            result = []
            total_tokens = 0
            
            for conv in conversations:
                summary_tokens = count_tokens(conv.get('summary', ''))
                if total_tokens + summary_tokens <= max_tokens:
                    result.append({
                        '_id': conv.get('_id'),
                        'summary': conv.get('summary', ''),
                        'timestamp': conv['timestamp'],
                        'source': 'recent'
                    })
                    total_tokens += summary_tokens
                else:
                    break
            
            return result
            
        except Exception as e:
            print(f"Error getting recent conversations: {e}")
            return []
    
    def _get_similarity_conversations(self, current_query: str, session_id: str, max_tokens: int, exclude_conversations: List[Dict]) -> List[Dict]:
        """Get conversations based on cosine similarity to current query"""
        try:
            # Get all conversations except recent ones
            all_conversations = list(self.collection.find(
                {"session_id": session_id, "type": "conversation"}
            ).sort("timestamp", -1))
            
            excluded_ids = {conv.get('_id', '') for conv in exclude_conversations}
            candidates = [conv for conv in all_conversations if conv.get('_id', '') not in excluded_ids]
            
            if not candidates:
                return []
            
            # Prepare texts for similarity calculation
            query_text = current_query.lower().strip()
            conversation_texts = [f"{conv['query']} {conv['response']}".lower() for conv in candidates]
            all_texts = [query_text] + conversation_texts
            
            # Calculate cosine similarity using TF-IDF
            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=500, ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
                
            except Exception as e:
                # Simple fallback
                query_words = set(query_text.split())
                similarities = []
                for text in conversation_texts:
                    text_words = set(text.split())
                    overlap = len(query_words.intersection(text_words))
                    similarities.append(overlap / len(query_words) if query_words else 0)
            
            # Score and sort conversations
            scored_conversations = []
            for i, conv in enumerate(candidates):
                similarity_score = similarities[i] if i < len(similarities) else 0
                scored_conversations.append({
                    '_id': conv.get('_id'),
                    'summary': conv.get('summary', ''),
                    'timestamp': conv['timestamp'],
                    'similarity_score': similarity_score,
                    'source': 'similarity'
                })
            
            # Sort by similarity and filter by threshold
            scored_conversations.sort(key=lambda x: x['similarity_score'], reverse=True)
            min_threshold = 0.02  # Low threshold for inclusivity
            relevant = [conv for conv in scored_conversations if conv['similarity_score'] >= min_threshold]
            
            if not relevant:
                relevant = scored_conversations[:3]
            
            # Select conversations within token limit
            result = []
            total_tokens = 0
            
            for conv in relevant:
                summary_tokens = count_tokens(conv['summary'])
                if total_tokens + summary_tokens <= max_tokens:
                    result.append(conv)
                    total_tokens += summary_tokens
                else:
                    break
            
            return result
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

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
        except Exception as e:
            pass  # Silently ignore close errors
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        try:
            self.close()
        except:
            pass
