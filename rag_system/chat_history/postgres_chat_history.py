import psycopg2
from psycopg2.extras import RealDictCursor
import time
import json
import uuid
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from ..token_utils import count_tokens

# Load environment variables
load_dotenv()

class PostgreSQLChatHistoryManager:
    def __init__(self, db_host="localhost", db_port=5432, db_name="chat_history_db", 
                 db_user="postgres", db_password="postgres"):
        """Initialize PostgreSQL chat history manager"""
        self.db_params = {
            'host': db_host,
            'port': db_port,
            'database': db_name,
            'user': db_user,
            'password': db_password or os.getenv('POSTGRES_PASSWORD', 'your_password')
        }
        
        print(f"Connecting to PostgreSQL database: {db_name}")
        self._test_connection()
    
    def _test_connection(self):
        """Test database connection"""
        try:
            conn = psycopg2.connect(**self.db_params)
            conn.close()
            print("Database connection successful")
        except psycopg2.Error as e:
            print(f"Database connection failed: {e}")
            raise
    
    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_params)
    
    def add_conversation(self, query: str, response: str, session_id: str, timestamp: float = None):
        """Add a conversation pair to PostgreSQL history"""
        if timestamp is None:
            timestamp = time.time()
        
        id = str(uuid.uuid4())
        
        insert_sql = """
        INSERT INTO chat_history (id, session_id, query, response, timestamp)
        VALUES (%s, %s, %s, %s, to_timestamp(%s))
        """
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_sql, (
                        id,
                        session_id,
                        query,
                        response,
                        timestamp
                    ))
                    conn.commit()
            print(f"Conversation added (Session: {session_id})")
        except Exception as e:
            print(f"Error adding conversation: {e}")
            raise
    
    def find_most_relevant_conversation(self, current_query: str, session_id: str, n_results: int = 5, 
                                        max_tokens: int = 2000) -> List[Dict]:
        """Find most relevant conversations using PostgreSQL full-text search with fallback"""
        
        search_sql = """
        WITH ranked_conversations AS (
            SELECT query, response,
                   EXTRACT(EPOCH FROM timestamp) as timestamp_unix,
                   ts_rank_cd(
                       setweight(to_tsvector('english', query), 'A') ||
                       setweight(to_tsvector('english', response), 'B'),
                       plainto_tsquery('english', %s)
                   ) as relevance_score
            FROM chat_history 
            WHERE session_id = %s 
              AND to_tsvector('english', query || ' ' || response) @@ plainto_tsquery('english', %s)
            ORDER BY relevance_score DESC, timestamp DESC
        )
        SELECT * FROM ranked_conversations LIMIT %s;
        """
        
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(search_sql, (current_query, session_id, current_query, n_results * 2))
                    results = cur.fetchall()
            
            if not results:
                print("No full-text search matches found. Checking recent conversations...")
                results = self.get_recent_conversations(session_id, min(3, n_results), max_tokens)
                if not results:
                    return []
            else:
                print(f"Found {len(results)} conversations via full-text search")
            
            conversations = []
            total_tokens = 0
            
            for row in results:
                conv_tokens = count_tokens(row['query']) + count_tokens(row['response'])
                
                if total_tokens + conv_tokens <= max_tokens:
                    relevance_score = float(row['relevance_score'])
                    conversations.append({
                        'query': row['query'],
                        'response': row['response'],
                        'timestamp': row['timestamp_unix'],
                        'distance': 1.0 - relevance_score,
                        'similarity_score': relevance_score
                    })
                    total_tokens += conv_tokens
                else:
                    break
                    
                if len(conversations) >= n_results:
                    break
            
            print(f"Returning {len(conversations)} conversations ({total_tokens} tokens)")
            return conversations
            
        except Exception as e:
            print(f"Error searching conversations: {e}")
            return []
    

    def get_recent_conversations(self, session_id: str, n_recent: int = 5, max_tokens = 3000) -> List[Dict]:
        """Get recent conversations for a session"""
        
        recent_sql = """
        SELECT query, response, 
               EXTRACT(EPOCH FROM timestamp) as timestamp_unix
        FROM chat_history
        WHERE session_id = %s
        ORDER BY timestamp DESC
        LIMIT %s
        """
        
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(recent_sql, (session_id, n_recent))
                    results = cur.fetchall()
            total_tokens = 0
            conversations = []
            for row in results:
                conversations.append({
                    'query': row['query'],
                    'response': row['response'],
                    'timestamp_unix': row['timestamp_unix'],  
                    'relevance_score': 0.5  
                })
                total_tokens += count_tokens(row['query']) + count_tokens(row['response'])
            
            print(f"Retrieved {len(conversations)} recent conversations")
            if(total_tokens > max_tokens):
                conversations = conversations[:n_recent]
            return conversations
            
        except Exception as e:
            print(f"Error getting recent conversations: {e}")
            return []
    

    

    def clear_history(self, session_id: str = None):
        """Clear chat history for a session or all if session_id is None"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    if session_id:
                        cur.execute("DELETE FROM chat_history WHERE session_id = %s", (session_id,))
                    else:
                        cur.execute("DELETE FROM chat_history")
                    rows_deleted = cur.rowcount
                    conn.commit()
            print(f"Cleared {rows_deleted} conversations")
        except Exception as e:
            print(f"Error clearing history: {e}")
            raise
    
    def get_conversation_count(self, session_id: str = None) -> int:
        """Get total number of conversations"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    if session_id:
                        cur.execute("SELECT COUNT(*) FROM chat_history WHERE session_id = %s", (session_id,))
                    else:
                        cur.execute("SELECT COUNT(*) FROM chat_history")
                    count = cur.fetchone()[0]
            return count
        except Exception as e:
            print(f"Error getting conversation count: {e}")
            return 0