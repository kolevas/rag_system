# Ensure project root is on sys.path so package imports like `rag_system.*` resolve
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from rag_system.vanilla_engine import DocumentChatBot
from rag_system.llamaindex.llamaindex_engine import LlamaIndexEngine
import json
import re

# import the multi-agent pipeline after path adjustments
from query_classifier import process_query, process_query_step

st.set_page_config(page_title="Document ChatBot", layout="wide")

def parse_bot_response(response_text):
    """Parse bot response and extract JSON content if present"""
    try:
        # If it's already a dict (from multiagent), extract response and follow-up questions
        if isinstance(response_text, dict):
            return response_text.get('response', str(response_text)), response_text.get('follow_up_questions', [])
        
        # Try to find JSON in the response
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            parsed = json.loads(json_str)
            return parsed.get('response', response_text), parsed.get('follow_up_questions', [])
        
        # Try to parse the entire response as JSON
        try:
            parsed = json.loads(response_text)
            return parsed.get('response', response_text), parsed.get('follow_up_questions', [])
        except:
            pass
        
        # If no JSON found, return the original response
        return response_text, []
    except Exception as e:
        print(f"Error parsing bot response: {e}")
        return str(response_text), []

def display_message(role, content, follow_up_questions=None, msg_idx=None, engine_type="document"):
    """Display a chat message with proper formatting"""
    if role == "user":
        st.markdown(f"""
        <div style='text-align: right; color: #1a73e8; margin: 10px 0; padding: 15px; 
                    background-color: #f0f8ff; border-radius: 15px; border-left: 4px solid #1a73e8;'>
            <b>You:</b> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='text-align: left; color: #333; margin: 10px 0; padding: 15px; 
                    background-color: #f9f9f9; border-radius: 15px; border-left: 4px solid #4CAF50;'>
            <b>Bot:</b> {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Display follow-up questions if available
        if follow_up_questions and len(follow_up_questions) > 0:
            st.markdown("**💡 Follow-up questions:**")
            cols = st.columns(min(len(follow_up_questions), 3))  # Max 3 columns
            for i, question in enumerate(follow_up_questions):
                col_idx = i % 3
                with cols[col_idx]:
                    # Use msg_idx in the key if available, else fallback to old logic
                    key_val = f"followup_{msg_idx}_{i}" if msg_idx is not None else f"followup_{len(st.session_state['messages'])}_{i}"
                    if st.button(
                        question, 
                        key=key_val,
                        help="Click to ask this question",
                        use_container_width=True
                    ):
                        # Add the follow-up question as a new user message
                        return question
            st.markdown("---")
        
        # Check if this message contains PDF export information and provide download
        if isinstance(content, dict) and content.get("pdf_exported"):
            pdf_file_path = "/Users/snezhanakoleva/praksa/app/exported_research.pdf"
            if os.path.exists(pdf_file_path):
                with open(pdf_file_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download Research PDF",
                        data=pdf_file.read(),
                        file_name="exported_research.pdf",
                        mime="application/pdf",
                        key=f"pdf_download_{msg_idx}" if msg_idx is not None else f"pdf_download_{len(st.session_state['messages'])}",
                        use_container_width=True
                    )
        elif isinstance(content, str) and ("pdf_exported" in content or "exported_research.pdf" in content):
            pdf_file_path = "/Users/snezhanakoleva/praksa/app/exported_research.pdf"
            if os.path.exists(pdf_file_path):
                with open(pdf_file_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download Research PDF",
                        data=pdf_file.read(),
                        file_name="exported_research.pdf",
                        mime="application/pdf",
                        key=f"pdf_download_str_{msg_idx}" if msg_idx is not None else f"pdf_download_str_{len(st.session_state['messages'])}",
                        use_container_width=True
                    )
    return None

def fetch_user_conversations(user_id, chatbot_type="document"):
    """Fetch all sessions for the user and their messages"""
    try:
        if chatbot_type == "llamaindex":
            bot = LlamaIndexEngine(user_id=user_id)
            sessions = bot.get_sessions()
        else:
            bot = DocumentChatBot(user_id=user_id)
            sessions = bot.get_sessions()
            
        conversations = []

        for session in sessions:
            session_id = session['session_id']
            try:
                # Get all messages for this session
                if chatbot_type == "llamaindex":
                    messages = bot.chat_manager.get_conversation(session_id=session_id)
                else:
                    messages = bot.chat_manager.get_conversation(session_id=session_id)
                
                # Create a meaningful title from the first user message if available
                title = session.get('title', f"Session {session_id[:8]}")
                if messages and len(messages) > 0:
                    first_user_msg = messages[0][1] if messages[0][0] == "user" else ""
                    if first_user_msg:
                        # Use first 50 characters of first message as title
                        title = first_user_msg[:50] + "..." if len(first_user_msg) > 50 else first_user_msg

                conversations.append({
                    "title": title,
                    "messages": messages,
                    "session_id": session_id,
                    "created_at": session.get('created_at', 0)
                })
            except Exception as e:
                print(f"Error fetching messages for session {session_id}: {e}")
                # Still add the session even if we can't get messages
                conversations.append({
                    "title": session.get('title', f"Session {session_id[:8]}"),
                    "messages": [],
                    "session_id": session_id,
                    "created_at": session.get('created_at', 0)
                })
                continue

        # Sort conversations by creation time (newest first)
        conversations.sort(key=lambda x: x.get('created_at', 0), reverse=True)
        return conversations

    except Exception as e:
        print(f"Error fetching user conversations: {e}")
        return []

def create_chatbot_instance(user_id, chatbot_type="document"):
    """Create appropriate chatbot instance based on type"""
    if chatbot_type == "llamaindex":
        bot = LlamaIndexEngine(user_id=user_id)
        # Build index if needed
        if not bot.has_existing_index():
            print("⚠️ No existing LlamaIndex found. You may need to build the index first.")
        else:
            bot.build_index()
        return bot
    elif chatbot_type == "multiagent":
        # Multiagent pipeline is a stateless function; no long-lived chatbot instance required
        return None
    else:
        return DocumentChatBot(user_id=user_id)

def query_chatbot(chatbot, user_input, chatbot_type="document"):
    """Unified interface to query either chatbot type"""
    if chatbot_type == "llamaindex":
        # LlamaIndex interface
        return chatbot.query(user_input)
    elif chatbot_type == "multiagent":
        # Use the non-blocking multi-agent pipeline
        result = process_query_step(user_input)
        # Check if we need user choice for low relevance
        if isinstance(result, dict) and result.get("status") == "need_user_choice":
            # Store the query and return a special marker for the UI to handle
            return {"status": "need_user_choice", "data": result}
        # For multiagent, result is already a structured response (JSON string or dict)
        # Try to parse it as JSON to extract follow-up questions
        if isinstance(result, str):
            try:
                # Try to parse JSON from the response
                import re
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', result, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(1))
                    return parsed  # Return the parsed dict directly
                else:
                    # Try to parse the entire string as JSON
                    parsed = json.loads(result)
                    return parsed  # Return the parsed dict directly
            except:
                # If parsing fails, return as string
                pass
        return result
    else:
        # DocumentChatBot interface
        relevant_history = chatbot._get_conversation_history(user_input)
        context_string = chatbot._get_document_context(user_input)
        history_text = f"\n\nRelevant Context:\n{relevant_history}" if relevant_history else ""
        system_message = chatbot._build_system_message(context_string, history_text)
        return chatbot._generate_response(user_input, system_message)

# Initialize session state variables
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = "default_user"
if 'chatbot_type' not in st.session_state:
    st.session_state['chatbot_type'] = "document"
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'active_conv' not in st.session_state:
    st.session_state['active_conv'] = None
if 'chatbot' not in st.session_state:
    st.session_state['chatbot'] = create_chatbot_instance(st.session_state['user_id'], st.session_state['chatbot_type'])
if 'conversations' not in st.session_state:
    st.session_state['conversations'] = []
if 'pending_followup' not in st.session_state:
    st.session_state['pending_followup'] = None
if 'pending_user_choice' not in st.session_state:
    st.session_state['pending_user_choice'] = None
if 'current_query' not in st.session_state:
    st.session_state['current_query'] = None

# Create columns
left_col, main_col, right_col = st.columns([2, 5, 2], gap="large")

# --- Left: Previous Conversations ---
with left_col:
    st.markdown("## Previous Conversations")
    
    # Refresh conversations for the current user
    st.session_state['conversations'] = fetch_user_conversations(st.session_state['user_id'], st.session_state['chatbot_type'])
    
    # Display message if no conversations found
    if not st.session_state['conversations']:
        st.markdown("*No previous conversations found.*")
    
    # List previous conversations
    for idx, conv in enumerate(st.session_state['conversations']):
        # Use a unique key and check if this is the active conversation
        is_active = st.session_state.get('active_conv') == idx
        button_label = f"📝 {conv['title']}" if not is_active else f"📖 {conv['title']}"
        
        if st.button(button_label, key=f"conv_{idx}", use_container_width=True):
            # Load the selected conversation
            st.session_state['messages'] = conv['messages'].copy()
            st.session_state['active_conv'] = idx
            
            # Update the chatbot to use the correct session
            st.session_state['chatbot'].session_id = conv['session_id']
            
            # Force a rerun to update the display
            st.rerun()
    
    st.markdown("---")
    
    # New Conversation button
    if st.button("New Conversation", use_container_width=True, type="primary"):
        # Create a new session in the database
        bot = st.session_state['chatbot']
        new_session = bot.chat_manager.create_session(st.session_state['user_id'], project_id="1")
        
        # Update the chatbot session
        st.session_state['chatbot'].session_id = new_session['session_id']
        
        # Clear current messages and set as new conversation
        st.session_state['messages'] = []
        st.session_state['active_conv'] = None
        
        # Refresh conversations list
        st.session_state['conversations'] = fetch_user_conversations(st.session_state['user_id'], st.session_state['chatbot_type'])
        
        # Force a rerun to update the display
        st.rerun()

# --- Right: ChatBot Settings ---
with right_col:
    st.markdown("## ChatBot Settings")
    
    # Chatbot Type Selection
    st.markdown("### Chatbot Engine")
    chatbot_type = st.selectbox(
        "Choose chatbot engine:",
        options=["document", "llamaindex", "multiagent"],
        format_func=lambda x: "Document ChatBot" if x == "document" else ("LlamaIndex ChatBot" if x == "llamaindex" else "MultiAgent System"),
        index=0 if st.session_state['chatbot_type'] == "document" else (1 if st.session_state['chatbot_type'] == "llamaindex" else 2),
        help="Document ChatBot uses traditional retrieval. LlamaIndex uses advanced indexing. MultiAgent runs the multi-agent pipeline."
    )

    # Show current engine status
    current_engine = "Document ChatBot" if st.session_state['chatbot_type'] == "document" else ("LlamaIndex ChatBot" if st.session_state['chatbot_type'] == "llamaindex" else "MultiAgent System")
    if chatbot_type != st.session_state['chatbot_type']:
        st.info(f"🔄 Engine will switch from **{current_engine}** to **{'Document ChatBot' if chatbot_type == 'document' else ('LlamaIndex ChatBot' if chatbot_type == 'llamaindex' else 'MultiAgent System') }** when you switch.")
    else:
        st.success(f"✅ Currently using: **{current_engine}**")
    
    st.markdown("---")
    
    # User ID input
    user_id = st.text_input("User ID", value=st.session_state.get('user_id', 'default_user'))
    
    if st.button("Switch User/Engine", key="new_bot", use_container_width=True):
        # Update user ID and chatbot type, create new chatbot instance
        st.session_state['user_id'] = user_id
        st.session_state['chatbot_type'] = chatbot_type

        with st.spinner(f"Initializing {'LlamaIndex' if chatbot_type == 'llamaindex' else ('MultiAgent' if chatbot_type == 'multiagent' else 'Document')} ChatBot..."):
            st.session_state['chatbot'] = create_chatbot_instance(user_id, chatbot_type)

        st.session_state['messages'] = []
        st.session_state['active_conv'] = None
        # For multiagent we don't have persistent conversations in DB, so only fetch when not multiagent
        if chatbot_type != 'multiagent':
            st.session_state['conversations'] = fetch_user_conversations(user_id, chatbot_type)
        else:
            st.session_state['conversations'] = []
        st.success(f"✅ Switched to {'LlamaIndex' if chatbot_type == 'llamaindex' else ('MultiAgent' if chatbot_type == 'multiagent' else 'Document')} ChatBot!")
        st.rerun()
    
    # Show engine-specific information
    st.markdown("### Engine Information")
    if st.session_state['chatbot_type'] == "llamaindex":
        st.info("""
        **🦙 LlamaIndex Features:**
        - Advanced semantic indexing
        - Vector similarity search
        - Built-in conversation history
        - Automatic context management
        """)
    elif st.session_state['chatbot_type'] == "multiagent":
        st.info("""
        **🤖 MultiAgent System Features:**
        - Multiple AI agents for diverse tasks
        - Dynamic task allocation
        - Context-aware agent switching
        - Integrated learning from interactions
        """)
    else:
        st.info("""
        **🤖 Document ChatBot Features:**
        - Traditional keyword search
        - Custom context retrieval  
        - Manual conversation management
        - Configurable search parameters
        """)
    
    st.markdown("---")
    
    # Display current session info
    st.markdown("### Session Info")
    engine_name = "LlamaIndex" if st.session_state['chatbot_type'] == "llamaindex" else "Document ChatBot"
    st.write(f"**Engine:** {engine_name}")
    
    if st.session_state.get('active_conv') is not None:
        active_session = st.session_state['conversations'][st.session_state['active_conv']]
        st.write(f"**Session:** {active_session['session_id'][:8]}...")
        st.write(f"**Messages:** {len(st.session_state['messages'])}")
        st.write(f"**User:** {st.session_state['user_id']}")
    else:
        st.write("**Status:** New conversation")
        st.write(f"**User:** {st.session_state['user_id']}")
    
    # Clear chat button
    if st.button("🗑️ Clear Current Chat", use_container_width=True, type="secondary"):
        st.session_state['messages'] = []
        st.rerun()

# --- Main: Chat Window ---
with main_col:
    # Dynamic title based on engine
    engine_icon = "🦙" if st.session_state['chatbot_type'] == "llamaindex" else "🤖"
    engine_name = "LlamaIndex ChatBot" if st.session_state['chatbot_type'] == "llamaindex" else "Document ChatBot"
    st.title(f"{engine_icon} {engine_name}")
    
    chatbot = st.session_state['chatbot']

    st.markdown("---")
    
    # Chat messages container
    chat_container = st.container()
    with chat_container:
        if st.session_state['messages']:
            pending_question = None
            for msg_idx, (role, msg) in enumerate(st.session_state['messages']):
                if role == "bot":
                    # Parse bot response for JSON content
                    parsed_response, follow_up_questions = parse_bot_response(msg)
                    clicked_question = display_message(role, parsed_response, follow_up_questions, msg_idx=msg_idx, engine_type=st.session_state['chatbot_type'])
                    if clicked_question:
                        pending_question = clicked_question
                else:
                    display_message(role, msg, engine_type=st.session_state['chatbot_type'])
            
            # Handle follow-up question clicks
            if pending_question:
                st.session_state['pending_followup'] = pending_question
                st.rerun()
        else:
            engine_name = "LlamaIndex ChatBot" if st.session_state['chatbot_type'] == "llamaindex" else "Document ChatBot"
            engine_description = "LlamaIndex for advanced semantic search" if st.session_state['chatbot_type'] == "llamaindex" else "Document ChatBot for traditional retrieval"
            st.markdown(f"""
            <div style='text-align: center; color: #666; font-style: italic; padding: 50px; 
                        border: 2px dashed #ccc; border-radius: 10px; background-color: #fafafa;'>
                <h3>🎯 Welcome to {engine_name}!</h3>
                <p>Start a conversation by typing your question below.</p>
                <p>I can help you with questions about your documents.</p>
                <p><strong>Engine:</strong> Using {engine_description}</p>
            </div>
            """, unsafe_allow_html=True)

    # Input section at the bottom
    st.markdown("---")
    
    # Handle pending user choice for multiagent low relevance
    if st.session_state.get('pending_user_choice') and st.session_state['chatbot_type'] == "multiagent":
        choice_data = st.session_state['pending_user_choice']
        st.warning("⚠️ " + choice_data.get("message", "Low relevance documents found."))
        
        st.write("**Choose how to proceed:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Use RAG Results", use_container_width=True, type="secondary"):
                # Process with RAG choice
                with st.spinner("🔍 Processing with RAG..."):
                    final_response = process_query_step(st.session_state['current_query'], user_choice="rag")
                    # Parse JSON response for proper formatting
                    if isinstance(final_response, str):
                        try:
                            import re
                            json_match = re.search(r'```json\s*(\{.*?\})\s*```', final_response, re.DOTALL)
                            if json_match:
                                final_response = json.loads(json_match.group(1))
                            else:
                                final_response = json.loads(final_response)
                        except:
                            pass  # Keep as string if parsing fails
                
                # Add both messages to chat
                st.session_state['messages'].append(("user", st.session_state['current_query']))
                st.session_state['messages'].append(("bot", final_response))
                
                # Clear pending state
                st.session_state['pending_user_choice'] = None
                st.session_state['current_query'] = None
                st.rerun()
        
        with col2:
            if st.button("🔬 Run Research", use_container_width=True, type="primary"):
                # Process with research choice
                with st.spinner("🔍 Running research, fact-checking, and generating PDF..."):
                    final_response = process_query_step(st.session_state['current_query'], user_choice="research")
                    # Parse JSON response for proper formatting
                    if isinstance(final_response, str):
                        try:
                            import re
                            json_match = re.search(r'```json\s*(\{.*?\})\s*```', final_response, re.DOTALL)
                            if json_match:
                                final_response = json.loads(json_match.group(1))
                            else:
                                final_response = json.loads(final_response)
                        except:
                            pass  # Keep as string if parsing fails
                
                # Check if PDF was exported and show success message with download
                pdf_file_path = "/Users/snezhanakoleva/praksa/app/exported_research.pdf"
                if isinstance(final_response, dict) and final_response.get("pdf_exported"):
                    st.success(f"✅ Research completed! PDF exported as: {final_response['pdf_exported']}")
                    # Add download button
                    if os.path.exists(pdf_file_path):
                        with open(pdf_file_path, "rb") as pdf_file:
                            st.download_button(
                                label="📥 Download Research PDF",
                                data=pdf_file.read(),
                                file_name="exported_research.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                elif "exported_research.pdf" in str(final_response):
                    st.success("✅ Research completed! PDF exported as: exported_research.pdf")
                    # Add download button
                    if os.path.exists(pdf_file_path):
                        with open(pdf_file_path, "rb") as pdf_file:
                            st.download_button(
                                label="📥 Download Research PDF",
                                data=pdf_file.read(),
                                file_name="exported_research.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                
                # Add both messages to chat
                st.session_state['messages'].append(("user", st.session_state['current_query']))
                st.session_state['messages'].append(("bot", final_response))
                
                # Clear pending state
                st.session_state['pending_user_choice'] = None
                st.session_state['current_query'] = None
                st.rerun()
        
        st.markdown("---")
        st.info("Please make a choice above to continue.")
    else:
        # Normal input flow when not waiting for user choice
        # Handle pending follow-up question
        if st.session_state.get('pending_followup'):
            user_input = st.session_state['pending_followup']
            st.session_state['pending_followup'] = None
            send_button = True
            clear_button = False
        else:
            # Create a form for better UX
            with st.form(key='chat_form', clear_on_submit=True):
                user_input = st.text_input(
                    "💬 Enter your question or command:", 
                    key="user_input", 
                    placeholder="Type your message here...",
                    help="Ask questions about your documents or use the follow-up suggestions above"
                )
                col1, col2, col3 = st.columns([2, 2, 4])
                
                with col1:
                    send_button = st.form_submit_button("📤 Send", use_container_width=True, type="primary")
                
                with col2:
                    clear_button = st.form_submit_button("🗑️ Clear", use_container_width=True, type="secondary")

        # Handle form submissions
        if send_button and user_input and user_input.strip():
            try:
                # Add user message to chat
                st.session_state['messages'].append(("user", user_input))
                
                # Get response using unified interface
                with st.spinner(f"🔍 Processing with {engine_name}..."):
                    response = query_chatbot(chatbot, user_input, st.session_state['chatbot_type'])
                
                # Check if multiagent needs user choice for low relevance
                if (st.session_state['chatbot_type'] == "multiagent" and 
                    isinstance(response, dict) and response.get("status") == "need_user_choice"):
                    # Store the pending state
                    st.session_state['pending_user_choice'] = response["data"]
                    st.session_state['current_query'] = user_input
                    # Remove the user message we just added since we'll re-add it after choice
                    st.session_state['messages'].pop()
                    st.rerun()
                else:
                    # Add bot response to chat (store the raw response)
                    st.session_state['messages'].append(("bot", response))
                    
                    # Save conversation to database (both engines support this)
                    if st.session_state['chatbot_type'] == "llamaindex":
                        # LlamaIndex handles saving internally in the query method
                        pass
                    elif st.session_state['chatbot_type'] != "multiagent":
                        # DocumentChatBot needs explicit saving (multiagent doesn't have persistent chat)
                        chatbot._save_conversation(user_input, response)
                    
                    # Update the conversations list if this was a new conversation
                    if st.session_state['active_conv'] is None and st.session_state['chatbot_type'] != "multiagent":
                        st.session_state['conversations'] = fetch_user_conversations(st.session_state['user_id'], st.session_state['chatbot_type'])
                        # Set this as the active conversation (it should be the first one now)
                        st.session_state['active_conv'] = 0
                    elif st.session_state['chatbot_type'] != "multiagent":
                        # Update the existing conversation in the list
                        st.session_state['conversations'][st.session_state['active_conv']]['messages'] = st.session_state['messages'].copy()
                    
                    # Rerun to update the display
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.error("Please try again or check your configuration.")
        
        # Handle clear button (only if not processing a follow-up)
        if not st.session_state.get('pending_followup') and 'clear_button' in locals() and clear_button:
            st.session_state['messages'] = []
            st.rerun()

# Add some custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #ddd;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stTextInput > div > div > input {
        border-radius: 8px;
    }
    
    /* Custom scrollbar for chat */
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* Follow-up question buttons */
    .stButton[data-testid="followup"] > button {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        color: #262730;
        font-size: 0.9rem;
        padding: 0.5rem;
        margin: 0.2rem 0;
    }
    
    .stButton[data-testid="followup"] > button:hover {
        background-color: #e6e9ef;
        border-color: #c4c8d0;
    }
    
    /* Engine specific styling */
    .llamaindex-engine {
        border-left: 4px solid #ff6b35 !important;
        background: linear-gradient(135deg, #fff5f0 0%, #ffe8d6 100%) !important;
    }
    
    .document-engine {
        border-left: 4px solid #4CAF50 !important;
        background: linear-gradient(135deg, #f0fff4 0%, #e8f5e8 100%) !important;
    }
    
    /* Info boxes for engines */
    .stAlert > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)