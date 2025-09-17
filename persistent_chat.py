import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import redis.asyncio as redis
from dataclasses import dataclass
from enum import Enum

# ==========================================================
#                   1. DATA MODELS & REDIS KEY DESIGN
# ==========================================================

@dataclass
class ChatMessage:
    """Represents a single chat message"""
    message_id: str
    conversation_id: str
    user_id: Optional[str]
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        return cls(
            message_id=data["message_id"],
            conversation_id=data["conversation_id"],
            user_id=data.get("user_id"),
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )

@dataclass
class ConversationInfo:
    """Metadata about a conversation"""
    conversation_id: str
    user_id: str
    title: Optional[str]
    created_at: datetime
    last_message_at: datetime
    message_count: int
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "last_message_at": self.last_message_at.isoformat(),
            "message_count": self.message_count,
            "is_active": self.is_active
        }

"""
REDIS KEY DESIGN:

1. conversation:{conversation_id}:messages - LIST of message JSON objects (ordered chronologically)
2. conversation:{conversation_id}:info - HASH with conversation metadata
3. user:{user_id}:conversations - SET of conversation IDs for this user
4. session:{session_id}:context - LIST for short-term session context (optional, TTL)
5. session:{session_id}:conversation - STRING mapping session to conversation_id (TTL 24h)

TTL Recommendations:
- Messages: No TTL (persistent)
- Conversation info: No TTL (persistent) 
- Session context: 1 hour TTL
- Session mapping: 24 hour TTL
"""

# ==========================================================
#                   2. ENHANCED CONVERSATION MANAGER
# ==========================================================

class PersistentConversationManager:
    """Enhanced conversation manager with persistent storage"""
    
    def __init__(self, redis_client: redis.Redis, session_context_limit: int = 20):
        self.redis = redis_client
        self.session_context_limit = session_context_limit
    
    async def create_conversation(self, user_id: str, title: Optional[str] = None) -> str:
        """Create a new conversation and return conversation_id"""
        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Create conversation info
        conv_info = ConversationInfo(
            conversation_id=conversation_id,
            user_id=user_id,
            title=title or f"Chat {now.strftime('%Y-%m-%d %H:%M')}",
            created_at=now,
            last_message_at=now,
            message_count=0
        )
        
        # Store conversation metadata
        await self.redis.hset(
            f"conversation:{conversation_id}:info",
            mapping=conv_info.to_dict()
        )
        
        # Add to user's conversation list
        await self.redis.sadd(f"user:{user_id}:conversations", conversation_id)
        
        return conversation_id
    
    async def get_or_create_conversation(self, user_id: str, session_id: str, title: Optional[str] = None) -> str:
        """Get existing conversation for session or create new one"""
        # Check if session already has a conversation
        existing_conv = await self.redis.get(f"session:{session_id}:conversation")
        if existing_conv:
            # Verify conversation exists and belongs to user
            conv_info = await self.redis.hgetall(f"conversation:{existing_conv}:info")
            if conv_info and conv_info.get("user_id") == user_id:
                return existing_conv
        
        # Create new conversation
        conversation_id = await self.create_conversation(user_id, title)
        
        # Link session to conversation (24h TTL)
        await self.redis.setex(f"session:{session_id}:conversation", 86400, conversation_id)
        
        return conversation_id
    
    async def add_message(self, conversation_id: str, role: str, content: str, 
                         user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a message to persistent conversation history"""
        message_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        message = ChatMessage(
            message_id=message_id,
            conversation_id=conversation_id,
            user_id=user_id,
            role=role,
            content=content,
            timestamp=now,
            metadata=metadata
        )
        
        # Use Redis transaction to ensure atomicity
        async with self.redis.pipeline() as pipe:
            # Add message to conversation
            await pipe.rpush(f"conversation:{conversation_id}:messages", json.dumps(message.to_dict()))
            
            # Update conversation metadata
            await pipe.hset(f"conversation:{conversation_id}:info", mapping={
                "last_message_at": now.isoformat(),
                "message_count": await self.redis.llen(f"conversation:{conversation_id}:messages") + 1
            })
            
            await pipe.execute()
        
        return message_id
    
    async def get_conversation_messages(self, conversation_id: str, 
                                      limit: Optional[int] = None, 
                                      offset: int = 0) -> List[ChatMessage]:
        """Retrieve messages from a conversation"""
        if limit:
            # Get last N messages
            raw_messages = await self.redis.lrange(
                f"conversation:{conversation_id}:messages", 
                -(limit + offset), 
                -1 - offset if offset > 0 else -1
            )
        else:
            # Get all messages
            raw_messages = await self.redis.lrange(
                f"conversation:{conversation_id}:messages", 
                offset, -1
            )
        
        messages = []
        for raw_msg in raw_messages:
            try:
                msg_data = json.loads(raw_msg)
                messages.append(ChatMessage.from_dict(msg_data))
            except (json.JSONDecodeError, KeyError) as e:
                # Skip corrupted messages but log the error
                print(f"Warning: Corrupted message in conversation {conversation_id}: {e}")
                continue
        
        return messages
    
    async def get_user_conversations(self, user_id: str, limit: int = 50) -> List[ConversationInfo]:
        """Get all conversations for a user, sorted by last message time"""
        conversation_ids = await self.redis.smembers(f"user:{user_id}:conversations")
        conversations = []
        
        for conv_id in conversation_ids:
            conv_data = await self.redis.hgetall(f"conversation:{conv_id}:info")
            if conv_data:
                try:
                    conv_info = ConversationInfo(
                        conversation_id=conv_data["conversation_id"],
                        user_id=conv_data["user_id"],
                        title=conv_data.get("title"),
                        created_at=datetime.fromisoformat(conv_data["created_at"]),
                        last_message_at=datetime.fromisoformat(conv_data["last_message_at"]),
                        message_count=int(conv_data.get("message_count", 0)),
                        is_active=conv_data.get("is_active", "true").lower() == "true"
                    )
                    conversations.append(conv_info)
                except (KeyError, ValueError) as e:
                    print(f"Warning: Corrupted conversation info {conv_id}: {e}")
                    continue
        
        # Sort by last message time, most recent first
        conversations.sort(key=lambda x: x.last_message_at, reverse=True)
        return conversations[:limit]
    
    async def save_session_message(self, session_id: str, role: str, content: str):
        """Save message to session context (short-term buffer with TTL)"""
        message_data = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add to session context list
        await self.redis.rpush(f"session:{session_id}:context", json.dumps(message_data))
        
        # Trim to keep only recent messages
        await self.redis.ltrim(f"session:{session_id}:context", -self.session_context_limit, -1)
        
        # Set TTL on session context (1 hour)
        await self.redis.expire(f"session:{session_id}:context", 3600)
    
    async def build_session_context(self, session_id: str, limit: int = 10) -> str:
        """Build conversation context from session buffer"""
        raw_messages = await self.redis.lrange(f"session:{session_id}:context", -limit, -1)
        
        context_lines = []
        for raw_msg in raw_messages:
            try:
                msg_data = json.loads(raw_msg)
                role = msg_data["role"]
                content = msg_data["content"]
                if role == "user":
                    context_lines.append(f"User: {content}")
                else:
                    context_lines.append(f"Assistant: {content}")
            except (json.JSONDecodeError, KeyError):
                continue
        
        return "\n".join(context_lines)
    
    async def load_conversation_into_session(self, conversation_id: str, session_id: str, 
                                           recent_messages: int = 10):
        """Load recent conversation messages into session context"""
        messages = await self.get_conversation_messages(conversation_id, limit=recent_messages)
        
        # Clear existing session context
        await self.redis.delete(f"session:{session_id}:context")
        
        # Add messages to session context
        for message in messages:
            await self.save_session_message(session_id, message.role, message.content)
    
    async def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        """Delete a conversation and all its messages"""
        # Verify ownership
        conv_info = await self.redis.hgetall(f"conversation:{conversation_id}:info")
        if not conv_info or conv_info.get("user_id") != user_id:
            return False
        
        # Delete conversation data
        async with self.redis.pipeline() as pipe:
            await pipe.delete(f"conversation:{conversation_id}:messages")
            await pipe.delete(f"conversation:{conversation_id}:info")
            await pipe.srem(f"user:{user_id}:conversations", conversation_id)
            await pipe.execute()
        
        return True
    
    async def update_conversation_title(self, user_id: str, conversation_id: str, title: str) -> bool:
        """Update conversation title"""
        # Verify ownership
        conv_info = await self.redis.hgetall(f"conversation:{conversation_id}:info")
        if not conv_info or conv_info.get("user_id") != user_id:
            return False
        
        await self.redis.hset(f"conversation:{conversation_id}:info", "title", title)
        return True

# ==========================================================
#                   3. FASTAPI WEBSOCKET INTEGRATION
# ==========================================================

class WebSocketChatHandler:
    """Handles WebSocket chat with persistent history"""
    
    def __init__(self, redis_client: redis.Redis):
        self.conv_manager = PersistentConversationManager(redis_client)
    
    async def handle_user_login(self, user_id: str, session_id: str) -> str:
        """Handle user login - get or create conversation"""
        conversation_id = await self.conv_manager.get_or_create_conversation(user_id, session_id)
        
        # Load recent conversation history into session context
        await self.conv_manager.load_conversation_into_session(
            conversation_id, session_id, recent_messages=10
        )
        
        return conversation_id
    
    async def handle_message(self, session_id: str, user_message: str, 
                           user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process incoming user message"""
        # Get conversation ID from session
        conversation_id = await self.redis.get(f"session:{session_id}:conversation")
        if not conversation_id:
            # Anonymous session - create temporary conversation
            if user_id:
                conversation_id = await self.conv_manager.create_conversation(user_id)
                await self.redis.setex(f"session:{session_id}:conversation", 86400, conversation_id)
            else:
                conversation_id = f"anonymous_{session_id}"
        
        # Save user message to session context (for immediate response generation)
        await self.conv_manager.save_session_message(session_id, "user", user_message)
        
        # If authenticated, also save to persistent history
        if user_id and not conversation_id.startswith("anonymous_"):
            await self.conv_manager.add_message(
                conversation_id, "user", user_message, user_id
            )
        
        # Build context for response generation
        context = await self.conv_manager.build_session_context(session_id, limit=10)
        
        return {
            "conversation_id": conversation_id,
            "context": context,
            "user_message": user_message
        }
    
    async def handle_bot_response(self, session_id: str, bot_response: str, 
                                user_id: Optional[str] = None, 
                                metadata: Optional[Dict[str, Any]] = None):
        """Process bot response and save to history"""
        conversation_id = await self.redis.get(f"session:{session_id}:conversation")
        
        # Save to session context
        await self.conv_manager.save_session_message(session_id, "assistant", bot_response)
        
        # If authenticated, save to persistent history
        if user_id and conversation_id and not conversation_id.startswith("anonymous_"):
            await self.conv_manager.add_message(
                conversation_id, "assistant", bot_response, user_id, metadata
            )

# ==========================================================
#                   4. FASTAPI ROUTE EXAMPLES
# ==========================================================

# Add these to your FastAPI app:

@app.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: UserInDB = Depends(get_current_user)
):
    """Get messages from a conversation"""
    conv_manager = PersistentConversationManager(services.redis_client)
    
    # Verify user owns this conversation
    conv_info = await services.redis_client.hgetall(f"conversation:{conversation_id}:info")
    if not conv_info or conv_info.get("user_id") != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = await conv_manager.get_conversation_messages(conversation_id, limit, offset)
    
    return {
        "conversation_id": conversation_id,
        "messages": [msg.to_dict() for msg in messages],
        "count": len(messages)
    }

@app.get("/api/conversations")
async def list_user_conversations(
    limit: int = 20,
    current_user: UserInDB = Depends(get_current_user)
):
    """List user's conversations"""
    conv_manager = PersistentConversationManager(services.redis_client)
    conversations = await conv_manager.get_user_conversations(current_user.id, limit)
    
    return {
        "conversations": [conv.to_dict() for conv in conversations],
        "count": len(conversations)
    }

@app.post("/api/conversations")
async def create_new_conversation(
    title: Optional[str] = None,
    current_user: UserInDB = Depends(get_current_user)
):
    """Create a new conversation"""
    conv_manager = PersistentConversationManager(services.redis_client)
    conversation_id = await conv_manager.create_conversation(current_user.id, title)
    
    return {
        "conversation_id": conversation_id,
        "message": "Conversation created successfully"
    }

# ==========================================================
#                   5. UPDATED WEBSOCKET ENDPOINT
# ==========================================================

@app.websocket("/ws/chat")
async def websocket_chat_with_history(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    chat_handler = WebSocketChatHandler(services.redis_client)
    user_id = None
    conversation_id = None
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "auth":
                # Handle authentication
                token = data.get("token")
                if token:
                    try:
                        current_user = await security_service.get_current_user(token)
                        user_id = current_user.id
                        
                        # Set up persistent conversation
                        conversation_id = await chat_handler.handle_user_login(user_id, session_id)
                        
                        await websocket.send_json({
                            "type": "auth_success",
                            "user_id": user_id,
                            "conversation_id": conversation_id,
                            "message": "Authentication successful, chat history loaded"
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "auth_error",
                            "error": str(e)
                        })
                        
            elif message_type in ["text", "audio"]:
                # Handle chat message
                user_message = data.get("content", "")
                if not user_message:
                    continue
                
                # Process user message
                message_info = await chat_handler.handle_message(session_id, user_message, user_id)
                
                # Generate bot response (your existing logic here)
                bot_response = await generate_bot_response(message_info["context"], user_message)
                
                # Save bot response
                await chat_handler.handle_bot_response(
                    session_id, bot_response, user_id, 
                    metadata={"response_time": datetime.utcnow().isoformat()}
                )
                
                await websocket.send_json({
                    "type": "response",
                    "conversation_id": message_info["conversation_id"],
                    "user_message": user_message,
                    "bot_response": bot_response,
                    "session_id": session_id
                })
                
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "error": str(e)
        })

# ==========================================================
#                   6. BEST PRACTICES FOR RACE CONDITIONS
# ==========================================================

class SafeConversationManager(PersistentConversationManager):
    """Version with additional safety measures"""
    
    async def add_message_safe(self, conversation_id: str, role: str, content: str,
                              user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                              max_retries: int = 3) -> str:
        """Add message with retry logic to handle race conditions"""
        for attempt in range(max_retries):
            try:
                return await self.add_message(conversation_id, role, content, user_id, metadata)
            except redis.RedisError as e:
                if attempt == max_retries - 1:
                    raise e
                # Wait briefly before retry
                await asyncio.sleep(0.1 * (attempt + 1))
        
    async def get_message_count_atomic(self, conversation_id: str) -> int:
        """Get accurate message count atomically"""
        return await self.redis.llen(f"conversation:{conversation_id}:messages")
    
    async def ensure_conversation_consistency(self, conversation_id: str):
        """Verify and fix conversation data consistency"""
        # Get actual message count
        actual_count = await self.get_message_count_atomic(conversation_id)
        
        # Update stored count if different
        stored_info = await self.redis.hgetall(f"conversation:{conversation_id}:info")
        if stored_info:
            stored_count = int(stored_info.get("message_count", 0))
            if stored_count != actual_count:
                await self.redis.hset(
                    f"conversation:{conversation_id}:info",
                    "message_count", actual_count
                )