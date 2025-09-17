# Enhanced conversation_history.py - Complete Implementation
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import uuid
import re

logger = logging.getLogger(__name__)

@dataclass
class ConversationSummary:
    conversation_id: str
    session_id: str
    user_id: str
    title: str
    created_at: datetime
    last_message_at: datetime
    message_count: int
    preview: str

@dataclass
class ConversationMessage:
    message_id: str
    conversation_id: str
    role: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict] = None

class EnhancedConversationManager:
    def __init__(self, redis_client, history_limit: int = 50, model_service=None):
        self.redis_client = redis_client
        self.history_limit = history_limit
        self.model_service = model_service  # Your Gemini or other AI model service
        self.conversation_summaries = {}

    # ==== TITLE GENERATION METHODS (NEW/ENHANCED) ====
    
    async def generate_conversation_title(self, first_message: str, fallback_only: bool = False) -> str:
        """Generate a meaningful title from the first message"""
        # Clean and prepare the message
        cleaned_message = self._clean_message_for_title(first_message)
        
        # If AI generation is disabled or unavailable, use fallback
        if fallback_only or not self.model_service:
            return self._generate_fallback_title(cleaned_message)
        
        try:
            # Use AI model to generate title
            title_prompt = self._create_title_prompt(cleaned_message)
            
            def _generate_title():
                response = self.model_service.generate_content(title_prompt)
                return response.text.strip() if response and hasattr(response, 'text') else None
            
            # Run AI generation in thread to avoid blocking
            ai_title = await asyncio.to_thread(_generate_title)
            
            if ai_title:
                # Validate and clean the AI-generated title
                cleaned_title = self._validate_and_clean_title(ai_title)
                if cleaned_title:
                    logger.info(f"Generated AI title: '{cleaned_title}' from message: '{first_message[:50]}...'")
                    return cleaned_title
            
        except Exception as e:
            logger.warning(f"AI title generation failed: {e}")
        
        # Always fallback to rule-based generation
        fallback_title = self._generate_fallback_title(cleaned_message)
        logger.info(f"Using fallback title: '{fallback_title}'")
        return fallback_title

    def _clean_message_for_title(self, message: str) -> str:
        """Clean and prepare message text for title generation"""
        if not message:
            return ""
        
        # Remove extra whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', message.strip())
        
        # Remove common conversation starters that don't help with titles
        patterns_to_remove = [
            r'^(hi|hello|hey|greetings?)[,\s]*',
            r'^(can you|could you|please|help me)[,\s]*',
            r'^(i need|i want|i would like)[,\s]*',
        ]
        
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()

    def _create_title_prompt(self, message: str) -> str:
        """Create an optimized prompt for title generation"""
        return f"""Generate a short, descriptive title for a conversation that starts with this message:

"{message[:300]}"

Requirements:
- Maximum 50 characters
- Be specific and descriptive
- Avoid generic words like "help", "question", "chat"
- Focus on the main topic or request
- Use title case
- Return only the title, no quotes or explanation

Title:"""

    def _validate_and_clean_title(self, title: str) -> Optional[str]:
        """Validate and clean AI-generated title"""
        if not title:
            return None
        
        # Remove quotes and extra formatting
        title = title.strip('"\'""''')
        title = title.strip()
        
        # Remove common prefixes that AI might add
        prefixes_to_remove = ['Title:', 'title:', 'TITLE:', 'Chat about', 'Discussion about']
        for prefix in prefixes_to_remove:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        
        # Ensure it's not too long
        if len(title) > 50:
            # Try to truncate at word boundary
            words = title.split()
            truncated = ""
            for word in words:
                if len(truncated + word) <= 47:  # Leave room for "..."
                    truncated += word + " "
                else:
                    break
            title = truncated.strip() + "..." if truncated else title[:47] + "..."
        
        # Validate it's meaningful (not just generic words)
        generic_titles = {
            'chat', 'conversation', 'question', 'help', 'assistance', 
            'new chat', 'untitled', 'discussion', 'talk'
        }
        
        if title.lower() in generic_titles or len(title) < 3:
            return None
        
        return title

    def _generate_fallback_title(self, message: str) -> str:
        """Generate fallback title using rule-based approach"""
        if not message:
            return f"Chat {datetime.utcnow().strftime('%b %d, %H:%M')}"
        
        # Extract key information based on common patterns
        title = self._extract_topic_based_title(message)
        
        if not title:
            # Use first meaningful words
            words = message.split()
            meaningful_words = []
            
            # Skip common filler words
            skip_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                         'should', 'may', 'might', 'must', 'can', 'to', 'of', 'in', 'on', 'at',
                         'by', 'for', 'with', 'about', 'what', 'how', 'when', 'where', 'why'}
            
            for word in words[:10]:  # Look at first 10 words
                clean_word = re.sub(r'[^\w\s]', '', word.lower())
                if clean_word and clean_word not in skip_words and len(clean_word) > 2:
                    meaningful_words.append(word)
                if len(meaningful_words) >= 6:
                    break
            
            if meaningful_words:
                title = " ".join(meaningful_words)
            else:
                title = " ".join(words[:4])
        
        # Ensure proper length
        if len(title) > 50:
            title = title[:47] + "..."
        elif len(title) < 3:
            title = f"Chat {datetime.utcnow().strftime('%b %d')}"
        
        # Convert to title case
        title = title.title()
        
        return title

    def _extract_topic_based_title(self, message: str) -> Optional[str]:
        """Extract topic-based title using pattern matching"""
        message_lower = message.lower()
        
        # Pattern for questions
        question_patterns = [
            (r'how (?:do i|can i|to) (.+?)(?:\?|$)', r'How to \1'),
            (r'what (?:is|are) (.+?)(?:\?|$)', r'About \1'),
            (r'why (?:is|are|do|does) (.+?)(?:\?|$)', r'Why \1'),
            (r'when (?:is|are|do|does|should) (.+?)(?:\?|$)', r'When \1'),
            (r'where (?:is|are|can i find) (.+?)(?:\?|$)', r'Where \1'),
        ]
        
        for pattern, replacement in question_patterns:
            match = re.search(pattern, message_lower)
            if match:
                topic = match.group(1).strip()
                if len(topic) > 3:
                    return replacement.replace(r'\1', topic.title())
        
        # Pattern for requests
        request_patterns = [
            (r'(?:create|make|build|develop) (?:a |an |)(.+?)(?:\.|$)', r'Create \1'),
            (r'(?:write|draft) (?:a |an |)(.+?)(?:\.|$)', r'Write \1'),
            (r'(?:explain|describe) (.+?)(?:\.|$)', r'Explain \1'),
            (r'(?:analyze|review) (.+?)(?:\.|$)', r'Analyze \1'),
        ]
        
        for pattern, replacement in request_patterns:
            match = re.search(pattern, message_lower)
            if match:
                topic = match.group(1).strip()
                if len(topic) > 3 and len(topic) < 40:
                    return replacement.replace(r'\1', topic.title())
        
        return None

    async def update_conversation_title(self, conversation_id: str, user_id: str, new_title: str) -> bool:
        """Allow users to manually update conversation titles"""
        # Verify ownership
        conv_data = await self.redis_client.hgetall(f"conversation:{conversation_id}")
        if not conv_data or conv_data.get("user_id") != user_id:
            return False
        
        # Validate and clean the new title
        cleaned_title = self._validate_and_clean_title(new_title)
        if not cleaned_title:
            cleaned_title = new_title[:50]  # At least truncate it
        
        await self.redis_client.hset(f"conversation:{conversation_id}", "title", cleaned_title)
        logger.info(f"Updated conversation title to '{cleaned_title}' for {conversation_id}")
        return True

    async def regenerate_conversation_title(self, conversation_id: str, user_id: str) -> Optional[str]:
        """Regenerate title for an existing conversation based on its messages"""
        # Verify ownership
        conv_data = await self.redis_client.hgetall(f"conversation:{conversation_id}")
        if not conv_data or conv_data.get("user_id") != user_id:
            return None
        
        # Get the first user message
        messages = await self.get_conversation_messages(conversation_id, limit=5)
        first_user_message = None
        
        for msg in messages:
            if msg.role == "user":
                first_user_message = msg.content
                break
        
        if not first_user_message:
            return None
        
        # Generate new title
        new_title = await self.generate_conversation_title(first_user_message)
        
        # Update the conversation
        await self.redis_client.hset(f"conversation:{conversation_id}", "title", new_title)
        logger.info(f"Regenerated title '{new_title}' for conversation {conversation_id}")
        
        return new_title

    # ==== ORIGINAL CONVERSATION MANAGEMENT METHODS ====

    async def create_conversation(self, user_id: str, session_id: str, title: Optional[str] = None) -> str:
        """Create a new conversation thread for a user"""
        conversation_id = f"conv_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{session_id[:8]}"
        
        # Auto-generate title if not provided
        if not title:
            title = f"Chat {datetime.utcnow().strftime('%b %d, %Y %H:%M')}"
        
        conversation_data = {
            "conversation_id": conversation_id,
            "session_id": session_id,
            "user_id": user_id,
            "title": title,
            "created_at": datetime.utcnow().isoformat(),
            "last_message_at": datetime.utcnow().isoformat(),
            "message_count": 0,
            "preview": "",
            "is_active": True
        }
        
        # Store conversation metadata
        await self.redis_client.hset(
            f"conversation:{conversation_id}", 
            mapping={k: str(v) for k, v in conversation_data.items()}
        )
        
        # Add to user's conversation list
        await self.redis_client.sadd(f"user_conversations:{user_id}", conversation_id)
        
        # Map session to conversation for quick lookup
        await self.redis_client.setex(f"session_conversation:{session_id}", 3600 * 24, conversation_id)
        
        logger.info(f"Created conversation {conversation_id} for user {user_id}")
        return conversation_id

    async def create_conversation_with_title(self, user_id: str, session_id: str, first_message: str = None) -> str:
        """Create conversation with auto-generated title (ENHANCED VERSION)"""
        conversation_id = f"conv_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Generate title
        title = "New Chat"
        if first_message:
            title = await self.generate_conversation_title(first_message)
        
        conversation_data = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "session_id": session_id,
            "title": title,
            "created_at": datetime.utcnow().isoformat(),
            "last_message_at": datetime.utcnow().isoformat(),
            "message_count": "0",
            "preview": first_message[:100] if first_message else "",
            "is_active": "true"
        }
        
        # Store conversation metadata
        await self.redis_client.hset(f"conversation:{conversation_id}", mapping=conversation_data)
        
        # Add to user's conversation list
        await self.redis_client.sadd(f"user_conversations:{user_id}", conversation_id)
        
        # Map session to conversation
        await self.redis_client.setex(f"session_conversation:{session_id}", 3600 * 24, conversation_id)
        
        logger.info(f"Created conversation '{title}' ({conversation_id}) for user {user_id}")
        return conversation_id

    async def get_or_create_user_conversation(self, user_id: str, session_id: str) -> str:
        """Get the user's latest conversation or create a new one"""
        try:
            # Try to get the latest conversation for this user
            conversations = await self.get_user_conversations(user_id, limit=1)
            if conversations:
                # Use the most recent conversation
                conversation_id = conversations[0].conversation_id
                
                # Update session mapping to point to this conversation
                await self.redis_client.setex(
                    f"session_conversation:{session_id}", 
                    3600 * 24,  # 24 hours
                    conversation_id
                )
                
                logger.info(f"Using existing conversation {conversation_id} for user {user_id}")
                return conversation_id
            
            # Create a new conversation if none exists
            return await self.create_conversation(user_id, session_id)
            
        except Exception as e:
            logger.error(f"Error getting or creating conversation for user {user_id}: {e}")
            # Fallback to creating a new conversation
            return await self.create_conversation(user_id, session_id)

    async def get_conversation_by_session(self, session_id: str) -> Optional[str]:
        """Get conversation ID from session ID"""
        return await self.redis_client.get(f"session_conversation:{session_id}")

    async def save_message(self, user_id: str, session_id: str, role: str, message: str, metadata: Dict = None):
        """Save a message to the conversation history"""
        # Get or create conversation
        conversation_id = await self.get_conversation_by_session(session_id)
        if not conversation_id:
            conversation_id = await self.create_conversation(user_id, session_id)
        
        # Create message
        message_id = f"msg_{conversation_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        message_data = {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": message,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": json.dumps(metadata or {})
        }
        
        # Store message
        await self.redis_client.hset(
            f"message:{message_id}",
            mapping={k: str(v) for k, v in message_data.items()}
        )
        
        # Add message to conversation's message list
        await self.redis_client.lpush(f"conversation_messages:{conversation_id}", message_id)
        
        # Trim to history limit
        await self.redis_client.ltrim(f"conversation_messages:{conversation_id}", 0, self.history_limit - 1)
        
        # Update conversation metadata
        await self._update_conversation_metadata(conversation_id, message, role)
        
        logger.info(f"Saved {role} message to conversation {conversation_id}")

    async def add_message_to_conversation(self, conversation_id: str, role: str, content: str, metadata: Dict = None) -> str:
        """Add a message to a specific conversation"""
        # Create message
        message_id = f"msg_{conversation_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        message_data = {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": json.dumps(metadata or {})
        }
        
        # Store message
        await self.redis_client.hset(
            f"message:{message_id}",
            mapping={k: str(v) for k, v in message_data.items()}
        )
        
        # Add message to conversation's message list
        await self.redis_client.lpush(f"conversation_messages:{conversation_id}", message_id)
        
        # Trim to history limit
        await self.redis_client.ltrim(f"conversation_messages:{conversation_id}", 0, self.history_limit - 1)
        
        # Update conversation metadata
        await self._update_conversation_metadata(conversation_id, content, role)
        
        logger.info(f"Added {role} message to conversation {conversation_id}")
        return message_id

    async def _update_conversation_metadata(self, conversation_id: str, latest_message: str, role: str):
        """Update conversation metadata with latest message info (ENHANCED VERSION)"""
        conversation_data = await self.redis_client.hgetall(f"conversation:{conversation_id}")
        if not conversation_data:
            return
        
        # Update last message time and count
        current_count = int(conversation_data.get("message_count", 0))
        updates = {
            "last_message_at": datetime.utcnow().isoformat(),
            "message_count": str(current_count + 1)
        }
        
        # Update preview with first user message if empty
        if not conversation_data.get("preview") and role == "user":
            preview = latest_message[:100] + "..." if len(latest_message) > 100 else latest_message
            updates["preview"] = preview
            
            # Auto-generate better title from first message using our enhanced method
            if conversation_data.get("title", "").startswith("Chat "):
                try:
                    new_title = await self.generate_conversation_title(latest_message)
                    updates["title"] = new_title
                except Exception as e:
                    logger.warning(f"Failed to generate title for conversation {conversation_id}: {e}")
                    # Fallback to simple title generation
                    title_words = latest_message.split()[:6]
                    new_title = " ".join(title_words)
                    if len(new_title) > 50:
                        new_title = new_title[:47] + "..."
                    updates["title"] = new_title
        
        await self.redis_client.hset(f"conversation:{conversation_id}", mapping=updates)

    async def get_user_conversations(self, user_id: str, limit: int = 20) -> List[ConversationSummary]:
        """Get list of user's conversations with summaries"""
        conversation_ids = await self.redis_client.smembers(f"user_conversations:{user_id}")
        
        if not conversation_ids:
            return []
        
        conversations = []
        for conv_id in conversation_ids:
            conv_data = await self.redis_client.hgetall(f"conversation:{conv_id}")
            if conv_data and conv_data.get("is_active", "true").lower() == "true":
                conversations.append(ConversationSummary(
                    conversation_id=conv_id,
                    session_id=conv_data.get("session_id", ""),
                    user_id=conv_data.get("user_id", ""),
                    title=conv_data.get("title", "Untitled Chat"),
                    created_at=datetime.fromisoformat(conv_data.get("created_at")),
                    last_message_at=datetime.fromisoformat(conv_data.get("last_message_at")),
                    message_count=int(conv_data.get("message_count", 0)),
                    preview=conv_data.get("preview", "")
                ))
        
        # Sort by last message time (most recent first)
        conversations.sort(key=lambda x: x.last_message_at, reverse=True)
        return conversations[:limit]

    async def get_conversation_messages(self, conversation_id: str, limit: int = None) -> List[ConversationMessage]:
        """Get all messages in a conversation"""
        if limit is None:
            limit = self.history_limit
        
        message_ids = await self.redis_client.lrange(f"conversation_messages:{conversation_id}", 0, limit - 1)
        
        messages = []
        for msg_id in message_ids:
            msg_data = await self.redis_client.hgetall(f"message:{msg_id}")
            if msg_data:
                metadata = {}
                try:
                    metadata = json.loads(msg_data.get("metadata", "{}"))
                except:
                    pass
                
                messages.append(ConversationMessage(
                    message_id=msg_data.get("message_id", ""),
                    conversation_id=msg_data.get("conversation_id", ""),
                    role=msg_data.get("role", ""),
                    content=msg_data.get("content", ""),
                    timestamp=datetime.fromisoformat(msg_data.get("timestamp")),
                    metadata=metadata
                ))
        
        # Reverse to get chronological order (oldest first)
        return list(reversed(messages))

    async def build_context_text(self, user_id: str, session_id: str, limit: Optional[int] = None) -> str:
        """Build context text from conversation history"""
        conversation_id = await self.get_conversation_by_session(session_id)
        if not conversation_id:
            return ""
        
        messages = await self.get_conversation_messages(conversation_id, limit or 10)
        
        lines = []
        for msg in messages:
            if msg.role.lower() == "user":
                lines.append(f"User: {msg.content}")
            else:
                lines.append(f"Bot: {msg.content}")
        
        return "\n".join(lines)

    async def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        """Delete a conversation (soft delete by marking inactive)"""
        # Verify ownership
        conv_data = await self.redis_client.hgetall(f"conversation:{conversation_id}")
        if not conv_data or conv_data.get("user_id") != user_id:
            return False
        
        # Mark as inactive
        await self.redis_client.hset(f"conversation:{conversation_id}", "is_active", "false")
        logger.info(f"Deleted conversation {conversation_id} for user {user_id}")
        return True

    async def search_conversations(self, user_id: str, query: str, limit: int = 10) -> List[ConversationSummary]:
        """Search user's conversations by content"""
        conversations = await self.get_user_conversations(user_id)
        
        # Simple text search in titles and previews
        query_lower = query.lower()
        matching = []
        
        for conv in conversations:
            if (query_lower in conv.title.lower() or 
                query_lower in conv.preview.lower()):
                matching.append(conv)
        
        return matching[:limit]

    async def get_conversation_stats(self, user_id: str) -> Dict:
        """Get user's conversation statistics"""
        conversations = await self.get_user_conversations(user_id, limit=1000)
        
        if not conversations:
            return {
                "total_conversations": 0,
                "total_messages": 0,
                "oldest_conversation": None,
                "most_active_conversation": None
            }
        
        total_messages = sum(conv.message_count for conv in conversations)
        oldest_conv = min(conversations, key=lambda x: x.created_at)
        most_active_conv = max(conversations, key=lambda x: x.message_count)
        
        return {
            "total_conversations": len(conversations),
            "total_messages": total_messages,
            "oldest_conversation": {
                "id": oldest_conv.conversation_id,
                "title": oldest_conv.title,
                "created_at": oldest_conv.created_at.isoformat()
            },
            "most_active_conversation": {
                "id": most_active_conv.conversation_id,
                "title": most_active_conv.title,
                "message_count": most_active_conv.message_count
            }
        }

    async def cleanup_old_conversations(self, days_old: int = 30):
        """Clean up conversations older than specified days"""
        # This would be run as a background task
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        # Implementation would scan for old conversations and archive them
        pass

# Example usage:
"""
# Initialize the manager with your services
conversation_manager = EnhancedConversationManager(
    redis_client=your_redis_client,
    model_service=your_gemini_service,  # Optional for AI title generation
    history_limit=50
)

# Create a conversation with intelligent title generation
conversation_id = await conversation_manager.create_conversation_with_title(
    user_id="user123",
    session_id="session456",
    first_message="How do I create a REST API in Python using FastAPI with authentication?"
)
# Might generate: "Create REST API Python FastAPI Auth"

# Save messages (will auto-update titles for new conversations)
await conversation_manager.save_message(
    user_id="user123",
    session_id="session456", 
    role="user",
    message="I need help with FastAPI"
)

# User can manually update title
await conversation_manager.update_conversation_title(
    conversation_id, 
    "user123", 
    "FastAPI Development Help"
)

# Regenerate title from conversation history
new_title = await conversation_manager.regenerate_conversation_title(
    conversation_id, 
    "user123"
)

# Get conversations with improved titles
conversations = await conversation_manager.get_user_conversations("user123")
for conv in conversations:
    print(f"Title: {conv.title}, Preview: {conv.preview}")
"""