# ==========================================================
#                   ENHANCED MEMORY SYSTEM
# ==========================================================

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

class UserMemoryManager:
    """
    Complete recall system based on user login with session-based short-term memory
    """
    
    def __init__(self, services, vector_db):
        self.services = services
        self.vector_db = vector_db
        self.short_term_limit = 15  # Messages to keep in session memory
        self.long_term_retrieval_limit = 20  # Messages to retrieve from user history
        self.similarity_threshold = 0.25  # Lower threshold for better recall
    
    async def initialize_user_memory(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Initialize complete memory context for a user session"""
        try:
            # Load user's complete interaction history
            user_context = await self._load_user_long_term_memory(user_id)
            
            # Initialize short-term session memory
            await self._initialize_session_memory(session_id, user_context["recent_messages"])
            
            # Load user's persistent preferences and patterns
            user_patterns = await self._load_user_behavioral_patterns(user_id)
            
            return {
                "user_id": user_id,
                "session_id": session_id,
                "total_interactions": user_context["total_count"],
                "loaded_recent_messages": len(user_context["recent_messages"]),
                "behavioral_patterns": user_patterns,
                "memory_initialized": True
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize user memory for {user_id}: {e}")
            return {"memory_initialized": False, "error": str(e)}
    
    async def get_complete_context(self, user_id: str, session_id: str, current_message: str) -> Dict[str, Any]:
        """
        Retrieve complete conversational context combining:
        - Current session (short-term)
        - User's complete history (long-term)
        - Semantic similarity search across all user interactions
        """
        
        # 1. Get short-term session context
        session_context = await self._get_session_context(session_id)
        
        # 2. Get relevant long-term memories via semantic search
        semantic_memories = await self._search_user_semantic_memory(user_id, current_message)
        
        # 3. Get recent user interactions (chronological)
        recent_interactions = await self._get_recent_user_interactions(user_id, exclude_session=session_id)
        
        # 4. Get user behavioral patterns
        behavioral_context = await self._get_user_behavioral_context(user_id)
        
        # 5. Combine all contexts intelligently
        combined_context = await self._combine_memory_contexts(
            session_context, semantic_memories, recent_interactions, behavioral_context
        )
        
        return combined_context
    
    async def save_interaction_to_user_memory(self, user_id: str, session_id: str, 
                                            user_message: str, bot_reply: str, 
                                            metadata: Dict[str, Any]) -> bool:
        """Save interaction to both session and permanent user memory"""
        try:
            # Save to session (short-term)
            await self._save_to_session(session_id, user_message, bot_reply, metadata)
            
            # Save to permanent user memory (long-term)
            await self._save_to_user_permanent_memory(user_id, session_id, user_message, bot_reply, metadata)
            
            # Update user behavioral patterns
            await self._update_user_behavioral_patterns(user_id, user_message, bot_reply, metadata)
            
            # Save to vector database with user-specific indexing
            await self._save_to_user_vector_memory(user_id, session_id, user_message, bot_reply, metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save interaction to user memory: {e}")
            return False
    
    async def _load_user_long_term_memory(self, user_id: str) -> Dict[str, Any]:
        """Load user's complete interaction history"""
        try:
            # Get total interaction count
            total_count = await self.services.redis_client.llen(f"user_history:{user_id}")
            
            # Load recent interactions (more than session limit)
            recent_interactions = await self.services.redis_client.lrange(
                f"user_history:{user_id}", -self.long_term_retrieval_limit, -1
            )
            
            recent_messages = []
            for interaction in recent_interactions:
                try:
                    data = json.loads(interaction)
                    recent_messages.append(data)
                except json.JSONDecodeError:
                    continue
            
            return {
                "total_count": total_count,
                "recent_messages": recent_messages,
                "loaded_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error loading user long-term memory: {e}")
            return {"total_count": 0, "recent_messages": []}
    
    async def _initialize_session_memory(self, session_id: str, recent_messages: List[Dict]) -> None:
        """Initialize session with recent user interactions"""
        try:
            # Clear any existing session data
            await self.services.redis_client.delete(f"chat:{session_id}")
            
            # Load last few interactions into session memory
            for msg in recent_messages[-self.short_term_limit:]:
                if msg.get("user_message"):
                    await self.services.redis_client.rpush(
                        f"chat:{session_id}",
                        json.dumps({
                            "role": "user",
                            "message": msg["user_message"],
                            "timestamp": msg["timestamp"]
                        })
                    )
                if msg.get("bot_reply"):
                    await self.services.redis_client.rpush(
                        f"chat:{session_id}",
                        json.dumps({
                            "role": "bot", 
                            "message": msg["bot_reply"],
                            "timestamp": msg["timestamp"]
                        })
                    )
            
            logger.info(f"Initialized session {session_id} with {len(recent_messages)} recent interactions")
            
        except Exception as e:
            logger.error(f"Error initializing session memory: {e}")
    
    async def _search_user_semantic_memory(self, user_id: str, current_message: str, top_k: int = 10) -> List[Dict]:
        """Search user's complete history using semantic similarity"""
        try:
            # Search vector DB with user-specific filter
            results = await self.vector_db.search_similar_async(
                query=current_message,
                top_k=top_k,
                min_similarity_threshold=self.similarity_threshold,
                time_decay=True,
                metadata_filter={"user_id": user_id}  # Filter to this user only
            )
            
            # Enrich results with additional context
            enriched_results = []
            for result in results:
                enriched_result = {
                    **result,
                    "days_ago": (datetime.utcnow() - datetime.fromisoformat(result.get("timestamp", ""))).days if result.get("timestamp") else None,
                    "context_type": "semantic_memory"
                }
                enriched_results.append(enriched_result)
            
            logger.info(f"Found {len(enriched_results)} semantically similar memories for user {user_id}")
            return enriched_results
            
        except Exception as e:
            logger.error(f"Error searching semantic memory: {e}")
            return []
    
    async def _get_recent_user_interactions(self, user_id: str, limit: int = 10, exclude_session: str = None) -> List[Dict]:
        """Get user's most recent interactions from permanent storage"""
        try:
            # Get recent interactions
            interactions = await self.services.redis_client.lrange(f"user_history:{user_id}", -limit, -1)
            
            recent_interactions = []
            for interaction in interactions:
                try:
                    data = json.loads(interaction)
                    # Skip interactions from current session to avoid duplication
                    if exclude_session and data.get("session_id") == exclude_session:
                        continue
                    
                    data["context_type"] = "recent_interaction"
                    recent_interactions.append(data)
                except json.JSONDecodeError:
                    continue
            
            return recent_interactions
            
        except Exception as e:
            logger.error(f"Error getting recent user interactions: {e}")
            return []
    
    async def _load_user_behavioral_patterns(self, user_id: str) -> Dict[str, Any]:
        """Load user's behavioral patterns and preferences"""
        try:
            patterns_data = await self.services.redis_client.hgetall(f"user_patterns:{user_id}")
            
            if not patterns_data:
                return self._initialize_default_patterns()
            
            # Deserialize stored patterns
            patterns = {}
            for key, value in patterns_data.items():
                try:
                    patterns[key] = json.loads(value)
                except json.JSONDecodeError:
                    patterns[key] = value
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error loading user behavioral patterns: {e}")
            return self._initialize_default_patterns()
    
    def _initialize_default_patterns(self) -> Dict[str, Any]:
        """Initialize default behavioral patterns"""
        return {
            "preferred_topics": {},
            "conversation_style": "neutral",
            "response_length_preference": "medium",
            "mood_patterns": {},
            "interaction_times": [],
            "preferred_personalities": {},
            "total_interactions": 0,
            "avg_session_length": 0
        }
    
    async def _get_user_behavioral_context(self, user_id: str) -> Dict[str, Any]:
        """Get current behavioral context for the user"""
        try:
            patterns = await self._load_user_behavioral_patterns(user_id)
            
            # Create contextual summary
            context = {
                "context_type": "behavioral_patterns",
                "total_interactions": patterns.get("total_interactions", 0),
                "preferred_style": patterns.get("conversation_style", "neutral"),
                "top_topics": list(sorted(
                    patterns.get("preferred_topics", {}).items(),
                    key=lambda x: x[1], reverse=True
                )[:5]),
                "mood_tendency": self._get_dominant_mood(patterns.get("mood_patterns", {})),
                "response_preference": patterns.get("response_length_preference", "medium")
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting behavioral context: {e}")
            return {"context_type": "behavioral_patterns", "error": str(e)}
    
    def _get_dominant_mood(self, mood_patterns: Dict[str, int]) -> str:
        """Get the user's most common mood"""
        if not mood_patterns:
            return "neutral"
        return max(mood_patterns.items(), key=lambda x: x[1])[0]
    
    async def _combine_memory_contexts(self, session_context: Dict, semantic_memories: List[Dict],
                                     recent_interactions: List[Dict], behavioral_context: Dict) -> Dict[str, Any]:
        """Intelligently combine all memory contexts"""
        
        # Build comprehensive context string
        context_parts = []
        
        # Add behavioral context
        if behavioral_context.get("total_interactions", 0) > 0:
            behavior_summary = (
                f"User interaction history: {behavioral_context['total_interactions']} total conversations. "
                f"Preferred style: {behavioral_context['preferred_style']}. "
                f"Common mood: {behavioral_context['mood_tendency']}."
            )
            if behavioral_context.get("top_topics"):
                topics = [topic[0] for topic in behavioral_context["top_topics"][:3]]
                behavior_summary += f" Frequently discusses: {', '.join(topics)}."
            context_parts.append(behavior_summary)
        
        # Add recent interactions context
        if recent_interactions:
            context_parts.append(f"Recent conversations (last {len(recent_interactions)} interactions):")
            for interaction in recent_interactions[-5:]:  # Last 5 recent interactions
                if interaction.get("user_message") and interaction.get("bot_reply"):
                    context_parts.append(
                        f"User: {interaction['user_message'][:100]}... "
                        f"Bot: {interaction['bot_reply'][:100]}..."
                    )
        
        # Add semantic memories
        if semantic_memories:
            context_parts.append(f"Relevant past conversations (similarity-based):")
            for memory in semantic_memories[:3]:  # Top 3 most relevant
                context_parts.append(
                    f"Similar past exchange (similarity: {memory.get('similarity', 0):.2f}): "
                    f"User: '{memory.get('user_message', '')}' -> "
                    f"Bot: '{memory.get('bot_reply', '')[:100]}...'"
                )
        
        # Add current session context
        if session_context.get("messages"):
            context_parts.append("Current session context:")
            context_parts.append(session_context["formatted_context"])
        
        return {
            "combined_context": "\n".join(context_parts),
            "context_stats": {
                "session_messages": len(session_context.get("messages", [])),
                "semantic_memories": len(semantic_memories),
                "recent_interactions": len(recent_interactions),
                "total_user_interactions": behavioral_context.get("total_interactions", 0),
                "has_behavioral_data": behavioral_context.get("total_interactions", 0) > 0
            },
            "memory_sources": {
                "session": session_context,
                "semantic": semantic_memories,
                "recent": recent_interactions,
                "behavioral": behavioral_context
            }
        }
    
    async def _get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get current session context"""
        try:
            items = await self.services.redis_client.lrange(f"chat:{session_id}", 0, -1)
            
            messages = []
            for item in items:
                try:
                    messages.append(json.loads(item))
                except json.JSONDecodeError:
                    continue
            
            # Format for context
            formatted_lines = []
            for msg in messages[-self.short_term_limit:]:
                role = msg.get("role", "unknown")
                content = msg.get("message", "")
                if role == "user":
                    formatted_lines.append(f"User: {content}")
                elif role == "bot":
                    formatted_lines.append(f"Bot: {content}")
            
            return {
                "messages": messages,
                "formatted_context": "\n".join(formatted_lines),
                "message_count": len(messages)
            }
            
        except Exception as e:
            logger.error(f"Error getting session context: {e}")
            return {"messages": [], "formatted_context": "", "message_count": 0}
    
    async def _save_to_session(self, session_id: str, user_message: str, bot_reply: str, metadata: Dict) -> None:
        """Save to session memory (short-term)"""
        try:
            timestamp = datetime.utcnow().isoformat()
            
            # Save user message
            await self.services.redis_client.rpush(
                f"chat:{session_id}",
                json.dumps({
                    "role": "user",
                    "message": user_message,
                    "timestamp": timestamp,
                    "metadata": metadata
                })
            )
            
            # Save bot reply
            await self.services.redis_client.rpush(
                f"chat:{session_id}",
                json.dumps({
                    "role": "bot",
                    "message": bot_reply,
                    "timestamp": timestamp,
                    "metadata": metadata
                })
            )
            
            # Trim to keep only recent messages
            await self.services.redis_client.ltrim(f"chat:{session_id}", -self.short_term_limit*2, -1)
            
        except Exception as e:
            logger.error(f"Error saving to session: {e}")
    
    async def _save_to_user_permanent_memory(self, user_id: str, session_id: str, 
                                           user_message: str, bot_reply: str, metadata: Dict) -> None:
        """Save to permanent user memory (long-term)"""
        try:
            interaction = {
                "user_id": user_id,
                "session_id": session_id,
                "user_message": user_message,
                "bot_reply": bot_reply,
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Save to user's permanent history
            await self.services.redis_client.rpush(
                f"user_history:{user_id}",
                json.dumps(interaction)
            )
            
            # Update user's total interaction count
            await self.services.redis_client.incr(f"user_stats:{user_id}:total_interactions")
            
            logger.info(f"Saved interaction to permanent memory for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error saving to permanent memory: {e}")
    
    async def _save_to_user_vector_memory(self, user_id: str, session_id: str,
                                        user_message: str, bot_reply: str, metadata: Dict) -> None:
        """Save to vector database with user-specific metadata"""
        try:
            vector_metadata = {
                **metadata,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.vector_db.add_interaction_async(
                session_id, user_message, bot_reply, vector_metadata
            )
            
        except Exception as e:
            logger.error(f"Error saving to vector memory: {e}")
    
    async def _update_user_behavioral_patterns(self, user_id: str, user_message: str, 
                                             bot_reply: str, metadata: Dict) -> None:
        """Update user's behavioral patterns based on interaction"""
        try:
            patterns = await self._load_user_behavioral_patterns(user_id)
            
            # Update interaction count
            patterns["total_interactions"] = patterns.get("total_interactions", 0) + 1
            
            # Update mood patterns
            user_mood = metadata.get("mood", "neutral")
            mood_patterns = patterns.get("mood_patterns", {})
            mood_patterns[user_mood] = mood_patterns.get(user_mood, 0) + 1
            patterns["mood_patterns"] = mood_patterns
            
            # Update preferred personalities based on positive interactions
            personality = metadata.get("personality", "empathetic")
            response_time = metadata.get("response_time_ms", 0)
            if response_time < 5000:  # Quick, likely positive interaction
                personalities = patterns.get("preferred_personalities", {})
                personalities[personality] = personalities.get(personality, 0) + 1
                patterns["preferred_personalities"] = personalities
            
            # Save updated patterns
            serialized_patterns = {}
            for key, value in patterns.items():
                if isinstance(value, (dict, list)):
                    serialized_patterns[key] = json.dumps(value)
                else:
                    serialized_patterns[key] = str(value)
            
            await self.services.redis_client.hset(f"user_patterns:{user_id}", mapping=serialized_patterns)
            
        except Exception as e:
            logger.error(f"Error updating behavioral patterns: {e}")
    
    async def get_user_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive memory statistics for a user"""
        try:
            # Get total interactions
            total_interactions = await self.services.redis_client.llen(f"user_history:{user_id}")
            
            # Get behavioral patterns
            patterns = await self._load_user_behavioral_patterns(user_id)
            
            # Get vector DB entries count (approximate)
            vector_results = await self.vector_db.search_similar_async(
                "test", top_k=1000, metadata_filter={"user_id": user_id}
            )
            
            return {
                "user_id": user_id,
                "total_permanent_interactions": total_interactions,
                "vector_db_entries": len(vector_results),
                "behavioral_patterns_count": len(patterns),
                "memory_system_status": "complete_recall_enabled",
                "dominant_mood": self._get_dominant_mood(patterns.get("mood_patterns", {})),
                "total_topics_discussed": len(patterns.get("preferred_topics", {}))
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}