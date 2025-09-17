import re
import json
import asyncio
from collections import defaultdict, Counter
from typing import List, Dict, Set
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class UserPreferenceManager:
    def __init__(self, redis_client=None, vector_db=None):
        self.redis_client = redis_client
        self.vector_db = vector_db 
        self.session_preferences = defaultdict(dict)
        
        # Enhanced preference categories
        self.preference_categories = {
            'topics': set(),
            'activities': set(), 
            'emotions': set(),
            'learning_style': set(),
            'communication_style': set()
        }
        
        # Keyword patterns for different preference types
        self.preference_patterns = {
            'topics': {
                'technology': r'\b(programming|coding|ai|machine learning|technology|software|computer|tech)\b',
                'health': r'\b(health|fitness|exercise|diet|nutrition|wellness|medical)\b',
                'education': r'\b(learning|study|education|school|university|course|lesson)\b',
                'entertainment': r'\b(movies|music|games|gaming|books|reading|tv|shows)\b',
                'travel': r'\b(travel|vacation|trip|journey|adventure|explore)\b',
                'food': r'\b(food|cooking|recipe|cuisine|restaurant|meal|eat)\b',
                'sports': r'\b(sports|football|basketball|soccer|tennis|golf|running)\b',
                'art': r'\b(art|painting|drawing|creative|design|artistic|music)\b',
                'business': r'\b(business|work|career|job|professional|entrepreneur)\b',
                'science': r'\b(science|research|experiment|discovery|theory|physics|chemistry)\b'
            },
            'learning_style': {
                'visual': r'\b(visual|see|show|image|picture|diagram|chart)\b',
                'auditory': r'\b(hear|listen|audio|sound|music|voice|explain)\b',
                'hands_on': r'\b(hands.on|practice|do|try|experiment|build|make)\b',
                'step_by_step': r'\b(step|guide|tutorial|process|sequence|order)\b'
            },
            'communication_style': {
                'casual': r'\b(casual|relaxed|informal|chill|easy)\b',
                'formal': r'\b(formal|professional|structured|proper)\b',
                'detailed': r'\b(detail|thorough|complete|comprehensive|in.depth)\b',
                'concise': r'\b(brief|short|quick|summary|concise|simple)\b'
            },
            'emotions': {
                'curious': r'\b(curious|wonder|interested|fascinated|intrigued)\b',
                'excited': r'\b(excited|thrilled|enthusiastic|pumped|eager)\b',
                'calm': r'\b(calm|peaceful|relaxed|serene|tranquil)\b',
                'motivated': r'\b(motivated|driven|determined|ambitious|goal)\b'
            }
        }

    async def extract_preferences_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract structured preferences from user message using pattern matching"""
        text_lower = text.lower()
        extracted_prefs = defaultdict(list)
        
        # Extract preferences by category
        for category, patterns in self.preference_patterns.items():
            for pref_name, pattern in patterns.items():
                if re.search(pattern, text_lower, re.IGNORECASE):
                    if pref_name not in extracted_prefs[category]:
                        extracted_prefs[category].append(pref_name)
        
        # Extract explicit mentions
        explicit_preferences = self._extract_explicit_preferences(text)
        if explicit_preferences:
            extracted_prefs['explicit'].extend(explicit_preferences)
        
        return dict(extracted_prefs)

    def _extract_explicit_preferences(self, text: str) -> List[str]:
        """Extract explicitly stated preferences like 'I like...', 'I prefer...', etc."""
        preference_phrases = [
            r'i (?:really )?(?:like|love|enjoy|prefer|am interested in) (.+?)(?:\.|,|$)',
            r'i\'m (?:really )?(?:into|interested in) (.+?)(?:\.|,|$)',
            r'my (?:favorite|favourite) (.+?) (?:is|are) (.+?)(?:\.|,|$)',
            r'i (?:don\'t|dont|do not) (?:like|enjoy|prefer) (.+?)(?:\.|,|$)'
        ]
        
        explicit_prefs = []
        text_lower = text.lower()
        
        for pattern in preference_phrases:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Clean up the extracted preference
                pref = match.group(1).strip()
                if len(pref) > 2 and len(pref) < 50:  # Reasonable length
                    explicit_prefs.append(pref)
        
        return explicit_prefs

    async def update_preferences(self, session_id: str, user_message: str) -> Dict[str, any]:
        """Enhanced preference extraction and storage"""
        try:
            # Extract preferences from current message
            new_prefs = await self.extract_preferences_from_text(user_message)
            
            # Get existing preferences for this session
            existing_prefs = await self.get_session_preferences(session_id)
            
            # Merge preferences with scoring
            updated_prefs = self._merge_preferences(existing_prefs, new_prefs)
            
            # Save to Redis if available
            if self.redis_client:
                await self.save_preferences_to_redis(session_id, updated_prefs)
            else:
                # Fallback to memory storage
                self.session_preferences[session_id] = updated_prefs
            
            # Return simplified list for backward compatibility + full structure
            simple_list = self._flatten_preferences_for_display(updated_prefs)
            
            return {
                'preferences_list': simple_list,
                'preferences_detailed': updated_prefs,
                'newly_detected': list(new_prefs.keys()) if new_prefs else []
            }
            
        except Exception as e:
            logger.error(f"Error updating preferences for session {session_id}: {e}")
            return {
                'preferences_list': [],
                'preferences_detailed': {},
                'newly_detected': []
            }

    def _merge_preferences(self, existing: Dict, new_prefs: Dict) -> Dict:
        """Merge new preferences with existing ones, using frequency scoring"""
        merged = existing.copy() if existing else {}
        
        for category, prefs in new_prefs.items():
            if category not in merged:
                merged[category] = {}
            
            for pref in prefs:
                if pref in merged[category]:
                    # Increment score for repeated preferences
                    merged[category][pref] = merged[category][pref] + 1
                else:
                    # New preference starts with score of 1
                    merged[category][pref] = 1
                    
        return merged

    def _flatten_preferences_for_display(self, detailed_prefs: Dict) -> List[str]:
        """Convert detailed preferences to simple list for UI display"""
        flattened = []
        
        for category, prefs in detailed_prefs.items():
            if isinstance(prefs, dict):
                # Sort by frequency and take top preferences
                sorted_prefs = sorted(prefs.items(), key=lambda x: x[1], reverse=True)
                for pref, score in sorted_prefs[:3]:  # Top 3 per category
                    if score >= 2 or category == 'explicit':  # Only include if mentioned multiple times or explicit
                        flattened.append(f"{pref}")
            elif isinstance(prefs, list):
                flattened.extend(prefs[:3])  # Take first 3
                
        return list(set(flattened))[:10]  # Max 10 unique preferences

    async def get_session_preferences(self, session_id: str) -> Dict:
        """Retrieve preferences for a session"""
        if self.redis_client:
            try:
                prefs_json = await self.redis_client.get(f"preferences:{session_id}")
                if prefs_json:
                    return json.loads(prefs_json)
            except Exception as e:
                logger.error(f"Error loading preferences from Redis: {e}")
        
        return self.session_preferences.get(session_id, {})

    async def save_preferences_to_redis(self, session_id: str, preferences: Dict):
        """Save preferences to Redis with expiration"""
        try:
            prefs_json = json.dumps(preferences)
            await self.redis_client.setex(f"preferences:{session_id}", 3600 * 24, prefs_json)  # 24 hour expiry
            logger.info(f"Saved preferences for session {session_id}")
        except Exception as e:
            logger.error(f"Error saving preferences to Redis: {e}")

    async def get_preference_summary(self, session_id: str) -> Dict:
        """Get a comprehensive summary of user preferences"""
        prefs = await self.get_session_preferences(session_id)
        
        if not prefs:
            return {'summary': 'No preferences detected yet', 'categories': {}}
        
        summary = {}
        total_score = 0
        
        for category, category_prefs in prefs.items():
            if isinstance(category_prefs, dict):
                top_pref = max(category_prefs.items(), key=lambda x: x[1])
                summary[category] = {
                    'top_preference': top_pref[0],
                    'confidence': top_pref[1],
                    'all_preferences': category_prefs
                }
                total_score += sum(category_prefs.values())
        
        return {
            'summary': f"Detected {len(summary)} preference categories with total engagement score of {total_score}",
            'categories': summary,
            'engagement_level': 'high' if total_score > 10 else 'medium' if total_score > 5 else 'low'
        }

    def get_preferences_for_prompt(self, session_id: str) -> str:
        """Generate a preference context string for LLM prompts"""
        prefs = self.session_preferences.get(session_id, {})
        if not prefs:
            return ""
        
        context_parts = []
        for category, category_prefs in prefs.items():
            if isinstance(category_prefs, dict) and category_prefs:
                top_prefs = sorted(category_prefs.items(), key=lambda x: x[1], reverse=True)[:2]
                pref_names = [p[0] for p in top_prefs]
                context_parts.append(f"{category}: {', '.join(pref_names)}")
        
        return f"User preferences - {'; '.join(context_parts)}" if context_parts else ""
    
    async def update_preferences(self, session_id: str, user_message: str) -> Dict[str, any]:
        """Enhanced preference extraction with FAISS storage"""
        try:
            # Extract preferences from current message
            new_prefs = await self.extract_preferences_from_text(user_message)
            
            # Get existing preferences for this session
            existing_prefs = await self.get_session_preferences(session_id)
            
            # Merge preferences with scoring
            updated_prefs = self._merge_preferences(existing_prefs, new_prefs)
            
            # Save to Redis if available
            if self.redis_client:
                await self.save_preferences_to_redis(session_id, updated_prefs)
            else:
                # Fallback to memory storage
                self.session_preferences[session_id] = updated_prefs
            
            # Store preferences in FAISS for semantic search
            await self._store_preferences_in_faiss(session_id, updated_prefs, user_message)
            
            # Return simplified list for backward compatibility + full structure
            simple_list = self._flatten_preferences_for_display(updated_prefs)
            
            return {
                'preferences_list': simple_list,
                'preferences_detailed': updated_prefs,
                'newly_detected': list(new_prefs.keys()) if new_prefs else []
            }
            
        except Exception as e:
            logger.error(f"Error updating preferences for session {session_id}: {e}")
            return {
                'preferences_list': [],
                'preferences_detailed': {},
                'newly_detected': []
            }

    async def _store_preferences_in_faiss(self, session_id: str, preferences: Dict, context_message: str):
        """Store extracted preferences in FAISS vector database"""
        if not self.vector_db:
            return
            
        try:
            # Store each preference category with context
            for category, category_prefs in preferences.items():
                if isinstance(category_prefs, dict):
                    for pref_name, score in category_prefs.items():
                        if score >= 2:  # Only store preferences mentioned multiple times
                            preference_text = f"{category}: {pref_name} - {context_message[:100]}..."
                            
                            metadata = {
                                "preference_category": category,
                                "preference_name": pref_name,
                                "confidence_score": score,
                                "source_message": context_message[:200],  # Truncate
                                "is_preference": True
                            }
                            
                            # Store in FAISS
                            await self.vector_db.add_preference_embedding_async(
                                session_id, preference_text, category, metadata
                            )
                            
            logger.info(f"Stored {sum(len(v) for v in preferences.values() if isinstance(v, dict))} preferences in FAISS")
            
        except Exception as e:
            logger.error(f"Error storing preferences in FAISS: {e}")

    async def get_similar_preferences(self, session_id: str, query: str, top_k: int = 3) -> List[Dict]:
        """Get similar preferences using FAISS semantic search"""
        if not self.vector_db:
            return []
            
        try:
            similar_prefs = await self.vector_db.search_similar_preferences_async(
                query, session_id, top_k
            )
            return similar_prefs
        except Exception as e:
            logger.error(f"Error searching similar preferences: {e}")
            return []

    async def get_preference_context(self, session_id: str, current_message: str) -> str:
        """Get relevant preference context for the current conversation"""
        if not self.vector_db:
            return ""
            
        try:
            # Search for similar preferences
            similar_prefs = await self.get_similar_preferences(session_id, current_message, top_k=3)
            
            if not similar_prefs:
                return ""
                
            context_parts = []
            for pref in similar_prefs:
                category = pref.get("preference_category", "preference")
                name = pref.get("preference_name", "unknown")
                confidence = pref.get("confidence_score", 1)
                
                context_parts.append(
                    f"{category}: {name} (confidence: {confidence})"
                )
            
            return f"User preferences context: {', '.join(context_parts)}"
            
        except Exception as e:
            logger.error(f"Error getting preference context: {e}")
            return ""