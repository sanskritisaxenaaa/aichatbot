import faiss
import numpy as np
import json
import os
import pickle
import asyncio
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import List, Dict, Optional, Union
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(
        self, 
        embedding_model="all-MiniLM-L6-v2", 
        vector_dim=384,
        use_gemini=False,
        gemini_api_key=None,
        persistence_path="vector_db"
    ):
        """
        Initialize FAISS-based vector store with optional Gemini embeddings.
        
        Args:
            embedding_model: SentenceTransformer model name or Gemini model name
            vector_dim: Dimension of embeddings (384 for MiniLM, 768 for Gemini text-embedding-004)
            use_gemini: Whether to use Gemini embeddings instead of local model
            gemini_api_key: Google AI API key (or set GOOGLE_API_KEY env var)
            persistence_path: Directory to save/load the vector database
        """
        self.use_gemini = use_gemini
        self.persistence_path = persistence_path
        os.makedirs(persistence_path, exist_ok=True)
        
        if use_gemini:
            # Gemini embeddings setup
            api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key is required. Set GOOGLE_API_KEY env var or pass gemini_api_key parameter")
            
            genai.configure(api_key=api_key)
            self.embedding_model = "models/text-embedding-004"  # Latest Gemini embedding model
            self.vector_dim = 768  # text-embedding-004 dimension
            self.model = None
            logger.info("Using Gemini embeddings (text-embedding-004)")
        else:
            # Local SentenceTransformer setup
            self.model = SentenceTransformer(embedding_model)
            self.embedding_model = embedding_model
            self.vector_dim = vector_dim
            logger.info(f"Using local SentenceTransformer: {embedding_model}")
        
        # Initialize FAISS index with proper dimension
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.metadata_store = []
        
        # Load existing data
        self.load_database()

    def _embed_local(self, text: str) -> np.ndarray:
        """Convert text into embedding vector using local model."""
        return np.array(self.model.encode([text]), dtype=np.float32)

    async def _embed_gemini(self, text: str) -> np.ndarray:
        """Convert text into embedding vector using Gemini API."""
        try:
            # Clean text for Gemini API
            cleaned_text = text.replace("\n", " ").strip()
            if not cleaned_text:
                return np.zeros((1, self.vector_dim), dtype=np.float32)
            
            # Generate embedding using Gemini
            response = await asyncio.to_thread(
                genai.embed_content,
                model=self.embedding_model,
                content=cleaned_text,
                task_type="retrieval_document"  # Optimized for retrieval tasks
            )
            
            embedding = response['embedding']
            return np.array([embedding], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            # Fallback to zero vector
            return np.zeros((1, self.vector_dim), dtype=np.float32)

    def _embed(self, text: str) -> np.ndarray:
        """Synchronous embedding wrapper."""
        if self.use_gemini:
            # Run async function in sync context
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self._embed_gemini(text))
            except RuntimeError:
                # Create new event loop if none exists
                return asyncio.run(self._embed_gemini(text))
        else:
            return self._embed_local(text)

    async def _embed_async(self, text: str) -> np.ndarray:
        """Async embedding method."""
        if self.use_gemini:
            return await self._embed_gemini(text)
        else:
            return await asyncio.to_thread(self._embed_local, text)

    def add_interaction(self, session_id: str, user_message: str, bot_reply: str, metadata: Dict = None):
        """
        Store a key interaction: user message + bot reply.
        Enhanced with metadata and better text combination.
        """
        # Create more contextual combined text
        combined_text = f"User: {user_message.strip()}\nBot: {bot_reply.strip()}"
        
        try:
            vector = self._embed(combined_text)
            self.index.add(vector)
            
            # Enhanced metadata storage
            interaction_metadata = {
                "session_id": session_id,
                "user_message": user_message,
                "bot_reply": bot_reply,
                "timestamp": datetime.utcnow().isoformat(),
                "interaction_id": len(self.metadata_store),  # Unique ID
                "text_length": len(combined_text),
                "embedding_model": self.embedding_model
            }
            
            # Add custom metadata if provided
            if metadata:
                interaction_metadata.update(metadata)
            
            self.metadata_store.append(interaction_metadata)
            
            # Auto-save every 10 interactions
            if len(self.metadata_store) % 10 == 0:
                self.save_database()
            
            logger.info(f"[VectorDB] Stored interaction #{len(self.metadata_store)} for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error adding interaction: {e}")

    async def add_interaction_async(self, session_id: str, user_message: str, bot_reply: str, metadata: Dict = None):
        """Async version of add_interaction for better performance."""
        combined_text = f"User: {user_message.strip()}\nBot: {bot_reply.strip()}"
        
        try:
            vector = await self._embed_async(combined_text)
            self.index.add(vector)
            
            interaction_metadata = {
                "session_id": session_id,
                "user_message": user_message,
                "bot_reply": bot_reply,
                "timestamp": datetime.utcnow().isoformat(),
                "interaction_id": len(self.metadata_store),
                "text_length": len(combined_text),
                "embedding_model": self.embedding_model
            }
            
            if metadata:
                interaction_metadata.update(metadata)
            
            self.metadata_store.append(interaction_metadata)
            
            if len(self.metadata_store) % 10 == 0:
                await asyncio.to_thread(self.save_database)
            
            logger.info(f"[VectorDB] Async stored interaction #{len(self.metadata_store)} for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error adding interaction async: {e}")

    def search_similar(
        self, 
        query: str, 
        top_k: int = 3, 
        min_similarity_threshold: float = 0.1,
        session_filter: Optional[str] = None,
        time_decay: bool = False
    ) -> List[Dict]:
        """
        Retrieve top-k similar past interactions from FAISS.
        Enhanced with filtering and time decay options.
        """
        if self.index.ntotal == 0:
            return []

        try:
            query_vector = self._embed(query)
            
            # Search more candidates than needed for filtering
            search_k = min(top_k * 3, self.index.ntotal)
            distances, indices = self.index.search(query_vector, search_k)

            results = []
            current_time = datetime.utcnow()
            
            for distance, idx in zip(distances[0], indices[0]):
                if idx >= len(self.metadata_store):
                    continue
                
                metadata = self.metadata_store[idx]
                
                # Apply session filter if specified
                if session_filter and metadata["session_id"] != session_filter:
                    continue
                
                # Calculate similarity score (lower distance = higher similarity)
                similarity = 1.0 / (1.0 + distance)
                
                if similarity < min_similarity_threshold:
                    continue
                
                # Apply time decay if requested
                if time_decay:
                    interaction_time = datetime.fromisoformat(metadata["timestamp"])
                    hours_ago = (current_time - interaction_time).total_seconds() / 3600
                    time_factor = max(0.1, 1.0 / (1.0 + hours_ago * 0.01))  # Gradual decay
                    similarity *= time_factor
                
                result = metadata.copy()
                result["similarity"] = float(similarity)
                result["distance"] = float(distance)
                results.append(result)
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching similar interactions: {e}")
            return []

    async def search_similar_async(
        self, 
        query: str, 
        top_k: int = 3, 
        min_similarity_threshold: float = 0.1,
        session_filter: Optional[str] = None,
        time_decay: bool = False
    ) -> List[Dict]:
        """Async version of search_similar."""
        if self.index.ntotal == 0:
            return []

        try:
            query_vector = await self._embed_async(query)
            
            # Run FAISS search in thread
            search_k = min(top_k * 3, self.index.ntotal)
            distances, indices = await asyncio.to_thread(
                self.index.search, query_vector, search_k
            )

            results = []
            current_time = datetime.utcnow()
            
            for distance, idx in zip(distances[0], indices[0]):
                if idx >= len(self.metadata_store):
                    continue
                
                metadata = self.metadata_store[idx]
                
                if session_filter and metadata["session_id"] != session_filter:
                    continue
                
                similarity = 1.0 / (1.0 + distance)
                
                if similarity < min_similarity_threshold:
                    continue
                
                if time_decay:
                    interaction_time = datetime.fromisoformat(metadata["timestamp"])
                    hours_ago = (current_time - interaction_time).total_seconds() / 3600
                    time_factor = max(0.1, 1.0 / (1.0 + hours_ago * 0.01))
                    similarity *= time_factor
                
                result = metadata.copy()
                result["similarity"] = float(similarity)
                result["distance"] = float(distance)
                results.append(result)
            
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in async search: {e}")
            return []

    def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent interactions for a specific session."""
        session_interactions = [
            metadata for metadata in self.metadata_store 
            if metadata["session_id"] == session_id
        ]
        # Sort by timestamp and return most recent
        session_interactions.sort(
            key=lambda x: x["timestamp"], reverse=True
        )
        return session_interactions[:limit]

    def get_database_stats(self) -> Dict:
        """Get statistics about the vector database."""
        if not self.metadata_store:
            return {"total_interactions": 0}
        
        sessions = set(metadata["session_id"] for metadata in self.metadata_store)
        total_chars = sum(metadata.get("text_length", 0) for metadata in self.metadata_store)
        
        # Find date range
        timestamps = [metadata["timestamp"] for metadata in self.metadata_store]
        timestamps.sort()
        
        return {
            "total_interactions": len(self.metadata_store),
            "unique_sessions": len(sessions),
            "total_characters": total_chars,
            "embedding_model": self.embedding_model,
            "vector_dimension": self.vector_dim,
            "oldest_interaction": timestamps[0] if timestamps else None,
            "newest_interaction": timestamps[-1] if timestamps else None,
            "using_gemini": self.use_gemini
        }

    def save_database(self):
        """Save the vector database to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(self.persistence_path, "faiss.index"))
            
            # Save metadata
            with open(os.path.join(self.persistence_path, "metadata.pkl"), "wb") as f:
                pickle.dump(self.metadata_store, f)
            
            # Save configuration
            config = {
                "embedding_model": self.embedding_model,
                "vector_dim": self.vector_dim,
                "use_gemini": self.use_gemini,
                "total_interactions": len(self.metadata_store)
            }
            
            with open(os.path.join(self.persistence_path, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved vector database with {len(self.metadata_store)} interactions")
            
        except Exception as e:
            logger.error(f"Error saving database: {e}")

    def load_database(self):
        """Load the vector database from disk."""
        try:
            index_path = os.path.join(self.persistence_path, "faiss.index")
            metadata_path = os.path.join(self.persistence_path, "metadata.pkl")
            config_path = os.path.join(self.persistence_path, "config.json")
            
            # Load configuration first to verify compatibility
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    saved_config = json.load(f)
                
                # Check if models match
                if saved_config.get("embedding_model") != self.embedding_model:
                    logger.warning(f"Model mismatch: saved={saved_config.get('embedding_model')}, current={self.embedding_model}")
                    return
            
            # Load FAISS index
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    self.metadata_store = pickle.load(f)
                logger.info(f"Loaded {len(self.metadata_store)} metadata entries")
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            # Initialize fresh database on load error
            self.index = faiss.IndexFlatL2(self.vector_dim)
            self.metadata_store = []

    def clear_database(self):
        """Clear all data from the vector database."""
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.metadata_store = []
        logger.info("Cleared vector database")

    def export_conversations(self, output_file: str = "conversations.json"):
        """Export all conversations to a JSON file."""
        try:
            export_data = {
                "metadata": self.get_database_stats(),
                "conversations": self.metadata_store
            }
            
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported {len(self.metadata_store)} conversations to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting conversations: {e}")
    def add_preference_embedding(self, session_id: str, preference_text: str, preference_type: str, metadata: Dict = None):
        """
        Store user preferences as embeddings for better personalization
        """
        try:
            vector = self._embed(preference_text)
            self.index.add(vector)
            
            preference_metadata = {
                "session_id": session_id,
                "preference_text": preference_text,
                "preference_type": preference_type,
                "timestamp": datetime.utcnow().isoformat(),
                "is_preference": True,
                "embedding_model": self.embedding_model
            }
            
            if metadata:
                preference_metadata.update(metadata)
            
            self.metadata_store.append(preference_metadata)
            logger.info(f"[VectorDB] Stored preference for session {session_id}: {preference_type}")
            
        except Exception as e:
            logger.error(f"Error adding preference embedding: {e}")

    async def add_preference_embedding_async(self, session_id: str, preference_text: str, preference_type: str, metadata: Dict = None):
        """Async version for preference embedding"""
        try:
            vector = await self._embed_async(preference_text)
            self.index.add(vector)
            
            preference_metadata = {
                "session_id": session_id,
                "preference_text": preference_text,
                "preference_type": preference_type,
                "timestamp": datetime.utcnow().isoformat(),
                "is_preference": True,
                "embedding_model": self.embedding_model
            }
            
            if metadata:
                preference_metadata.update(metadata)
            
            self.metadata_store.append(preference_metadata)
            logger.info(f"[VectorDB] Async stored preference for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error adding preference embedding async: {e}")

    def search_similar_preferences(self, query: str, session_id: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar preferences in the vector store
        """
        results = self.search_similar(
            query, 
            top_k=top_k * 2,  # Get more results for filtering
            session_filter=session_id,
            min_similarity_threshold=0.2
        )
        
        # Filter to only preference entries
        preference_results = [
            result for result in results 
            if result.get("is_preference", False)
        ]
        
        return preference_results[:top_k]

    async def search_similar_preferences_async(self, query: str, session_id: str, top_k: int = 5) -> List[Dict]:
        """Async version of preference search"""
        results = await self.search_similar_async(
            query, 
            top_k=top_k * 2,
            session_filter=session_id,
            min_similarity_threshold=0.2
        )
        
        preference_results = [
            result for result in results 
            if result.get("is_preference", False)
        ]
        
        return preference_results[:top_k]

    def get_session_preferences(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get all preferences for a specific session"""
        session_preferences = [
            metadata for metadata in self.metadata_store 
            if metadata.get("session_id") == session_id and metadata.get("is_preference", False)
        ]
        
        session_preferences.sort(key=lambda x: x["timestamp"], reverse=True)
        return session_preferences[:limit]

# Example usage:
if __name__ == "__main__":
    # Initialize with Gemini embeddings
    vector_db = VectorDBManager(
        use_gemini=True,
        gemini_api_key="your-gemini-api-key-here",  # Or set GOOGLE_API_KEY env var
        persistence_path="vector_db_gemini"
    )
    
    # Add some sample interactions
    vector_db.add_interaction(
        session_id="session_1",
        user_message="How do I train a machine learning model?",
        bot_reply="To train a machine learning model, you need to prepare your data, choose an algorithm, split your data into training and testing sets, fit the model on training data, and evaluate its performance."
    )
    
    # Search for similar interactions
    results = vector_db.search_similar("machine learning training process", top_k=3)
    print("Similar interactions found:")
    for result in results:
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"User: {result['user_message']}")
        print(f"Bot: {result['bot_reply'][:100]}...")
        print("-" * 50)