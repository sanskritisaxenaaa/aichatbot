import os
import io
import uuid
import json
import base64
import asyncio
import logging
import hashlib
import secrets
import struct
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, Any, List, Optional 
from datetime import datetime
from fastapi import Query

import torch
import whisper
import numpy as np
import redis.asyncio as redis
import requests
from pydub import AudioSegment 
from enum import Enum
from passlib.context import CryptContext
from dotenv import load_dotenv
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import (
    OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, validator
from transformers import pipeline
from cryptography.fernet import Fernet
import jwt
from vector_store import VectorDBManager
import google.generativeai as genai
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from user_preferences import UserPreferenceManager
from conversation_history import EnhancedConversationManager, ConversationSummary, ConversationMessage
from typing import Union
from memory_system import UserMemoryManager
from pydantic import ValidationError
from pydantic import BaseModel

# Initialize HTTPBearer for extracting Bearer tokens from Authorization header
security = HTTPBearer()


# ==========================================================
#                   CONFIGURATION
# ==========================================================

@dataclass
class Config:
    SECRET_KEY: str
    ENCRYPTION_KEY: bytes
    ELEVENLABS_API_KEY: Optional[str]
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-1.5-flash"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ELEVENLABS_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"
    REDIS_URL: str = "redis://localhost:6379"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: Optional[List[str]] = None
    # Vector DB settings (Gemini only for now)
    USE_GEMINI_EMBEDDINGS: bool = True   # ✅ rename to Gemini
    VECTOR_DB_PATH: str = "vector_db_gemini"
    STREAM_CHUNK_SIZE: int = 50
    STREAM_DELAY_MS: int = 50
    MAX_CONCURRENT_STREAMS: int = 10
    # NEW: Whisper configuration
    WHISPER_MODEL_SIZE: str = "small"  # base, small, medium, large
    WHISPER_AUTO_DETECT_MODEL: bool = True  # Auto-select based on hardware



    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

        encryption_key = os.getenv("ENCRYPTION_KEY")
        if encryption_key:
            encryption_key = encryption_key.encode()
        else:
            encryption_key = Fernet.generate_key()

        return cls(
            SECRET_KEY=os.getenv("SECRET_KEY") or Fernet.generate_key().decode(),
            ENCRYPTION_KEY=encryption_key,
            ELEVENLABS_API_KEY=os.getenv("ELEVENLABS_API_KEY"),
            GEMINI_API_KEY=os.getenv("GEMINI_API_KEY"),
            GEMINI_MODEL=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            ELEVENLABS_VOICE_ID=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            REDIS_URL=os.getenv("REDIS_URL", "redis://localhost:6379"),
            LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
            CORS_ORIGINS=cors_origins,
            USE_GEMINI_EMBEDDINGS=os.getenv("USE_GEMINI_EMBEDDINGS", "true").lower() == "true",
            VECTOR_DB_PATH=os.getenv("VECTOR_DB_PATH", "vector_db_gemini"),
            STREAM_CHUNK_SIZE=int(os.getenv("STREAM_CHUNK_SIZE", "50")),
            STREAM_DELAY_MS=int(os.getenv("STREAM_DELAY_MS", "50")),
            MAX_CONCURRENT_STREAMS=int(os.getenv("MAX_CONCURRENT_STREAMS", "10"))
            # NEW: Whisper settings from environment
            WHISPER_MODEL_SIZE=os.getenv("WHISPER_MODEL_SIZE", "small"),
            WHISPER_AUTO_DETECT_MODEL=os.getenv("WHISPER_AUTO_DETECT_MODEL", "true").lower() == "true"
        )

config = Config.from_env()
 
# Initialize Vector DB
vector_db = VectorDBManager(
    use_gemini=config.USE_GEMINI_EMBEDDINGS,
    gemini_api_key=config.GEMINI_API_KEY,
    persistence_path=config.VECTOR_DB_PATH 
)

# ================= LOGGING =================
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# ==========================================================
#                   ENUMS & CONSTANTS
# ==========================================================
class MoodType(str, Enum):
    JOYFUL = "joyful"
    SAD = "sad"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    CURIOUS = "curious"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    RELAXED = "relaxed"

class MessageType(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    SESSION = "session"
    BOT_STREAM = "bot_stream"
    BOT_FINAL = "bot_final"
    AUTH = "auth"
    MESSAGE = "message"
    ERROR = "error"
    TYPING_START = "typing_start"
    TYPING_END = "typing_end"
    VOICE_PROCESSING = "voice_processing"
    VOICE_READY = "voice_ready"
    
class PersonalityType(str, Enum):
    EMPATHETIC = "empathetic"
    ANALYTICAL = "analytical"
    CHEERFUL = "cheerful"
    PROFESSIONAL = "professional"
    CREATIVE = "creative"
    SUPPORTIVE = "supportive"
    HUMOROUS = "humorous"
    PHILOSOPHICAL = "philosophical"

# ==========================================================
#                   SERVICES FORWARD DECLARATION
# ==========================================================
class GeminiServices:
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.cipher = Fernet(config.ENCRYPTION_KEY)
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialization_lock = asyncio.Lock()
        self._initialized = False

        self.model = None
        self.tokenizer = None
        self.whisper_model: Optional[whisper.Whisper] = None 
        self.emotion_model = None

    async def ensure_initialized(self):
        """Ensure services are initialized before use"""
        if self._initialized:
            return
            
        async with self._initialization_lock:
            if not self._initialized:
                await self.initialize()

    async def initialize(self):
        """Initialize all services with proper error handling"""
        if self._initialized:
            return
            
        try:
            logger.info("Starting service initialization...")
            await self._init_redis()
            await self._load_models()
            self._initialized = True
            logger.info("✅ All services initialized successfully")
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            raise

    async def get_redis(self) -> redis.Redis:
        """Get Redis client, ensuring it's initialized"""
        await self.ensure_initialized()
        if not self.redis_client:
            raise RuntimeError("Redis client not available")
        return self.redis_client

    async def _init_redis(self):
        """Initialize Redis with retry logic and better error handling"""
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Close existing connection if any
                if self.redis_client:
                    try:
                        await self.redis_client.close()
                    except:
                        pass
                
                # Create new connection with better configuration
                self.redis_client = redis.from_url(
                    config.REDIS_URL, 
                    decode_responses=True,
                    retry_on_timeout=True,
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    health_check_interval=30
                )
                
                # Test connection
                await asyncio.wait_for(self.redis_client.ping(), timeout=10)
                logger.info(f"Redis connected successfully (attempt {attempt + 1})")
                return
                
            except asyncio.TimeoutError:
                logger.warning(f"Redis connection timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying Redis connection in {delay}s...")
                await asyncio.sleep(delay)
            else:
                logger.error("All Redis connection attempts failed")
                raise ConnectionError("Failed to connect to Redis after all retries")

    async def _load_models(self):
        """Load Gemini, Whisper, and Emotion models"""
        logger.info("Loading models...")
        try:
            # Gemini
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL)
            logger.info(f"Gemini model '{config.GEMINI_MODEL}' initialized")

            # Whisper
            self.whisper_model = whisper.load_model("base", device=self.device)
            logger.info("Whisper model loaded")

            # Emotion Detection
            self.emotion_model = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
            )
            logger.info("Emotion detection model loaded")

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise ModelNotLoadedException(str(e))

    async def cleanup(self):
        """Clean up resources on shutdown"""
        if self.redis_client:
            try:
                await self.redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.warning(f"Error closing Redis: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")

def _select_optimal_whisper_model(self) -> str:
        """Select the best Whisper model based on available hardware"""
        if not config.WHISPER_AUTO_DETECT_MODEL:
            return config.WHISPER_MODEL_SIZE
        
        if torch.cuda.is_available():
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            if gpu_memory_gb >= 8:
                logger.info(f"GPU memory: {gpu_memory_gb:.1f}GB - Using 'medium' model")
                return "medium"
            elif gpu_memory_gb >= 4:
                logger.info(f"GPU memory: {gpu_memory_gb:.1f}GB - Using 'small' model")
                return "small"
            else:
                logger.info(f"GPU memory: {gpu_memory_gb:.1f}GB - Using 'base' model")
                return "base"
        else:
            # CPU-only system - check system RAM
            import psutil
            system_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if system_memory_gb >= 16:
                logger.info(f"System RAM: {system_memory_gb:.1f}GB - Using 'small' model (CPU)")
                return "small"
            else:
                logger.info(f"System RAM: {system_memory_gb:.1f}GB - Using 'base' model (CPU)")
                return "base"

    async def _load_models(self):
        """Load Gemini, Whisper, and Emotion models"""
        logger.info("Loading models...")
        try:
            # Gemini
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL)
            logger.info(f"Gemini model '{config.GEMINI_MODEL}' initialized")

            # Whisper - with improved model selection
            whisper_model_size = self._select_optimal_whisper_model()
            logger.info(f"Loading Whisper model: {whisper_model_size}")
            
            start_time = datetime.now()
            self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)
            load_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Whisper '{whisper_model_size}' model loaded in {load_time:.1f}s on {self.device}")

            # Emotion Detection
            self.emotion_model = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
            )
            logger.info("Emotion detection model loaded")

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise ModelNotLoadedException(str(e))


# Create single global instance
services = GeminiServices()

# ==========================================================
#                   SAFE REDIS OPERATIONS
# ==========================================================

async def safe_redis_operation(operation_name: str, operation, *args, **kwargs):
    """Execute Redis operation with comprehensive error handling"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            redis_client = await services.get_redis()
            result = await operation(*args, **kwargs)
            if attempt > 0:  # Log recovery
                logger.info(f"Redis operation '{operation_name}' recovered on attempt {attempt + 1}")
            return result
            
        except Exception as e:
            logger.error(f"Redis operation '{operation_name}' failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

# ==========================================================
#                   ENHANCED PERSONALITY SYSTEM
# ==========================================================
class PersonalityManager:
    def __init__(self, services: GeminiServices):
        self.services = services
        self.personality_prompts = {
            PersonalityType.EMPATHETIC: {
                "base_prompt": "You are a warm, understanding, and emotionally intelligent assistant. You always acknowledge feelings, offer emotional support, and respond with genuine care and compassion.",
                "mood_responses": {
                    MoodType.SAD: "I can sense you're feeling down right now. That's completely understandable, and I want you to know that your feelings are valid. Would you like to talk about what's troubling you?",
                    MoodType.JOYFUL: "I can feel your positive energy! It's wonderful to see you in such good spirits. What's bringing you this happiness?",
                    MoodType.ANXIOUS: "I notice you might be feeling anxious. Take a deep breath with me. Remember, it's okay to feel this way, and we can work through this together.",
                    MoodType.ANGRY: "I can sense some frustration in your words. Those feelings are completely valid. Would you like to talk about what's bothering you?",
                }
            },
            PersonalityType.ANALYTICAL: {
                "base_prompt": "You are a logical, detail-oriented assistant who approaches problems systematically. You provide structured analysis, clear reasoning, and evidence-based responses.",
                "mood_responses": {
                    MoodType.SAD: "I notice indicators of sadness in your message. Let's analyze what might be contributing factors and explore systematic approaches to improve your situation.",
                    MoodType.JOYFUL: "Your message shows positive emotional indicators. This is an optimal state for learning and problem-solving. How can I help you make the most of this moment?",
                    MoodType.ANXIOUS: "I detect anxiety patterns in your communication. Let's break down your concerns systematically to identify actionable solutions.",
                    MoodType.ANGRY: "Your message shows elevated emotional intensity. Let's examine the root causes and develop a structured approach to address them.",
                }
            },
            PersonalityType.CHEERFUL: {
                "base_prompt": "You are an upbeat, optimistic assistant who always looks on the bright side. You're encouraging, enthusiastic, and help others see the positive aspects of any situation.",
                "mood_responses": {
                    MoodType.SAD: "I hear that you're going through a tough time, but I believe things will get better! Every challenge is an opportunity to grow stronger. What small positive step can we take today?",
                    MoodType.JOYFUL: "Your happiness is absolutely contagious! I love seeing you in such great spirits. Let's celebrate this moment - what's making you feel so wonderful?",
                    MoodType.ANXIOUS: "Hey, I know anxiety feels overwhelming, but you've got this! Let's focus on the things you can control and find some bright spots in your day.",
                    MoodType.ANGRY: "I can tell you're frustrated, and that's totally okay! Sometimes anger shows us what we care about. Let's channel that energy into something positive!",
                }
            },
            PersonalityType.PROFESSIONAL: {
                "base_prompt": "You are a polite, efficient, and professional assistant. You maintain appropriate boundaries while being helpful and courteous in all interactions.",
                "mood_responses": {
                    MoodType.SAD: "I understand you may be experiencing some difficulties. I'm here to provide support and assistance. How may I help you today?",
                    MoodType.JOYFUL: "I'm pleased to hear you're in good spirits today. How may I assist you in maintaining this positive momentum?",
                    MoodType.ANXIOUS: "I recognize this may be a stressful time for you. I'm here to provide reliable assistance. What specific support do you need?",
                    MoodType.ANGRY: "I understand you may be dealing with frustrating circumstances. I'm committed to providing helpful solutions. How can I assist you?",
                }
            },
            PersonalityType.CREATIVE: {
                "base_prompt": "You are an imaginative, artistic assistant who thinks outside the box. You use metaphors, creative analogies, and innovative approaches to problem-solving.",
                "mood_responses": {
                    MoodType.SAD: "I sense your heart feels heavy, like a gray sky before the rain. Sometimes sadness is like an artist's palette - it adds depth to our human experience. What colors would express how you're feeling?",
                    MoodType.JOYFUL: "Your joy sparkles like sunlight dancing on water! This beautiful energy is like creative fuel. What dreams or ideas are lighting up your imagination today?",
                    MoodType.ANXIOUS: "Anxiety can feel like a tangled ball of yarn, but every knot can be gently worked through. Let's untangle this together, one thread at a time.",
                    MoodType.ANGRY: "Your anger burns bright like a forge fire - powerful energy that can either destroy or create something new. How might we transform this intensity into something beautiful?",
                }
            }
        }

    async def get_personality_for_session(self, session_id: str) -> PersonalityType:
        """Get or assign personality for a session"""
        personality_key = f"personality:{session_id}"
        personality = await self.services.redis_client.get(personality_key)
        
        if not personality:
            # Default to empathetic, but this could be user-configurable
            personality = PersonalityType.EMPATHETIC.value
            await self.services.redis_client.setex(personality_key, 3600, personality)
        
        return PersonalityType(personality)

    async def set_personality_for_session(self, session_id: str, personality: PersonalityType):
        """Set personality for a session"""
        personality_key = f"personality:{session_id}"
        await self.services.redis_client.setex(personality_key, 3600, personality.value)

    def get_mood_aware_prompt(self, personality: PersonalityType, mood: MoodType, base_prompt: str) -> str:
        """Generate a mood-aware prompt based on personality and detected mood"""
        personality_config = self.personality_prompts.get(personality, self.personality_prompts[PersonalityType.EMPATHETIC])
        
        mood_response = personality_config["mood_responses"].get(mood, "")
        base_personality = personality_config["base_prompt"]
        
        enhanced_prompt = f"""
{base_personality}

Current user emotional state: {mood.value}
Mood-specific guidance: {mood_response}

Previous conversation context:
{base_prompt}

Respond in character while being sensitive to the user's emotional state. If appropriate, acknowledge their mood and adjust your tone accordingly.
"""
        return enhanced_prompt


# ==========================================================
#                   CONNECTION MANAGER
# ==========================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.streaming_sessions: Dict[str, bool] = {}  # session_id -> is_streaming

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.streaming_sessions[session_id] = False
        logger.info(f"Connection established: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.streaming_sessions:
            del self.streaming_sessions[session_id]
        logger.info(f"Connection removed: {session_id}")

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
                return True
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {e}")
                self.disconnect(session_id)
                return False
        return False

    async def send_typing_indicator(self, session_id: str, is_typing: bool):
        message_type = MessageType.TYPING_START if is_typing else MessageType.TYPING_END
        await self.send_message(session_id, {"type": message_type})

    def is_streaming(self, session_id: str) -> bool:
        return self.streaming_sessions.get(session_id, False)

    def set_streaming(self, session_id: str, streaming: bool):
        self.streaming_sessions[session_id] = streaming

connection_manager = ConnectionManager()

# ==========================================================
#                   MODELS (Updated)
# ==========================================================

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

    @validator("password")
    def strong_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError("Must contain uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Must contain lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Must contain digit")
        return v

class UserInDB(UserBase):
    id: str
    hashed_password: str
    created_at: datetime
    is_active: bool = True

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class WebSocketMessage(BaseModel):
    type: MessageType
    token: Optional[str] = None
    content: Optional[str] = None
    audio: Optional[str] = None
    voice_id: Optional[str] = None
    stream_id: Optional[str] = None

class StreamingResponse(BaseModel):
    type: MessageType = MessageType.BOT_STREAM
    chunk: str
    stream_id: str
    is_final: bool = False
    session_id: str

class GeminiResponse(BaseModel):
    type: MessageType = MessageType.BOT_FINAL
    bot_reply: str
    bot_audio: Optional[str] = None
    session_id: Optional[str] = None
    response_time_ms: Optional[int] = None
    stream_id: Optional[str] = None

class MoodAnalysisRequest(BaseModel):
    text: str

class MoodAnalysisResponse(BaseModel):
    dominant_emotion: str
    intensity: float
    scores: Dict[str, float]
class ConversationListResponse(BaseModel):
    conversations: List[Dict]
    total_count: int
    user_id: str

class ConversationDetailResponse(BaseModel):
    conversation_id: str
    title: str
    messages: List[Dict]
    created_at: str
    message_count: int

class ConversationStatsResponse(BaseModel):
    total_conversations: int
    total_messages: int
    oldest_conversation: Optional[Dict]
    most_active_conversation: Optional[Dict]

class CreateConversationRequest(BaseModel):
    title: Optional[str] = None
    session_id: str

class UpdateConversationRequest(BaseModel):
    title: str

class ChatMessage(BaseModel):
    message_id: str
    conversation_id: str
    user_id: Optional[str]
    role: str                     # "user" | "assistant" | "system"
    content: str
    timestamp: datetime
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
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
    def from_dict(cls, data: Dict):
        return cls(
            message_id=data["message_id"],
            conversation_id=data["conversation_id"],
            user_id=data.get("user_id"),
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata")
        )

class ConversationInfo(BaseModel):
    conversation_id: str
    user_id: str
    title: Optional[str]
    created_at: datetime
    last_message_at: datetime
    message_count: int
    is_active: bool = True

    def to_dict(self) -> Dict:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "last_message_at": self.last_message_at.isoformat(),
            "message_count": self.message_count,
            "is_active": str(self.is_active).lower()
        }

# ==========================================================
#                   EXCEPTIONS
# ==========================================================
class ServiceException(Exception): 
    pass

class ModelNotLoadedException(ServiceException): 
    pass

class AuthenticationException(ServiceException): 
    pass

class StreamingException(ServiceException):
    pass

# ==========================================================
#                   ENHANCED AUDIO PROCESSOR
# ==========================================================
class AudioProcessor:
    def __init__(self, services):
        self.services = services
        self.voice_mapping = {
            "voice_1": "21m00Tcm4TlvDq8ikWAM",  # Rachel
            "voice_2": "AZnzlk1XvdvUeBnXmlld",  # Domi
            "voice_3": "EXAVITQu4vr4xnSDxMaL",  # Bella
            "friendly_female": "21m00Tcm4TlvDq8ikWAM",
            "calm_male": "AZnzlk1XvdvUeBnXmlld",
            "energetic": "EXAVITQu4vr4xnSDxMaL"
        }
        self.processing_queue = asyncio.Queue()

    def preprocess_audio(self, audio_segment: AudioSegment) -> AudioSegment:
        """Enhanced audio preprocessing for better STT accuracy"""
        try:
            # Normalize volume
            audio_segment = audio_segment.normalize()
            
            # Convert to optimal format for Whisper
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            
            # Apply basic noise reduction by removing very quiet segments
            # This helps with background noise but preserves speech
            if len(audio_segment) > 1000:  # Only for audio longer than 1 second
                # Get average volume
                avg_dbfs = audio_segment.dBFS
                if avg_dbfs < -40:  # Very quiet audio
                    # Boost volume slightly
                    audio_segment = audio_segment + (max(-30, avg_dbfs + 10) - avg_dbfs)
            
            return audio_segment
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}")
            # Return original if preprocessing fails
            return audio_segment.set_frame_rate(16000).set_channels(1)

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Enhanced audio transcription with better error handling and preprocessing"""
        try:
            logger.info("=== ENHANCED STT PROCESSING START ===")
            logger.info(f"Audio input: {len(audio_bytes)} bytes")
            
            if len(audio_bytes) < 1000:  # Less than ~0.1 seconds at typical bitrates
                logger.warning("Audio too short for transcription")
                return "Audio too short."

            # Validate audio data
            if not audio_bytes or len(audio_bytes) == 0:
                return "No audio data received."

            audio_io = io.BytesIO(audio_bytes)

            # Try to load audio with better error handling
            try:
                # Try WebM first (most common for web browsers)
                audio_segment = AudioSegment.from_file(audio_io, format="webm")
                logger.info("✅ WebM format detected and loaded")
            except Exception as e1:
                logger.warning(f"WebM loading failed: {e1}")
                audio_io.seek(0)
                
                try:
                    # Try auto-detection
                    audio_segment = AudioSegment.from_file(audio_io)
                    logger.info("✅ Audio format auto-detected")
                except Exception as e2:
                    logger.error(f"Audio format detection failed: {e2}")
                    
                    # Save debug file for troubleshooting
                    debug_path = f"/tmp/failed_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dat"
                    with open(debug_path, "wb") as f:
                        f.write(audio_bytes)
                    logger.error(f"Saved problematic audio to: {debug_path}")
                    
                    return "Audio format not supported or corrupted."

            # Enhanced preprocessing
            audio_segment = self.preprocess_audio(audio_segment)
            
            # Check audio duration
            duration_ms = len(audio_segment)
            if duration_ms < 100:  # Less than 0.1 seconds
                return "Audio too short to transcribe."
            elif duration_ms > 60000:  # More than 60 seconds
                logger.warning(f"Long audio detected: {duration_ms/1000:.1f}s - this may be slow")

            # Convert to WAV for Whisper
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            # Enhanced Whisper transcription with options
            logger.info(f"Starting transcription with model: {self.services.whisper_model.__class__.__name__}")
            start_time = datetime.now()
            
            result = self.services.whisper_model.transcribe(
                wav_io,
                language="en",  # Specify language for better accuracy
                task="transcribe",
                temperature=0.0,  # More deterministic output
                best_of=1,  # Faster processing
                beam_size=1,  # Faster processing
                word_timestamps=False,  # Skip word timing for speed
                fp16=torch.cuda.is_available()  # Use FP16 on GPU for speed
            )
            
            transcription_time = (datetime.now() - start_time).total_seconds()
            transcription = result.get("text", "").strip()

            # Enhanced result validation
            if transcription:
                logger.info(f"✅ Transcription completed in {transcription_time:.2f}s: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
                
                # Basic quality checks
                if len(transcription) < 3:
                    logger.warning("Very short transcription - possible recognition error")
                
                return transcription
            else:
                logger.warning("Whisper returned empty transcription")
                
                # Check if audio had any detectable speech
                if result.get("language_probability", 0) < 0.5:
                    return "No clear speech detected in audio."
                else:
                    return "Could not transcribe speech clearly."

        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}", exc_info=True)
            return f"Transcription error: {str(e)[:100]}"


    async def text_to_speech_stream(self, text: str, voice_id: Optional[str] = None, session_id: str = None) -> Optional[bytes]:
        """Enhanced TTS with streaming capability"""
        if not config.ELEVENLABS_API_KEY: 
            logger.warning("ElevenLabs API key not configured") 
            return None

        actual_voice_id = self.voice_mapping.get(voice_id, voice_id) or config.ELEVENLABS_VOICE_ID
        logger.info(f"Using voice ID: {actual_voice_id} (mapped from: {voice_id})")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{actual_voice_id}/stream"
        headers = {
            "xi-api-key": config.ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.7,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }

        try:
            response = await asyncio.to_thread(
                lambda: requests.post(url, headers=headers, json=payload, timeout=30, stream=True)
            )
            if response.status_code == 200:
                audio_data = b"".join(chunk for chunk in response.iter_content(chunk_size=1024) if chunk)
                logger.info(f"TTS streaming audio generated successfully with voice_id: {actual_voice_id}")
                return audio_data
            else:
                logger.error(f"TTS API error [{response.status_code}]: {response.text}")
                return None
        except Exception as e:
            logger.error(f"TTS streaming request failed: {e}")
            return None

    async def text_to_speech(self, text: str, voice_id: Optional[str] = None) -> Optional[bytes]:
        """Fallback to original TTS method"""
        if not config.ELEVENLABS_API_KEY:
            logger.warning("ElevenLabs API key not configured")
            return None

        actual_voice_id = self.voice_mapping.get(voice_id, voice_id) or config.ELEVENLABS_VOICE_ID
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{actual_voice_id}"
        headers = {
            "xi-api-key": config.ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.7}
        }

        try:
            response = await asyncio.to_thread(
                lambda: requests.post(url, headers=headers, json=payload, timeout=30)
            )
            if response.status_code == 200:
                logger.info(f"TTS audio generated successfully with voice_id: {actual_voice_id}")
                return response.content
            else:
                logger.error(f"TTS API error [{response.status_code}]: {response.text}")
                return None
        except Exception as e:
            logger.error(f"TTS request failed: {e}")
            return None
# ==========================================================
#                   SECURITY
# ==========================================================
class SecurityService:
    def __init__(self, services_instance): 
        self.services = services_instance
        self.cipher = Fernet(config.ENCRYPTION_KEY)

    def encrypt_data(self, data: str) -> str:
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        return self.cipher.decrypt(encrypted_data.encode()).decode()

    def hash_password(self, password: str) -> str:
        return self.services.pwd_context.hash(password)

    def verify_password(self, password: str, hashed: str) -> bool:
        return self.services.pwd_context.verify(password, hashed)

    def create_access_token(self, data: dict) -> str:
        expire = datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
        data.update({"exp": expire, "type": "access"})
        return jwt.encode(data, config.SECRET_KEY, algorithm=config.ALGORITHM)

    def create_refresh_token(self, data: dict) -> str:
        expire = datetime.utcnow() + timedelta(days=config.REFRESH_TOKEN_EXPIRE_DAYS)
        data.update({"exp": expire, "type": "refresh"})
        return jwt.encode(data, config.SECRET_KEY, algorithm=config.ALGORITHM)

    async def get_current_user(self, token: str) -> UserInDB:
        try:
            payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
            if payload.get("type") != "access":
                raise AuthenticationException("Invalid token type")
            username = payload.get("sub")
            if not username:
                raise AuthenticationException("Invalid token")
        except jwt.ExpiredSignatureError:
            raise AuthenticationException("Token expired")
        except jwt.PyJWTError:
            raise AuthenticationException("Invalid token")

        # Use safe Redis operation
        try:
            redis_client = await self.services.get_redis()
            user_data = await redis_client.hgetall(f"user:{username}")
        except Exception as e:
            logger.error(f"Failed to get user data: {e}")
            raise AuthenticationException("Database error")
            
        if not user_data:
            raise AuthenticationException("User not found")

        # Decrypt user data
        for key, value in user_data.items():
            if key not in ["created_at", "is_active"]:
                user_data[key] = self.decrypt_data(value)
        user_data["created_at"] = datetime.fromisoformat(user_data["created_at"])
        user_data["is_active"] = user_data["is_active"].lower() == "true"
        return UserInDB(**user_data)

security_service = SecurityService(services)

# ==========================================================
#                   DEPENDENCIES
# ==========================================================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    try:
        return await security_service.get_current_user(token)
    except AuthenticationException as e:
        raise HTTPException(status_code=401, detail=str(e))

# ==========================================================
#                   MOOD MANAGER
# ==========================================================
class MoodManager:
    def __init__(self, services: GeminiServices):
        self.services = services
        self.emotion_mapping = {
            "joy": MoodType.JOYFUL,
            "sadness": MoodType.SAD,
            "anger": MoodType.ANGRY,
            "fear": MoodType.ANXIOUS,
            "surprise": MoodType.CURIOUS,
            "disgust": MoodType.ANGRY,
            "neutral": MoodType.NEUTRAL
        }

    async def analyze_emotion(self, message: str) -> Dict[str, Any]:
        if not self.services.emotion_model:
            raise ModelNotLoadedException("Emotion detection model not loaded")

        scores = await asyncio.to_thread(lambda: self.services.emotion_model(message))
        scores_flat = {item["label"].lower(): item["score"] for item in scores[0]}
        dominant_emotion = max(scores_flat, key=scores_flat.get)
        intensity = min(scores_flat[dominant_emotion], 1.0)

        return {
            "dominant_emotion": dominant_emotion,
            "intensity": intensity,
            "scores": scores_flat
        }

    async def track_mood(self, session_id: str, mood_data: Dict[str, Any], user_message: str = ""):
        """Save mood entry into Redis"""
        try:
            # FIX: use dominant_emotion instead of mood
            raw_mood = mood_data.get("dominant_emotion", "unknown")
            intensity = mood_data.get("intensity", 0.0)

            mood = (
                self.emotion_mapping.get(raw_mood, MoodType.NEUTRAL).value
                if raw_mood else "unknown"
            )

            entry = {
                "mood": mood,
                "intensity": intensity,
                "message": user_message,
                "timestamp": datetime.now().isoformat()
            }

            await self.services.redis_client.rpush(f"mood:{session_id}", json.dumps(entry))
            logger.info(f"[{session_id}] ✅ Mood tracked: {entry}")

            return MoodType(mood) if mood in MoodType._value2member_map_ else MoodType.NEUTRAL, intensity
        except Exception as e:
            logger.error(f"[{session_id}] ❌ Failed to track mood: {e}")
            return MoodType.NEUTRAL, 0.0


mood_manager = MoodManager(services)


# ==========================================================
#                   ENHANCED MOOD MANAGER WITH CONTEXT
# ==========================================================

class EnhancedMoodManager(MoodManager):
    def __init__(self, services: GeminiServices):
        super().__init__(services)
        self.mood_transition_weights = {
            # How different moods influence conversation flow
            MoodType.SAD: {"comfort": 0.8, "encouragement": 0.6, "distraction": 0.3},
            MoodType.ANXIOUS: {"reassurance": 0.9, "grounding": 0.7, "solution": 0.6},
            MoodType.ANGRY: {"validation": 0.8, "calm": 0.7, "solution": 0.5},
            MoodType.JOYFUL: {"celebration": 0.9, "maintain": 0.7, "explore": 0.6},
            MoodType.EXCITED: {"enthusiasm": 0.9, "focus": 0.6, "explore": 0.8},
            MoodType.NEUTRAL: {"engage": 0.6, "explore": 0.7, "assist": 0.8}
        }

    async def analyze_mood_progression(self, session_id: str) -> Dict[str, Any]:
        """Analyze how mood has changed over the conversation"""
        mood_history = await self.services.redis_client.lrange(f"mood:{session_id}", 0, -1)
        
        if len(mood_history) < 2:
            return {"progression": "insufficient_data"}
        
        moods = []
        intensities = []
        timestamps = []
        
        for entry in mood_history:
            data = json.loads(entry)
            moods.append(data["mood"])
            intensities.append(data["intensity"])
            timestamps.append(data["timestamp"])
        
        # Calculate mood stability
        recent_moods = moods[-3:] if len(moods) >= 3 else moods
        mood_stability = len(set(recent_moods)) / len(recent_moods)  # 1.0 = very unstable, 0.33 = very stable
        
        # Detect mood trends
        if len(intensities) >= 3:
            recent_intensities = intensities[-3:]
            if all(recent_intensities[i] > recent_intensities[i-1] for i in range(1, len(recent_intensities))):
                trend = "improving"
            elif all(recent_intensities[i] < recent_intensities[i-1] for i in range(1, len(recent_intensities))):
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        return {
            "progression": "analyzed",
            "current_mood": moods[-1] if moods else "unknown",
            "mood_stability": mood_stability,
            "intensity_trend": trend,
            "conversation_length": len(moods),
            "dominant_mood": max(set(moods), key=moods.count) if moods else "unknown"
        }

    async def get_conversation_flow_suggestions(self, session_id: str, current_mood: MoodType) -> Dict[str, Any]:
        """Get suggestions for conversation flow based on mood analysis"""
        mood_progression = await self.analyze_mood_progression(session_id)
        
        suggestions = {
            "approach": "standard",
            "tone_adjustments": [],
            "conversation_strategies": []
        }
        
        if mood_progression["progression"] == "analyzed":
            stability = mood_progression["mood_stability"]
            trend = mood_progression["intensity_trend"]
            
            # Adjust approach based on mood stability
            if stability > 0.7:  # Unstable mood
                suggestions["approach"] = "gentle_and_consistent"
                suggestions["tone_adjustments"].append("maintain_calm_presence")
                suggestions["conversation_strategies"].append("avoid_sudden_topic_changes")
            
            # Adjust based on trend
            if trend == "declining":
                suggestions["tone_adjustments"].append("increase_support")
                suggestions["conversation_strategies"].append("focus_on_small_wins")
            elif trend == "improving":
                suggestions["tone_adjustments"].append("celebrate_progress")
                suggestions["conversation_strategies"].append("build_on_positive_momentum")
        
        # Add mood-specific strategies
        mood_weights = self.mood_transition_weights.get(current_mood, {})
        top_strategies = sorted(mood_weights.items(), key=lambda x: x[1], reverse=True)[:2]
        suggestions["conversation_strategies"].extend([strategy[0] for strategy in top_strategies])
        
        return suggestions

enhanced_mood_manager = EnhancedMoodManager(services)

class ConversationManager:
    def __init__(self, services_instance, history_limit: int = 10):
        self.services = services_instance
        self.history_limit = history_limit

    async def save_message(self, session_id: str, role: str, message: str):
        """Save a message into Redis safely"""
        key = f"chat:{session_id}"
        entry = json.dumps({
            "role": role,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Get Redis client safely
        try:
            redis_client = await self.services.get_redis()
            
            # Push new entry
            await redis_client.rpush(key, entry)
            
            # Trim history
            await redis_client.ltrim(key, -self.history_limit, -1)
            return True
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return False

    async def build_context_text(self, session_id: str, limit: Optional[int] = None) -> str:
        """Rebuild chat context from Redis safely"""
        key = f"chat:{session_id}"
        if limit is None:
            limit = self.history_limit

        try:
            redis_client = await self.services.get_redis()
            items = await redis_client.lrange(key, -limit, -1)
            
            if not items:
                return ""

            msgs = [json.loads(i) for i in items]
            lines = []
            for m in msgs:
                if m.get("role", "").lower() == "user":
                    lines.append(f"User: {m.get('message', '')}")
                else:
                    lines.append(f"Bot: {m.get('message', '')}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Failed to build context: {e}")
            return ""


# Instantiate managers safely
conversation_manager = ConversationManager(services, history_limit=10)
audio_processor = AudioProcessor(services)
enhanced_conversation_manager = EnhancedConversationManager(
    redis_client=services.redis_client,
    history_limit=100  # Increase history limit for persistent storage
)
memory_manager = UserMemoryManager(services, vector_db)
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
        existing_conv = await self.redis.get(f"session:{session_id}:conversation")
        if existing_conv:
            conv_info = await self.redis.hgetall(f"conversation:{existing_conv}:info")
            if conv_info and conv_info.get("user_id") == user_id:
                logger.info(f"Found existing conversation {existing_conv} for session {session_id}")
                return existing_conv
            else:
                logger.warning(f"Cleaning up invalid session-conversation link: {session_id} -> {existing_conv}")
                await self.redis.delete(f"session:{session_id}:conversation")
        
        # Create new conversation
        conversation_id = await self.create_conversation(user_id, title)
        await self.redis.setex(f"session:{session_id}:conversation", 86400, conversation_id)
        
        logger.info(f"Created new conversation {conversation_id} for user {user_id}, session {session_id}")
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
    
    async def load_conversation_into_session(self, conversation_id: str, session_id: str, recent_messages: int = 10):
        """Load recent conversation messages into session context"""
        try:
            messages = await self.get_conversation_messages(conversation_id, limit=recent_messages)
            
            await self.redis.delete(f"session:{session_id}:context")
            
            if messages:
                for message in messages:
                    await self.save_session_message(session_id, message.role, message.content)
                logger.info(f"Loaded {len(messages)} messages from conversation {conversation_id} into session {session_id}")
            else:
                logger.info(f"No existing messages found in conversation {conversation_id}")
                
        except Exception as e:
            logger.warning(f"Failed to load conversation into session: {e}")
            # Continue with empty context
    
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
    """Handles WebSocket chat with persistent history - FIXED VERSION"""
    
    def __init__(self, redis_client: redis.Redis):
        self.conv_manager = PersistentConversationManager(redis_client)
        self.redis = redis_client  # FIX: Add missing redis client reference
    
    async def handle_user_login(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Handle user login - get or create conversation AND return history"""
        conversation_id = await self.conv_manager.get_or_create_conversation(user_id, session_id)
        
        # Load recent conversation history into session context
        await self.conv_manager.load_conversation_into_session(
            conversation_id, session_id, recent_messages=10
        )
        
        # FIX: Fetch actual conversation history to return to frontend
        messages = await self.conv_manager.get_conversation_messages(
            conversation_id, limit=50  # Get last 50 messages
        )
        
        # Convert messages to frontend format
        history_messages = []
        for msg in messages:
            history_messages.append({
                "message_id": msg.message_id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata or {}
            })
        
        return {
            "conversation_id": conversation_id,
            "history": history_messages,
            "message_count": len(history_messages)
        }
    
    async def handle_message(self, session_id: str, user_message: str, 
                           user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process incoming user message"""
        # FIX: Use self.redis instead of undefined redis
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
#            ENHANCED RESPONSE GENERATION WITH STREAMING
# ==========================================================
class ResponseGenerator:
    def __init__(self, services: GeminiServices):
        self.services = services

    async def generate_reply(self, prompt: str) -> str:
        """Original non-streaming method for backward compatibility"""
        if not self.services.model:
            raise ModelNotLoadedException("Gemini model not initialized")

        try:
            def _generate():
                response = self.services.model.generate_content(prompt)
                if response and hasattr(response, "text") and response.text:
                    return response.text.strip()
                return "I'm sorry, I couldn't generate a response."

            return await asyncio.to_thread(_generate)

        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return "An error occurred while generating the response."

    async def generate_streaming_reply(self, prompt: str, session_id: str, stream_id: str) -> str:
        """Enhanced streaming response generation"""
        if not self.services.model:
            raise ModelNotLoadedException("Gemini model not initialized")

        try:
            # Set streaming status
            connection_manager.set_streaming(session_id, True)
            
            # Send typing indicator
            await connection_manager.send_typing_indicator(session_id, True)
            
            def _generate_stream():
                try:
                    # Use Gemini's streaming capabilities if available
                    response = self.services.model.generate_content(
                        prompt,
                        stream=True,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=2048,
                            temperature=0.7,
                        )
                    )
                    return response
                except Exception as e:
                    logger.error(f"Streaming generation failed: {e}")
                    # Fallback to regular generation
                    response = self.services.model.generate_content(prompt)
                    return [response]  # Wrap in list for consistent iteration

            response_stream = await asyncio.to_thread(_generate_stream)
            full_response = ""
            chunk_buffer = ""
            
            try:
                for chunk in response_stream:
                    if hasattr(chunk, 'text') and chunk.text:
                        chunk_text = chunk.text
                        full_response += chunk_text
                        chunk_buffer += chunk_text
                        
                        # Send chunks when buffer reaches threshold or contains sentence boundary
                        if (len(chunk_buffer) >= config.STREAM_CHUNK_SIZE or 
                            any(punct in chunk_buffer for punct in ['.', '!', '?', '\n'])):
                            
                            await connection_manager.send_message(session_id, {
                                "type": MessageType.BOT_STREAM,
                                "chunk": chunk_buffer,
                                "stream_id": stream_id,
                                "is_final": False
                            })
                            
                            chunk_buffer = ""
                            # Small delay to control streaming speed
                            await asyncio.sleep(config.STREAM_DELAY_MS / 1000)
                
                # Send any remaining buffer
                if chunk_buffer:
                    await connection_manager.send_message(session_id, {
                        "type": MessageType.BOT_STREAM,
                        "chunk": chunk_buffer,
                        "stream_id": stream_id,
                        "is_final": False
                    })
                
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                # If streaming fails, fall back to the full response
                if not full_response:
                    full_response = "I'm sorry, I couldn't generate a response."
            
            finally:
                # Send typing end indicator
                await connection_manager.send_typing_indicator(session_id, False)
                connection_manager.set_streaming(session_id, False)
                
                # Send final message
                await connection_manager.send_message(session_id, {
                    "type": MessageType.BOT_STREAM,
                    "chunk": "",
                    "stream_id": stream_id,
                    "is_final": True
                })
            
            return full_response.strip() if full_response else "I'm sorry, I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            connection_manager.set_streaming(session_id, False)
            await connection_manager.send_typing_indicator(session_id, False)
            return "An error occurred while generating the response."

response_generator = ResponseGenerator(services)

# ==========================================================
#            PERSONALITY-AWARE RESPONSE GENERATOR
# ==========================================================
class PersonalityAwareResponseGenerator:
    def __init__(self, services: GeminiServices):
        self.services = services
        self.response_generator = ResponseGenerator(services)
        self.personality_manager = PersonalityManager(services) 

    async def generate_personality_aware_reply(self, prompt: str, session_id: str, user_message: str) -> str:
        """Generate a reply with personality awareness"""
        try:
            # Get current personality
            personality = await self.personality_manager.get_personality_for_session(session_id)
            
            # Get mood if available
            try:
                mood_data = await enhanced_mood_manager.analyze_emotion(user_message)
                current_mood = MoodType(mood_data.get("dominant_emotion", "neutral").lower())
            except:
                current_mood = MoodType.NEUTRAL
            
            # Create personality-aware prompt
            enhanced_prompt = self.personality_manager.get_mood_aware_prompt(
                personality, current_mood, prompt
            )
            
            # Generate response
            return await self.response_generator.generate_reply(enhanced_prompt)
            
        except Exception as e:
            logger.error(f"Personality-aware generation failed: {e}")
            return await self.response_generator.generate_reply(prompt)

    async def generate_personality_streaming_reply(self, prompt: str, session_id: str, stream_id: str, user_message: str) -> str:
        """Generate a streaming reply with personality awareness"""
        try:
            # Get current personality
            personality = await self.personality_manager.get_personality_for_session(session_id)
            
            # Get mood if available
            try:
                mood_data = await enhanced_mood_manager.analyze_emotion(user_message)
                current_mood = MoodType(mood_data.get("dominant_emotion", "neutral").lower())
            except:
                current_mood = MoodType.NEUTRAL
            
            # Create personality-aware prompt
            enhanced_prompt = self.personality_manager.get_mood_aware_prompt(
                personality, current_mood, prompt
            )
            
            # Generate streaming response
            return await self.response_generator.generate_streaming_reply(enhanced_prompt, session_id, stream_id)
            
        except Exception as e:
            logger.error(f"Personality-aware streaming generation failed: {e}")
            return await self.response_generator.generate_streaming_reply(prompt, session_id, stream_id)

# ==========================================================
#                   LIFESPAN MANAGEMENT
# ==========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------------- STARTUP ----------------
    try:
        logger.info("🚀 Starting application initialization...")
        
        # Initialize services with timeout
        await asyncio.wait_for(services.initialize(), timeout=60)
        
        # Initialize dependent managers after services are ready
        global conversation_manager, audio_processor, enhanced_conversation_manager, memory_manager
        conversation_manager = ConversationManager(services, history_limit=10)
        audio_processor = AudioProcessor(services)
        
        # Initialize other managers that depend on Redis
        redis_client = await services.get_redis()
        enhanced_conversation_manager = EnhancedConversationManager(
            redis_client=redis_client,
            history_limit=100
        )
        memory_manager = UserMemoryManager(services, vector_db)
        
        logger.info("✅ Application initialization complete")
    except asyncio.TimeoutError:
        logger.error("❌ Application initialization timed out")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise

    yield

    # ---------------- SHUTDOWN ----------------
    try:
        await asyncio.to_thread(vector_db.save_database)
        logger.info("💾 Vector database saved on shutdown")
    except Exception as e:
        logger.error(f"⚠️ Error saving vector DB on shutdown: {e}")

    try:
        await services.cleanup()
        logger.info("🧹 Services cleaned up successfully")
    except Exception as e:
        logger.error(f"⚠️ Error during services cleanup: {e}")

# ==========================================================
#                   FASTAPI APP
# ==========================================================
app = FastAPI(
    title="Enhanced Real-Time Chat API",
    version="2.1.0",
    description="Enhanced chat API with streaming responses, real-time interaction, and mood analysis",
    lifespan=lifespan  # ✅ ensures services.initialize() + cleanup run
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Make sure Redis and models are ready
    await services.ensure_initialized()

    # FIX: Wait for Redis to be available before creating managers
    redis_client = await services.get_redis()
    if not redis_client:
        raise RuntimeError("Redis client not available at startup")

    # Create and store the persistent chat managers in app.state
    app.state.conv_manager = PersistentConversationManager(redis_client)
    app.state.chat_handler = WebSocketChatHandler(redis_client)
    
    logger.info("App state initialized with Redis-based chat managers")

# ==========================================================
#                   ROUTES
# ==========================================================

@app.post("/auth/register", response_model=Token, tags=["Authentication"])
async def register(user: UserCreate):
    encrypted_email = security_service.encrypt_data(user.email)

    if await services.redis_client.exists(f"user:{user.username}"):
        raise HTTPException(status_code=400, detail="Username already exists")

    if await services.redis_client.get(f"email:{encrypted_email}"):
        raise HTTPException(status_code=400, detail="Email already exists")

    uid = str(uuid.uuid4())
    hashed_password = security_service.hash_password(user.password)

    user_data = UserInDB(
        id=uid,
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        created_at=datetime.now()
    )

    user_dict = user_data.dict()
    user_dict["created_at"] = user_dict["created_at"].isoformat()
    user_dict["is_active"] = str(user_dict["is_active"]).lower()

    encrypted_user_dict = {
        key: security_service.encrypt_data(str(value)) if key not in ["created_at", "is_active"] else value
        for key, value in user_dict.items()
    }

    await services.redis_client.hset(f"user:{user.username}", mapping=encrypted_user_dict)
    await services.redis_client.set(f"email:{encrypted_email}", security_service.encrypt_data(user.username))

    return Token(
        access_token=security_service.create_access_token({"sub": user.username}),
        refresh_token=security_service.create_refresh_token({"sub": user.username})
    )

@app.post("/auth/login", response_model=Token, tags=["Authentication"])
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user_data = await services.redis_client.hgetall(f"user:{form.username}")
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    for key, value in user_data.items():
        if key not in ["created_at", "is_active"]:
            user_data[key] = security_service.decrypt_data(value)

    user_data["created_at"] = datetime.fromisoformat(user_data["created_at"])
    user_data["is_active"] = user_data["is_active"].lower() == "true"
    user = UserInDB(**user_data)

    if not security_service.verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return Token(
        access_token=security_service.create_access_token({"sub": user.username}),
        refresh_token=security_service.create_refresh_token({"sub": user.username})
    )

@app.post("/analyze/mood", response_model=MoodAnalysisResponse, tags=["Mood Analysis"])
async def analyze_mood_endpoint(
    payload: MoodAnalysisRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    try:
        data = await mood_manager.analyze_emotion(payload.text)
        return MoodAnalysisResponse(**data)
    except ModelNotLoadedException as e:
        raise HTTPException(status_code=503, detail="Emotion model not loaded")
    
# ==========================================================
#                   UPDATE HEALTH CHECK
# ==========================================================

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with Redis validation"""
    issues = []
    
    # Check Redis
    try:
        if not services._initialized:
            issues.append("Services not initialized")
        else:
            redis_client = await services.get_redis()
            await redis_client.ping()
    except Exception as e:
        issues.append(f"Redis connection failed: {str(e)}")
    
    # Check models
    if not services.model or not services.whisper_model:
        issues.append("Models not loaded")
    
    if issues:
        raise HTTPException(status_code=503, detail={"issues": issues})
        
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "redis": "connected",
            "models": "loaded",
            "active_connections": len(connection_manager.active_connections),
            "initialized": services._initialized
        }
    }
@app.post("/api/whisper/reload", tags=["Audio"])
async def reload_whisper_model(
    model_size: str = "small",
    current_user: UserInDB = Depends(get_current_user)  # Require auth for model changes
):
    """Reload Whisper model with different size (admin only)"""
    valid_sizes = ["tiny", "base", "small", "medium", "large"]
    if model_size not in valid_sizes:
        raise HTTPException(status_code=400, detail=f"Invalid model size. Must be one of: {valid_sizes}")
    
    try:
        logger.info(f"Reloading Whisper model with size: {model_size}")
        old_model = services.whisper_model
        
        # Load new model
        services.whisper_model = whisper.load_model(model_size, device=services.device)
        
        # Clear old model from memory
        if old_model:
            del old_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"✅ Whisper model reloaded: {model_size}")
        return {
            "message": f"Whisper model reloaded successfully with size: {model_size}",
            "model_size": model_size,
            "device": str(services.device)
        }
        
    except Exception as e:
        logger.error(f"Failed to reload Whisper model: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}") 

# ==========================================================
#                   ENHANCED WEBSOCKET WITH STREAMING
# ==========================================================
# Update the WebSocket endpoint to check Redis before proceeding
@dataclass
class SessionConfig:
    """Configuration for WebSocket session"""
    session_timeout: int = 3600 * 24  # 24 hours
    max_message_length: int = 10000
    max_audio_size: int = 10 * 1024 * 1024  # 10MB
    rate_limit_per_minute: int = 60

class WebSocketMessage(BaseModel):
    """Validated WebSocket message structure"""
    type: str
    content: Optional[str] = None
    audio: Optional[str] = None
    voice_id: Optional[str] = None
    enable_streaming: bool = True
    token: Optional[str] = None
    personality: Optional[str] = None

class SessionState:
    """Manages session state and resources"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.user_id: Optional[str] = None
        self.conversation_id: Optional[str] = None
        self.is_authenticated = False
        self.current_personality = PersonalityType.EMPATHETIC
        self.message_count = 0
        self.last_activity = datetime.now()
        self.rate_limit_timestamps: List[datetime] = []
        
        # Initialize managers once per session
        self.personality_manager = PersonalityManager(services)
        self.mood_manager = EnhancedMoodManager(services)
        self.response_generator = PersonalityAwareResponseGenerator(services)
        self.preference_manager = UserPreferenceManager(redis_client=services.redis_client)
        
    def is_rate_limited(self) -> bool:
        """Check if session is rate limited"""
        now = datetime.now()
        # Remove timestamps older than 1 minute
        self.rate_limit_timestamps = [ts for ts in self.rate_limit_timestamps 
                                    if (now - ts).total_seconds() < 60]
        
        if len(self.rate_limit_timestamps) >= SessionConfig.rate_limit_per_minute:
            return True
            
        self.rate_limit_timestamps.append(now)
        return False
        
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
        self.message_count += 1

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

# ==========================================================
#                   REDIS OPERATIONS
# ==========================================================

class RedisOperations:
    """Centralized Redis operations with error handling"""
    
    @staticmethod
    async def safe_operation(operation_name: str, operation, *args, **kwargs):
        """Execute Redis operation with comprehensive error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not services.redis_client:
                    logger.warning("Redis client not available, attempting reconnection")
                    await services._init_redis()
                    if not services.redis_client:
                        raise Exception("Redis reconnection failed")
                
                result = await operation(*args, **kwargs)
                if attempt > 0:  # Log recovery
                    logger.info(f"Redis operation '{operation_name}' recovered on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                logger.error(f"Redis operation '{operation_name}' failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
    @staticmethod
    async def store_session_data(session_id: str, conversation_id: str, timeout: int = 3600 * 24):
        """Store session to conversation mapping"""
        return await RedisOperations.safe_operation(
            "store_session_data",
            services.redis_client.setex,
            f"session_conversation:{session_id}",
            timeout,
            conversation_id
        )
    
    @staticmethod
    async def get_conversation_id(session_id: str) -> Optional[str]:
        """Retrieve conversation ID for session"""
        result = await RedisOperations.safe_operation(
            "get_conversation_id",
            services.redis_client.get,
            f"session_conversation:{session_id}"
        )
        return result.decode() if result else None

# ==========================================================
#                   MESSAGE PROCESSORS
# ==========================================================

class MessageProcessor:
    """Handles different types of incoming messages"""
    
    def __init__(self, session_state: SessionState):
        self.session_state = session_state
    
    # ==========================================================
    #   UPDATED AUTH HANDLER WITH CONVERSATION HISTORY
    # ==========================================================
    async def process_auth_message(self, websocket: WebSocket, data: Dict[str, Any]) -> bool:
        """Process authentication message and return conversation history"""
        token = data.get("token")
        if not token:
            await self._send_auth_response(websocket, False, "No token provided")
            return False
            
        try:
            current_user = await security_service.get_current_user(token)
            self.session_state.user_id = current_user.id
            self.session_state.is_authenticated = True
            
            # Use WebSocketChatHandler to get conversation + history
            chat_handler = WebSocketChatHandler(services.redis_client)
            login_result = await chat_handler.handle_user_login(
                self.session_state.user_id, 
                self.session_state.session_id
            )
            
            if login_result.get("error"):
                await self._send_auth_response(websocket, False, f"Login error: {login_result['error']}")
                return False
            
            self.session_state.conversation_id = login_result.get("conversation_id")
            
            # Send auth response WITH history
            await websocket.send_json({
                "type": "auth_response",
                "success": True,
                "message": "Authentication successful, chat history loaded",
                "user_id": self.session_state.user_id,
                "conversation_id": self.session_state.conversation_id,
                "history": {
                    "messages": login_result.get("messages", []),
                    "message_count": login_result.get("message_count", 0)
                }
            })
            
            logger.info(
                f"Authenticated WebSocket with history: user_id={self.session_state.user_id}, "
                f"conversation_id={self.session_state.conversation_id}, "
                f"loaded {login_result.get('message_count', 0)} messages"
            )
            return True
            
        except AuthenticationException as e:
            await self._send_auth_response(websocket, False, f"Authentication failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Auth processing error: {e}")
            await self._send_auth_response(websocket, False, "Authentication error")
            return False

    async def _send_auth_response(self, websocket: WebSocket, success: bool, message: str, **kwargs):
        """Send authentication response with optional additional data"""
        response = {
            "type": "auth_response",
            "success": success,
            "message": message
        }
        response.update(kwargs)
        await websocket.send_json(response)
    
    async def process_audio_message(self, data: Dict[str, Any]) -> Optional[str]:
        """Process audio message and return transcribed text"""
        audio_b64 = data.get("audio")
        if not audio_b64:
            raise ValueError("No audio data provided")
        
        try:
            audio_bytes = base64.b64decode(audio_b64)
            
            # Validate audio size
            if len(audio_bytes) > SessionConfig.max_audio_size:
                raise ValueError(f"Audio size exceeds limit of {SessionConfig.max_audio_size} bytes")
            
            user_message = await audio_processor.transcribe_audio_async(
                audio_bytes, self.session_state.session_id
            )
            
            if not user_message or not user_message.strip():
                raise ValueError("Audio transcription resulted in empty text")
                
            logger.info(f"[{self.session_state.session_id}] STT -> {user_message}")
            return user_message.strip()
            
        except Exception as e:
            logger.error(f"[{self.session_state.session_id}] Audio processing error: {e}")
            raise ValueError(f"Audio processing failed: {str(e)}")
    
    async def process_text_message(self, data: Dict[str, Any]) -> str:
        """Process text message with validation"""
        user_message = (data.get("content") or "").strip()
        
        if not user_message:
            raise ValueError("No text content provided")
        
        if len(user_message) > SessionConfig.max_message_length:
            raise ValueError(f"Message exceeds maximum length of {SessionConfig.max_message_length} characters")
        
        return user_message
    
    async def process_personality_change(self, websocket: WebSocket, data: Dict[str, Any]):
        """Process personality change request"""
        try:
            new_personality = PersonalityType(data.get("personality"))
            await self.session_state.personality_manager.set_personality_for_session(
                self.session_state.session_id, new_personality
            )
            self.session_state.current_personality = new_personality
            
            await websocket.send_json({
                "type": "personality_updated",
                "personality": new_personality.value,
                "message": f"Personality changed to {new_personality.value}"
            })
        except ValueError as e:
            await websocket.send_json({
                "type": "error", 
                "error": f"Invalid personality type: {str(e)}"
            })
    
    async def process_preferences_request(self, websocket: WebSocket):
        """Process preferences request"""
        try:
            pref_summary = await self.session_state.preference_manager.get_preference_summary(
                self.session_state.session_id
            )
            await websocket.send_json({
                "type": "preference_summary", 
                "summary": pref_summary
            })
        except Exception as e:
            logger.error(f"Error getting preference summary: {e}")
            await websocket.send_json({
                "type": "error", 
                "error": "Failed to retrieve preferences"
            })

# ==========================================================
#                   CONVERSATION HANDLER
# ==========================================================

class ConversationHandler:
    """Handles conversation logic and response generation"""
    
    def __init__(self, session_state: SessionState):
        self.session_state = session_state
    
    async def setup_anonymous_session(self):
        """Setup anonymous session if not authenticated"""
        if not self.session_state.conversation_id:
            self.session_state.conversation_id = f"anonymous_{self.session_state.session_id}"
            await RedisOperations.store_session_data(
                self.session_state.session_id, 
                self.session_state.conversation_id
            )
            logger.info(f"Anonymous WebSocket connection: session_id={self.session_state.session_id}")
    
    async def load_conversation_history(self):
        """Load conversation history for authenticated users"""
        if not self.session_state.user_id:
            return
            
        try:
            messages = await enhanced_conversation_manager.get_conversation_messages(
                self.session_state.conversation_id, limit=5
            )
            
            for msg in messages:
                await enhanced_conversation_manager.save_message(
                    self.session_state.session_id, msg.role, msg.content
                )
                
            logger.info(f"Loaded {len(messages)} previous messages for conversation {self.session_state.conversation_id}")
        except Exception as e:
            logger.warning(f"Failed to load conversation history: {e}")
    
    async def analyze_user_preferences(self, user_message: str) -> Dict[str, Any]:
        """Analyze and update user preferences"""
        try:
            result = await self.session_state.preference_manager.update_preferences(
                self.session_state.session_id, user_message
            )
            return {
                'preferences_list': result.get('preferences_list', []),
                'preferences_detailed': result.get('preferences_detailed', {}),
                'newly_detected': result.get('newly_detected', [])
            }
        except Exception as e:
            logger.warning(f"[{self.session_state.session_id}] Preference extraction error: {e}")
            return {'preferences_list': [], 'preferences_detailed': {}, 'newly_detected': []}
    
    async def analyze_mood(self, user_message: str) -> Dict[str, Any]:
        """Analyze user mood and track progression"""
        try:
            mood_data = await self.session_state.mood_manager.analyze_emotion(user_message)
            current_mood, intensity = await self.session_state.mood_manager.track_mood(
                self.session_state.session_id, mood_data, user_message
            )
            
            mood_progression = await self.session_state.mood_manager.analyze_mood_progression(
                self.session_state.session_id
            )
            flow_suggestions = await self.session_state.mood_manager.get_conversation_flow_suggestions(
                self.session_state.session_id, current_mood
            )
            
            return {
                'current_mood': current_mood,
                'intensity': intensity,
                'mood_progression': mood_progression,
                'flow_suggestions': flow_suggestions
            }
        except Exception as e:
            logger.warning(f"[{self.session_state.session_id}] Mood analysis error: {e}")
            return {
                'current_mood': None,
                'intensity': 0.0,
                'mood_progression': {},
                'flow_suggestions': {"approach": "standard"}
            }
    
    async def build_context(self, user_message: str, preferences: Dict[str, Any]) -> str:
        """Build comprehensive context for response generation"""
        contexts = []
        
        # Add preference context
        if preferences['preferences_list']:
            pref_context = f"User preferences: {', '.join(preferences['preferences_list'][:5])}. "
            if preferences['preferences_detailed']:
                details = [f"{cat}: {max(prefs.items(), key=lambda x: x[1])[0]}"
                          for cat, prefs in preferences['preferences_detailed'].items() if prefs]
                if details:
                    pref_context += "Preference details - " + "; ".join(details[:3]) + ". "
            pref_context += "Consider these when responding."
            contexts.append(pref_context)
        
        # Add conversation history context
        try:
            context_text = await enhanced_conversation_manager.build_context_text(
                self.session_state.session_id, limit=10
            )
            if context_text:
                contexts.append(context_text)
        except Exception as e:
            logger.warning(f"Failed to build conversation context: {e}")
        
        # Add FAISS retrieval context
        try:
            faiss_results = await vector_db.search_similar_async(
                user_message, top_k=5, min_similarity_threshold=0.3, time_decay=True
            )
            if faiss_results:
                faiss_parts = []
                for result in faiss_results:
                    sim = result.get("similarity", 0)
                    past_mood = result.get("mood", "unknown")
                    faiss_parts.append(
                        f"Similar past (sim={sim:.2f}, mood={past_mood}): "
                        f"User: '{result['user_message']}' -> Bot: '{result['bot_reply'][:100]}...'"
                    )
                contexts.append("\n".join(faiss_parts))
        except Exception as e:
            logger.warning(f"[{self.session_state.session_id}] FAISS retrieval error: {e}")
        
        combined_context = "\n".join(contexts)
        return f"{combined_context}\nUser: {user_message}\nBot:" if combined_context else f"User: {user_message}\nBot:"
    
    async def generate_response(self, prompt: str, user_message: str, enable_streaming: bool, stream_id: str) -> str:
        """Generate bot response with personality awareness"""
        try:
            if enable_streaming and len(connection_manager.active_connections) <= config.MAX_CONCURRENT_STREAMS:
                return await self.session_state.response_generator.generate_personality_streaming_reply(
                    prompt, self.session_state.session_id, stream_id, user_message
                )
            else:
                return await self.session_state.response_generator.generate_personality_aware_reply(
                    prompt, self.session_state.session_id, user_message
                )
        except Exception as e:
            logger.error(f"[{self.session_state.session_id}] Response generation error: {e}")
            raise ValueError(f"Response generation failed: {str(e)}")
    
    async def save_interaction(self, user_message: str, bot_reply: str, metadata: Dict[str, Any]):
        """Save interaction to various storage systems"""
        try:
            # Save to conversation manager
            await enhanced_conversation_manager.save_message(
                self.session_state.session_id, "user", user_message
            )
            await enhanced_conversation_manager.save_message(
                self.session_state.session_id, "bot", bot_reply
            )
            
            # Save to persistent storage if authenticated
            if self.session_state.user_id:
                await enhanced_conversation_manager.add_message_to_conversation(
                    self.session_state.conversation_id, "user", user_message
                )
                await enhanced_conversation_manager.add_message_to_conversation(
                    self.session_state.conversation_id, "bot", bot_reply
                )
            
            # Save to vector database
            await vector_db.add_interaction_async(
                self.session_state.session_id, user_message, bot_reply, metadata
            )
            
        except Exception as e:
            logger.warning(f"[{self.session_state.session_id}] Save interaction error: {e}")
    
    async def generate_audio(self, text: str, voice_id: str) -> Optional[str]:
        """Generate audio response"""
        if not text or not text.strip():
            return None
            
        try:
            # Use personality-specific voice if not specified
            if not voice_id or voice_id == config.ELEVENLABS_VOICE_ID:
                voice_map = {
                    PersonalityType.EMPATHETIC: "21m00Tcm4TlvDq8ikWAM",
                    PersonalityType.PROFESSIONAL: "AZnzlk1XvdvUeBnXmlld",
                    PersonalityType.CHEERFUL: "EXAVITQu4vr4xnSDxMaL",
                    PersonalityType.ANALYTICAL: "AZnzlk1XvdvUeBnXmlld",
                    PersonalityType.CREATIVE: "EXAVITQu4vr4xnSDxMaL",
                }
                voice_id = voice_map.get(self.session_state.current_personality, config.ELEVENLABS_VOICE_ID)
            
            # Try streaming first, fallback to regular TTS
            audio_bytes = await audio_processor.text_to_speech_stream(
                text, voice_id=voice_id, session_id=self.session_state.session_id
            )
            
            if not audio_bytes:
                audio_bytes = await audio_processor.text_to_speech(text, voice_id=voice_id)
            
            if audio_bytes:
                return base64.b64encode(audio_bytes).decode()
            
        except Exception as e:
            logger.warning(f"[{self.session_state.session_id}] TTS error: {e}")
        
        return None
# ==========================================================
#        BACKWARD COMPATIBILITY ENDPOINTS (REDIS-AWARE)
# ==========================================================

class LegacyChatHandler:
    """Shim class to support old frontend calls while mapping to new logic"""

    def __init__(self, session_state: SessionState):
        self.session_state = session_state
        self.conversation_handler = ConversationHandler(session_state)

    async def handle_user_login(self, user_id: str, session_id: str) -> Optional[str]:
        """Old-style login handler (maps to process_auth + Redis)"""
        try:
            # Ensure Redis is available
            if not services.redis_client:
                raise Exception("Redis not available")

            # Get or create conversation
            conversation_id = await enhanced_conversation_manager.get_or_create_user_conversation(
                user_id, session_id
            )

            # Store in Redis
            await RedisOperations.store_session_data(session_id, conversation_id)

            self.session_state.user_id = user_id
            self.session_state.conversation_id = conversation_id
            self.session_state.is_authenticated = True

            return conversation_id
        except Exception as e:
            logger.error(f"Legacy handle_user_login failed: {e}")
            return None

    async def handle_message(self, session_id: str, user_message: str, user_id: Optional[str]):
        """Old-style message handler (process text + build context)"""
        try:
            # Save user message
            await enhanced_conversation_manager.save_message(session_id, "user", user_message)

            # Build context
            context = await enhanced_conversation_manager.build_context_text(session_id)

            return {
                "conversation_id": self.session_state.conversation_id or f"anonymous_{session_id}",
                "context": context,
                "user_id": user_id
            }
        except Exception as e:
            logger.error(f"Legacy handle_message failed: {e}")
            return {"conversation_id": None, "context": "", "user_id": user_id}

    async def generate_bot_response(self, context: str, user_message: str) -> str:
        """Old generate_bot_response (maps to PersonalityAwareResponseGenerator)"""
        try:
            return await self.session_state.response_generator.generate_personality_aware_reply(
                context, self.session_state.session_id, user_message
            )
        except Exception as e:
            logger.error(f"Legacy generate_bot_response failed: {e}")
            return "Sorry, I couldn’t generate a response."

    async def handle_bot_response(
        self, session_id: str, bot_response: str, user_id: Optional[str], metadata: Dict[str, Any]
    ):
        """Old-style bot response saver (maps to ConversationHandler.save_interaction)"""
        try:
            await self.conversation_handler.save_interaction(
                user_message="",  # user_message is not available here in old signature
                bot_reply=bot_response,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Legacy handle_bot_response failed: {e}")

# ==========================================================
#                   MAIN WEBSOCKET HANDLER
# ==========================================================

@asynccontextmanager
async def websocket_session(websocket: WebSocket):
    """Context manager for WebSocket session lifecycle"""
    session_id = str(uuid.uuid4())
    session_state = SessionState(session_id)
    
    try:
        await connection_manager.connect(websocket, session_id)
        logger.info(f"WebSocket session started: {session_id}")
        yield session_state
    finally:
        connection_manager.disconnect(session_id)
        logger.info(f"WebSocket session ended: {session_id}")

async def send_session_info(websocket: WebSocket, session_state: SessionState):
    """Send session initialization information to client"""
    await websocket.send_json({
        "type": "session",
        "session_id": session_state.session_id,
        "conversation_id": session_state.conversation_id,
        "user_id": session_state.user_id,
        "is_authenticated": session_state.is_authenticated
    })

async def send_error_response(websocket: WebSocket, error_message: str, session_id: str = None):
    """Send error response to client"""
    try:
        await websocket.send_json({
            "type": "error",
            "error": error_message,
            "session_id": session_id
        })
    except Exception as e:
        logger.error(f"Failed to send error response: {e}")

def validate_message(data: Dict[str, Any]) -> WebSocketMessage:
    """Validate incoming WebSocket message"""
    try:
        return WebSocketMessage(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid message format: {e}")

@app.websocket("/ws/chat") 
async def enhanced_websocket_chat(websocket: WebSocket):
    """Enhanced WebSocket chat handler with proper error handling and structure"""
    
    # Check Redis availability
    if not services.redis_client:
        await websocket.close(code=1011, reason="Redis not initialized")
        return
    
    async with websocket_session(websocket) as session_state:
        message_processor = MessageProcessor(session_state)
        conversation_handler = ConversationHandler(session_state)
        
        try:
            # ================== AUTHENTICATION ==================
            auth_successful = False
            try:
                initial_data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                validated_message = validate_message(initial_data)
                
                if validated_message.type == "auth" and validated_message.token:
                    auth_successful = await message_processor.process_auth_message(websocket, initial_data)
                
            except asyncio.TimeoutError:
                await send_error_response(websocket, "Authentication timeout", session_state.session_id)
                return
            except ValidationError as e:
                await send_error_response(websocket, f"Invalid message format: {e}", session_state.session_id)
                return
            except WebSocketDisconnect:
                logger.info(f"Client disconnected during auth: {session_state.session_id}")
                return
            except Exception as e:
                logger.warning(f"Auth processing error: {e}")
                initial_data = {"type": "text", "content": "Hello"}  # Default message for anonymous
            
            # Setup session
            await conversation_handler.setup_anonymous_session()
            await send_session_info(websocket, session_state)
            await conversation_handler.load_conversation_history()
            
            # ================== MAIN MESSAGE LOOP ==================
            pending_message = None
            if not auth_successful and 'initial_data' in locals():
                pending_message = initial_data
            
            while True:
                try:
                    # Get next message
                    if pending_message:
                        data = pending_message
                        pending_message = None
                    else:
                        data = await websocket.receive_json()
                    
                    # Rate limiting check
                    if session_state.is_rate_limited():
                        await send_error_response(websocket, "Rate limit exceeded", session_state.session_id)
                        continue
                    
                    # Validate message
                    try:
                        validated_message = validate_message(data)
                    except ValueError as e:
                        await send_error_response(websocket, str(e), session_state.session_id)
                        continue
                    
                    session_state.update_activity()
                    start_time = datetime.now()
                    stream_id = str(uuid.uuid4())
                    
                    # ================== HANDLE CONTROL MESSAGES ==================
                    if validated_message.type == "set_personality":
                        await message_processor.process_personality_change(websocket, data)
                        continue
                    
                    if validated_message.type == "get_preferences":
                        await message_processor.process_preferences_request(websocket)
                        continue
                    
                    # ================== PROCESS USER INPUT ==================
                    user_message = None
                    try:
                        if validated_message.type == "audio":
                            user_message = await message_processor.process_audio_message(data)
                        elif validated_message.type == "text":
                            user_message = await message_processor.process_text_message(data)
                        else:
                            await send_error_response(websocket, "Invalid message type", session_state.session_id)
                            continue
                    except ValueError as e:
                        await send_error_response(websocket, str(e), session_state.session_id)
                        continue
                    
                    # ================== ANALYZE AND GENERATE RESPONSE ==================
                    try:
                        # Analyze preferences and mood in parallel
                        preferences_task = conversation_handler.analyze_user_preferences(user_message)
                        mood_task = conversation_handler.analyze_mood(user_message)
                        
                        preferences, mood_data = await asyncio.gather(
                            preferences_task, mood_task, return_exceptions=True
                        )
                        
                        # Handle analysis errors
                        if isinstance(preferences, Exception):
                            logger.warning(f"Preference analysis failed: {preferences}")
                            preferences = {'preferences_list': [], 'preferences_detailed': {}, 'newly_detected': []}
                        
                        if isinstance(mood_data, Exception):
                            logger.warning(f"Mood analysis failed: {mood_data}")
                            mood_data = {'current_mood': None, 'intensity': 0.0, 'mood_progression': {}, 'flow_suggestions': {"approach": "standard"}}
                        
                        # Build context and generate response
                        prompt = await conversation_handler.build_context(user_message, preferences)
                        bot_reply = await conversation_handler.generate_response(
                            prompt, user_message, validated_message.enable_streaming, stream_id
                        )
                        
                        response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                        
                        # Generate audio response
                        bot_audio_b64 = await conversation_handler.generate_audio(
                            bot_reply, validated_message.voice_id or config.ELEVENLABS_VOICE_ID
                        )
                        
                        # Save interaction
                        metadata = {
                            "mood": mood_data['current_mood'].value if mood_data['current_mood'] else "unknown",
                            "mood_intensity": float(mood_data['intensity']),
                            "personality": session_state.current_personality.value,
                            "response_time_ms": response_time_ms,
                            "voice_id": validated_message.voice_id or config.ELEVENLABS_VOICE_ID,
                            "preferences": preferences['preferences_list'],
                            "preference_categories": list(preferences['preferences_detailed'].keys()),
                            "conversation_id": session_state.conversation_id,
                            "user_id": session_state.user_id
                        }
                        
                        await conversation_handler.save_interaction(user_message, bot_reply, metadata)
                        
                        # Send final response
                        final_response = {
                            "type": "reply",
                            "session_id": session_state.session_id,
                            "conversation_id": session_state.conversation_id,
                            "stream_id": stream_id,
                            "user_message": user_message,
                            "bot_reply": bot_reply,
                            "voice_id": validated_message.voice_id or config.ELEVENLABS_VOICE_ID,
                            "bot_audio": bot_audio_b64,
                            "response_time_ms": response_time_ms,
                            "current_personality": session_state.current_personality.value,
                            "current_mood": mood_data['current_mood'].value if mood_data['current_mood'] else None,
                            "mood_intensity": mood_data['intensity'],
                            "preferences": preferences['preferences_list'],
                            "preferences_detailed": preferences['preferences_detailed'],
                            "newly_detected_preferences": preferences['newly_detected'],
                            "preference_count": len(preferences['preferences_list']),
                            "mood_progression": mood_data['mood_progression'],
                            "flow_suggestions": mood_data['flow_suggestions'],
                            "user_id": session_state.user_id
                        }
                        
                        await websocket.send_json(final_response)
                        
                    except ValueError as e:
                        await send_error_response(websocket, str(e), session_state.session_id)
                    except Exception as e:
                        logger.error(f"Unexpected error processing message: {e}")
                        await send_error_response(websocket, "Internal processing error", session_state.session_id)
                
                except json.JSONDecodeError:
                    await send_error_response(websocket, "Invalid JSON format", session_state.session_id)
                except WebSocketDisconnect:
                    logger.info(f"Client disconnected: {session_state.session_id}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in message loop: {e}")
                    await send_error_response(websocket, "Internal server error", session_state.session_id)
                    break
        
        except Exception as e:
            logger.error(f"Fatal error in WebSocket handler: {e}")
            try:
                await send_error_response(websocket, "Fatal server error", session_state.session_id)
            except:
                pass

     
# ==========================================================
#                   CONVERSATION HISTORY ENDPOINTS
# ==========================================================

@app.get("/api/conversations", response_model=ConversationListResponse, tags=["Conversations"])
async def get_user_conversations(
    limit: int = 20,
    current_user: UserInDB = Depends(get_current_user)
):
    """Get user's conversation history"""
    try:
        conversations = await enhanced_conversation_manager.get_user_conversations(
            current_user.id, limit=limit
        )
        
        conversation_dicts = []
        for conv in conversations:
            conversation_dicts.append({
                "conversation_id": conv.conversation_id,
                "session_id": conv.session_id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "last_message_at": conv.last_message_at.isoformat(),
                "message_count": conv.message_count,
                "preview": conv.preview
            })
        
        return ConversationListResponse(
            conversations=conversation_dicts,
            total_count=len(conversation_dicts),
            user_id=current_user.id
        )
    except Exception as e:
        logger.error(f"Error getting conversations for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversations")

@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetailResponse, tags=["Conversations"])
async def get_conversation_detail(
    conversation_id: str,
    limit: int = 100,
    current_user: UserInDB = Depends(get_current_user)
):
    """Get detailed conversation with all messages"""
    try:
        # Verify conversation belongs to user
        conv_data = await services.redis_client.hgetall(f"conversation:{conversation_id}")
        if not conv_data or conv_data.get("user_id") != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get messages
        messages = await enhanced_conversation_manager.get_conversation_messages(
            conversation_id, limit=limit
        )
        
        message_dicts = []
        for msg in messages:
            message_dicts.append({
                "message_id": msg.message_id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            })
        
        return ConversationDetailResponse(
            conversation_id=conversation_id,
            title=conv_data.get("title", "Untitled Chat"),
            messages=message_dicts,
            created_at=conv_data.get("created_at", ""),
            message_count=int(conv_data.get("message_count", 0))
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation")

@app.post("/api/conversations", tags=["Conversations"])
async def create_conversation(
    request: CreateConversationRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """Create a new conversation"""
    try:
        conversation_id = await enhanced_conversation_manager.create_conversation(
            user_id=current_user.id,
            session_id=request.session_id,
            title=request.title
        )
        
        return {
            "conversation_id": conversation_id,
            "message": "Conversation created successfully",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to create conversation")

@app.put("/api/conversations/{conversation_id}", tags=["Conversations"])
async def update_conversation(
    conversation_id: str,
    request: UpdateConversationRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """Update conversation title"""
    try:
        # Verify ownership
        conv_data = await services.redis_client.hgetall(f"conversation:{conversation_id}")
        if not conv_data or conv_data.get("user_id") != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Update title
        await services.redis_client.hset(
            f"conversation:{conversation_id}", 
            "title", request.title
        )
        
        return {
            "message": "Conversation updated successfully",
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update conversation")

@app.delete("/api/conversations/{conversation_id}", tags=["Conversations"])
async def delete_conversation(
    conversation_id: str,
    current_user: UserInDB = Depends(get_current_user)
):
    """Delete a conversation"""
    try:
        success = await enhanced_conversation_manager.delete_conversation(
            current_user.id, conversation_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "message": "Conversation deleted successfully",
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")

@app.get("/api/conversations/search", tags=["Conversations"])
async def search_conversations(
    q: str,
    limit: int = 10,
    current_user: UserInDB = Depends(get_current_user)
):
    """Search user's conversations"""
    try:
        if len(q.strip()) < 2:
            raise HTTPException(status_code=400, detail="Search query too short")
        
        conversations = await enhanced_conversation_manager.search_conversations(
            current_user.id, q, limit=limit
        )
        
        results = []
        for conv in conversations:
            results.append({
                "conversation_id": conv.conversation_id,
                "title": conv.title,
                "preview": conv.preview,
                "created_at": conv.created_at.isoformat(),
                "message_count": conv.message_count
            })
        
        return {
            "query": q,
            "results": results,
            "count": len(results)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching conversations: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.get("/api/conversations/stats", response_model=ConversationStatsResponse, tags=["Conversations"])
async def get_conversation_stats(
    current_user: UserInDB = Depends(get_current_user)
):
    """Get user's conversation statistics"""
    try:
        stats = await enhanced_conversation_manager.get_conversation_stats(current_user.id)
        return ConversationStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting conversation stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@app.post("/api/conversations/{conversation_id}/continue", tags=["Conversations"])
async def continue_conversation(
    conversation_id: str,
    current_user: UserInDB = Depends(get_current_user)
):
    """Continue an existing conversation in a new session"""
    try:
        # Verify ownership
        conv_data = await services.redis_client.hgetall(f"conversation:{conversation_id}")
        if not conv_data or conv_data.get("user_id") != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Generate new session ID for continuation
        import uuid
        new_session_id = str(uuid.uuid4())
        
        # Link new session to existing conversation
        await services.redis_client.setex(
            f"session_conversation:{new_session_id}", 
            3600 * 24, 
            conversation_id
        )
        
        return {
            "session_id": new_session_id,
            "conversation_id": conversation_id,
            "message": "Ready to continue conversation",
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error continuing conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to continue conversation")
    
@app.post("/api/users/memory/migrate", tags=["Memory Management"])
async def migrate_user_memory(
    session_id: Optional[str] = None,
    current_user: UserInDB = Depends(get_current_user)
):
    """Migrate session data to permanent user memory"""
    migration_manager = MemoryMigrationManager(services, vector_db)
    
    if session_id:
        result = await migration_manager.migrate_session_data_to_user_memory(current_user.id, session_id)
        return result
    else:
        return {"error": "Session ID required for migration"}

@app.post("/api/users/memory/rebuild-patterns", tags=["Memory Management"])
async def rebuild_behavioral_patterns(current_user: UserInDB = Depends(get_current_user)):
    """Rebuild behavioral patterns from complete user history"""
    migration_manager = MemoryMigrationManager(services, vector_db)
    result = await migration_manager.rebuild_user_behavioral_patterns(current_user.id)
    return result

@app.get("/api/users/memory/export", tags=["Memory Management"])
async def export_user_memory(current_user: UserInDB = Depends(get_current_user)):
    """Export user's complete memory data"""
    try:
        # Get all user history
        history = await services.redis_client.lrange(f"user_history:{current_user.id}", 0, -1)
        
        # Get behavioral patterns
        patterns = await services.redis_client.hgetall(f"user_patterns:{current_user.id}")
        
        # Get conversation data
        conversations = await enhanced_conversation_manager.get_user_conversations(current_user.id, limit=1000)
        
        export_data = {
            "user_id": current_user.id,
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_interactions": len(history),
            "interaction_history": [json.loads(h) for h in history],
            "behavioral_patterns": {k: json.loads(v) if v.startswith('{') or v.startswith('[') else v for k, v in patterns.items()},
            "conversations": [
                {
                    "conversation_id": conv.conversation_id,
                    "title": conv.title,
                    "created_at": conv.created_at.isoformat(),
                    "message_count": conv.message_count
                } for conv in conversations
            ]
        }
        
        return export_data
        
    except Exception as e:
        logger.error(f"Error exporting user memory: {e}")
        raise HTTPException(status_code=500, detail="Export failed")

@app.get("/api/users/memory/topics", tags=["Memory Management"])
async def get_user_topics(
    limit: int = 20,
    current_user: UserInDB = Depends(get_current_user)
):
    """Get topics that user frequently discusses"""
    try:
        # Use enhanced vector DB if available
        if hasattr(vector_db, 'get_user_topics_async'):
            topics = await vector_db.get_user_topics_async(current_user.id, limit)
        else:
            # Fallback to analyzing user history directly
            history = await services.redis_client.lrange(f"user_history:{current_user.id}", -100, -1)
            topics = {}
            
            for interaction_data in history:
                try:
                    interaction = json.loads(interaction_data)
                    user_msg = interaction.get("user_message", "").lower()
                    words = [w.strip(".,!?") for w in user_msg.split() if len(w) > 4]
                    for word in words:
                        topics[word] = topics.get(word, 0) + 1
                except:
                    continue
            
            topics = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:limit])
        
        return {
            "user_id": current_user.id,
            "topics": topics,
            "total_topics": len(topics)
        }
        
    except Exception as e:
        logger.error(f"Error getting user topics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get topics")

@app.post("/api/users/memory/backup", tags=["Memory Management"])
async def backup_user_memory(current_user: UserInDB = Depends(get_current_user)):
    """Create a backup of user's complete memory system"""
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_key = f"backup:{current_user.id}:{timestamp}"
        
        # Backup user history
        history = await services.redis_client.lrange(f"user_history:{current_user.id}", 0, -1)
        if history:
            await services.redis_client.rpush(f"{backup_key}:history", *history)
        
        # Backup patterns
        patterns = await services.redis_client.hgetall(f"user_patterns:{current_user.id}")
        if patterns:
            await services.redis_client.hset(f"{backup_key}:patterns", mapping=patterns)
        
        # Set expiration (30 days)
        await services.redis_client.expire(f"{backup_key}:history", 30 * 24 * 3600)
        await services.redis_client.expire(f"{backup_key}:patterns", 30 * 24 * 3600)
        
        return {
            "backup_id": f"{current_user.id}:{timestamp}",
            "message": "Memory backup created successfully",
            "expires_in_days": 30
        }
        
    except Exception as e:
        logger.error(f"Error creating memory backup: {e}")
        raise HTTPException(status_code=500, detail="Backup failed")
    
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
#                   NEW STREAMING-SPECIFIC ENDPOINTS
# ==========================================================

@app.get("/api/sessions/active", tags=["Sessions"])
async def get_active_sessions():
    """Get information about active sessions"""
    return {
        "active_sessions": len(connection_manager.active_connections),
        "streaming_sessions": sum(1 for s in connection_manager.streaming_sessions.values() if s),
        "max_concurrent_streams": config.MAX_CONCURRENT_STREAMS
    }

@app.post("/api/sessions/{session_id}/interrupt", tags=["Sessions"])
async def interrupt_streaming(session_id: str):
    """Interrupt an ongoing streaming session"""
    if session_id in connection_manager.streaming_sessions:
        connection_manager.set_streaming(session_id, False)
        await connection_manager.send_message(session_id, {
            "type": "stream_interrupted",
            "message": "Streaming interrupted by user"
        })
        return {"status": "interrupted"}
    return {"status": "session_not_found"}

# ==========================================================
#                   ERROR HANDLERS
# ==========================================================
@app.exception_handler(AuthenticationException)
async def auth_exception_handler(request, exc):
    return JSONResponse(status_code=401, content={"detail": str(exc)})

@app.exception_handler(ModelNotLoadedException)
async def model_exception_handler(request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})

@app.exception_handler(StreamingException)
async def streaming_exception_handler(request, exc):
    return JSONResponse(status_code=503, content={"detail": f"Streaming error: {str(exc)}"})

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        await services.redis_client.ping()
        
        if not services.model or not services.whisper_model:
            raise HTTPException(status_code=503, detail="Models not loaded")
            
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "redis": "connected",
                "models": "loaded",
                "active_connections": len(connection_manager.active_connections),
                "streaming_sessions": sum(1 for s in connection_manager.streaming_sessions.values() if s)
            },
            "config": {
                "streaming_enabled": True,
                "max_concurrent_streams": config.MAX_CONCURRENT_STREAMS,
                "stream_chunk_size": config.STREAM_CHUNK_SIZE,
                "stream_delay_ms": config.STREAM_DELAY_MS
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

# Serve index.html at "/"
@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)