import os
import logging
from datetime import datetime
from typing import List
from contextlib import asynccontextmanager
import re

from fastapi import FastAPI, HTTPException, status, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, select, func
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

# ----------------------
# Environment Variables
# ----------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")
print(f"DATABASE_URL: {DATABASE_URL}")
print(f"TOGETHER_AI_API_KEY: {TOGETHER_AI_API_KEY}")

if not DATABASE_URL or not TOGETHER_AI_API_KEY:
    raise RuntimeError("DATABASE_URL and TOGETHER_AI_API_KEY must be set in .env file.")

# Convert postgresql to postgresql+asyncpg
DATABASE_URL = re.sub(r'^postgresql:', 'postgresql+asyncpg:', DATABASE_URL)

# ----------------------
# Logging Setup
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# Database Setup
# ----------------------
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ----------------------
# SQLAlchemy Models
# ----------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    phone_number = Column(String, nullable=False, unique=True, index=True)
    preferred_call_time = Column(String, nullable=False)
    language_preference = Column(String, nullable=False)
    aboutme = Column(Text, nullable=False)
    chat_summaries = relationship("ChatSummary", back_populates="user")

class ChatSummary(Base):
    __tablename__ = "chat_summaries"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    conversation_summary = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    user = relationship("User", back_populates="chat_summaries")

# ----------------------
# Pydantic Schemas
# ----------------------
class UserCreate(BaseModel):
    name: str = Field(...)
    phone_number: str = Field(...)
    preferred_call_time:str = Field(..., alias="preferred_call_time")
    language_preference: str = Field(...)
    aboutme: str = Field(...)

class UserResponse(BaseModel):
    status: str
    user_id: int
    name: str
    phone_number: str
    preferred_call_time: str
    language_preference: str
    aboutme: str
    

class ChatSummaryCreate(BaseModel):
    user_id: int
    chat_data: str

class ChatSummaryResponse(BaseModel):
    status: str
    summary_id: int
    summary: str

class ChatSummaryListItem(BaseModel):
    summary_id: int
    conversation_summary: str
    created_at: datetime

class ChatSummaryListResponse(BaseModel):
    status: str
    summaries: List[ChatSummaryListItem]

# ----------------------
# Lifespan Context Manager
# ----------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created.")
    yield
    # Shutdown
    await engine.dispose()
# ----------------------
# FastAPI App
# ----------------------
app = FastAPI(title="ALKANE Voice Agent API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000", "http://127.0.0.1:8000","http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------
# Together AI API Call
# ----------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_summary(chat_data: str) -> str:
    url = "https://api.together.xyz/v1/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = (
        "Summarize this conversation into 100-200 words in a casual, diary-like style: "
        f"{chat_data}"
    )
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 300
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload, timeout=60) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Together AI API error: {resp.status} - {text}")
                raise HTTPException(status_code=500, detail="Together AI API error.")
            data = await resp.json()
            summary = data.get("choices", [{}])[0].get("text")
            if not summary:
                logger.error(f"Together AI API returned no summary: {data}")
                raise HTTPException(status_code=500, detail="Together AI API returned no summary.")
            return summary.strip()

# ----------------------
# Endpoints
# ----------------------

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    async with async_session() as session:
        async with session.begin():
            stmt = select(User).where(User.phone_number == user.phone_number)
            result = await session.execute(stmt)
            existing_user = result.scalar_one_or_none()
            if existing_user:
                logger.info(f"Duplicate phone number: {user.phone_number}")
                return UserResponse(
                    status="success",
                    user_id=existing_user.id,
                    name=existing_user.name,
                    phone_number=existing_user.phone_number,
                    preferred_call_time=existing_user.preferred_call_time,
                    language_preference=existing_user.language_preference,
                    aboutme=existing_user.aboutme
                )
            new_user = User(**user.model_dump())
            session.add(new_user)
            try:
                await session.flush()
                await session.refresh(new_user)
            except IntegrityError as e:
                logger.error(f"IntegrityError: {e}")
                raise HTTPException(status_code=400, detail="User with this phone number already exists.")
            return UserResponse(
                status="success",
                user_id=new_user.id,
                name=new_user.name,
                phone_number=new_user.phone_number,
                preferred_call_time=new_user.preferred_call_time,
                language_preference=new_user.language_preference,
                aboutme=new_user.aboutme
                
            )

@app.post("/chat-summaries", response_model=ChatSummaryResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_summary(summary: ChatSummaryCreate):
    async with async_session() as session:
        async with session.begin():
            # Validate user_id
            stmt = select(User).where(User.id == summary.user_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()
            if not user:
                logger.warning(f"User ID not found: {summary.user_id}")
                raise HTTPException(status_code=404, detail="User not found.")
            # Generate summary
            try:
                generated_summary = await generate_summary(summary.chat_data)
            except Exception as e:
                logger.error(f"Together AI API failed: {e}")
                raise HTTPException(status_code=500, detail="Failed to generate summary.")
            # Save summary
            chat_summary = ChatSummary(
                user_id=summary.user_id,
                conversation_summary=generated_summary
            )
            session.add(chat_summary)
            await session.flush()
            await session.refresh(chat_summary)
            return ChatSummaryResponse(
                status="success",
                summary_id=chat_summary.id,
                summary=chat_summary.conversation_summary
            )

@app.get("/chat-summaries/{user_id}", response_model=ChatSummaryListResponse)
async def get_chat_summaries(user_id: int = Path(..., gt=0)):
    async with async_session() as session:
        async with session.begin():
            # Validate user_id
            stmt = select(User).where(User.id == user_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()
            if not user:
                logger.warning(f"User ID not found: {user_id}")
                raise HTTPException(status_code=404, detail="User not found.")
            # Get summaries
            stmt = select(ChatSummary).where(ChatSummary.user_id == user_id).order_by(ChatSummary.created_at.desc())
            result = await session.execute(stmt)
            summaries = result.scalars().all()
            summary_list = [
                ChatSummaryListItem(
                    summary_id=s.id,
                    conversation_summary=s.conversation_summary,
                    created_at=s.created_at
                ) for s in summaries
            ]
            return ChatSummaryListResponse(status="success", summaries=summary_list)
@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int = Path(..., gt=0)):
                async with async_session() as session:
                    async with session.begin():
                        stmt = select(User).where(User.id == user_id)
                        result = await session.execute(stmt)
                        user = result.scalar_one_or_none()
                        if not user:
                            logger.warning(f"User ID not found: {user_id}")
                            raise HTTPException(status_code=404, detail="User not found.")
                        return UserResponse(
                            status="success",
                            user_id=user.id,
                            name=user.name,
                            phone_number=user.phone_number,
                            preferred_call_time=user.preferred_call_time,
                            language_preference=user.language_preference,
                            aboutme=user.aboutme
                        )
@app.get("/users", response_model=List[UserResponse])
async def list_users():
    async with async_session() as session:
        async with session.begin():
            stmt = select(User).order_by(User.id)
            result = await session.execute(stmt)
            users = result.scalars().all()
            user_list = [
                UserResponse(
                    status="success",
                    user_id=u.id,
                    name=u.name,
                    phone_number=u.phone_number,
                    preferred_call_time=u.preferred_call_time,
                    language_preference=u.language_preference,
                    aboutme=u.aboutme
                    
                ) for u in users
            ]
            return user_list
