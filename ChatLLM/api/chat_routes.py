from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..services import chat_service as svc
from ..models.schemas import (
    Msg, ChatMeta, SendReq, SendResp, RegenReq,
)

router = APIRouter(prefix="/api", tags=["Chat"])


class EditReq(BaseModel):
    content: str
    model: str | None = None
    temperature: float | None = None


class ChatUpdateReq(BaseModel):
    title: str
    model: str
    temperature: float
    stream: bool
    prompt: str


@router.get("/chats", response_model=list[ChatMeta])
def list_chats():
    return svc.list_chats()


@router.post("/chats", response_model=ChatMeta)
def create_chat():
    return svc.create_chat()


@router.delete("/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: str):
    svc.delete_chat(chat_id)


@router.get("/chats/{chat_id}", response_model=list[Msg])
def get_messages(chat_id: str):
    if chat_id not in svc._chat_store:
        raise HTTPException(404, "chat not found")
    return svc.get_messages(chat_id)


@router.post("/chats/{chat_id}/messages", response_model=SendResp)
def send(chat_id: str, req: SendReq):
    if chat_id not in svc._chat_store:
        raise HTTPException(404, "chat not found")
    chat = svc._chat_store[chat_id]
    if chat.get("stream"):
        generator = svc.stream_message(chat_id, req.content, req.model, req.temperature)
        return StreamingResponse(generator, media_type="text/plain")
    content = svc.send_message(chat_id, req.content, req.model, req.temperature)
    return {"content": content}


@router.patch("/chats/{chat_id}/messages/{idx}", response_model=SendResp)
def edit(chat_id: str, idx: int, req: EditReq):
    if chat_id not in svc._chat_store:
        raise HTTPException(404, "chat not found")
    content = svc.edit_message(chat_id, idx, req.content, req.model, req.temperature)
    return {"content": content}


@router.post("/chats/{chat_id}/messages/{idx}/regenerate", response_model=SendResp)
def regenerate(chat_id: str, idx: int, req: RegenReq | None = None):
    if chat_id not in svc._chat_store:
        raise HTTPException(404, "chat not found")
    model = req.model if req else None
    temperature = req.temperature if req else None
    chat = svc._chat_store[chat_id]
    if chat.get("stream"):
        gen = svc.stream_regenerate_assistant(chat_id, idx, model, temperature)
        return StreamingResponse(gen, media_type="text/plain")
    content = svc.regenerate_assistant(chat_id, idx, model, temperature)
    return {"content": content}

@router.patch("/chats/{chat_id}", response_model=ChatMeta)
def update_chat(chat_id: str, req: ChatUpdateReq):
    return svc.update_chat(chat_id, req.title, req.model, req.temperature, req.stream, req.prompt)


@router.get("/models", response_model=list[str])
def list_models():
    return svc.list_models()
