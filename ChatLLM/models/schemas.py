from pydantic import BaseModel


class Msg(BaseModel):
    role: str
    content: str


class ChatMeta(BaseModel):
    id: str
    title: str
    model: str
    temperature: float
    stream: bool


class SendReq(BaseModel):
    content: str
    model: str | None = None          # если None – берём настройки чата
    temperature: float | None = None  # если None – берём настройки чата


class SendResp(BaseModel):
    content: str


class EditReq(BaseModel):
    content: str


class RegenReq(BaseModel):
    model: str | None = None
    temperature: float | None = None
