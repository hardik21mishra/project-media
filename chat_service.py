import json
import os
from groq import Groq
from file_service import conversations

INTENT_NAMES = {"upload", "transcription", "conversion", "status", "general"}
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "20"))

def chat_reply(
    conversation_id: str,
    reply: str,
    job_id: str | None = None,
    status: str | None = None,
    result: dict | None = None,
    attachment: dict | None = None,
    action: str | None = None,
):
    response: dict[str, object] = {
        "conversation_id": conversation_id,
        "reply": reply,
        "reply_format": "markdown",
    }
    if job_id:
        response["job_id"] = job_id
    if status:
        response["status"] = status
    if result:
        response["result"] = result
    if attachment:
        response["attachment"] = attachment
    if action:
        response["action"] = action
    return response

def classify_intent(message: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    client = Groq(api_key=api_key, timeout=GROQ_TIMEOUT_SECONDS)
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        messages=[
            {
                "role": "system",
                "content": (
                   "Classify the user's media request. Return only a JSON object with one field, \"intent\". The value must be exactly one of: upload, transcription, conversion, status, general. Choose transcription to turn audio into written words (infer from meaning, not specific trigger words). Choose conversion for changing media formats. Choose upload when they provide a file but state no processing task. Choose status for progress/result questions. Return general for greetings, small talk, or when no media intent is expressed."
                ),
            },
            {"role": "user", "content": message},
        ],
        temperature=0,
        max_tokens=40,
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content or "{}"
    intent = json.loads(content).get("intent")
    if intent not in INTENT_NAMES:
        raise ValueError("Groq returned an unsupported intent.")
    return intent

def ask_groq(conversation_id: str, message: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    conversation = conversations.setdefault(conversation_id, {"messages": []})
    messages = conversation.setdefault("messages", [])
    messages.append({"role": "user", "content": message})
    client = Groq(api_key=api_key, timeout=GROQ_TIMEOUT_SECONDS)
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        messages=[
            {
                "role": "system",
                "content": ("You are Voxera, a concise and friendly media assistant. You can transcribe audio, convert videos to audio, and report job status. Never claim a file was processed unless the backend says the job is done."),
            },
            *messages[-10:],
        ],
        temperature=0.3,
    )
    reply = completion.choices[0].message.content or "How can I help with your media?"
    messages.append({"role": "assistant", "content": reply})
    return reply