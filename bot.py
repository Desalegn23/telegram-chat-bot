import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import ssl
from urllib.parse import parse_qs, unquote, urlparse

import pg8000.native
from chatgpt_md_converter import split_html_for_telegram, telegram_format
from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

TELEGRAM_TOKEN = os.environ["telegram_token"].strip()
OPENROUTER_KEY = os.environ["open_router_key"].strip()
GEMINI_KEY = os.environ.get("gemini_key", "").strip()
DATABASE_URL = os.environ.get("database_url", "").strip()
SYSTEM_PROMPT = """You are a helpful, friendly assistant chatting on Telegram.

Format replies so they look great in a Telegram chat, using Markdown:
- Use **bold** for key terms and short section titles (## headings also work).
- Use bullet or numbered lists for steps, options and comparisons.
- Use `inline code` and fenced ```language code blocks``` for any code, commands or file names.
- Use > quotes to highlight a tip, warning or key takeaway.
- Use ||spoiler|| only to hide answers to quizzes or riddles.
- Use a few relevant emojis to make sections easy to scan, but don't overdo it.
- Keep tables small (they are shown as monospace text); prefer lists when possible.
Keep paragraphs short. Match length to the question: brief for simple questions, structured for complex ones."""
MAX_HISTORY = 20  # messages kept per chat (user + assistant)
TELEGRAM_MSG_LIMIT = 4096
REQUEST_TIMEOUT = 90  # seconds per model request

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")

# One OpenAI-compatible client per provider.
clients = {
    "openrouter": AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY, timeout=REQUEST_TIMEOUT
    ),
}
if GEMINI_KEY:
    clients["gemini"] = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=GEMINI_KEY,
        timeout=REQUEST_TIMEOUT,
    )

# key -> (button label, provider, model id)
MODELS = {
    "ling": ("Ling 3.0 Flash VL", "openrouter", "inclusionai/ling-3.0-flash-vl"),
    "gemini-flash": ("Gemini 3.8 Flash", "gemini", "gemini-3.8-flash"),
    "gemini-lite": ("Gemini 3.5 Flash-Lite", "gemini", "gemini-3.5-flash-lite"),
}
MODELS = {k: v for k, v in MODELS.items() if v[1] in clients}  # hide models without a key
DEFAULT_MODEL = "ling"

class ChatStore:
    """Per-chat model choice and history: saved in Postgres when database_url is set, else kept in memory."""

    def __init__(self, url: str):
        self.url = urlparse(url) if url else None
        self.memory: dict[int, dict] = {}

    def _run(self, sql: str, **params):
        # A short-lived connection per query: simple, and survives the database suspending when idle.
        u = self.url
        sslmode = parse_qs(u.query).get("sslmode", ["require"])[0]
        con = pg8000.native.Connection(
            user=unquote(u.username or ""),
            password=unquote(u.password or ""),
            host=u.hostname,
            port=u.port or 5432,
            database=u.path.lstrip("/") or "postgres",
            ssl_context=None if sslmode == "disable" else ssl.create_default_context(),
            timeout=30,
        )
        try:
            return con.run(sql, **params)
        finally:
            con.close()

    async def init(self) -> None:
        if not self.url:
            log.warning("database_url not set: chat history is kept in memory and lost on restart")
            return
        await asyncio.to_thread(
            self._run,
            "CREATE TABLE IF NOT EXISTS chats ("
            " chat_id BIGINT PRIMARY KEY,"
            " model TEXT,"
            " history JSONB NOT NULL DEFAULT '[]',"
            " updated_at TIMESTAMPTZ NOT NULL DEFAULT now())",
        )
        log.info("Chat history is saved in Postgres (%s)", self.url.hostname)

    async def load(self, chat_id: int) -> dict:
        if not self.url:
            return self.memory.get(chat_id) or {"model": None, "history": []}
        rows = await asyncio.to_thread(self._run, "SELECT model, history FROM chats WHERE chat_id = :id", id=chat_id)
        return {"model": rows[0][0], "history": rows[0][1]} if rows else {"model": None, "history": []}

    async def save(self, chat_id: int, state: dict) -> None:
        state = {"model": state["model"], "history": state["history"][-MAX_HISTORY:]}
        if not self.url:
            self.memory[chat_id] = state
            return
        await asyncio.to_thread(
            self._run,
            "INSERT INTO chats (chat_id, model, history) VALUES (:id, :model, CAST(:history AS JSONB))"
            " ON CONFLICT (chat_id) DO UPDATE"
            " SET model = EXCLUDED.model, history = EXCLUDED.history, updated_at = now()",
            id=chat_id,
            model=state["model"],
            history=json.dumps(state["history"]),
        )


store = ChatStore(DATABASE_URL)


def model_key(state: dict) -> str:
    return state["model"] if state["model"] in MODELS else DEFAULT_MODEL


async def ask_model(chat_id: int, user_content) -> str:
    state = await store.load(chat_id)
    _, provider, model_id = MODELS[model_key(state)]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *state["history"], {"role": "user", "content": user_content}]
    resp = await clients[provider].chat.completions.create(model=model_id, messages=messages)
    reply = (resp.choices[0].message.content or "").strip() or "(empty response)"
    # Store images as a text placeholder so history stays small.
    if isinstance(user_content, list):
        text = next((p["text"] for p in user_content if p["type"] == "text"), "")
        user_content = f"[sent an image] {text}".strip()
    state["history"] = [
        *state["history"],
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": reply},
    ]
    await store.save(chat_id, state)
    return reply


TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEPARATOR = re.compile(r"^[\s|:\-]+$")
HORIZONTAL_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
INLINE_MARKUP = re.compile(r"(\*\*|__|~~|`)")
DIVIDER = "━━━━━━━━━━━━"


def render_table(lines: list[str]) -> list[str]:
    """Turn Markdown table lines into an aligned monospace table inside a code fence."""
    rows = [
        [INLINE_MARKUP.sub("", cell).strip() for cell in line.strip().strip("|").split("|")]
        for line in lines
        if not TABLE_SEPARATOR.match(line)
    ]
    cols = max(len(r) for r in rows)
    rows = [r + [""] * (cols - len(r)) for r in rows]
    widths = [max(len(r[i]) for r in rows) for i in range(cols)]
    fmt = lambda r: " │ ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
    out = [fmt(rows[0])]
    if len(rows) > 1:
        out.append("─┼─".join("─" * w for w in widths))
        out += [fmt(r) for r in rows[1:]]
    return ["```", *out, "```"]


def prepare_markdown(markdown: str) -> str:
    """Adapt Markdown for Telegram: tables become aligned monospace blocks, --- rules become a divider line."""
    out, table, in_code = [], [], False
    for line in markdown.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not in_code and TABLE_LINE.match(line):
            table.append(line)
            continue
        if table:
            out += render_table(table)
            table = []
        out.append(DIVIDER if not in_code and HORIZONTAL_RULE.match(line) else line)
    if table:
        out += render_table(table)
    return "\n".join(out)


async def send_long(update: Update, text: str) -> None:
    """Send a Markdown reply as rich Telegram HTML, falling back to plain text if Telegram rejects it."""
    try:
        chunks = split_html_for_telegram(telegram_format(prepare_markdown(text)), max_length=TELEGRAM_MSG_LIMIT)
    except Exception:
        log.exception("Markdown conversion failed")
        chunks = None
    if chunks:
        try:
            for chunk in chunks:
                await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
            return
        except BadRequest:
            log.exception("Telegram rejected formatted message; resending as plain text")
    for i in range(0, len(text), TELEGRAM_MSG_LIMIT):
        await update.message.reply_text(text[i : i + TELEGRAM_MSG_LIMIT])


def model_keyboard(selected: str) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(("✅ " if key == selected else "") + label, callback_data=f"model:{key}")]
        for key, (label, _, _) in MODELS.items()
    ]
    return InlineKeyboardMarkup(buttons)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = await store.load(update.effective_chat.id)
    label = MODELS[model_key(state)][0]
    await update.message.reply_text(
        f"Hi! I'm currently using {label}. Send me a message or a photo.\n"
        "/model switches the AI model\n/reset clears our conversation."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    state = await store.load(chat_id)
    await store.save(chat_id, {"model": state["model"], "history": []})
    await update.message.reply_text("Conversation cleared.")


async def choose_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = await store.load(update.effective_chat.id)
    await update.message.reply_text("Choose a model:", reply_markup=model_keyboard(model_key(state)))


async def model_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    key = query.data.removeprefix("model:")
    if key not in MODELS:
        await query.answer("That model is no longer available.")
        return
    chat_id = update.effective_chat.id
    state = await store.load(chat_id)
    await store.save(chat_id, {"model": key, "history": state["history"]})
    label = MODELS[key][0]
    await query.answer(f"Switched to {label}")
    await query.edit_message_text(f"Now using {label}.", reply_markup=model_keyboard(key))


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
    try:
        reply = await ask_model(chat_id, update.message.text)
    except Exception:
        log.exception("Model request failed")
        reply = "Sorry, something went wrong talking to the model. Try again, or pick another one with /model."
    await send_long(update, reply)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
    try:
        photo_file = await update.message.photo[-1].get_file()  # largest size
        data = await photo_file.download_as_bytearray()
        b64 = base64.b64encode(bytes(data)).decode()
        content = [
            {"type": "text", "text": update.message.caption or "Describe this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        reply = await ask_model(chat_id, content)
    except Exception:
        log.exception("Image request failed")
        reply = "Sorry, I couldn't process that image. Try again, or pick another model with /model."
    await send_long(update, reply)


async def post_init(app: Application) -> None:
    await store.init()
    # Populates the command menu next to the chat input box.
    await app.bot.set_my_commands(
        [
            BotCommand("model", "Choose the AI model"),
            BotCommand("reset", "Clear the conversation"),
            BotCommand("start", "Show the welcome message"),
        ]
    )


def main() -> None:
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .get_updates_connect_timeout(30)
        .get_updates_read_timeout(30)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("model", choose_model))
    app.add_handler(CallbackQueryHandler(model_selected, pattern=r"^model:"))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    log.info("Bot running. Models: %s", ", ".join(label for label, _, _ in MODELS.values()))

    # On Render (or any host that sets a public URL) use a webhook; locally, poll.
    public_url = os.environ.get("WEBHOOK_URL") or os.environ.get("RENDER_EXTERNAL_URL")
    if public_url:
        # Secret path + header derived from the token, so only Telegram can post updates.
        secret = hashlib.sha256(TELEGRAM_TOKEN.encode()).hexdigest()[:32]
        app.run_webhook(
            listen="0.0.0.0",
            port=int(os.environ.get("PORT", 10000)),
            url_path=secret,
            webhook_url=f"{public_url.rstrip('/')}/{secret}",
            secret_token=secret,
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
