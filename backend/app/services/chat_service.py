import asyncio
import json
import base64
import logging
import re
from typing import List, AsyncGenerator
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from openai import AsyncOpenAI
from app.core.config import settings
from app.models.chat import Chat, Message
from app.models.knowledge import KnowledgeBase
from app.services.retrieval import hybrid_search

# ── Constants ─────────────────────────────────────────────────────────────────

# Number of most-recent full user/assistant turn-pairs to include verbatim.
# 3 pairs = 6 messages (3 human + 3 assistant).
_SLIDING_WINDOW_PAIRS = 3
_SLIDING_WINDOW_MESSAGES = _SLIDING_WINDOW_PAIRS * 2  # 6

_IDENTITY_PATTERNS = re.compile(
    r"^\s*(who\s+are\s+you|what\s+are\s+you|introduce\s+yourself|tell\s+me\s+about\s+yourself|"
    r"what\s+is\s+your\s+name|what('s| is)\s+your\s+purpose|what\s+can\s+you\s+do)\s*\??\s*$",
    re.IGNORECASE,
)

_IDENTITY_RESPONSE = (
    "I'm professional AI based Knowledge Assistant that answers questions using "
    "the documents and knowledge bases you've uploaded. "
    "Ask me anything about your content and I'll retrieve the most relevant information "
    "and give you a clear, cited answer."
)


def _is_identity_question(query: str) -> bool:
    return bool(_IDENTITY_PATTERNS.match(query.strip()))


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        temperature=0,
        streaming=True,
        model=settings.OPENAI_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
    )


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── Summary ───────────────────────────────────────────────────────────────────

async def _summarise_older_messages(
    messages_to_summarise: List[dict],
    existing_summary: str | None,
    chat_id: int,
) -> str:
    """
    Produce (or update) a rolling summary that covers all dialogue *outside*
    the sliding window.

    If an existing summary exists, it is passed in and the new batch of
    messages is folded into it — so the summary is always cumulative.
    """
    # Build a plain-text transcript of the messages being summarised
    transcript_lines = []
    for m in messages_to_summarise:
        role = "User" if m["role"] == "user" else "Assistant"
        # Strip the base64 context prefix that assistant messages contain
        content = m["content"]
        if "__LLM_RESPONSE__" in content:
            content = content.split("__LLM_RESPONSE__")[-1]
        transcript_lines.append(f"{role}: {content.strip()}")
    transcript = "\n".join(transcript_lines)

    if existing_summary:
        system_prompt = (
            "You are a precise dialogue summariser. "
            "You will be given a running summary of an earlier part of a conversation "
            "and a new batch of exchanges to fold into it.\n\n"
            "Rules:\n"
            "- Produce a single, compact summary that covers everything: the existing summary PLUS the new exchanges.\n"
            "- Preserve every fact, decision, preference, and piece of information the user provided or the assistant stated.\n"
            "- Capture questions asked and answers given — especially facts extracted from documents.\n"
            "- Keep it dense but readable — use short bullet points or tightly written prose.\n"
            "- Do NOT omit details; losing information defeats the purpose.\n"
            "- Output ONLY the updated summary — no preamble, no labels, no extra text."
        )
        user_prompt = (
            f"EXISTING SUMMARY:\n{existing_summary}\n\n"
            f"NEW EXCHANGES TO FOLD IN:\n{transcript}"
        )
    else:
        system_prompt = (
            "You are a precise dialogue summariser. "
            "You will be given a conversation transcript to summarise.\n\n"
            "Rules:\n"
            "- Produce a compact summary that captures every significant fact, question, answer, and decision.\n"
            "- Include what documents or topics were discussed and what was found.\n"
            "- Keep it dense but readable — use short bullet points or tightly written prose.\n"
            "- Do NOT omit details; losing information defeats the purpose.\n"
            "- Output ONLY the summary — no preamble, no labels, no extra text."
        )
        user_prompt = f"CONVERSATION TRANSCRIPT:\n{transcript}"

    logger.info("[SUMMARY] chat_id=%d | summarising %d messages | has_existing=%s",
                chat_id, len(messages_to_summarise), bool(existing_summary))

    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_API_BASE,
    )
    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        stream=False,
    )
    summary = _strip_think(response.choices[0].message.content or "")
    logger.info("[SUMMARY] chat_id=%d | summary_length=%d chars", chat_id, len(summary))
    return summary


async def _maybe_update_summary(
    chat_id: int,
    all_prior_messages: List[dict],
    existing_summary: str | None,
) -> None:
    """
    Called as a fire-and-forget background task after the response stream
    completes. Checks whether there are messages beyond the sliding window
    and, if so, summarises them and persists the result.

    Uses a fresh DB session so it doesn't interfere with the main request.
    """
    from app.db.session import SessionLocal

    # Messages outside the window are everything except the last N
    # (window) + the pair just completed (2 more = current user + bot turn).
    # At call time all_prior_messages already includes only historical turns
    # (not the current one) — the current pair gets appended here.
    # Actually: all_prior_messages is the full history BEFORE the current turn.
    # The current turn was just committed. We need to reload from DB.
    db = SessionLocal()
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            return

        # Full ordered message list from DB (just committed, so current turn is included)
        db_messages = (
            db.query(Message)
            .filter(Message.chat_id == chat_id)
            .order_by(Message.id)
            .all()
        )

        # Convert to plain dicts, excluding empty bot placeholders
        all_msgs = [
            {"role": m.role, "content": m.content}
            for m in db_messages
            if m.content.strip()
        ]

        total = len(all_msgs)
        if total <= _SLIDING_WINDOW_MESSAGES:
            # Nothing outside the window yet — no summary needed
            logger.info("[SUMMARY] chat_id=%d | total=%d msgs <= window=%d, skip",
                        chat_id, total, _SLIDING_WINDOW_MESSAGES)
            return

        # Messages older than the sliding window
        older = all_msgs[: total - _SLIDING_WINDOW_MESSAGES]

        # Only re-summarise if there are new messages beyond what was
        # previously summarised. Track this by comparing count.
        # Simple heuristic: re-summarise whenever we have more "older"
        # messages than before (i.e. the window has slid forward).
        # We persist the summary unconditionally each time for simplicity.
        summary = await _summarise_older_messages(
            messages_to_summarise=older,
            existing_summary=existing_summary,
            chat_id=chat_id,
        )

        chat.history_summary = summary
        db.commit()
        logger.info("[SUMMARY] chat_id=%d | summary persisted (%d chars)", chat_id, len(summary))

    except Exception as e:
        logger.error("[SUMMARY] chat_id=%d | error: %s", chat_id, e)
    finally:
        db.close()


# ── Query rewrite ──────────────────────────────────────────────────────────────

async def _rewrite_query(
    query: str,
    recent_history: List,  # LangChain message objects
) -> str:
    """
    Condense the current query + recent chat history into a self-contained
    standalone question for retrieval.
    """
    if not recent_history:
        return query

    llm = _make_llm()
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is. "
            "Your only task is to rewrite the user's question as a fully self-contained question "
            "that can be understood without the chat history. "
            "Output ONLY the rewritten question — no explanations, no answers, no extra text. "
            "If the question is already self-contained, output it unchanged. "
            "Never answer the question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    condense_chain = contextualize_q_prompt | llm | StrOutputParser()
    raw_rewrite = (await condense_chain.ainvoke(
        {"input": query, "chat_history": recent_history}
    )).strip()

    had_think = bool(re.search(r"<think>", raw_rewrite))
    standalone = _strip_think(raw_rewrite) or query
    logger.info("[STEP 1] raw_rewrite=%r | had_think=%s | standalone=%r",
                raw_rewrite[:300], had_think, standalone)
    return standalone


# ── Main entry point ──────────────────────────────────────────────────────────

async def generate_response(
    query: str,
    messages: dict,
    knowledge_base_ids: List[int],
    chat_id: int,
    db: Session
) -> AsyncGenerator[str, None]:
    logger.info("=" * 70)
    logger.info("[CHAT] chat_id=%s | kb_ids=%s | query=%r", chat_id, knowledge_base_ids, query)

    try:
        # Persist user message
        user_message = Message(content=query, role="user", chat_id=chat_id)
        db.add(user_message)
        db.commit()

        # Persist bot placeholder
        bot_message = Message(content="", role="assistant", chat_id=chat_id)
        db.add(bot_message)
        db.commit()

        # ── Identity shortcut ──────────────────────────────────────────────
        if _is_identity_question(query):
            logger.info("[CHAT] identity shortcut — skipping RAG")
            yield f'0:{json.dumps(_IDENTITY_RESPONSE)}\n'
            yield 'd:{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0}}\n'
            bot_message.content = _IDENTITY_RESPONSE
            db.commit()
            return

        # ── Check knowledge bases ──────────────────────────────────────────
        knowledge_bases = (
            db.query(KnowledgeBase)
            .filter(KnowledgeBase.id.in_(knowledge_base_ids))
            .all()
        )
        if not knowledge_bases:
            error_msg = "I don't have any knowledge base to help answer your question."
            yield f'0:"{error_msg}"\n'
            yield 'd:{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0}}\n'
            bot_message.content = error_msg
            db.commit()
            return

        # ── Load chat-level state (existing summary) ───────────────────────
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        existing_summary: str | None = chat.history_summary if chat else None

        # ── Build sliding window from prior messages ───────────────────────
        # prior_messages = all messages in payload except the current one
        prior_messages = messages["messages"][:-1]
        logger.info("[CHAT] total_payload_msgs=%d | prior=%d",
                    len(messages["messages"]), len(prior_messages))

        # Take only the last N messages (sliding window)
        window_messages = prior_messages[-_SLIDING_WINDOW_MESSAGES:]
        older_count = max(0, len(prior_messages) - _SLIDING_WINDOW_MESSAGES)

        # Convert window to LangChain message objects for query rewrite
        recent_lc_history = []
        for m in window_messages:
            if m["role"] == "user":
                recent_lc_history.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                content = m["content"]
                if "__LLM_RESPONSE__" in content:
                    content = content.split("__LLM_RESPONSE__")[-1]
                recent_lc_history.append(AIMessage(content=content))

        logger.info("[CHAT] sliding_window=%d msgs | older=%d msgs | has_summary=%s",
                    len(window_messages), older_count, bool(existing_summary))

        # ── Step 1: Rewrite query using sliding window ─────────────────────
        logger.info("[STEP 1] condense | window_turns=%d", len(recent_lc_history))
        standalone_question = await _rewrite_query(query, recent_lc_history)
        logger.info("[STEP 1] standalone_question=%r", standalone_question)

        # ── Step 2: Hybrid retrieval ───────────────────────────────────────
        logger.info("[STEP 2] hybrid_search | query=%r", standalone_question)
        docs = await hybrid_search(
            query=standalone_question,
            kb_ids=knowledge_base_ids,
            db=db,
        )
        logger.info("[STEP 2] returned %d docs", len(docs))
        for i, doc in enumerate(docs):
            snippet = doc.page_content[:120].replace("\n", " ")
            logger.info("  chunk[%d] meta=%s | text=%r", i, doc.metadata, snippet)

        # ── Step 3: Emit base64 context chunk ─────────────────────────────
        serializable_context = [
            {
                "page_content": doc.page_content.replace('"', '\\"'),
                "metadata": doc.metadata,
            }
            for doc in docs
        ]
        base64_context = base64.b64encode(
            json.dumps({
                "context": serializable_context,
                "rewritten_query": standalone_question,
            }).encode()
        ).decode()
        separator = "__LLM_RESPONSE__"
        yield f'0:"{base64_context}{separator}"\n'
        full_response = base64_context + separator

        # ── Step 4: QA answer with context + sliding window + summary ──────
        formatted_context = "\n\n".join(
            f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(docs)
        )

        # Build the system prompt — inject summary when it exists
        summary_section = ""
        if existing_summary:
            summary_section = (
                "\n\n## Earlier Conversation Summary\n"
                "The following is a summary of the earlier part of this conversation "
                "(before the recent exchanges shown in the chat history below). "
                "Use it for context but prioritise the retrieved documents and recent exchanges:\n"
                f"{existing_summary}"
            )

        qa_system_prompt = (
            "You are a professional AI-based Knowledge Assistant that answers questions using the provided context documents.\n\n"
            "## Formatting\n"
            "- Use **bold** for key terms, concepts, and important phrases.\n"
            "- Use *italics* for definitions, technical terms, or emphasis.\n"
            "- Use numbered lists (1. 2. 3.) for sequential steps or ordered items.\n"
            "- Use bullet points (- or *) for non-ordered lists, features, or comparisons.\n"
            "- Use headings (##, ###) only for longer multi-section answers.\n"
            "- Keep paragraphs short and well-separated for readability.\n\n"
            "## Citations\n"
            "You will be given context documents numbered sequentially starting from 1.\n"
            "You MUST cite sources using EXACTLY this format: [citation:x] — for example: 'The sky is blue [citation:1].'\n"
            "Do NOT use any other citation format such as [1], (1), Context [1], or footnotes.\n"
            "If a sentence draws from multiple contexts, list all applicable citations: [citation:1][citation:2].\n\n"
            "## General\n"
            "- Your answer must be correct, accurate, and written in a professional, unbiased tone.\n"
            "- Limit your response to 2048 tokens.\n"
            "- Do not include information unrelated to the question, and do not repeat yourself.\n"
            "- If the provided context does not contain sufficient information, say so briefly and professionally.\n"
            "- Write in the same language as the question (except for code, citations, and proper nouns).\n"
            "- Do not blindly repeat the contexts verbatim; synthesise and explain.\n\n"
            f"Context:\n{formatted_context}"
            f"{summary_section}"
        )

        # Build OpenAI messages directly — do NOT use ChatPromptTemplate here.
        # Template formatting parses curly-braces in user content (e.g. LaTeX
        # citations like {author1, author2}) as variable placeholders, raising
        # KeyError when the context contains arbitrary document text.
        logger.info("[STEP 4] QA | model=%s | standalone=%r | context_chunks=%d | window_msgs=%d | has_summary=%s",
                    settings.OPENAI_MODEL, standalone_question, len(docs),
                    len(recent_lc_history), bool(existing_summary))

        openai_messages = [{"role": "system", "content": qa_system_prompt}]
        for lc_msg in recent_lc_history:
            if isinstance(lc_msg, HumanMessage):
                openai_messages.append({"role": "user", "content": lc_msg.content})
            elif isinstance(lc_msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": lc_msg.content})
        openai_messages.append({"role": "user", "content": standalone_question})

        openai_client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
        )

        stream = await openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=openai_messages,
            temperature=0,
            stream=True,
        )

        # reasoning_content is emitted by models like carnice/QwQ/DeepSeek-R1
        # via the OpenAI-style `delta.reasoning_content` field instead of
        # wrapping in <think> tags inline. We collect it and emit a synthetic
        # <think>...</think> block so the frontend ThinkBlock renders correctly.
        reasoning_buf = ""
        reasoning_closed = False  # track whether we've sent </think> yet

        async for chunk in stream:
            delta = chunk.choices[0].delta

            # ── reasoning tokens ──────────────────────────────────────────
            reasoning_token = getattr(delta, "reasoning_content", None) or ""
            if reasoning_token:
                if not reasoning_buf:
                    # Open the think block on the first reasoning token
                    open_tag = "<think>"
                    full_response += open_tag
                    yield f'0:{json.dumps(open_tag)}\n'
                reasoning_buf += reasoning_token
                full_response += reasoning_token
                yield f'0:{json.dumps(reasoning_token)}\n'
                continue

            # ── answer tokens ─────────────────────────────────────────────
            chunk_text = delta.content or ""
            if not chunk_text:
                continue

            # Close the think block exactly once, on the first answer token
            if reasoning_buf and not reasoning_closed:
                close_tag = "</think>"
                full_response += close_tag
                yield f'0:{json.dumps(close_tag)}\n'
                reasoning_closed = True

            full_response += chunk_text
            yield f'0:{json.dumps(chunk_text)}\n'

        # Edge case: model produced reasoning but no answer tokens at all
        if reasoning_buf and not reasoning_closed:
            close_tag = "</think>"
            full_response += close_tag
            yield f'0:{json.dumps(close_tag)}\n'

        logger.info("[STEP 4] streaming complete | response_length=%d chars", len(full_response))
        bot_message.content = full_response
        db.commit()

        # ── Post-turn: schedule summary update (fire-and-forget) ──────────
        # Runs after the stream is fully consumed by the client.
        asyncio.create_task(
            _maybe_update_summary(
                chat_id=chat_id,
                all_prior_messages=prior_messages,
                existing_summary=existing_summary,
            )
        )

    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        print(error_message)
        yield '3:{text}\n'.format(text=error_message)
        if 'bot_message' in locals():
            bot_message.content = error_message
            db.commit()
    finally:
        db.close()
