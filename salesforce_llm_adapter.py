# salesforce_llm_adapter.py
import json, re
import requests
from typing import Any, Dict, List, Optional

from salesforce_agent_API import SalesforceAgentAPI

def _safe_str(x: Any) -> str:
    try:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

def _coerce_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        out = []
        for p in x:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict):
                # common shapes
                if isinstance(p.get("text"), str):
                    out.append(p["text"])
                elif isinstance(p.get("content"), str):
                    out.append(p["content"])
                else:
                    out.append(_safe_str(p))
            else:
                out.append(_safe_str(p))
        return "\n\n".join([s for s in out if s]).strip()
    if isinstance(x, dict):
        if isinstance(x.get("text"), str):
            return x["text"]
        if isinstance(x.get("content"), str):
            return x["content"]
    return _safe_str(x)

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # turn literal "\n" into real newlines, same for \t and \r
    s = s.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    # collapse 3+ blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _extract_sf_text(reply: Dict[str, Any]) -> str:
    """
    Try multiple known Salesforce Agent response shapes; return a non-empty string.
    """
    # 1) Direct message.text
    v = reply.get("message", {})
    if isinstance(v, dict):
        t = v.get("text")
        if isinstance(t, str) and t.strip():
            return t.strip()

    # 2) messages[-1].content[*].text or .content string
    msgs = reply.get("messages")
    if isinstance(msgs, list) and msgs:
        last = msgs[-1]
        content = last.get("content")
        if isinstance(content, list):
            text = _coerce_text(content)
            if text.strip():
                return text.strip()
        elif isinstance(content, str):
            if content.strip():
                return content.strip()

    # 3) outputs[0].content
    outs = reply.get("outputs")
    if isinstance(outs, list) and outs:
        c = outs[0].get("content")
        if isinstance(c, str) and c.strip():
            return c.strip()

    # 4) data.answer / result / text (other backends sometimes map here)
    for key in ("answer", "result", "text"):
        t = reply.get(key)
        if isinstance(t, str) and t.strip():
            return t.strip()

    # 5) Fallback to pretty JSON (never return None)
    return _safe_str(reply)

class SalesforceAgentLLM:
    def __init__(self, api: Optional[SalesforceAgentAPI] = None):
        self.api = api or SalesforceAgentAPI()
        self._token: Optional[str] = None
        self._session_id: Optional[str] = None
        self._sequence_id: int = 1

    def _ensure_session(self):
        if not self._token:
            self._token = self.api.get_access_token()
        if not self._session_id:
            self._session_id = self.api.start_session(self._token)
            self._sequence_id = 1

    def _refresh_and_reopen(self):
        self._token = self.api.get_access_token()
        self._session_id = self.api.start_session(self._token)
        self._sequence_id = 1

    @staticmethod
    def _render_messages(messages: List[Dict[str, Any]]) -> str:
        # Flatten OpenAI-style chat list to a single string
        lines: List[str] = []
        for m in messages or []:
            role = (m.get("role") or "user").upper()
            text = _coerce_text(m.get("content"))
            if text:
                lines.append(f"{role}: {text}")
        return "\n\n".join(lines).strip()

    def call(self, prompt: Optional[str] = None, messages: Optional[List[Dict[str, Any]]] = None, **_: Any) -> str:
        self._ensure_session()

        text = self._render_messages(messages) if messages else _coerce_text(prompt or "")
        # Defensive clamp to avoid accidental huge payloads
        if len(text) > 12000:
            text = text[:11800] + "\n\n[truncated]"

        try:
            resp = self.api.send_message_sync(self._token, self._session_id, text, sequence_id=self._sequence_id)
        except requests.HTTPError as e:
            if "401" in str(e):
                self._refresh_and_reopen()
                resp = self.api.send_message_sync(self._token, self._session_id, text, sequence_id=self._sequence_id)
            else:
                raise

        self._sequence_id += 1

        extracted = _extract_sf_text(resp).strip()
        if not extracted:
            # absolutely never hand back None/empty
            extracted = "[Salesforce Agent returned no content]"
        return _clean_text(extracted)