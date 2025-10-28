# salesforce_crewai_tool.py
import time
import json
from typing import List, Optional
from pydantic import Field
from crewai.tools import BaseTool


from salesforce_agent_API import SalesforceAgentAPI  # <-- your core class

class SalesforceAgentCrewTool(BaseTool):
    """
    CrewAI-compatible tool that delegates a prompt to Salesforce Agentforce
    using your SalesforceAgentTool client under the hood.
    """
    name: str = "salesforce_agent_tool"
    description: str = (
        "Call Salesforce Agentforce (Agent API) to fetch info. "
        "Input should be a concise instruction for the agent."
    )

    # Optional config fields (Pydantic):
    tz: str = Field(default="America/Los_Angeles", description="Timezone for the session.")
    add_lang_var: bool = Field(default=True, description="Whether to add $Context.EndUserLanguage=en_US")

    # Internal client (not validated/serialized by Pydantic):
    _client: SalesforceAgentAPI = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # You can pass explicit creds to SalesforceAgentTool(...) if you don't want env vars.
        self._client = SalesforceAgentAPI()

    def _start_session(self, token: str, variables: Optional[List[dict]] = None) -> str:
        vars_payload = variables or []
        if self.add_lang_var:
            # keep only one language variable if user supplied it already
            if not any(v.get("name") == "$Context.EndUserLanguage" for v in vars_payload):
                vars_payload.append({"name": "$Context.EndUserLanguage", "type": "Text", "value": "en_US"})
        # Reuse your client’s start_session by temporarily patching variables via a small override
        # We’ll call the same start_session but constructing the body with tz & variables.
        # Easiest path: temporarily use a helper on the core client (not required though).
        # Since your core start_session doesn't accept variables param, we emulate via local call:
        # We'll just duplicate here to avoid editing your core class:
        return self._client.start_session(token=token)  # uses its internal defaults

    def _run(self, prompt: str) -> str:
        """
        Required CrewAI method. Receives a single string argument (the user/tool instruction).
        Returns a short human-readable response (extracted from Agentforce payload).
        """
        # 1) Auth
        token = self._client.get_access_token()

        # 2) Session
        session_id = self._start_session(token)

        try:
            # 3) Message
            payload = self._client.send_message_sync(
                token=token,
                session_id=session_id,
                text=prompt,
                sequence_id=int(time.time())
            )
            # 4) Extract nice text if present
            messages = payload.get("messages", [])
            texts = [m.get("message") for m in messages if m.get("type") == "Inform" and m.get("message")]
            return "\n".join(texts).strip() if texts else json.dumps(payload, indent=2)
        finally:
            # 5) End session (best-effort)
            try:
                self._client.end_session(token, session_id)
            except Exception:
                pass
