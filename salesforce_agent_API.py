import os
import uuid
import json
import requests
from dotenv import load_dotenv, find_dotenv

# Load .env
load_dotenv(find_dotenv(), override=False)

DEFAULT_TIMEOUT = 30


class SalesforceAgentAPI:
    def __init__(
        self,
        my_domain=None,
        client_id=None,
        client_secret=None,
        agent_id=None,
        sf_api_host=None,
    ):
        # Read from .env (use defaults if not provided)
        self.MY_DOMAIN = my_domain or os.getenv("SF_ORG_DOMAIN", "https://your-domain.my.salesforce.com")
        self.CLIENT_ID = client_id or os.getenv("SF_CLIENT_ID", "")
        self.CLIENT_SECRET = client_secret or os.getenv("SF_CLIENT_SECRET", "")
        self.AGENT_ID = agent_id or os.getenv("SF_AGENT_ID", "")
        self.SF_API_HOST = sf_api_host or os.getenv("SF_API_HOST", "https://api.salesforce.com")

    # --- Utility ---
    def _raise_for_status_with_body(self, resp: requests.Response):
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            msg = f"HTTP {resp.status_code} {resp.reason} for {resp.request.method} {resp.url}\n"
            try:
                msg += json.dumps(resp.json(), indent=2)
            except Exception:
                msg += resp.text
            raise requests.HTTPError(msg) from e

    # --- Core Methods ---
    def get_access_token(self) -> str:
        """OAuth2 client-credentials authentication."""
        url = f"{self.MY_DOMAIN}/services/oauth2/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.CLIENT_ID,
            "client_secret": self.CLIENT_SECRET,
        }
        r = requests.post(url, data=data, timeout=DEFAULT_TIMEOUT)
        self._raise_for_status_with_body(r)
        return r.json()["access_token"]

    def start_session(self, token: str) -> str:
        """Start a new Agent session."""
        url = f"{self.SF_API_HOST}/einstein/ai-agent/v1/agents/{self.AGENT_ID}/sessions"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        body = {
            "externalSessionKey": str(uuid.uuid4()),
            "instanceConfig": {"endpoint": self.MY_DOMAIN},
            "tz": "America/Los_Angeles",
            "bypassUser": True,
            "featureSupport": "Streaming",
            "streamingCapabilities": {"chunkTypes": ["Text"]},
            "variables": [
                {"name": "$Context.EndUserLanguage", "type": "Text", "value": "en_US"}
            ],
        }
        r = requests.post(url, json=body, headers=headers, timeout=DEFAULT_TIMEOUT)
        self._raise_for_status_with_body(r)
        return r.json()["sessionId"]

    def send_message_sync(self, token: str, session_id: str, text: str, sequence_id: int = 1) -> dict:
        """Send a synchronous text message to the session."""
        url = f"{self.SF_API_HOST}/einstein/ai-agent/v1/sessions/{session_id}/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        body = {
            "message": {"sequenceId": sequence_id, "type": "Text", "text": text},
            "variables": [],
        }
        r = requests.post(url, json=body, headers=headers, timeout=DEFAULT_TIMEOUT)
        self._raise_for_status_with_body(r)
        return r.json()

    def end_session(self, token: str, session_id: str, reason: str = "UserRequest") -> dict:
        """End the session gracefully."""
        url = f"{self.SF_API_HOST}/einstein/ai-agent/v1/sessions/{session_id}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "x-session-end-reason": reason,
        }
        r = requests.delete(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        self._raise_for_status_with_body(r)
        try:
            return r.json()
        except Exception:
            return {"ok": True, "status": r.status_code}


if __name__ == "__main__":
    # Reads everything from .env automatically
    api = SalesforceAgentAPI()

    token = api.get_access_token()
    print("Got token OK (length):", len(token))

    sid = api.start_session(token)
    print("Session started:", sid)

    try:
        reply = api.send_message_sync(token, sid, "Hello, what services do you offer?")
        print("Reply payload:", json.dumps(reply, indent=2))
    finally:
        ended = api.end_session(token, sid)
        print("Session ended:", ended)
