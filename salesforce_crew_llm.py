# salesforce_crew_llm.py
from typing import Optional, List, Dict, Any
from crewai import BaseLLM      # <-- use BaseLLM as per docs
from salesforce_agent_API import SalesforceAgentAPI
from salesforce_llm_adapter import SalesforceAgentLLM


from salesforce_agent_API import SalesforceAgentAPI  # your env-only API client from earlier
from salesforce_llm_adapter import SalesforceAgentLLM  # the simple adapter with .call()

class SalesforceCrewLLM(BaseLLM):
    """
    CrewAI-compatible LLM that routes calls directly to Salesforce Einstein Agent
    via SalesforceAgentLLM. This bypasses LiteLLM completely.
    """
    def __init__(self, api: Optional[SalesforceAgentAPI] = None):
        # pass benign values to parent; they won't be used because we override call()
        super().__init__(model="salesforce-einstein", temperature=None) 
        self._sf = SalesforceAgentLLM(api=api or SalesforceAgentAPI())

    # CrewAI will call this; DO NOT call super().call()
    def call(self, prompt: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None, **kwargs: Any) -> str:
        # You can optionally trim messages here if needed
        return self._sf.call(prompt=prompt, messages=messages, **kwargs)

    # optional: expose a clean close for session hygiene
    def close(self, reason: str = "UserRequest"):
        try:
            self._sf.close(reason)
        except Exception:
            pass
