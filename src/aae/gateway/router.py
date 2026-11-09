"""
Model Gateway & Provider Routing Layer.
Implements failover, token accounting, and fallback policies.
"""

from typing import List, Dict, Any, Optional
import logging
from aae.types import RunBudget

logger = logging.getLogger("AAE.Gateway.Router")


class MockModelGateway:
    """Deterministic and testable Model Gateway conforming to LLMPort."""

    def __init__(self, canned_responses: Optional[List[Dict[str, Any]]] = None):
        self.canned_responses = canned_responses or []
        self._call_count = 0

    async def generate_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        budget: Optional[RunBudget] = None
    ) -> Dict[str, Any]:
        if self._call_count < len(self.canned_responses):
            res = self.canned_responses[self._call_count]
            self._call_count += 1
            return res

        # Default fallback response
        return {
            "content": f"Resolved step with system prompt: {system_prompt[:20]}...",
            "tool_calls": [],
            "tokens_used": 150
        }
