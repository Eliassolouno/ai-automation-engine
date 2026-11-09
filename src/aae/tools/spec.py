"""
Tool Definition and Registry with Type Validation.
"""

from typing import Callable, Dict, Any, List, Optional
import inspect
from pydantic import BaseModel, create_model


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    handler: Any


class ToolRegistry:
    """Central registry of type-safe tools for agent execution."""

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, name: Optional[str] = None, description: Optional[str] = None):
        """Decorator to register a typed python function as an agent tool."""
        def decorator(func: Callable):
            tool_name = name or func.__name__
            tool_doc = description or (func.__doc__ or "").strip() or f"Executes {tool_name}"
            
            # Simple schema generation from signature
            sig = inspect.signature(func)
            fields = {}
            for param_name, param in sig.parameters.items():
                annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
                default = ... if param.default == inspect.Parameter.empty else param.default
                fields[param_name] = (annotation, default)

            param_model = create_model(f"{tool_name}_params", **fields)
            schema = param_model.model_json_schema()

            spec = ToolSpec(
                name=tool_name,
                description=tool_doc,
                parameters_schema=schema,
                handler=func
            )
            self._tools[tool_name] = spec
            return func
        return decorator

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema
                }
            }
            for t in self._tools.values()
        ]
