"""
Disaster Relief Coordination — Tool Registry
==============================================
Maps tool name strings to handler functions, validates arguments, and
provides tool metadata for observations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from src.env.state import WorldState
from src.env.tools_intake import assess_report_urgency, classify_report, verify_report
from src.env.tools_dispatch import get_resources, send_resource, reroute_resource
from src.env.tools_monitor import check_operation, close_case, mark_false_report
from src.env.tools_coordinator import call_dispatch_agent, call_intake_agent, call_monitor_agent


# ---------------------------------------------------------------------------
# Tool metadata: name → (handler, required_args, optional_args, description)
# ---------------------------------------------------------------------------

ToolHandler = Callable[..., Tuple[str, float]]

_TOOL_DEFS: Dict[str, Dict[str, Any]] = {
    # --- Intake ---
    "classify_report": {
        "handler": classify_report,
        "required": ["report_id"],
        "optional": [],
        "description": "Classify a report's category (flood, medical, fire, etc.) from its raw text.",
    },
    "assess_report_urgency": {
        "handler": assess_report_urgency,
        "required": ["report_id"],
        "optional": [],
        "description": "Assess a report's urgency on a 0-10 scale.",
    },
    "verify_report": {
        "handler": verify_report,
        "required": ["report_id"],
        "optional": [],
        "description": "Verify a report's authenticity (real, duplicate, or false alarm).",
    },
    # --- Dispatch ---
    "get_resources": {
        "handler": get_resources,
        "required": [],
        "optional": [],
        "description": "List all resources and their current status.",
    },
    "send_resource": {
        "handler": send_resource,
        "required": ["resource_id", "report_id"],
        "optional": [],
        "description": "Dispatch a resource to respond to a report.",
    },
    "reroute_resource": {
        "handler": reroute_resource,
        "required": ["resource_id"],
        "optional": ["route_hint"],
        "description": "Reroute a deployed resource that is stuck on a blocked route.",
    },
    # --- Monitor ---
    "check_operation": {
        "handler": check_operation,
        "required": ["target_id"],
        "optional": [],
        "description": "Check the status of a report (RPT-xxx) or assignment (ASG-xxx).",
    },
    "close_case": {
        "handler": close_case,
        "required": ["report_id"],
        "optional": ["resolution_note"],
        "description": "Close a resolved case and mark the report as complete.",
    },
    "mark_false_report": {
        "handler": mark_false_report,
        "required": ["report_id"],
        "optional": ["reason"],
        "description": "Flag a report as a false alarm or duplicate.",
    },
    # --- Coordinator (delegation) ---
    "call_intake_agent": {
        "handler": call_intake_agent,
        "required": ["report_id"],
        "optional": ["instruction"],
        "description": "Delegate full intake processing (classify + urgency + verify) to the intake sub-agent.",
    },
    "call_dispatch_agent": {
        "handler": call_dispatch_agent,
        "required": ["resource_id", "report_id"],
        "optional": [],
        "description": "Delegate resource dispatch to the dispatch sub-agent (shows inventory + dispatches).",
    },
    "call_monitor_agent": {
        "handler": call_monitor_agent,
        "required": ["target_id"],
        "optional": ["instruction"],
        "description": "Delegate monitoring to the monitor sub-agent. Add 'close' in instruction to auto-close.",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tool_names() -> List[str]:
    """Return all available tool names."""
    return list(_TOOL_DEFS.keys())


def get_tool_descriptions() -> Dict[str, str]:
    """Return {tool_name: description} for all tools."""
    return {name: defn["description"] for name, defn in _TOOL_DEFS.items()}


def get_tool_signatures() -> List[str]:
    """Return human-readable tool signatures for the observation."""
    sigs = []
    for name, defn in _TOOL_DEFS.items():
        required = ", ".join(defn["required"])
        optional = ", ".join(f"{a}=..." for a in defn["optional"])
        params = ", ".join(filter(None, [required, optional]))
        sigs.append(f"{name}({params}) — {defn['description']}")
    return sigs


def execute_tool(
    state: WorldState,
    tool_name: str,
    args: Dict[str, Any],
) -> Tuple[str, float]:
    """
    Execute a tool by name with the given arguments.

    Returns (result_text, reward_delta).
    Handles unknown tools and missing arguments gracefully.
    """
    if tool_name not in _TOOL_DEFS:
        available = ", ".join(get_tool_names())
        return (
            f"Error: Unknown tool '{tool_name}'. Available tools: {available}.",
            -0.5,
        )

    defn = _TOOL_DEFS[tool_name]
    handler: ToolHandler = defn["handler"]
    required: List[str] = defn["required"]

    # Validate required arguments
    missing = [arg for arg in required if arg not in args]
    if missing:
        return (
            f"Error: Tool '{tool_name}' missing required arguments: {missing}. "
            f"Required: {required}, Optional: {defn['optional']}.",
            -0.5,
        )

    # Build kwargs: state + declared args only (ignore unknown args)
    kwargs: Dict[str, Any] = {}
    for arg_name in required + defn["optional"]:
        if arg_name in args:
            kwargs[arg_name] = args[arg_name]

    try:
        return handler(state, **kwargs)
    except Exception as e:
        return f"Error executing '{tool_name}': {e}", -0.5
