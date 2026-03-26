from __future__ import annotations

from typing import Any, Dict, Tuple


def create_demo_meeting(fields: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "meeting_id": "demo_001", "fields": fields}


def create_support_ticket(fields: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "ticket_id": "ticket_001", "fields": fields}


def send_whatsapp_message(fields: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "message_id": "wa_001", "fields": fields}


def update_crm_record(fields: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "record_id": "crm_001", "fields": fields}


def run_action(action_text: str, fields: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    action = (action_text or "").lower()
    tool_call = {"name": "unknown_action", "input": fields, "output": {"status": "skipped"}}
    summary = "I could not run that action, but I noted your details."

    if "demo" in action or "meeting" in action:
        output = create_demo_meeting(fields)
        tool_call = {"name": "create_demo_meeting", "input": fields, "output": output}
        summary = "I have scheduled a demo and will confirm the details shortly."
    elif "ticket" in action or "support" in action:
        output = create_support_ticket(fields)
        tool_call = {"name": "create_support_ticket", "input": fields, "output": output}
        summary = "I have created a support ticket and will share the next steps."
    elif "whatsapp" in action or "message" in action:
        output = send_whatsapp_message(fields)
        tool_call = {"name": "send_whatsapp_message", "input": fields, "output": output}
        summary = "I have sent a WhatsApp message with the requested information."
    elif "crm" in action or "update" in action:
        output = update_crm_record(fields)
        tool_call = {"name": "update_crm_record", "input": fields, "output": output}
        summary = "I have updated your record in the CRM."

    return tool_call, summary
