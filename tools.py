from datetime import datetime
from typing import Any
import sys
import os
sys.path.insert(
0,
"/home/karam-ali-jan/Documents/Research_work/system/final_corpus_package"
)
from final_corpus_package.retrieval import search_chunks,generate_rag_answer,generate_rag_prompt

 
 
# ---------------------------------------------------------------------------
# Mock data stores
# ---------------------------------------------------------------------------
 
_LEAVE_DB = {
    "EMP1001": {"annual_leave": 12, "sick_leave": 5},
    "EMP1042": {"annual_leave": 8, "sick_leave": 3},
    "EMP1077": {"annual_leave": 15, "sick_leave": 7},
}
 
_DIRECTORY = {
    "ahmed raza": {"extension": "4471", "department": "Finance", "email": "ahmed.raza@company.com"},
    "sara khan": {"extension": "2210", "department": "HR", "email": "sara.khan@company.com"},
    "bilal ahmed": {"extension": "3390", "department": "Engineering", "email": "bilal.ahmed@company.com"},
}
_POLICY_CORPUS = {
    "policy_doc_travel_03": (
        "Travel Reimbursement Policy. International business class flights: "
        "maximum reimbursable amount is PKR 250,000 per trip. Manager "
        "pre-approval is required for any single expense above PKR 100,000. "
        "Domestic economy flights do not require pre-approval."
    ),
    "policy_doc_expense_07": (
        "Expense Policy. Eligible categories: transport, meals, accommodation, "
        "client entertainment, and approved training. Personal fitness, gym "
        "memberships, and other wellness expenses are not covered under this "
        "policy."
    ),
    "policy_doc_leave_02": (
        "Leave Policy. Employees accrue 1.5 annual leave days per month. "
        "Sick leave is capped at 10 days per year and does not carry over. "
        "Unused annual leave up to 5 days may carry over to the next year."
    ),
}
 
_TICKETS: list[dict] = []
_EXPENSES: list[dict] = []
_MEETINGS: dict[str, dict] = {}  # keyed by title, so re-scheduling updates in place
_ROOM_BOOKINGS: list[dict] = []
 
 
def check_leave_balance(employee_id:str)->dict:
    record=_LEAVE_DB.get(employee_id.strip().upper())
    if record is None:
        return{"error":f"No employee record for '{employee_id}'"}
    return{"employee_id":employee_id,**record} 

def employee_directory_lookup(name:str)->dict:
    key=name.strip().lower()
    record=_DIRECTORY.get(key)
    if record is None:
        return{"error":f"No directory match for '{name}'"}
    return{"name":name,**record}

def policy_search(query_topic:str)->dict:
    results = search_chunks(query_topic, top_k=3)
    prompt = generate_rag_prompt(results, query_topic)
    answer = generate_rag_answer(prompt)
    return {
        "answer": answer,
        "retrieved_chunk_ids": [r["chunk_id"] for r in results],
        "retrieved_scores": [r["score"] for r in results],
    }
    

def create_it_ticket(subject:str,priority:str="medium",description:str="")->dict:
    priority=priority.strip().lower()
    if priority not in {"low","medium","high"}:
        return{"error":f"invalid priority {priority} --must be high low or medium "}
    ticket={}
    ticket = {
        "ticket_id": f"TCK{len(_TICKETS) + 1:04d}",
        "subject": subject,
        "priority": priority,
        "description": description,
    }
    _TICKETS.append(ticket)
    return{"status":"created",**ticket}

def submit_expense_reimbursement(amount:float,currency:str,category:str,date:str)->dict:
    try:
     datetime.strptime(date,"%Y-%m-%d")
    except ValueError as e:
        return {"error":f"invalid date format :{e}"}
    if amount <=0:
        return {"error":"Amount must be positive"}
    entry={
        "expense_id":f"EXP{len(_EXPENSES)+1:04d}",
        "amount":amount,
        "currency":currency.upper(),
        "category":category,
        "date":date
    }
    _EXPENSES.append(entry)

def schedule_meeting(title: str, date: str, time: str, attendees: list[str] | None = None) -> dict:
    """Schedule (or reschedule, if the same title exists) a meeting.
 
    Using title as the key lets a multi-turn task update the same meeting
    across turns (e.g. change the time) without a separate 'update' tool --
    keeps the multiturn_goal_persistence tasks simple to write.
    """
    try:
        datetime.strptime(date, "%Y-%m-%d")
        datetime.strptime(time, "%H:%M")
    except ValueError as e:
        return {"error": f"Invalid date/time format: {e}"}
 
    attendees = attendees or []
    if title in _MEETINGS:
        # Update in place; keep prior attendees unless new ones are given.
        _MEETINGS[title]["date"] = date
        _MEETINGS[title]["time"] = time
        if attendees:
            _MEETINGS[title]["attendees"] = attendees
        return {"status": "updated", "title": title, **_MEETINGS[title]}
 
    _MEETINGS[title] = {"date": date, "time": time, "attendees": attendees}
    return {"status": "created", "title": title, **_MEETINGS[title]}
 
 
def book_meeting_room(room: str = "", date: str = "", time: str = "", party_size: int = 0) -> dict:
    """Book a meeting room. Returns a clarification request if fields are
    missing -- this is the tool your ambiguous_clarification tasks target."""
    missing = [f for f, v in [("room", room), ("date", date), ("time", time), ("party_size", party_size)] if not v]
    if missing:
        return {"status": "needs_clarification", "missing_fields": missing}
 
    booking = {"room": room, "date": date, "time": time, "party_size": party_size}
    _ROOM_BOOKINGS.append(booking)
    return {"status": "booked", **booking}
 
 
# ---------------------------------------------------------------------------
# Schemas (Anthropic tool-use format -- adapt trivially for OpenAI's format)
# ---------------------------------------------------------------------------
 
TOOL_SCHEMAS = [
    {
        "name": "check_leave_balance",
        "description": "Look up an employee's remaining annual, casual, and sick leave balance. Use this when the user is asking how many leave days they have left or how much leave they've used. Do not use this to request or apply for leave.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {
                    "type": "string"
                }
            },
            "required": ["employee_id"]
        }
    },

    {
        "name": "employee_directory_lookup",
        "description": "Look up a colleague's contact information such as extension number, department, and email by their name. Use this when the user wants to find or contact another employee.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                }
            },
            "required": ["name"]
        }
    },

    {
        "name": "policy_search",
        "description": "Search the company's workplace policy documents to answer questions about company rules, eligibility, entitlements, and procedures. Use this when the user is asking about a policy or procedure rather than performing an action.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_topic": {
                    "type": "string"
                }
            },
            "required": ["query_topic"]
        }
    },

    {
        "name": "create_it_ticket",
        "description": "Create an IT support ticket for a hardware, software, or access problem the user is experiencing. Use this when the user is reporting an actual IT problem that needs support.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string"
                },
                "priority": {
                    "type": "string",
                    "enum": [
                        "low",
                        "medium",
                        "high"
                    ]
                },
                "description": {
                    "type": "string"
                }
            },
            "required": ["subject"]
        }
    },

    {
        "name": "submit_expense_reimbursement",
        "description": "Submit a request to be reimbursed for a business expense the user has already incurred. Use this when the user wants to claim money back for something they paid for.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number"
                },
                "currency": {
                    "type": "string"
                },
                "category": {
                    "type": "string"
                },
                "date": {
                    "type": "string",
                    "description": "YYYY-MM-DD"
                }
            },
            "required": [
                "amount",
                "currency",
                "category",
                "date"
            ]
        }
    },

    {
        "name": "schedule_meeting",
        "description": "Schedule a new internal meeting between colleagues. Use this for scheduling meetings. Do not use this for booking a physical meeting room.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string"
                },
                "date": {
                    "type": "string",
                    "description": "YYYY-MM-DD"
                },
                "time": {
                    "type": "string",
                    "description": "HH:MM, 24-hour"
                },
                "attendees": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": [
                "title",
                "date",
                "time"
            ]
        }
    },

    {
        "name": "book_meeting_room",
        "description": "Book a physical meeting room. Use this when the user needs a room for an in-person meeting. Requires a room, date, time, and party size. If information is missing, call the tool with the information available instead of guessing.",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {
                    "type": "string"
                },
                "date": {
                    "type": "string"
                },
                "time": {
                    "type": "string"
                },
                "party_size": {
                    "type": "integer"
                }
            }
        }
    }
]
 
def get_tool_registry() -> dict[str, Any]:
    """Return {tool_name: callable} for the harness to execute against."""
    return {
        "check_leave_balance": check_leave_balance,
        "employee_directory_lookup": employee_directory_lookup,
        "policy_search": policy_search,
        "create_it_ticket": create_it_ticket,
        "submit_expense_reimbursement": submit_expense_reimbursement,
        "schedule_meeting": schedule_meeting,
        "book_meeting_room": book_meeting_room,
    }
 
 
if __name__ == "__main__":
    # Quick smoke test: python enterprise_tools.py
    print(check_leave_balance("EMP1042"))
    print(employee_directory_lookup("Ahmed Raza"))
    print(policy_search("international business class flight reimbursement"))
    print(policy_search("gym membership"))  # should trigger low-similarity / error
    print(create_it_ticket("Laptop won't turn on", priority="high", description="Urgent, client call in 1 hour"))
    print(submit_expense_reimbursement(4500, "PKR", "transport", "2026-03-12"))
    print(schedule_meeting("Budget Review", "2026-03-20", "14:00", ["Sara", "Bilal"]))
    print(schedule_meeting("Budget Review", "2026-03-20", "15:00"))  # reschedule, same title
    print(book_meeting_room())  # should ask for clarification
    print(book_meeting_room(room="Room A", date="2026-03-21", time="10:00", party_size=4))
