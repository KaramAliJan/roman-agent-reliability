
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable
import sys

from config import GEMINI_API_KEY
from google import genai
from google.genai import types



# ---------------------------------------------------------
# Import tools
# ---------------------------------------------------------

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "tools")
)

from tools import (
    TOOL_SCHEMAS,
    get_tool_registry,
    check_leave_balance,
    submit_expense_reimbursement,
    book_meeting_room,
    employee_directory_lookup,
    policy_search,
    create_it_ticket,
    schedule_meeting,
)


# ---------------------------------------------------------
# Gemini client
# ---------------------------------------------------------

client = genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------
# Result structure
# ---------------------------------------------------------

@dataclass
class RunResult:
    task_id: str
    variant: str = ""
    input_text: str = ""
    predicted_intent: str = ""
    tool_called: str = ""
    tool_params: dict = field(default_factory=dict)
    tool_output: dict = field(default_factory=dict)
    final_answer: str = ""
    raw_trace: list = field(default_factory=list)


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

class LLM:

    @staticmethod
    def call_model(
        text: str,
        tool_schema: list[dict]
    ) -> dict:

        system_prompt = """
You are an enterprise assistant.

Choose the appropriate tool when the user's request requires
one of the available tools.

Return a function call when a tool is required.
Do not call a tool when no tool is appropriate.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                 tools=[
        types.Tool(
            function_declarations=tool_schema
        )
    ],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="AUTO"
                    )
                ),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

        function_call = None
        model_text = ""

        # -------------------------------------------------
        # Extract model response
        # -------------------------------------------------

        for candidate in response.candidates or []:

            if not candidate.content:
                continue

            for part in candidate.content.parts:

                if part.function_call is not None:
                    function_call = part.function_call
                    break

                if part.text:
                    model_text += part.text

            if function_call is not None:
                break

        # -------------------------------------------------
        # No tool was called
        # -------------------------------------------------

        if function_call is None:
            return {
                "intent": "no_tool_called",
                "tool": None,
                "params": {},
                "model_text": model_text,
            }

        # -------------------------------------------------
        # Tool was called
        # -------------------------------------------------

        return {
            "intent": function_call.name,
            "tool": function_call.name,
            "params": (
                dict(function_call.args)
                if function_call.args
                else {}
            ),
            "model_text": model_text,
        }

    # -----------------------------------------------------
    # Run one task
    # -----------------------------------------------------

    @staticmethod
    def run_task(
        task: dict,
        variant: str,
        call_model: Callable[[str, list], dict]
    ) -> RunResult:

        text_fields = {
            "english": "english_text",
            "roman_urdu_natural": "roman_urdu_natural",
            "roman_urdu_codemixed": "roman_urdu_codemixed",
        }

        text_field = text_fields[variant]

        input_text = task[text_field]

        result = RunResult(
            task_id=task["task_id"],
            variant=variant,
            input_text=input_text,
        )

        # -------------------------------------------------
        # Call LLM
        # -------------------------------------------------

        model_out = call_model(
            input_text,
            TOOL_SCHEMAS
        )

        result.predicted_intent = model_out.get(
            "intent",
            ""
        )

        result.tool_called = model_out.get(
            "tool"
        ) or ""

        result.tool_params = model_out.get(
            "params",
            {}
        )

        result.raw_trace.append(model_out)

        # -------------------------------------------------
        # Execute tool
        # -------------------------------------------------

        if result.tool_called:

            registry = get_tool_registry()

            fn = registry.get(
                result.tool_called
            )

            if fn is not None:

                try:

                    result.tool_output = fn(
                        **result.tool_params
                    )

                except TypeError as e:

                    result.tool_output = {
                        "error": (
                            f"Bad params for "
                            f"{result.tool_called}: {e}"
                        )
                    }

                except Exception as e:

                    result.tool_output = {
                        "error": (
                            f"Tool execution failed "
                            f"for {result.tool_called}: {e}"
                        )
                    }

            else:

                result.tool_output = {
                    "error": (
                        f"Unknown tool: "
                        f"{result.tool_called}"
                    )
                }

        # -------------------------------------------------
        # Final answer
        # -------------------------------------------------

        if result.tool_called:

            result.final_answer = json.dumps(
                result.tool_output,
                ensure_ascii=False
            )

        else:

            result.final_answer = model_out.get(
                "model_text",
                ""
            )

        return result

    # -----------------------------------------------------
    # Run complete dataset
    # -----------------------------------------------------

    @staticmethod
    def run_dataset(
        dataset_path: str,
        call_model: Callable[[str, list], dict],
        variants: list[str] | None = None
    ) -> list[RunResult]:

        if variants is None:

            variants = [
                "english",
                "roman_urdu_natural",
                "roman_urdu_codemixed",
            ]

        # -------------------------------------------------
        # Load dataset
        # -------------------------------------------------

        with open(
            dataset_path,
            "r",
            encoding="utf-8"
        ) as f:

            tasks = json.load(f)

        results = []

        # -------------------------------------------------
        # Run every task × every language variant
        # -------------------------------------------------

        text_fields = {
            "english": "english_text",
            "roman_urdu_natural": "roman_urdu_natural",
            "roman_urdu_codemixed": "roman_urdu_codemixed",
        }

        for task in tasks:

            for variant in variants:

                text_field = text_fields[variant]

                # Skip if this variant is missing
                if not task.get(text_field):
                    continue

                result = LLM.run_task(
                    task,
                    variant,
                    call_model
                )

                results.append(result)

        return results

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    @staticmethod
    def save_results(
        results: list[RunResult],
        out_path: str
    ) -> None:

        rows = []

        for r in results:

            rows.append(
                {
                    "task_id": r.task_id,
                    "variant": r.variant,
                    "input_text": r.input_text,
                    "predicted_intent": r.predicted_intent,
                    "tool_called": r.tool_called,
                    "tool_params": r.tool_params,
                    "tool_output": r.tool_output,
                    "final_answer": r.final_answer,
                }
            )

        # Make sure output directory exists
        os.makedirs(
            os.path.dirname(out_path),
            exist_ok=True
        )

        with open(
            out_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                rows,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"Saved {len(rows)} result rows "
            f"to {out_path}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    here = os.path.dirname(
        os.path.abspath(__file__)
    )

    dataset_path = os.path.join(
        here,
        "data",
        "sample_dataset.json"
    )

    out_path = os.path.join(
        here,
        "results",
        "sample_run_results.json"
    )

    results = LLM.run_dataset(
        dataset_path,
        call_model=LLM.call_model
    )

    LLM.save_results(
        results,
        out_path
    )

    # -----------------------------------------------------
    # Print first 3 results
    # -----------------------------------------------------

    for r in results[:3]:

        print(
            f"[{r.variant}] "
            f"{r.task_id}: "
            f"intent={r.predicted_intent}, "
            f"tool={r.tool_called}"
        )
