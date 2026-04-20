"""
REPL with HITL interrupt/resume loop for the multi-agent research system.
Langfuse tracing is attached per-run via CallbackHandler.
"""
import uuid
from langgraph.types import Command

from supervisor import build_supervisor, get_langfuse_handler

SESSION_ID = str(uuid.uuid4())
USER_ID = "researcher-user"


def _stream_supervisor(supervisor, messages, config):
    final_answer = None
    interrupted_payload = None

    for event in supervisor.stream(
        {"messages": messages},
        config=config,
        stream_mode="values",
    ):
        msgs = event.get("messages", [])
        if msgs:
            last = msgs[-1]
            role = getattr(last, "type", "")
            if role == "ai" and hasattr(last, "content") and last.content:
                final_answer = last.content

    state = supervisor.get_state(config)
    if state.next:
        interrupted_payload = None
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupted_payload = task.interrupts[0].value
                break

    return final_answer, interrupted_payload


def _resume_stream(supervisor, command, config):
    final_answer = None

    for event in supervisor.stream(command, config=config, stream_mode="values"):
        msgs = event.get("messages", [])
        if msgs:
            last = msgs[-1]
            if hasattr(last, "type") and last.type == "ai" and last.content:
                final_answer = last.content

    state = supervisor.get_state(config)
    interrupted_payload = None
    if state.next:
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupted_payload = task.interrupts[0].value
                break

    return final_answer, interrupted_payload


def handle_hitl(supervisor, payload, config):
    while payload is not None:
        filename = payload.get("filename", "report.md")
        preview = payload.get("content_preview", "")

        print("\n" + "=" * 60)
        print("⏸️  ACTION REQUIRES APPROVAL")
        print("=" * 60)
        print(f"  Tool:  save_report")
        print(f"  File:  {filename}")
        print(f"\n--- Report Preview ---")
        print(preview)
        print("----------------------")

        while True:
            try:
                action = input("\n👉 approve / edit / reject: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                action = "reject"

            if action == "approve":
                cmd = Command(resume={"action": "approve"})
                final, payload = _resume_stream(supervisor, cmd, config)
                return final

            elif action == "edit":
                try:
                    feedback = input("✏️  Your feedback: ").strip()
                except (EOFError, KeyboardInterrupt):
                    feedback = ""
                cmd = Command(resume={"action": "edit", "feedback": feedback})
                final, payload = _resume_stream(supervisor, cmd, config)
                if payload is not None:
                    break
                return final

            elif action == "reject":
                cmd = Command(resume={"action": "reject", "reason": "User rejected"})
                final, payload = _resume_stream(supervisor, cmd, config)
                return final

            else:
                print("  Please enter 'approve', 'edit', or 'reject'.")

    return None


def main():
    print("🤖 Multi-Agent Research System (type 'exit' to quit)")
    print("-" * 60)

    supervisor = build_supervisor()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        thread_id = str(uuid.uuid4())
        lf_handler = get_langfuse_handler(
            session_id=SESSION_ID,
            user_id=USER_ID,
            trace_name=f"research: {user_input[:60]}",
        )

        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [lf_handler],
        }

        try:
            messages = [{"role": "user", "content": user_input}]
            final_answer, interrupted_payload = _stream_supervisor(supervisor, messages, config)

            if interrupted_payload is not None:
                final_answer = handle_hitl(supervisor, interrupted_payload, config)

            if final_answer:
                print(f"\nAgent: {final_answer}")

            lf_handler.flush()

        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
