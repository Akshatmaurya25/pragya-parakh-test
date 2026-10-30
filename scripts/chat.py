"""Chat with the master agent in the terminal.

Run: python -m scripts.chat
"""

import json

from parakh.agent.master_agent import MasterAgent


def main() -> None:
    agent = MasterAgent()
    print("Parakh master agent. Ctrl+C to quit.")
    while True:
        try:
            text = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not text:
            continue
        result = agent.handle(text)
        print(f"\n[{result.action}] {result.message}")
        print(json.dumps(result.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
