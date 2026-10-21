import argparse
import json
import os
import sys

from ai_gateway_client import GatewayClient

from .runner import run_checks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gateway-harness", description="Run contract checks.")
    p.add_argument("--url", default=os.environ.get("AI_GATEWAY_URL", "http://127.0.0.1:8000"))
    p.add_argument("--api-key", default=os.environ.get("AI_GATEWAY_API_KEY"))
    p.add_argument("--model", action="append", help="model to test (repeatable; default: all)")
    p.add_argument("--capability", action="append", help="only this endpoint (repeatable)")
    p.add_argument("--json", action="store_true", help="emit JSON results")
    a = p.parse_args(argv)

    with GatewayClient(a.url, api_key=a.api_key) as client:
        results = run_checks(client, models=a.model, capabilities=a.capability)

    if a.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
    else:
        for r in results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"{mark}  {r.model:<20} {r.check:<34} {r.seconds * 1000:7.1f} ms")
            if r.error:
                print(f"      {r.error}")
        failed = sum(not r.passed for r in results)
        print(f"\n{len(results) - failed}/{len(results)} passed")
    return 1 if any(not r.passed for r in results) or not results else 0


if __name__ == "__main__":
    sys.exit(main())
