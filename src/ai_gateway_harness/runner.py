import time
import traceback
from dataclasses import dataclass

from ai_gateway_client import GatewayClient

from .checks import CHECKS, Check


@dataclass
class Result:
    model: str
    check: str
    capability: str
    passed: bool
    seconds: float
    error: str | None = None


def plan(
    client: GatewayClient,
    models: list[str] | None = None,
    capabilities: list[str] | None = None,
    checks: list[Check] | None = None,
) -> list[tuple[str, Check]]:
    """Pair each model with the checks its advertised capabilities cover."""
    available = {m["name"]: set(m["capabilities"]) for m in client.models()}
    names = models or list(available)
    pairs: list[tuple[str, Check]] = []
    for name in names:
        caps = available.get(name, set())
        for chk in checks or CHECKS:
            if capabilities and chk.capability not in (*capabilities, "*"):
                continue
            if chk.capability == "*" or chk.capability in caps:
                pairs.append((name, chk))
    return pairs


def run_check(client: GatewayClient, model: str, chk: Check) -> Result:
    t0 = time.perf_counter()
    try:
        chk.fn(client, model)
        return Result(model, chk.name, chk.capability, True, time.perf_counter() - t0)
    except Exception as e:
        msg = "".join(traceback.format_exception_only(type(e), e)).strip()
        return Result(model, chk.name, chk.capability, False, time.perf_counter() - t0, msg)


def run_checks(client: GatewayClient, **plan_kwargs) -> list[Result]:
    return [run_check(client, m, chk) for m, chk in plan(client, **plan_kwargs)]
