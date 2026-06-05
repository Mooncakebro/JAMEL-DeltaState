from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import os
import random
import re
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from PIL import Image, ImageDraw

from jamel.core.env.web import Observer, get_environment, stop_envrionment
from jamel.core.env.web.utils import StepHistory
from jamel.core.reward.web.utils import (
    compute_monocart_coverage_reward_details,
)
from jamel.coverage_artifact import (
    build_coverage_artifact_fields,
    coverage_artifact_extra_fields,
)
from jamel.log import log_utils
from jamel.weak_model_labeling_utils import is_action_execution_valid

logger = log_utils.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RELEASE_SCALEWOB_ROOT = REPO_ROOT / "env" / "browser_env" / "scalewob-env"
DEFAULT_SCALEWOB_ROOT = Path(
    os.environ.get(
        "SCALEWOB_ROOT",
        str(_DEFAULT_RELEASE_SCALEWOB_ROOT),
    )
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "baseline_gui_eval"
GENERATE_REPORT_SCRIPT = REPO_ROOT / "jamel/core/env/web/javascript/generate_report.js"
DEFAULT_QWEN_TOKENIZER = "Qwen/Qwen3-235B-A22B"
DEFAULT_MODEL_CONTEXT_TOKENS = 131_072
QWEN_PLUS_CONTEXT_TOKENS = 1_000_000
TOKEN_COUNT_CHUNK_CHARS = 16_384
MODEL_SYSTEM_PROMPT = (
    "You are a browser GUI exploration agent. Return exactly "
    "<think>...</think><action>...</action>. The action must be "
    "one BrowserGym action call from the provided action space."
)

BASELINE_ACTION_SPACE = """
noop(wait_ms: float = 1000)
send_msg_to_user(text: str)
report_infeasible(reason: str)
scroll(delta_x: float, delta_y: float)
mouse_click(x: float, y: float, button: Literal['left', 'middle', 'right'] = 'left')
fill(bid: str, value: str)
select_option(bid: str, options: str | list[str])
click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
hover(bid: str)
press(bid: str, key_comb: str)
focus(bid: str)
clear(bid: str)
drag_and_drop(from_bid: str, to_bid: str)
upload_file(bid: str, file: str | list[str])
tab_close()
tab_focus(index: int)
new_tab()
go_back()
go_forward()
goto(url: str)
reset()
""".strip()

ACTION_NAMES = tuple(
    line.split("(", 1)[0].strip()
    for line in BASELINE_ACTION_SPACE.splitlines()
    if line.strip() and "(" in line
)

BID_ACTIONS = {
    "click",
    "dblclick",
    "hover",
    "focus",
    "clear",
    "fill",
    "press",
    "select_option",
    "upload_file",
}

PROMPT_TEMPLATE = """
You are an autonomous browser exploration agent.
You have {max_steps} total steps in this continuous evaluation session.
This is session step {session_step_idx} of {max_steps}.
Your goal is to maximize novel JavaScript execution coverage in the target app.

Target app: {target_app}
Start URL: {start_url}

You may call reset() whenever you want to return the browser to the initial app state.
reset() does not clear your long-term session memory and does not reset the cumulative coverage baseline.

Browser action space:
{action_space}

{memory_block}

Current observation:
{observation}

Current valid interactive element ids:
{interactive_elements}

Respond with exactly:
<think>short reason</think><action>one action</action>

The <action> content must be one single BrowserGym action call. If the action
uses a bid, use one exact id from the valid interactive element list. Never
invent bids and never combine two actions in one response.
""".strip()

MAI_UI_SYSTEM_PROMPT = (
    "You are a MAI-UI browser GUI exploration agent. Use the requested "
    "MAI-UI-style XML tool-call protocol and output exactly one tool call."
)

MOBILE_AGENT_SYSTEM_PROMPT = (
    "You are a Mobile-Agent-v3.5 browser GUI exploration agent. Use the "
    "requested Thought/Action tool-call protocol and output exactly one action."
)

MAI_UI_PROMPT_TEMPLATE = """
You are controlling a browser by looking at the screenshot and visible UI elements.
You have {max_steps} total steps in this continuous evaluation session.
This is session step {session_step_idx} of {max_steps}.
Your goal is to maximize novel JavaScript execution coverage in the target app.

Target app: {target_app}
Start URL: {start_url}

You may call reset whenever you want to return the browser to the initial app state.
reset does not clear your session memory and does not reset cumulative coverage.

Coordinates must be normalized integers from 0 to 999 relative to the screenshot:
x=0 is the left edge, x=999 is the right edge, y=0 is the top edge, y=999 is the bottom edge.
You may also target a visible element by BrowserGym bid when the bid is known.

Available tools, expressed as JSON inside one <tool_call> tag:
- click: {{"name":"click","arguments":{{"x":500,"y":420}}}} or {{"name":"click","arguments":{{"bid":"12"}}}}
- type: {{"name":"type","arguments":{{"bid":"12","text":"search text"}}}}
- press: {{"name":"press","arguments":{{"bid":"12","key":"Enter"}}}}
- scroll: {{"name":"scroll","arguments":{{"direction":"down"}}}}
- go_back: {{"name":"go_back","arguments":{{}}}}
- wait: {{"name":"wait","arguments":{{}}}}
- reset: {{"name":"reset","arguments":{{}}}}

{memory_block}

Visible elements:
{interactive_elements}

Respond with exactly:
<thinking>short reason</thinking><tool_call>{{"name":"one_tool","arguments":{{...}}}}</tool_call>

Do not output BrowserGym code directly. Do not output more than one tool call.
""".strip()

MOBILE_AGENT_PROMPT_TEMPLATE = """
You are controlling a browser with Mobile-Agent-v3.5 style actions.
You have {max_steps} total steps in this continuous evaluation session.
This is session step {session_step_idx} of {max_steps}.
Your goal is to maximize novel JavaScript execution coverage in the target app.

Target app: {target_app}
Start URL: {start_url}

The screenshot is annotated with visible element ids when possible. The same ids
are listed below as BrowserGym bids. You may call reset whenever you want to
return the browser to the initial app state. reset keeps your session memory and
cumulative coverage.

Available actions, expressed as JSON inside one <tool_call> tag after Action:
- click: {{"name":"click","arguments":{{"bid":"12"}}}} or {{"name":"click","arguments":{{"x":500,"y":420}}}}
- type: {{"name":"type","arguments":{{"bid":"12","text":"search text"}}}}
- press: {{"name":"press","arguments":{{"bid":"12","key":"Enter"}}}}
- scroll: {{"name":"scroll","arguments":{{"direction":"down"}}}}
- go_back: {{"name":"go_back","arguments":{{}}}}
- wait: {{"name":"wait","arguments":{{}}}}
- reset: {{"name":"reset","arguments":{{}}}}

{memory_block}

Visible elements:
{interactive_elements}

Respond with exactly:
Thought: short reason
Action: <tool_call>{{"name":"one_tool","arguments":{{...}}}}</tool_call>

Do not output BrowserGym code directly. Do not output more than one action.
""".strip()


@dataclass
class BaselineAgentDecision:
    response: str
    think: str
    action: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuiElement:
    bid: str
    role: str
    label: str
    line: str
    bbox: tuple[float, float, float, float] | None = None
    visibility: float | None = None
    clickable: bool | None = None
    set_of_marks: bool | None = None

    @property
    def center(self) -> tuple[float, float] | None:
        if self.bbox is None:
            return None
        x, y, width, height = self.bbox
        return x + width / 2, y + height / 2


@dataclass
class NativeGuiAction:
    kind: str
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    point: tuple[float, float] | None = None
    bid: str | None = None
    label: str | None = None
    text: str | None = None
    key: str | None = None
    direction: str | None = None
    raw: str = ""
    parse_valid: bool = True
    parse_error: str | None = None


class DecisionTimeoutError(TimeoutError):
    pass


class EnvStepTimeoutError(TimeoutError):
    pass


@dataclass
class BaselineSessionState:
    run_id: str
    session_id: str
    agent_id: str
    agent_type: str
    target_app: str
    start_url: str
    max_steps: int
    seed: int
    session_step_idx: int = 1
    episode_idx: int = 0
    step_idx: int = 1
    cumulative_reward: float = 0.0
    last_coverage_score: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    output_dir: Path | None = None

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.session_step_idx + 1)


class BaselineAgentAdapter(Protocol):
    agent_id: str
    agent_type: str

    def decide(self, obs: dict[str, Any], session_state: BaselineSessionState) -> BaselineAgentDecision:
        ...

    def observe(self, step_record: StepHistory) -> None:
        ...


def decide_with_hard_timeout(
    agent: BaselineAgentAdapter,
    obs: dict[str, Any],
    session_state: BaselineSessionState,
    timeout_seconds: int,
) -> BaselineAgentDecision:
    if timeout_seconds <= 0 or threading.current_thread() is not threading.main_thread():
        return agent.decide(obs, session_state)

    def _handle_timeout(signum: int, frame: Any) -> None:
        raise DecisionTimeoutError(f"agent.decide exceeded {timeout_seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    signal.signal(signal.SIGALRM, _handle_timeout)
    try:
        return agent.decide(obs, session_state)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def env_step_with_hard_timeout(env: Any, action: str, timeout_seconds: int) -> tuple[Any, Any, Any, Any, Any]:
    if timeout_seconds <= 0 or threading.current_thread() is not threading.main_thread():
        return env.step(action)

    def _handle_timeout(signum: int, frame: Any) -> None:
        raise EnvStepTimeoutError(f"env.step exceeded {timeout_seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    signal.signal(signal.SIGALRM, _handle_timeout)
    try:
        return env.step(action)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


class LocalStaticServer:
    def __init__(
        self,
        root_dir: Path,
        host: str = "127.0.0.1",
        port: int = 8000,
        port_search: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.host = host
        self.port = port
        self.port_search = port_search
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> int:
        if self._server is not None:
            return self.port

        last_error: OSError | None = None
        ports = [self.port]
        if self.port_search:
            ports.extend(range(self.port + 1, self.port + 100))

        handler = partial(SimpleHTTPRequestHandler, directory=str(self.root_dir))
        for candidate_port in ports:
            try:
                self._server = ThreadingHTTPServer((self.host, candidate_port), handler)
                self.port = candidate_port
                break
            except OSError as exc:
                last_error = exc

        if self._server is None:
            raise RuntimeError(f"Failed to start static server: {last_error}")

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="baseline-scalewob-static-server",
            daemon=True,
        )
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None


def parse_apps(raw_apps: str) -> list[str]:
    stripped = raw_apps.strip()
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in stripped.split(",") if item.strip()]


def parse_agents(raw_agents: str) -> list[str]:
    return [item.strip() for item in raw_agents.split(",") if item.strip()]


def start_url_for_app(host: str, port: int, app: str) -> str:
    return f"http://{host}:{port}/{app}/index.html"


def httpx_trust_env_for_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return False
    if host.startswith("127."):
        return False
    return True


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "id"


def extract_interactive_elements(observation: str) -> list[dict[str, str]]:
    elements: list[dict[str, str]] = []
    pattern = re.compile(r"\[(\d+)\]\s+([A-Za-z]+)\s+'([^']*)'")
    for raw_line in observation.splitlines():
        line = raw_line.strip()
        match = pattern.search(line)
        if not match:
            continue
        bid, role, label = match.groups()
        elements.append(
            {
                "bid": bid,
                "role": role.lower(),
                "label": label.strip(),
                "line": line,
            }
        )
    return elements


def gui_elements_from_obs(obs: dict[str, Any] | None, *, observation: str | None = None) -> list[GuiElement]:
    observation_text_value = observation if observation is not None else (observation_text(obs) if isinstance(obs, dict) else "")
    text_elements = extract_interactive_elements(observation_text_value)
    element_by_bid: dict[str, GuiElement] = {
        item["bid"]: GuiElement(
            bid=item["bid"],
            role=item["role"],
            label=item["label"],
            line=item["line"],
        )
        for item in text_elements
    }

    if isinstance(obs, dict):
        extra_properties = obs.get("extra_element_properties")
        if isinstance(extra_properties, dict):
            for bid, properties in extra_properties.items():
                if not isinstance(properties, dict):
                    continue
                element = element_by_bid.get(str(bid))
                if element is None:
                    element = GuiElement(
                        bid=str(bid),
                        role="unknown",
                        label="",
                        line=f"[{bid}] unknown ''",
                    )
                    element_by_bid[str(bid)] = element
                bbox = properties.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        element.bbox = tuple(float(value) for value in bbox)  # type: ignore[assignment]
                    except (TypeError, ValueError):
                        element.bbox = None
                visibility = properties.get("visibility")
                if isinstance(visibility, (int, float)):
                    element.visibility = float(visibility)
                if "clickable" in properties:
                    element.clickable = bool(properties.get("clickable"))
                if "set_of_marks" in properties:
                    element.set_of_marks = bool(properties.get("set_of_marks"))

    return list(element_by_bid.values())


def format_gui_elements(obs: dict[str, Any] | None, *, observation: str | None = None, limit: int = 120) -> str:
    elements = gui_elements_from_obs(obs, observation=observation)
    if not elements:
        return "(none)"
    lines: list[str] = []
    for item in elements[:limit]:
        extras: list[str] = []
        if item.center is not None:
            center_x, center_y = item.center
            extras.append(f"center=({center_x:.0f},{center_y:.0f})")
        if item.set_of_marks:
            extras.append("som=true")
        if item.visibility is not None:
            extras.append(f"visibility={item.visibility:.2f}")
        suffix = f" [{' '.join(extras)}]" if extras else ""
        lines.append(f"- {item.bid}: {item.role} {item.label!r}{suffix}")
    if len(elements) > limit:
        lines.append(f"- ... {len(elements) - limit} more omitted")
    return "\n".join(lines)


def format_interactive_elements(observation: str, limit: int = 120) -> str:
    elements = extract_interactive_elements(observation)
    if not elements:
        return "(none)"
    lines = [f"- {item['bid']}: {item['role']} {item['label']!r}" for item in elements[:limit]]
    if len(elements) > limit:
        lines.append(f"- ... {len(elements) - limit} more omitted")
    return "\n".join(lines)


def screenshot_dimensions(obs: dict[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(obs, dict):
        return None
    screenshot = obs.get("screenshot")
    shape = getattr(screenshot, "shape", None)
    if not shape or len(shape) < 2:
        return None
    return int(shape[1]), int(shape[0])


def normalized_point_to_pixels(point: tuple[float, float], obs: dict[str, Any] | None) -> tuple[float, float]:
    x, y = point
    dimensions = screenshot_dimensions(obs)
    if dimensions is None:
        return x, y
    width, height = dimensions
    if 0 <= x <= 1 and 0 <= y <= 1:
        return x * width, y * height
    if 0 <= x <= 999 and 0 <= y <= 999:
        return x / 999.0 * width, y / 999.0 * height
    return x, y


def element_at_point(
    point: tuple[float, float],
    obs: dict[str, Any] | None,
    *,
    preferred_roles: set[str] | None = None,
) -> GuiElement | None:
    px, py = normalized_point_to_pixels(point, obs)
    candidates: list[tuple[float, GuiElement]] = []
    for element in gui_elements_from_obs(obs):
        if element.bbox is None:
            continue
        x, y, width, height = element.bbox
        if x <= px <= x + width and y <= py <= y + height:
            role_bonus = -10_000.0 if preferred_roles and element.role in preferred_roles else 0.0
            area = max(1.0, width * height)
            candidates.append((role_bonus + area, element))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def find_element_for_native_action(native: NativeGuiAction, obs: dict[str, Any] | None) -> GuiElement | None:
    elements = gui_elements_from_obs(obs)
    if native.bid:
        for element in elements:
            if element.bid == str(native.bid) and element.role != "unknown":
                return element
    if native.label:
        normalized_label = _normalize_label(native.label)
        for element in elements:
            if _normalize_label(element.label) == normalized_label:
                return element
        for element in elements:
            if normalized_label and normalized_label in _normalize_label(element.label):
                return element
    if native.point is not None:
        preferred_roles = {"textbox", "searchbox", "combobox"} if native.kind in {"type", "press"} else None
        return element_at_point(native.point, obs, preferred_roles=preferred_roles)
    return None


def extract_action_response(response: str) -> tuple[str, str, bool]:
    think_match = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
    action_matches = re.findall(r"<action>(.*?)</action>", response, flags=re.DOTALL)
    if len(action_matches) != 1:
        return (
            think_match.group(1).strip() if think_match else "",
            action_matches[0].strip() if action_matches else "",
            False,
        )
    think = think_match.group(1).strip() if think_match else ""
    action = action_matches[0].strip()
    return think, action, bool(action)


def parse_action_call(action: str) -> ast.Call | None:
    try:
        parsed = ast.parse(action.strip(), mode="eval")
    except SyntaxError:
        return None
    if not isinstance(parsed.body, ast.Call):
        return None
    if not isinstance(parsed.body.func, ast.Name):
        return None
    if parsed.body.func.id not in ACTION_NAMES:
        return None
    return parsed.body


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _keyword_str(call: ast.Call, name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return _literal_str(keyword.value)
    return None


def action_referenced_bids(action: str) -> tuple[list[str], str | None]:
    call = parse_action_call(action)
    if call is None:
        return [], "action is not a single supported BrowserGym call"
    if not isinstance(call.func, ast.Name):
        return [], "action function is invalid"
    name = call.func.id

    def positional(index: int) -> str | None:
        if len(call.args) <= index:
            return None
        return _literal_str(call.args[index])

    if name in BID_ACTIONS:
        bid = positional(0) or _keyword_str(call, "bid")
        if not bid:
            return [], f"{name} action is missing a string bid"
        return [bid], None
    if name == "drag_and_drop":
        from_bid = positional(0) or _keyword_str(call, "from_bid")
        to_bid = positional(1) or _keyword_str(call, "to_bid")
        missing = [label for label, bid in (("from_bid", from_bid), ("to_bid", to_bid)) if not bid]
        if missing:
            return [], f"drag_and_drop action is missing {', '.join(missing)}"
        return [str(from_bid), str(to_bid)], None
    return [], None


def validate_action(action: str, observation: str) -> tuple[bool, str | None]:
    referenced_bids, parse_error = action_referenced_bids(action)
    if parse_error:
        return False, parse_error
    valid_bids = {item["bid"] for item in extract_interactive_elements(observation)}
    invalid_bids = [bid for bid in referenced_bids if bid not in valid_bids]
    if invalid_bids:
        return False, f"action references missing bid(s): {', '.join(invalid_bids)}"
    return True, None


def replace_known_hanging_action(
    *,
    target_app: str,
    session_step_idx: int,
    action: str,
) -> str | None:
    if target_app != "airbnb":
        return None
    normalized = " ".join(action.strip().split())
    if normalized in {"click(bid='33')", 'click(bid="33")'}:
        return "noop(wait_ms=1000)"
    if session_step_idx >= 25 and normalized in {
        "click(bid='1154')",
        'click(bid="1154")',
        "click(bid='37')",
        'click(bid="37")',
    }:
        return "noop(wait_ms=1000)"
    return None


def quote_action_arg(value: str) -> str:
    return repr(str(value))


def encode_screenshot(obs: dict[str, Any] | None) -> bytes | None:
    if not isinstance(obs, dict):
        return None
    screenshot = obs.get("screenshot")
    if screenshot is None:
        return None
    image = Image.fromarray(screenshot.astype("uint8"))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def save_screenshot_file(obs: dict[str, Any] | None, path: Path) -> str | None:
    data = encode_screenshot(obs)
    if data is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def screenshot_data_url(obs: dict[str, Any] | None) -> str | None:
    if not isinstance(obs, dict):
        return None
    screenshot = obs.get("screenshot")
    if screenshot is None:
        return None
    image = Image.fromarray(screenshot.astype("uint8")).convert("RGB")
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def observation_text(obs: dict[str, Any]) -> str:
    try:
        return Observer.get_observation(obs)
    except Exception as exc:
        logger.warning("Failed to build observation text", error=str(exc))
        return str(obs)


def build_prompt(
    *,
    obs: dict[str, Any],
    session_state: BaselineSessionState,
    memory_block: str,
) -> str:
    current_observation = observation_text(obs)
    return PROMPT_TEMPLATE.format(
        max_steps=session_state.max_steps,
        session_step_idx=session_state.session_step_idx,
        target_app=session_state.target_app,
        start_url=session_state.start_url,
        action_space=BASELINE_ACTION_SPACE,
        memory_block=memory_block,
        observation=current_observation,
        interactive_elements=format_interactive_elements(current_observation),
    )


def build_external_gui_prompt(
    *,
    obs: dict[str, Any],
    session_state: BaselineSessionState,
    memory_block: str,
    template: str,
) -> str:
    current_observation = observation_text(obs)
    return template.format(
        max_steps=session_state.max_steps,
        session_step_idx=session_state.session_step_idx,
        target_app=session_state.target_app,
        start_url=session_state.start_url,
        memory_block=memory_block,
        interactive_elements=format_gui_elements(obs, observation=current_observation),
    )


class TokenCounter(Protocol):
    name: str
    model_max_length: int

    def count_text(self, text: str) -> int:
        ...

    def count_chat(self, *, system_text: str, user_text: str) -> int:
        ...


class HuggingFaceTokenCounter:
    def __init__(self, tokenizer_name: str) -> None:
        from transformers import AutoTokenizer

        self.name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        raw_max_length = int(getattr(self.tokenizer, "model_max_length", 0) or 0)
        if raw_max_length <= 0 or raw_max_length > 10_000_000:
            raw_max_length = DEFAULT_MODEL_CONTEXT_TOKENS
        self.model_max_length = raw_max_length

    def count_text(self, text: str) -> int:
        if len(text) > TOKEN_COUNT_CHUNK_CHARS:
            return sum(
                len(self.tokenizer.encode(text[index : index + TOKEN_COUNT_CHUNK_CHARS], add_special_tokens=False))
                for index in range(0, len(text), TOKEN_COUNT_CHUNK_CHARS)
            )
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_text_up_to(self, text: str, limit: int) -> int:
        if limit < 0:
            return 0
        total = 0
        for index in range(0, len(text), TOKEN_COUNT_CHUNK_CHARS):
            chunk = text[index : index + TOKEN_COUNT_CHUNK_CHARS]
            total += len(self.tokenizer.encode(chunk, add_special_tokens=False))
            if total > limit:
                return limit + 1
        return total

    def count_chat(self, *, system_text: str, user_text: str) -> int:
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return self.count_text(str(rendered))
            except Exception as exc:
                logger.warning("Tokenizer chat template failed; using role-tag fallback", error=str(exc))
        rendered = f"<|system|>\n{system_text}\n<|user|>\n{user_text}\n<|assistant|>\n"
        return self.count_text(rendered)

    def count_chat_up_to(self, *, system_text: str, user_text: str, limit: int) -> int:
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return self.count_text_up_to(str(rendered), limit)
            except Exception as exc:
                logger.warning("Tokenizer chat template failed; using role-tag fallback", error=str(exc))
        rendered = f"<|system|>\n{system_text}\n<|user|>\n{user_text}\n<|assistant|>\n"
        return self.count_text_up_to(rendered, limit)


class WhitespaceTokenCounter:
    def __init__(self, model_max_length: int = DEFAULT_MODEL_CONTEXT_TOKENS) -> None:
        self.name = "whitespace-test-tokenizer"
        self.model_max_length = model_max_length

    def count_text(self, text: str) -> int:
        return max(1, len(text.split()))

    def count_text_up_to(self, text: str, limit: int) -> int:
        if limit < 0:
            return 0
        count = 0
        for _ in text.split():
            count += 1
            if count > limit:
                return limit + 1
        return max(1, count)

    def count_chat(self, *, system_text: str, user_text: str) -> int:
        return self.count_text(f"<system> {system_text} <user> {user_text} <assistant>")

    def count_chat_up_to(self, *, system_text: str, user_text: str, limit: int) -> int:
        return self.count_text_up_to(
            f"<system> {system_text} <user> {user_text} <assistant>",
            limit,
        )


_TOKEN_COUNTER_CACHE: dict[str, TokenCounter] = {}


def infer_tokenizer_name(model_name: str, explicit_name: str | None = None) -> str:
    if explicit_name:
        return explicit_name
    if "qwen" in model_name.lower():
        return DEFAULT_QWEN_TOKENIZER
    return DEFAULT_QWEN_TOKENIZER


def infer_model_context_tokens(model_name: str, tokenizer_context_tokens: int) -> int:
    normalized = model_name.lower()
    if "qwen3.6-plus" in normalized or "qwen3.5-plus" in normalized:
        return QWEN_PLUS_CONTEXT_TOKENS
    if tokenizer_context_tokens > 0:
        return tokenizer_context_tokens
    return DEFAULT_MODEL_CONTEXT_TOKENS


def get_token_counter(tokenizer_name: str) -> TokenCounter:
    if tokenizer_name not in _TOKEN_COUNTER_CACHE:
        _TOKEN_COUNTER_CACHE[tokenizer_name] = HuggingFaceTokenCounter(tokenizer_name)
    return _TOKEN_COUNTER_CACHE[tokenizer_name]


def _first_present(mapping: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in mapping and mapping[name] not in (None, ""):
            return mapping[name]
    return None


def _parse_jsonish_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    repaired_coordinate = re.sub(
        r'("x"\s*:\s*)(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(\s*[}\]])',
        r'\1[\2,\3]\4',
        stripped,
    )
    if repaired_coordinate != stripped:
        candidates.append(repaired_coordinate)
    open_braces = stripped.count("{") - stripped.count("}")
    if open_braces > 0:
        candidates.append(stripped + ("}" * open_braces))
        if repaired_coordinate != stripped:
            candidates.append(repaired_coordinate + ("}" * open_braces))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(candidate)
            return parsed if isinstance(parsed, dict) else None
        except (ValueError, SyntaxError):
            pass
    return None


def _extract_jsonish_from_tag(response: str, tag: str = "tool_call") -> tuple[dict[str, Any] | None, str | None]:
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", flags=re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(response)
    if len(matches) != 1:
        return None, f"expected exactly one <{tag}> block, found {len(matches)}"
    parsed = _parse_jsonish_object(matches[0])
    if parsed is None:
        return None, f"<{tag}> block is not a JSON object"
    return parsed, None


def _point_from_args(args: dict[str, Any]) -> tuple[float, float] | None:
    x = _first_present(args, ("x", "X", "coordinate_x", "point_x"))
    y = _first_present(args, ("y", "Y", "coordinate_y", "point_y"))
    if y is None and isinstance(x, (list, tuple)) and len(x) >= 2:
        x, y = x[0], x[1]
    elif y is None and isinstance(x, dict):
        y = _first_present(x, ("y", "Y"))
        x = _first_present(x, ("x", "X"))
    if x is None or y is None:
        point = _first_present(args, ("point", "coordinate", "coordinates", "position"))
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
        elif isinstance(point, dict):
            x = _first_present(point, ("x", "X"))
            y = _first_present(point, ("y", "Y"))
        elif isinstance(point, str):
            numbers = re.findall(r"-?\d+(?:\.\d+)?", point)
            if len(numbers) >= 2:
                x, y = numbers[0], numbers[1]
    if x is None or y is None:
        return None
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


def _direction_from_text(text: str) -> str | None:
    normalized = text.lower()
    if any(word in normalized for word in ("down", "向下", "上滑")):
        return "down"
    if any(word in normalized for word in ("up", "向上", "下滑")):
        return "up"
    if any(word in normalized for word in ("left", "向左", "右滑")):
        return "left"
    if any(word in normalized for word in ("right", "向右", "左滑")):
        return "right"
    return None


def _tool_call_to_native(tool: dict[str, Any], *, raw: str) -> NativeGuiAction:
    name = str(_first_present(tool, ("name", "tool", "action", "type", "function")) or "").strip()
    args = _first_present(tool, ("arguments", "args", "parameters", "parameter"))
    if isinstance(args, str):
        args = _parse_jsonish_object(args) or {"text": args}
    if not isinstance(args, dict):
        args = {}
    normalized_name = name.lower().replace("-", "_").replace(" ", "_")
    if normalized_name in {"one_tool", "tool_call", "call_tool"} and args:
        inner_name = _first_present(args, ("name", "tool", "action", "type", "function"))
        inner_args = _first_present(args, ("arguments", "args", "parameters", "parameter"))
        if inner_name is not None:
            name = str(inner_name).strip()
            if isinstance(inner_args, str):
                inner_args = _parse_jsonish_object(inner_args) or {"text": inner_args}
            args = dict(inner_args) if isinstance(inner_args, dict) else {}
            normalized_name = name.lower().replace("-", "_").replace(" ", "_")
    if normalized_name in {"tap", "click", "left_click", "double_click", "hover"}:
        kind = "click"
    elif normalized_name in {"input", "type", "text", "enter_text", "set_text"}:
        kind = "type"
    elif normalized_name in {"keyboard", "press", "hotkey"}:
        kind = "press"
    elif normalized_name in {"scroll", "swipe"}:
        kind = "scroll"
    elif normalized_name in {"back", "go_back"}:
        kind = "go_back"
    elif normalized_name in {"wait", "noop", "do_nothing"}:
        kind = "wait"
    elif normalized_name in {"reset", "restart"}:
        kind = "reset"
    else:
        kind = normalized_name or "unknown"

    bid = _first_present(args, ("bid", "element_id", "element", "mark", "target_id"))
    label = _first_present(args, ("label", "name", "target", "element_text"))
    point = _point_from_args(args)
    text_value = _first_present(args, ("text", "value", "content", "input_text", "query"))
    key_value = _first_present(args, ("key", "keys", "key_comb", "hotkey"))
    direction_value = _first_present(args, ("direction", "dir"))
    if direction_value is None:
        direction_value = _direction_from_text(json.dumps(args, ensure_ascii=False))

    return NativeGuiAction(
        kind=kind,
        name=name,
        args=dict(args),
        point=point,
        bid=str(bid) if bid is not None else None,
        label=str(label) if label is not None else None,
        text=str(text_value) if text_value is not None else None,
        key=str(key_value) if key_value is not None else None,
        direction=str(direction_value).lower() if direction_value else None,
        raw=raw,
    )


def parse_mai_ui_native_action(response: str) -> tuple[str, NativeGuiAction, bool]:
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", response, flags=re.DOTALL | re.IGNORECASE)
    think = thinking_match.group(1).strip() if thinking_match else ""
    tool, error = _extract_jsonish_from_tag(response, "tool_call")
    if error:
        return think, NativeGuiAction(kind="unknown", raw=response, parse_valid=False, parse_error=error), False
    native = _tool_call_to_native(tool or {}, raw=response)
    return think, native, True


def parse_mobile_agent_native_action(response: str) -> tuple[str, NativeGuiAction, bool]:
    thought_match = re.search(r"(?:Thought|思考)\s*:\s*(.*?)(?:\n\s*(?:Action|动作)\s*:|$)", response, flags=re.DOTALL | re.IGNORECASE)
    think = thought_match.group(1).strip() if thought_match else ""
    tool, error = _extract_jsonish_from_tag(response, "tool_call")
    if error is None:
        native = _tool_call_to_native(tool or {}, raw=response)
        return think, native, True

    action_match = re.search(r"(?:Action|动作)\s*:\s*(.*)", response, flags=re.DOTALL | re.IGNORECASE)
    action_text = action_match.group(1).strip() if action_match else response.strip()
    lowered = action_text.lower()
    if "reset" in lowered:
        return think, NativeGuiAction(kind="reset", raw=response), True
    if "wait" in lowered or "noop" in lowered:
        return think, NativeGuiAction(kind="wait", raw=response), True
    if "back" in lowered:
        return think, NativeGuiAction(kind="go_back", raw=response), True
    direction = _direction_from_text(action_text)
    if direction:
        return think, NativeGuiAction(kind="scroll", direction=direction, raw=response), True
    numbers = re.findall(r"-?\d+(?:\.\d+)?", action_text)
    if "click" in lowered and len(numbers) >= 1:
        if len(numbers) >= 2:
            return think, NativeGuiAction(kind="click", point=(float(numbers[0]), float(numbers[1])), raw=response), True
        return think, NativeGuiAction(kind="click", bid=numbers[0], raw=response), True
    if any(word in lowered for word in ("type", "input", "text")):
        bid = numbers[0] if numbers else None
        text_match = re.search(r"['\"]([^'\"]+)['\"]", action_text)
        text_value = text_match.group(1) if text_match else ""
        return think, NativeGuiAction(kind="type", bid=bid, text=text_value, raw=response), True
    return think, NativeGuiAction(kind="unknown", raw=response, parse_valid=False, parse_error=error), False


def native_gui_action_to_browsergym(native: NativeGuiAction, obs: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "native_action_kind": native.kind,
        "native_action_name": native.name,
        "native_action_parse_valid": native.parse_valid,
    }
    if native.parse_error:
        metadata["native_action_parse_error"] = native.parse_error
    if native.args:
        metadata["native_action_args"] = json.dumps(native.args, ensure_ascii=False)

    kind = native.kind.lower()
    if kind == "reset":
        return "reset()", {**metadata, "native_action_conversion": "reset"}
    if kind == "wait":
        return "noop(wait_ms=1000)", {**metadata, "native_action_conversion": "wait_to_noop"}
    if kind == "go_back":
        return "go_back()", {**metadata, "native_action_conversion": "go_back"}
    if kind == "scroll":
        direction = native.direction or "down"
        scroll_map = {
            "down": "scroll(0, 700)",
            "up": "scroll(0, -700)",
            "right": "scroll(700, 0)",
            "left": "scroll(-700, 0)",
        }
        return scroll_map.get(direction, "scroll(0, 700)"), {
            **metadata,
            "native_action_conversion": "scroll",
            "native_action_direction": direction,
        }

    if native.point is not None and kind == "click":
        px, py = normalized_point_to_pixels(native.point, obs)
        metadata.update(
            {
                "native_action_conversion": "coordinate",
                "native_action_point_x": round(px, 3),
                "native_action_point_y": round(py, 3),
            }
        )
        return f"mouse_click({px:.3f}, {py:.3f})", metadata

    element = find_element_for_native_action(native, obs)
    if element is not None:
        metadata.update(
            {
                "native_action_matched_bid": element.bid,
                "native_action_matched_role": element.role,
                "native_action_matched_label": element.label,
                "native_action_conversion": "element",
            }
        )
        bid = quote_action_arg(element.bid)
        if kind == "type":
            return f"fill({bid}, {quote_action_arg(native.text or '')})", metadata
        if kind == "press":
            return f"press({bid}, {quote_action_arg(native.key or 'Enter')})", metadata
        return f"click({bid})", metadata

    metadata["native_action_conversion"] = "unknown_to_noop"
    return "noop(wait_ms=1000)", metadata


def _normalize_label(label: str) -> str:
    compact = " ".join(label.lower().split())
    compact = re.sub(r"\d+", "#", compact)
    return compact[:80]


def _state_signature(observation: str) -> str:
    lines: list[str] = []
    for raw_line in observation.splitlines():
        line = raw_line.strip()
        if any(keyword in line.lower() for keyword in ("button", "link", "tab", "search", "textbox", "menu")):
            lines.append(re.sub(r"\[\d+\]", "[]", line))
    return "\n".join(lines[:80])


class HeuristicExplorer:
    def __init__(self, target_app: str, start_url: str, seed: int) -> None:
        self.target_app = target_app
        self.start_url = start_url
        self.rng = random.Random(seed)
        self.cursor = 0
        self.fill_cursor = 0
        self.no_reward_streak = 0
        self.pending_text_bid: str | None = None
        self.tried_state_actions: set[tuple[str, str]] = set()
        self.tried_action_signatures: set[str] = set()
        self.rewarded_action_signatures: set[str] = set()
        self.failed_action_signatures: set[str] = set()

    def _fill_values(self) -> list[str]:
        target = self.target_app.lower()
        if target in {"agoda", "airbnb", "trip", "huazhu"}:
            return ["Tokyo", "Paris", "2 guests", "2026-05-01"]
        if target in {"douban", "bilibili", "weibo", "ximalaya", "qqmusic"}:
            return ["movie", "travel", "music", "OpenAI"]
        if target == "wikipedia":
            return ["Artificial intelligence", "Travel", "China", "Machine learning"]
        return ["OpenAI", "Travel", "News", "Music"]

    def _action_signature(self, action: str, observation: str) -> str:
        elements = {item["bid"]: item for item in extract_interactive_elements(observation)}
        bid_match = re.search(r"['\"]?(\d+)['\"]?", action)
        bid = bid_match.group(1) if bid_match else ""
        element = elements.get(bid, {})
        action_type = action.split("(", 1)[0]
        role = element.get("role", "")
        label = _normalize_label(element.get("label", ""))
        if action_type in {"click", "hover", "fill", "press", "dblclick"}:
            return f"{action_type}:{role}:{label}"
        return action_type

    def _element_score(self, element: dict[str, str]) -> float:
        role = element["role"]
        label = element["label"].lower()
        semantic_click = f"click:{role}:{_normalize_label(element['label'])}"
        score = 0.0
        if semantic_click not in self.tried_action_signatures:
            score += 10.0
        if semantic_click in self.rewarded_action_signatures:
            score -= 15.0
        if semantic_click in self.failed_action_signatures:
            score -= 6.0
        score += {
            "button": 10.0,
            "link": 9.0,
            "textbox": 8.0,
            "searchbox": 8.0,
            "combobox": 7.0,
            "tab": 7.0,
            "menuitem": 6.0,
            "heading": 2.0,
            "generic": -4.0,
            "image": -4.0,
        }.get(role, 1.0)
        for keyword, bonus in {
            "search": 6.0,
            "next": 5.0,
            "discover": 4.0,
            "profile": 4.0,
            "setting": 4.0,
            "message": 3.0,
            "home": -12.0,
            "logo": -12.0,
        }.items():
            if keyword in label:
                score += bonus
        return score

    def choose_action(self, observation: str, session_state: BaselineSessionState) -> str:
        if self.pending_text_bid:
            bid = self.pending_text_bid
            self.pending_text_bid = None
            return f"press({quote_action_arg(bid)}, 'Enter')"

        if self.no_reward_streak >= 8 and session_state.remaining_steps > 3:
            self.no_reward_streak = 0
            return "reset()"

        elements = extract_interactive_elements(observation)
        state_signature = _state_signature(observation)
        candidates: list[tuple[float, str]] = []
        fill_values = self._fill_values()
        for element in elements:
            bid = element["bid"]
            role = element["role"]
            base = self._element_score(element)
            bid_arg = quote_action_arg(bid)
            if role in {"button", "link", "tab", "menuitem", "heading", "generic", "image"}:
                candidates.append((base + 1.0, f"click({bid_arg})"))
                candidates.append((base - 2.0, f"hover({bid_arg})"))
            if role in {"textbox", "searchbox", "combobox"}:
                value = fill_values[self.fill_cursor % len(fill_values)]
                candidates.append((base + 9.0, f"fill({bid_arg}, {quote_action_arg(value)})"))
                candidates.append((base + 3.0, f"click({bid_arg})"))

        candidates.extend([(4.0, "scroll(0, 700)"), (2.0, "scroll(0, -500)")])
        if self.no_reward_streak >= 4:
            candidates.extend([(7.0, "go_back()"), (6.0, f"goto({quote_action_arg(self.start_url)})")])

        filtered: list[tuple[float, str]] = []
        last_action = session_state.history[-1]["action"] if session_state.history else ""
        for score, action in candidates:
            signature = self._action_signature(action, observation)
            if (state_signature, signature) in self.tried_state_actions:
                continue
            if action == last_action:
                continue
            filtered.append((score, action))

        if not filtered:
            fallback = [
                "scroll(0, 800)",
                "scroll(0, -600)",
                "go_back()",
                f"goto({quote_action_arg(self.start_url)})",
                "noop(300)",
            ]
            action = fallback[self.cursor % len(fallback)]
        else:
            max_score = max(score for score, _ in filtered)
            top = [action for score, action in filtered if score >= max_score - 2.0]
            action = top[self.cursor % len(top)]

        self.cursor += 1
        if action.startswith("fill("):
            self.fill_cursor += 1
        return action

    def observe(self, step_record: StepHistory) -> None:
        action = str((step_record.extra_fields or {}).get("action") or "")
        before_observation = step_record.before_observation or ""
        signature = self._action_signature(action, before_observation)
        self.tried_action_signatures.add(signature)
        self.tried_state_actions.add((_state_signature(before_observation), signature))
        if action.startswith("fill("):
            bids, _ = action_referenced_bids(action)
            if bids:
                self.pending_text_bid = bids[0]
        if float(step_record.reward or 0.0) > 0:
            self.rewarded_action_signatures.add(signature)
            self.no_reward_streak = 0
        else:
            self.failed_action_signatures.add(signature)
            self.no_reward_streak += 1


class OpenAICompatibleReActAdapter:
    def __init__(
        self,
        *,
        agent_id: str,
        agent_type: str,
        target_app: str,
        start_url: str,
        seed: int,
        model_config: dict[str, Any],
        policy: str,
        use_vision: bool = False,
        history_window: int = 0,
        history_mode: str = "full",
        history_budget_mode: str = "char",
        history_char_budget: int = 120_000,
        history_observation_char_budget: int = 0,
        tokenizer_name: str | None = None,
        model_context_tokens: int = 0,
        context_margin_tokens: int = 2_048,
        token_counter: TokenCounter | None = None,
        memory_mode: str = "none",
    ) -> None:
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.target_app = target_app
        self.start_url = start_url
        self.use_vision = use_vision
        self.history_window = history_window
        self.history_mode = history_mode
        self.history_budget_mode = history_budget_mode
        self.history_char_budget = history_char_budget
        self.history_observation_char_budget = history_observation_char_budget
        self.tokenizer_name = tokenizer_name or infer_tokenizer_name(str(model_config.get("model") or ""))
        self.model_context_tokens = model_context_tokens
        self.context_margin_tokens = context_margin_tokens
        self._token_counter = token_counter
        self.memory_mode = memory_mode
        self.model_config = model_config
        self.policy = policy
        self.heuristic = HeuristicExplorer(target_app, start_url, seed)
        self.history: list[dict[str, Any]] = []
        self._history_record_token_cache: dict[int, int] = {}
        self.memory_summary = ""
        self.last_history_render_stats: dict[str, Any] = {}

    def _memory_block(self, session_state: BaselineSessionState) -> str:
        if self.history_mode == "full" and self.memory_mode == "none":
            return self._full_history_block_char_budget()

        lines: list[str] = []
        if self.history_window > 0 and self.history:
            recent = [self._history_record_for_prompt(item) for item in self.history[-self.history_window :]]
            lines.append("Recent session history:")
            lines.append(json.dumps(recent, ensure_ascii=False, indent=2))
        if self.memory_mode == "summary" and self.memory_summary:
            lines.append("Long-term session memory summary:")
            lines.append(self.memory_summary)
        if self.memory_mode == "compressed" and self.history:
            def compressed_item(item: dict[str, Any]) -> dict[str, Any]:
                return {
                    "session_step_idx": item.get("session_step_idx"),
                    "episode_idx": item.get("episode_idx"),
                    "step_idx": item.get("step_idx"),
                    "action": item.get("action"),
                    "reward": item.get("reward"),
                    "coverage_delta_score": item.get("coverage_delta_score"),
                    "action_execution_valid": item.get("action_execution_valid"),
                    "coverage_skip_reason": item.get("coverage_skip_reason"),
                }

            positive = [
                compressed_item(item)
                for item in self.history
                if float(item.get("reward", 0.0) or 0.0) > 0
            ]
            failed = [
                compressed_item(item)
                for item in self.history
                if not item.get("action_execution_valid", True)
            ]
            compressed = {
                "positive_actions": positive[-12:],
                "invalid_or_failed_actions": failed[-12:],
                "last_coverage_score": session_state.last_coverage_score,
                "cumulative_reward": session_state.cumulative_reward,
            }
            lines.append("Compressed session memory:")
            lines.append(json.dumps(compressed, ensure_ascii=False, indent=2))
        if not lines:
            return "Session memory: no previous steps in this session."
        return "\n".join(lines)

    def _history_record_for_prompt(self, item: dict[str, Any]) -> dict[str, Any]:
        record = {
            "session_step_idx": item.get("session_step_idx"),
            "episode_idx": item.get("episode_idx"),
            "step_idx": item.get("step_idx"),
            "think": item.get("think"),
            "action": item.get("action"),
            "reward": item.get("reward"),
            "coverage_delta_score": item.get("coverage_delta_score"),
            "action_format_valid": item.get("action_format_valid"),
            "action_validation_error": item.get("action_validation_error"),
            "action_execution_valid": item.get("action_execution_valid"),
            "coverage_skip_reason": item.get("coverage_skip_reason"),
            "after_last_action_error": item.get("after_last_action_error"),
        }
        observation = str(item.get("after_observation") or "")
        if self.history_observation_char_budget > 0 and len(observation) > self.history_observation_char_budget:
            record["after_observation"] = observation[: self.history_observation_char_budget]
            record["after_observation_truncated_chars"] = len(observation) - self.history_observation_char_budget
        else:
            record["after_observation"] = observation
            record["after_observation_truncated_chars"] = 0
        return record

    def _render_full_history_records(self, records: list[dict[str, Any]], omitted: int) -> str:
        block = (
            "Full session history so far, ordered oldest to newest. "
            "Preserve this trajectory when choosing the next action. "
            "Only oldest records are omitted if the configured context budget is exceeded.\n"
            f"History records included: {len(records)} / {len(self.history)}.\n"
        )
        if omitted:
            block += f"Oldest records omitted due to context budget: {omitted}.\n"
        block += json.dumps(records, ensure_ascii=False, indent=2)
        return block

    def _full_history_block_char_budget(self) -> str:
        if not self.history:
            self.last_history_render_stats = {
                "history_mode": "full",
                "history_budget_mode": "char",
                "history_total_records": 0,
                "history_included_records": 0,
                "history_omitted_records": 0,
                "history_char_budget": self.history_char_budget,
                "history_render_chars": 0,
            }
            return "Session history: no previous steps in this session."

        records = [self._history_record_for_prompt(item) for item in self.history]
        omitted = 0
        while records:
            block = self._render_full_history_records(records, omitted)
            if self.history_char_budget <= 0 or len(block) <= self.history_char_budget:
                self.last_history_render_stats = {
                    "history_mode": "full",
                    "history_budget_mode": "char",
                    "history_total_records": len(self.history),
                    "history_included_records": len(records),
                    "history_omitted_records": omitted,
                    "history_char_budget": self.history_char_budget,
                    "history_render_chars": len(block),
                }
                return block
            records = records[1:]
            omitted += 1

        self.last_history_render_stats = {
            "history_mode": "full",
            "history_budget_mode": "char",
            "history_total_records": len(self.history),
            "history_included_records": 0,
            "history_omitted_records": len(self.history),
            "history_char_budget": self.history_char_budget,
            "history_render_chars": 0,
        }
        return (
            "Session history exists but all records were omitted because the "
            "configured context budget is too small."
        )

    def _get_token_counter(self) -> TokenCounter:
        if self._token_counter is None:
            self._token_counter = get_token_counter(self.tokenizer_name)
        return self._token_counter

    def _effective_context_tokens(self, counter: TokenCounter) -> int:
        if self.model_context_tokens > 0:
            return self.model_context_tokens
        return infer_model_context_tokens(str(self.model_config.get("model") or ""), counter.model_max_length)

    def _input_token_budget(self, counter: TokenCounter) -> tuple[int, int]:
        context_tokens = self._effective_context_tokens(counter)
        max_output_tokens = int(self.model_config.get("max_tokens", 512) or 512)
        budget = max(0, context_tokens - max_output_tokens - max(0, self.context_margin_tokens))
        return budget, context_tokens

    def _count_prompt_tokens(self, prompt: str, counter: TokenCounter, limit: int | None = None) -> int:
        if limit is not None and hasattr(counter, "count_chat_up_to"):
            return int(
                getattr(counter, "count_chat_up_to")(
                    system_text=MODEL_SYSTEM_PROMPT,
                    user_text=prompt,
                    limit=limit,
                )
            )
        return counter.count_chat(system_text=MODEL_SYSTEM_PROMPT, user_text=prompt)

    def _history_record_token_count(self, item: dict[str, Any], counter: TokenCounter) -> int:
        session_step_idx = int(item.get("session_step_idx", len(self._history_record_token_cache) + 1) or 0)
        if session_step_idx in self._history_record_token_cache:
            return self._history_record_token_cache[session_step_idx]
        record = self._history_record_for_prompt(item)
        token_count = counter.count_text(json.dumps(record, ensure_ascii=False, indent=2))
        self._history_record_token_cache[session_step_idx] = token_count
        return token_count

    def _history_record_token_count_up_to(
        self,
        item: dict[str, Any],
        counter: TokenCounter,
        limit: int,
    ) -> int:
        session_step_idx = int(item.get("session_step_idx", len(self._history_record_token_cache) + 1) or 0)
        if session_step_idx in self._history_record_token_cache:
            return self._history_record_token_cache[session_step_idx]
        record = self._history_record_for_prompt(item)
        rendered = json.dumps(record, ensure_ascii=False, indent=2)
        if hasattr(counter, "count_text_up_to"):
            token_count = int(getattr(counter, "count_text_up_to")(rendered, limit))
        else:
            token_count = counter.count_text(rendered)
        if token_count <= limit:
            self._history_record_token_cache[session_step_idx] = token_count
        return token_count

    def _full_history_block_token_budget(
        self,
        *,
        obs: dict[str, Any],
        session_state: BaselineSessionState,
    ) -> str:
        counter = self._get_token_counter()
        input_budget, context_tokens = self._input_token_budget(counter)
        max_output_tokens = int(self.model_config.get("max_tokens", 512) or 512)

        if not self.history:
            block = "Session history: no previous steps in this session."
            prompt = build_prompt(obs=obs, session_state=session_state, memory_block=block)
            prompt_tokens = self._count_prompt_tokens(prompt, counter, input_budget)
            self.last_history_render_stats = {
                "history_mode": "full",
                "history_budget_mode": "token",
                "history_total_records": 0,
                "history_included_records": 0,
                "history_omitted_records": 0,
                "history_char_budget": self.history_char_budget,
                "history_render_chars": 0,
                "history_token_budget": input_budget,
                "history_prompt_tokens": prompt_tokens,
                "history_context_tokens": context_tokens,
                "history_context_margin_tokens": self.context_margin_tokens,
                "history_max_output_tokens": max_output_tokens,
                "history_tokenizer_name": counter.name,
                "history_over_budget": prompt_tokens > input_budget,
            }
            return block

        records = [self._history_record_for_prompt(item) for item in self.history]
        empty_history_block = self._render_full_history_records([], len(records))
        empty_prompt = build_prompt(obs=obs, session_state=session_state, memory_block=empty_history_block)
        base_tokens = self._count_prompt_tokens(empty_prompt, counter, input_budget)
        separator_tokens = 8

        selected_start = len(records)
        estimated_tokens = base_tokens
        if base_tokens > input_budget:
            best_omitted = len(records)
            best_block = self._render_full_history_records([], best_omitted)
            best_tokens = base_tokens
            self.last_history_render_stats = {
                "history_mode": "full",
                "history_budget_mode": "token",
                "history_total_records": len(self.history),
                "history_included_records": 0,
                "history_omitted_records": best_omitted,
                "history_char_budget": self.history_char_budget,
                "history_render_chars": len(best_block),
                "history_token_budget": input_budget,
                "history_prompt_tokens": best_tokens,
                "history_context_tokens": context_tokens,
                "history_context_margin_tokens": self.context_margin_tokens,
                "history_max_output_tokens": max_output_tokens,
                "history_tokenizer_name": counter.name,
                "history_prompt_tokens_estimated": True,
                "history_base_prompt_tokens": base_tokens,
                "history_over_budget": True,
            }
            return best_block

        for index in range(len(records) - 1, -1, -1):
            remaining = input_budget - estimated_tokens - separator_tokens
            if remaining < 0:
                break
            record_tokens = self._history_record_token_count_up_to(
                self.history[index],
                counter,
                remaining,
            )
            candidate_tokens = estimated_tokens + record_tokens + separator_tokens
            if candidate_tokens > input_budget:
                break
            selected_start = index
            estimated_tokens = candidate_tokens

        best_omitted = selected_start
        best_block = self._render_full_history_records(records[best_omitted:], best_omitted)
        best_tokens = estimated_tokens

        self.last_history_render_stats = {
            "history_mode": "full",
            "history_budget_mode": "token",
            "history_total_records": len(self.history),
            "history_included_records": len(records) - best_omitted,
            "history_omitted_records": best_omitted,
            "history_char_budget": self.history_char_budget,
            "history_render_chars": len(best_block),
            "history_token_budget": input_budget,
            "history_prompt_tokens": best_tokens,
            "history_context_tokens": context_tokens,
            "history_context_margin_tokens": self.context_margin_tokens,
            "history_max_output_tokens": max_output_tokens,
            "history_tokenizer_name": counter.name,
            "history_prompt_tokens_estimated": True,
            "history_base_prompt_tokens": base_tokens,
            "history_over_budget": best_tokens > input_budget,
        }
        return best_block

    def _has_model_config(self) -> bool:
        return bool(
            self.model_config.get("model")
            and self.model_config.get("base_url")
            and self.model_config.get("api_key")
        )

    def _effective_policy(self) -> str:
        if self.policy == "heuristic":
            return "heuristic"
        if self.policy == "model":
            if not self._has_model_config():
                missing = [key for key in ("model", "base_url", "api_key") if not self.model_config.get(key)]
                raise ValueError(f"Model policy requested but missing config fields: {missing}")
            return "model"
        return "model" if self._has_model_config() else "heuristic"

    def _model_response(self, prompt: str, obs: dict[str, Any]) -> str:
        content: str | list[dict[str, Any]] = prompt
        if self.use_vision:
            image_url = screenshot_data_url(obs)
            if image_url:
                content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
        payload = {
            "model": self.model_config["model"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a browser GUI exploration agent. Return exactly "
                        "<think>...</think><action>...</action>. The action must be "
                        "one BrowserGym action call from the provided action space."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": float(self.model_config.get("temperature", 0.2)),
            "max_tokens": int(self.model_config.get("max_tokens", 512)),
        }
        if self.model_config.get("reasoning_effort"):
            payload["reasoning_effort"] = self.model_config["reasoning_effort"]
        url = chat_completions_url(str(self.model_config["base_url"]))
        headers = {
            "Authorization": f"Bearer {self.model_config['api_key']}",
            "Content-Type": "application/json",
        }
        retries = int(self.model_config.get("retries", 3) or 0)
        retry_backoff = float(self.model_config.get("retry_backoff", 10.0) or 0.0)
        last_error: Exception | None = None
        with httpx.Client(
            timeout=int(self.model_config.get("timeout", 120)),
            trust_env=httpx_trust_env_for_url(url),
        ) as client:
            for attempt in range(retries + 1):
                try:
                    response = client.post(url, headers=headers, json=payload)
                    if response.is_error:
                        message = (
                            f"Baseline model request failed: status={response.status_code}, "
                            f"url={url}, body={response.text[:1000]}"
                        )
                        if response.status_code in {408, 409, 429, 500, 502, 503, 504} and attempt < retries:
                            time.sleep(retry_backoff * (attempt + 1))
                            continue
                        raise RuntimeError(message)
                    data = response.json()
                    return str(data["choices"][0]["message"]["content"])
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc
                    if attempt < retries:
                        time.sleep(retry_backoff * (attempt + 1))
                        continue
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Baseline model request failed without a response")

    def decide(self, obs: dict[str, Any], session_state: BaselineSessionState) -> BaselineAgentDecision:
        if self.history_mode == "full" and self.memory_mode == "none" and self.history_budget_mode == "token":
            memory_block = self._full_history_block_token_budget(obs=obs, session_state=session_state)
        else:
            memory_block = self._memory_block(session_state)
        prompt = build_prompt(
            obs=obs,
            session_state=session_state,
            memory_block=memory_block,
        )
        effective_policy = self._effective_policy()
        if effective_policy == "model":
            response = self._model_response(prompt, obs)
            think, action, tag_valid = extract_action_response(response)
        else:
            obs_text = observation_text(obs)
            action = self.heuristic.choose_action(obs_text, session_state)
            think = "Choose a valid exploratory browser action likely to reveal new UI state."
            response = f"<think>{think}</think><action>{action}</action>"
            tag_valid = True
        return BaselineAgentDecision(
            response=response,
            think=think,
            action=action,
            metadata={
                "prompt": prompt,
                "decision_policy": effective_policy,
                "response_tag_valid": tag_valid,
                "response_has_think": bool(think),
                "use_vision": self.use_vision,
                "history_window": self.history_window,
                "history_mode": self.history_mode,
                "history_budget_mode": self.history_budget_mode,
                **self.last_history_render_stats,
                "memory_mode": self.memory_mode,
                "model_name": self.model_config.get("model") if effective_policy == "model" else None,
            },
        )

    def observe(self, step_record: StepHistory) -> None:
        extra = step_record.extra_fields or {}
        item = {
            "session_step_idx": int(extra.get("session_step_idx", step_record.step)),
            "episode_idx": int(extra.get("episode_idx", 0) or 0),
            "step_idx": int(extra.get("step_idx", step_record.step) or 0),
            "think": extra.get("think"),
            "action": extra.get("action"),
            "reward": float(step_record.reward or 0.0),
            "coverage_delta_score": int(extra.get("coverage_delta_score", 0) or 0),
            "action_format_valid": bool(extra.get("action_format_valid", False)),
            "action_validation_error": extra.get("action_validation_error"),
            "action_execution_valid": bool(extra.get("action_execution_valid", False)),
            "coverage_skip_reason": extra.get("coverage_skip_reason"),
            "after_last_action_error": (
                step_record.after_obs.get("last_action_error")
                if isinstance(step_record.after_obs, dict)
                else None
            ),
            "after_observation": step_record.after_observation,
        }
        self.history.append(item)
        if self.memory_mode == "summary":
            summary_lines = [
                f"Step {item['session_step_idx']}: {item['action']} "
                f"reward={item['reward']} delta={item['coverage_delta_score']}"
            ]
            if self.memory_summary:
                summary_lines.insert(0, self.memory_summary)
            combined_summary = "\n".join(summary_lines)
            self.memory_summary = "\n".join(combined_summary.splitlines()[-40:])
        self.heuristic.observe(step_record)


class OpenAICompatibleExternalGuiAdapter:
    def __init__(
        self,
        *,
        agent_id: str,
        agent_type: str,
        target_app: str,
        start_url: str,
        seed: int,
        model_config: dict[str, Any],
        policy: str,
        prompt_template: str,
        system_prompt: str,
        parser_name: str,
        model_name_override: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.target_app = target_app
        self.start_url = start_url
        self.model_config = dict(model_config)
        if model_name_override:
            self.model_config["model"] = model_name_override
        self.policy = policy
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.parser_name = parser_name
        self.heuristic = HeuristicExplorer(target_app, start_url, seed)
        self.history: list[dict[str, Any]] = []

    def _has_model_config(self) -> bool:
        return bool(
            self.model_config.get("model")
            and self.model_config.get("base_url")
            and self.model_config.get("api_key")
        )

    def _effective_policy(self) -> str:
        if self.policy == "heuristic":
            return "heuristic"
        if self.policy == "model":
            if not self._has_model_config():
                missing = [key for key in ("model", "base_url", "api_key") if not self.model_config.get(key)]
                raise ValueError(f"Model policy requested but missing config fields: {missing}")
            return "model"
        return "model" if self._has_model_config() else "heuristic"

    def _memory_block(self) -> str:
        if not self.history:
            return "Session memory: no previous steps in this session."
        recent = self.history[-40:]
        return "Recent session memory:\n" + json.dumps(recent, ensure_ascii=False, indent=2)

    def _model_response(self, prompt: str, obs: dict[str, Any]) -> str:
        content: str | list[dict[str, Any]] = prompt
        image_url = screenshot_data_url(obs)
        if image_url:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        payload = {
            "model": self.model_config["model"],
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": float(self.model_config.get("temperature", 0.0)),
            "max_tokens": int(self.model_config.get("max_tokens", 512)),
        }
        if self.model_config.get("reasoning_effort"):
            payload["reasoning_effort"] = self.model_config["reasoning_effort"]
        url = chat_completions_url(str(self.model_config["base_url"]))
        headers = {
            "Authorization": f"Bearer {self.model_config['api_key']}",
            "Content-Type": "application/json",
        }
        retries = int(self.model_config.get("retries", 3) or 0)
        retry_backoff = float(self.model_config.get("retry_backoff", 10.0) or 0.0)
        with httpx.Client(
            timeout=int(self.model_config.get("timeout", 120)),
            trust_env=httpx_trust_env_for_url(url),
        ) as client:
            for attempt in range(retries + 1):
                try:
                    response = client.post(url, headers=headers, json=payload)
                    if response.is_error:
                        if response.status_code in {408, 409, 429, 500, 502, 503, 504} and attempt < retries:
                            time.sleep(retry_backoff * (attempt + 1))
                            continue
                        raise RuntimeError(
                            f"External GUI model request failed: status={response.status_code}, "
                            f"url={url}, body={response.text[:1000]}"
                        )
                    data = response.json()
                    return str(data["choices"][0]["message"]["content"])
                except (httpx.TimeoutException, httpx.TransportError):
                    if attempt < retries:
                        time.sleep(retry_backoff * (attempt + 1))
                        continue
                    raise
        raise RuntimeError("External GUI model request failed without a response")

    def _parse_native(self, response: str) -> tuple[str, NativeGuiAction, bool]:
        if self.parser_name == "mai-ui":
            return parse_mai_ui_native_action(response)
        if self.parser_name == "mobile-agent-v35":
            return parse_mobile_agent_native_action(response)
        raise ValueError(f"Unsupported external GUI parser: {self.parser_name}")

    def decide(self, obs: dict[str, Any], session_state: BaselineSessionState) -> BaselineAgentDecision:
        prompt = build_external_gui_prompt(
            obs=obs,
            session_state=session_state,
            memory_block=self._memory_block(),
            template=self.prompt_template,
        )
        effective_policy = self._effective_policy()
        if effective_policy == "model":
            response = self._model_response(prompt, obs)
            think, native_action, tag_valid = self._parse_native(response)
            action, conversion_metadata = native_gui_action_to_browsergym(native_action, obs)
        else:
            obs_text = observation_text(obs)
            action = self.heuristic.choose_action(obs_text, session_state)
            think = "Use heuristic fallback while preserving external baseline schema."
            native_action = NativeGuiAction(kind="heuristic", raw=action)
            tag_valid = True
            conversion_metadata = {
                "native_action_kind": "heuristic",
                "native_action_parse_valid": True,
                "native_action_conversion": "heuristic_browsergym",
            }
            if self.parser_name == "mai-ui":
                response = (
                    f"<thinking>{think}</thinking>"
                    f"<tool_call>{{\"name\":\"browsergym\",\"arguments\":{{\"action\":{json.dumps(action)}}}}}</tool_call>"
                )
            else:
                response = (
                    f"Thought: {think}\n"
                    f"Action: <tool_call>{{\"name\":\"browsergym\",\"arguments\":{{\"action\":{json.dumps(action)}}}}}</tool_call>"
                )
        return BaselineAgentDecision(
            response=response,
            think=think,
            action=action,
            metadata={
                "prompt": prompt,
                "decision_policy": effective_policy,
                "response_tag_valid": tag_valid,
                "response_has_think": bool(think),
                "use_vision": True,
                "history_mode": "external_recent_memory",
                "history_window": 40,
                "memory_mode": "recent",
                "model_name": self.model_config.get("model") if effective_policy == "model" else None,
                "external_parser": self.parser_name,
                "native_response_raw": native_action.raw,
                **conversion_metadata,
            },
        )

    def observe(self, step_record: StepHistory) -> None:
        extra = step_record.extra_fields or {}
        item = {
            "session_step_idx": int(extra.get("session_step_idx", step_record.step)),
            "episode_idx": int(extra.get("episode_idx", 0) or 0),
            "step_idx": int(extra.get("step_idx", step_record.step) or 0),
            "action": extra.get("action"),
            "reward": float(step_record.reward or 0.0),
            "coverage_delta_score": int(extra.get("coverage_delta_score", 0) or 0),
            "action_execution_valid": bool(extra.get("action_execution_valid", False)),
            "coverage_skip_reason": extra.get("coverage_skip_reason"),
            "native_action_kind": extra.get("agent_metadata_native_action_kind"),
            "native_action_conversion": extra.get("agent_metadata_native_action_conversion"),
        }
        self.history.append(item)
        if len(self.history) > 80:
            self.history = self.history[-80:]
        self.heuristic.observe(step_record)


class JAMELMemoryAugAdapter:
    def __init__(
        self,
        *,
        agent_id: str,
        checkpoint: str,
        compressor_model: str,
        device: str,
        memory_max_items: int,
        memory_builder: str = "online_tokens",
        delta_rank: int = 8,
        delta_memory_slots: int = 8,
        delta_seed: int = 13,
        hybrid_recent_items: int = 32,
    ) -> None:
        if not checkpoint:
            raise ValueError("jamel-memory-aug requires --checkpoint")
        if not compressor_model:
            raise ValueError("jamel-memory-aug requires --compressor-model")
        self.agent_id = agent_id
        self.agent_type = "jamel-memory-aug"
        self.checkpoint = checkpoint
        self.compressor_model = compressor_model
        self.device = device
        self.memory_max_items = memory_max_items
        self.memory_builder_type = str(memory_builder).strip().lower().replace("-", "_")
        self.delta_rank = int(delta_rank)
        self.delta_memory_slots = int(delta_memory_slots)
        self.delta_seed = int(delta_seed)
        self.hybrid_recent_items = int(hybrid_recent_items)
        self._loaded = False
        self._history_records: list[dict[str, Any]] = []

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoProcessor
        from jamel.train.memory.delta_state_encoder import make_history_memory_builder
        from jamel.train.memory.modeling import MemoryAugmentedCausalLM

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.model = MemoryAugmentedCausalLM.from_pretrained(
            self.checkpoint,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.aligner = self.model.aligner.to(dtype=torch.bfloat16)
        self.model.eval()
        mem_cfg_path = Path(self.checkpoint) / "memory_augment_config.json"
        if mem_cfg_path.exists():
            mem_cfg = json.loads(mem_cfg_path.read_text(encoding="utf-8"))
            self.memory_hidden_size = int(mem_cfg.get("memory_hidden_size", 2048))
        else:
            self.memory_hidden_size = 2048
        self.use_online_delta_state = getattr(self.model, "delta_state_memory", None) is not None
        builder_type = self.memory_builder_type
        if self.use_online_delta_state and builder_type not in {"delta_state", "online_delta_state"}:
            builder_type = "online_delta_state"
        self.memory_builder = make_history_memory_builder(
            memory_builder=builder_type,
            compressor_model_name=self.compressor_model,
            memory_hidden_size=self.memory_hidden_size,
            history_window=self.memory_max_items,
            max_memory_items=self.memory_max_items,
            torch_dtype="bfloat16",
            device_map=self.device,
            cache_history_memory=True,
            delta_rank=self.delta_rank,
            delta_memory_slots=self.delta_memory_slots,
            delta_seed=self.delta_seed,
            hybrid_recent_items=self.hybrid_recent_items,
        )
        self.memory_builder_type = builder_type
        compressor = self.memory_builder.compressor
        tokenizer = getattr(getattr(compressor, "processor", None), "tokenizer", None)
        if tokenizer is not None:
            tokenizer.add_eos_token = True
        self._loaded = True

    def decide(self, obs: dict[str, Any], session_state: BaselineSessionState) -> BaselineAgentDecision:
        self._ensure_loaded()
        prompt = build_prompt(
            obs=obs,
            session_state=session_state,
            memory_block="Session memory is provided through learned memory tokens.",
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt_text], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        inputs.pop("second_per_grid_ts", None)
        if self.use_online_delta_state:
            history_tokens, history_mask = self.memory_builder.build_step_token_inputs(
                batch_size=1,
                history_records=[self._history_records],
            )
            screenshot = obs.get("screenshot") if isinstance(obs, dict) else None
            if screenshot is None:
                query_tokens = self.torch.zeros((1, 1, self.memory_hidden_size), dtype=self.torch.float32)
                query_mask = self.torch.zeros((1, 1), dtype=self.torch.long)
            else:
                query_image = Image.fromarray(screenshot.astype("uint8"))
                query_tokens, query_mask = self.memory_builder.build_current_query_inputs(
                    images=[query_image],
                )
            memory_kwargs = {
                "history_memory_tokens": history_tokens.to(self.device, dtype=self.torch.bfloat16),
                "history_memory_attention_mask": history_mask.to(self.device, dtype=self.torch.long),
                "current_memory_query_tokens": query_tokens.to(self.device, dtype=self.torch.bfloat16),
                "current_memory_query_attention_mask": query_mask.to(self.device, dtype=self.torch.long),
            }
        else:
            memory_tokens, memory_mask = self.memory_builder.build_memory_inputs(
                batch_size=1,
                history_records=[self._history_records],
            )
            memory_kwargs = {
                "memory_tokens": memory_tokens.to(self.device, dtype=self.torch.bfloat16),
                "memory_attention_mask": memory_mask.to(self.device, dtype=self.torch.long),
            }
        input_len = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                **memory_kwargs,
                max_new_tokens=256,
                do_sample=False,
            )
        raw = self.processor.batch_decode(
            generated[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        think, action, tag_valid = extract_action_response(raw)
        return BaselineAgentDecision(
            response=raw,
            think=think,
            action=action,
            metadata={
                "prompt": prompt,
                "decision_policy": "jamel-memory-aug",
                "response_tag_valid": tag_valid,
                "response_has_think": bool(think),
                "checkpoint": self.checkpoint,
                "memory_max_items": self.memory_max_items,
                "memory_builder": self.memory_builder_type,
                "delta_rank": self.delta_rank,
                "delta_memory_slots": self.delta_memory_slots,
            },
        )

    def observe(self, step_record: StepHistory) -> None:
        self._ensure_loaded()
        obs = step_record.before_obs or {}
        action = str((step_record.extra_fields or {}).get("action") or "")
        screenshot = obs.get("screenshot") if isinstance(obs, dict) else None
        image = Image.fromarray(screenshot.astype("uint8")) if screenshot is not None else None
        self._history_records.append({"image_obs": image, "action": action})
        if len(self._history_records) > self.memory_max_items:
            self._history_records = self._history_records[-self.memory_max_items :]


def env_first(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def resolve_model_config(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(REPO_ROOT / ".env", override=False)
    model_name = args.model_name or env_first("BASELINE_MODEL_NAME", "MODEL_NAME", "OPENAI_MODEL")
    base_url = args.model_base_url or env_first("BASELINE_MODEL_BASE_URL", "MODEL_BASE_URL", "OPENAI_BASE_URL")
    api_key = args.model_api_key or env_first("BASELINE_MODEL_API_KEY", "MODEL_API_KEY", "OPENAI_API_KEY")
    return {
        "model": model_name,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": args.model_temperature,
        "max_tokens": args.model_max_tokens,
        "timeout": args.model_timeout,
        "retries": args.model_retries,
        "retry_backoff": args.model_retry_backoff,
        "context_tokens": args.model_context_tokens,
        "reasoning_effort": getattr(args, "model_reasoning_effort", None),
    }


def chat_completions_url(base_url: str) -> str:
    url = str(base_url).rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def build_agent(
    agent_name: str,
    *,
    target_app: str,
    start_url: str,
    seed: int,
    args: argparse.Namespace,
) -> BaselineAgentAdapter:
    model_config = resolve_model_config(args)
    agent_name = agent_name.strip()
    if agent_name == "react-text":
        return OpenAICompatibleReActAdapter(
            agent_id=agent_name,
            agent_type=agent_name,
            target_app=target_app,
            start_url=start_url,
            seed=seed,
            model_config=model_config,
            policy=args.policy,
            history_mode="full",
            history_budget_mode=args.history_budget_mode,
            history_char_budget=args.history_char_budget,
            history_observation_char_budget=args.history_observation_char_budget,
            tokenizer_name=args.history_tokenizer_name,
            model_context_tokens=args.model_context_tokens,
            context_margin_tokens=args.context_margin_tokens,
        )
    if agent_name == "react-vision":
        return OpenAICompatibleReActAdapter(
            agent_id=agent_name,
            agent_type=agent_name,
            target_app=target_app,
            start_url=start_url,
            seed=seed,
            model_config=model_config,
            policy=args.policy,
            use_vision=True,
            history_mode="full",
            history_budget_mode=args.history_budget_mode,
            history_char_budget=args.history_char_budget,
            history_observation_char_budget=args.history_observation_char_budget,
            tokenizer_name=args.history_tokenizer_name,
            model_context_tokens=args.model_context_tokens,
            context_margin_tokens=args.context_margin_tokens,
        )
    if agent_name == "react-history":
        return OpenAICompatibleReActAdapter(
            agent_id=agent_name,
            agent_type=agent_name,
            target_app=target_app,
            start_url=start_url,
            seed=seed,
            model_config=model_config,
            policy=args.policy,
            history_window=args.history_window,
            history_mode="window",
            history_budget_mode=args.history_budget_mode,
            history_char_budget=args.history_char_budget,
            history_observation_char_budget=args.history_observation_char_budget,
            tokenizer_name=args.history_tokenizer_name,
            model_context_tokens=args.model_context_tokens,
            context_margin_tokens=args.context_margin_tokens,
        )
    if agent_name == "react-summary":
        return OpenAICompatibleReActAdapter(
            agent_id=agent_name,
            agent_type=agent_name,
            target_app=target_app,
            start_url=start_url,
            seed=seed,
            model_config=model_config,
            policy=args.policy,
            history_window=args.history_window,
            history_mode="summary",
            history_budget_mode=args.history_budget_mode,
            history_char_budget=args.history_char_budget,
            history_observation_char_budget=args.history_observation_char_budget,
            tokenizer_name=args.history_tokenizer_name,
            model_context_tokens=args.model_context_tokens,
            context_margin_tokens=args.context_margin_tokens,
            memory_mode="summary",
        )
    if agent_name == "react-compressed":
        return OpenAICompatibleReActAdapter(
            agent_id=agent_name,
            agent_type=agent_name,
            target_app=target_app,
            start_url=start_url,
            seed=seed,
            model_config=model_config,
            policy=args.policy,
            history_window=args.history_window,
            history_mode="compressed",
            history_budget_mode=args.history_budget_mode,
            history_char_budget=args.history_char_budget,
            history_observation_char_budget=args.history_observation_char_budget,
            tokenizer_name=args.history_tokenizer_name,
            model_context_tokens=args.model_context_tokens,
            context_margin_tokens=args.context_margin_tokens,
            memory_mode="compressed",
        )
    if agent_name == "mai-ui":
        return OpenAICompatibleExternalGuiAdapter(
            agent_id=agent_name,
            agent_type=agent_name,
            target_app=target_app,
            start_url=start_url,
            seed=seed,
            model_config=model_config,
            policy=args.policy,
            prompt_template=MAI_UI_PROMPT_TEMPLATE,
            system_prompt=MAI_UI_SYSTEM_PROMPT,
            parser_name="mai-ui",
            model_name_override=args.mai_ui_model_name,
        )
    if agent_name in {"mobile-agent-v35", "mobile-agent-v3.5"}:
        return OpenAICompatibleExternalGuiAdapter(
            agent_id="mobile-agent-v35",
            agent_type="mobile-agent-v35",
            target_app=target_app,
            start_url=start_url,
            seed=seed,
            model_config=model_config,
            policy=args.policy,
            prompt_template=MOBILE_AGENT_PROMPT_TEMPLATE,
            system_prompt=MOBILE_AGENT_SYSTEM_PROMPT,
            parser_name="mobile-agent-v35",
            model_name_override=args.mobile_agent_model_name,
        )
    if agent_name == "jamel-memory-aug":
        return JAMELMemoryAugAdapter(
            agent_id=agent_name,
            checkpoint=args.checkpoint,
            compressor_model=args.compressor_model,
            device=args.device,
            memory_max_items=args.memory_max_items,
            memory_builder=args.memory_builder,
            delta_rank=args.delta_rank,
            delta_memory_slots=args.delta_memory_slots,
            delta_seed=args.delta_seed,
            hybrid_recent_items=args.hybrid_recent_items,
        )
    raise ValueError(f"Unsupported baseline agent: {agent_name}")


def compute_coverage_reward_details(
    *,
    current_path: Path | None,
    baseline_paths: list[str | Path],
    previous_score: int = 0,
) -> dict[str, Any]:
    return compute_monocart_coverage_reward_details(
        current_path=current_path,
        baseline_paths=baseline_paths,
        previous_score=previous_score,
    )


def reset_reward_details(previous_score: int) -> dict[str, Any]:
    return {
        "reward": 0.0,
        "previous_score": int(previous_score),
        "current_score": int(previous_score),
        "delta_score": 0,
        "skip_reason": "reset_action",
    }


def invalid_reward_details(previous_score: int, reason: str) -> dict[str, Any]:
    return {
        "reward": 0.0,
        "previous_score": int(previous_score),
        "current_score": int(previous_score),
        "delta_score": 0,
        "skip_reason": reason,
    }


def step_error_indicates_dead_browser(error: str) -> bool:
    lowered = error.lower()
    return any(
        marker in lowered
        for marker in (
            "target crashed",
            "target page, context or browser has been closed",
            "browser has been closed",
            "page has been closed",
            "context has been closed",
        )
    )


def advance_session_indices_after_action(
    session_state: BaselineSessionState,
    *,
    action: str,
    action_format_valid: bool,
) -> None:
    if action.strip() == "reset()" and action_format_valid:
        session_state.episode_idx += 1
        session_state.step_idx = 1
        return
    session_state.step_idx += 1


def build_step_history(
    *,
    before_obs: dict[str, Any],
    after_obs: dict[str, Any],
    before_info: dict[str, Any],
    after_info: dict[str, Any],
    before_observation: str,
    after_observation: str,
    session_state: BaselineSessionState,
    decision: BaselineAgentDecision,
    action_format_valid: bool,
    action_validation_error: str | None,
    action_execution_valid: bool,
    reward_details: dict[str, Any],
    artifact_fields: dict[str, Any],
    screenshot_paths: dict[str, str | None],
    result: dict[str, Any],
) -> StepHistory:
    metadata = dict(decision.metadata or {})
    prompt = str(metadata.pop("prompt", ""))
    reward = float(reward_details.get("reward", 0.0) or 0.0)
    extra_fields = {
        "run_id": session_state.run_id,
        "session_id": session_state.session_id,
        "agent_id": session_state.agent_id,
        "agent_type": session_state.agent_type,
        "target_app": session_state.target_app,
        "target_url": session_state.start_url,
        "start_url": session_state.start_url,
        "episode_idx": session_state.episode_idx,
        "step_idx": session_state.step_idx,
        "session_step_idx": session_state.session_step_idx,
        "prompt": prompt,
        "response": decision.response,
        "think": decision.think,
        "action": decision.action,
        "action_format_valid": bool(action_format_valid),
        "action_validation_error": action_validation_error,
        "action_execution_valid": bool(action_execution_valid),
        "reward_source": "coverage" if reward > 0 else "none",
        "coverage_previous_score": int(reward_details.get("previous_score", 0) or 0),
        "coverage_current_score": int(reward_details.get("current_score", 0) or 0),
        "coverage_delta_score": int(reward_details.get("delta_score", 0) or 0),
        "coverage_skip_reason": reward_details.get("skip_reason"),
        "before_screenshot_path": screenshot_paths.get("before_screenshot_path"),
        "after_screenshot_path": screenshot_paths.get("after_screenshot_path"),
        **coverage_artifact_extra_fields(artifact_fields),
        **{f"agent_metadata_{key}": value for key, value in metadata.items()},
    }
    return StepHistory(
        before_obs=before_obs,
        after_obs=after_obs,
        before_info=before_info,
        after_info=after_info,
        before_observation=before_observation,
        after_observation=after_observation,
        step=session_state.session_step_idx,
        reward=reward,
        raw_content=decision.response,
        memory_content=None,
        parsed_content={"think": decision.think, "action": decision.action},
        result=result,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        extra_fields=extra_fields,
    )


def write_reward_curve(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    curve = []
    cumulative = 0.0
    for row in rows:
        reward = float(row.get("reward", 0.0) or 0.0)
        cumulative += reward
        curve.append(
            {
                "session_step_idx": int(row.get("session_step_idx", len(curve) + 1) or len(curve) + 1),
                "reward": reward,
                "cumulative_reward": cumulative,
            }
        )

    json_path = output_dir / "reward_curve.json"
    json_path.write_text(json.dumps(curve, indent=2, ensure_ascii=False), encoding="utf-8")

    png_path = output_dir / "reward_curve.png"
    width, height = 900, 420
    margin_left, margin_top, margin_right, margin_bottom = 60, 30, 30, 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    draw.rectangle(
        [margin_left, margin_top, margin_left + plot_w, margin_top + plot_h],
        outline=(190, 190, 190),
    )
    if curve:
        max_step = max(item["session_step_idx"] for item in curve) or 1
        max_cum = max(1.0, max(item["cumulative_reward"] for item in curve))

        def point(step: int, value: float, max_value: float) -> tuple[int, int]:
            x = margin_left + int((step - 1) / max(1, max_step - 1) * plot_w)
            y = margin_top + plot_h - int(value / max_value * plot_h)
            return x, y

        reward_points = [point(item["session_step_idx"], item["reward"], 1.0) for item in curve]
        cumulative_points = [point(item["session_step_idx"], item["cumulative_reward"], max_cum) for item in curve]
        if len(reward_points) > 1:
            draw.line(reward_points, fill=(33, 113, 181), width=2)
            draw.line(cumulative_points, fill=(35, 139, 69), width=3)
        for x, y in reward_points:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(33, 113, 181))
        for x, y in cumulative_points:
            draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill=(35, 139, 69))
    draw.text((margin_left, height - 35), "blue: per-step reward    green: cumulative reward", fill=(50, 50, 50))
    image.save(png_path)
    return {"reward_curve_json": str(json_path), "reward_curve_png": str(png_path)}


def generate_optional_html_report(coverage_paths: list[str | Path], output_dir: Path) -> str | None:
    existing_paths = [str(Path(path).resolve()) for path in coverage_paths if Path(path).exists()]
    if not existing_paths:
        return None
    if not GENERATE_REPORT_SCRIPT.exists():
        logger.warning("Monocart report script missing", script=str(GENERATE_REPORT_SCRIPT))
        return None
    report_dir = output_dir / "coverage_report"
    try:
        subprocess.run(
            ["node", str(GENERATE_REPORT_SCRIPT), "--output-dir", str(report_dir), *existing_paths],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("Failed to generate Monocart HTML report", error=str(exc))
        return None
    index_path = report_dir / "index.html"
    return str(index_path) if index_path.exists() else str(report_dir)


def history_rows_for_curve(history: list[StepHistory]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in history:
        extra = step.extra_fields or {}
        rows.append(
            {
                "session_step_idx": extra.get("session_step_idx", step.step),
                "reward": step.reward,
            }
        )
    return rows


def run_session(
    *,
    agent: BaselineAgentAdapter,
    target_app: str,
    start_url: str,
    run_id: str,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    session_id = sanitize_id(f"{agent.agent_id}__{target_app}__seed{args.seed}")
    session_dir = output_root / session_id
    coverage_dir = session_dir / "coverage"
    screenshot_dir = session_dir / "screenshots"
    session_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    state = BaselineSessionState(
        run_id=run_id,
        session_id=session_id,
        agent_id=agent.agent_id,
        agent_type=agent.agent_type,
        target_app=target_app,
        start_url=start_url,
        max_steps=args.max_steps,
        seed=args.seed,
        output_dir=session_dir,
    )
    env_context = get_environment(
        start_url,
        headless=not args.show_browser,
        record_coverage=True,
        timeout=args.timeout,
    )
    obs, info = env_context.obs, env_context.info
    history: list[StepHistory] = []
    reward_coverage_paths: list[str] = []
    all_coverage_paths: list[str] = []
    terminated_count = 0
    truncated_count = 0

    try:
        for session_step_idx in range(1, args.max_steps + 1):
            state.session_step_idx = session_step_idx
            before_obs = obs
            before_info = info
            before_observation = observation_text(before_obs)
            try:
                decision = decide_with_hard_timeout(
                    agent,
                    before_obs,
                    state,
                    int(getattr(args, "decision_timeout", 0) or 0),
                )
            except DecisionTimeoutError as exc:
                logger.warning(
                    "Agent decision timed out; recording noop step",
                    target_app=target_app,
                    session_step_idx=session_step_idx,
                    error=str(exc),
                )
                think = f"Agent decision timed out after {int(getattr(args, 'decision_timeout', 0) or 0)} seconds."
                action = "noop(wait_ms=1000)"
                response = f"<think>{think}</think><action>{action}</action>"
                decision = BaselineAgentDecision(
                    response=response,
                    think=think,
                    action=action,
                    metadata={
                        "prompt": "",
                        "decision_policy": "timeout_fallback",
                        "decision_timeout": True,
                        "decision_timeout_seconds": int(getattr(args, "decision_timeout", 0) or 0),
                        "decision_timeout_error": str(exc),
                        "response_tag_valid": True,
                        "response_has_think": True,
                    },
                )
            if not decision.action:
                think, action, tag_valid = extract_action_response(decision.response)
                decision.think = decision.think or think
                decision.action = decision.action or action
                decision.metadata["response_tag_valid"] = tag_valid
                decision.metadata["response_has_think"] = bool(decision.think)

            replacement_action = replace_known_hanging_action(
                target_app=target_app,
                session_step_idx=session_step_idx,
                action=decision.action,
            )
            if replacement_action is not None:
                logger.warning(
                    "Replacing known hanging action with noop",
                    target_app=target_app,
                    session_step_idx=session_step_idx,
                    original_action=decision.action,
                    replacement_action=replacement_action,
                )
                decision.metadata["model_proposed_action_before_hang_guard"] = decision.action
                decision.metadata["action_replaced_by_hang_guard"] = True
                decision.action = replacement_action

            action_format_valid, action_validation_error = validate_action(decision.action, before_observation)
            if not bool(decision.metadata.get("response_tag_valid", True)):
                action_format_valid = False
                action_validation_error = action_validation_error or "response is missing exactly one action tag"

            before_screenshot_path = save_screenshot_file(
                before_obs,
                screenshot_dir / f"step_{session_step_idx:03d}_before.png",
            )
            coverage_path = coverage_dir / f"coverage_{session_step_idx}.json"

            after_obs = before_obs
            after_info = before_info
            result: dict[str, Any] = {
                "terminated": False,
                "truncated": False,
                "raw_reward": 0.0,
            }
            action_execution_valid = False
            skip_coverage_write_reason: str | None = None

            if action_format_valid and decision.action.strip() == "reset()":
                if env_context.save_step_coverage is not None:
                    env_context.save_step_coverage(coverage_path, session_step_idx)
                artifact_fields = build_coverage_artifact_fields(coverage_path)
                if artifact_fields["coverage_exists_at_write"]:
                    all_coverage_paths.append(str(coverage_path))
                stop_envrionment(env_context)
                env_context = get_environment(
                    start_url,
                    headless=not args.show_browser,
                    record_coverage=True,
                    timeout=args.timeout,
                )
                after_obs, after_info = env_context.obs, env_context.info
                action_execution_valid = True
                reward_details = reset_reward_details(state.last_coverage_score)
                result["episode_boundary"] = True
            elif action_format_valid:
                try:
                    after_obs, raw_reward, terminated, truncated, after_info = env_step_with_hard_timeout(
                        env_context.env,
                        decision.action,
                        int(getattr(args, "env_step_timeout", 0) or 0),
                    )
                    result.update(
                        {
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "raw_reward": float(raw_reward or 0.0),
                        }
                    )
                    if terminated:
                        terminated_count += 1
                    if truncated:
                        truncated_count += 1
                    action_execution_valid = bool(is_action_execution_valid(after_obs))
                except Exception as exc:
                    step_error = str(exc)
                    logger.warning("BrowserGym step failed", action=decision.action, error=step_error)
                    result["step_error"] = step_error
                    action_execution_valid = False
                    if step_error_indicates_dead_browser(step_error):
                        skip_coverage_write_reason = "browser_crash_before_coverage"
                        result["needs_env_restart"] = True
                    elif isinstance(exc, EnvStepTimeoutError):
                        skip_coverage_write_reason = "env_step_timeout_before_coverage"
                        result["needs_env_restart"] = True
                        result["restart_without_close"] = True

                if skip_coverage_write_reason:
                    artifact_fields = build_coverage_artifact_fields(None)
                elif env_context.save_step_coverage is not None:
                    env_context.save_step_coverage(coverage_path, session_step_idx)
                    artifact_fields = build_coverage_artifact_fields(coverage_path)
                else:
                    artifact_fields = build_coverage_artifact_fields(None)
                if artifact_fields["coverage_exists_at_write"]:
                    all_coverage_paths.append(str(coverage_path))
                if action_execution_valid:
                    reward_details = compute_coverage_reward_details(
                        current_path=coverage_path,
                        baseline_paths=reward_coverage_paths,
                        previous_score=state.last_coverage_score,
                    )
                else:
                    reason = (
                        skip_coverage_write_reason
                        or ("step_error" if result.get("step_error") else "invalid_action_execution")
                    )
                    reward_details = invalid_reward_details(state.last_coverage_score, reason)
                if artifact_fields["coverage_exists_at_write"]:
                    reward_coverage_paths.append(str(coverage_path))
                state.last_coverage_score = int(reward_details.get("current_score", state.last_coverage_score) or 0)
            else:
                if env_context.save_step_coverage is not None:
                    env_context.save_step_coverage(coverage_path, session_step_idx)
                artifact_fields = build_coverage_artifact_fields(coverage_path)
                if artifact_fields["coverage_exists_at_write"]:
                    all_coverage_paths.append(str(coverage_path))
                reward_details = invalid_reward_details(state.last_coverage_score, "invalid_action_format")

            if (
                not artifact_fields["coverage_exists_at_write"]
                and decision.action.strip() != "reset()"
                and not reward_details.get("skip_reason")
            ):
                reward_details = invalid_reward_details(state.last_coverage_score, "missing_coverage_at_write")

            after_observation = observation_text(after_obs)
            after_screenshot_path = save_screenshot_file(
                after_obs,
                screenshot_dir / f"step_{session_step_idx:03d}_after.png",
            )
            step_history = build_step_history(
                before_obs=before_obs,
                after_obs=after_obs,
                before_info=before_info,
                after_info=after_info,
                before_observation=before_observation,
                after_observation=after_observation,
                session_state=state,
                decision=decision,
                action_format_valid=action_format_valid,
                action_validation_error=action_validation_error,
                action_execution_valid=action_execution_valid,
                reward_details=reward_details,
                artifact_fields=artifact_fields,
                screenshot_paths={
                    "before_screenshot_path": before_screenshot_path,
                    "after_screenshot_path": after_screenshot_path,
                },
                result=result,
            )
            history.append(step_history)
            agent.observe(step_history)

            state.cumulative_reward += float(step_history.reward or 0.0)
            state.history.append(
                {
                    "session_step_idx": session_step_idx,
                    "episode_idx": state.episode_idx,
                    "step_idx": state.step_idx,
                    "action": decision.action,
                    "reward": float(step_history.reward or 0.0),
                    "coverage_delta_score": int(reward_details.get("delta_score", 0) or 0),
                    "action_execution_valid": action_execution_valid,
                }
            )

            if decision.action.strip() == "reset()" and action_format_valid:
                advance_session_indices_after_action(
                    state,
                    action=decision.action,
                    action_format_valid=action_format_valid,
                )
                obs, info = after_obs, after_info
            else:
                advance_session_indices_after_action(
                    state,
                    action=decision.action,
                    action_format_valid=action_format_valid,
                )
                obs, info = after_obs, after_info
                if result.get("terminated") or result.get("truncated") or result.get("needs_env_restart"):
                    if result.get("restart_without_close"):
                        logger.warning(
                            "Restarting environment after timed out env.step without blocking close",
                            target_app=target_app,
                            session_step_idx=session_step_idx,
                        )
                        env_context = None
                    else:
                        stop_envrionment(env_context)
                    env_context = get_environment(
                        start_url,
                        headless=not args.show_browser,
                        record_coverage=True,
                        timeout=args.timeout,
                    )
                    obs, info = env_context.obs, env_context.info

        trajectory_path = Observer.save_trajectory(
            history=history,
            history_dir=str(session_dir),
            filename="trajectory.parquet",
            metadata={
                "schema_version": 1,
                "run_id": run_id,
                "session_id": session_id,
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "target_app": target_app,
                "start_url": start_url,
                "max_steps": args.max_steps,
                "seed": args.seed,
            },
        )
        trajectory_path_obj = Path(trajectory_path) if trajectory_path else session_dir / "trajectory.parquet"
        curve_paths = write_reward_curve(history_rows_for_curve(history), session_dir)
        html_report = generate_optional_html_report(all_coverage_paths, session_dir) if args.monocart_html_report else None

        rewards = [float(step.reward or 0.0) for step in history]
        coverage_deltas = [int((step.extra_fields or {}).get("coverage_delta_score", 0) or 0) for step in history]
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "session_id": session_id,
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "target_app": target_app,
            "start_url": start_url,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "steps": len(history),
            "episodes": max(1, state.episode_idx + 1),
            "resets": sum(1 for step in history if (step.extra_fields or {}).get("action") == "reset()"),
            "cumulative_reward": sum(rewards),
            "coverage_delta_score_total": sum(coverage_deltas),
            "coverage_current_score": state.last_coverage_score,
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "trajectory_path": str(trajectory_path_obj),
            "coverage_dir": str(coverage_dir),
            "screenshots_dir": str(screenshot_dir),
            **curve_paths,
            "monocart_html_report": html_report,
            "actions": [(step.extra_fields or {}).get("action") for step in history],
            "rewards": rewards,
            "coverage_delta_scores": coverage_deltas,
        }
        summary_path = session_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
    finally:
        if env_context is not None:
            stop_envrionment(
                env_context,
                record_coverage=True,
                record_coverage_path=coverage_dir / "coverage_final.json",
            )


def run_baseline_eval(args: argparse.Namespace) -> dict[str, Any]:
    if not args.run_id:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir or (DEFAULT_OUTPUT_ROOT / args.run_id)).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    scalewob_root = Path(args.scalewob_root or DEFAULT_SCALEWOB_ROOT).resolve()
    if not scalewob_root.exists() and not args.use_existing_server:
        raise FileNotFoundError(f"ScaleWoB root not found: {scalewob_root}")

    server = None
    port = args.port
    if args.use_existing_server:
        if not port_is_open(args.host, args.port):
            raise RuntimeError(f"--use-existing-server was set, but {args.host}:{args.port} is not reachable")
    else:
        server = LocalStaticServer(
            scalewob_root,
            host=args.host,
            port=args.port,
            port_search=not args.no_port_search,
        )
        port = server.start()

    apps = parse_apps(args.apps)
    agents = parse_agents(args.agents)
    summaries: list[dict[str, Any]] = []
    try:
        for target_app in apps:
            start_url = start_url_for_app(args.host, port, target_app)
            for agent_name in agents:
                agent = build_agent(
                    agent_name,
                    target_app=target_app,
                    start_url=start_url,
                    seed=args.seed,
                    args=args,
                )
                logger.info(
                    "Running baseline GUI eval session",
                    agent=agent_name,
                    target_app=target_app,
                    start_url=start_url,
                    max_steps=args.max_steps,
                )
                try:
                    summaries.append(
                        run_session(
                            agent=agent,
                            target_app=target_app,
                            start_url=start_url,
                            run_id=args.run_id,
                            output_root=output_root,
                            args=args,
                        )
                    )
                except Exception as exc:
                    logger.error(
                        "Baseline GUI eval session failed",
                        agent=agent_name,
                        target_app=target_app,
                        error=str(exc),
                        exc_info=True,
                    )
                    summaries.append(
                        {
                            "schema_version": 1,
                            "run_id": args.run_id,
                            "session_id": sanitize_id(f"{agent.agent_id}__{target_app}__seed{args.seed}"),
                            "agent_id": agent.agent_id,
                            "agent_type": agent.agent_type,
                            "target_app": target_app,
                            "start_url": start_url,
                            "seed": args.seed,
                            "max_steps": args.max_steps,
                            "steps": 0,
                            "episodes": 0,
                            "resets": 0,
                            "cumulative_reward": 0.0,
                            "coverage_delta_score_total": 0,
                            "coverage_current_score": 0,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
        manifest = {
            "schema_version": 1,
            "run_id": args.run_id,
            "output_root": str(output_root),
            "apps": apps,
            "agents": agents,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "sessions": summaries,
        }
        (output_root / "summary.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        with (output_root / "manifest.jsonl").open("w", encoding="utf-8") as fout:
            for summary in summaries:
                fout.write(json.dumps(summary, ensure_ascii=False) + "\n")
        return manifest
    finally:
        if server is not None:
            server.stop()


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--apps", default="weibo", help="Comma-separated ScaleWoB app ids, e.g. weibo,agoda")
    parser.add_argument(
        "--agents",
        default="react-text",
        help=(
            "Comma-separated baseline agents: react-text, react-vision, react-history, "
            "react-summary, react-compressed, mai-ui, mobile-agent-v35, "
            "jamel-memory-aug"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scalewob-root", default=str(DEFAULT_SCALEWOB_ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-port-search", action="store_true")
    parser.add_argument("--use-existing-server", action="store_true")
    parser.add_argument("--timeout", type=int, default=60000)
    parser.add_argument("--show-browser", action="store_true")
    parser.add_argument("--policy", choices=["auto", "model", "heuristic"], default="auto")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-base-url", default=None)
    parser.add_argument("--model-api-key", default=None)
    parser.add_argument("--model-temperature", type=float, default=0.2)
    parser.add_argument("--model-max-tokens", type=int, default=512)
    parser.add_argument("--model-timeout", type=int, default=120)
    parser.add_argument("--model-retries", type=int, default=3)
    parser.add_argument("--model-retry-backoff", type=float, default=10.0)
    parser.add_argument(
        "--model-reasoning-effort",
        default=None,
        help=(
            "Optional OpenAI-compatible reasoning_effort value. "
            "Use disable or the endpoint's lowest supported value when disabling model thinking."
        ),
    )
    parser.add_argument(
        "--mai-ui-model-name",
        default=None,
        help="Optional model name override for --agents mai-ui, e.g. a local vLLM-served MAI-UI-8B id.",
    )
    parser.add_argument(
        "--mobile-agent-model-name",
        default=None,
        help=(
            "Optional model name override for --agents mobile-agent-v35, "
            "e.g. a local vLLM-served GUI-Owl-1.5-8B-Instruct id."
        ),
    )
    parser.add_argument(
        "--decision-timeout",
        type=int,
        default=0,
        help=(
            "Optional hard wall-clock timeout around one agent decision. "
            "When exceeded, records a valid noop fallback step instead of hanging."
        ),
    )
    parser.add_argument(
        "--env-step-timeout",
        type=int,
        default=0,
        help=(
            "Optional hard wall-clock timeout around one BrowserGym env.step call. "
            "When exceeded, records a zero-reward failed step and restarts the environment."
        ),
    )
    parser.add_argument(
        "--model-context-tokens",
        type=int,
        default=0,
        help=(
            "Model context length in tokens. Default 0 infers from the tokenizer "
            f"or known model family ({QWEN_PLUS_CONTEXT_TOKENS} for Qwen Plus models)."
        ),
    )
    parser.add_argument(
        "--context-margin-tokens",
        type=int,
        default=2_048,
        help=(
            "Safety margin left unused in the model context window after reserving "
            "generation tokens. Keep this small to maximize ReAct history."
        ),
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=8,
        help="Fixed recent-history window for ablation agents such as react-history.",
    )
    parser.add_argument(
        "--history-budget-mode",
        choices=["token", "char"],
        default="token",
        help=(
            "Budgeting mode for default ReAct history. Token mode uses a tokenizer "
            "and the full chat prompt budget; char mode is kept for compatibility."
        ),
    )
    parser.add_argument(
        "--history-tokenizer-name",
        default=None,
        help=(
            "HuggingFace tokenizer name/path used for token budgeting. "
            f"Default infers {DEFAULT_QWEN_TOKENIZER} for Qwen models."
        ),
    )
    parser.add_argument(
        "--history-char-budget",
        type=int,
        default=120_000,
        help=(
            "Deprecated compatibility budget for char mode. Token mode ignores this "
            "for inclusion decisions but still records it in metadata."
        ),
    )
    parser.add_argument(
        "--history-observation-char-budget",
        type=int,
        default=0,
        help=(
            "Optional per-history-record observation character budget. "
            "Default 0 disables per-record truncation, so ReAct keeps full observations "
            "for included records and only drops oldest records when the full history budget is exceeded."
        ),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--compressor-model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-max-items", type=int, default=512)
    parser.add_argument(
        "--memory-builder",
        choices=["online_tokens", "delta_state", "online_delta_state", "hybrid"],
        default=os.environ.get("MEMORY_BUILDER", "online_tokens"),
    )
    parser.add_argument("--delta-rank", type=int, default=int(os.environ.get("DELTA_RANK", "8")))
    parser.add_argument("--delta-memory-slots", type=int, default=int(os.environ.get("DELTA_MEMORY_SLOTS", "8")))
    parser.add_argument("--delta-seed", type=int, default=int(os.environ.get("DELTA_SEED", "13")))
    parser.add_argument("--hybrid-recent-items", type=int, default=int(os.environ.get("HYBRID_RECENT_ITEMS", "32")))
    parser.add_argument("--monocart-html-report", action="store_true")
    return parser


def build_parser() -> argparse.ArgumentParser:
    return configure_parser(argparse.ArgumentParser(description="Run GUI baseline session-level evaluation."))


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = build_parser()
        args = parser.parse_args()
    result = run_baseline_eval(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
