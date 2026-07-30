"""Real API execution and immutable delivery for P4-API-R2B v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import run_p4_api_r2b_20_val_smoke as core
from utils.cae_intent import validate_intent_result
from utils.cae_language_grounding.llm_adapter_r0 import (
    LLMProtocolError,
    LLMTransportError,
    _default_transport,
)
from utils.cae_language_grounding.reliable_runner import AtomicJSONCache
from utils.cae_query_frame import QueryFrameCompileError, compile_query_frame, validate_query_frame


R2BError = core.R2BError


class RequestLedger(core.RequestLedger):
    def completed(self, request_hash: str) -> bool:
        return any(
            row["request_hash"] == request_hash and row.get("completed") is True
            for row in self.value["entries"]
        )

    def complete(
        self,
        *,
        request_hash: str,
        status: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        rows = [row for row in self.value["entries"] if row["request_hash"] == request_hash]
        if len(rows) != 1:
            raise R2BError("R2B_E_LEDGER_REQUEST_IDENTITY")
        row = rows[0]
        if row.get("completed"):
            if (
                row.get("status") != status
                or row.get("input_tokens", 0) != input_tokens
                or row.get("output_tokens", 0) != output_tokens
            ):
                raise R2BError("R2B_E_LEDGER_COMPLETION_CONFLICT")
            return
        row.update(
            completed=True,
            status=status,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
        )
        counters = self.value["counters"]
        counters["input_tokens"] = counters.get("input_tokens", 0) + int(input_tokens)
        counters["output_tokens"] = counters.get("output_tokens", 0) + int(output_tokens)
        core._atomic_write_json(self.path, self.value)


def _request_hash(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _strict_object(raw: str) -> dict[str, Any]:
    duplicates: list[str] = []

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                duplicates.append(str(key))
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=pairs_hook)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LLMProtocolError("QF_E_JSON_TYPE") from exc
    if duplicates or not isinstance(value, dict):
        raise LLMProtocolError("QF_E_JSON_TYPE")
    return value


def _schema_summary(error_code: str, field_path: str) -> dict[str, Any]:
    return {
        "field_path": field_path if field_path.startswith("/") else "/",
        "required": [],
        "allowed_enum": [],
        "rule": error_code,
    }


class R2BClient:
    def __init__(
        self,
        *,
        api_key: str,
        ledger: RequestLedger,
        cache_dir: Path,
        transport=None,
        timeout_seconds: float = 120.0,
    ) -> None:
        if not api_key:
            raise R2BError("R2B_E_API_KEY_MISSING")
        self._api_key = api_key
        self.ledger = ledger
        self.cache = AtomicJSONCache(cache_dir)
        self.transport = transport or _default_transport
        self.timeout_seconds = timeout_seconds

    def _fetch(
        self,
        *,
        sample_id: str,
        stage: str,
        request: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], bool, str]:
        request_hash = _request_hash(request)
        namespace = f"p4_api_r2b_{stage}"
        cached = self.cache.get(namespace, request_hash)
        if cached is not None:
            if not self.ledger.completed(request_hash):
                raise R2BError("R2B_E_CACHE_LEDGER_IDENTITY")
            response = cached.get("response")
            if not isinstance(response, Mapping):
                raise R2BError("R2B_E_CACHE_RESPONSE")
            return response, True, request_hash
        self.ledger.reserve(sample_id=sample_id, stage=stage, request_hash=request_hash)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        try:
            response = self.transport(
                str(request["endpoint"]),
                headers,
                request["payload"],
                self.timeout_seconds,
            )
        except Exception:
            self.ledger.complete(request_hash=request_hash, status="transport_error")
            raise
        if not isinstance(response, Mapping):
            self.ledger.complete(request_hash=request_hash, status="protocol_error")
            raise LLMProtocolError("E_RUNTIME_RESPONSE_TYPE")
        usage = response.get("usage")
        input_tokens = output_tokens = 0
        if isinstance(usage, Mapping):
            try:
                input_tokens = int(usage.get("prompt_tokens", 0))
                output_tokens = int(usage.get("completion_tokens", 0))
            except (TypeError, ValueError):
                pass
        self.ledger.complete(
            request_hash=request_hash,
            status="response_received",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.cache.put(namespace, request_hash, {"response": dict(response)})
        return response, False, request_hash

    @staticmethod
    def _decode(response: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            choice = response["choices"][0]
            raw = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProtocolError("E_RUNTIME_RESPONSE_SHAPE") from exc
        if finish_reason == "length":
            raise LLMProtocolError("E_RUNTIME_TRUNCATED")
        if not isinstance(raw, str) or not raw.strip():
            raise LLMProtocolError("E_RUNTIME_EMPTY_CONTENT")
        actual_model = str(response.get("model") or "")
        if actual_model != core.AUTHORIZED_MODEL:
            raise LLMProtocolError("R2B_E_ACTUAL_MODEL_MISMATCH")
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            raise LLMProtocolError("E_RUNTIME_USAGE_MISSING")
        frame = _strict_object(raw)
        return frame, {
            "actual_model": actual_model,
            "finish_reason": finish_reason,
            "input_tokens": int(usage["prompt_tokens"]),
            "output_tokens": int(usage["completion_tokens"]),
        }

    @staticmethod
    def _validate_compile(frame: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        validation = validate_query_frame(frame)
        if not validation["valid"]:
            first = validation["errors"][0]
            raise QueryFrameCompileError(
                str(first["error_code"]), str(first["field_path"]), str(first["message"])
            )
        compiled = compile_query_frame(frame)
        validation = validate_intent_result(compiled)
        if not validation["valid"]:
            first = validation["errors"][0]
            raise QueryFrameCompileError(
                "QF_E_CANONICAL_REJECTED",
                str(first.get("field_path") or "/"),
                str(first.get("message") or "P2 validation failed"),
            )
        return dict(frame), compiled

    def infer(self, *, sample_id: str, inference_input: Mapping[str, Any]) -> dict[str, Any]:
        outcome: dict[str, Any] = {
            "sample_id": sample_id,
            "first_round_json_parse_valid": False,
            "first_round_frame_schema_valid": False,
            "post_repair_frame_valid": False,
            "p2_valid": False,
            "repair_count": 0,
            "cache_hit": False,
            "failure_phase": None,
            "failure_code": None,
            "error_path": None,
            "frame": None,
            "compiled_result": None,
            "requests": [],
        }
        try:
            response, cache_hit, request_hash = self._fetch(
                sample_id=sample_id,
                stage="first_round",
                request=core.build_request(inference_input),
            )
            outcome["cache_hit"] = cache_hit
            outcome["requests"].append(
                {"stage": "first_round", "request_hash": request_hash, "cache_hit": cache_hit}
            )
            frame, metadata = self._decode(response)
            outcome["first_round_json_parse_valid"] = True
            frame, compiled = self._validate_compile(frame)
            outcome.update(
                first_round_frame_schema_valid=True,
                post_repair_frame_valid=True,
                p2_valid=True,
                frame=frame,
                compiled_result=compiled,
                metadata=metadata,
            )
            return outcome
        except LLMTransportError as exc:
            outcome.update(failure_phase="transport", failure_code=exc.code)
            return outcome
        except LLMProtocolError as exc:
            code = str(exc.code)
            if code != "QF_E_JSON_TYPE":
                outcome.update(failure_phase="protocol", failure_code=code)
                return outcome
            phase, path = "json_parse", "/"
        except QueryFrameCompileError as exc:
            phase, code, path = "frame_validation", exc.code, exc.field_path

        if not core.repair_allowed(phase, code):
            outcome.update(failure_phase=phase, failure_code=code, error_path=path)
            return outcome
        outcome["repair_count"] = 1
        try:
            response, cache_hit, request_hash = self._fetch(
                sample_id=sample_id,
                stage="repair",
                request=core.build_request(
                    inference_input, repair=(code, _schema_summary(code, path))
                ),
            )
            outcome["requests"].append(
                {"stage": "repair", "request_hash": request_hash, "cache_hit": cache_hit}
            )
            frame, metadata = self._decode(response)
            frame, compiled = self._validate_compile(frame)
            outcome.update(
                post_repair_frame_valid=True,
                p2_valid=True,
                frame=frame,
                compiled_result=compiled,
                metadata=metadata,
                failure_phase=None,
                failure_code=None,
                error_path=None,
            )
        except LLMTransportError as exc:
            outcome.update(failure_phase="transport", failure_code=exc.code)
        except LLMProtocolError as exc:
            outcome.update(failure_phase="repair_protocol", failure_code=str(exc.code))
        except QueryFrameCompileError as exc:
            outcome.update(
                failure_phase="repair_frame_validation",
                failure_code=exc.code,
                error_path=exc.field_path,
            )
        return outcome

