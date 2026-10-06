from __future__ import annotations

import atexit
import socket
import subprocess
import time
from pathlib import Path

import httpx

from openjev_ja.common.types import ScoreResult
from openjev_ja.methods.decider.scorer import build_questions, parse_answer

PRIMITIVES = ("noul", "choice", "score")

# variant -> (Hub repo, revision, file). ggml-org's own conversion of Cloudflare/clef-flash;
# its decision head is stored at Q8_0 in every quantized file, the backbone is quantized.
RELEASES: dict[str, tuple[str, str, str]] = {
    "q4_k_m": ("ggml-org/Clef-Flash-GGUF", "4a192915ef971886004b5b13294f2b4c7a7fc39d", "Clef-Flash-Q4_K_M.gguf"),
    "q8_0": ("ggml-org/Clef-Flash-GGUF", "4a192915ef971886004b5b13294f2b4c7a7fc39d", "Clef-Flash-Q8_0.gguf"),
    "bf16": ("ggml-org/Clef-Flash-GGUF", "4a192915ef971886004b5b13294f2b4c7a7fc39d", "Clef-Flash-BF16.gguf"),
}
DEFAULT_SERVER = "models/llama.cpp/b11430/llama-server.exe"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ClefGgufScorer:
    """Cloudflare/clef-flash as a GGUF, served by llama.cpp's `llama-server` through its
    `/v1/systemone` endpoint (decision-model support merged in ggml-org/llama.cpp#29831,
    release b11430 or later). The same wire format as ClefScorer / DeciderScorer is posted
    to the server; this scorer starts one `llama-server` subprocess for its lifetime and
    stops it on close()/interpreter exit.
    """

    name = "clef-gguf"

    def __init__(
        self,
        model_name: str = "q4_k_m",
        *,
        primitive: str,
        device: str = "cuda",
        server_path: str = DEFAULT_SERVER,
        n_gpu_layers: int = 99,
        context_size: int = 32768,
        batch_size: int = 16384,
        startup_timeout_s: float = 300.0,
        model_id: str | None = None,
    ) -> None:
        if primitive not in PRIMITIVES:
            raise ValueError(f"unsupported primitive: {primitive}")
        if model_name not in RELEASES:
            raise ValueError(f"unknown clef GGUF variant {model_name!r}; choose one of {sorted(RELEASES)}")
        server = Path(server_path)
        if not server.exists():
            raise RuntimeError(
                f"llama-server not found at {server}: download a llama.cpp release >= b11430 "
                "(Windows CUDA build) and point `server_path` at its llama-server executable"
            )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Install the clef extra: pip install -e '.[clef]'") from exc
        self.model_name = model_name
        self.repo_id, self.revision, filename = RELEASES[model_name]
        self.primitive = primitive
        self.device = device
        self.model_id = model_id or f"clef-flash-gguf-{model_name}"
        self.gguf_path = hf_hub_download(self.repo_id, filename, revision=self.revision)
        self.port = _free_port()
        self._process = subprocess.Popen(
            [
                str(server), "-m", self.gguf_path, "-ngl", str(n_gpu_layers),
                "-c", str(context_size), "--parallel", "1",
                # a decision model reads the whole prompt in one non-causal pass, so the physical
                # batch must hold the longest prompt (the default 512 returns HTTP 500 on
                # prompts of ~1300 tokens, e.g. long MMMLU items).
                "-b", str(batch_size), "-ub", str(batch_size),
                "--host", "127.0.0.1", "--port", str(self.port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(self.close)
        self._client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=300.0)
        self._wait_ready(startup_timeout_s)

    def _wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"llama-server exited early (code {self._process.returncode})")
            try:
                if self._client.get("/health").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
        self.close()
        raise RuntimeError("llama-server did not become ready in time")

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()

    def __del__(self) -> None:
        self.close()

    def score(self, question: str, options: list[str]) -> ScoreResult:
        state, questions = build_questions(self.primitive, question, options)
        started = time.perf_counter()
        response = self._client.post(
            "/v1/systemone", json={"model": self.model_id, "state": state, "questions": questions}
        )
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        body = response.json()
        answer = body["answers"]["answer"]
        scores, probabilities, predicted = parse_answer(self.primitive, answer, len(options))
        return ScoreResult(
            scores=scores,
            probabilities=probabilities,
            predicted_index=predicted,
            latency_ms=latency_ms,
            metadata={"primitive": self.primitive, "usage": body.get("usage", {})},
        )

    def metadata(self) -> dict[str, object]:
        return {
            "scorer": self.name,
            "model": self.model_id,
            "release": f"{self.repo_id}@{self.revision}",
            "gguf": Path(self.gguf_path).name,
            "base_model": "Cloudflare/clef-flash",
            "primitive": self.primitive,
            "device": self.device,
            "runtime": "llama.cpp llama-server /v1/systemone",
            "generation": False,
        }
