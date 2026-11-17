"""Builds one client per role from config and assembles the role wrappers."""

from __future__ import annotations

from functools import cached_property

from .config import Config
from .llm import LLMClient, Responder, TraceSink, make_client
from .roles import Coder, Dispatcher, Extractor, Planner, RepoQA


class ModelRegistry:
    """Owns the per-role clients and exposes ready-to-use role wrappers.

    In real mode each role gets an :class:`OpenAICompatClient` pointed at its
    server (roles that ``share`` a model point at the same server). In mock mode
    every role shares one ``responder`` (typically a
    :class:`~coding_agent.mock_brain.MockBrain`).
    """

    def __init__(
        self,
        config: Config,
        responder: Responder | None = None,
        tracer: TraceSink | None = None,
    ):
        self.config = config
        self.mock = config.mock
        if self.mock and responder is None:
            raise ValueError(
                "mock mode requires a responder (e.g. a MockBrain); "
                "pass one to ModelRegistry(...)."
            )
        self._responder = responder
        self.tracer = tracer
        self._clients: dict[str, LLMClient] = {}

    def client(self, role: str) -> LLMClient:
        if role not in self._clients:
            role_cfg = self.config.role(role)
            self._clients[role] = make_client(
                role_cfg, mock=self.mock, responder=self._responder, tracer=self.tracer
            )
        return self._clients[role]

    # Role wrappers are created once and reused (tests may monkeypatch them).
    @cached_property
    def planner(self) -> Planner:
        return Planner(self.client("planner"))

    @cached_property
    def coder(self) -> Coder:
        return Coder(self.client("coder"))

    @cached_property
    def dispatcher(self) -> Dispatcher:
        return Dispatcher(self.client("dispatcher"))

    @cached_property
    def extractor(self) -> Extractor:
        return Extractor(self.client("extractor"))

    @cached_property
    def rag(self) -> RepoQA:
        return RepoQA(self.client("rag"))

    def has_role(self, role: str) -> bool:
        return role in self.config.roles
