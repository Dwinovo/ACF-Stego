import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)


class RemoteAgent:
    """
    通用远程用户 Agent。
    - 只负责发送消息并返回模型回复文本。
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        retry_delay: float = 1.0,
    ) -> None:
        if not api_key:
            raise ValueError("缺少 OPENAI_API_KEY，无法创建 Gemini 模拟用户 Agent。")
        kwargs = {"api_key": api_key, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model_name
        self.retry_delay = max(0.0, float(retry_delay))

    def invoke(self, messages: list[dict[str, Any]], temperature: float = 1.0) -> str:
        """
        发送一组 OpenAI 兼容 chat messages，返回文本回复。
        """
        request_messages = messages
        # 工程约束：当前远程模型通常只会在「最后一条是 user」时继续回复。
        # 如果上下文以 assistant 结尾，模型大概率不会输出内容。
        # 因此这里在发送前做一次 role 互换（user <-> assistant），
        # 让远程模型把该轮输入视作来自 user，从而正常返回回复。
        if messages and str(messages[-1].get("role", "")).lower() == "assistant":
            request_messages = []
            for message in messages:
                swapped = dict(message)
                role = str(swapped.get("role", "")).lower()
                if role == "user":
                    swapped["role"] = "assistant"
                elif role == "assistant":
                    swapped["role"] = "user"
                request_messages.append(swapped)

        retry_count = 0
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=request_messages,
                    temperature=temperature,
                )
                choices = getattr(resp, "choices", None)
                if not choices:
                    retry_count += 1
                    continue

                return (choices[0].message.content or "").strip()
            except APIStatusError as exc:
                status_code = getattr(exc, "status_code", None)
                retryable_status = status_code is None or status_code == 429 or status_code >= 500
                if not retryable_status:
                    raise
                retry_count += 1
            except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError):
                retry_count += 1

            print(f"[UserAgent] retry_count={retry_count}")
            if self.retry_delay > 0:
                time.sleep(self.retry_delay)
