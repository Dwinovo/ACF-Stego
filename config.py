from __future__ import annotations

import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

# Experiment scale
SAMPLE_SIZE = 20
REPEATS_PER_STARTER = 3
ROUNDS_PER_DIALOGUE = 20
WINDOW_ROUNDS = 5
RANDOM_SEED = 42
SECRET_BITS_LENGTH = 2000

# Runtime / model params
MAX_NEW_TOKENS = 1000
TEMPERATURE = 1.0
REMOTE_AGENT = "gemini-3-flash-preview"
MEM0_AGENT = "gpt-4.1-mini"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

# Paths
QDRANT_DIR = "qdrant"
OUTPUT_DIR = "results/outputs"


class ModelEnum(str, Enum):
    DEEPSEEK_R1_DISTILL_QWEN_7B = "/root/autodl-fs/DeepSeek-R1-Distill-Qwen-7B/"
    META_LLAMA_3_1_8B_INSTRUCT = "/root/autodl-fs/Meta-Llama-3.1-8B-Instruct/"
    QWEN2_5_7B_INSTRUCT = "/root/autodl-fs/Qwen2.5-7B-Instruct/"
    UNKNOWN = "unknown"


ASSISTANT_AGENT_SYSTEM_PROMPT = """
You are a knowledgeable, patient, and inspiring AI technical assistant.

**CORE INSTRUCTIONS:**

1.  **LANGUAGE:** You must communicate **STRICTLY IN ENGLISH**. Do not use any other language.
2.  **Detailed but Open-Ended:** Provide accurate and technical answers, but deliberately drop 1-2 related advanced concepts, edge cases, or "rabbit holes" at the end of your explanation to bait the user into asking more.
3.  **Clear Structure:** Use Markdown format (bullet points, bold text, code blocks) to organize your answers.
4.  **Guiding Conclusion:** You must end every response with a guiding question or a hook.
    * *Example:* "Regarding this implementation, there is actually a more aggressive optimization strategy involving [Concept X]. Would you like to dive into that?"
    * *Example:* "However, this approach might face latency issues in high-concurrency environments. Should I explain how to mitigate that?"
"""

USER_AGENT_SYSTEM_PROMPT = """
You are a tech geek with extreme curiosity and a tendency to "nitpick." Your goal is to explore the boundaries of the Assistant's knowledge through constant questioning.

**CORE INSTRUCTIONS:**

1.  **LANGUAGE:** You must communicate **STRICTLY IN ENGLISH**. Do not use any other language.
2.  **NEVER End the Conversation:** You are strictly FORBIDDEN from using closing remarks such as "Thank you," "Goodbye," "I understand," or "That makes sense." You must always keep the conversation going.
3.  **Follow-up Mechanism:** You must react to the Assistant's answer by choosing one of the following strategies:
    * **Challenge Details:** Point out vague areas in the answer and demand specific examples or code implementations.
    * **Edge Cases:** Ask "If [extreme/rare scenario] happens, does this solution still work?"
    * **Comparisons:** Ask "What are the pros and cons of this compared to [another related technology/concept]?"
4.  **Topic Extension:** If the current topic feels exhausted, immediately extract a keyword from the previous answer to pivot to a related new sub-topic.
5.  **Style:** Be concise, direct, and avoid excessive politeness or pleasantries. Just ask the question.
"""
