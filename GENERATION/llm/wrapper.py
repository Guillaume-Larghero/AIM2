"""LLM wrapper supporting OpenAI and Anthropic APIs for report generation."""

import os
import time
import logging
from typing import Dict, List
from abc import ABC, abstractmethod

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        pass


class OpenAILLM(BaseLLM):
    def __init__(self, api_key=None, model_name="gpt-4o-mini", max_tokens=512,
                 temperature=0.0, top_p=1.0, frequency_penalty=0.0,
                 presence_penalty=0.0, timeout=60, max_retries=3):
        if OpenAI is None:
            raise ImportError("openai package not installed. Install with: pip install openai")

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY)")

        self.client = OpenAI(api_key=key, timeout=timeout)
        self.model = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_retries = max_retries
        logger.info(f"Initialized OpenAI LLM: {model_name}")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        params = {
            'model': kwargs.get('model', self.model),
            'messages': messages,
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'top_p': kwargs.get('top_p', self.top_p),
            'frequency_penalty': kwargs.get('frequency_penalty', self.frequency_penalty),
            'presence_penalty': kwargs.get('presence_penalty', self.presence_penalty),
        }
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError("Failed after maximum retries")


class AnthropicLLM(BaseLLM):
    def __init__(self, api_key=None, model_name="claude-3-5-sonnet-20241022",
                 max_tokens=512, temperature=0.0, top_p=1.0, timeout=60, max_retries=3):
        if Anthropic is None:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Anthropic API key required (set ANTHROPIC_API_KEY)")

        self.client = Anthropic(api_key=key, timeout=timeout)
        self.model = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        logger.info(f"Initialized Anthropic LLM: {model_name}")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        system_message = None
        filtered_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                filtered_messages.append(msg)

        params = {
            'model': kwargs.get('model', self.model),
            'messages': filtered_messages,
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'top_p': kwargs.get('top_p', self.top_p),
        }
        if system_message:
            params['system'] = system_message

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(**params)
                return response.content[0].text
            except Exception as e:
                logger.warning(f"Anthropic API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError("Failed after maximum retries")


class LLMWrapper:
    """Unified LLM wrapper that selects provider based on config."""

    def __init__(self, config):
        provider = config.generation.LLM_PROVIDER.lower()
        model_name = config.generation.LLM_MODEL

        if provider == "openai":
            self.llm = OpenAILLM(
                api_key=os.getenv(config.generation.API_KEY_ENV),
                model_name=model_name,
                max_tokens=config.generation.MAX_TOKENS,
                temperature=config.generation.TEMPERATURE,
                top_p=config.generation.TOP_P,
                frequency_penalty=config.generation.FREQUENCY_PENALTY,
                presence_penalty=config.generation.PRESENCE_PENALTY,
                timeout=config.generation.API_TIMEOUT,
                max_retries=config.generation.MAX_RETRIES
            )
        elif provider == "anthropic":
            self.llm = AnthropicLLM(
                api_key=os.getenv(config.generation.API_KEY_ENV),
                model_name=model_name,
                max_tokens=config.generation.MAX_TOKENS,
                temperature=config.generation.TEMPERATURE,
                top_p=config.generation.TOP_P,
                timeout=config.generation.API_TIMEOUT,
                max_retries=config.generation.MAX_RETRIES
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        self._provider = provider
        self._model = model_name
        logger.info(f"LLM Wrapper initialized: {provider}/{model_name}")

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.llm.generate(messages, **kwargs)

    def generate_complete_report(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, str]:
        """Generate and parse into findings/impression sections."""
        full_report = self.generate(system_prompt, user_prompt, **kwargs)
        return self._parse_report_sections(full_report)

    def _parse_report_sections(self, report_text: str) -> Dict[str, str]:
        lines = report_text.strip().split('\n')
        current_section = None
        findings_lines = []
        impression_lines = []

        for line in lines:
            line_lower = line.lower().strip()
            if 'findings:' in line_lower or line_lower.startswith('findings'):
                current_section = 'findings'
                line = line.split(':', 1)[-1].strip()
                if line:
                    findings_lines.append(line)
                continue
            elif 'impression:' in line_lower or line_lower.startswith('impression'):
                current_section = 'impression'
                line = line.split(':', 1)[-1].strip()
                if line:
                    impression_lines.append(line)
                continue

            if current_section == 'findings' and line.strip():
                findings_lines.append(line)
            elif current_section == 'impression' and line.strip():
                impression_lines.append(line)
            elif current_section is None and line.strip():
                findings_lines.append(line)

        findings = '\n'.join(findings_lines).strip()
        impression = '\n'.join(impression_lines).strip()
        if not findings:
            findings = report_text.strip()

        return {'findings': findings, 'impression': impression}

    def __repr__(self):
        return f"LLMWrapper(provider={self._provider}, model={self._model})"