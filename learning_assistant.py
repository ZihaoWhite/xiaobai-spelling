"""Dictionary grounding, NVIDIA generation, and local cache for learning cards."""

from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import certifi


DICTIONARY_ENDPOINT = "https://api.dictionaryapi.dev/api/v2/entries/en"
NVIDIA_CHAT_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
CACHE_VERSION = 1
HTTPS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


class LearningAssistantError(RuntimeError):
    """A user-facing learning assistant request failed safely."""


def empty_dictionary_result(word: str) -> dict[str, Any]:
    return {
        "found": False,
        "word": word,
        "phonetic": "",
        "part_of_speech": "",
        "definition_en": "",
        "example_en": "",
        "origin": "",
        "source_url": (
            f"{DICTIONARY_ENDPOINT}/{quote(word.strip(), safe='')}"
        ),
    }


def parse_dictionary_payload(payload: Any, word: str) -> dict[str, Any]:
    """Extract a small, stable subset from DictionaryAPI's response."""
    result = empty_dictionary_result(word)
    if not isinstance(payload, list) or not payload:
        return result
    entry = payload[0]
    if not isinstance(entry, dict):
        return result

    phonetic = str(entry.get("phonetic") or "").strip()
    if not phonetic:
        for item in entry.get("phonetics") or []:
            if isinstance(item, dict) and str(item.get("text") or "").strip():
                phonetic = str(item["text"]).strip()
                break

    part_of_speech = ""
    definition_en = ""
    example_en = ""
    for meaning in entry.get("meanings") or []:
        if not isinstance(meaning, dict):
            continue
        definitions = meaning.get("definitions") or []
        for definition in definitions:
            if not isinstance(definition, dict):
                continue
            candidate = str(definition.get("definition") or "").strip()
            if not candidate:
                continue
            part_of_speech = str(meaning.get("partOfSpeech") or "").strip()
            definition_en = candidate
            example_en = str(definition.get("example") or "").strip()
            break
        if definition_en:
            break

    result.update(
        {
            "found": bool(definition_en or phonetic or entry.get("origin")),
            "word": str(entry.get("word") or word).strip(),
            "phonetic": phonetic,
            "part_of_speech": part_of_speech,
            "definition_en": definition_en,
            "example_en": example_en,
            "origin": str(entry.get("origin") or "").strip(),
        }
    )
    return result


def _request_json(
    request: Request,
    *,
    timeout: float,
    urlopen_fn: Callable[..., Any] = urlopen,
) -> Any:
    try:
        with urlopen_fn(
            request,
            timeout=timeout,
            context=HTTPS_CONTEXT,
        ) as response:
            raw = response.read()
    except HTTPError:
        raise
    except (URLError, TimeoutError, OSError) as exc:
        raise LearningAssistantError("网络连接暂时不可用，请稍后再试。") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LearningAssistantError("学习服务返回了无法解析的数据。") from exc


def lookup_dictionary(
    word: str,
    *,
    timeout: float = 8.0,
    urlopen_fn: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Look up factual dictionary fields. A missing word is not an error."""
    url = f"{DICTIONARY_ENDPOINT}/{quote(word.strip(), safe='')}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "XiaobaiSpelling/1.0",
        },
    )
    try:
        payload = _request_json(
            request,
            timeout=timeout,
            urlopen_fn=urlopen_fn,
        )
    except HTTPError as exc:
        if exc.code == 404:
            return empty_dictionary_result(word)
        raise LearningAssistantError(
            f"词典服务暂时不可用（HTTP {exc.code}）。"
        ) from exc
    return parse_dictionary_payload(payload, word)


def build_learning_prompt(
    word: str,
    chinese_meaning: str,
    dictionary: dict[str, Any],
) -> str:
    """Build a constrained prompt that keeps facts separate from mnemonics."""
    verified = {
        key: dictionary.get(key, "")
        for key in (
            "phonetic",
            "part_of_speech",
            "definition_en",
            "example_en",
            "origin",
        )
    }
    return f"""
请为英语学习者制作一张简洁、实用的中英双语记忆卡。

目标单词：{word}
用户词表中的中文释义：{chinese_meaning}
外部词典核验信息：{json.dumps(verified, ensure_ascii=False)}

只返回一个 JSON 对象，不要 Markdown，不要解释过程，字段必须是：
{{
  "example_en": "8到18个词的自然英文例句",
  "example_zh": "准确、自然的中文翻译",
  "definition_en": "用简明自然的英语解释这个单词，不超过25个英文词",
  "usage_note": "一句中文用法提醒，指出常见搭配或语境",
  "spelling_tip": "一句中文拼写观察，强调容易漏写或写错的字母顺序",
  "mnemonic": "一句明确标注为联想的记忆提示，不冒充真实词源",
  "word_family": ["最多4个真实且常用的同族词"]
}}

严格要求：
1. 不得编造词根、词缀、历史来源或权威词源。
2. mnemonic 只能是帮助记忆的联想，必须使用“联想：”开头。
3. 如果无法确认同族词，word_family 返回空数组。
4. definition_en 必须直接解释目标单词，不能只重复例句。
5. 例句必须正确使用目标单词，难度适合中国英语学习者。
6. 不输出思考过程或 <think> 标签。
""".strip()


def parse_ai_card_text(content: str) -> dict[str, Any]:
    """Parse a model response defensively and normalize the expected fields."""
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        str(content),
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise LearningAssistantError("AI 没有返回完整的记忆卡，请重试。")
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LearningAssistantError("AI 返回的记忆卡格式不完整，请重试。") from exc
    if not isinstance(payload, dict):
        raise LearningAssistantError("AI 返回的记忆卡格式不正确。")

    normalized: dict[str, Any] = {}
    for key in (
        "example_en",
        "example_zh",
        "definition_en",
        "usage_note",
        "spelling_tip",
        "mnemonic",
    ):
        normalized[key] = str(payload.get(key) or "").strip()
    mnemonic = normalized["mnemonic"]
    if mnemonic and not mnemonic.startswith("联想："):
        normalized["mnemonic"] = f"联想：{mnemonic}"

    family = payload.get("word_family") or []
    if isinstance(family, str):
        family = [item.strip() for item in re.split(r"[,，;；]", family)]
    if not isinstance(family, list):
        family = []
    normalized["word_family"] = [
        str(item).strip()
        for item in family
        if str(item).strip()
    ][:4]

    if (
        not normalized["example_en"]
        or not normalized["example_zh"]
        or not normalized["definition_en"]
    ):
        raise LearningAssistantError("AI 返回的释义或例句不完整，请重试。")
    return normalized


def _friendly_nvidia_http_error(code: int) -> str:
    if code in (401, 403):
        return "NVIDIA API 密钥无效或没有模型权限。"
    if code == 404:
        return "该 NVIDIA 模型端点当前不可用。"
    if code == 429:
        return "NVIDIA API 已达到临时速率限制，请稍后再试。"
    if code >= 500:
        return "NVIDIA 服务暂时繁忙。"
    return f"NVIDIA 请求失败（HTTP {code}）。"


def request_nvidia_card(
    *,
    api_key: str,
    models: Sequence[str],
    word: str,
    chinese_meaning: str,
    dictionary: dict[str, Any],
    timeout: float = 30.0,
    urlopen_fn: Callable[..., Any] = urlopen,
) -> tuple[dict[str, Any], str]:
    """Try configured models in order and return the first valid card."""
    if not api_key.strip():
        raise LearningAssistantError("尚未配置 NVIDIA_API_KEY。")
    model_list = list(dict.fromkeys(model.strip() for model in models if model.strip()))
    if not model_list:
        raise LearningAssistantError("尚未配置可用的 NVIDIA 模型。")

    prompt = build_learning_prompt(word, chinese_meaning, dictionary)
    errors: list[str] = []
    for model in model_list:
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是严谨的英语学习助手。只输出指定 JSON，"
                        "不要编造词源，不要输出思考过程。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.6,
            "top_p": 0.8,
            "max_tokens": 800,
            "stream": False,
        }
        if "deepseek" in model.casefold():
            body["chat_template_kwargs"] = {"thinking": False}
        request = Request(
            NVIDIA_CHAT_ENDPOINT,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "XiaobaiSpelling/1.0",
            },
        )
        try:
            payload = _request_json(
                request,
                timeout=timeout,
                urlopen_fn=urlopen_fn,
            )
            choices = payload.get("choices") if isinstance(payload, dict) else None
            message = (
                choices[0].get("message", {})
                if isinstance(choices, list) and choices
                else {}
            )
            content = message.get("content") if isinstance(message, dict) else ""
            return parse_ai_card_text(str(content or "")), model
        except HTTPError as exc:
            errors.append(f"{model}：{_friendly_nvidia_http_error(exc.code)}")
        except LearningAssistantError as exc:
            errors.append(f"{model}：{exc}")

    detail = "；".join(errors)
    raise LearningAssistantError(
        f"主模型和备用模型都没有生成成功。{detail}"
    )


def cache_path_for_word(cache_dir: Path, word: str) -> Path:
    normalized = word.strip().casefold()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{digest}.json"


def load_cached_bundle(cache_dir: Path, word: str) -> dict[str, Any] | None:
    path = cache_path_for_word(cache_dir, word)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("cache_version") != CACHE_VERSION:
        return None
    if str(payload.get("word") or "").casefold() != word.strip().casefold():
        return None
    return payload


def save_cached_bundle(
    cache_dir: Path,
    word: str,
    bundle: dict[str, Any],
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_path_for_word(cache_dir, word)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".learning_",
            suffix=".tmp",
            dir=cache_dir,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(bundle, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        return target
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def generate_learning_bundle(
    *,
    word: str,
    chinese_meaning: str,
    api_key: str,
    models: Sequence[str],
    cache_dir: Path,
    dictionary_timeout: float = 8.0,
    nvidia_timeout: float = 30.0,
    urlopen_fn: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Generate one cached card; dictionary failure never blocks AI generation."""
    dictionary_error = ""
    try:
        dictionary = lookup_dictionary(
            word,
            timeout=dictionary_timeout,
            urlopen_fn=urlopen_fn,
        )
    except LearningAssistantError as exc:
        dictionary = empty_dictionary_result(word)
        dictionary_error = str(exc)

    ai_card, used_model = request_nvidia_card(
        api_key=api_key,
        models=models,
        word=word,
        chinese_meaning=chinese_meaning,
        dictionary=dictionary,
        timeout=nvidia_timeout,
        urlopen_fn=urlopen_fn,
    )
    bundle = {
        "cache_version": CACHE_VERSION,
        "word": word.strip(),
        "chinese_meaning": chinese_meaning.strip(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": used_model,
        "dictionary": dictionary,
        "dictionary_error": dictionary_error,
        "ai": ai_card,
    }
    save_cached_bundle(cache_dir, word, bundle)
    return bundle
