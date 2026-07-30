import json
from urllib.error import HTTPError

from learning_assistant import (
    build_learning_prompt,
    load_cached_bundle,
    parse_ai_card_text,
    parse_dictionary_payload,
    request_nvidia_card,
    save_cached_bundle,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


def test_parse_dictionary_payload_extracts_grounded_fields() -> None:
    payload = [
        {
            "word": "hello",
            "phonetic": "həˈləʊ",
            "origin": "early 19th century",
            "meanings": [
                {
                    "partOfSpeech": "exclamation",
                    "definitions": [
                        {
                            "definition": "used as a greeting",
                            "example": "Hello there!",
                        }
                    ],
                }
            ],
        }
    ]
    result = parse_dictionary_payload(payload, "hello")
    assert result["found"] is True
    assert result["phonetic"] == "həˈləʊ"
    assert result["origin"] == "early 19th century"
    assert result["definition_en"] == "used as a greeting"


def test_parse_ai_card_text_strips_thinking_and_code_fence() -> None:
    content = """
    <think>private reasoning</think>
    ```json
    {
      "example_en": "The young bird is still unfledged.",
      "example_zh": "这只幼鸟还没有长出羽毛。",
      "definition_en": "Not yet fully developed or experienced.",
      "usage_note": "常用于描述尚未成熟的人或事物。",
      "spelling_tip": "注意中间的 fl 连写。",
      "mnemonic": "把 fledged 想成羽毛长成。",
      "word_family": "fledge, fledgling"
    }
    ```
    """
    card = parse_ai_card_text(content)
    assert card["example_en"].startswith("The young bird")
    assert card["definition_en"].startswith("Not yet")
    assert card["mnemonic"].startswith("联想：")
    assert card["word_family"] == ["fledge", "fledgling"]


def test_prompt_forbids_invented_etymology() -> None:
    prompt = build_learning_prompt(
        "unfledged",
        "未长羽毛的",
        {"origin": "", "definition_en": "not having feathers"},
    )
    assert "不得编造词根" in prompt
    assert "不冒充真实词源" in prompt


def test_cache_round_trip(tmp_path) -> None:
    bundle = {
        "cache_version": 1,
        "word": "Apple",
        "model": "test/model",
        "dictionary": {},
        "ai": {"example_en": "I ate an apple."},
    }
    save_cached_bundle(tmp_path, "Apple", bundle)
    assert load_cached_bundle(tmp_path, "apple") == bundle


def test_nvidia_request_falls_back_after_unavailable_primary() -> None:
    requests = []

    def fake_urlopen(request, timeout, **kwargs):
        requests.append(request)
        body = json.loads(request.data.decode("utf-8"))
        if body["model"] == "qwen/deprecated":
            raise HTTPError(request.full_url, 404, "not found", None, None)
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "example_en": "The bird is still unfledged.",
                                    "example_zh": "这只鸟尚未长出羽毛。",
                                    "definition_en": "Not yet fully developed.",
                                    "usage_note": "可形容尚不成熟。",
                                    "spelling_tip": "注意双写与字母顺序。",
                                    "mnemonic": "联想：羽毛还没长全。",
                                    "word_family": ["fledge"],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )

    card, model = request_nvidia_card(
        api_key="secret",
        models=("qwen/deprecated", "deepseek-ai/deepseek-v4-flash"),
        word="unfledged",
        chinese_meaning="未长羽毛的",
        dictionary={},
        urlopen_fn=fake_urlopen,
    )
    assert model == "deepseek-ai/deepseek-v4-flash"
    assert card["example_zh"]
    fallback_body = json.loads(requests[-1].data.decode("utf-8"))
    assert fallback_body["chat_template_kwargs"] == {"thinking": False}
