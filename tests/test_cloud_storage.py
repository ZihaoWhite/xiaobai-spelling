import json
from urllib.error import HTTPError

import pandas as pd
import pytest

from cloud_storage import (
    CloudConflictError,
    SupabaseConfig,
    SupabaseStorage,
    cloud_id_from_ref,
    cloud_ref,
    dataframe_from_payload,
    dataframe_to_payload,
    is_cloud_ref,
    learning_rows_from_dataframe,
)


class FakeResponse:
    def __init__(self, payload):
        if payload is None:
            self.payload = b""
        else:
            self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


def test_cloud_reference_round_trip() -> None:
    value = cloud_ref("abc-123")
    assert is_cloud_ref(value)
    assert cloud_id_from_ref(value) == "abc-123"


def test_dataframe_payload_preserves_index_and_chinese_columns() -> None:
    frame = pd.DataFrame(
        {"单词": ["apple"], "当天正确": [2]},
        index=[9],
    )
    restored = dataframe_from_payload(dataframe_to_payload(frame))
    assert list(restored.index) == [9]
    assert restored.at[9, "单词"] == "apple"
    assert restored.at[9, "当天正确"] == 2


def test_list_word_lists_does_not_expose_key_in_record() -> None:
    def fake_urlopen(request, timeout, context):
        assert request.headers["Apikey"] == "server-secret"
        return FakeResponse(
            [
                {
                    "id": "list-1",
                    "source_name": "today.csv",
                    "source_date": "2026-07-29",
                    "encoding": "utf-8-sig",
                    "row_count": 3,
                    "revision": 2,
                    "updated_at": "2026-07-29T12:00:00+00:00",
                }
            ]
        )

    storage = SupabaseStorage(
        SupabaseConfig("https://example.supabase.co", "server-secret"),
        urlopen_fn=fake_urlopen,
    )
    records = storage.list_word_lists()
    assert records[0]["path"] == "cloud://list-1"
    assert records[0]["parent"] == "Supabase 云端"
    assert "server-secret" not in json.dumps(records, ensure_ascii=False)


def test_optimistic_save_reports_conflict_when_no_row_matches() -> None:
    def fake_urlopen(request, timeout, context):
        return FakeResponse([])

    storage = SupabaseStorage(
        SupabaseConfig("https://example.supabase.co", "server-secret"),
        urlopen_fn=fake_urlopen,
    )
    with pytest.raises(CloudConflictError):
        storage.save_word_list(
            record_id="list-1",
            dataframe=pd.DataFrame({"单词": ["apple"]}),
            expected_revision=2,
        )


def test_delete_word_list_requests_exact_cloud_record() -> None:
    captured = {}

    def fake_urlopen(request, timeout, context):
        captured["method"] = request.method
        captured["url"] = request.full_url
        return FakeResponse([{"id": "list-1"}])

    storage = SupabaseStorage(
        SupabaseConfig("https://example.supabase.co", "server-secret"),
        urlopen_fn=fake_urlopen,
    )
    storage.delete_word_list("list-1")
    assert captured["method"] == "DELETE"
    assert "id=eq.list-1" in captured["url"]


def test_http_auth_error_is_user_friendly() -> None:
    def fake_urlopen(request, timeout, context):
        raise HTTPError(request.full_url, 401, "bad key", None, None)

    storage = SupabaseStorage(
        SupabaseConfig("https://example.supabase.co", "server-secret"),
        urlopen_fn=fake_urlopen,
    )
    with pytest.raises(Exception, match="密钥无效"):
        storage.list_word_lists()


def test_learning_rows_merge_duplicates_and_skip_unpracticed() -> None:
    frame = pd.DataFrame(
        [
            {
                "单词": "Apple",
                "中文释义": "苹果",
                "类型": "新学",
                "当前状态": 1,
                "当天答题次数": 2,
                "当天正确": 1,
                "当天错误": 1,
            },
            {
                "单词": "apple",
                "中文释义": "",
                "类型": "",
                "当前状态": 2,
                "当天答题次数": 1,
                "当天正确": 1,
                "当天错误": 0,
            },
            {
                "单词": "banana",
                "中文释义": "香蕉",
                "类型": "新学",
                "当前状态": 0,
                "当天答题次数": 0,
                "当天正确": 0,
                "当天错误": 0,
            },
        ]
    )
    rows = learning_rows_from_dataframe(
        frame,
        word_list_id="list-1",
        source_name="today.csv",
        source_date="2026-07-29",
    )
    assert len(rows) == 1
    assert rows[0]["normalized_word"] == "apple"
    assert rows[0]["attempts"] == 3
    assert rows[0]["correct"] == 2
    assert rows[0]["wrong"] == 1
    assert rows[0]["current_status"] == 2


def test_sync_learning_snapshot_uses_atomic_rpc() -> None:
    captured = {}

    def fake_urlopen(request, timeout, context):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(1)

    storage = SupabaseStorage(
        SupabaseConfig("https://example.supabase.co", "server-secret"),
        urlopen_fn=fake_urlopen,
    )
    count = storage.sync_learning_snapshot(
        word_list_id="list-1",
        source_name="today.csv",
        source_date="2026-07-29",
        dataframe=pd.DataFrame(
            [
                {
                    "单词": "apple",
                    "中文释义": "苹果",
                    "类型": "新学",
                    "当前状态": 1,
                    "当天答题次数": 1,
                    "当天正确": 1,
                    "当天错误": 0,
                }
            ]
        ),
    )
    assert count == 1
    assert "/rest/v1/rpc/xb_sync_daily_words" in captured["url"]
    assert captured["body"]["p_rows"][0]["normalized_word"] == "apple"


def test_list_learned_words_reads_aggregate_view() -> None:
    def fake_urlopen(request, timeout, context):
        assert "/rest/v1/xb_learned_words" in request.full_url
        return FakeResponse(
            [
                {
                    "normalized_word": "apple",
                    "word": "apple",
                    "total_attempts": 3,
                    "has_ai_card": True,
                }
            ]
        )

    storage = SupabaseStorage(
        SupabaseConfig("https://example.supabase.co", "server-secret"),
        urlopen_fn=fake_urlopen,
    )
    rows = storage.list_learned_words()
    assert rows[0]["word"] == "apple"
    assert rows[0]["has_ai_card"] is True


def test_list_ai_card_words_returns_normalized_set() -> None:
    def fake_urlopen(request, timeout, context):
        assert "/rest/v1/xb_ai_cards" in request.full_url
        return FakeResponse(
            [
                {"normalized_word": "apple"},
                {"normalized_word": "Well-Being"},
            ]
        )

    storage = SupabaseStorage(
        SupabaseConfig("https://example.supabase.co", "server-secret"),
        urlopen_fn=fake_urlopen,
    )
    assert storage.list_ai_card_words() == {"apple", "well-being"}
