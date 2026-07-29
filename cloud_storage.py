"""Small, dependency-light Supabase storage client for 小白拼写."""

from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import certifi
import pandas as pd


WORD_LISTS_TABLE = "xb_word_lists"
AI_CARDS_TABLE = "xb_ai_cards"
DAILY_WORDS_TABLE = "xb_daily_words"
LEARNED_WORDS_VIEW = "xb_learned_words"
CLOUD_REF_PREFIX = "cloud://"
HTTPS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


class CloudStorageError(RuntimeError):
    """A cloud request failed with a message safe to show in the UI."""


class CloudConflictError(CloudStorageError):
    """The remote word list changed after this device loaded it."""


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    service_role_key: str

    def normalized(self) -> "SupabaseConfig":
        return SupabaseConfig(
            url=self.url.strip().rstrip("/"),
            service_role_key=self.service_role_key.strip(),
        )

    @property
    def configured(self) -> bool:
        value = self.normalized()
        return bool(
            value.url.startswith("https://")
            and value.service_role_key
        )


def cloud_ref(record_id: str) -> str:
    return f"{CLOUD_REF_PREFIX}{record_id.strip()}"


def is_cloud_ref(value: Any) -> bool:
    return str(value or "").startswith(CLOUD_REF_PREFIX)


def cloud_id_from_ref(value: str) -> str:
    if not is_cloud_ref(value):
        raise ValueError("不是有效的云端词表引用。")
    record_id = value[len(CLOUD_REF_PREFIX) :].strip()
    if not record_id or "/" in record_id:
        raise ValueError("云端词表引用无效。")
    return record_id


def dataframe_to_payload(df: pd.DataFrame) -> dict[str, Any]:
    """Serialize columns, index and values without leaking pandas-only types."""
    return json.loads(
        df.to_json(
            orient="split",
            force_ascii=False,
            date_format="iso",
        )
    )


def dataframe_from_payload(payload: Any) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise CloudStorageError("云端词表数据格式不正确。")
    columns = payload.get("columns")
    index = payload.get("index")
    data = payload.get("data")
    if not isinstance(columns, list) or not isinstance(data, list):
        raise CloudStorageError("云端词表数据不完整。")
    if not isinstance(index, list) or len(index) != len(data):
        index = list(range(len(data)))
    try:
        return pd.DataFrame(data=data, columns=columns, index=index)
    except (TypeError, ValueError) as exc:
        raise CloudStorageError("无法还原云端词表。") from exc


def learning_rows_from_dataframe(
    dataframe: pd.DataFrame,
    *,
    word_list_id: str,
    source_name: str,
    source_date: str | None,
) -> list[dict[str, Any]]:
    """Build idempotent per-list learning rows, merging duplicate words."""
    effective_date = (
        str(source_date or "").strip()
        or datetime.now(timezone.utc).date().isoformat()
    )
    merged: dict[str, dict[str, Any]] = {}
    for _, item in dataframe.iterrows():
        word = str(item.get("单词", "")).strip()
        normalized = word.casefold()
        if not normalized:
            continue
        try:
            attempts = max(int(item.get("当天答题次数", 0) or 0), 0)
            correct = max(int(item.get("当天正确", 0) or 0), 0)
            wrong = max(int(item.get("当天错误", 0) or 0), 0)
            current_status = int(item.get("当前状态", 0) or 0)
        except (TypeError, ValueError):
            attempts = correct = wrong = current_status = 0
        if attempts <= 0:
            continue
        existing = merged.get(normalized)
        if existing is None:
            merged[normalized] = {
                "word_list_id": str(word_list_id),
                "source_name": str(source_name).strip() or "云端词表.csv",
                "source_date": effective_date,
                "normalized_word": normalized,
                "word": word,
                "chinese_meaning": str(item.get("中文释义", "")).strip(),
                "learning_type": str(item.get("类型", "")).strip(),
                "current_status": current_status,
                "attempts": attempts,
                "correct": correct,
                "wrong": wrong,
            }
            continue
        existing["attempts"] += attempts
        existing["correct"] += correct
        existing["wrong"] += wrong
        existing["current_status"] = max(
            int(existing["current_status"]),
            current_status,
        )
        if not existing["chinese_meaning"]:
            existing["chinese_meaning"] = str(
                item.get("中文释义", "")
            ).strip()
        if not existing["learning_type"]:
            existing["learning_type"] = str(item.get("类型", "")).strip()
    return list(merged.values())


def _parse_timestamp(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc).timestamp()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return datetime.now(timezone.utc).timestamp()


class SupabaseStorage:
    """Use Supabase PostgREST with a server-side service-role key."""

    def __init__(
        self,
        config: SupabaseConfig,
        *,
        timeout: float = 12.0,
        urlopen_fn: Callable[..., Any] = urlopen,
    ) -> None:
        normalized = config.normalized()
        if not normalized.configured:
            raise ValueError("Supabase 配置不完整。")
        self.base_url = normalized.url
        self._service_role_key = normalized.service_role_key
        self.timeout = timeout
        self._urlopen = urlopen_fn

    def _request(
        self,
        method: str,
        table: str,
        *,
        query: dict[str, str] | None = None,
        body: Any = None,
        prefer: str = "",
    ) -> Any:
        query_string = f"?{urlencode(query or {})}" if query else ""
        url = f"{self.base_url}/rest/v1/{table}{query_string}"
        data = (
            json.dumps(body, ensure_ascii=False).encode("utf-8")
            if body is not None
            else None
        )
        headers = {
            "apikey": self._service_role_key,
            "Authorization": f"Bearer {self._service_role_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "XiaobaiSpelling/1.0",
        }
        if prefer:
            headers["Prefer"] = prefer
        request = Request(
            url,
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with self._urlopen(
                request,
                timeout=self.timeout,
                context=HTTPS_CONTEXT,
            ) as response:
                raw = response.read()
        except HTTPError as exc:
            if exc.code in (401, 403):
                message = "Supabase 密钥无效或权限不足。"
            elif exc.code == 404:
                message = (
                    "云端数据表尚未初始化，请先运行 "
                    "docs/supabase_schema.sql。"
                )
            elif exc.code == 409:
                message = "云端数据发生冲突，请刷新后重试。"
            elif exc.code == 429:
                message = "Supabase 请求过于频繁，请稍后再试。"
            elif exc.code >= 500:
                message = "Supabase 服务暂时繁忙。"
            else:
                message = f"Supabase 请求失败（HTTP {exc.code}）。"
            raise CloudStorageError(message) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise CloudStorageError(
                "暂时无法连接 Supabase，请检查网络后重试。"
            ) from exc

        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CloudStorageError("Supabase 返回了无法解析的数据。") from exc

    def list_word_lists(self) -> list[dict[str, Any]]:
        rows = self._request(
            "GET",
            WORD_LISTS_TABLE,
            query={
                "select": (
                    "id,source_name,source_date,encoding,row_count,"
                    "revision,updated_at"
                ),
                "order": "updated_at.desc",
            },
        )
        if not isinstance(rows, list):
            raise CloudStorageError("无法读取云端词表列表。")
        records: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            name = str(row.get("source_name") or "云端词表.csv")
            records.append(
                {
                    "path": cloud_ref(str(row["id"])),
                    "name": name,
                    "parent": "Supabase 云端",
                    "size": 0,
                    "mtime": _parse_timestamp(row.get("updated_at")),
                    "date": str(row.get("source_date") or "") or None,
                    "missing": False,
                    "cloud": True,
                    "cloud_id": str(row["id"]),
                    "revision": int(row.get("revision") or 1),
                    "row_count": int(row.get("row_count") or 0),
                    "encoding": str(row.get("encoding") or "utf-8-sig"),
                }
            )
        return records

    def load_word_list(self, record_id: str) -> dict[str, Any]:
        rows = self._request(
            "GET",
            WORD_LISTS_TABLE,
            query={
                "id": f"eq.{record_id}",
                "select": "*",
                "limit": "1",
            },
        )
        if not isinstance(rows, list) or not rows:
            raise CloudStorageError("所选云端词表不存在或已被删除。")
        row = rows[0]
        return {
            "id": str(row["id"]),
            "source_name": str(row.get("source_name") or "云端词表.csv"),
            "encoding": str(row.get("encoding") or "utf-8-sig"),
            "revision": int(row.get("revision") or 1),
            "updated_at": str(row.get("updated_at") or ""),
            "dataframe": dataframe_from_payload(row.get("data")),
        }

    def upsert_word_list(
        self,
        *,
        source_name: str,
        dataframe: pd.DataFrame,
        encoding: str = "utf-8-sig",
        source_date: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        body = {
            "source_name": source_name,
            "source_date": source_date,
            "encoding": encoding or "utf-8-sig",
            "data": dataframe_to_payload(dataframe),
            "row_count": len(dataframe),
            "revision": 1,
            "updated_at": now,
        }
        rows = self._request(
            "POST",
            WORD_LISTS_TABLE,
            query={"on_conflict": "source_name"},
            body=body,
            prefer="resolution=merge-duplicates,return=representation",
        )
        if not isinstance(rows, list) or not rows:
            raise CloudStorageError("词表上传后没有返回确认信息。")
        return rows[0]

    def save_word_list(
        self,
        *,
        record_id: str,
        dataframe: pd.DataFrame,
        expected_revision: int,
    ) -> int:
        next_revision = int(expected_revision) + 1
        rows = self._request(
            "PATCH",
            WORD_LISTS_TABLE,
            query={
                "id": f"eq.{record_id}",
                "revision": f"eq.{int(expected_revision)}",
            },
            body={
                "data": dataframe_to_payload(dataframe),
                "row_count": len(dataframe),
                "revision": next_revision,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            prefer="return=representation",
        )
        if not isinstance(rows, list) or not rows:
            raise CloudConflictError(
                "另一台设备已经更新了这个词表。请点击“从云端重新加载”后继续。"
            )
        return int(rows[0].get("revision") or next_revision)

    def load_ai_card(self, word: str) -> dict[str, Any] | None:
        normalized = str(word).strip().casefold()
        rows = self._request(
            "GET",
            AI_CARDS_TABLE,
            query={
                "normalized_word": f"eq.{normalized}",
                "select": "bundle",
                "limit": "1",
            },
        )
        if not isinstance(rows, list) or not rows:
            return None
        bundle = rows[0].get("bundle")
        return bundle if isinstance(bundle, dict) else None

    def save_ai_card(self, word: str, bundle: dict[str, Any]) -> None:
        normalized = str(word).strip().casefold()
        self._request(
            "POST",
            AI_CARDS_TABLE,
            query={"on_conflict": "normalized_word"},
            body={
                "normalized_word": normalized,
                "word": str(word).strip(),
                "model": str(bundle.get("model") or ""),
                "bundle": bundle,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            prefer="resolution=merge-duplicates,return=minimal",
        )

    def sync_learning_snapshot(
        self,
        *,
        word_list_id: str,
        source_name: str,
        source_date: str | None,
        dataframe: pd.DataFrame,
    ) -> int:
        """Atomically replace learned rows for one uploaded CSV snapshot."""
        rows = learning_rows_from_dataframe(
            dataframe,
            word_list_id=word_list_id,
            source_name=source_name,
            source_date=source_date,
        )
        result = self._request(
            "POST",
            "rpc/xb_sync_daily_words",
            body={
                "p_word_list_id": str(word_list_id),
                "p_source_name": str(source_name).strip() or "云端词表.csv",
                "p_source_date": (
                    str(source_date or "").strip()
                    or datetime.now(timezone.utc).date().isoformat()
                ),
                "p_rows": rows,
            },
        )
        try:
            return int(result or 0)
        except (TypeError, ValueError) as exc:
            raise CloudStorageError("云端未返回有效的学习记录数量。") from exc

    def upsert_learning_row(
        self,
        *,
        word_list_id: str,
        source_name: str,
        source_date: str | None,
        row: pd.Series,
    ) -> None:
        rows = learning_rows_from_dataframe(
            pd.DataFrame([row]),
            word_list_id=word_list_id,
            source_name=source_name,
            source_date=source_date,
        )
        if not rows:
            return
        rows[0]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._request(
            "POST",
            DAILY_WORDS_TABLE,
            query={"on_conflict": "word_list_id,normalized_word"},
            body=rows[0],
            prefer="resolution=merge-duplicates,return=minimal",
        )

    def list_learned_words(self) -> list[dict[str, Any]]:
        rows = self._request(
            "GET",
            LEARNED_WORDS_VIEW,
            query={
                "select": "*",
                "order": "last_seen.desc,normalized_word.asc",
            },
        )
        if not isinstance(rows, list):
            raise CloudStorageError("无法读取已学单词本。")
        return [row for row in rows if isinstance(row, dict)]
