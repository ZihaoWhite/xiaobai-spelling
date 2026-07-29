-- 小白拼写：个人多设备云同步数据库
-- 在 Supabase Dashboard > SQL Editor 中完整运行一次。

create extension if not exists pgcrypto;

create table if not exists public.xb_word_lists (
    id uuid primary key default gen_random_uuid(),
    source_name text not null unique,
    source_date date,
    encoding text not null default 'utf-8-sig',
    data jsonb not null,
    row_count integer not null default 0 check (row_count >= 0),
    revision bigint not null default 1 check (revision >= 1),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists xb_word_lists_updated_at_idx
    on public.xb_word_lists (updated_at desc);

create table if not exists public.xb_ai_cards (
    normalized_word text primary key,
    word text not null,
    model text not null default '',
    bundle jsonb not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.xb_daily_words (
    word_list_id uuid not null
        references public.xb_word_lists(id) on delete cascade,
    source_name text not null,
    source_date date not null,
    normalized_word text not null,
    word text not null,
    chinese_meaning text not null default '',
    learning_type text not null default '',
    current_status integer not null default 0,
    attempts integer not null default 0 check (attempts >= 0),
    correct integer not null default 0 check (correct >= 0),
    wrong integer not null default 0 check (wrong >= 0),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (word_list_id, normalized_word)
);

create index if not exists xb_daily_words_normalized_word_idx
    on public.xb_daily_words (normalized_word);
create index if not exists xb_daily_words_source_date_idx
    on public.xb_daily_words (source_date desc);

create or replace function public.xb_sync_daily_words(
    p_word_list_id uuid,
    p_source_name text,
    p_source_date date,
    p_rows jsonb
)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
    inserted_count integer := 0;
begin
    delete from public.xb_daily_words
    where word_list_id = p_word_list_id;

    insert into public.xb_daily_words (
        word_list_id,
        source_name,
        source_date,
        normalized_word,
        word,
        chinese_meaning,
        learning_type,
        current_status,
        attempts,
        correct,
        wrong,
        updated_at
    )
    select
        p_word_list_id,
        coalesce(nullif(trim(p_source_name), ''), '云端词表.csv'),
        p_source_date,
        trim(item.normalized_word),
        item.word,
        coalesce(item.chinese_meaning, ''),
        coalesce(item.learning_type, ''),
        greatest(coalesce(item.current_status, 0), 0),
        greatest(coalesce(item.attempts, 0), 0),
        greatest(coalesce(item.correct, 0), 0),
        greatest(coalesce(item.wrong, 0), 0),
        now()
    from jsonb_to_recordset(coalesce(p_rows, '[]'::jsonb)) as item(
        word_list_id text,
        source_name text,
        source_date date,
        normalized_word text,
        word text,
        chinese_meaning text,
        learning_type text,
        current_status integer,
        attempts integer,
        correct integer,
        wrong integer
    )
    where coalesce(item.attempts, 0) > 0
      and nullif(trim(item.normalized_word), '') is not null;

    get diagnostics inserted_count = row_count;
    return inserted_count;
end;
$$;

create or replace view public.xb_learned_words
with (security_invoker = true)
as
with practiced as (
    select *
    from public.xb_daily_words
    where attempts > 0
),
totals as (
    select
        normalized_word,
        min(source_date) as first_seen,
        max(source_date) as last_seen,
        count(distinct source_date)::integer as study_days,
        count(distinct word_list_id)::integer as source_count,
        sum(attempts)::bigint as total_attempts,
        sum(correct)::bigint as total_correct,
        sum(wrong)::bigint as total_wrong,
        max(current_status)::integer as current_status
    from practiced
    group by normalized_word
),
latest as (
    select distinct on (normalized_word)
        normalized_word,
        word,
        chinese_meaning,
        learning_type,
        source_name
    from practiced
    order by normalized_word, source_date desc, updated_at desc
)
select
    totals.normalized_word,
    latest.word,
    latest.chinese_meaning,
    latest.learning_type,
    latest.source_name as latest_source_name,
    totals.first_seen,
    totals.last_seen,
    totals.study_days,
    totals.source_count,
    totals.total_attempts,
    totals.total_correct,
    totals.total_wrong,
    totals.current_status,
    case
        when totals.total_attempts > 0
        then round(totals.total_correct::numeric / totals.total_attempts, 4)
        else 0
    end as accuracy,
    exists (
        select 1
        from public.xb_ai_cards
        where xb_ai_cards.normalized_word = totals.normalized_word
    ) as has_ai_card
from totals
join latest using (normalized_word);

alter table public.xb_word_lists enable row level security;
alter table public.xb_ai_cards enable row level security;
alter table public.xb_daily_words enable row level security;

revoke all on table public.xb_word_lists from anon, authenticated;
revoke all on table public.xb_ai_cards from anon, authenticated;
revoke all on table public.xb_daily_words from anon, authenticated;
revoke all on table public.xb_learned_words from anon, authenticated;
revoke all on function public.xb_sync_daily_words(
    uuid, text, date, jsonb
) from public, anon, authenticated;
grant select, insert, update, delete on table public.xb_word_lists to service_role;
grant select, insert, update, delete on table public.xb_ai_cards to service_role;
grant select, insert, update, delete on table public.xb_daily_words to service_role;
grant select on table public.xb_learned_words to service_role;
grant execute on function public.xb_sync_daily_words(
    uuid, text, date, jsonb
) to service_role;

-- 不创建公开策略：浏览器端无法直接读取数据。
-- 小白拼写只在服务器端使用 Secret key（映射为 service_role）访问。
