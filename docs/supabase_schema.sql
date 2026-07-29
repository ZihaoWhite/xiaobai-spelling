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

alter table public.xb_word_lists enable row level security;
alter table public.xb_ai_cards enable row level security;

revoke all on table public.xb_word_lists from anon, authenticated;
revoke all on table public.xb_ai_cards from anon, authenticated;
grant select, insert, update, delete on table public.xb_word_lists to service_role;
grant select, insert, update, delete on table public.xb_ai_cards to service_role;

-- 不创建公开策略：浏览器端无法直接读取数据。
-- 小白拼写只在服务器端使用 Secret key（映射为 service_role）访问。
