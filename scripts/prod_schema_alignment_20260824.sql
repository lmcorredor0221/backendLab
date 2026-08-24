-- Production schema alignment for Supabase project oacyylgnzwcmtnwoiuak
-- Date: 2026-08-24
-- Purpose:
--   1. Create the schema objects introduced by revisions 20260823_0015 to 20260824_0017.
--   2. Normalize the Hotmart webhook unique constraint introduced in 20260823_0016.
--   3. Stamp alembic_version at head once the schema matches production expectations.
--
-- Safe re-run behavior:
--   - Tables are created with IF NOT EXISTS.
--   - Indexes are created with IF NOT EXISTS.
--   - Existing Hotmart constraints are checked before changes.
--   - alembic_version is rewritten to the current head at the end.

begin;

create table if not exists public.commercial_quota_product_configs (
    id uuid primary key,
    product_key text not null,
    display_name text not null,
    enabled boolean not null,
    initial_free_units integer not null,
    consumption_priority jsonb not null,
    checkout_required_on_zero_balance boolean not null,
    fifo_auto_approval_enabled boolean not null,
    default_blocked_request_ttl_hours integer not null,
    default_checkout_ttl_minutes integer not null,
    debt_enabled boolean not null,
    allow_manual_override_without_charge boolean not null,
    allow_courtesy boolean not null,
    allow_debt_pending boolean not null,
    catalog_priority_strategy text not null,
    sync_retry_limit integer not null,
    duplicate_conflict_visibility text not null,
    metadata jsonb not null,
    created_at timestamp not null,
    updated_at timestamp not null,
    constraint uq_commercial_quota_product_config_product_key unique (product_key)
);

create index if not exists ix_commercial_quota_product_configs_product_key
    on public.commercial_quota_product_configs (product_key);

create table if not exists public.commercial_quota_workspace_overrides (
    id uuid primary key,
    workspace_id uuid not null references public.workspaces(id),
    product_key text not null,
    is_active boolean not null,
    enabled_override boolean null,
    free_units_override integer null,
    consumption_priority_override jsonb not null,
    checkout_required_on_zero_balance_override boolean null,
    fifo_auto_approval_enabled_override boolean null,
    default_blocked_request_ttl_hours_override integer null,
    default_checkout_ttl_minutes_override integer null,
    debt_enabled_override boolean null,
    effective_from timestamp null,
    effective_to timestamp null,
    notes text not null,
    updated_by_user_id uuid null references public.users(id),
    metadata jsonb not null,
    created_at timestamp not null,
    updated_at timestamp not null,
    constraint uq_commercial_quota_workspace_override_workspace_product
        unique (workspace_id, product_key)
);

create index if not exists ix_commercial_quota_workspace_overrides_workspace_id
    on public.commercial_quota_workspace_overrides (workspace_id);

create index if not exists ix_commercial_quota_workspace_overrides_product_key
    on public.commercial_quota_workspace_overrides (product_key);

create table if not exists public.commercial_balance_buckets (
    id uuid primary key,
    workspace_id uuid not null references public.workspaces(id),
    product_key text not null,
    bucket_key text not null,
    source_kind text not null,
    status text not null,
    units_granted integer not null,
    units_consumed integer not null,
    source_ref text not null,
    granted_by_user_id uuid null references public.users(id),
    order_id uuid null references public.commercial_orders(id),
    payment_id uuid null references public.commercial_payments(id),
    starts_at timestamp not null,
    ends_at timestamp null,
    metadata jsonb not null,
    created_at timestamp not null,
    updated_at timestamp not null,
    constraint uq_commercial_balance_bucket_workspace_product_key
        unique (workspace_id, product_key, bucket_key)
);

create index if not exists ix_commercial_balance_buckets_workspace_id
    on public.commercial_balance_buckets (workspace_id);

create index if not exists ix_commercial_balance_buckets_product_key
    on public.commercial_balance_buckets (product_key);

create index if not exists ix_commercial_balance_buckets_bucket_key
    on public.commercial_balance_buckets (bucket_key);

create index if not exists ix_commercial_balance_buckets_source_ref
    on public.commercial_balance_buckets (source_ref);

create table if not exists public.commercial_balance_ledger (
    id uuid primary key,
    workspace_id uuid not null references public.workspaces(id),
    product_key text not null,
    bucket_id uuid null references public.commercial_balance_buckets(id),
    movement_type text not null,
    source_kind text not null,
    delta_units integer not null,
    balance_before_units integer not null,
    balance_after_units integer not null,
    bucket_balance_before_units integer not null,
    bucket_balance_after_units integer not null,
    source_ref text not null,
    actor_user_id uuid null references public.users(id),
    order_id uuid null references public.commercial_orders(id),
    payment_id uuid null references public.commercial_payments(id),
    access_request_id uuid null references public.commercial_access_requests(id),
    metadata jsonb not null,
    created_at timestamp not null
);

create index if not exists ix_commercial_balance_ledger_workspace_id
    on public.commercial_balance_ledger (workspace_id);

create index if not exists ix_commercial_balance_ledger_product_key
    on public.commercial_balance_ledger (product_key);

create index if not exists ix_commercial_balance_ledger_bucket_id
    on public.commercial_balance_ledger (bucket_id);

create index if not exists ix_commercial_balance_ledger_source_ref
    on public.commercial_balance_ledger (source_ref);

create table if not exists public.commercial_package_catalog (
    id uuid primary key,
    package_code text not null,
    display_name text not null,
    product_key text not null,
    package_type text not null,
    enabled boolean not null,
    granted_units integer not null,
    granted_units_blueprint_pro integer not null,
    granted_units_acp integer not null,
    validity_days integer null,
    billing_cycle text not null,
    renewal_policy text not null,
    recommendation_priority integer not null,
    hotmart_environment text not null,
    hotmart_product_id text not null,
    hotmart_product_ucode text not null,
    offer_code text not null,
    plan_code text not null,
    checkout_currency_mode text not null,
    hotmart_price_strategy text not null,
    metadata jsonb not null,
    created_at timestamp not null,
    updated_at timestamp not null,
    constraint uq_commercial_package_catalog_code unique (package_code)
);

create index if not exists ix_commercial_package_catalog_package_code
    on public.commercial_package_catalog (package_code);

create index if not exists ix_commercial_package_catalog_product_key
    on public.commercial_package_catalog (product_key);

create table if not exists public.commercial_debts (
    id uuid primary key,
    workspace_id uuid not null references public.workspaces(id),
    product_key text not null,
    access_request_id uuid null references public.commercial_access_requests(id),
    order_id uuid null references public.commercial_orders(id),
    status text not null,
    reason_code text not null,
    reason_label text not null,
    summary text not null,
    amount_cents integer not null,
    settled_amount_cents integer not null,
    currency text not null,
    opened_by_user_id uuid null references public.users(id),
    resolved_by_user_id uuid null references public.users(id),
    due_at timestamp null,
    resolved_at timestamp null,
    metadata jsonb not null,
    created_at timestamp not null,
    updated_at timestamp not null
);

create index if not exists ix_commercial_debts_workspace_id
    on public.commercial_debts (workspace_id);

create index if not exists ix_commercial_debts_product_key
    on public.commercial_debts (product_key);

create table if not exists public.commercial_debt_settlements (
    id uuid primary key,
    debt_id uuid not null references public.commercial_debts(id),
    workspace_id uuid not null references public.workspaces(id),
    order_id uuid null references public.commercial_orders(id),
    payment_id uuid null references public.commercial_payments(id),
    settled_amount_cents integer not null,
    currency text not null,
    settlement_kind text not null,
    actor_user_id uuid null references public.users(id),
    metadata jsonb not null,
    created_at timestamp not null
);

create index if not exists ix_commercial_debt_settlements_debt_id
    on public.commercial_debt_settlements (debt_id);

create index if not exists ix_commercial_debt_settlements_workspace_id
    on public.commercial_debt_settlements (workspace_id);

do $$
begin
    if exists (
        select 1
        from information_schema.tables
        where table_schema = 'public'
          and table_name = 'hotmart_webhook_events'
    ) then
        alter table public.hotmart_webhook_events
            drop constraint if exists uq_hotmart_webhook_event_id;

        if not exists (
            select 1
            from pg_constraint
            where conname = 'uq_hotmart_webhook_event_id_type'
              and conrelid = 'public.hotmart_webhook_events'::regclass
        ) then
            alter table public.hotmart_webhook_events
                add constraint uq_hotmart_webhook_event_id_type unique (event_id, event_type);
        end if;
    end if;
end
$$;

create table if not exists public.hotmart_pending_activations (
    id uuid primary key,
    source_workspace_id uuid not null references public.workspaces(id),
    environment text not null,
    status text not null,
    provider_ref text not null,
    event_id text not null,
    webhook_event_id uuid null references public.hotmart_webhook_events(id),
    hotmart_product_id text not null,
    hotmart_product_ucode text not null,
    offer_code text not null,
    plan_code text not null,
    product_key text not null,
    package_code text not null,
    resolution_strategy text not null,
    buyer_name text not null,
    buyer_email text not null,
    buyer_document text not null,
    currency text not null,
    amount_cents integer not null,
    activation_token text not null,
    claimed_by_user_id uuid null references public.users(id),
    claimed_workspace_id uuid null references public.workspaces(id),
    claimed_session_id uuid null references public.sessions(id),
    adopted_order_id uuid null references public.commercial_orders(id),
    adopted_payment_id uuid null references public.commercial_payments(id),
    claimed_at timestamp null,
    canceled_at timestamp null,
    metadata jsonb not null,
    created_at timestamp not null,
    updated_at timestamp not null,
    constraint uq_hotmart_pending_activation_workspace_provider_ref
        unique (source_workspace_id, environment, provider_ref),
    constraint uq_hotmart_pending_activation_token unique (activation_token)
);

create index if not exists ix_hotmart_pending_activations_source_workspace_id
    on public.hotmart_pending_activations (source_workspace_id);

create index if not exists ix_hotmart_pending_activations_environment
    on public.hotmart_pending_activations (environment);

create index if not exists ix_hotmart_pending_activations_status
    on public.hotmart_pending_activations (status);

create index if not exists ix_hotmart_pending_activations_provider_ref
    on public.hotmart_pending_activations (provider_ref);

create index if not exists ix_hotmart_pending_activations_event_id
    on public.hotmart_pending_activations (event_id);

create index if not exists ix_hotmart_pending_activations_hotmart_product_id
    on public.hotmart_pending_activations (hotmart_product_id);

create index if not exists ix_hotmart_pending_activations_product_key
    on public.hotmart_pending_activations (product_key);

create index if not exists ix_hotmart_pending_activations_buyer_email
    on public.hotmart_pending_activations (buyer_email);

create index if not exists ix_hotmart_pending_activations_activation_token
    on public.hotmart_pending_activations (activation_token);

create table if not exists public.alembic_version (
    version_num varchar(32) not null
);

delete from public.alembic_version;
insert into public.alembic_version (version_num) values ('20260824_0017');

commit;

select
    version_num as alembic_version
from public.alembic_version;
