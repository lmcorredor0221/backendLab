-- Lean Agent Builder production repair for Rebill commerce provider tables.
-- Run this once in the Supabase SQL Editor for project oacyylgnzwcmtnwoiuak.
-- The restricted runtime role cannot CREATE tables, so this must run as the
-- dashboard/admin owner. It is intentionally idempotent.

begin;

create table if not exists public.platform_provider_secrets (
    id uuid primary key,
    provider_key varchar not null,
    secret_kind varchar not null,
    secret_ciphertext varchar not null,
    secret_ref varchar not null,
    status varchar not null,
    last_rotated_at timestamp without time zone null,
    updated_by_user_id uuid null references public.users(id),
    created_at timestamp without time zone not null,
    updated_at timestamp without time zone not null,
    constraint uq_platform_provider_secret unique (provider_key, secret_kind)
);

create index if not exists ix_platform_provider_secrets_provider_key
    on public.platform_provider_secrets (provider_key);

create table if not exists public.commerce_provider_configs (
    id uuid primary key,
    workspace_id uuid not null references public.workspaces(id),
    provider_key varchar not null,
    environment varchar not null,
    enabled boolean not null,
    status varchar not null,
    api_base_url varchar not null,
    webhook_public_url varchar not null,
    capabilities jsonb not null default '{}'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    last_checked_at timestamp without time zone null,
    last_health_status varchar not null,
    last_health_message varchar not null,
    created_by_user_id uuid null references public.users(id),
    updated_by_user_id uuid null references public.users(id),
    created_at timestamp without time zone not null,
    updated_at timestamp without time zone not null,
    constraint uq_commerce_provider_config_workspace_provider_environment
        unique (workspace_id, provider_key, environment)
);

create index if not exists ix_commerce_provider_config_workspace_provider_status
    on public.commerce_provider_configs (workspace_id, provider_key, environment, status);

create table if not exists public.commerce_provider_secrets (
    id uuid primary key,
    workspace_id uuid not null references public.workspaces(id),
    provider_key varchar not null,
    environment varchar not null,
    secret_kind varchar not null,
    secret_ciphertext varchar not null,
    secret_ref varchar not null,
    configured boolean not null,
    status varchar not null,
    last_rotated_at timestamp without time zone null,
    updated_by_user_id uuid null references public.users(id),
    created_at timestamp without time zone not null,
    updated_at timestamp without time zone not null,
    constraint uq_commerce_provider_secret_workspace_provider_environment_kind
        unique (workspace_id, provider_key, environment, secret_kind)
);

create index if not exists ix_commerce_provider_secret_workspace_provider
    on public.commerce_provider_secrets (workspace_id, provider_key, environment);

create table if not exists public.commerce_provider_product_mappings (
    id uuid primary key,
    workspace_id uuid not null references public.workspaces(id),
    provider_key varchar not null,
    environment varchar not null,
    internal_product_key varchar not null,
    package_code varchar not null,
    billing_mode varchar not null,
    currency varchar not null,
    internal_unit_amount_usd_cents integer not null,
    provider_product_id varchar not null,
    provider_plan_id varchar not null,
    provider_price_id varchar not null,
    provider_payment_link_id varchar not null,
    provider_offer_ref varchar not null,
    price_strategy varchar not null,
    grants_tier varchar not null,
    entitlement_scope varchar not null,
    is_active boolean not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamp without time zone not null,
    updated_at timestamp without time zone not null,
    constraint uq_commerce_provider_mapping_workspace_provider_product_package
        unique (workspace_id, provider_key, environment, internal_product_key, package_code)
);

create index if not exists ix_commerce_provider_mapping_workspace_provider_active
    on public.commerce_provider_product_mappings (workspace_id, provider_key, environment, is_active);

create table if not exists public.commerce_provider_checkout_records (
    id uuid primary key,
    workspace_id uuid not null references public.workspaces(id),
    provider_key varchar not null,
    environment varchar not null,
    order_id uuid not null references public.commercial_orders(id),
    checkout_ref varchar not null,
    provider_checkout_id varchar not null,
    provider_payment_link_id varchar not null,
    provider_customer_id varchar not null,
    checkout_url varchar not null,
    status varchar not null,
    amount_cents integer not null,
    currency varchar not null,
    request_payload_redacted jsonb not null default '{}'::jsonb,
    response_payload_redacted jsonb not null default '{}'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    expires_at timestamp without time zone null,
    created_at timestamp without time zone not null,
    updated_at timestamp without time zone not null,
    constraint uq_commerce_provider_checkout_ref unique (provider_key, checkout_ref)
);

create index if not exists ix_commerce_provider_checkout_order
    on public.commerce_provider_checkout_records (order_id);

create index if not exists ix_commerce_provider_checkout_provider_checkout_id
    on public.commerce_provider_checkout_records (provider_key, provider_checkout_id);

create index if not exists ix_commerce_provider_checkout_provider_payment_link_id
    on public.commerce_provider_checkout_records (provider_key, provider_payment_link_id);

create table if not exists public.commerce_provider_webhook_events (
    id uuid primary key,
    provider_key varchar not null,
    environment varchar not null,
    event_id varchar not null,
    event_type varchar not null,
    provider_resource_id varchar not null,
    workspace_id uuid null references public.workspaces(id),
    order_id uuid null references public.commercial_orders(id),
    payment_id uuid null references public.commercial_payments(id),
    signature_validated boolean not null,
    processing_status varchar not null,
    retries integer not null,
    payload_hash varchar not null,
    payload_redacted jsonb not null default '{}'::jsonb,
    error_code varchar not null,
    error_message varchar not null,
    processed_at timestamp without time zone null,
    created_at timestamp without time zone not null,
    constraint uq_commerce_provider_webhook_event unique (provider_key, event_id, event_type)
);

create index if not exists ix_commerce_provider_webhook_workspace_status_created
    on public.commerce_provider_webhook_events (workspace_id, processing_status, created_at);

create index if not exists ix_commerce_provider_webhook_provider_resource
    on public.commerce_provider_webhook_events (provider_key, provider_resource_id);

grant usage on schema public to luism_corredor_lab;

grant select, insert, update, delete on public.platform_provider_secrets to luism_corredor_lab;
grant select, insert, update, delete on public.commerce_provider_configs to luism_corredor_lab;
grant select, insert, update, delete on public.commerce_provider_secrets to luism_corredor_lab;
grant select, insert, update, delete on public.commerce_provider_product_mappings to luism_corredor_lab;
grant select, insert, update, delete on public.commerce_provider_checkout_records to luism_corredor_lab;
grant select, insert, update, delete on public.commerce_provider_webhook_events to luism_corredor_lab;

commit;
