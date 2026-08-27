create or replace function public.export_lab_project_snapshot(p_session uuid)
returns jsonb
language plpgsql
as $$
declare
    v_workspace uuid;
    v_result jsonb := '{}'::jsonb;
    v_table record;
    v_data jsonb;
begin
    select workspace_id into v_workspace
    from public.sessions
    where id = p_session;

    if v_workspace is null then
        raise exception 'sessions.id % no existe', p_session;
    end if;

    select coalesce(jsonb_agg(to_jsonb(t)), '[]'::jsonb)
    into v_data
    from (
        select *
        from public.sessions
        where id = p_session
    ) t;
    v_result := v_result || jsonb_build_object('sessions', v_data);

    select coalesce(jsonb_agg(to_jsonb(t)), '[]'::jsonb)
    into v_data
    from (
        select *
        from public.workspaces
        where id = v_workspace
    ) t;
    v_result := v_result || jsonb_build_object('workspaces', v_data);

    for v_table in
        select t.table_name
        from information_schema.tables t
        where t.table_schema = 'public'
          and t.table_type = 'BASE TABLE'
          and t.table_name not in (
              'sessions',
              'workspaces',
              'runtime_settings_audit',
              'users',
              'commercial_order_lines',
              'evaluation_cases',
              'evaluation_results',
              'workspace_provider_secrets',
              'hotmart_integration_secrets'
          )
          and exists (
              select 1
              from information_schema.columns c
              where c.table_schema = t.table_schema
                and c.table_name = t.table_name
                and c.column_name = 'session_id'
          )
        order by t.table_name
    loop
        execute format(
            'select coalesce(jsonb_agg(to_jsonb(x)), ''[]''::jsonb) from (select * from public.%I where session_id = $1) x',
            v_table.table_name
        )
        into v_data
        using p_session;

        v_result := v_result || jsonb_build_object(v_table.table_name, coalesce(v_data, '[]'::jsonb));
    end loop;

    for v_table in
        select t.table_name
        from information_schema.tables t
        where t.table_schema = 'public'
          and t.table_type = 'BASE TABLE'
          and t.table_name not in (
              'sessions',
              'workspaces',
              'runtime_settings_audit',
              'users',
              'workspace_provider_secrets',
              'hotmart_integration_secrets'
          )
          and exists (
              select 1
              from information_schema.columns c
              where c.table_schema = t.table_schema
                and c.table_name = t.table_name
                and c.column_name = 'workspace_id'
          )
          and not exists (
              select 1
              from information_schema.columns c
              where c.table_schema = t.table_schema
                and c.table_name = t.table_name
                and c.column_name = 'session_id'
          )
        order by t.table_name
    loop
        execute format(
            'select coalesce(jsonb_agg(to_jsonb(x)), ''[]''::jsonb) from (select * from public.%I where workspace_id = $1) x',
            v_table.table_name
        )
        into v_data
        using v_workspace;

        v_result := v_result || jsonb_build_object(v_table.table_name, coalesce(v_data, '[]'::jsonb));
    end loop;

    select coalesce(jsonb_agg(to_jsonb(t)), '[]'::jsonb)
    into v_data
    from (
        select *
        from public.runtime_settings_audit
        where scope_id in (p_session::text, v_workspace::text)
    ) t;
    v_result := v_result || jsonb_build_object('runtime_settings_audit', v_data);

    select coalesce(jsonb_agg(to_jsonb(t)), '[]'::jsonb)
    into v_data
    from (
        select *
        from public.evaluation_cases
        where dataset_id in (
            select id
            from public.evaluation_datasets
            where session_id = p_session
        )
    ) t;
    v_result := v_result || jsonb_build_object('evaluation_cases', v_data);

    select coalesce(jsonb_agg(to_jsonb(t)), '[]'::jsonb)
    into v_data
    from (
        select *
        from public.evaluation_results
        where run_id in (
            select id
            from public.evaluation_runs
            where session_id = p_session
        )
    ) t;
    v_result := v_result || jsonb_build_object('evaluation_results', v_data);

    select coalesce(jsonb_agg(to_jsonb(t)), '[]'::jsonb)
    into v_data
    from (
        select *
        from public.commercial_order_lines
        where order_id in (
            select id
            from public.commercial_orders
            where session_id = p_session
               or workspace_id = v_workspace
        )
    ) t;
    v_result := v_result || jsonb_build_object('commercial_order_lines', v_data);

    select coalesce(jsonb_agg(to_jsonb(t)), '[]'::jsonb)
    into v_data
    from (
        select *
        from public.hotmart_pending_activations
        where claimed_session_id = p_session
           or source_workspace_id = v_workspace
           or claimed_workspace_id = v_workspace
    ) t;
    v_result := v_result || jsonb_build_object('hotmart_pending_activations', v_data);

    select coalesce(jsonb_agg(to_jsonb(t)), '[]'::jsonb)
    into v_data
    from (
        with user_ids as (
            select user_id as id
            from public.workspace_memberships
            where workspace_id = v_workspace
            union
            select created_by_user_id
            from public.workspaces
            where id = v_workspace
            union
            select user_id
            from public.sessions
            where id = p_session
            union
            select archived_by_user_id
            from public.sessions
            where id = p_session
            union
            select deleted_by_user_id
            from public.sessions
            where id = p_session
            union
            select actor_user_id
            from public.runtime_settings_audit
            where scope_id in (p_session::text, v_workspace::text)
            union
            select answered_by_user_id
            from public.construction_question_responses
            where session_id = p_session
            union
            select created_by_user_id
            from public.acp_build_runs
            where session_id = p_session
            union
            select user_id
            from public.export_jobs
            where session_id = p_session
            union
            select user_id
            from public.acp_launch_reports
            where session_id = p_session
            union
            select requested_by_user_id
            from public.diagram_generation_jobs_v3
            where session_id = p_session
            union
            select created_by_user_id
            from public.product_build_runs_v1
            where session_id = p_session
            union
            select user_id
            from public.stage_operations
            where session_id = p_session
            union
            select approved_by_user_id
            from public.journey_stage_artifacts
            where session_id = p_session
            union
            select actor_user_id
            from public.journey_stage_decisions
            where session_id = p_session
            union
            select created_by_user_id
            from public.deliverable_prompt_versions_v1
            where workspace_id = v_workspace
            union
            select actor_user_id
            from public.deliverable_prompt_audit_v1
            where workspace_id = v_workspace
            union
            select actor_user_id
            from public.deliverable_governance_audit_v1
            where workspace_id = v_workspace
            union
            select updated_by_user_id
            from public.deliverable_governance_v1
            where workspace_id = v_workspace
        )
        select *
        from public.users
        where id in (select id from user_ids where id is not null)
    ) t;
    v_result := v_result || jsonb_build_object('users', v_data);

    return v_result;
end;
$$;

select export_lab_project_snapshot('__PROJECT_ID__'::uuid) as snapshot;
