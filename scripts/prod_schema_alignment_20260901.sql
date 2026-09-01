-- Production schema alignment for Render release 2026-09-01.
-- Applies Alembic revisions 20260828_0018 and 20260828_0019 when production is at 20260824_0017.
-- Run from Supabase SQL Editor with a schema-owner role or "Run without RLS".

BEGIN;

CREATE TABLE journey_state_current (
    id UUID NOT NULL,
    workspace_id UUID,
    session_id UUID NOT NULL,
    state_key VARCHAR NOT NULL,
    substate VARCHAR NOT NULL,
    product_key VARCHAR NOT NULL,
    stage_key VARCHAR NOT NULL,
    progress_percent INTEGER NOT NULL,
    blocking BOOLEAN NOT NULL,
    revision INTEGER NOT NULL,
    source_contracts JSONB NOT NULL,
    state_payload JSONB NOT NULL,
    last_transition_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uq_journey_state_current_session UNIQUE (session_id),
    FOREIGN KEY(workspace_id) REFERENCES workspaces (id),
    FOREIGN KEY(session_id) REFERENCES sessions (id)
);

CREATE INDEX ix_journey_state_current_workspace_id ON journey_state_current (workspace_id);
CREATE INDEX ix_journey_state_current_session_id ON journey_state_current (session_id);
CREATE INDEX ix_journey_state_current_state_key ON journey_state_current (state_key);
CREATE INDEX ix_journey_state_current_substate ON journey_state_current (substate);
CREATE INDEX ix_journey_state_current_last_transition_at ON journey_state_current (last_transition_at);

CREATE TABLE journey_state_transitions (
    id UUID NOT NULL,
    workspace_id UUID,
    session_id UUID NOT NULL,
    sequence INTEGER NOT NULL,
    event_key VARCHAR NOT NULL,
    from_state_key VARCHAR NOT NULL,
    from_substate VARCHAR NOT NULL,
    to_state_key VARCHAR NOT NULL,
    to_substate VARCHAR NOT NULL,
    actor_type VARCHAR NOT NULL,
    actor_user_id UUID,
    reason VARCHAR NOT NULL,
    correlation_id VARCHAR NOT NULL,
    transition_payload JSONB NOT NULL,
    occurred_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uq_journey_state_transition_sequence UNIQUE (session_id, sequence),
    CONSTRAINT uq_journey_state_transition_correlation UNIQUE (session_id, correlation_id),
    FOREIGN KEY(workspace_id) REFERENCES workspaces (id),
    FOREIGN KEY(session_id) REFERENCES sessions (id),
    FOREIGN KEY(actor_user_id) REFERENCES users (id)
);

CREATE INDEX ix_journey_state_transitions_workspace_id ON journey_state_transitions (workspace_id);
CREATE INDEX ix_journey_state_transitions_session_id ON journey_state_transitions (session_id);
CREATE INDEX ix_journey_state_transitions_event_key ON journey_state_transitions (event_key);
CREATE INDEX ix_journey_state_transitions_actor_user_id ON journey_state_transitions (actor_user_id);
CREATE INDEX ix_journey_state_transitions_occurred_at ON journey_state_transitions (occurred_at);

UPDATE alembic_version
SET version_num = '20260828_0018'
WHERE version_num = '20260824_0017';

CREATE TABLE tool_pattern_learning_candidates (
    id UUID NOT NULL,
    workspace_id UUID NOT NULL,
    session_id UUID NOT NULL,
    source_artifact_id UUID,
    source_blueprint_version INTEGER,
    candidate_pattern_id VARCHAR NOT NULL,
    capability_key VARCHAR NOT NULL,
    family_key VARCHAR NOT NULL,
    label VARCHAR NOT NULL,
    source_level VARCHAR NOT NULL,
    promotion_status VARCHAR NOT NULL,
    global_promotion_allowed BOOLEAN NOT NULL,
    dedupe_signature VARCHAR NOT NULL,
    replacement_global_pattern_id VARCHAR NOT NULL,
    contract_quality VARCHAR NOT NULL,
    risk_flags JSONB NOT NULL,
    source_refs JSONB NOT NULL,
    evidence_refs JSONB NOT NULL,
    contract_seed_payload JSONB NOT NULL,
    metadata JSONB NOT NULL,
    observation_count INTEGER NOT NULL,
    first_seen_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    last_seen_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uq_tool_pattern_learning_candidate_session_signature UNIQUE (workspace_id, session_id, dedupe_signature),
    FOREIGN KEY(workspace_id) REFERENCES workspaces (id),
    FOREIGN KEY(session_id) REFERENCES sessions (id),
    FOREIGN KEY(source_artifact_id) REFERENCES journey_stage_artifacts (id)
);

CREATE INDEX ix_tool_pattern_learning_candidates_workspace_id ON tool_pattern_learning_candidates (workspace_id);
CREATE INDEX ix_tool_pattern_learning_candidates_session_id ON tool_pattern_learning_candidates (session_id);
CREATE INDEX ix_tool_pattern_learning_candidates_source_artifact_id ON tool_pattern_learning_candidates (source_artifact_id);
CREATE INDEX ix_tool_pattern_learning_candidates_source_blueprint_version ON tool_pattern_learning_candidates (source_blueprint_version);
CREATE INDEX ix_tool_pattern_learning_candidates_candidate_pattern_id ON tool_pattern_learning_candidates (candidate_pattern_id);
CREATE INDEX ix_tool_pattern_learning_candidates_capability_key ON tool_pattern_learning_candidates (capability_key);
CREATE INDEX ix_tool_pattern_learning_candidates_family_key ON tool_pattern_learning_candidates (family_key);
CREATE INDEX ix_tool_pattern_learning_candidates_source_level ON tool_pattern_learning_candidates (source_level);
CREATE INDEX ix_tool_pattern_learning_candidates_promotion_status ON tool_pattern_learning_candidates (promotion_status);
CREATE INDEX ix_tool_pattern_learning_candidates_global_promotion_allowed ON tool_pattern_learning_candidates (global_promotion_allowed);
CREATE INDEX ix_tool_pattern_learning_candidates_dedupe_signature ON tool_pattern_learning_candidates (dedupe_signature);
CREATE INDEX ix_tool_pattern_learning_candidates_replacement_pattern_id ON tool_pattern_learning_candidates (replacement_global_pattern_id);
CREATE INDEX ix_tool_pattern_learning_candidates_contract_quality ON tool_pattern_learning_candidates (contract_quality);
CREATE INDEX ix_tool_pattern_learning_candidates_last_seen_at ON tool_pattern_learning_candidates (last_seen_at);
CREATE INDEX ix_tool_pattern_learning_candidates_workspace_status ON tool_pattern_learning_candidates (workspace_id, promotion_status);
CREATE INDEX ix_tool_pattern_learning_candidates_workspace_capability ON tool_pattern_learning_candidates (workspace_id, capability_key);

UPDATE alembic_version
SET version_num = '20260828_0019'
WHERE version_num = '20260828_0018';

COMMIT;
