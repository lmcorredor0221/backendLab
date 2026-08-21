# Workspace Contracts

Contratos operativos que no deben derivarse por intuicion.

## Session Snapshot

Shape estable de lectura integral del workspace:

- `contract_version`
- `session`
- `discovery`
- `canvas`
- `blueprint`
- `evaluation`
- `evaluation_dataset`
- `evaluation_rubric`
- `evaluation_runs`
- `validations`
- `activity`
- `blueprint_versions`
- `selected_workflow_template_key`
- `approvals`
- `artifact_records`
- `metric_snapshots`
- `alert_events`
- `integration_statuses`
- `workflow_templates`
- `handoff_records`
- `governance_policies`
- `subagent_runs`
- `workspace_contract`
- `skill_catalog`
- `skill_runs`

Referencias actuales:

- backend: `backend/app/api/routes/sessions.py` -> `build_snapshot()`
- frontend: `frontend/src/types.ts` -> `SessionSnapshot`
- cliente: `frontend/src/lib/api.ts` -> `normalizeSessionSnapshot()`

## ACP Preview

Shape estable de paquete continuable:

- `package_version`
- `session_id`
- `blueprint_version_number`
- `manifest_path`
- `files`
- `validation`
- `construction_readiness`

Subcontratos ACP que deben mantenerse sincronizados:

- `ACPFileEntry`
- `ACPValidationReport`
- `ConstructionGapEntry`
- `ConstructionQuestionViewEntry`
- `ConstructionReadinessReport`
- `BlueprintKnowledgeGraph`

Referencias actuales:

- backend rutas: `backend/app/api/routes/session_acp.py`
- backend generacion: `backend/app/services/acp_generator.py`
- frontend tipos ACP: `frontend/src/types/acp.ts`
- frontend vista ACP: `frontend/src/components/AcpPreviewPanel.tsx`

## Workspace Contract

Shape estable de navegacion/capacidad:

- `contract_version`
- `sections[]`

Cada `section` debe declarar como minimo:

- `key`
- `label`
- `view_kind`
- `capability_status`
- `source_of_truth`
- `read_only`
- `summary`

Referencias actuales:

- backend bootstrap: `backend/app/services/workspace_bootstrap.py`
- snapshot serializado: `build_snapshot()`
- frontend consumo: `frontend/src/types.ts`

## Regla de cambio

Si se modifica cualquiera de estos contratos, el cambio debe actualizar como minimo:

1. modelo/tipo fuente;
2. normalizador frontend o serializador backend afectado;
3. prueba focalizada de frontera.
