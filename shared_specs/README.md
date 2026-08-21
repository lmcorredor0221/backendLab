# Shared Specs

Contratos compartidos del MVP de `Lean Agent Builder`.

## Documentos activos

- `domain_contracts.md`: dominios canonicos del builder y reglas base.
- `workspace_contracts.md`: shape estable de `SessionSnapshot`, `ACPPreview` y `workspace_contract`.
- `artifact-diagram-taxonomy.v1.json`: inventario canonico de artefactos y diagramas por producto, etapa, acceso, formato y portabilidad.

## Artefactos base

- `session`
- `discovery`
- `canvas`
- `blueprint`
- `validation_report`
- `artifact_diagram_taxonomy`

## Etapas del flujo

1. `draft_capture`
2. `input_validation`
3. `normalize_discovery`
4. `build_canvas`
5. `build_blueprint`
6. `post_validation`
7. `ready_for_export`

## Estados permitidos

- `draft`
- `ready`
- `needs_review`
- `failed`

## Principios

- rule-first
- schema-first
- state-first
- unknown over invention
- partial regeneration
