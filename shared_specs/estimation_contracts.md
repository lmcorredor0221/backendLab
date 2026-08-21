# Estimation Contracts

## Objetivo

Este documento fija la decision de `Etapa 0` para la capacidad de estimacion comparativa del Lean Agent Builder.

## Decision de persistencia

El artefacto `estimation_report` vive en dos capas:

- `SessionSnapshot.estimation_report` para consumo inmediato del frontend.
- `artifact_records` con `artifact_kind=estimation_report` para trazabilidad, versionado y recarga posterior.

Esta decision evita crear una ruta paralela o una tabla nueva en la etapa 0.

## Contratos principales

- `EstimationReportArtifact`
- `TraditionalEstimate`
- `AgenticEstimate`
- `ConfidenceBreakdown`
- `WorkstreamEstimate`

## Catalogos gobernados reutilizando `runtime_catalog_entries`

La etapa 0 reutiliza `RuntimeCatalogEntryRecord` como almacenamiento semilla para los insumos de estimacion.

Catalogos definidos:

- `estimation_maturity_stages`
- `estimation_role_rates`
- `estimation_workstream_effort`
- `estimation_automation_matrix`
- `estimation_pricing_profiles`
- `estimation_confidence_bands`

## Regla de uso

En etapa 0 estos catalogos quedan como `seed contracts`.

Eso significa:

- la estructura ya es estable;
- los item keys ya son gobernados;
- la etapa 1 puede leerlos sin redisenar storage;
- los valores economicos y de calibracion todavia pueden estar en estado `seed`.

## Regla de evolucion

- etapa 1 llena formulas y effort model;
- etapa 2 consume `estimation_automation_matrix`;
- etapa 3 sincroniza `estimation_pricing_profiles`;
- etapa 4 usa `estimation_confidence_bands`;
- etapa 6 captura actuals y recalibra.
