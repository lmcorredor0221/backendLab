# Domain Contracts

## Discovery

Campos minimos:

- `problem_statement`
- `current_user`
- `current_process`
- `desired_outcome`
- `autonomy_level`
- `constraints`

Campos derivados:

- `case_type`
- `value_statement`

Valores canonicos:

- `autonomy_level`: `low`, `medium`, `high`
- `case_type`: `informacion`, `automatizacion`, `copiloto`, `operador_autonomo`, `sistema_multiagente`

## Canvas

Campos:

- `user_goal`
- `mvp_scope`
- `out_of_scope`
- `success_metric`
- `primary_risk`

## Blueprint

Campos:

- `architecture`
- `reasoning_pattern`
- `memory_strategy`
- `tools`
- `memory_profile`
- `safety_checks`
- `guardrails`
- `readiness_state`
- `narrative`

## Evaluation

Campos:

- `completeness_status`
- `coherence_status`
- `cases`
- `gaps`
- `recommendations`
- `scores`

## Envelope agentico

Toda operacion debe devolver:

- `status`
- `stage`
- `data`
- `missing_fields`
- `assumptions`
- `warnings`
- `evidence`
- `next_action`

## Control de acceso y trazabilidad

Objetos minimos adicionales:

- `user`
- `auth_token`
- `session.owner`
- `execution_log`
- `validation_report`
- `blueprint_version`

Reglas:

- toda sesion pertenece a un usuario;
- toda mutacion relevante del blueprint genera una nueva version;
- la actividad debe quedar registrada por etapa y estado;
- la exportacion final siempre se deriva del ultimo estado persistido.
