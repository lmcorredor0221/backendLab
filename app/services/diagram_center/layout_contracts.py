from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.services.diagram_center.contracts import DiagramNotation, SAFE_IDENTIFIER


class DiagramLayoutStrategy(StrEnum):
    fixed_grid = "fixed_grid"
    layered = "layered"
    bpmn_swimlane = "bpmn_swimlane"
    uml_use_case = "uml_use_case"
    uml_activity = "uml_activity"
    split_required = "split_required"


class DiagramLayoutRisk(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class DiagramLayoutPoint(BaseModel):
    x: float
    y: float


class DiagramLayoutViewport(BaseModel):
    width: int = Field(ge=320)
    height: int = Field(ge=240)
    min_scale: float = Field(default=0.35, ge=0.1, le=1)
    max_scale: float = Field(default=2.5, ge=1, le=6)
    fit_mode: Literal["contain", "width", "actual"] = "contain"


class DiagramLayoutMetrics(BaseModel):
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    edge_density: float = Field(ge=0)
    max_degree: int = Field(ge=0)
    label_avg_chars: float = Field(ge=0)
    lane_count: int = Field(default=0, ge=0)
    pool_count: int = Field(default=0, ge=0)
    estimated_crossing_risk: DiagramLayoutRisk = DiagramLayoutRisk.low
    split_recommended: bool = False


class DiagramLayoutNodeBox(BaseModel):
    node_id: str
    x: float
    y: float
    width: int = Field(ge=24)
    height: int = Field(ge=24)
    kind: str
    layer: int = Field(default=0, ge=0)
    group_id: str | None = None
    pool_id: str | None = None
    lane_id: str | None = None
    label_lines: list[str] = Field(default_factory=list)

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, value: str) -> str:
        normalized = value.strip()
        if not SAFE_IDENTIFIER.fullmatch(normalized):
            raise ValueError("layout node id must be a stable safe identifier")
        return normalized


class DiagramLayoutEdgeRoute(BaseModel):
    edge_id: str
    source: str
    target: str
    points: list[DiagramLayoutPoint] = Field(default_factory=list, min_length=2)
    label: str = ""
    label_position: DiagramLayoutPoint | None = None
    crossing_risk: DiagramLayoutRisk = DiagramLayoutRisk.low

    @field_validator("edge_id", "source", "target")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not SAFE_IDENTIFIER.fullmatch(normalized):
            raise ValueError("layout edge identifiers must be stable and safe")
        return normalized


class DiagramSplitRecommendation(BaseModel):
    recommended: bool = False
    reason: str = ""
    suggested_chunks: list[str] = Field(default_factory=list)


class DiagramLayoutPlan(BaseModel):
    schema_version: Literal["diagram-layout-plan.v1"] = "diagram-layout-plan.v1"
    diagram_key: str
    notation: DiagramNotation
    strategy: DiagramLayoutStrategy
    metrics: DiagramLayoutMetrics
    viewport: DiagramLayoutViewport
    nodes: list[DiagramLayoutNodeBox] = Field(default_factory=list)
    edges: list[DiagramLayoutEdgeRoute] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    split_recommendation: DiagramSplitRecommendation = Field(default_factory=DiagramSplitRecommendation)
    renderer_revision: str = ""

    @field_validator("diagram_key")
    @classmethod
    def validate_diagram_key(cls, value: str) -> str:
        normalized = value.strip()
        if not SAFE_IDENTIFIER.fullmatch(normalized):
            raise ValueError("layout diagram key must be a stable safe identifier")
        return normalized

    @model_validator(mode="after")
    def validate_layout_integrity(self) -> "DiagramLayoutPlan":
        node_ids = {node.node_id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("layout node ids must be unique")
        invalid_edges = [
            edge.edge_id
            for edge in self.edges
            if edge.source not in node_ids or edge.target not in node_ids
        ]
        if invalid_edges:
            raise ValueError(f"layout edges reference unknown nodes: {', '.join(invalid_edges)}")
        return self
