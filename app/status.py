"""Health and Telemetry Status Data Structures for SS_PROJECT_VCR_FRESH."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from app.constants import ExecutionMode, FaultCode
from app.hardware.interfaces import ObstacleState


class SubsystemStatus(str, Enum):
    """Lifecycle/health state of an individual robot subsystem."""
    UNINITIALIZED = "UNINITIALIZED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    FAULT = "FAULT"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class SystemHealth:
    """Comprehensive, immutable snapshot of robot state and subsystem health."""
    mode: ExecutionMode
    is_healthy: bool
    active_fault: FaultCode = FaultCode.OK
    subsystems: Dict[str, SubsystemStatus] = field(default_factory=dict)
    motors_enabled: bool = False
    current_speeds: Tuple[float, float] = (0.0, 0.0)
    last_distance_cm: Optional[float] = None
    obstacle_state: ObstacleState = ObstacleState.FAULT
    uptime_sec: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert health snapshot to a JSON-serializable dictionary."""
        return {
            "mode": self.mode.value,
            "is_healthy": self.is_healthy,
            "active_fault": self.active_fault.value,
            "subsystems": {k: v.value for k, v in self.subsystems.items()},
            "motors_enabled": self.motors_enabled,
            "current_speeds": {
                "left": self.current_speeds[0],
                "right": self.current_speeds[1],
            },
            "last_distance_cm": self.last_distance_cm,
            "obstacle_state": self.obstacle_state.value,
            "uptime_sec": round(self.uptime_sec, 2),
            "timestamp": self.timestamp,
        }
