"""Robot joint state management, analogous to rbs_physics Articulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

from stiff_physics.engine import Engine


@dataclass
class JointInfo:
    """Metadata for a single joint (revolute or prismatic)."""

    name: str
    index: int
    lower_limit: float      # radians (revolute) or metres (prismatic)
    upper_limit: float
    strength_ratio: float
    is_prismatic: bool

    @property
    def lower_limit_deg(self) -> float:
        return math.degrees(self.lower_limit) if not self.is_prismatic else self.lower_limit

    @property
    def upper_limit_deg(self) -> float:
        return math.degrees(self.upper_limit) if not self.is_prismatic else self.upper_limit


class Robot:
    """Manages joint state for a robot loaded into the engine.

    Provides name-based and index-based access to joint targets with
    automatic degree/radian conversion and limit clamping.
    """

    def __init__(self, engine: Engine):
        self._engine = engine
        self._revolute_joints: list[JointInfo] = []
        self._prismatic_joints: list[JointInfo] = []
        self._name_to_joint: dict[str, JointInfo] = {}
        self._targets_deg: dict[int, float] = {}        # revolute: cached deg
        self._targets_mm: dict[int, float] = {}          # prismatic: cached mm

        self._populate_joints()

    def _populate_joints(self) -> None:
        for i in range(self._engine.num_revolute_joints):
            info = self._engine.get_revolute_joint_info(i)
            ji = JointInfo(
                name=info.name,
                index=i,
                lower_limit=info.lower_limit,
                upper_limit=info.upper_limit,
                strength_ratio=info.strength_ratio,
                is_prismatic=False,
            )
            self._revolute_joints.append(ji)
            self._name_to_joint[info.name] = ji
            self._targets_deg[i] = math.degrees(info.target)

        for i in range(self._engine.num_prismatic_joints):
            info = self._engine.get_prismatic_joint_info(i)
            ji = JointInfo(
                name=info.name,
                index=i,
                lower_limit=info.lower_limit,
                upper_limit=info.upper_limit,
                strength_ratio=info.strength_ratio,
                is_prismatic=True,
            )
            self._prismatic_joints.append(ji)
            self._name_to_joint[info.name] = ji
            self._targets_mm[i] = info.target * 1000.0

    @property
    def revolute_joints(self) -> list[JointInfo]:
        return self._revolute_joints

    @property
    def prismatic_joints(self) -> list[JointInfo]:
        return self._prismatic_joints

    @property
    def all_joints(self) -> list[JointInfo]:
        return self._revolute_joints + self._prismatic_joints

    def get_joint(self, name_or_index: Union[str, int]) -> JointInfo:
        if isinstance(name_or_index, str):
            return self._name_to_joint[name_or_index]
        for j in self.all_joints:
            if j.index == name_or_index and not j.is_prismatic:
                return j
        raise KeyError(f"Joint {name_or_index} not found")

    # ---- Revolute joints ----

    def set_revolute_position(
        self,
        index: int,
        value: float,
        degree: bool = False,
    ) -> None:
        """Set target angle for a revolute joint.

        Args:
            index: Joint index in the revolute joint list.
            value: Target angle.
            degree: If True, value is in degrees; otherwise radians.
        """
        ji = self._revolute_joints[index]
        if degree:
            angle_rad = math.radians(value)
        else:
            angle_rad = value

        angle_rad = max(ji.lower_limit, min(ji.upper_limit, angle_rad))
        self._engine.set_revolute_target(index, angle_rad)
        self._targets_deg[index] = math.degrees(angle_rad)

    def set_revolute_initial_offset(self, index: int, offset_rad: float) -> None:
        """Set the initial angle offset for FK-posed joints.

        When a joint is pre-positioned via FK before parsing, this offset
        ensures the driving energy uses the correct reference so that
        target_angle semantics remain absolute URDF angles.
        """
        self._engine.set_revolute_initial_offset(index, offset_rad)

    def get_revolute_target_deg(self, index: int) -> float:
        return self._targets_deg.get(index, 0.0)

    # ---- Prismatic joints ----

    def set_prismatic_position(
        self,
        index: int,
        value: float,
        millimeters: bool = False,
    ) -> None:
        """Set target distance for a prismatic joint.

        Args:
            index: Joint index in the prismatic joint list.
            value: Target distance.
            millimeters: If True, value is in mm; otherwise metres.
        """
        ji = self._prismatic_joints[index]
        if millimeters:
            dist_m = value / 1000.0
        else:
            dist_m = value

        dist_m = max(ji.lower_limit, min(ji.upper_limit, dist_m))
        self._engine.set_prismatic_target(index, dist_m)
        self._targets_mm[index] = dist_m * 1000.0

    def get_prismatic_target_mm(self, index: int) -> float:
        return self._targets_mm.get(index, 0.0)

    # ---- Convenience ----

    def set_joint_position(
        self,
        name_or_index: Union[str, int],
        value: float,
        degree: bool = False,
    ) -> None:
        """Set position for any joint by name or index."""
        if isinstance(name_or_index, str):
            ji = self._name_to_joint[name_or_index]
            if ji.is_prismatic:
                self.set_prismatic_position(ji.index, value)
            else:
                self.set_revolute_position(ji.index, value, degree=degree)
        else:
            self.set_revolute_position(name_or_index, value, degree=degree)

    def reset_all(self) -> None:
        """Reset all joint targets to zero."""
        for i in range(len(self._revolute_joints)):
            self.set_revolute_position(i, 0.0)
        for i in range(len(self._prismatic_joints)):
            self.set_prismatic_position(i, 0.0)
