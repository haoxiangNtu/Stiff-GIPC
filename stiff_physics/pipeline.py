"""Main simulation + Polyscope visualization pipeline.

Analogous to rbs_physics.pipeline.PipelineBase, this class wires together
the StiffGIPC engine, a Robot joint controller, and Polyscope rendering
with ImGui-based joint sliders.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional

import polyscope as ps
import polyscope.imgui as psim

from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot


class Pipeline:
    """Polyscope-driven simulation loop for StiffGIPC.

    Usage::

        pipeline = Pipeline(
            urdf_path="Assets/sim_data/urdf/xarm/xarm6_robot_white.urdf",
            global_scale=0.3,
            root_fixed=True,
        )
        pipeline.run()
    """

    def __init__(
        self,
        urdf_path: str,
        config: Optional[Config] = None,
        global_scale: float = 1.0,
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        root_fixed: bool = True,
        revolute_as_motor: bool = False,
        default_young: float = 1e7,
        up_dir: str = "y_up",
    ):
        # Polyscope (OpenGL) must init BEFORE the CUDA engine to avoid
        # GL context corrupting the CUDA context on some drivers.
        ps.init()
        ps.set_up_dir(up_dir)
        ps.set_ground_plane_mode("shadow_only")

        self.config = config or Config()
        self.engine = Engine(self.config)

        self.engine.load_urdf(
            urdf_path,
            scale=global_scale,
            translation=translation,
            root_fixed=root_fixed,
            revolute_as_motor=revolute_as_motor,
            default_young=default_young,
        )
        self.engine.finalize()

        self.robot = Robot(self.engine)
        self.is_running = False
        self._step_count = 0

        self._ps_mesh = None
        self._surf_faces = None

    def _setup_polyscope(self) -> None:
        verts = self.engine.get_vertices()
        self._surf_faces = self.engine.get_surface_faces()

        self._ps_mesh = ps.register_surface_mesh(
            "robot", verts, self._surf_faces, smooth_shade=True
        )
        self._ps_mesh.set_color((0.6, 0.7, 0.8))

        ps.set_user_callback(self._callback)

    def _callback(self) -> None:
        # ---- Control panel ----
        psim.SetNextWindowPos((10, 10), psim.ImGuiCond_FirstUseEver)
        psim.SetNextWindowSize((340, 0), psim.ImGuiCond_FirstUseEver)

        if psim.Begin("StiffGIPC Control"):
            # Run / Pause
            if self.is_running:
                if psim.Button("Pause"):
                    self.is_running = False
            else:
                if psim.Button("Run"):
                    self.is_running = True

            psim.SameLine()
            psim.Text(f"Step: {self._step_count}")

            psim.Separator()

            # Revolute joint sliders
            if self.robot.revolute_joints:
                psim.Text("Revolute Joints")
                psim.Separator()
                for i, ji in enumerate(self.robot.revolute_joints):
                    lo_deg = math.degrees(ji.lower_limit)
                    hi_deg = math.degrees(ji.upper_limit)
                    cur_deg = self.robot.get_revolute_target_deg(i)
                    changed, new_val = psim.SliderFloat(
                        ji.name, cur_deg, lo_deg, hi_deg
                    )
                    if changed:
                        self.robot.set_revolute_position(i, new_val, degree=True)

            # Prismatic joint sliders
            if self.robot.prismatic_joints:
                psim.Spacing()
                psim.Text("Prismatic Joints")
                psim.Separator()
                for i, ji in enumerate(self.robot.prismatic_joints):
                    lo_mm = ji.lower_limit * 1000.0
                    hi_mm = ji.upper_limit * 1000.0
                    cur_mm = self.robot.get_prismatic_target_mm(i)
                    changed, new_val = psim.SliderFloat(
                        ji.name, cur_mm, lo_mm, hi_mm
                    )
                    if changed:
                        self.robot.set_prismatic_position(
                            i, new_val, millimeters=True
                        )

            psim.Spacing()
            if psim.Button("Reset All Joints"):
                self.robot.reset_all()

            self.user_gui()

        psim.End()

        # ---- Simulation step ----
        if self.is_running:
            self.engine.step()
            self._step_count += 1
            self._update_mesh()

    def _update_mesh(self) -> None:
        verts = self.engine.get_vertices()
        self._ps_mesh.update_vertex_positions(verts)

    def user_gui(self) -> None:
        """Override in subclass to add custom ImGui controls."""
        pass

    def run(self) -> None:
        """Launch the Polyscope viewer and enter the main loop."""
        self._setup_polyscope()
        ps.show()
