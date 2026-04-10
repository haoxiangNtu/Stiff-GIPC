"""stiff_physics - Python interface for the StiffGIPC IPC physics engine."""


def __getattr__(name):
    if name in ("Engine", "Config"):
        from stiff_physics.engine import Engine, Config
        globals()["Engine"] = Engine
        globals()["Config"] = Config
        return globals()[name]
    if name in ("Robot", "JointInfo"):
        from stiff_physics.robot import Robot, JointInfo
        globals()["Robot"] = Robot
        globals()["JointInfo"] = JointInfo
        return globals()[name]
    if name == "Pipeline":
        from stiff_physics.pipeline import Pipeline
        globals()["Pipeline"] = Pipeline
        return Pipeline
    raise AttributeError(f"module 'stiff_physics' has no attribute {name!r}")


__all__ = ["Engine", "Config", "Robot", "JointInfo", "Pipeline"]
