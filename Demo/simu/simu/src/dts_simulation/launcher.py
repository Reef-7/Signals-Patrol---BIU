"""Virtual-only launcher presentation state with no hardware-control capabilities."""

from dataclasses import dataclass

from .models import Point2D, Point3D


@dataclass(slots=True)
class VirtualLauncher:
    """Store a simulated aiming target for future visualization use only."""

    target_xy: Point2D | None = None
    target_xyz: Point3D | None = None

    def aim_at(self, target_xy: Point2D) -> None:
        self.target_xy = target_xy

    def aim_at_xyz(self, target_xyz: Point3D) -> None:
        """Store a virtual 3D aim target without firing or hardware control."""
        self.target_xyz = target_xyz
