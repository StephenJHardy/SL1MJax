"""Finite-pixel integration plans and the NumPy subcell reference.

Integration nodes are deterministic samples of a fitted parent. They are
not extra sky parameters. A uniform square of width ``w`` at depth ``d``
becomes ``4**d`` equal subcells of width ``w / 2**d``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import numpy as np
from jax import Array
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_aware_imaging import (
    SkyBasisType,
    SkyComponent,
    SkyComponentTable,
    VoltageIntegrationMode,
)
from sl1mjax.beam_conventions import BeamCalibrationState, require_beam_calibration_state
from sl1mjax.beam_operator import (
    BeamOperatorConfig,
    BeamOperatorResult,
    SkyStokesPlanes,
    adjoint_voltage_beam,
    predict_voltage_beam,
)
from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import circular_parallactic_jones
from sl1mjax.sky import GaussianApproximation
from sl1mjax.voltage_beam import BeamCoordinates, BeamEvaluation, _pointing_relative_lm

NODE_BUCKET_SIZES = (
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1_048_576,
)


class IntegrationParentPolicy(StrEnum):
    """Which fitted parents the integration plan keeps."""

    ALL_ACTIVE = "all_active"
    POSITIVE_FLUX = "positive_flux"
MODE_DEPTH = {
    VoltageIntegrationMode.POINT_CENTRE: 0,
    VoltageIntegrationMode.ANALYTIC_SQUARE: 0,
    VoltageIntegrationMode.SUBCELL_2X2: 1,
    VoltageIntegrationMode.SUBCELL_4X4: 2,
}


@dataclass(frozen=True)
class IntegrationPlan:
    """Packed numerical nodes in one mosaic tangent frame.

    ``l_rad`` and ``m_rad`` are mosaic-frame direction cosines. Prediction
    transforms them into each block's phase-centre frame. Beam-only
    pointing offsets stay on ``BeamOperatorConfig``.
    """

    parent_index: NDArray[np.int32]
    l_rad: NDArray[np.float64]
    m_rad: NDArray[np.float64]
    width_rad: NDArray[np.float64]
    weight: NDArray[np.float64]
    node_valid: NDArray[np.bool_]
    parent_id: tuple[str, ...]
    mode: tuple[str, ...]
    mosaic_phase_centre_rad: tuple[float, float]
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD

    def __post_init__(self) -> None:
        sizes = {
            self.parent_index.size,
            self.l_rad.size,
            self.m_rad.size,
            self.width_rad.size,
            self.weight.size,
            self.node_valid.size,
        }
        if len(sizes) != 1 or 0 in sizes:
            raise ValueError("integration-plan arrays must be nonempty and aligned")
        if len(self.parent_id) != self.parent_count:
            raise ValueError("parent_id must contain one label per fitted parent")
        if np.any(self.weight[self.node_valid] <= 0.0):
            raise ValueError("valid integration nodes must have positive weight")
        if np.any(self.width_rad < 0.0):
            raise ValueError("node widths must be non-negative")
        if len(self.mosaic_phase_centre_rad) != 2 or not np.all(
            np.isfinite(self.mosaic_phase_centre_rad)
        ):
            raise ValueError("mosaic_phase_centre_rad must contain finite RA and Dec")

    @property
    def capacity(self) -> int:
        return int(self.node_valid.size)

    @property
    def node_count(self) -> int:
        return int(np.count_nonzero(self.node_valid))

    @property
    def parent_count(self) -> int:
        if not np.any(self.node_valid):
            raise ValueError("integration plan has no valid nodes")
        return int(np.max(self.parent_index[self.node_valid]) + 1)

    def node_flux(self, parent_flux: ArrayLike) -> NDArray[np.float64]:
        """Gather parent coefficients onto nodes and apply fixed weights."""

        values = np.asarray(parent_flux, dtype=np.float64).reshape(-1)
        if values.size != self.parent_count:
            raise ValueError("parent_flux must match the number of fitted parents")
        flux = values[self.parent_index] * self.weight
        return np.where(self.node_valid, flux, 0.0)

    def reduce_node_gradient(
        self, node_gradient: ArrayLike
    ) -> NDArray[np.float64]:
        """Scatter node derivatives back onto fitted parents."""

        values = np.asarray(node_gradient, dtype=np.float64)
        if values.ndim == 2:
            values = values.sum(axis=1)
        if values.shape[0] != self.capacity:
            raise ValueError("node gradient must match the packed plan")
        parent = np.zeros(self.parent_count, dtype=np.float64)
        valid = self.node_valid
        np.add.at(
            parent,
            self.parent_index[valid],
            values[valid] * self.weight[valid],
        )
        return parent

    def local_directions(
        self, block_phase_centre_rad: tuple[float, float]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Rotate mosaic-frame nodes into a pointing's tangent plane."""

        mosaic = np.asarray(self.mosaic_phase_centre_rad, dtype=np.float64)
        block = np.asarray(block_phase_centre_rad, dtype=np.float64)
        if np.allclose(mosaic, block, rtol=0.0, atol=1.0e-15):
            local_l = np.where(self.node_valid, self.l_rad, 0.0)
            local_m = np.where(self.node_valid, self.m_rad, 0.0)
            return (
                np.asarray(local_l, dtype=np.float64),
                np.asarray(local_m, dtype=np.float64),
            )
        sky_ra, sky_dec = lmn_to_radec(
            self.mosaic_phase_centre_rad[0],
            self.mosaic_phase_centre_rad[1],
            self.l_rad,
            self.m_rad,
        )
        local_l, local_m, _n = radec_to_lmn(
            block_phase_centre_rad[0],
            block_phase_centre_rad[1],
            sky_ra,
            sky_dec,
        )
        local_l = np.where(self.node_valid, local_l, 0.0)
        local_m = np.where(self.node_valid, local_m, 0.0)
        return (
            np.asarray(local_l, dtype=np.float64),
            np.asarray(local_m, dtype=np.float64),
        )


@dataclass(frozen=True)
class ManufacturedVoltageBeam:
    """Polynomial Jones used as a manufactured-beam oracle.

    ``Jones(l, m) = C0 + Cl l + Cm m + Cll l^2 + Clm l m + Cmm m^2``.
    Coordinates are the pointing-relative directions already passed to
    ``evaluate``. Optional parallactic rotation uses CASA circular P.
    """

    intercept: np.ndarray
    grad_l: np.ndarray | None = None
    grad_m: np.ndarray | None = None
    hess_ll: np.ndarray | None = None
    hess_lm: np.ndarray | None = None
    hess_mm: np.ndarray | None = None
    valid_radius_rad: float | None = None
    off_diagonal_radius_rad: float | None = None
    off_diagonal_valid: bool = True
    rotate_parallactic: bool = False
    model_id: str = "manufactured_voltage"

    @property
    def antenna_planes_from_parallactic(self) -> bool:
        return self.rotate_parallactic

    def evaluate(
        self,
        coordinates: BeamCoordinates,
        *,
        calibration_state: BeamCalibrationState | str,
    ) -> BeamEvaluation:
        state = require_beam_calibration_state(calibration_state)
        l_rad, m_rad = _pointing_relative_lm(coordinates)
        n_channel = int(np.asarray(coordinates.frequency_hz).size)
        plane = _polynomial_jones(
            self,
            l_rad,
            m_rad,
        )
        if self.rotate_parallactic:
            chi = np.asarray(coordinates.parallactic_angle_rad, dtype=np.float64)
            if chi.size == 1:
                chi = np.full(1, float(chi))
            rotated = []
            for angle in chi.reshape(-1):
                rotated.append(_apply_circular_p(plane, float(angle), state))
            jones = np.stack(rotated, axis=0)
        else:
            jones = plane[None, ...]
        jones = np.broadcast_to(
            jones[:, :, None, :, :],
            (jones.shape[0], l_rad.size, n_channel, 2, 2),
        ).copy()
        valid = np.ones(jones.shape[:3], dtype=bool)
        if self.valid_radius_rad is not None:
            inside = (l_rad * l_rad + m_rad * m_rad) <= self.valid_radius_rad**2
            valid &= inside[None, :, None]
        leakage = valid if self.off_diagonal_valid else np.zeros_like(valid)
        if self.off_diagonal_radius_rad is not None:
            leak_inside = (l_rad * l_rad + m_rad * m_rad) <= (
                self.off_diagonal_radius_rad**2
            )
            leakage = leakage & leak_inside[None, :, None]
        return BeamEvaluation(
            jones=jones,
            valid=valid,
            provenance={"model_id": self.model_id},
            off_diagonal_valid=leakage,
        )


def subcell_nodes(
    l_rad: float,
    m_rad: float,
    width_rad: float,
    depth: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Return subcell centres, width, and parent-flux weight for one square.

    Centres form a regular ``2**depth`` by ``2**depth`` grid that tiles the
    parent square without gaps, overlap, reflection, or axis exchange.
    """

    if depth < 0:
        raise ValueError("subdivision depth must be non-negative")
    if not np.isfinite(width_rad) or width_rad <= 0.0:
        raise ValueError("square width_rad must be finite and positive")
    n_side = 2**depth
    sub_width = width_rad / n_side
    half = (n_side - 1) / 2.0
    iy, ix = np.meshgrid(
        np.arange(n_side, dtype=np.float64),
        np.arange(n_side, dtype=np.float64),
        indexing="ij",
    )
    d_l = (ix - half) * sub_width
    d_m = (iy - half) * sub_width
    return (
        np.asarray(l_rad + d_l.reshape(-1), dtype=np.float64),
        np.asarray(m_rad + d_m.reshape(-1), dtype=np.float64),
        float(sub_width),
        float(1.0 / (n_side * n_side)),
    )


def choose_node_capacity(node_count: int, capacity: int | None = None) -> int:
    """Return the smallest fixed node bucket that can hold ``node_count``."""

    if node_count < 1:
        raise ValueError("node_count must be positive")
    if capacity is not None:
        if capacity < node_count:
            raise ValueError("requested node capacity is smaller than the plan")
        return int(capacity)
    for size in NODE_BUCKET_SIZES:
        if size >= node_count:
            return size
    raise ValueError(
        f"node count {node_count} exceeds the largest bucket {NODE_BUCKET_SIZES[-1]}"
    )


def integration_plan_from_table(
    table: SkyComponentTable,
    *,
    mode: VoltageIntegrationMode | str = VoltageIntegrationMode.ANALYTIC_SQUARE,
    depth_by_parent: dict[str, int] | None = None,
    parent_policy: IntegrationParentPolicy | str = IntegrationParentPolicy.ALL_ACTIVE,
    include_zero_flux: bool | None = None,
    include_inactive: bool = False,
    pad: bool = False,
    capacity: int | None = None,
    approximation: GaussianApproximation | str = GaussianApproximation.WIDE_FIELD,
) -> IntegrationPlan:
    """Expand fitted sky components into packed integration nodes.

    The imaging default keeps every active parent, including zero-flux
    leaves that the optimizer must be able to fill. Pass
    ``parent_policy='positive_flux'`` only for the sealed transfer
    diagnostic that dropped non-positive atoms.
    """

    selected_mode = VoltageIntegrationMode(mode)
    policy = IntegrationParentPolicy(parent_policy)
    keep_nonpositive = (
        include_zero_flux
        if include_zero_flux is not None
        else policy is IntegrationParentPolicy.ALL_ACTIVE
    )
    selected = [
        component
        for component in table.components
        if (include_inactive or component.active)
        and (keep_nonpositive or component.stokes_i_jy > 0.0)
    ]
    if not selected:
        raise ValueError("sky table has no selected parents")
    parent_index: list[int] = []
    l_rad: list[float] = []
    m_rad: list[float] = []
    width_rad: list[float] = []
    weight: list[float] = []
    parent_ids = tuple(component.component_id for component in selected)
    node_modes: list[str] = []
    for parent, component in enumerate(selected):
        depth = _component_depth(component, selected_mode, depth_by_parent)
        if component.basis_type is SkyBasisType.GAUSSIAN:
            raise ValueError(
                "Gaussian finite-pixel integration is not implemented; "
                "declare a delta or uniform_square basis"
            )
        if component.basis_type not in {
            SkyBasisType.DELTA,
            SkyBasisType.UNIFORM_SQUARE,
        }:
            raise ValueError(
                f"unsupported sky basis {component.basis_type.value!r}"
            )
        use_delta = (
            selected_mode is VoltageIntegrationMode.POINT_CENTRE
            or component.basis_type is SkyBasisType.DELTA
            or component.width_rad == 0.0
        )
        if use_delta:
            parent_index.append(parent)
            l_rad.append(component.l_rad)
            m_rad.append(component.m_rad)
            width_rad.append(0.0)
            weight.append(1.0)
            node_modes.append(VoltageIntegrationMode.POINT_CENTRE.value)
            continue
        node_l, node_m, node_width, node_weight = subcell_nodes(
            component.l_rad,
            component.m_rad,
            component.width_rad,
            depth,
        )
        parent_index.extend(int(parent) for _ in node_l)
        l_rad.extend(float(value) for value in node_l)
        m_rad.extend(float(value) for value in node_m)
        width_rad.extend(node_width for _ in node_l)
        weight.extend(node_weight for _ in node_l)
        label = _mode_for_depth(depth)
        node_modes.extend(label for _ in node_l)
    plan = IntegrationPlan(
        parent_index=np.asarray(parent_index, dtype=np.int32),
        l_rad=np.asarray(l_rad, dtype=np.float64),
        m_rad=np.asarray(m_rad, dtype=np.float64),
        width_rad=np.asarray(width_rad, dtype=np.float64),
        weight=np.asarray(weight, dtype=np.float64),
        node_valid=np.ones(len(parent_index), dtype=bool),
        parent_id=parent_ids,
        mode=tuple(node_modes),
        mosaic_phase_centre_rad=_phase_centre(table.mosaic_phase_centre_rad),
        approximation=GaussianApproximation(approximation),
    )
    _reject_weight_drift(plan)
    if pad or capacity is not None:
        return pad_integration_plan(plan, capacity=capacity)
    return plan


def pad_integration_plan(
    plan: IntegrationPlan,
    *,
    capacity: int | None = None,
) -> IntegrationPlan:
    """Pad nodes into a fixed-capacity bucket. Masks are generated here."""

    selected = choose_node_capacity(plan.node_count, capacity)
    if selected == plan.capacity and bool(np.all(plan.node_valid)):
        return plan
    extra = selected - plan.node_count
    valid = plan.node_valid
    return IntegrationPlan(
        parent_index=np.concatenate(
            (plan.parent_index[valid], np.zeros(extra, dtype=np.int32))
        ),
        l_rad=np.concatenate((plan.l_rad[valid], np.zeros(extra))),
        m_rad=np.concatenate((plan.m_rad[valid], np.zeros(extra))),
        width_rad=np.concatenate((plan.width_rad[valid], np.zeros(extra))),
        weight=np.concatenate((plan.weight[valid], np.zeros(extra))),
        node_valid=np.concatenate(
            (np.ones(plan.node_count, dtype=bool), np.zeros(extra, dtype=bool))
        ),
        parent_id=plan.parent_id,
        mode=tuple(mode for mode, keep in zip(plan.mode, valid, strict=True) if keep)
        + ("pad",) * extra,
        mosaic_phase_centre_rad=plan.mosaic_phase_centre_rad,
        approximation=plan.approximation,
    )


def predict_voltage_from_plan(
    block: VisibilityBlock,
    plan: IntegrationPlan,
    parent_flux: ArrayLike,
    beam: Any,
    *,
    antenna_position_m: ArrayLike,
    calibration_state: BeamCalibrationState | str,
    config: BeamOperatorConfig | None = None,
    backend: str = "numpy",
    split_parents: bool = False,
) -> BeamOperatorResult:
    """Predict visibilities from fitted parent fluxes and a frozen plan."""

    sky = SkyStokesPlanes(stokes_i=plan.node_flux(parent_flux))
    local_l, local_m = plan.local_directions(block.phase_centre_rad)
    kwargs = dict(
        block=block,
        l_rad=local_l,
        m_rad=local_m,
        sky=sky,
        beam=beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
        width_rad=plan.width_rad,
        node_valid=plan.node_valid,
        kernel_approximation=plan.approximation,
    )
    if split_parents:
        if backend != "numpy":
            raise ValueError("split_parents is only implemented for the NumPy backend")
        kwargs["parent_index"] = plan.parent_index
    if backend == "numpy":
        return predict_voltage_beam(**kwargs)
    if backend == "jax":
        from sl1mjax.voltage_operator_jax import predict_voltage_beam_jax

        return predict_voltage_beam_jax(**kwargs)
    raise ValueError(f"unknown finite-pixel backend {backend!r}")


def adjoint_voltage_from_plan(
    residual: ArrayLike,
    block: VisibilityBlock,
    plan: IntegrationPlan,
    beam: Any,
    *,
    antenna_position_m: ArrayLike,
    calibration_state: BeamCalibrationState | str,
    config: BeamOperatorConfig | None = None,
    backend: str = "numpy",
) -> NDArray[np.float64]:
    """Return the Stokes-I adjoint reduced onto fitted parents."""

    local_l, local_m = plan.local_directions(block.phase_centre_rad)
    if backend == "jax":
        import jax.numpy as jnp

        from sl1mjax.voltage_operator_jax import _adjoint_voltage_beam_jax_arrays

        return cast(
            NDArray[np.float64],
            np.asarray(
                _adjoint_voltage_beam_jax_arrays(
                    jnp.asarray(residual),
                    block,
                    local_l,
                    local_m,
                    beam,
                    antenna_position_m=np.asarray(antenna_position_m),
                    calibration_state=calibration_state,
                    config=config,
                    width_rad=plan.width_rad,
                    node_valid=plan.node_valid,
                    kernel_approximation=plan.approximation,
                    parent_index=plan.parent_index,
                    node_weight=plan.weight,
                    parent_count=plan.parent_count,
                ),
                dtype=np.float64,
            ),
        )
    if backend != "numpy":
        raise ValueError(f"unknown finite-pixel adjoint backend {backend!r}")
    stokes_i, _q, _u, _v = adjoint_voltage_beam(
        residual,
        block,
        local_l,
        local_m,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
        width_rad=plan.width_rad,
        node_valid=plan.node_valid,
        kernel_approximation=plan.approximation,
    )
    return plan.reduce_node_gradient(stokes_i)


def predict_voltage_from_plan_value_and_grad(
    parent_flux: Any,
    block: VisibilityBlock,
    plan: IntegrationPlan,
    beam: Any,
    *,
    antenna_position_m: Any,
    calibration_state: BeamCalibrationState | str,
    config: BeamOperatorConfig | None = None,
    train_mask: Any = None,
    operator_mode: str = "vjp",
) -> tuple[Array, Array]:
    """Stokes-I value and parent gradient for a frozen JAX integration plan.

    ``operator_mode="vjp"`` is the tiled reverse-mode oracle.
    ``operator_mode="explicit_jax"`` streams one forward and one adjoint.
    """

    from sl1mjax.voltage_operator_jax import (
        predict_voltage_from_plan_value_and_grad_explicit_jax,
        predict_voltage_from_plan_value_and_grad_jax,
    )

    if operator_mode == "explicit_jax":
        return predict_voltage_from_plan_value_and_grad_explicit_jax(
            parent_flux,
            block,
            plan,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
            train_mask=train_mask,
        )
    if operator_mode != "vjp":
        raise ValueError("operator_mode must be 'vjp' or 'explicit_jax'")
    return predict_voltage_from_plan_value_and_grad_jax(
        parent_flux,
        block,
        plan,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
        train_mask=train_mask,
    )


def _component_depth(
    component: SkyComponent,
    mode: VoltageIntegrationMode,
    depth_by_parent: dict[str, int] | None,
) -> int:
    if depth_by_parent is not None and component.component_id in depth_by_parent:
        depth = int(depth_by_parent[component.component_id])
    else:
        depth = MODE_DEPTH[mode]
    if depth < 0:
        raise ValueError("subdivision depth must be non-negative")
    return depth


def _mode_for_depth(depth: int) -> str:
    inverse = {
        value: key.value
        for key, value in MODE_DEPTH.items()
        if key is not VoltageIntegrationMode.POINT_CENTRE
    }
    return inverse.get(depth, f"subcell_{2**depth}x{2**depth}")


def _phase_centre(phase_centre_rad: tuple[float, float]) -> tuple[float, float]:
    if len(phase_centre_rad) != 2:
        raise ValueError("mosaic_phase_centre_rad must contain two values")
    return (float(phase_centre_rad[0]), float(phase_centre_rad[1]))


def _reject_weight_drift(plan: IntegrationPlan) -> None:
    for parent in range(plan.parent_count):
        selected = plan.node_valid & (plan.parent_index == parent)
        total = float(plan.weight[selected].sum())
        if abs(total - 1.0) > 1.0e-12:
            raise ValueError(
                f"parent {plan.parent_id[parent]} node weights sum to {total}"
            )


def _coefficient(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros((2, 2), dtype=np.complex128)
    array = np.asarray(value, dtype=np.complex128)
    if array.shape != (2, 2):
        raise ValueError("manufactured Jones coefficients must have shape (2, 2)")
    return array


def _polynomial_jones(
    beam: ManufacturedVoltageBeam,
    l_rad: np.ndarray,
    m_rad: np.ndarray,
) -> np.ndarray:
    intercept = _coefficient(beam.intercept)
    plane = np.broadcast_to(intercept, (l_rad.size, 2, 2)).copy()
    plane = plane + _coefficient(beam.grad_l) * l_rad[:, None, None]
    plane = plane + _coefficient(beam.grad_m) * m_rad[:, None, None]
    plane = plane + _coefficient(beam.hess_ll) * (l_rad * l_rad)[:, None, None]
    plane = plane + _coefficient(beam.hess_lm) * (l_rad * m_rad)[:, None, None]
    plane = plane + _coefficient(beam.hess_mm) * (m_rad * m_rad)[:, None, None]
    return np.asarray(plane, dtype=np.complex128)


def _apply_circular_p(
    jones: np.ndarray,
    chi: float,
    state: BeamCalibrationState,
) -> np.ndarray:
    parallactic = circular_parallactic_jones(chi)
    if state is BeamCalibrationState.CASA_PARANG_TRUE:
        conjugate = np.conjugate(np.transpose(parallactic))
        return np.asarray(
            np.einsum("ij,...jk,kl->...il", conjugate, jones, parallactic),
            dtype=np.complex128,
        )
    return np.asarray(
        np.einsum("...ij,jk->...ik", jones, parallactic),
        dtype=np.complex128,
    )
