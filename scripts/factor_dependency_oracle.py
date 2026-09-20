#!/usr/bin/env python3
"""CPU dependency oracle for factor-streamed FFT layouts.

The oracle intentionally obtains inter-factor fan-in by matching enumerated
producer output addresses with enumerated consumer input addresses.  The
closed-form dependency and packet rules in this file are only used as an
independent verification/diagnostic path.

For a factor with log-size ``Lg`` and the sum of preceding logs ``Done`` the
layout is::

    P = 2**Done, F = 2**Lg, S = 2**(Total-Done-Lg)
    j = s*P + p
    input[row]  = row*(P*S) + j
    output[u]   = (s*F + u)*P + p

The optional batch dimension is a compact batch-major prefix of ``2**Total``
elements.  It is part of the address enumeration so that cross-batch
dependencies cannot be hidden by using local addresses.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from typing import Iterable, Mapping, Sequence, Tuple


TransformId = Tuple[int, int]  # (batch, local transform j)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _require_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class FactorGroup:
    """One factor's logical shape and transform coordinate helpers."""

    index: int
    log: int
    done: int
    P: int
    F: int
    S: int

    @property
    def transform_count(self) -> int:
        return self.P * self.S

    def transform_id(self, s: int, p: int) -> int:
        if not 0 <= s < self.S or not 0 <= p < self.P:
            raise ValueError(
                f"group {self.index} transform coordinates out of range: "
                f"s={s}, p={p}, shape S={self.S}, P={self.P}"
            )
        return s * self.P + p

    def coordinates(self, j: int) -> tuple[int, int]:
        if not 0 <= j < self.transform_count:
            raise ValueError(
                f"group {self.index} transform j={j} outside [0, {self.transform_count})"
            )
        return divmod(j, self.P)

    def transforms(self) -> range:
        return range(self.transform_count)


@dataclass(frozen=True)
class AddressMatch:
    """One consumer input element matched to one producer output element."""

    batch: int
    consumer_j: int
    row: int
    address: int
    producer_j: int
    producer_u: int

    def as_dict(self) -> dict[str, int]:
        return {
            "batch": self.batch,
            "consumer_j": self.consumer_j,
            "row": self.row,
            "address": self.address,
            "producer_j": self.producer_j,
            "producer_u": self.producer_u,
        }


@dataclass(frozen=True)
class EdgeDependency:
    """Enumerated dependencies between two adjacent factor groups.

    ``fan_in`` is keyed by ``(batch, consumer_j)`` and contains the matching
    ``(batch, producer_j)`` ids.  ``matches`` retains the address-level proof;
    it is useful when a failed verification needs to be diagnosed rather than
    just reported as a set mismatch.
    """

    producer: FactorGroup
    consumer: FactorGroup
    fan_in: Mapping[TransformId, frozenset[TransformId]]
    matches: Mapping[TransformId, tuple[AddressMatch, ...]]
    producer_to_consumers: Mapping[TransformId, frozenset[TransformId]]

    @property
    def producer_group(self) -> int:
        return self.producer.index

    @property
    def consumer_group(self) -> int:
        return self.consumer.index

    def fan_in_for(self, batch: int, consumer_j: int) -> frozenset[int]:
        """Return local producer ``j`` ids for one consumer transform."""

        key = (batch, consumer_j)
        try:
            return frozenset(j for _, j in self.fan_in[key])
        except KeyError as exc:
            raise KeyError(f"unknown consumer transform {key}") from exc

    def matches_for(self, batch: int, consumer_j: int) -> tuple[AddressMatch, ...]:
        try:
            return self.matches[(batch, consumer_j)]
        except KeyError as exc:
            raise KeyError(f"unknown consumer transform {(batch, consumer_j)}") from exc

    def as_dict(self, *, include_matches: bool = False) -> dict[str, object]:
        consumers: list[dict[str, object]] = []
        for key in sorted(self.fan_in):
            producer_ids = sorted(self.fan_in[key])
            record: dict[str, object] = {
                "batch": key[0],
                "consumer_j": key[1],
                "producer_transforms": [
                    {"batch": batch, "j": j} for batch, j in producer_ids
                ],
            }
            if include_matches:
                record["matches"] = [m.as_dict() for m in self.matches[key]]
            consumers.append(record)
        producers = [
            {
                "batch": key[0],
                "producer_j": key[1],
                "consumer_transforms": [
                    {"batch": batch, "j": j} for batch, j in sorted(consumers_for)
                ],
            }
            for key, consumers_for in sorted(self.producer_to_consumers.items())
        ]
        return {
            "producer_group": self.producer.index,
            "consumer_group": self.consumer.index,
            "fan_in": consumers,
            "producer_to_consumers": producers,
        }


@dataclass(frozen=True)
class PacketLayout:
    """Last-factor input packet partition.

    Packets partition ``s mod K`` for every group before the last factor.  The
    last factor has no ``s`` packet dimension (its ``S`` is one), so a
    dependency entering that factor intentionally fans into every packet.
    """

    total_log: int
    last_log: int
    K: int
    slices: int
    tail_width: int
    columns: int | None

    @property
    def packet_count(self) -> int:
        return self.slices

    def packet_range(self, packet: int) -> range:
        if not 0 <= packet < self.slices:
            raise ValueError(f"packet {packet} outside [0, {self.slices})")
        start = packet * self.tail_width
        return range(start, start + self.tail_width)

    def packet_for_s(self, s: int) -> int:
        if s < 0:
            raise ValueError("s must be non-negative")
        return (s % self.K) // self.tail_width

    def transform_ids(self, group: FactorGroup, packet: int) -> frozenset[int]:
        """Return compact transform ids belonging to ``packet``.

        This method applies only to groups before the last factor.  The
        compact form is exactly ``j=(outer*K+tail)*P+p`` from the contract.
        """

        if group.S < self.K:
            raise ValueError(
                f"group {group.index} has S={group.S} smaller than last-factor K={self.K}"
            )
        ids = {
            (outer * self.K + tail) * group.P + p
            for outer in range(group.S // self.K)
            for tail in self.packet_range(packet)
            for p in range(group.P)
        }
        return frozenset(ids)

    def as_dict(self) -> dict[str, int | None]:
        return {
            "K": self.K,
            "slices": self.slices,
            "tail_width": self.tail_width,
            "columns": self.columns,
        }


@dataclass(frozen=True)
class DependencyOracle:
    """Complete enumerated dependency graph and independent validations."""

    total_log: int
    factor_logs: tuple[int, ...]
    batch: int
    groups: tuple[FactorGroup, ...]
    edges: tuple[EdgeDependency, ...]
    packets: PacketLayout | None

    @property
    def N(self) -> int:
        return 1 << self.total_log

    @property
    def edge_map(self) -> dict[tuple[int, int], EdgeDependency]:
        return {(edge.producer.index, edge.consumer.index): edge for edge in self.edges}

    def edge(self, producer_group: int) -> EdgeDependency:
        if not 0 <= producer_group < len(self.edges):
            raise IndexError(f"no edge after group {producer_group}")
        return self.edges[producer_group]

    def packet_transform_ids(self, group_index: int, packet: int) -> frozenset[int]:
        if self.packets is None:
            raise ValueError("packetization was not requested")
        if group_index < 0:
            raise ValueError("group index must be non-negative")
        if group_index >= len(self.groups) - 1:
            raise ValueError("packet transform sets are defined only before the last factor")
        return self.packets.transform_ids(self.groups[group_index], packet)

    def verification(self) -> dict[str, object]:
        """Run all independent checks and return a machine-readable report."""

        report: dict[str, object] = {
            "address_fanin": verify_address_fanin(self),
            "closed_form_fanin": verify_closed_form_fanin(self),
        }
        if self.packets is not None:
            report["packets"] = verify_packet_analytic(self)
        report["ok"] = all(
            bool(value.get("ok"))
            for value in report.values()
            if isinstance(value, dict) and "ok" in value
        )
        return report

    def as_dict(self, *, include_matches: bool = False) -> dict[str, object]:
        return {
            "total_log": self.total_log,
            "factor_logs": list(self.factor_logs),
            "batch": self.batch,
            "N": self.N,
            "groups": [
                {
                    "index": group.index,
                    "log": group.log,
                    "done": group.done,
                    "P": group.P,
                    "F": group.F,
                    "S": group.S,
                }
                for group in self.groups
            ],
            "edges": [
                edge.as_dict(include_matches=include_matches) for edge in self.edges
            ],
            "packets": self.packets.as_dict() if self.packets is not None else None,
            "verification": self.verification(),
        }


def make_groups(total_log: int, factor_logs: Sequence[int]) -> tuple[FactorGroup, ...]:
    """Build and validate the factor shapes from the mathematical contract."""

    total_log = _require_int("total_log", total_log)
    if total_log < 1:
        raise ValueError("total_log must be positive")
    logs = tuple(_require_int("factor log", value) for value in factor_logs)
    if not logs or any(value < 1 for value in logs):
        raise ValueError("factor_logs must contain positive logs")
    if sum(logs) != total_log:
        raise ValueError(
            f"factor logs sum to {sum(logs)}, but total_log is {total_log}"
        )

    groups: list[FactorGroup] = []
    done = 0
    for index, log in enumerate(logs):
        groups.append(
            FactorGroup(
                index=index,
                log=log,
                done=done,
                P=1 << done,
                F=1 << log,
                S=1 << (total_log - done - log),
            )
        )
        done += log
    return tuple(groups)


def _validate_batch(batch: int) -> int:
    batch = _require_int("batch", batch)
    if batch < 1:
        raise ValueError("batch must be positive")
    return batch


def _transform_refs(group: FactorGroup, batch: int) -> Iterable[TransformId]:
    for b in range(batch):
        for j in group.transforms():
            yield (b, j)


def enumerate_input_addresses(
    total_log: int,
    group: FactorGroup,
    *,
    batch: int = 1,
) -> dict[TransformId, tuple[int, ...]]:
    """Enumerate every input address consumed by every group transform."""

    batch = _validate_batch(batch)
    N = 1 << total_log
    addresses: dict[TransformId, tuple[int, ...]] = {}
    for b, j in _transform_refs(group, batch):
        s, p = group.coordinates(j)
        base = b * N
        addresses[(b, j)] = tuple(
            base + row * (group.P * group.S) + s * group.P + p
            for row in range(group.F)
        )
    return addresses


def enumerate_output_addresses(
    total_log: int,
    group: FactorGroup,
    *,
    batch: int = 1,
) -> dict[TransformId, tuple[int, ...]]:
    """Enumerate every output address written by every group transform."""

    batch = _validate_batch(batch)
    N = 1 << total_log
    addresses: dict[TransformId, tuple[int, ...]] = {}
    for b, j in _transform_refs(group, batch):
        s, p = group.coordinates(j)
        base = b * N
        addresses[(b, j)] = tuple(
            base + (s * group.F + u) * group.P + p for u in range(group.F)
        )
    return addresses


# Short aliases make the address-level API convenient in exploratory tests.
enumerate_inputs = enumerate_input_addresses
enumerate_outputs = enumerate_output_addresses


def _reverse_output_addresses(
    total_log: int, producer: FactorGroup, *, batch: int
) -> dict[int, tuple[TransformId, int]]:
    reverse: dict[int, tuple[TransformId, int]] = {}
    for transform, addresses in enumerate_output_addresses(
        total_log, producer, batch=batch
    ).items():
        for u, address in enumerate(addresses):
            if address in reverse:
                previous = reverse[address]
                raise AssertionError(
                    "producer output address is written twice: "
                    f"{address} by {previous} and {(transform, u)}"
                )
            reverse[address] = (transform, u)
    expected = batch * (1 << total_log)
    if len(reverse) != expected:
        raise AssertionError(
            f"producer group {producer.index} covers {len(reverse)} addresses, "
            f"expected {expected}"
        )
    return reverse


def enumerate_edge_dependency(
    total_log: int,
    producer: FactorGroup,
    consumer: FactorGroup,
    *,
    batch: int = 1,
) -> EdgeDependency:
    """Derive one edge's fan-in solely from enumerated address matching."""

    if consumer.index != producer.index + 1:
        raise ValueError("only adjacent factor groups form an edge")
    batch = _validate_batch(batch)
    output_owner = _reverse_output_addresses(total_log, producer, batch=batch)
    consumer_inputs = enumerate_input_addresses(total_log, consumer, batch=batch)
    fan_in: dict[TransformId, frozenset[TransformId]] = {}
    matches: dict[TransformId, tuple[AddressMatch, ...]] = {}
    producer_to_consumers: dict[TransformId, set[TransformId]] = {}
    for consumer_id, addresses in consumer_inputs.items():
        consumer_matches: list[AddressMatch] = []
        producer_ids: set[TransformId] = set()
        for row, address in enumerate(addresses):
            try:
                producer_id, producer_u = output_owner[address]
            except KeyError as exc:
                raise AssertionError(
                    f"consumer input address {address} has no producer output"
                ) from exc
            producer_ids.add(producer_id)
            consumer_matches.append(
                AddressMatch(
                    batch=consumer_id[0],
                    consumer_j=consumer_id[1],
                    row=row,
                    address=address,
                    producer_j=producer_id[1],
                    producer_u=producer_u,
                )
            )
            producer_to_consumers.setdefault(producer_id, set()).add(consumer_id)
        if len(producer_ids) != consumer.F:
            raise AssertionError(
                f"consumer {consumer_id} has {len(producer_ids)} producer transforms, "
                f"expected {consumer.F}"
            )
        fan_in[consumer_id] = frozenset(producer_ids)
        matches[consumer_id] = tuple(consumer_matches)

    return EdgeDependency(
        producer=producer,
        consumer=consumer,
        fan_in=fan_in,
        matches=matches,
        producer_to_consumers={
            key: frozenset(value) for key, value in producer_to_consumers.items()
        },
    )


def make_packet_layout(
    total_log: int,
    factor_logs: Sequence[int],
    *,
    slices: int = 1,
    columns: int | None = None,
) -> PacketLayout:
    """Validate and construct the last-factor packet partition."""

    groups = make_groups(total_log, factor_logs)
    slices = _require_int("slices", slices)
    if not _is_power_of_two(slices):
        raise ValueError("slices must be a positive power of two")
    K = groups[-1].F
    if slices > K:
        raise ValueError(f"slices={slices} exceeds last-factor K={K}")
    tail_width = K // slices
    if columns is not None:
        columns = _require_int("columns", columns)
        if columns < 1:
            raise ValueError("columns must be positive")
        # The first-factor packet is the only one whose packet tail supplies
        # its columns.  Subsequent factors get columns from their P dimension.
        if tail_width < columns or tail_width % columns:
            raise ValueError(
                "first factor packet width K/slices must be >= Columns and "
                "divisible by Columns"
            )
        for group in groups[1:]:
            if group.P < columns:
                raise ValueError(
                    f"group {group.index} has P={group.P}, below Columns={columns}"
                )
    return PacketLayout(
        total_log=total_log,
        last_log=groups[-1].log,
        K=K,
        slices=slices,
        tail_width=tail_width,
        columns=columns,
    )


def build_dependency_oracle(
    total_log: int,
    factor_logs: Sequence[int],
    *,
    batch: int = 1,
    slices: int | None = None,
    columns: int | None = None,
) -> DependencyOracle:
    """Build all adjacent edge dependencies and optional packet metadata."""

    factor_logs = tuple(factor_logs)
    groups = make_groups(total_log, factor_logs)
    batch = _validate_batch(batch)
    packets = (
        make_packet_layout(
            total_log, factor_logs, slices=1 if slices is None else slices, columns=columns
        )
        if slices is not None or columns is not None
        else None
    )
    edges = tuple(
        enumerate_edge_dependency(total_log, groups[index], groups[index + 1], batch=batch)
        for index in range(len(groups) - 1)
    )
    return DependencyOracle(
        total_log=total_log,
        factor_logs=tuple(factor_logs),
        batch=batch,
        groups=groups,
        edges=edges,
        packets=packets,
    )


build_oracle = build_dependency_oracle


def analytic_fan_in(
    producer: FactorGroup, consumer: FactorGroup, consumer_j: int, *, batch: int
) -> frozenset[TransformId]:
    """Closed-form fan-in used only to check the address-derived oracle."""

    if consumer.index != producer.index + 1:
        raise ValueError("only adjacent factor groups form an edge")
    consumer_s, consumer_p = consumer.coordinates(consumer_j)
    # The consumer's p coordinate contains the producer p plus an Fp-sized
    # digit.  This is a separate derivation from address enumeration.
    producer_p = consumer_p % producer.P
    return frozenset(
        (
            batch,
            (consumer_s + row * consumer.S) * producer.P + producer_p,
        )
        for row in range(consumer.F)
    )


def verify_address_fanin(oracle: DependencyOracle) -> dict[str, object]:
    """Check address coverage, uniqueness, and expected local fan-in size."""

    failures: list[str] = []
    N = oracle.N
    for edge in oracle.edges:
        for consumer_id, fanin in edge.fan_in.items():
            if len(fanin) != edge.consumer.F:
                failures.append(
                    f"edge {edge.producer.index}->{edge.consumer.index} "
                    f"consumer {consumer_id} fan-in={len(fanin)} expected {edge.consumer.F}"
                )
            if any(producer_id[0] != consumer_id[0] for producer_id in fanin):
                failures.append(f"edge crosses batch at consumer {consumer_id}")
        if len(edge.producer_to_consumers) != oracle.batch * edge.producer.transform_count:
            failures.append(
                f"edge {edge.producer.index}->{edge.consumer.index} does not use all "
                "producer transforms"
            )
        for producer_id, consumers in edge.producer_to_consumers.items():
            if len(consumers) != edge.producer.F:
                failures.append(
                    f"edge {edge.producer.index}->{edge.consumer.index} producer "
                    f"{producer_id} fan-out={len(consumers)} expected {edge.producer.F}"
                )
    # A valid edge's producer outputs and consumer inputs each cover exactly
    # the compact batch address domain.  This catches indexing errors even
    # when an individual fan-in cardinality happens to look plausible.
    for group in oracle.groups:
        inputs = enumerate_input_addresses(oracle.total_log, group, batch=oracle.batch)
        outputs = enumerate_output_addresses(oracle.total_log, group, batch=oracle.batch)
        if len({address for values in inputs.values() for address in values}) != oracle.batch * N:
            failures.append(f"group {group.index} input addresses are not a permutation")
        if len({address for values in outputs.values() for address in values}) != oracle.batch * N:
            failures.append(f"group {group.index} output addresses are not a permutation")
    return {"ok": not failures, "failures": failures, "method": "enumerated-address-match"}


def verify_closed_form_fanin(oracle: DependencyOracle) -> dict[str, object]:
    """Compare enumerated fan-in with the independently derived closed form."""

    failures: list[str] = []
    for edge in oracle.edges:
        for consumer_id, actual in edge.fan_in.items():
            expected = analytic_fan_in(
                edge.producer,
                edge.consumer,
                consumer_id[1],
                batch=consumer_id[0],
            )
            if actual != expected:
                failures.append(
                    f"edge {edge.producer.index}->{edge.consumer.index} "
                    f"consumer {consumer_id}: actual={sorted(actual)} expected={sorted(expected)}"
                )
    return {"ok": not failures, "failures": failures, "method": "closed-form-cross-check"}


def _packet_ids_for_batch(
    layout: PacketLayout, group: FactorGroup, packet: int, batch: int
) -> frozenset[TransformId]:
    return frozenset((b, j) for b in range(batch) for j in layout.transform_ids(group, packet))


def verify_packet_analytic(oracle: DependencyOracle) -> dict[str, object]:
    """Verify the packet closure rule against enumerated fan-in sets.

    For an edge whose consumer is not the last factor, packet ``q`` consumes
    only producer packet ``q``.  For the edge entering the last factor, each
    consumer requires producer transforms from every packet.
    """

    if oracle.packets is None:
        return {"ok": True, "failures": [], "method": "packet-check-skipped"}
    layout = oracle.packets
    failures: list[str] = []
    last = len(oracle.groups) - 1
    for edge in oracle.edges:
        consumer_is_last = edge.consumer.index == last
        for packet in range(layout.packet_count):
            if consumer_is_last:
                consumer_ids = tuple(sorted(edge.fan_in))
                expected_producers = frozenset(
                    transform
                    for producer_packet in range(layout.packet_count)
                    for transform in _packet_ids_for_batch(
                        layout, edge.producer, producer_packet, oracle.batch
                    )
                )
            else:
                consumer_ids = tuple(
                    sorted(
                        _packet_ids_for_batch(
                            layout, edge.consumer, packet, oracle.batch
                        )
                    )
                )
                expected_producers = _packet_ids_for_batch(
                    layout, edge.producer, packet, oracle.batch
                )
            actual_producers = frozenset(
                producer
                for consumer_id in consumer_ids
                for producer in edge.fan_in[consumer_id]
            )
            if actual_producers != expected_producers:
                failures.append(
                    f"edge {edge.producer.index}->{edge.consumer.index} packet {packet}: "
                    f"actual={sorted(actual_producers)} expected={sorted(expected_producers)}"
                )
            # For middle edges each individual consumer must be packet-local.
            if not consumer_is_last:
                for consumer_id in consumer_ids:
                    if not edge.fan_in[consumer_id] <= expected_producers:
                        failures.append(
                            f"edge {edge.producer.index}->{edge.consumer.index} "
                            f"consumer {consumer_id} escapes packet {packet}"
                        )
    return {
        "ok": not failures,
        "failures": failures,
        "method": "compact-last-factor-packet-closure",
        "last_edge_uses_all_packets": bool(oracle.edges)
        and oracle.edges[-1].consumer.index == last,
    }


def _parse_factor_logs(value: str) -> tuple[int, ...]:
    try:
        logs = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("factor logs must be comma-separated integers") from exc
    if not logs:
        raise argparse.ArgumentTypeError("at least one factor log is required")
    return logs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-log", type=int, required=True)
    parser.add_argument("--factors", type=_parse_factor_logs, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--slices", type=int)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--include-matches", action="store_true")
    args = parser.parse_args(argv)
    oracle = build_dependency_oracle(
        args.total_log,
        args.factors,
        batch=args.batch,
        slices=args.slices,
        columns=args.columns,
    )
    print(json.dumps(oracle.as_dict(include_matches=args.include_matches), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
