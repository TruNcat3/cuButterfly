"""Independent CPU checks for the factor-streamed dependency oracle."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from factor_dependency_oracle import (  # noqa: E402
    build_dependency_oracle,
    enumerate_input_addresses,
    enumerate_output_addresses,
    make_groups,
    make_packet_layout,
)


def _expected_fanin_from_addresses(total_log, producer, consumer, batch=1):
    """A test-side address join, deliberately independent of oracle internals."""

    output_owner = {}
    for b in range(batch):
        for producer_j, addresses in enumerate_output_addresses(
            total_log, producer, batch=1
        ).items():
            # The helper's batch=1 addresses start at zero; add the explicit
            # batch base here so this test also checks batch isolation.
            local_b, local_j = producer_j
            assert local_b == 0
            for u, address in enumerate(addresses):
                global_address = b * (1 << total_log) + address
                assert global_address not in output_owner
                output_owner[global_address] = (b, local_j, u)

    result = {}
    for b in range(batch):
        for consumer_j, addresses in enumerate_input_addresses(
            total_log, consumer, batch=1
        ).items():
            local_b, local_j = consumer_j
            assert local_b == 0
            producer_ids = set()
            for address in addresses:
                producer_b, producer_j, _ = output_owner[b * (1 << total_log) + address]
                assert producer_b == b
                producer_ids.add((b, producer_j))
            result[(b, local_j)] = frozenset(result for result in producer_ids)
    return result


@pytest.mark.parametrize(
    "factor_logs,total_log,slices,columns,batch",
    [
        ((2, 3), 5, 2, 2, 1),
        ((1, 2, 3), 6, 4, 2, 1),
        ((2, 1, 2, 1), 6, 2, 1, 1),
        ((2, 3), 5, 2, 2, 3),
    ],
)
def test_address_join_is_the_fanin_source_of_truth(
    factor_logs, total_log, slices, columns, batch
):
    oracle = build_dependency_oracle(
        total_log,
        factor_logs,
        batch=batch,
        slices=slices,
        columns=columns,
    )
    assert oracle.verification()["ok"]
    for edge in oracle.edges:
        assert dict(edge.fan_in) == _expected_fanin_from_addresses(
            total_log, edge.producer, edge.consumer, batch=batch
        )
        assert all(len(producers) == edge.consumer.F for producers in edge.fan_in.values())


def test_shapes_and_exact_two_factor_fanin():
    groups = make_groups(5, (2, 3))
    first, last = groups
    assert (first.P, first.F, first.S) == (1, 4, 8)
    assert (last.P, last.F, last.S) == (4, 8, 1)

    oracle = build_dependency_oracle(5, (2, 3), batch=1, slices=2, columns=2)
    edge = oracle.edges[0]
    # Consumer j=1 has s=0,p=1.  Its 8 input rows come from all producer
    # s values because this is the edge entering the last factor.
    assert edge.fan_in_for(0, 1) == frozenset(range(8))
    assert len(edge.matches_for(0, 1)) == 8
    assert {match.address for match in edge.matches_for(0, 1)} == {
        row * 4 + 1 for row in range(8)
    }


def test_middle_edge_stays_in_the_same_packet_and_last_edge_uses_all_packets():
    oracle = build_dependency_oracle(6, (1, 2, 3), batch=1, slices=4, columns=2)
    first_edge, last_edge = oracle.edges
    layout = oracle.packets
    assert layout is not None
    assert (layout.K, layout.tail_width) == (8, 2)

    # A group-1 transform in packet 3 can only fan into group-0 transforms in
    # packet 3.  The group-2 consumer is the final factor and needs all four.
    group_1_packet_3 = oracle.packet_transform_ids(1, 3)
    assert group_1_packet_3
    for consumer in group_1_packet_3:
        assert first_edge.fan_in[(0, consumer)] <= {
            (0, producer) for producer in oracle.packet_transform_ids(0, 3)
        }

    all_first_packets = set().union(
        *(oracle.packet_transform_ids(0, packet) for packet in range(4))
    )
    all_second_packets = set().union(
        *(oracle.packet_transform_ids(1, packet) for packet in range(4))
    )
    assert {(0, producer) for producer in all_second_packets} == {
        producer for consumers in last_edge.fan_in.values() for producer in consumers
    }
    assert all_first_packets == set(first_edge.producer.transforms())


def test_packet_compact_sets_cover_each_pre_last_group_once():
    oracle = build_dependency_oracle(7, (2, 1, 2, 2), batch=2, slices=2, columns=2)
    assert oracle.packets is not None
    for group in oracle.groups[:-1]:
        sets = [
            oracle.packet_transform_ids(group.index, packet)
            for packet in range(oracle.packets.packet_count)
        ]
        assert set().union(*sets) == set(group.transforms())
        assert all(left.isdisjoint(right) for i, left in enumerate(sets) for right in sets[i + 1 :])
        assert sum(map(len, sets)) == group.transform_count


@pytest.mark.parametrize(
    "factor_logs,slices,columns",
    [
        ((2, 3), 16, 2),  # slices > K
        ((2, 3), 3, 1),  # slices not a power of two
        ((2, 3), 2, 8),  # first packet tail too narrow
        ((1, 2, 3), 4, 4),  # a later P is smaller than Columns
    ],
)
def test_columns_and_slice_contract_is_enforced(factor_logs, slices, columns):
    total_log = sum(factor_logs)
    with pytest.raises(ValueError):
        make_packet_layout(total_log, factor_logs, slices=slices, columns=columns)


def test_batch_addresses_do_not_cross_and_all_batches_are_covered():
    oracle = build_dependency_oracle(5, (2, 3), batch=3, slices=2, columns=2)
    N = 1 << oracle.total_log
    for edge in oracle.edges:
        for consumer_id, producers in edge.fan_in.items():
            assert {batch for batch, _ in producers} == {consumer_id[0]}
    for group in oracle.groups:
        output_addresses = enumerate_output_addresses(oracle.total_log, group, batch=3)
        assert {address // N for values in output_addresses.values() for address in values} == {
            0,
            1,
            2,
        }


def test_verification_report_exposes_methods_and_exact_fanin_conclusion():
    oracle = build_dependency_oracle(7, (2, 2, 3), batch=2, slices=4, columns=2)
    report = oracle.verification()
    assert report["ok"]
    assert report["address_fanin"]["method"] == "enumerated-address-match"
    assert report["closed_form_fanin"]["method"] == "closed-form-cross-check"
    assert report["packets"]["method"] == "compact-last-factor-packet-closure"
    assert report["packets"]["last_edge_uses_all_packets"]
