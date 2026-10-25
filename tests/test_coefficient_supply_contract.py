import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class CoefficientSupplyContractTest(unittest.TestCase):
    """Keep NTT-only coefficient reuse from becoming a silent common knob."""

    def read(self, relative):
        return (ROOT / relative).read_text(encoding="utf-8")

    def test_public_execution_descriptor_has_no_ntt_reuse_field(self):
        mixed = self.read("include/cuntt/mixed_dataflow.hpp")
        execution_group = mixed.split("struct ExecutionGroup", 1)[1].split(
            "struct ExecutionBoundary", 1
        )[0]
        self.assertNotIn("coefficient_reuse_stages", execution_group)
        self.assertNotIn("twiddles_shoup", execution_group)

    def test_shared_templates_consume_operator_semantics_only(self):
        iterative = self.read("src/shared_iterative.cuh")
        resident = self.read("src/shared_resident.cuh")
        for source in (iterative, resident):
            self.assertIn("op.apply(", source)
            self.assertNotIn("coefficient_reuse_stages", source)
            self.assertNotIn("apply_coefficient(", source)

    def test_ntt_supply_is_explicitly_narrow(self):
        ntt_traits = self.read("src/ntt_traits.cuh")
        ntt_mapping = self.read("include/cuntt/ntt.hpp")
        self.assertIn("apply_coefficient(", ntt_traits)
        self.assertIn("coefficient_reuse_stages", ntt_mapping)
        self.assertIn("twiddles_shoup", ntt_traits)

    def test_non_ntt_operator_traits_do_not_claim_ntt_supply(self):
        operators = self.read("src/operator_traits.cuh")
        self.assertIn("struct FftOperator", operators)
        self.assertIn("struct FwhtOperator", operators)
        self.assertIn("struct BooleanZetaOperator", operators)
        self.assertNotIn("apply_coefficient(", operators)


if __name__ == "__main__":
    unittest.main()
