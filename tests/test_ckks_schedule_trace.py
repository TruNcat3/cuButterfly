import importlib.util
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "paper/experiments/analyze_ckks_schedule_trace.py"
spec = importlib.util.spec_from_file_location("ckks_schedule_trace", SCRIPT)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)
sys.path.insert(0, str(SCRIPT.parent))


class ScheduleTraceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / "trace.sqlite"
        self.connection = sqlite3.connect(self.database)
        self.addCleanup(self.connection.close)
        self.connection.executescript("""
            CREATE TABLE StringIds(id INTEGER,value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,demangledName INTEGER,graphNodeId INTEGER);
        """)
        self.connection.executemany("INSERT INTO StringIds VALUES (?,?)", [
            (1, "cudaGraphLaunch_v10000"), (2, "cuntt::rns_fused_detail::stage_kernel<int>()"),
            (3, "cubutterfly_heongpu::(anonymous namespace)::transfer<true>()"),
            (4, "cubutterfly_heongpu::(anonymous namespace)::transfer<false>()"),
            (5, "heongpu::multiply()"), (6, "ckks_measure/rotate_rows_shift1/cu_stage_batch/0")])
        self.connection.execute("INSERT INTO NVTX_EVENTS VALUES (0,10000,7,NULL,6)")
        self.connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)", [
            (10, 20, 7, 100, 1), (30, 40, 7, 101, 1), (50, 60, 8, 102, 1)])
        # GPU events deliberately extend beyond the host range. Correlation, not
        # GPU timestamp containment, identifies their measurement owner.
        self.connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)", [
            (10000, 12000, 100, 3, 1), (11000, 14000, 100, 2, 2),
            (14000, 15000, 100, 4, 3), (15000, 16000, 100, 5, 4),
            (20000, 23000, 101, 2, 2), (21000, 22000, 101, 5, 4),
            (1, 90000, 102, 5, 20)])
        self.connection.commit()

    def test_correlations_threads_and_concurrency(self):
        result = trace.analyze(self.database)
        self.assertEqual(result["unassigned_graph_launches"], 1)
        row, = result["measurements"]
        self.assertEqual((row["variant"], row["round"], row["graph_launches"]), ("cu_stage_batch", 0, 2))
        self.assertEqual(row["kernel_count"], 6)
        self.assertEqual(row["graph_node_count"], 4)
        self.assertEqual(row["kernel_sum_us"], 11)
        self.assertEqual(row["kernel_busy_us"], 9)
        self.assertEqual(row["gpu_span_us"], 13)
        self.assertEqual(row["per_graph_mean"]["gpu_span_us"], 4.5)
        self.assertEqual(row["categories"]["ntt"]["kernel_us_per_graph"], 3)
        self.assertEqual(row["categories"]["gather"]["kernel_count"], 1)
        self.assertEqual(row["categories"]["scatter"]["kernel_count"], 1)
        self.assertEqual(row["categories"]["non_ntt"]["kernel_count"], 2)

    def test_inline_nvtx_text(self):
        self.connection.execute("UPDATE NVTX_EVENTS SET text='ckks_measure/square_rescale_keyswitch/cu_loop/2',textId=NULL")
        self.connection.commit()
        row, = trace.analyze(self.database)["measurements"]
        self.assertEqual((row["scene"], row["variant"], row["round"]), ("square_rescale_keyswitch", "cu_loop", 2))

    def test_missing_kernel_activity_rejected(self):
        self.connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=101")
        self.connection.commit()
        with self.assertRaisesRegex(ValueError, "missing node-level"):
            trace.analyze(self.database)

    def test_duplicate_identity_rejected(self):
        self.connection.execute("INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS")
        self.connection.commit()
        with self.assertRaisesRegex(ValueError, "duplicate measurement"):
            trace.analyze(self.database)

    def test_ambiguous_overlapping_ranges_rejected(self):
        self.connection.execute("INSERT INTO NVTX_EVENTS VALUES (0,10000,7,'ckks_measure/rotate_rows_shift1/cu_loop/0',NULL)")
        self.connection.commit()
        with self.assertRaisesRegex(ValueError, "overlapping measurement"):
            trace.analyze(self.database)

    def test_old_trace_without_nvtx_rejected(self):
        self.connection.execute("DROP TABLE NVTX_EVENTS")
        self.connection.commit()
        with self.assertRaisesRegex(ValueError, "NVTX_EVENTS missing"):
            trace.analyze(self.database)

    def test_categories_and_invalid_intervals(self):
        self.assertEqual(trace.category("gpuntt::GPU_NTT<int>()"), "ntt")
        self.assertEqual(trace.category("cubutterfly_heongpu::transfer<(bool)0>()"), "scatter")
        with self.assertRaisesRegex(ValueError, "unrecognized stage transfer"):
            trace.category("cubutterfly_heongpu::transfer<unknown>()")
        with self.assertRaisesRegex(ValueError, "positive duration"):
            trace.interval_stats([(2, 2)])

    def test_request_geometry_and_prepared_plan_matching(self):
        log = Path(self.directory.name) / "plans.log"
        log.write_text(
            "ntt_adapter_plan,N=16384,B=2,inverse=1,domains=1,n1_log=8,core_family=fused-ctgs,"
            "rows_per_block=1,threads_per_block=256,second_rows_per_block=4,"
            "second_threads_per_block=0,compute_unit=radix4\n"
            "ntt_adapter_domain,N=16384,B=2,inverse=1,domain=0,q=17,forward_psi=3\n")
        def stage(name, start, end):
            return dict(name=name, start=start, end=end, gridX=128, gridY=1, gridZ=1,
                        blockX=256, blockY=1, blockZ=1, registersPerThread=32,
                        dynamicSharedMemory=4096)
        native = [stage("gpuntt::InverseCorePolyOrdered()", 0, 1000),
                  stage("gpuntt::InverseCorePolyOrdered()", 1000, 2000)]
        cu = [stage("cuntt::rns_fused_detail::stage_kernel<(unsigned int)8, (unsigned int)1, (bool)1, (unsigned int)2, (bool)0, (bool)0>()", 0, 2000),
              stage("cuntt::rns_fused_detail::stage_kernel<(unsigned int)6, (unsigned int)4, (bool)1, (unsigned int)2, (bool)0, (bool)0>()", 2000, 4000)]
        result = {"measurements": [dict(variant="native_loop", launches=[dict(kernels=native)]),
                                   dict(variant="cu_loop", launches=[dict(kernels=cu)])]}
        decoded = trace.loop_requests(result, 16384, log)
        row, = decoded["requests"]
        self.assertEqual((row["B"], row["inverse"], row["native_order"]), (2, 1, "polynomial"))
        self.assertEqual(row["excess_us"], 2)
        self.assertEqual(row["distinct_observable_plan_semantics"], 1)
        self.assertEqual(len(row["matching_prepared_plans"]), 1)
        native[0]["name"] = "gpuntt::ForwardCore()"
        with self.assertRaisesRegex(ValueError, "directions disagree"):
            trace.loop_requests(result, 16384, log)

    def fused_stages(self, inverse, arity=7):
        logs = (7, 5) if inverse else (5, 7)
        result = []
        for index, local_log in enumerate(logs):
            arguments = [f"(unsigned int){local_log}", "(unsigned int)1", f"(bool){int(inverse)}"]
            if arity >= 4:
                arguments.append("(unsigned int)2")
            if arity >= 6:
                arguments += ["(bool)0", "(bool)0"]
            if arity == 7:
                arguments.append(f"(bool){int(index == 0)}")
            result.append(dict(
                name="void cuntt::rns_fused_detail::stage_kernel<" + ", ".join(arguments) + ">()",
                gridX=2 * (4096 >> local_log), gridY=1, gridZ=1,
                blockX=32, blockY=1, blockZ=1, registersPerThread=32,
                dynamicSharedMemory=(1 << local_log) * 8))
        return result

    def test_first_template_forward_inverse_and_32_thread_geometry(self):
        from analyze_ckks_rns_workloads import fused_request, plan_matches_request
        for inverse in (False, True):
            with self.subTest(inverse=inverse):
                decoded = fused_request(self.fused_stages(inverse), 4096)
                self.assertEqual((decoded["B"], decoded["inverse"], decoded["n1_log"]), (2, int(inverse), 7))
                self.assertEqual(decoded["threads_per_block"], 32)
                self.assertEqual([stage["first"] for stage in decoded["stage_mappings"]], [True, False])
                plan = dict(N="4096", B="2", inverse=str(int(inverse)), core_family="fused-ctgs",
                            n1_log="7", rows_per_block="1", threads_per_block="32", compute_unit="radix4")
                self.assertTrue(plan_matches_request(plan, decoded, 4096))

    def test_old_template_arities_preserve_defaults_and_stage_schema(self):
        from analyze_ckks_rns_workloads import fused_request
        for arity in (3, 4, 6):
            for inverse in (False, True):
                with self.subTest(arity=arity, inverse=inverse):
                    decoded = fused_request(self.fused_stages(inverse, arity), 4096)
                    self.assertEqual(decoded["B"], 2)
                    self.assertEqual(decoded["compute_unit"], "radix2" if arity == 3 else "radix4")
                    for stage in decoded["stage_mappings"]:
                        self.assertEqual(stage["arithmetic"], "shoup")
                        self.assertFalse(stage["compact"])
                        self.assertNotIn("first", stage)

    def test_first_role_and_unknown_template_rejected(self):
        from analyze_ckks_rns_workloads import fused_request
        for inverse in (False, True):
            for index in (0, 1):
                stages = self.fused_stages(inverse)
                stages[index]["name"] = stages[index]["name"].replace(
                    f", (bool){int(index == 0)}>()", f", (bool){int(index != 0)}>()")
                with self.subTest(inverse=inverse, index=index), self.assertRaisesRegex(ValueError, "First template role"):
                    fused_request(stages, 4096)
        for invalid in (
            "stage_kernel<(unsigned int)5, (unsigned int)1, (bool)0, (unsigned int)3>()",
            "stage_kernel<(unsigned int)5, (unsigned int)1, (bool)0, (unsigned int)2, (bool)0>()",
            "stage_kernel<(unsigned int)5, (unsigned int)1, (bool)0, (unsigned int)2, (bool)0, (bool)0, (bool)2>()",
            "stage_kernel<(unsigned int)5, (unsigned int)1, (bool)0, (unsigned int)2, (bool)0, (bool)0, (bool)1, (bool)0>()",
        ):
            stages = self.fused_stages(False)
            stages[0]["name"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "unsupported fused kernel template"):
                fused_request(stages, 4096)


if __name__ == "__main__":
    unittest.main()
