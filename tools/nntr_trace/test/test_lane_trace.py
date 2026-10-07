# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location(
    "lane_trace", Path(__file__).resolve().parents[1] / "lane_trace_to_trace.py")
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)


def capture(recs, dropped=0):
    words = [0x4e4c5431, 1, 19200000, len(recs), dropped, 1920, 8, 8]
    words += [v for rec in recs for v in rec]
    return struct.pack("<%dI" % len(words), *words)


class LaneTraceTest(unittest.TestCase):
    def test_parallel_union_and_caller_help(self):
        raw = capture([
            [10, 0, 480, 96, 5, 13, 2, 0],  # publish 25..30us
            [1, 0, 192, 768, 0, 0, 0, 0],  # HMX 10..50us
            [13, 1, 576, 960, 5, 0, 2, 0],  # HVX 30..80us
            [13, 2, 768, 384, 5, 1, 2, 0],  # HVX 40..60us
            [6, 0, 960, 768, 8, 12, 1, 0],  # BG wait 50..90us
            [12, 0, 1152, 384, 8, 0, 1, 1],  # caller help 60..80us
            [7, 0, 192, 1536, 4096, 1, 576, 0],  # DMA bound 10..90
        ])
        trace, m = lane.convert(raw)
        self.assertAlmostEqual(m["hmx_hvx_overlap_us"], 20)
        self.assertAlmostEqual(m["hvx_any_callback_us"], 50)
        self.assertAlmostEqual(m["exposed_wait_us"]["activation pack"], 20)
        self.assertAlmostEqual(m["dma"]["max_completion_uncertainty_us"], 60)
        self.assertTrue(m["complete"])
        self.assertEqual(m["jobs_by_operation"]["gate_up dequant + SwiGLU"]["jobs"], 1)
        self.assertAlmostEqual(m["longest_publish_to_first_start_jobs"][0]["publish_to_first_start_upper_us"], 5)
        self.assertIn("dsp.dma.bound", {e.get("cat") for e in trace["traceEvents"]})
        hmx = next(e for e in trace["traceEvents"] if e.get("cat") == "dsp.hmx")
        self.assertEqual(hmx["args"]["display_label"], "FC matmul gate/up + acc_read")
        hvx = next(e for e in trace["traceEvents"] if e.get("cat") == "dsp.hvx")
        self.assertEqual(hvx["args"]["operation"], "dequant + SwiGLU")

    def test_drop_disables_complete(self):
        self.assertFalse(lane.convert(capture([], 1))[1]["complete"])

    def test_conv_pipeline_and_stage_wait(self):
        trace, m = lane.convert(capture([
            [18, 0, 0, 192, 0, 1, 16, 32],
            [21, 1, 96, 96, 1, 0, 2, 0],
            [19, 0, 192, 384, 1, 0, 0, 32],
            [22, 2, 384, 192, 2, 0, 2, 0],
            [23, 1, 576, 384, 3, 0, 8, 1],
            [6, 0, 576, 576, 3, 23, 8, 0],
            [23, 0, 768, 192, 3, 1, 8, 1],
            [20, 0, 1152, 384, 0, 1, 32, 32],
        ]))
        self.assertEqual(m["operation"], "conv")
        self.assertEqual(trace["traceEvents"][0]["name"], "Conv block DSP entry")
        self.assertAlmostEqual(m["hmx_api_window_us"], 50)
        self.assertAlmostEqual(m["hmx_hvx_overlap_us"], 15)
        self.assertAlmostEqual(m["exposed_wait_us"]["conv gate + requant + pack"], 20)
        stage = m["jobs_by_operation"]["conv gate + requant + pack"]
        self.assertEqual(stage["jobs"], 1)
        self.assertAlmostEqual(stage["summed_callback_us"], 30)
        events = [e for e in trace["traceEvents"] if e.get("cat") == "dsp.hmx"]
        self.assertEqual(events[0]["args"]["column_tile"], 16)
        self.assertEqual(events[-1]["args"]["block"], 0)
        self.assertNotIn("expert", events[0]["args"])

    def test_bad_input_rejected(self):
        for raw in [b"", capture([])[:-4],
                    capture([[3, 9, 0, 1, 0, 0, 0, 0]]),
                    capture([[1, 0, 1919, 2, 0, 0, 0, 0]]),
                    capture([[7, 0, 0, 10, 1, 1, 11, 0]]),
                    capture([[13, 1, 0, 10, 1, 0, 2, 0],
                             [14, 1, 5, 10, 2, 0, 2, 0]]),
                    capture([[21, 1, 0, 10, 1, 0, 2, 0],
                             [23, 1, 5, 10, 2, 0, 2, 1]])]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                lane.convert(raw)


if __name__ == "__main__":
    unittest.main()
