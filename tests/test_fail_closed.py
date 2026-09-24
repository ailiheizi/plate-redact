#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
fail-closed 链路的自测：全部使用**合成素材**，不读取任何真实视频、权重或素材。

覆盖的行为（对应 DESIGN.md 第 2/3/5 节）:

* 正常路径：输入冻结 → 排他 lease → 两遍渲染一致 → 输出侧审计通过 → 原子发布；
* 阻断路径：渲染漏打（计划框在成品里不是纯色）、成品里仍有未被计划覆盖的牌区、
  车辆轨迹没有可解释牌区（UNKNOWN）；
* 清理失败：冻结副本清理 / lease 释放 / 参考遍临时产物清理 任一失败都写
  failure marker 并隔离最终名；
* 复核门禁：人工安全裁决只对视觉候选生效，未知 id 或摘要不符一律阻断；
* 运行前旧成品必须被移除（文件存在 == 本次通过）；诊断运行永远不产最终成品。

运行::

    python tests/test_fail_closed.py
    python -m unittest discover -s tests -p "test_*.py"

依赖: numpy + opencv-python。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "examples"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fail_closed as fc            # noqa: E402
import fail_closed_synthetic as demo  # noqa: E402


class FailClosedChainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="fail-closed-selftest-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------------- 辅助 ----------------
    def _source(self, scenario_mode: str = "pass") -> Path:
        scenario = demo.scenario_for_mode(scenario_mode)
        source = self.tmp / f"src_{scenario_mode}.mp4"
        demo.write_source(source, scenario)
        return source

    def _run(self, source: Path, name: str, *, plan=None, **kwargs) -> fc.RunReport:
        return fc.run_fail_closed(source, self.tmp / f"{name}.mp4",
                                  plan=plan or demo.make_plan(), **kwargs)

    def _scenario_run(self, mode: str, **kwargs) -> tuple[fc.RunReport, Path]:
        outdir = self.tmp / f"scenario_{mode}"
        report = demo.run_scenario(outdir, mode, **kwargs)
        return report, outdir

    def _ids(self, report: fc.RunReport) -> list[str]:
        return [str(issue.get("id")) for issue in report.issues]

    def _types(self, report: fc.RunReport) -> list[str]:
        return [str(issue.get("type")) for issue in report.issues]

    def _issue(self, report: fc.RunReport, issue_id: str) -> dict:
        for issue in report.issues:
            if str(issue.get("id")) == issue_id:
                return issue
        raise AssertionError(f"未找到问题 {issue_id}: {report.issues}")

    def _read_report(self, path: Path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    # ---------------- 1. 正常路径 ----------------
    def test_pass_path_publishes_atomically(self) -> None:
        source = self._source()
        before = fc.sha256_file(source)
        report = self._run(source, "pass_out")
        self.assertEqual(report.status, "PASS", report.summary())
        self.assertEqual(report.issues, [])
        # 产物文件存在 == 本次通过：最终名存在、draft 已被原子改名消耗掉
        self.assertTrue(report.output.exists())
        self.assertIsNone(report.draft)
        self.assertFalse(fc.canonical_draft_path(report.output).exists())
        # 报告与产物绑定同一个哈希
        payload = self._read_report(report.report_path)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["artifacts"]["output_sha256"], fc.sha256_file(report.output))
        self.assertEqual(payload["decoded_output_audit"]["flat_checked"], payload["two_pass"]["planned_frames"])
        # 源文件从头到尾没被改动
        self.assertEqual(fc.sha256_file(source), before)
        # 没有生命周期 marker、没有遗留 lease
        for kind in fc.MARKER_KINDS:
            self.assertFalse(fc.failure_marker_path(report.output, kind).exists(), kind)
        self.assertFalse(fc.output_lock_path(report.output).exists())

    def test_plan_receives_frozen_copy_not_original(self) -> None:
        source = self._source()
        seen: dict[str, object] = {}
        inner = demo.make_plan()

        def plan(frozen_source: Path) -> fc.PlanBundle:
            # 冻结副本只在运行期间存在，哈希要在这里取。
            seen["path"] = Path(frozen_source)
            seen["sha256"] = fc.sha256_file(frozen_source)
            return inner(frozen_source)

        report = self._run(source, "frozen_out", plan=plan)
        self.assertEqual(report.status, "PASS", report.summary())
        frozen = seen["path"]
        self.assertNotEqual(frozen, source)
        self.assertNotEqual(frozen.parent, source.parent)
        self.assertEqual(seen["sha256"], fc.sha256_file(source))
        self.assertEqual(report.extra["frozen_inputs"]["source"]["sha256"],
                         fc.sha256_file(source))
        # 冻结副本在运行结束后必须已被清理
        self.assertFalse(frozen.exists())

    def test_stale_final_output_is_removed_first(self) -> None:
        source = self._source()
        stale = self.tmp / "stale_out.mp4"
        stale.write_bytes(b"this is a stale artifact from a previous run")
        report = self._run(source, "stale_out")
        self.assertEqual(report.status, "PASS", report.summary())
        self.assertNotEqual(stale.read_bytes(), b"this is a stale artifact from a previous run")
        self.assertIn("output:stale_final",
                      [str(w.get("id")) for w in report.warnings])

    def test_second_run_is_locked_out(self) -> None:
        source = self._source()
        output = self.tmp / "locked.mp4"
        lease = fc.acquire_output_lock(output)
        try:
            with self.assertRaises(fc.OutputLockError):
                fc.acquire_output_lock(output)
            report = self._run(source, "locked")
            self.assertEqual(report.status, "INCOMPLETE")
            self.assertIn("engine:pipeline", self._ids(report))
            self.assertFalse(output.exists())
        finally:
            lease.release()
        # 锁释放后可以正常跑通
        report = self._run(source, "locked")
        self.assertEqual(report.status, "PASS", report.summary())

    def test_smoke_run_never_produces_final_output(self) -> None:
        source = self._source()
        report = self._run(source, "smoke_out", max_frames=8)
        self.assertEqual(report.status, "SMOKE_TEST", report.summary())
        self.assertFalse(report.output.exists())
        self.assertIsNotNone(report.draft)
        self.assertTrue(report.draft.exists())

    # ---------------- 2. 阻断路径 ----------------
    def test_missing_mask_is_rejected(self) -> None:
        report, outdir = self._scenario_run("missing_mask")
        self.assertEqual(report.status, "BLOCKED", report.summary())
        self.assertTrue(any(issue.startswith("render:flat:")
                            for issue in self._ids(report)), report.issues)
        self.assertFalse(report.output.exists())
        self.assertTrue((outdir / "synthetic_missing_mask_out.draft.mp4").exists())

    def test_residual_plate_texture_is_rejected(self) -> None:
        report, _ = self._scenario_run("residual_plate")
        self.assertEqual(report.status, "BLOCKED", report.summary())
        issue = self._issue(report, "vehicle_track:0")
        self.assertEqual(issue["decision"], "UNKNOWN")
        self.assertIn("residual_plate_texture", issue["reasons"])
        self.assertGreater(issue["reasons"]["residual_plate_texture"][0]["regions"].__len__(), 0)
        self.assertFalse(report.output.exists())

    def test_unexplained_vehicle_blocks_until_reviewed(self) -> None:
        source = self._source("unexplained_vehicle")
        # 先看一遍「不解码成品」的轨迹级预览：同样按轨迹报，不按帧报。
        bundle = demo.make_plan()(source)
        preview = fc.unresolved_vehicle_issues(bundle.plans, bundle.vehicle_tracks)
        self.assertEqual([issue["id"] for issue in preview], ["vehicle_track:1"])
        self.assertEqual(preview[0]["first_frame"], 0)
        self.assertEqual(preview[0]["last_frame"], 29)

        report = self._run(source, "unexplained_out")
        self.assertEqual(report.status, "BLOCKED", report.summary())
        issue = self._issue(report, "vehicle_track:1")
        self.assertEqual(issue["type"], "vehicle_without_explainable_plate_region")
        self.assertIn("no_planned_mask_for_vehicle", issue["reasons"])
        self.assertFalse(report.output.exists())

        # 绑定摘要的真人安全裁决可以让它通过（只对这条轨迹生效）
        digest = report.extra and self._read_report(report.report_path)["review_issue_bindings"]["vehicle_track:1"]
        reviewed = self._run(source, "unexplained_out",
                             safe_issue_ids={"vehicle_track:1": digest})
        self.assertEqual(reviewed.status, "PASS", reviewed.summary())

    def test_review_gate_rejects_unknown_id_and_bad_binding(self) -> None:
        source = self._source("unexplained_vehicle")
        unknown = self._run(source, "review_unknown",
                            safe_issue_ids={"vehicle_track:99": None})
        self.assertEqual(unknown.status, "BLOCKED")
        self.assertIn("review:unknown:vehicle_track:99", self._ids(unknown))

        bad = self._run(source, "review_bad",
                        safe_issue_ids={"vehicle_track:1": "0" * 64})
        self.assertEqual(bad.status, "BLOCKED")
        self.assertIn("review:binding:vehicle_track:1", self._ids(bad))

    def test_review_cannot_dismiss_infrastructure_issues(self) -> None:
        # 基础设施问题（render:*）即使被写进 safe_issue_ids 也不可豁免
        report, outdir = self._scenario_run("missing_mask")
        flat_ids = [issue for issue in self._ids(report) if issue.startswith("render:flat:")]
        self.assertTrue(flat_ids)
        source = outdir / "synthetic_missing_mask_source.mp4"
        again = fc.run_fail_closed(source, outdir / "again.mp4", plan=demo.make_plan(),
                                   render=demo.defective_render(
                                       skip_mask_frames=(15,)),
                                   safe_issue_ids={flat_ids[0]: None})
        self.assertEqual(again.status, "BLOCKED")
        self.assertIn(f"review:non_dismissible:{flat_ids[0]}", self._ids(again))

    def test_two_pass_chain_mismatch_blocks(self) -> None:
        source = self._source()
        real_render = fc.render_masked_draft
        calls = {"count": 0}

        def perturbed(source_path, destination, plans, **kwargs):
            result = real_render(source_path, destination, plans, **kwargs)
            calls["count"] += 1
            if calls["count"] == 2:
                _patch_one_pixel(Path(destination))
            return result

        with mock.patch.object(fc, "render_masked_draft", side_effect=perturbed):
            report = self._run(source, "twopass_out")
        self.assertEqual(report.status, "BLOCKED", report.summary())
        self.assertIn("render:decoded_chain", self._ids(report))
        self.assertFalse(report.output.exists())

    # ---------------- 3. 清理失败也算阻断 ----------------
    def test_frozen_input_cleanup_failure_blocks(self) -> None:
        source = self._source()
        real_cleanup = fc.FrozenInputSet.cleanup

        def failing_cleanup(self):
            return real_cleanup(self) + ["注入的冻结副本清理失败"]

        with mock.patch.object(fc.FrozenInputSet, "cleanup", failing_cleanup):
            report = self._run(source, "cleanup_frozen")
        self.assertEqual(report.status, "INCOMPLETE", report.summary())
        self.assertIn("cleanup:frozen_input_cleanup", self._ids(report))
        marker = fc.failure_marker_path(report.output, "frozen_input_cleanup")
        self.assertTrue(marker.exists())
        self.assertEqual(self._read_report(marker)["status"], "INCOMPLETE")
        # 清理失败不允许把中间态留下当成功：最终名被隔离，报告降级
        self.assertFalse(report.output.exists())
        self.assertIn("quarantined", report.artifacts)
        self.assertEqual(self._read_report(report.report_path)["status"], "INCOMPLETE")

    def test_lease_release_failure_blocks(self) -> None:
        source = self._source()
        with mock.patch.object(fc.OutputLease, "release",
                               side_effect=RuntimeError("注入的 lease 释放失败")):
            report = self._run(source, "cleanup_lease")
        self.assertEqual(report.status, "INCOMPLETE", report.summary())
        self.assertIn("cleanup:output_lock_release", self._ids(report))
        self.assertTrue(fc.failure_marker_path(report.output,
                                               "output_lock_release").exists())
        self.assertFalse(report.output.exists())

    def test_draft_cleanup_failure_blocks(self) -> None:
        source = self._source()
        with mock.patch.object(fc, "_remove_temp_leaf",
                               side_effect=RuntimeError("注入的参考遍清理失败")):
            report = self._run(source, "cleanup_draft")
        self.assertEqual(report.status, "INCOMPLETE", report.summary())
        self.assertIn("cleanup:draft_cleanup", self._ids(report))
        self.assertTrue(fc.failure_marker_path(report.output, "draft_cleanup").exists())
        self.assertFalse(report.output.exists())

    # ---------------- 4. 直接调用组件的边界 ----------------
    def test_lease_release_detects_replaced_lock(self) -> None:
        output = self.tmp / "swap.mp4"
        lease = fc.acquire_output_lock(output)
        lease.path.unlink()
        lease.path.write_text("attacker", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            lease.release()

    def test_no_vehicle_evidence_blocks(self) -> None:
        source = self._source()

        def plan_without_vehicles(frozen_source: Path) -> fc.PlanBundle:
            bundle = demo.make_plan()(frozen_source)
            bundle.vehicle_tracks = []
            return bundle

        report = self._run(source, "novehicle_out", plan=plan_without_vehicles)
        self.assertEqual(report.status, "BLOCKED", report.summary())
        self.assertIn("coverage:no_vehicle_tracks", self._ids(report))

    def test_camera_source_is_rejected_by_cli(self) -> None:
        with self.assertRaises(SystemExit):
            fc.main(['--source', '0', '--output', str(self.tmp / 'cam.mp4')])


def _patch_one_pixel(path: Path) -> None:
    """把成品重编码一遍并在角落改一个像素：两遍解码链必须因此不一致。"""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    frames[0][0, 0] = (255, 255, 255)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    unittest.main(verbosity=2)
