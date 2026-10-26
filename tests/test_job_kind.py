"""Focused tests for analyze-only jobs in the shared render queue."""
from __future__ import annotations

import unittest
from pathlib import Path

from webui import store
from webui.job import Job, JobQueue, JobSpec


def _spec(kind: str = "analyze") -> JobSpec:
    return JobSpec(
        book=Path("book.epub"),
        title="Analyze · Book",
        cast_file=Path("cast.json"),
        workdir=Path("work"),
        out_path=Path("cast.json"),
        args=["--analyze-only"],
        kind=kind,
    )


class AnalyzeJobTests(unittest.TestCase):
    def test_default_kind_is_render(self):
        spec = _spec("render")
        self.assertEqual(spec.kind, "render")
        self.assertFalse(Job(spec=spec, log_path=Path("job.log")).is_analyze)

    def test_analyze_job_has_no_audio_outputs(self):
        spec = _spec("analyze")
        self.assertTrue(Job(spec=spec, log_path=Path("job.log")).is_analyze)
        self.assertEqual(store.job_outputs(spec), [])

    def test_analyze_progress_and_cast_path_are_parsed(self):
        job = Job(spec=_spec("analyze"), log_path=Path("job.log"))
        queue = JobQueue(slots=1)
        queue._parse(job, "[1/2] Reconciling the cast")
        queue._parse(job, "Cast saved to /tmp/runs/book/cast.json")
        self.assertEqual(job.chapter_index, 1)
        self.assertEqual(job.chapter_total, 2)
        self.assertEqual(job.chapter_title, "Reconciling the cast")
        self.assertEqual(job.cast_path, "/tmp/runs/book/cast.json")


if __name__ == "__main__":
    unittest.main()
