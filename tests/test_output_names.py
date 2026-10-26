"""Tests for EPUB-named outputs, output cleanup, and deleting finished jobs."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from btlib import progress
from webui import store
from webui.job import Job, JobQueue, JobSpec


class OutputNameTests(unittest.TestCase):
    def test_output_path_uses_input_filename(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            book = Path("books/The Secret Garden.epub")
            self.assertEqual(store.out_path(ws, False, book),
                             ws / "The Secret Garden.mp3")
            self.assertEqual(store.out_path(ws, True, book),
                             ws / "The Secret Garden")

    def test_output_path_keeps_legacy_default(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            self.assertEqual(store.out_path(ws, False), ws / "out.mp3")
            self.assertEqual(store.out_path(ws, True), ws / "out")

    def test_per_chapter_output_avoids_workspace_directories(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            self.assertEqual(store.out_path(ws, True, "books/work.epub"),
                             ws / "work (audio)")

    def test_named_output_is_shared_under_its_filename(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / "My Book.mp3").write_bytes(b"audio")
            self.assertEqual(store.output_for_sharing(ws), ws / "My Book.mp3")
            self.assertEqual(store.share_name(ws), "My Book.mp3")

    def test_analyze_output_path_does_not_become_shared_audio(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / "My Book.mp3").write_bytes(b"audio")
            (ws / "cast.json").write_text("{}", encoding="utf-8")
            self.assertEqual(store.output_for_sharing(ws, ws / "cast.json"),
                             ws / "My Book.mp3")

    def test_per_chapter_output_gets_matching_zip_name(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            outdir = ws / "My Book"
            outdir.mkdir()
            (outdir / "01_Chapter.mp3").write_bytes(b"audio")
            store.save_meta(ws, output=str(outdir))
            shared = store.output_for_sharing(ws)
            self.assertEqual(shared, ws / "My Book.zip")
            self.assertEqual(store.share_name(ws, shared), "My Book.zip")

    def test_legacy_out_mp3_uses_input_name_for_sharing(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / "out.mp3").write_bytes(b"audio")
            book = Path(td) / "Input Book.epub"
            store.save_meta(ws, last_argv=[str(book), "-o", str(ws / "out.mp3")])
            self.assertEqual(store.share_name(ws), "Input Book.mp3")

    def test_invalidation_removes_named_output_and_zip(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            work = ws / "work"
            work.mkdir()
            progress_data = {
                "chapters": [{
                    "index": 1,
                    "num_chunks": 1,
                    "chunk_done": [True],
                    "chapter_combined_done": True,
                    "chapter_output_done": True,
                }],
                "final_done": True,
            }
            (work / "progress.json").write_text(json.dumps(progress_data), encoding="utf-8")
            named_dir = ws / "My Book"
            named_dir.mkdir()
            (named_dir / "01_Chapter.mp3").write_bytes(b"chapter")
            (ws / "My Book.mp3").write_bytes(b"audio")
            (ws / "My Book.zip").write_bytes(b"zip")
            self.assertEqual(store.invalidate_chapters(ws, [1]), [1])
            self.assertFalse((named_dir / "01_Chapter.mp3").exists())
            self.assertFalse((ws / "My Book.mp3").exists())
            self.assertFalse((ws / "My Book.zip").exists())


class ProgressRenameTests(unittest.TestCase):
    @staticmethod
    def _args(output: Path):
        return SimpleNamespace(
            engine="edge", voice=None, rate="+0%", gap_ms=350,
            chapter_pause_ms=900, min_chapter_words=150, per_chapter=False,
            output=str(output), no_dialogue_voice=False, dialogue_voice=None,
            dialogue_rate=None, dialogue_pitch=None, dialogue_pause_ms=200,
            multi_voice=True, max_characters=20, cast_file="cast.json",
        )

    def test_output_rename_does_not_invalidate_progress(self):
        with tempfile.TemporaryDirectory() as td:
            book = Path(td) / "Book.epub"
            book.write_bytes(b"epub")
            old = self._args(Path(td) / "out.mp3")
            new = self._args(Path(td) / "Book.mp3")
            store_obj = progress.ProgressStore(Path(td) / "work")
            store_obj.data = {
                "input": progress._input_signature(book),
                "params": progress._params_signature(old),
            }
            self.assertTrue(store_obj.matches(book, new))


class DeleteJobTests(unittest.TestCase):
    @staticmethod
    def _job(log: Path) -> Job:
        spec = JobSpec(
            book=Path("book.epub"), title="Book", cast_file=Path("cast.json"),
            workdir=Path("work"), out_path=Path("Book.mp3"), args=[],
        )
        return Job(spec=spec, log_path=log)

    def test_delete_finished_job_removes_history_and_log(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "job.log"
            log.write_text("log", encoding="utf-8")
            job = self._job(log)
            job.ended = time.time()
            job.returncode = 0
            queue = JobQueue(slots=1)
            queue._jobs[job.id] = job
            queue._order.append(job.id)
            self.assertTrue(queue.delete(job.id))
            self.assertEqual(queue.jobs, [])
            self.assertFalse(log.exists())

    def test_delete_refuses_running_job(self):
        class RunningProc:
            @staticmethod
            def poll():
                return None

        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "job.log"
            job = self._job(log)
            job.proc = RunningProc()
            queue = JobQueue(slots=1)
            queue._jobs[job.id] = job
            queue._order.append(job.id)
            self.assertFalse(queue.delete(job.id))
            self.assertEqual([j.id for j in queue.jobs], [job.id])


if __name__ == "__main__":
    unittest.main()
