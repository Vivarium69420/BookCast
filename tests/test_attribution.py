"""Focused tests for the attribution changes.  Run with:

    ./.venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from btlib import cast as castlib
from btlib.cast_llm import _apply_cast_updates, _apply_results
from btlib.cast_scene import analyze_book_scene
from btlib.llm import LLMClient
from btlib.models import Book, Chapter
from webui import cast as webcast
from evaltools.corpora import Doc
from evaltools.runner import _alias_to_main, _make_resolver


class ResolverTests(unittest.TestCase):
    def test_gold_alias_is_canonicalised(self):
        doc = Doc(
            doc_id="x", title="X", chapters=[],
            quotes=[], genders={"miss honeychurch": "F"},
            aliases={"miss honeychurch": ["miss honeychurch", "lucy"]},
        )
        resolve = _make_resolver(doc)
        self.assertTrue(resolve("Lucy", "Miss Honeychurch"))
        self.assertTrue(resolve("Miss Honeychurch", "Lucy"))

    def test_short_form_token_subset(self):
        doc = Doc(doc_id="x", title="X", chapters=[], quotes=[])
        resolve = _make_resolver(doc)
        self.assertTrue(resolve("Jake", "Jake Barnes"))
        self.assertFalse(resolve("Jake", "Barnes Smith"))

    def test_alias_map_values_are_normalised(self):
        doc = Doc(doc_id="x", title="X", chapters=[], quotes=[],
                  aliases={"Jake Barnes": ["Jake Barnes", "Mr. Barnes"]})
        a2m = _alias_to_main(doc)
        self.assertEqual(a2m["mr barnes"], "jake barnes")
        self.assertEqual(a2m["jake barnes"], "jake barnes")


def _cast():
    return {
        "characters": [
            {"id": "c1", "name": "Mary", "aliases": ["Mary"], "gender": None,
             "narrator": False, "line_count": 1},
            {"id": "c2", "name": "Mistress Mary", "aliases": ["Mistress Mary"],
             "gender": "female", "narrator": False, "line_count": 1},
            {"id": "c3", "name": "London", "aliases": ["London"], "gender": None,
             "narrator": False, "line_count": 1},
            {"id": "minor_f", "name": "Unnamed woman (minor)", "aliases": [],
             "gender": "female", "line_count": 0},
            {"id": "minor_m", "name": "Unnamed man (minor)", "aliases": [],
             "gender": "male", "line_count": 0},
            {"id": "unknown", "name": "Unknown / Minor Characters", "aliases": [],
             "gender": None, "line_count": 0},
        ],
        "attribution": {"1": [
            {"seg_idx": 1, "character_id": "c1"},
            {"seg_idx": 3, "character_id": "c2"},
            {"seg_idx": 5, "character_id": "c3"},
        ]},
    }


class CastUpdateTests(unittest.TestCase):
    def test_merge_drop_gender_and_narrator(self):
        data = _cast()
        stats = _apply_cast_updates(data, {"characters": [
            {"id": "c2", "name": "Mistress Mary", "merge_into": "c1"},
            {"id": "c1", "name": "Mary", "aliases": ["Mary", "Mistress Mary"],
             "gender": "female", "is_narrator": True, "merge_into": None},
            {"id": "c3", "is_person": False},
        ]})
        self.assertEqual(stats["merged"], 1)
        self.assertEqual(stats["dropped"], 1)
        ids = [c["id"] for c in data["characters"]]
        self.assertNotIn("c2", ids)
        self.assertNotIn("c3", ids)
        mary = next(c for c in data["characters"] if c["id"] == "c1")
        self.assertEqual(mary["name"], "Mary")  # merged source must not rename the target
        self.assertEqual(mary["gender"], "female")
        self.assertTrue(mary["narrator"])
        self.assertEqual(data["attribution"]["1"][1]["character_id"], "c1")
        self.assertEqual(data["attribution"]["1"][2]["character_id"], "unknown")

    def test_narrator_merge_keeps_gender_override_safe(self):
        data = _cast()
        data["characters"][1]["narrator"] = True
        data["characters"][1]["person"] = "first"
        _apply_cast_updates(
            data,
            {"characters": [{"id": "c2", "merge_into": "c1"},
                            {"id": "c1", "is_narrator": True}]},
            narrator_gender="female",
        )
        mary = next(c for c in data["characters"] if c["id"] == "c1")
        self.assertTrue(mary["narrator"])
        self.assertEqual(mary["gender"], "female")

    def test_low_confidence_cast_edits_are_ignored(self):
        data = _cast()
        stats = _apply_cast_updates(data, {"characters": [
            {"id": "c2", "merge_into": "c1", "confidence": 0.2},
            {"id": "c3", "is_person": False, "confidence": 0.2},
        ]})
        self.assertEqual(stats["rejected_low_confidence"], 2)
        self.assertEqual([c["id"] for c in data["characters"]][:3], ["c1", "c2", "c3"])
        self.assertEqual(data["attribution"]["1"][2]["character_id"], "c3")

    def test_low_confidence_override_is_ignored(self):
        data = _cast()
        stats = _apply_results(
            data, "1",
            [{"seg_idx": 1, "speaker_id": "c2", "confidence": 0.2}],
            {1}, threshold=0.6, allow_unknown=True,
        )
        self.assertEqual(stats["rejected_low_confidence"], 1)
        self.assertEqual(data["attribution"]["1"][0]["character_id"], "c1")

    def test_closed_set_override(self):
        data = _cast()
        stats = _apply_results(
            data, "1",
            [{"seg_idx": 1, "speaker_id": "c2", "confidence": 0.9,
              "evidence": "she said"}],
            {1}, threshold=0.6, allow_unknown=True,
        )
        self.assertEqual(stats["changed"], 1)
        row = data["attribution"]["1"][0]
        self.assertEqual(row["character_id"], "c2")
        self.assertEqual(row["method"], "llm-adjudicated")


class _Handler(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))))
        self.__class__.calls += 1
        payload = {
            "response": json.dumps({"ok": True, "model": body.get("model")}),
            "prompt_eval_count": 7,
            "eval_count": 3,
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class ClientCacheTests(unittest.TestCase):
    def test_second_query_is_cached(self):
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                client = LLMClient(
                    model="fake",
                    endpoint=f"http://127.0.0.1:{server.server_port}/api/generate",
                    cache_dir=Path(tmp),
                    retries=0,
                    verbose=False,
                )
                first = client.query_json("hello")
                second = client.query_json("hello")
                self.assertEqual(first, second)
                self.assertEqual(_Handler.calls, 1)
                self.assertEqual(client.usage.cache_hits, 1)
        finally:
            server.shutdown()
            server.server_close()



class FirstAppearanceTests(unittest.TestCase):
    def test_embedded_self_introduction_resolves_speaker(self):
        cid = castlib.self_introduced_name(
            "“Hi, Lulu, I’m Willem.”",
            {"Willem": "Willem", "Lulu": "Lulu"},
            {"Willem": "c2", "Lulu": "c4"},
        )
        self.assertEqual(cid, "c2")

    def test_self_introduction_ignores_non_names(self):
        self.assertIsNone(castlib.self_introduced_name(
            "“I’m fine.”", {"Willem": "Willem"}, {"Willem": "c2"}))

    def test_strict_action_beat_rejects_distant_mention(self):
        segments = [
            ("narration", "She thought about it. Was all this just buttering "
                          "me up so he could hit on Melanie?"),
            ("dialogue", "“No.”"),
            ("narration", " He peers at her. "),
        ]
        broad = castlib._tag_dialogue_segments(segments, {"Melanie"})
        strict = castlib._tag_dialogue_segments(
            segments, {"Melanie"}, strict_action_beats=True)
        self.assertEqual(broad[1][0], "Melanie")
        self.assertIsNone(strict[1][0])

    def test_strict_action_beat_rejects_quoted_word(self):
        segments = [
            ("narration", "\n\nLouise\n. I like that.\n\n"),
            ("dialogue", "“No, not Louise. Lulu.”"),
            ("narration", " He reaches out his hand. "),
        ]
        strict = castlib._tag_dialogue_segments(
            segments, {"Louise", "Lulu"}, strict_action_beats=True)
        self.assertIsNone(strict[1][0])

    def test_tag_in_previous_paragraph_does_not_bleed_forward(self):
        segments = [
            ("narration", "Willem says there is another station nearby.\n\n"),
            ("dialogue", "“I can just drag it behind me.”"),
            ("narration", " I’m joking.\n\n"),
        ]
        strict = castlib._tag_dialogue_segments(
            segments, {"Willem"}, strict_action_beats=True,
            paragraph_scoped=True)
        self.assertIsNone(strict[1][0])

    def test_later_paragraph_after_quote_does_not_tag_it(self):
        segments = [
            ("dialogue", "“Oh.”"),
            ("narration", " Surprise. And relief.\n\nAnd then Willem says, "
                          "ever so casually, "),
            ("dialogue", "“We just fell in love once.”"),
        ]
        strict = castlib._tag_dialogue_segments(
            segments, {"Willem"}, strict_action_beats=True,
            paragraph_scoped=True)
        self.assertIsNone(strict[0][0])
        self.assertEqual(strict[2][0], "Willem")

    def test_action_beat_rejects_possessive_and_prepositional_mentions(self):
        for beat in ("I sat with Melanie’s family.\n\n",
                     "I looked at Melanie.\n\n"):
            segments = [
                ("narration", beat),
                ("dialogue", "“No.”"),
            ]
            strict = castlib._tag_dialogue_segments(
                segments, {"Melanie"}, strict_action_beats=True)
            self.assertIsNone(strict[1][0], beat)

    def test_paragraph_scoped_action_beat_skips_trailing_blank_line(self):
        segments = [
            ("narration", "I drag Melanie out of bed.\n\n"),
            ("dialogue", "“No.”"),
        ]
        strict = castlib._tag_dialogue_segments(
            segments, {"Melanie"}, strict_action_beats=True,
            paragraph_scoped=True)
        self.assertIsNone(strict[1][0])

    def test_action_beat_accepts_main_clause_subject(self):
        segments = [
            ("dialogue", "“We can walk.”"),
            ("narration", "\n\nWe head back. Before we get to the doors, "
                          "Willem turns to me and asks, "),
            ("dialogue", "“Are you ready for Paris?”"),
        ]
        strict = castlib._tag_dialogue_segments(
            segments, {"Willem"}, strict_action_beats=True)
        self.assertIsNone(strict[0][0])
        self.assertEqual(strict[2][0], "Willem")

    def test_narrator_contraction_tags_the_speaker(self):
        segments = [
            ("dialogue", "“What’s this?”"),
            ("narration", " I’d asked. We were sitting at the kitchen table. "),
        ]
        strict = castlib._tag_dialogue_segments(
            segments, set(), strict_action_beats=True)
        self.assertEqual(strict[0][0], castlib.NARRATOR_SENTINEL)
        self.assertEqual(strict[0][2], "narrator")

    def test_i_tell_name_does_not_make_name_the_speaker(self):
        segments = [
            ("dialogue", "“She’s new to this hangover thing.”"),
            ("narration", " I tell Willem. "),
        ]
        strict = castlib._tag_dialogue_segments(
            segments, {"Willem"}, strict_action_beats=True)
        self.assertEqual(strict[0][0], castlib.NARRATOR_SENTINEL)
        self.assertEqual(strict[0][2], "narrator")

    def test_unnamed_male_pronoun_gets_male_minor_voice(self):
        book = Book(title="Tiny", chapters=[Chapter(title="One", text=(
            "Melanie and Allyson stood by the door, fanning themselves. "
            "“It is hot,” Melanie said. "
            "“Very hot,” Allyson said. "
            "A stranger walked up to them. "
            "“I think she is right,” he said. "
            "“Thank you,” Melanie said. "
            "“You are welcome,” he said. "
        ))])
        data = analyze_book_scene(book, max_characters=8)
        rows = {e["seg_idx"]: e for e in data["attribution"]["1"]}
        self.assertEqual(rows[5]["character_id"], "minor_m")
        self.assertEqual(rows[5]["method"], "pronoun-male")


class DuplicateVoiceTests(unittest.TestCase):
    def test_named_characters_cannot_share_a_voice(self):
        data = {
            "characters": [
                {"id": "c1", "name": "Allyson", "voice": "voice-a"},
                {"id": "c2", "name": "Melanie", "voice": "voice-b"},
            ],
            "attribution": {},
        }
        ok, msg = webcast.apply(
            data, "c2", name="Changed", gender="male", voice="voice-a")
        self.assertFalse(ok)
        self.assertIn("already used", msg)
        self.assertEqual(data["characters"][1]["voice"], "voice-b")
        self.assertEqual(data["characters"][1]["name"], "Melanie")
        self.assertNotIn("gender", data["characters"][1])

    def test_narrator_voice_cannot_be_reused(self):
        data = {
            "characters": [
                {"id": "narrator", "name": "Narrator (I)", "voice": "voice-a"},
                {"id": "c1", "name": "Allyson", "voice": "voice-b"},
            ],
            "attribution": {},
        }
        ok, msg = webcast.apply(data, "c1", voice="voice-a")
        self.assertFalse(ok)
        self.assertIn("already used", msg)
        self.assertEqual(data["characters"][1]["voice"], "voice-b")


if __name__ == "__main__":
    unittest.main()
