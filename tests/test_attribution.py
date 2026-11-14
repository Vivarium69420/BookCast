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
from btlib.extract import split_dialogue_narration
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


class SceneAttributionTests(unittest.TestCase):
    """JOD-style family/class scenes that exercise the scene engine."""

    _WARMUP = "".join(f"“Warmup {i},” I said.\n\n" for i in range(10))

    def _scene(self, text):
        book = Book(title="Tiny", chapters=[Chapter(title="One", text=text)])
        data = analyze_book_scene(book, max_characters=10)
        segments = split_dialogue_narration(text)
        rows = {e["seg_idx"]: e for e in data["attribution"]["1"]}
        chars = {c["id"]: c for c in data["characters"]}
        by_quote = {}
        for idx, (kind, part) in enumerate(segments):
            if kind == "dialogue":
                by_quote[part] = rows.get(idx, {})
        return by_quote, chars

    def _speaker(self, by_quote, chars, needle):
        for quote, row in by_quote.items():
            if needle in quote:
                cid = row.get("character_id")
                return chars.get(cid, {"name": cid, "id": cid})
        self.fail(f"no quote containing {needle!r}")

    def _assert_name(self, by_quote, chars, needle, name):
        speaker = self._speaker(by_quote, chars, needle)
        self.assertEqual(speaker.get("name"), name,
                         f"{needle!r} -> {speaker}")

    def test_four_person_family_tags_each_speaker(self):
        text = self._WARMUP + (
            "Grandma folded her napkin. “She’s not a child,” Grandma said.\n\n"
            "“She’s still in school,” Mom said.\n\n"
            "“I can decide for myself,” I said.\n\n"
            "“Then decide,” Dad said.\n\n"
            "“Don’t talk to your father that way,” Mom said.\n\n"
            "“Let her speak,” Grandma said.\n\n"
        )
        by_quote, chars = self._scene(text)
        self._assert_name(by_quote, chars, "not a child", "Grandma")
        self._assert_name(by_quote, chars, "still in school", "Mom")
        self.assertTrue(self._speaker(by_quote, chars,
                                      "decide for myself").get("narrator"))
        self._assert_name(by_quote, chars, "Then decide", "Dad")
        self._assert_name(by_quote, chars, "father that way", "Mom")
        self._assert_name(by_quote, chars, "Let her speak", "Grandma")

    def test_paragraph_break_she_action_returns_to_mother(self):
        text = self._WARMUP + (
            "Mom set down her checkbook. “Paris is expensive,” Mom said.\n\n"
            "“I know,” I said.\n\n"
            "“You should take the car at night,” Mom said.\n\n"
            "“That’s okay. I don’t want to leave you stranded,” I said.\n\n"
            "“Well, call me for a ride,” Mom said.\n\n"
            "“It’s late. And I usually get a lift home from someone.”\n\n"
            "She takes the check back and with a violence that surprises me, "
            "rips it up.\n\n"
            "“Well, I can’t do anything for you anymore, can I?”\n\n"
            "“What does that mean?”\n\n"
            "“You don’t want my money or my car or my ride,” Mom said.\n\n"
        )
        by_quote, chars = self._scene(text)
        self._assert_name(by_quote, chars, "anything for you anymore", "Mom")
        self.assertTrue(self._speaker(by_quote, chars,
                                      "What does that mean?").get("narrator"))
        self._assert_name(by_quote, chars, "my money or my car", "Mom")

    def test_cognition_beat_lets_narrator_reply(self):
        text = self._WARMUP + (
            "Mom set down her fork. “You never listen,” Mom said.\n\n"
            "I don’t entirely believe her.\n\n"
            "“If that’s true, maybe you should stop.”\n\n"
            "“Am I wrong?” Mom asked.\n\n"
        )
        by_quote, chars = self._scene(text)
        self.assertTrue(self._speaker(by_quote, chars,
                                      "maybe you should stop").get("narrator"))
        self._assert_name(by_quote, chars, "Am I wrong?", "Mom")

    def test_rapid_grandma_exchange_keeps_narrator_turns(self):
        text = self._WARMUP + (
            "Mom smoothed her dress. “Paris is expensive,” Mom said.\n\n"
            "“I know,” I said.\n\n"
            "“Your mom is having such a hard time,” Grandma said.\n\n"
            "“I don’t see why. She has a charmed life.”\n\n"
            "“Funny, that’s what she says about you whenever she thinks "
            "you’re being ungrateful.”\n\n"
            "I spit out my lemonade.\n\n"
            "“She’s forty-seven.”\n\n"
            "“She could adopt.”\n\n"
            "Grandma waves her hand.\n\n"
            "“One of those Chinese orphans. Lucy Rosenbaum got a cute one as "
            "a granddaughter.”\n\n"
            "“They’re not dogs, Grandma!”\n\n"
            "“I know that. Still, she could get an older one.”\n\n"
            "“Did you tell Mom that?”\n\n"
            "“Of course I did.”\n\n"
        )
        by_quote, chars = self._scene(text)
        self._assert_name(by_quote, chars, "having such a hard time", "Grandma")
        self.assertTrue(self._speaker(by_quote, chars,
                                      "She has a charmed life").get("narrator"))
        self.assertTrue(self._speaker(by_quote, chars,
                                      "She’s forty-seven").get("narrator"))
        self._assert_name(by_quote, chars, "She could adopt", "Grandma")
        self._assert_name(by_quote, chars, "Chinese orphans", "Grandma")
        self.assertTrue(self._speaker(by_quote, chars,
                                      "They’re not dogs").get("narrator"))
        self._assert_name(by_quote, chars, "I know that", "Grandma")
        self.assertTrue(self._speaker(by_quote, chars,
                                      "Did you tell Mom").get("narrator"))
        self._assert_name(by_quote, chars, "Of course I did", "Grandma")

    def test_french_class_lines_do_not_leak_to_parent(self):
        text = self._WARMUP + (
            "I am learning French this summer. Mom smoothed her dress. "
            "“Paris is expensive,” Mom said.\n\n"
            "“I know,” I said.\n\n"
            "“Je m’appelle Thérèse Lambert,” the teacher said.\n\n"
            "“Comment vous appelez-vous?” the teacher asked.\n\n"
            "“Je m’appelle Allyson,” I said.\n\n"
            "“Bravo! Enchantée, Allyson,” the teacher said.\n\n"
            "“Pourquoi voulez-vous apprendre le français?” the teacher asked.\n\n"
            "“Je veux apprendre le français parce que,” a girl whispered. "
            "She raised her hand.\n\n"
            "“Je veux apprendre le français parce que le divorce!”\n\n"
            "“Excellent,” the teacher said.\n\n"
        )
        by_quote, chars = self._scene(text)
        self.assertTrue(self._speaker(by_quote, chars,
                                      "Je m’appelle Allyson").get("narrator"))
        for needle in ("Thérèse Lambert", "Bravo! Enchantée",
                       "le divorce!"):
            speaker = self._speaker(by_quote, chars, needle)
            self.assertIn(speaker.get("id"), ("minor_f", "minor_m"),
                          f"{needle!r} -> {speaker}")


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
        self.assertIn("reserved for the narrator", msg)
        self.assertEqual(data["characters"][1]["voice"], "voice-b")

    def test_minor_bucket_cannot_take_narrator_voice(self):
        data = {
            "characters": [
                {"id": "narrator", "name": "Narrator (I)", "voice": "voice-a"},
                {"id": "minor_f", "name": "Unnamed woman (minor)", "voice": "voice-b"},
            ],
            "attribution": {},
        }
        ok, msg = webcast.apply(data, "minor_f", voice="voice-a")
        self.assertFalse(ok)
        self.assertIn("reserved for the narrator", msg)
        self.assertEqual(data["characters"][1]["voice"], "voice-b")

    def test_narrator_cannot_take_a_character_voice(self):
        data = {
            "characters": [
                {"id": "narrator", "name": "Narrator (I)", "voice": "voice-a"},
                {"id": "c1", "name": "Allyson", "voice": "voice-b"},
                {"id": "minor_m", "name": "Unnamed man (minor)", "voice": "voice-c"},
            ],
            "attribution": {},
        }
        ok, msg = webcast.apply(data, "narrator", voice="voice-c")
        self.assertFalse(ok)
        self.assertIn("already used", msg)
        self.assertEqual(data["characters"][0]["voice"], "voice-a")


class NarratorCastTests(unittest.TestCase):
    def test_narrator_entry_is_added_and_voice_reserved(self):
        characters = [
            {"id": "c1", "name": "Allyson", "gender": "female",
             "narrator": True, "person": "first", "line_count": 803},
            {"id": "c2", "name": "Willem", "gender": "male", "line_count": 580},
            {"id": "c3", "name": "Melanie", "gender": "female", "line_count": 220},
            {"id": "minor_f", "name": "Unnamed woman (minor)",
             "gender": "female", "line_count": 0},
            {"id": "minor_m", "name": "Unnamed man (minor)",
             "gender": "male", "line_count": 0},
            {"id": "unknown", "name": "Unknown / Minor Characters",
             "gender": None, "line_count": 0},
        ]
        narrator = castlib.ensure_narrator_character(characters)
        self.assertIsNotNone(narrator)
        self.assertEqual(narrator["id"], "narrator")
        self.assertEqual(narrator["name"], "Narrator (I)")
        self.assertEqual(narrator["line_count"], 0)
        castlib.assign_voices(characters)
        narrator_voice = narrator["voice"]
        self.assertTrue(narrator_voice)
        self.assertNotIn(narrator_voice,
                         [c["voice"] for c in characters if c is not narrator])

    def test_character_rows_pin_narrator_first(self):
        data = {
            "characters": [
                {"id": "c1", "name": "Allyson", "line_count": 803, "voice": "a"},
                {"id": "narrator", "name": "Narrator (I)", "line_count": 0, "voice": "b"},
            ],
            "attribution": {},
        }
        rows = webcast.characters_rows(data)
        self.assertEqual(rows[0][0], "narrator")

    def test_old_top_level_narrator_becomes_an_editable_row(self):
        data = {
            "characters": [
                {"id": "c1", "name": "Allyson", "line_count": 803, "voice": "a"},
            ],
            "narrator": "voice-n",
            "attribution": {},
        }
        narrator = webcast.ensure_narrator_entry(data)
        self.assertIsNotNone(narrator)
        self.assertEqual(narrator["id"], "narrator")
        self.assertEqual(narrator["voice"], "voice-n")
        self.assertEqual(narrator["line_count"], 0)

    def test_import_rejects_narrator_voice_collision(self):
        data = {
            "characters": [
                {"id": "narrator", "name": "Narrator (I)", "voice": "voice-a"},
                {"id": "c1", "name": "Allyson", "voice": "voice-a"},
                {"id": "unknown", "name": "Unknown / Minor Characters", "voice": "voice-u"},
            ],
            "narrator": "voice-a",
            "attribution": {},
        }
        ok, problems, _summary = webcast.validate_import(data)
        self.assertFalse(ok)
        self.assertTrue(any("reserved for the narrator" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
