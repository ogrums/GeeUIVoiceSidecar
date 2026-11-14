"""Tâches Hermes : intention vocale, création de run, suivi, résumé parlé."""

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import emo_server as server
import hermes_tasks


class FakeHermes:
    """API Runs minimale : POST /v1/runs, GET /v1/runs/<id>, POST /v1/runs/<id>/stop."""

    def __init__(self):
        self.runs = {}
        self.inputs = []
        self.auth = []
        self.next_status = "running"
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, payload):
                raw = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                owner.auth.append(self.headers.get("Authorization"))
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if self.path == "/v1/runs":
                    data = json.loads(body)
                    owner.inputs.append(data["input"])
                    run_id = f"run_{len(owner.runs) + 1}"
                    owner.runs[run_id] = {"run_id": run_id, "status": owner.next_status}
                    self._send(200, owner.runs[run_id])
                elif self.path.endswith("/stop"):
                    run_id = self.path.split("/")[3]
                    owner.runs[run_id]["status"] = "cancelled"
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {})

            def do_GET(self):
                owner.auth.append(self.headers.get("Authorization"))
                run_id = self.path.rsplit("/", 1)[1]
                if run_id not in owner.runs:
                    self._send(404, {"error": "unknown"})
                    return
                self._send(200, owner.runs[run_id])

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d/v1" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def finish(self, run_id, output, status="completed"):
        self.runs[run_id].update(status=status, output=output)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeLemonade:
    def __init__(self, reply="Hermes a trouvé trois fichiers dans le dossier demo.", fail=False):
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                owner.requests.append(json.loads(body))
                if fail:
                    self.send_response(500)
                    self.end_headers()
                    return
                raw = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d/api/v1" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class IntentTests(unittest.TestCase):
    def intent(self, text):
        return hermes_tasks.parse_intent(text)

    def test_create_phrases(self):
        self.assertEqual(
            {"intent": "create", "prompt": "compter les fichiers du dossier demo"},
            self.intent("Confie à Hermes de compter les fichiers du dossier demo."))
        self.assertEqual("analyser le dossier GeeUIFace",
                         self.intent("demande à Hermès d'analyser le dossier GeeUIFace")["prompt"])
        self.assertEqual("liste les projets",
                         self.intent("RUX, délègue à Hermes : liste les projets")["prompt"])
        self.assertEqual("create", self.intent("demande a hermes")["intent"])
        self.assertIsNone(self.intent("demande a hermes")["prompt"])

    def test_read_phrases(self):
        for text in ("Lis le résultat", "lis-moi le résultat.", "Résultat Hermes",
                     "quel est le résultat de Hermes ?", "donne-moi le résultat"):
            self.assertEqual("read", self.intent(text)["intent"], text)

    def test_status_phrases(self):
        self.assertEqual("status", self.intent("Où en est Hermes ?")["intent"])
        self.assertEqual("status", self.intent("Hermes a fini ?")["intent"])

    def test_ordinary_sentences_are_ignored(self):
        for text in ("Demande à maman de venir", "donne-moi le résultat du match d'hier soir",
                     "quelle heure est-il", "", "lis une histoire"):
            self.assertIsNone(self.intent(text)["intent"], text)


class PlainTests(unittest.TestCase):
    def test_lines_become_sentences(self):
        self.assertEqual("Dossiers : 3. Fichiers : 20.",
                         hermes_tasks.plain("**Dossiers** : 3  \nFichiers : 20."))


class SidecarHermesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.hermes = FakeHermes()
        self.lemonade = FakeLemonade()
        self.env = {
            "HERMES_URL": self.hermes.url,
            "HERMES_API_KEY": "test-key-not-a-secret",
            "HERMES_TASKS_FILE": os.path.join(self.tmp.name, "tasks.json"),
            "HERMES_POLL_SECONDS": "1",
            "HERMES_TASKS_TOKEN": "",
            "LEMONADE_HOST": self.lemonade.url,
            "EMO_CHAT_MODEL": "gemma-test",
            "EMO_HOST": "http://127.0.0.1:0",
            "NANOBOT_URL": "",
        }
        self.old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        hermes_tasks._store = None
        hermes_tasks._last_refresh[0] = 0
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        self.base = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.hermes.close()
        self.lemonade.close()
        for key, value in self.old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        hermes_tasks._store = None
        self.tmp.cleanup()

    def call(self, method, path, payload=None, headers=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.base + path, data=data, method=method,
                          headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def test_create_track_speak_and_read(self):
        code, task = self.call("POST", "/hermes/tasks",
                               {"text": "Confie à Hermes de compter les fichiers de demo"})
        self.assertEqual(201, code)
        self.assertEqual("running", task["status"])
        self.assertEqual("compter les fichiers de demo", task["prompt"])
        self.assertEqual("run_1", task["run_id"])
        sent = self.hermes.inputs[0]
        self.assertIn("compter les fichiers de demo", sent)
        self.assertIn("outil terminal", sent)
        self.assertIn("N'invente rien", sent)
        self.assertEqual("Bearer test-key-not-a-secret", self.hermes.auth[0])

        code, listing = self.call("GET", "/hermes/tasks")
        self.assertEqual((0, 1), (listing["unread"], listing["running"]))

        self.hermes.finish("run_1", "Il y a 3 fichiers : a.txt, b.txt, c.txt.")
        hermes_tasks._last_refresh[0] = 0
        code, listing = self.call("GET", "/hermes/tasks")
        self.assertEqual((1, 0), (listing["unread"], listing["running"]))
        self.assertEqual("completed", listing["tasks"][0]["status"])
        self.assertTrue(listing["tasks"][0]["finished"])

        code, spoken = self.call("POST", "/hermes/tasks/latest/speak", {})
        self.assertEqual(200, code)
        self.assertEqual("Hermes a trouvé trois fichiers dans le dossier demo.", spoken["text"])
        asked = self.lemonade.requests[0]
        self.assertEqual("gemma-test", asked["model"])
        self.assertIn("a.txt, b.txt, c.txt", asked["messages"][1]["content"])

        code, listing = self.call("GET", "/hermes/tasks")
        self.assertEqual(0, listing["unread"])
        # Le résumé est gardé : pas de second appel au modèle.
        self.call("POST", "/hermes/tasks/%s/speak" % task["id"], {})
        self.assertEqual(1, len(self.lemonade.requests))

        # La liste survit à un redémarrage (fichier JSON).
        hermes_tasks._store = None
        code, again = self.call("GET", "/hermes/tasks/" + task["id"])
        self.assertEqual("completed", again["status"])
        self.assertTrue(again["read"])

    def test_read_endpoint_and_unknown_task(self):
        code, task = self.call("POST", "/hermes/tasks", {"prompt": "liste /workspace/data"})
        self.hermes.finish("run_1", "vide")
        hermes_tasks._last_refresh[0] = 0
        code, got = self.call("GET", "/hermes/tasks/" + task["id"])
        self.assertEqual("completed", got["status"])
        code, got = self.call("POST", "/hermes/tasks/%s/read" % task["id"], {})
        self.assertTrue(got["read"])
        code, _ = self.call("GET", "/hermes/tasks/nope")
        self.assertEqual(404, code)

    def test_running_task_speaks_progress(self):
        code, task = self.call("POST", "/hermes/tasks", {"prompt": "analyse les projets"})
        code, spoken = self.call("POST", "/hermes/tasks/%s/speak" % task["id"], {})
        self.assertIn("travaille encore", spoken["text"])
        self.assertEqual([], self.lemonade.requests)

    def test_failed_run_and_stop(self):
        code, task = self.call("POST", "/hermes/tasks", {"prompt": "x"})
        code, stopped = self.call("POST", "/hermes/tasks/%s/stop" % task["id"], {})
        self.assertEqual("failed", stopped["status"])
        self.assertEqual("cancelled", self.hermes.runs["run_1"]["status"])
        code, spoken = self.call("POST", "/hermes/tasks/%s/speak" % task["id"], {})
        self.assertIn("n'a pas abouti", spoken["text"])

    def test_summary_falls_back_to_plain_result(self):
        self.lemonade.close()
        self.lemonade = FakeLemonade(fail=True)
        os.environ["LEMONADE_HOST"] = self.lemonade.url
        code, task = self.call("POST", "/hermes/tasks", {"prompt": "x"})
        self.hermes.finish("run_1", "**3 fichiers** trouvés.")
        hermes_tasks._last_refresh[0] = 0
        self.call("GET", "/hermes/tasks")
        code, spoken = self.call("POST", "/hermes/tasks/latest/speak", {})
        self.assertEqual("3 fichiers trouvés.", spoken["text"])

    def test_hermes_down_marks_task_failed(self):
        os.environ["HERMES_URL"] = "http://127.0.0.1:9/v1"
        code, task = self.call("POST", "/hermes/tasks", {"prompt": "x"})
        self.assertEqual(502, code)
        self.assertEqual("failed", task["status"])
        self.assertNotIn("test-key", json.dumps(task))

    def test_disabled_and_empty_prompt(self):
        code, body = self.call("POST", "/hermes/tasks", {"text": "quelle heure est-il"})
        self.assertEqual(400, code)
        os.environ["HERMES_URL"] = ""
        code, body = self.call("GET", "/hermes/tasks")
        self.assertEqual(503, code)
        code, body = self.call("POST", "/hermes/intent", {"text": "lis le résultat"})
        self.assertEqual((200, "read"), (code, body["intent"]))

    def test_optional_token(self):
        os.environ["HERMES_TASKS_TOKEN"] = "jeton"
        code, _ = self.call("GET", "/hermes/tasks")
        self.assertEqual(401, code)
        code, _ = self.call("GET", "/hermes/tasks", headers={"X-Hermes-Token": "jeton"})
        self.assertEqual(200, code)

    def test_config_hides_key(self):
        code, cfg = self.call("GET", "/config")
        self.assertTrue(cfg["hermes_enabled"])
        self.assertTrue(cfg["hermes_key_set"])
        self.assertNotIn("test-key-not-a-secret", json.dumps(cfg))


if __name__ == "__main__":
    unittest.main()
