# Rapport de test sidecar — mock

- Date : 2026-11-05T21:33:25.304963+01:00
- Cible : `amont Lemonade simulé localement`
- **Résultat : 26/26 réussis, 0 échoué(s)**
- Latence moyenne : 19.29 ms ; p95 : 96.42 ms

## Détail des cas

| ID | Objectif | Statut | Attendu | Observé | Latence (ms) |
|---|---|---|---|---|---:|
| STATUS-01 | Diagnostic du classifieur ONNX | **PASS** | HTTP 200, configuration, session et compteurs exposés | HTTP 200, b'{"onnx_path": "", "onnx_configured": false, "onnx_file_exists": false, "onnxruntime_importable": true, "onnxruntime_version": "1.31.1", "available_providers": ["AzureExecutionProvider", "CPUExecutionProvider"], "selected_providers": ["CPUExecutionProvider"], "head_path": "", "head_loaded": false, "labels": null, "session_loaded": false, "last_error": null, "classifications": 0, "failures": 0}' | 83.64 |
| EMO-01 | Transcription et émotion neutral | **PASS** | HTTP 200, emotion=neutral, confiance numérique | HTTP 200, emotion=neutral, confiance=0.93 | 36.53 |
| EMO-02 | Transcription et émotion neutral | **PASS** | HTTP 200, emotion=neutral, confiance numérique | HTTP 200, emotion=neutral, confiance=0.93 | 7.63 |
| EMO-03 | Transcription et émotion happy | **PASS** | HTTP 200, emotion=happy, confiance numérique | HTTP 200, emotion=happy, confiance=0.93 | 6.03 |
| EMO-04 | Transcription et émotion sad | **PASS** | HTTP 200, emotion=sad, confiance numérique | HTTP 200, emotion=sad, confiance=0.93 | 5.7 |
| EMO-05 | Transcription et émotion angry | **PASS** | HTTP 200, emotion=angry, confiance numérique | HTTP 200, emotion=angry, confiance=0.93 | 5.95 |
| EMO-06 | Transcription et émotion fear | **PASS** | HTTP 200, emotion=fear, confiance numérique | HTTP 200, emotion=fear, confiance=0.93 | 5.19 |
| EMO-07 | Transcription et émotion neutral | **PASS** | HTTP 200, emotion=neutral, confiance numérique | HTTP 200, emotion=neutral, confiance=0.93 | 5.8 |
| EMO-08 | Transcription et émotion surprise | **PASS** | HTTP 200, emotion=surprise, confiance numérique | HTTP 200, emotion=surprise, confiance=0.93 | 5.6 |
| EMO-09 | Transcription et émotion neutral | **PASS** | HTTP 200, emotion=neutral, confiance numérique | HTTP 200, emotion=neutral, confiance=0.93 | 5.45 |
| EMO-10 | Transcription et émotion neutral | **PASS** | HTTP 200, emotion=neutral, confiance numérique | HTTP 200, emotion=neutral, confiance=0.93 | 5.84 |
| EDGE-01 | Transcription et émotion neutral | **PASS** | HTTP 200, emotion=neutral, confiance numérique | HTTP 200, emotion=neutral, confiance=0.0 | 4.78 |
| EDGE-02 | Transcription et émotion neutral | **PASS** | HTTP 200, emotion=neutral, confiance numérique | HTTP 200, emotion=neutral, confiance=0.93 | 5.25 |
| EDGE-03 | Transcription et émotion neutral | **PASS** | HTTP 200, emotion=neutral, confiance numérique | HTTP 200, emotion=neutral, confiance=0.93 | 6.51 |
| EDGE-04 | Rejet d'un fichier non audio | **PASS** | HTTP 400 amont conservé, aucun champ émotion | HTTP 400, b'{"error":{"message":"invalid audio","type":"invalid_request_error"}}' | 5.15 |
| EDGE-05 | Champ file absent | **PASS** | HTTP 400 missing audio file | HTTP 400, b'{"error": "missing audio file"}' | 1.13 |
| EDGE-06 | Transcription et émotion neutral | **PASS** | HTTP 200, emotion=neutral, confiance numérique | HTTP 200, emotion=neutral, confiance=0.93 | 96.42 |
| EDGE-07 | Requête client sans champ model | **PASS** | HTTP 200, le sidecar ajoute model=Whisper-Small | HTTP 200, model reçu par Lemonade=Whisper-Small | 8.53 |
| CONF-01 | Transcription et émotion neutral | **PASS** | HTTP 200, emotion=neutral, confiance numérique | HTTP 200, emotion=neutral, confiance=0.0 | 5.3 |
| CONF-02 | Transcription et émotion neutral | **PASS** | HTTP 200, emotion=neutral, confiance numérique | HTTP 200, emotion=neutral, confiance=0.0 | 9.3 |
| UP-01 | Lemonade indisponible | **PASS** | Erreur contrôlée HTTP 500 (comportement actuel) | HTTP 500, b'{"error": "URLError"}' | 6.22 |
| UP-02 | Dépassement de délai Lemonade | **PASS** | Erreur contrôlée HTTP 500 (comportement actuel) | HTTP 500, b'{"error": "TimeoutError"}' | 106.68 |
| PROXY-01 | Proxy chat SSE et chunks | **PASS** | SSE complet Bon+jour+[DONE] | HTTP 200, b'data: {"choices":[{"delta":{"content":"Bon"}}]}\n\ndata: {"choices":[{"delta":{"content":"jour"}}]}\n\ndata: [DONE]\n\n' | 62.51 |
| PROXY-02 | Proxy synthèse vocale WAV | **PASS** | HTTP 200 audio/wav RIFF | HTTP 200, audio/wav, 3244 octets | 2.28 |
| PROXY-03 | Proxy liste des modèles | **PASS** | HTTP 200 et liste Whisper inchangée | HTTP 200, b'{"object": "list", "data": [{"id": "Whisper-Small"}]}' | 1.94 |
| PERF-01 | Mesure de latence transcription | **PASS** | 5 mesures valides | 5/5 mesures | 6.05 |

Les détails JSON complets de chaque réponse sont dans `resultats.json`.
