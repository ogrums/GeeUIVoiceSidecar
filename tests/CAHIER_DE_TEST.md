# Cahier de test — passerelle émotion du sidecar

## Objet et périmètre

Cette campagne valide `emo_server.py` et ses routes OpenAI compatibles. Le mode **mock** teste le serveur HTTP réel, lancé dans le processus de test, avec un serveur Lemonade local simulé et un classifieur déterministe. Il ne requiert ni réseau, ni ONNX Runtime, ni modèle. Le mode **live** envoie de vrais WAV à un sidecar déjà démarré.

La réponse de transcription attendue conserve le JSON Whisper et ajoute `emotion` (une des six émotions robot), `emotion_confidence` (nombre entre 0 et 1) et `emotion_source` (`onnx` ou `fallback`). Les labels proviennent du JSON de tête, normalisés après `/` et en minuscules ; `EMO_AUDIO_LABELS` explicite est prioritaire. Le dernier repli est `angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown`. `fearful` devient `fear`, `surprised` devient `surprise` ; `disgusted`, `other` et `unknown` deviennent `neutral`. Un WAV de moins d'une seconde renvoie `neutral` avec une confiance de 0.

## Préparation

- Python 3.9 ou supérieur ; **aucune dépendance Python supplémentaire** pour le script de campagne (bibliothèque standard).
- En live : sidecar accessible, Lemonade et modèles configurés ; WAV PCM 16 kHz, mono, 16 bits.
- Le petit corpus RAVDESS attribué est versionné ; les autres téléchargements et rapports restent exclus de Git.

## Cas de test

Les identifiants et l'ordre ci-dessous correspondent exactement à ceux du script `run_emotion_tests.py` et du rapport.

### Diagnostic (mock et live)

| ID | Objectif | Entrée / simulation | Résultat attendu |
|---|---|---|---|
| STATUS-01 | État du classifieur sans modèle | GET `/v1/emotion/status` en mock | HTTP 200, `session_loaded=false`, `onnx_configured=false`, compteurs présents |
| PRE-01 | Préflight ONNX en live | GET `/v1/emotion/status` avant les WAV | `session_loaded=true`, sinon FAIL avec la dernière erreur |

### Émotions (mock)

Chaque cas envoie un WAV de 1,2 s. Le classifieur simulé renvoie le label brut indiqué, puis le sidecar le convertit en émotion robot. Les labels RAVDESS sont repris tels quels ; `other` et `unknown` sont des labels propres à emotion2vec+.

| ID | Label brut simulé | Code RAVDESS | Émotion robot attendue | Vérifications |
|---|---|---|---|---|
| EMO-01 | `neutral` | 01 | `neutral` | HTTP 200, confiance numérique |
| EMO-02 | `calm` | 02 | `neutral` | HTTP 200, confiance numérique |
| EMO-03 | `happy` | 03 | `happy` | HTTP 200, confiance numérique, champs Whisper conservés (`text`, `language`, `segments`) |
| EMO-04 | `sad` | 04 | `sad` | HTTP 200, confiance numérique |
| EMO-05 | `angry` | 05 | `angry` | HTTP 200, confiance numérique |
| EMO-06 | `fearful` | 06 | `fear` | HTTP 200, confiance numérique |
| EMO-07 | `disgusted` | 07 | `neutral` | HTTP 200, confiance numérique |
| EMO-08 | `surprised` | 08 | `surprise` | HTTP 200, confiance numérique |
| EMO-09 | `other` | aucun | `neutral` | HTTP 200, confiance numérique |
| EMO-10 | `unknown` | aucun | `neutral` | HTTP 200, confiance numérique |

### Cas limites, amont, configuration et proxy (mock)

| ID | Objectif | Entrée / simulation | Résultat attendu |
|---|---|---|---|
| EDGE-01 | Clip trop court | WAV tonal de 0,5 s, vrai classifieur sans modèle | HTTP 200, `neutral`, confiance 0, aucun appel ONNX |
| EDGE-02 | Silence | WAV de silence de 1,2 s, label `neutral` simulé | HTTP 200, `neutral` |
| EDGE-03 | Bruit | WAV de bruit pseudo-aléatoire de 1,2 s, label `neutral` simulé | HTTP 200, `neutral` |
| EDGE-04 | Fichier corrompu | Octets non audio nommés `corrompu.wav` | HTTP 400 de l'amont conservé, pas de champ `emotion` |
| EDGE-05 | Champ manquant | Multipart sans partie `file` | HTTP 400, `{"error":"missing audio file"}` |
| EDGE-06 | Fichier volumineux | WAV de 3 s suivi d'environ 2 Mio de données | HTTP 200, pas de troncature ni de crash, `neutral` |
| EDGE-07 | Champ `model` absent | Multipart avec `file` mais sans `model` | HTTP 200 ; le sidecar ajoute `model=EMO_STT_MODEL` (défaut `Whisper-Small`) avant d'appeler Lemonade, qui l'exige |
| CONF-01 | Modèle absent | `EMO_AUDIO_ONNX` et `EMO_AUDIO_CMD` vides | HTTP 200, repli `neutral` sans erreur |
| CONF-02 | Modèle mal configuré | `EMO_AUDIO_ONNX=/chemin/inexistant.onnx` | HTTP 200, erreur ONNX interceptée, repli `neutral` |
| UP-01 | Lemonade arrêté | URL amont vers un port local fermé | HTTP 500 contrôlé `{"error":"URLError"}`, le serveur continue |
| UP-02 | Lemonade trop lent | Réponse amont retardée au-delà du timeout | HTTP 500 contrôlé `TimeoutError`, le serveur continue |
| PROXY-01 | Chat en streaming | POST `/v1/chat/completions` avec `stream=true`, trois événements SSE | HTTP 200, `text/event-stream`, fragments `Bon` et `jour` puis `[DONE]`, sans altération |
| PROXY-02 | Synthèse vocale | POST `/v1/audio/speech` | HTTP 200, `Content-Type: audio/wav`, contenu RIFF transmis |
| PROXY-03 | Modèles | GET `/v1/models` | HTTP 200, JSON de l'amont inchangé |
| PERF-01 | Latence | Cinq transcriptions successives | 5 mesures valides ; moyenne et p95 dans le rapport, sans seuil imposé |

Total mock : 26 cas. Le faux Lemonade refuse, comme le vrai, une transcription sans champ `model` (HTTP 400 `missing_argument`).

### Mode live

Le préflight `PRE-01` est inclus dans le rapport JSON et Markdown. Un cas par WAV du dossier audio suit (24 avec le corpus RAVDESS commité, 3 par expression). Le résultat attendu vient de la colonne `expected_emotion` du manifeste. Pour tout clip d'au moins une seconde, `emotion_source=fallback`, ou une confiance 0 hors silence, fait échouer le cas avec `classifieur inactif (repli)`, même si `neutral` est attendu. `LIVE-SILENCE` doit renvoyer `neutral` avec `emotion_source=silence` : un signal dont la trame de 20 ms la plus forte reste sous `EMO_MIN_RMS` (0,003 par défaut, environ −50 dBFS) n'est pas envoyé au modèle. Le mode live ajoute aussi trois cas synthétiques, `LIVE-SHORT` (0,5 s, exempté), `LIVE-SILENCE` et `LIVE-NOISE` ; les autres erreurs ne sont validées qu'en mock.

## Exécution

Depuis la racine du dépôt :

```bash
# Tout simuler, hors ligne ; sortie non nulle si au moins un cas échoue
python3 tests/run_emotion_tests.py --mode mock

# Vrais fichiers, émotion attendue lue dans manifest.csv
python3 tests/run_emotion_tests.py --mode live \
  --url http://nimbus:13306 --stt-model Whisper-Small

# Dossier explicite équivalent (valeur par défaut) :
# --audio-dir tests/samples/ravdess
```

Le dossier audio contient un `manifest.csv` à trois colonnes : le fichier, le label RAVDESS brut et l'émotion robot attendue.

```csv
file,ravdess_label,expected_emotion
03-01-03-02-01-01-01.wav,happy,happy
03-01-08-02-01-01-01.wav,surprised,surprise
```

Sans manifeste, l'émotion attendue est déduite du nom de fichier, qui doit alors contenir un label (`happy`, `sad`, `angry`, `fearful`, `surprised`, `neutral`…). Les noms RAVDESS étant numériques, le manifeste est nécessaire pour ce corpus.

Le rapport live fournit l'exactitude par label brut et par émotion robot, ainsi que la matrice de confusion robot.

Les rapports `RAPPORT.md`, `resultats.json` et `rapport.html` sont créés dans `tests/reports/<horodatage>/`. Le JSON contient les réponses détaillées, l'exactitude par émotion, la matrice de confusion et les latences. Le HTML est autonome.

## Jeu d'échantillons réels

RAVDESS (Livingstone & Russo, 2018, PLoS ONE 13(5): e0196391), Zenodo **10.5281/zenodo.1188976**, est sous **CC BY-NC-SA 4.0**. Voir `samples/ravdess/SOURCE.md` pour l'attribution complète, la liste et les modifications. Les 24 WAV convertis, trois par expression, sont commités et totalisent moins de 3 Mio.

```bash
# Téléchargement sélectif par défaut (24 WAV, nécessite remotezip)
python3 -m pip install remotezip
python3 tests/fetch_samples.py --output-dir tests/samples/ravdess

# Alternative : archive complète (199 Mio), extraction des 1440 WAV
python3 tests/fetch_samples.py --full --output-dir tests/samples/ravdess-full
```

L'option `--full` télécharge exactement `https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1`. Aucune dépendance externe n'est requise dans ce mode. Le ZIP et les fichiers hors du petit corpus commité restent ignorés par Git.

## Tests unitaires existants

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
```

### Non-régression ONNX

`test_onnx_receives_batched_float32_signal` utilise une fausse session et exige un tenseur `float32` de forme `(1, 16000)`. `test_external_head_pool_and_linear_layer` vérifie le pooling et le produit matriciel ; `test_head_labels_override_default_and_normalize` vérifie la lecture et la normalisation des labels du JSON. Les erreurs ONNX réelles restent des replis contrôlés dans les cas CONF, mais sont désormais visibles dans le statut et le journal.
