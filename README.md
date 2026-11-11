# GeeUIVoiceSidecar

Sidecar HTTP compatible OpenAI pour le robot RUX. Il sert de passerelle entre l’application Android ([GeeUIVoice](https://github.com/ogrums/GeeUIVoice)) et les services de la machine d’inférence : Lemonade (STT, TTS Kokoro, repli chat), nanobot (cerveau, optionnel) et, si configuré, CosyVoice.

Fonctions principales :

- `POST /v1/audio/transcriptions` : transcrit via Lemonade, vote une émotion audio (emotion2vec ONNX) et l’ajoute à la réponse. Le texte est nettoyé : Whisper renvoie `" texte\n"`, le sidecar renvoie `"texte"` (JSON en UTF-8, sans échappement `\u00e9`).
- `POST /v1/chat/completions` : si `NANOBOT_URL` est défini, route vers nanobot (preset rapide ou lourd) avec repli automatique sur Lemonade ; sinon, proxy direct vers Lemonade.
- `POST /v1/audio/speech` : Pocket TTS d’abord si `POCKET_TTS_URL` est défini, repli Kokoro (Lemonade) en cas d’erreur ou de délai ; sinon proxy direct vers Kokoro. L’en-tête `X-TTS-Engine` (`pocket` ou `kokoro`) indique le moteur utilisé.
- Routes d’affect (`/affect/audio`, `/affect/text`, `/mood`, `/v1/emotion/status`).

Ce dépôt reprend le dossier `sidecar/` de [ogrums/GeeUIVoice](https://github.com/ogrums/GeeUIVoice) (commit `efd2cb6` de `main`) en le rendant autonome, lançable avec Docker Compose.

GeeUIVoice n’a pas de fichier LICENSE ; ce dépôt n’en ajoute pas. Les échantillons RAVDESS versionnés dans `tests/samples/ravdess/` restent sous [CC BY-NC-SA 4.0](tests/samples/ravdess/SOURCE.md).

## Démarrage rapide (Docker)

```bash
git clone https://github.com/ogrums/GeeUIVoiceSidecar.git
cd GeeUIVoiceSidecar
cp .env.example .env
# Éditer .env : IP LAN de l’hôte (pas 127.0.0.1, voir ci-dessous)
docker compose up -d
curl -s http://127.0.0.1:13306/v1/emotion/status
curl -s http://127.0.0.1:13306/config
```

Au premier démarrage, l’entrypoint télécharge `emotion2vec_plus_base.onnx` et `emotion2vec_head.json` depuis Hugging Face vers le volume `models`. Pour sauter ce téléchargement (le classifieur retombera alors sur `neutral`) :

```bash
# dans .env
EMO_AUDIO_DOWNLOAD=0
```

Arrêt : `docker compose down`. Les modèles restent dans le volume nommé `models`.

### Accès à Lemonade et nanobot depuis le conteneur

Dans un conteneur, `127.0.0.1` désigne le conteneur, pas la machine hôte. Renseigner l’IP LAN (exemple nimbus : `192.168.2.155`) ou `host.docker.internal` :

```bash
LEMONADE_HOST=http://192.168.2.155:13305/api/v1
NANOBOT_URL=http://192.168.2.155:8900/v1
```

Si nanobot n’écoute que sur `127.0.0.1:8900` (cas typique d’un Docker rootless), le conteneur **ne peut pas** l’atteindre, même via `host.docker.internal`. Deux options :

1. Faire écouter nanobot sur l’IP LAN (ou `0.0.0.0`).
2. Décommenter `network_mode: host` dans `docker-compose.yml` (et retirer alors la section `ports`, déjà inutile en mode host). Dans ce cas, `127.0.0.1:8900` et `127.0.0.1:13305` fonctionnent, car le processus partage la pile réseau de l’hôte.

## Variables d’environnement

Les variables vides ou absentes reprennent les défauts de `emo_server.py`. `EMO_HOST` sert aussi à calculer le port d’écoute.

| Variable | Défaut | Rôle |
|---|---|---|
| `EMO_HOST` | `http://nimbus:13306` (dans l’image : `http://0.0.0.0:13306`) | URL affichée et port d’écoute |
| `LEMONADE_HOST` | `http://nimbus:13305/api/v1` | Base OpenAI de Lemonade |
| `EMO_STT_MODEL` | `Whisper-Small` | Modèle STT ajouté au multipart s’il manque |
| `EMO_CHAT_MODEL` | `gemma4e-flash-e2b-FLM` | Modèle chat annoncé / repli Lemonade. Sur nimbus, préférer `gemma4-it-e2b-FLM` |
| `KOKORO_MODEL` | `kokoro-v1` | Modèle TTS Kokoro |
| `KOKORO_VOICE_FR` | `ff_siwis` | Voix française |
| `KOKORO_VOICE_EN` | `af_heart` | Voix anglaise |
| `POCKET_TTS_URL` | *(vide)* | Serveur Pocket TTS OpenAI (ex. `http://127.0.0.1:49112`). Vide = Kokoro seul |
| `POCKET_TTS_VOICE` | `alba` | Voix Pocket par défaut (remplace une voix Kokoro française ou inconnue) |
| `POCKET_TTS_TIMEOUT` | `20` | Délai (s) avant repli sur Kokoro |
| `COSYVOICE_HOST` | *(vide)* | Second hôte TTS ; désactivé si vide |
| `COSYVOICE_MODEL` | `cosyvoice2` | Modèle CosyVoice |
| `NANOBOT_URL` | *(vide)* | Base nanobot (`…/v1`). Vide = chat directement vers Lemonade |
| `NANOBOT_API_KEY` | *(vide)* | Clé Bearer vers nanobot |
| `NANOBOT_SESSION` | `rux` | Identifiant de session nanobot |
| `NANOBOT_FAST` | `rapide` | Preset des tours courts |
| `NANOBOT_HEAVY` | `lourd` | Preset des tours longs ou « réfléchis » |
| `NANOBOT_HEAVY_WORDS` | `explique,résume,…` | Mots qui basculent vers le preset lourd |
| `NANOBOT_HEAVY_CHARS` | `200` | Seuil de caractères pour le preset lourd |
| `NANOBOT_TIMEOUT` | `120` | Timeout (secondes) des appels nanobot |
| `EMO_AUDIO_ONNX` | *(vide)* ; `/models/emotion2vec_plus_base.onnx` dans l’image | Chemin du modèle ONNX |
| `EMO_AUDIO_HEAD` | *(vide)* ; `/models/emotion2vec_head.json` dans l’image | Tête de classification JSON |
| `EMO_AUDIO_MODEL` | `emotion2vec` | Nom exposé dans les réponses |
| `EMO_AUDIO_CMD` | *(vide)* | Commande externe alternative au ONNX |
| `EMO_AUDIO_LABELS` | labels emotion2vec+ | Surcharge de l’ordre des labels |
| `EMO_MIN_RMS` | `0.003` | Seuil de silence (RMS) |
| `EMO_ORT_PROVIDERS` | *(vide)* | Fournisseurs ONNX Runtime ; vide = auto (CUDA, ROCm, puis CPU) |
| `EMO_AUDIO_DOWNLOAD` | `1` (entrypoint) | `0` pour ne pas télécharger les modèles au démarrage |

## Routes

| Méthode | Chemin | Rôle |
|---|---|---|
| `GET` | `/v1/emotion/status` | État du classifieur ONNX, compteurs (healthcheck) |
| `GET` | `/config` | Configuration effective (sans la clé API) |
| `GET` | `/v1/models` | Proxy vers l’amont (Lemonade) |
| `GET` | `/mood` | Émotion fusionnée courante |
| `POST` | `/v1/audio/transcriptions` | STT Lemonade + émotion audio |
| `POST` | `/v1/chat/completions` | nanobot ou Lemonade, streaming SSE |
| `POST` | `/v1/audio/speech` | Proxy TTS |
| `POST` | `/affect/audio` | Vote émotion sur un WAV |
| `POST` | `/affect/text` | Injecte une émotion texte |
| `POST` | `/route` | Choisit Kokoro ou CosyVoice selon l’émotion et la longueur |

Le serveur écoute sur `0.0.0.0`, port 13306 par défaut.

## Routage nanobot et repli

Si `NANOBOT_URL` est défini, `/v1/chat/completions` part vers nanobot :

- preset `NANOBOT_FAST` (défaut `rapide`) pour un tour court ;
- preset `NANOBOT_HEAVY` (défaut `lourd`) si le texte dépasse `NANOBOT_HEAVY_CHARS` ou contient un mot de `NANOBOT_HEAVY_WORDS`.

Le sidecar se rabat sur Lemonade lorsque :

- nanobot est injoignable, expire, ou répond une erreur HTTP ;
- le flux SSE est incomplet (pas de `[DONE]`) ;
- le texte renvoyé commence par `Error calling LLM` ou `Error: `.

Sans `NANOBOT_URL`, le chat est toujours proxifié vers Lemonade.

## Tests

Depuis la racine du dépôt (Python 3.9+, bibliothèque standard pour les unitaires ; `onnxruntime` et `numpy` sont optionnels, déjà dans `requirements.txt`) :

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
```

Campagne d’émotion mock (sans réseau ni modèle) et live : voir [tests/CAHIER_DE_TEST.md](tests/CAHIER_DE_TEST.md).

```bash
python3 tests/run_emotion_tests.py --mode mock
```

## Sans Docker

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# Optionnel : placer les fichiers ONNX et renseigner les chemins
export LEMONADE_HOST=http://192.168.2.155:13305/api/v1
export NANOBOT_URL=http://127.0.0.1:8900/v1
export NANOBOT_API_KEY=…   # ou laisser vide
python -u emo_server.py
```

Le serveur reste agnostique du matériel : ONNX Runtime choisit un accélérateur disponible, sinon le CPU.

## Pocket TTS

Pocket TTS (Kyutai, modèle `french_24l`) a de meilleures voix françaises que Kokoro, mais il est plus lent. Mesures sur nimbus (Ryzen AI 9 HX 470, CPU) : RTF ≈ 0,67 (2 s pour une phrase de 40 caractères, 3,6 s pour 110 caractères), contre RTF ≈ 0,16 pour Kokoro. Le premier son sort en ≈ 0,2 s si le client lit le flux WAV (`"response_format": "wav", "stream": true`) ; en MP3, il faut attendre toute la phrase.

Activation (mode host) :

```bash
# dans .env
POCKET_TTS_URL=http://127.0.0.1:49112
POCKET_TTS_VOICE=alba
docker compose up -d
curl -s -D - -o /tmp/t.wav http://127.0.0.1:13306/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"kokoro-v1","input":"Bonjour, je suis RUX.","voice":"ff_siwis","response_format":"wav"}' | grep -i x-tts-engine
```

Côté application GeeUIVoice, le chemin Kokoro appelle Lemonade directement. Pour passer par Pocket, renseigner le champ « Phonon » avec l’adresse du sidecar (`http://nimbus:13306`) et la voix voulue (`alba`, `cosette`…). Le sidecar ne transmet pas `speed` à Pocket : le lecteur du robot applique la vitesse liée à l’émotion.
