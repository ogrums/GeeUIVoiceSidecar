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
| `GET` | `/v1/speakers` | Personnes enregistrées, sans empreinte (clé `SIDECAR_ADMIN_KEY`) |
| `DELETE` | `/v1/speakers/<id>` | Efface l’empreinte d’une personne (clé `SIDECAR_ADMIN_KEY`) |

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

## Qui parle (empreinte vocale)

Optionnel, activé par `SPEAKER_MODEL` (le `.env.example` pointe vers `/models/wespeaker_en_voxceleb_resnet34_LM.onnx`, téléchargé au démarrage, 25 Mo, licence CC BY 4.0). Image sans cette dépendance : `WITH_SPEAKER=0 docker compose build`.

- Chaque WAV de `/v1/audio/transcriptions` donne aussi une empreinte (sherpa-onnx, CPU, ~80 ms), en parallèle de Whisper et d'emotion2vec. La réponse gagne `speaker` (`paul` ou `invite`), `speaker_name` et `speaker_score`. L'app actuelle les ignore.
- Score cosinus ≥ `SPEAKER_THRESHOLD` et écart ≥ `SPEAKER_MARGIN` avec la deuxième personne, sinon invité. Une phrase courte (< `SPEAKER_MIN_S`) ou un score douteux (≥ `SPEAKER_DOUBT`) garde la dernière personne si elle a parlé il y a moins de `SPEAKER_STICKY_S` secondes.
- Le chat suivant part dans la session nanobot `rux-<id>` (ou `rux-invite`) avec le préfixe « (Paul parle, ton entendu : sad) ». Tant que personne n'est enregistré, tout reste comme avant (session `rux`). `SPEAKER_GUEST_ROUTE=lemonade` envoie les invités directement à Lemonade.
- Phrases traitées par le sidecar, sans nanobot (réponse OpenAI JSON ou SSE, avec une balise d'émotion) :
  - « Retiens ma voix, je suis Paul » ou « Je m'appelle Paul, retiens ma voix » : puis trois phrases d'au moins 2 s. Le robot vérifie qu'elles se ressemblent. « Annule » arrête ; 2 minutes sans rien aussi. On ne peut pas remplacer la voix de quelqu'un d'autre.
  - « Oublie ma voix » : efface seulement l'empreinte de la personne reconnue.
  - « Qui suis-je ? », « Tu me reconnais ? ».
- `SPEAKERS_FILE` (`/data/speakers.json`, volume `data`, droits 600) : nom, empreinte moyenne (256 nombres), nombre de phrases, date, `consent: "voice"`. Aucun audio. Si le modèle change, les anciennes empreintes sont ignorées.

```bash
curl -H "Authorization: Bearer $SIDECAR_ADMIN_KEY" http://nimbus:13306/v1/speakers
curl -X DELETE -H "Authorization: Bearer $SIDECAR_ADMIN_KEY" http://nimbus:13306/v1/speakers/paul
```

Test avec le vrai modèle (ignoré s'il manque) :

```bash
pip install -r requirements-speaker.txt
python3 tests/fetch_samples.py      # extraits RAVDESS
SPEAKER_TEST_MODEL=/chemin/wespeaker_en_voxceleb_resnet34_LM.onnx python3 -m unittest test_speaker -v
```

## Mode enfant / invité

Chaque voix inscrite a un rôle dans `speakers.json` : `adulte` ou `enfant`. Une voix inconnue est `invite`.

- **Inscription** : la toute première voix est `adulte`. Les suivantes reçoivent `SPEAKER_NEW_ROLE` (`enfant` par défaut), pour qu'un invité ne puisse pas s'inscrire lui-même comme adulte. Un adulte reconnu change un rôle à la voix : « Marie est une adulte », « Léa est une enfant ». La phrase n'est interceptée que si le prénom est une voix connue, sinon elle part à nanobot. Pour l'admin : `curl -X POST -H "Authorization: Bearer $KEY" -d '{"role":"enfant"}' http://nimbus:13306/v1/speakers/lea/role`.
- **STT** : `/v1/audio/transcriptions` ajoute `speaker_role` et `allowed_actions` (balises d'action que l'app a le droit d'exécuter, réglées par `ACTIONS_ADULTE`, `ACTIONS_ENFANT` et `ACTIONS_INVITE`). C'est l'app qui filtre, dans son code ; le modèle ne décide pas.
- **Enfant** : session `rux-<prénom>`, préfixe « (Léa parle, c'est un enfant) » ; la soul adapte le vocabulaire et refuse la maison. Par défaut pas de marche ni de retour à la base.
- **Invité** : `SPEAKER_GUEST_ROUTE=lemonade` par défaut, donc pas de nanobot : pas de mémoire, pas d'outils (web, maison), seulement le modèle de chat. Avec `nanobot`, session `rux-invite` et la règle « rien d'écrit » du Dream.
- Sans reconnaissance de la voix (`SPEAKER_MODEL` vide, ou personne d'inscrit) : rien ne change, pas de `allowed_actions`, l'app ne filtre pas.

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
