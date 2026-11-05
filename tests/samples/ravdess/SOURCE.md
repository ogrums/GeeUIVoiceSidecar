# Source et licence des échantillons RAVDESS

Les 24 fichiers WAV et `manifest.csv` proviennent du **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**.

- Auteurs : Steven R. Livingstone et Frank A. Russo (2018).
- Publication : *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English*, **PLoS ONE 13(5): e0196391**.
- DOI : https://doi.org/10.1371/journal.pone.0196391
- Dépôt : Zenodo, https://doi.org/10.5281/zenodo.1188976
- Archive d'origine : `Audio_Speech_Actors_01-24.zip`.
- Licence : **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**, https://creativecommons.org/licenses/by-nc-sa/4.0/
- Conditions principales : attribution, usage non commercial et partage des adaptations sous la même licence.

## Modifications

Les WAV d'origine (48 kHz, mono, PCM 16 bits) ont été rééchantillonnés en **16 kHz, mono, PCM 16 bits**, par sélection déterministe d'échantillons, pour correspondre à l'entrée ONNX du sidecar. Aucun autre traitement ni rognage. Les noms de fichiers RAVDESS sont inchangés.

## Sélection

Trois prises de chacun des huit codes d'expression : `01 neutral`, `02 calm`, `03 happy`, `04 sad`, `05 angry`, `06 fearful`, `07 disgusted`, `08 surprised`. Acteurs 01 (femme) et 02 (homme) en intensité normale, puis acteur 01 en intensité forte, sauf le neutre pour lequel l'intensité forte n'existe pas : sa troisième prise vient de l'acteur 03 en intensité normale.

Correspondance robot : `calm` et `disgusted` sont attendus comme `neutral`, `fearful` comme `fear`, `surprised` comme `surprise`. Les autres labels gardent leur nom robot. Le manifeste conserve aussi chaque label brut RAVDESS.

| Fichier | Label brut | Émotion robot attendue |
|---|---|---|
| `03-01-01-01-01-01-01.wav` | neutral | neutral |
| `03-01-01-01-01-01-02.wav` | neutral | neutral |
| `03-01-01-01-01-01-03.wav` | neutral | neutral |
| `03-01-02-01-01-01-01.wav` | calm | neutral |
| `03-01-02-01-01-01-02.wav` | calm | neutral |
| `03-01-02-02-01-01-01.wav` | calm | neutral |
| `03-01-03-01-01-01-01.wav` | happy | happy |
| `03-01-03-01-01-01-02.wav` | happy | happy |
| `03-01-03-02-01-01-01.wav` | happy | happy |
| `03-01-04-01-01-01-01.wav` | sad | sad |
| `03-01-04-01-01-01-02.wav` | sad | sad |
| `03-01-04-02-01-01-01.wav` | sad | sad |
| `03-01-05-01-01-01-01.wav` | angry | angry |
| `03-01-05-01-01-01-02.wav` | angry | angry |
| `03-01-05-02-01-01-01.wav` | angry | angry |
| `03-01-06-01-01-01-01.wav` | fearful | fear |
| `03-01-06-01-01-01-02.wav` | fearful | fear |
| `03-01-06-02-01-01-01.wav` | fearful | fear |
| `03-01-07-01-01-01-01.wav` | disgusted | neutral |
| `03-01-07-01-01-01-02.wav` | disgusted | neutral |
| `03-01-07-02-01-01-01.wav` | disgusted | neutral |
| `03-01-08-01-01-01-01.wav` | surprised | surprise |
| `03-01-08-01-01-01-02.wav` | surprised | surprise |
| `03-01-08-02-01-01-01.wav` | surprised | surprise |
