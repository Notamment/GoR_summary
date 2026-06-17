#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gor_pipeline.py — Game of Rôles (Sheol) : télécharger -> transcrire+diariser -> résumer.

Pipeline en 3 étapes, pensé pour une RTX 2080 Ti (11 Go, Turing) :
  1. Téléchargement audio       (yt-dlp)
  2. Transcription + diarisation (WhisperX large-v3, FR, glossaire dans initial_prompt)
  3. Résumé                      (Ollama + Mistral Nemo, en map-reduce)

------------------------------------------------------------------------------
INSTALLATION (une seule fois)
------------------------------------------------------------------------------
  # 1) ffmpeg (système) -- ex. Windows: winget install Gyan.FFmpeg
  # 2) Torch AVEC CUDA (sinon WhisperX tourne sur CPU). Pour une 2080 Ti :
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
  # 3) Le reste
  pip install -r requirements.txt
  # 4) Ollama : https://ollama.com  puis :
  ollama pull mistral-nemo

------------------------------------------------------------------------------
DIARISATION : 2 prérequis OBLIGATOIRES (c'est LA source d'erreurs)
------------------------------------------------------------------------------
  a) Crée un token Hugging Face (https://huggingface.co/settings/tokens) et
     mets-le dans la variable d'environnement HF_TOKEN.
        Linux/macOS : export HF_TOKEN=hf_xxx
        Windows     : setx HF_TOKEN hf_xxx   (rouvrir le terminal ensuite)
  b) Accepte les conditions des modèles pyannote (gated) avec ce compte :
        https://huggingface.co/pyannote/speaker-diarization-3.1
        https://huggingface.co/pyannote/segmentation-3.0
     Sans ça, la diarisation échoue silencieusement ou renvoie une 401.

------------------------------------------------------------------------------
EXEMPLES
------------------------------------------------------------------------------
  # Tout d'un coup (épisode 2, lien YouTube du wiki) :
  python gor_pipeline.py --url "https://youtu.be/x0kh9Nnb2Rw" --episode 2 --name sheol_e02

  # Source Acast : recolle automatiquement les parties (1/3, 2/3, 3/3) + ignore les pubs :
  python gor_pipeline.py --acast --episode 4 --name sheol_e04 --min-speakers 4 --max-speakers 7
  #   …ou en donnant simplement l'URL de la page Acast d'UNE partie :
  python gor_pipeline.py --url "https://shows.acast.com/game-of-roles-magic/episodes/game-of-roles-sheol-episode-4-13" --name sheol_e04

  # CAMPAGNE ENTIÈRE : 1 dossier par épisode, .mp3 supprimés en fin d'épisode,
  # résumé détaillé + résumé court, et reprise auto là où ça s'est arrêté :
  python gor_pipeline.py --all --outdir sheol --min-speakers 4 --max-speakers 7
  #   …une plage :              python gor_pipeline.py --all --from 4 --to 10 --outdir sheol
  #   …une liste précise :      python gor_pipeline.py --episodes "4,5,6" --outdir sheol

  # Audio déjà là -> transcrire + résumer :
  python gor_pipeline.py --audio sheol_e02.mp3 --episode 2 --name sheol_e02

  # Transcription déjà faite -> juste re-résumer :
  python gor_pipeline.py --transcript sheol_e02_transcript.txt --name sheol_e02

  # Indices de diarisation (table Sheol = MJ + 4 joueurs, +invités possibles) :
  python gor_pipeline.py --url "..." --episode 2 --name sheol_e02 --min-speakers 4 --max-speakers 7
"""

import argparse
import gc
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

# ============================================================================
# GLOSSAIRE  (cf. glossaire-sheol-whisper.md — noyau + complément par épisode)
# Le initial_prompt de Whisper est plafonné ~224 tokens : on reste ciblé.
# ============================================================================
NOYAU = (
    "Game of Rôles, campagne Sheol (jeu de rôle Aria, MJ FibreTigre). "
    "Personnages : Arianne Duper, Hedy Abu Al Urssafi, Matin Miracle, Savroche Zerius, "
    "Milan Votov, la reine Julia, le chien Saucisse. "
    "Lieux : Sheol, Aria, Morneterre, Miragone, l'île d'Azur. "
    "Monde : les hommes de jadis, la langue de Jadis, le Dieu Protecteur, "
    "la Légion Silencieuse, les minotaures. "
    "Termes : méritomancie, perceptionner, dédale mental, pièce maudite, "
    "le Grimoire des sorts infinis."
)

EPISODES = {
    1: "Andreï Votov, le manoir de la clef, le détroit des mystères, le pays des Monastères, "
       "Maflak, le prêtre du Dieu Protecteur.",
    2: "Bartiméus, les juges pénitents, Valkar le Blème, Clovia Brogy, Alynata, Artyom Premier, "
       "Bibi la Malice, Perrine Delauno, Shazam, Pire Anna, Corneguigouille Onckelet, "
       "Sylvain Onckelet, la pierre de chatification, l'astéria.",
    3: "Anubis de Valora, Atalante, Alicia-Lasagua de Valora, Rex Torax, Hâsabha, Malgor X, "
       "le gouverneur Percegal, Albéric Luran, Kellis de Kellogs, Jaquenot, docteur Tox, "
       "Surimi, Biggie Sauce, BigBiz Doggodog, Petit Pépin, Clémence Gueidan, Cochenille, "
       "la vuvuzela du malaise.",
    4: "Altair, Arzach, Sirius Mingus, Origo, Hectolitre, Anathena, Darwin, Azenor, Zigobar, "
       "Nekoma, la Confrérie des Belettes, Mania, Nicolas le Loris, les Grizmelunes, "
       "l'hôtel du cœur perdu, le Labyrinthe, le Pain Parasol.",
}

# Corrections appliquées après transcription (variantes -> forme canonique)
POST_FIXES = {
    "Arthyom": "Artyom", "Arthium": "Artyom",
    "Annubis": "Anubis",
    "Clodia": "Clovia",
    "Hasaba": "Hâsabha", "Hasabha": "Hâsabha",
    "Malkar": "Valkar",
}

OLLAMA_URL = "http://localhost:11434/api/chat"


# ============================================================================
# Logs
# ============================================================================
_T0 = time.time()


def log(msg: str):
    """Affiche un message préfixé du temps écoulé depuis le lancement (MM:SS)."""
    el = int(time.time() - _T0)
    m, s = divmod(el, 60)
    print(f"[{m:02d}:{s:02d}] {msg}", flush=True)


def _fmt_dur(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# ============================================================================
# ÉTAPE 1 — Téléchargement
# ============================================================================
def step1_download(url: str, name: str, outdir: Path = Path(".")) -> Path:
    import yt_dlp
    log(f"[1/3] Téléchargement audio : {url}")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(outdir / f"{name}.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": False,
        "noprogress": False,   # laisse yt-dlp afficher sa propre barre de téléchargement
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    mp3 = outdir / f"{name}.mp3"
    if not mp3.exists():
        raise FileNotFoundError(f"Audio introuvable après téléchargement : {mp3}")
    size = mp3.stat().st_size / 1e6
    log(f"[1/3] ✅ Audio prêt : {mp3} ({size:.1f} Mo)")
    return mp3


# ============================================================================
# ÉTAPE 1 bis — Mode Acast multi-parties (téléchargement + collage)
# ============================================================================
# Le podcast Acast découpe chaque épisode en parties (1/3, 2/3, 3/3) et y insère
# des PUBLICITÉS. Ici on retrouve toutes les parties d'un épisode depuis le flux
# RSS, on les télécharge et on les recolle. Les pubs sont gérées plus loin, au
# moment du résumé : le modèle a pour consigne de les ignorer (cf. SYS).

ACAST_FEED_DEFAUT = "https://rss.acast.com/game-of-roles-magic"
_EP_RE = re.compile(r"\bEpisode\s+(\d+)\b", re.IGNORECASE)
_PART_RE = re.compile(r"\((\d+)\s*/\s*(\d+)\)")
_UA = {"User-Agent": "Mozilla/5.0 (gor_pipeline)"}


def acast_info_from_url(url: str):
    """Déduit (flux RSS, campagne, n° d'épisode) depuis l'URL d'une page Acast, p.ex.
    https://shows.acast.com/game-of-roles-magic/episodes/...-sheol-episode-4-13"""
    feed = campaign = episode = None
    mshow = re.search(r"acast\.com/([^/]+)/episodes", url)
    if mshow:
        feed = f"https://rss.acast.com/{mshow.group(1)}"
    mslug = re.search(r"-([a-zA-Zàâçéèêëîïôûù]+)-episode-(\d+)", url, re.IGNORECASE)
    if mslug:
        campaign = mslug.group(1).capitalize()
        episode = int(mslug.group(2))
    return feed, campaign, episode


def find_acast_parts(feed_url: str, campaign: str, episode: int):
    """Liste ordonnée [(x, y, titre, url_mp3), …] des parties de
    « <campaign> Episode <episode> » présentes dans le flux RSS."""
    import requests
    xml = requests.get(feed_url, headers=_UA, timeout=60).content
    root = ET.fromstring(xml)
    parts = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if campaign.lower() not in title.lower():       # bon épisode de la BONNE campagne
            continue
        mep = _EP_RE.search(title)
        if not mep or int(mep.group(1)) != episode:
            continue
        enc = item.find("enclosure")
        url = enc.get("url") if enc is not None else None
        if not url:
            continue
        mp = _PART_RE.search(title)
        x, y = (int(mp.group(1)), int(mp.group(2))) if mp else (1, 1)
        parts.append((x, y, title, url))
    # dédoublonne par n° de partie + trie dans l'ordre
    seen, uniq = set(), []
    for p in sorted(parts, key=lambda p: p[0]):
        if p[0] not in seen:
            seen.add(p[0])
            uniq.append(p)
    return uniq


def list_acast_episodes(feed_url: str, campaign: str):
    """Liste triée des numéros d'épisodes de la campagne présents dans le flux RSS."""
    import requests
    xml = requests.get(feed_url, headers=_UA, timeout=60).content
    root = ET.fromstring(xml)
    eps = set()
    for item in root.iter("item"):
        title = (item.findtext("title") or "")
        if campaign.lower() in title.lower():
            m = _EP_RE.search(title)
            if m:
                eps.add(int(m.group(1)))
    return sorted(eps)


def _download_file(url: str, dest: Path) -> Path:
    import requests
    with requests.get(url, headers=_UA, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                         desc=f"      {dest.name}", ncols=88) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def _concat_audio(parts, out_path: Path) -> Path:
    """Recolle des MP3 avec ffmpeg (concat demuxer). Repli en ré-encodage si besoin."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg introuvable sur le PATH (voir l'installation en en-tête).")
    listfile = out_path.with_suffix(".parts.txt")
    listfile.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    base = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile)]
    res = subprocess.run(base + ["-c", "copy", str(out_path)], capture_output=True, text=True)
    if res.returncode != 0:
        log("      ⚠️  Collage '-c copy' échoué, ré-encodage propre…")
        res = subprocess.run(base + ["-c:a", "libmp3lame", "-q:a", "2", str(out_path)],
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError("ffmpeg (collage) a échoué :\n" + res.stderr[-1500:])
    listfile.unlink(missing_ok=True)
    return out_path


def step1_download_acast(feed_url: str, campaign: str, episode: int, name: str,
                         outdir: Path = Path(".")) -> Path:
    log(f"[1/3] Mode Acast — parties de « {campaign} Episode {episode} »")
    log(f"      Flux RSS : {feed_url}")
    meta = find_acast_parts(feed_url, campaign, episode)
    if not meta:
        raise RuntimeError(
            f"Aucune partie trouvée pour « {campaign} Episode {episode} » dans le flux. "
            "L'épisode est peut-être trop ancien pour le flux courant — prends le lien "
            "YouTube, ou passe les URLs des parties à la main."
        )
    y = meta[0][1]
    found = [x for x, *_ in meta]
    log(f"      {len(meta)} partie(s) trouvée(s) : {found}" + (f"  (attendu 1..{y})" if y > 1 else ""))
    if y > 1 and len(meta) < y:
        log(f"      ⚠️  Parties manquantes ({len(meta)}/{y}) : le collage sera incomplet.")

    files = []
    for x, yy, title, url in meta:
        dest = outdir / f"{name}_p{x}.mp3"
        log(f"      → Partie {x}/{yy} : {title}")
        _download_file(url, dest)
        files.append(dest)

    out = outdir / f"{name}.mp3"
    log(f"      → Collage de {len(files)} partie(s) → {out}")
    _concat_audio(files, out)
    size = out.stat().st_size / 1e6
    log(f"[1/3] ✅ Audio assemblé : {out} ({size:.1f} Mo). "
        f"Parties conservées ({', '.join(f.name for f in files)}).")
    return out


# ============================================================================
# ÉTAPE 2 — Transcription + diarisation (WhisperX)
# ============================================================================
def _free_gpu(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_diarizer(hf_token: str, device: str):
    """L'emplacement de l'import ET le nom du kwarg ont changé selon les versions."""
    try:
        from whisperx.diarize import DiarizationPipeline      # versions récentes
    except Exception:
        from whisperx import DiarizationPipeline              # versions anciennes
    try:
        return DiarizationPipeline(token=hf_token, device=device)        # nouveau kwarg
    except TypeError:
        return DiarizationPipeline(use_auth_token=hf_token, device=device)  # ancien kwarg


def step2_transcribe(audio_path: Path, episode: int, name: str,
                     outdir: Path = Path("."),
                     batch_size: int = 8, compute_type: str = "float16",
                     diarize: bool = True,
                     min_speakers=None, max_speakers=None) -> Path:
    import json
    import whisperx

    device = "cuda"
    prompt = NOYAU + (" " + EPISODES.get(episode, "") if episode else "")

    log(f"[2/3] WhisperX large-v3 (FR) — batch={batch_size}, compute={compute_type}")
    log(f"      Glossaire injecté ({len(prompt)} car.) via initial_prompt"
        + (f" — épisode {episode}." if episode else " — noyau seul."))

    t = time.time()
    audio = whisperx.load_audio(str(audio_path))
    dur = len(audio) / 16000.0
    log(f"      Audio chargé : {_fmt_dur(dur)} d'enregistrement (lu en {time.time()-t:.0f}s).")

    # -- 2a. Transcription (faster-whisper batché). initial_prompt -> asr_options.
    log("      → [2a] Chargement du modèle large-v3…")
    t = time.time()
    model = whisperx.load_model(
        "large-v3", device, compute_type=compute_type, language="fr",
        asr_options={
            "initial_prompt": prompt,
            # Anti-boucles : sur la musique/les silences (intro, transitions, outro),
            # Whisper répète un mot à l'infini (« NON NON NON… », « B B B B… »).
            # Le mode batché de WhisperX n'a pas le repli en température qui corrige
            # ça d'habitude, donc on l'empêche de boucler directement au décodage :
            "no_repeat_ngram_size": 3,    # interdit de répéter un même 3-gramme
            "repetition_penalty": 1.1,    # pénalise légèrement les tokens déjà émis
        },
    )
    log(f"      → [2a] Transcription en cours ({_fmt_dur(dur)} d'audio à traiter)…")
    try:
        # print_progress affiche un % natif si la version de WhisperX le supporte
        result = model.transcribe(audio, batch_size=batch_size, print_progress=True)
    except TypeError:
        result = model.transcribe(audio, batch_size=batch_size)
    el = time.time() - t
    nseg = len(result.get("segments", []))
    rt = dur / el if el > 0 else 0
    log(f"      ✅ [2a] {nseg} segments en {_fmt_dur(el)} (≈ {rt:.1f}× temps réel).")
    _free_gpu(model)  # libère la VRAM avant l'alignement

    # -- 2b. Alignement (timestamps au mot)
    log("      → [2b] Alignement des timestamps (wav2vec2)…")
    t = time.time()
    align_model, meta = whisperx.load_align_model(language_code="fr", device=device)
    result = whisperx.align(result["segments"], align_model, meta, audio, device,
                            return_char_alignments=False)
    log(f"      ✅ [2b] Alignement terminé en {_fmt_dur(time.time()-t)}.")
    _free_gpu(align_model)

    # -- 2c. Diarisation (qui parle) -> labels SPEAKER_00, SPEAKER_01, …
    if diarize:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            log("      ⚠️  [2c] HF_TOKEN absent : diarisation ignorée (voir en-tête du script).")
        else:
            log("      → [2c] Diarisation (pyannote) — identification des locuteurs…")
            t = time.time()
            try:
                diarizer = _load_diarizer(hf_token, device)
                dkw = {}
                if min_speakers is not None:
                    dkw["min_speakers"] = min_speakers
                if max_speakers is not None:
                    dkw["max_speakers"] = max_speakers
                diarize_segments = diarizer(audio, **dkw)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                n_spk = len({s.get("speaker") for s in result.get("segments", [])
                             if s.get("speaker")})
                log(f"      ✅ [2c] {n_spk} locuteur(s) détecté(s) en {_fmt_dur(time.time()-t)}.")
                _free_gpu(diarizer)
            except Exception as e:
                log(f"      ⚠️  [2c] Diarisation échouée ({e}). Transcription gardée sans locuteurs.")

    # -- 2d. Écriture : JSON brut + transcript lisible (locuteurs regroupés)
    log("      → [2d] Écriture des fichiers…")
    json_path = outdir / f"{name}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    turns = _segments_to_turns(result.get("segments", []))
    turns = [(spk, _collapse_repeats(_apply_fixes(txt))) for spk, txt in turns]
    txt_path = outdir / f"{name}_transcript.txt"
    txt_path.write_text("\n".join(f"[{spk}] {txt}" for spk, txt in turns), encoding="utf-8")

    log(f"      💾 {json_path}  (brut, avec timestamps)")
    log(f"      💾 {txt_path}  ({len(turns)} prises de parole)")
    return txt_path


# ============================================================================
# ÉTAPE 3 — Résumé (Ollama, map-reduce)
# ============================================================================
def _segments_to_turns(segments):
    """Fusionne les segments consécutifs d'un même locuteur en 'prises de parole'."""
    turns = []
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_?")
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        if turns and turns[-1][0] == spk:
            turns[-1] = (spk, turns[-1][1] + " " + txt)
        else:
            turns.append((spk, txt))
    return turns


def _apply_fixes(text: str) -> str:
    for bad, good in POST_FIXES.items():
        text = re.sub(rf"\b{re.escape(bad)}\b", good, text)
    return text


# Filet de sécurité : si une boucle d'hallucination passe malgré tout le décodage,
# on l'écrase ici. Un même mot répété >= `threshold` fois d'affilée
# (« NON NON NON… », « B B B B… ») est ramené à une seule occurrence.
# Seuil volontairement haut : ça n'altère pas un vrai « non, non ».
_REPEAT_RE = re.compile(r"(\b[\wÀ-ÿ'’\-]+\b)(?:[\s,.;:!?…–\-]+\1\b){3,}", re.IGNORECASE)


def _collapse_repeats(text: str) -> str:
    prev = None
    while prev != text:        # ré-applique jusqu'à stabilité (boucles enchevêtrées)
        prev = text
        text = _REPEAT_RE.sub(r"\1", text)
    return text


def _parse_transcript_txt(path: Path):
    """Reconstruit les prises de parole depuis un transcript '[SPEAKER_xx] texte'."""
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\[([^\]]+)\]\s*(.*)$", line)
        if m:
            turns.append((m.group(1), m.group(2)))
        elif turns:
            turns[-1] = (turns[-1][0], turns[-1][1] + " " + line)
    # nettoie aussi d'éventuelles boucles d'un transcript déjà existant
    return [(spk, _collapse_repeats(txt)) for spk, txt in turns]


def _chunk_turns(turns, max_chars=8000):
    chunks, cur, cur_len = [], [], 0
    for spk, txt in turns:
        line = f"[{spk}] {txt}"
        if cur and cur_len + len(line) > max_chars:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _ollama_chat(model, system, user, num_ctx=8192, temperature=0.2):
    import requests
    r = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"num_ctx": num_ctx, "temperature": temperature},
    }, timeout=1200)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


SYS = ("Tu es l'assistant d'un maître du jeu. Tu traites des extraits de l'actual play "
       "Game of Rôles (jeu de rôle Aria, MJ FibreTigre), récupéré depuis un podcast qui "
       "CONTIENT DES PUBLICITÉS insérées (pré-roll, mid-roll, post-roll) sans rapport avec "
       "la partie. Règles impératives :\n"
       "1. IGNORE entièrement tout passage publicitaire/sponsor/promotionnel — pubs de "
       "produits ou de marques, « cet épisode vous est présenté par… », codes promo, appels "
       "au financement participatif (Ulule), mentions de la boutique. Ne les résume JAMAIS.\n"
       "2. Reste factuel, n'invente rien.\n"
       "3. Conserve TELS QUELS les noms propres et néologismes du show "
       "(perceptionner, méritomancie, dédale mental, etc.).")


def step3_summarize(transcript_txt: Path, name: str, outdir: Path = Path("."),
                    model="mistral-nemo", num_ctx=8192, max_chars=8000) -> Path:
    turns = _parse_transcript_txt(transcript_txt)
    chunks = _chunk_turns(turns, max_chars=max_chars)
    log(f"[3/3] Résumé via Ollama ({model}) — {len(turns)} prises de parole "
        f"→ {len(chunks)} segment(s) à digérer.")

    # -- MAP : un résumé par segment (barre de progression)
    partials = []
    t = time.time()
    for ch in tqdm(chunks, desc="      [3-map] Résumés", unit="seg", ncols=88):
        user = (
            "Résume cet extrait de transcription. Les [SPEAKER_xx] sont les différents "
            "joueurs/MJ (tu peux les ignorer si l'identité n'est pas claire). "
            "Donne les faits importants : événements, PNJ rencontrés, lieux, décisions, "
            "objets obtenus/perdus, jets de dés marquants. Concis, en français. "
            "Si l'extrait contient une publicité ou un message de sponsor, ignore-la "
            "entièrement (au besoin, réponds seulement « (publicité) »).\n\n"
            "--- EXTRAIT ---\n" + ch
        )
        partials.append(_ollama_chat(model, SYS, user, num_ctx=num_ctx))
    log(f"      ✅ [3-map] {len(chunks)} segments résumés en {_fmt_dur(time.time()-t)}.")

    # -- REDUCE : compte rendu d'épisode structuré
    log("      → [3-reduce] Synthèse finale du compte rendu…")
    t = time.time()
    user_red = (
        "Voici les résumés partiels d'un épisode, dans l'ordre chronologique. "
        "Rédige un compte rendu d'épisode structuré en français, en Markdown, avec ces sections :\n"
        "## En bref\n(4 à 6 puces des temps forts)\n"
        "## Résumé\n(le déroulé narratif)\n"
        "## PNJ rencontrés\n## Lieux\n## Quêtes & objets notables\n\n"
        "Reste fidèle aux résumés (n'ajoute rien), conserve les noms propres et néologismes.\n\n"
        "--- RÉSUMÉS PARTIELS ---\n"
        + "\n\n".join(f"[{i}] {p}" for i, p in enumerate(partials, 1))
    )
    final = _ollama_chat(model, SYS, user_red, num_ctx=max(num_ctx, 16384))
    log(f"      ✅ [3-reduce] Synthèse détaillée rédigée en {_fmt_dur(time.time()-t)}.")
    out = outdir / f"{name}_resume.md"
    out.write_text(final, encoding="utf-8")
    log(f"      💾 {out}")

    # -- COURT : version condensée 15-20 lignes, dérivée du compte rendu détaillé
    log("      → [3-court] Résumé court (15-20 lignes)…")
    t = time.time()
    user_court = (
        "Voici le compte rendu détaillé d'un épisode. Condense-le en un résumé COURT de "
        "15 à 20 lignes maximum (≈ 150-250 mots), en PROSE continue — sans titres ni puces. "
        "Couvre l'essentiel de ce qui s'est passé, conserve les noms propres, et n'ajoute "
        "aucune information absente du compte rendu.\n\n"
        "--- COMPTE RENDU DÉTAILLÉ ---\n" + final
    )
    court = _ollama_chat(model, SYS, user_court, num_ctx=num_ctx)
    log(f"      ✅ [3-court] Résumé court rédigé en {_fmt_dur(time.time()-t)}.")
    out_court = outdir / f"{name}_resume_court.md"
    out_court.write_text(court, encoding="utf-8")
    log(f"      💾 {out_court}")
    return out


# ============================================================================
# Traitement d'un épisode + boucle de campagne
# ============================================================================
def _delete_audio(outdir: Path):
    """Supprime les .mp3 volumineux d'un dossier d'épisode (garde .json/.txt/.md)."""
    freed, removed = 0, []
    for mp3 in outdir.glob("*.mp3"):
        try:
            freed += mp3.stat().st_size
            mp3.unlink()
            removed.append(mp3.name)
        except OSError:
            pass
    if removed:
        log(f"      🧹 {len(removed)} .mp3 supprimé(s), {freed/1e6:.0f} Mo libérés.")


def process_one_episode(episode: int, campaign: str, feed_url: str, outroot: str, args) -> str:
    """Traite un épisode de bout en bout dans son propre dossier.
    Renvoie 'done' ou 'skipped' ; toute exception est gérée par l'appelant."""
    name = f"{campaign.lower()}_e{episode:02d}"
    outdir = Path(outroot) / name
    outdir.mkdir(parents=True, exist_ok=True)

    resume = outdir / f"{name}_resume.md"
    if resume.exists() and not args.force:
        log(f"⏭️  {name} : déjà traité, on saute (--force pour refaire).")
        return "skipped"

    log(f"════════ {campaign} épisode {episode} → {outdir}/ ════════")
    audio = step1_download_acast(feed_url, campaign, episode, name, outdir=outdir)
    transcript = step2_transcribe(
        audio, episode, name, outdir=outdir,
        batch_size=args.batch_size, compute_type=args.compute_type,
        diarize=not args.no_diarize,
        min_speakers=args.min_speakers, max_speakers=args.max_speakers,
    )
    step3_summarize(transcript, name, outdir=outdir, model=args.model, num_ctx=args.num_ctx)
    if not args.keep_audio:
        _delete_audio(outdir)
    log(f"✅ {name} terminé.")
    return "done"


# ============================================================================
# Orchestration
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Pipeline Game of Rôles : DL -> WhisperX -> résumés Ollama.")
    # -- source (épisode unique) --
    ap.add_argument("--url", help="Lien à télécharger (YouTube/Twitch, ou page Acast d'une partie).")
    ap.add_argument("--acast", action="store_true",
                    help="Mode Acast : retrouve toutes les parties de l'épisode et les recolle.")
    ap.add_argument("--audio", help="Audio existant (saute l'étape 1).")
    ap.add_argument("--transcript", help="Transcript .txt existant (saute 1 et 2, résume seulement).")
    ap.add_argument("--name", help="Préfixe des fichiers de sortie (défaut : déduit).")
    ap.add_argument("--episode", type=int, default=0,
                    help="N° d'épisode (glossaire + recherche des parties en mode Acast).")
    # -- mode campagne (batch) --
    ap.add_argument("--all", action="store_true",
                    help="Traite TOUS les épisodes de la campagne trouvés dans le flux Acast.")
    ap.add_argument("--episodes", default=None,
                    help='Liste d\'épisodes à traiter, ex. "4,5,6" (au lieu de --all).')
    ap.add_argument("--from", type=int, default=None, dest="from_ep", help="Borne basse (avec --all).")
    ap.add_argument("--to", type=int, default=None, dest="to_ep", help="Borne haute (avec --all).")
    ap.add_argument("--outdir", default=".",
                    help="Dossier racine ; en batch, un sous-dossier par épisode y est créé.")
    ap.add_argument("--keep-audio", action="store_true",
                    help="Ne pas supprimer les .mp3 en fin d'épisode.")
    ap.add_argument("--force", action="store_true",
                    help="Retraiter un épisode même si son résumé existe déjà.")
    # -- communs --
    ap.add_argument("--campaign", default=None,
                    help="Campagne (défaut : déduit de l'URL, sinon Sheol).")
    ap.add_argument("--acast-feed", default=None,
                    help="Flux RSS Acast (défaut : déduit de l'URL, sinon le flux Game of Rôles).")
    ap.add_argument("--model", default="mistral-nemo", help="Modèle Ollama pour le résumé.")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch WhisperX (baisser si OOM : 4).")
    ap.add_argument("--compute-type", default="float16", help="float16 (2080 Ti) ou int8 si VRAM serrée.")
    ap.add_argument("--no-diarize", action="store_true", help="Désactiver la diarisation.")
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--num-ctx", type=int, default=8192, help="Contexte Ollama (12288/16384 si nécessaire).")
    args = ap.parse_args()

    # ===================== MODE CAMPAGNE (batch) =====================
    if args.all or args.episodes or args.from_ep or args.to_ep:
        feed = args.acast_feed or ACAST_FEED_DEFAUT
        campaign = args.campaign or "Sheol"
        if args.episodes:
            eps = sorted({int(x) for x in re.split(r"[,\s]+", args.episodes.strip()) if x})
        else:
            log(f"Recherche des épisodes « {campaign} » dans le flux Acast…")
            eps = list_acast_episodes(feed, campaign)
        if args.from_ep:
            eps = [e for e in eps if e >= args.from_ep]
        if args.to_ep:
            eps = [e for e in eps if e <= args.to_ep]
        if not eps:
            sys.exit(f"Aucun épisode « {campaign} » à traiter (flux vide ou plage hors limites).")
        log(f"=== Batch {campaign} — {len(eps)} épisode(s) : {eps}  (sortie dans {args.outdir}/) ===")

        stats = {"done": 0, "skipped": 0, "failed": 0}
        for ep in eps:
            try:
                stats[process_one_episode(ep, campaign, feed, args.outdir, args)] += 1
            except KeyboardInterrupt:
                log("⏹️  Interrompu (Ctrl-C).")
                break
            except Exception as e:
                stats["failed"] += 1
                log(f"❌ Épisode {ep} : échec ({type(e).__name__}: {e}). On passe au suivant.")
        log(f"=== ✅ Batch terminé en {_fmt_dur(time.time()-_T0)} — "
            f"{stats['done']} traité(s), {stats['skipped']} sauté(s), {stats['failed']} échec(s). ===")
        return

    # ===================== MODE ÉPISODE UNIQUE =====================
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    name = args.name
    if not name:
        src = args.transcript or args.audio or args.url or "episode"
        name = Path(src).stem if (args.transcript or args.audio) else "episode"

    log(f"=== Pipeline Game of Rôles — sortie : « {name} » ===")

    # Cas : on ne fait que résumer
    if args.transcript:
        step3_summarize(Path(args.transcript), name, outdir=outdir, model=args.model, num_ctx=args.num_ctx)
        log(f"=== ✅ Terminé en {_fmt_dur(time.time()-_T0)}. ===")
        return

    # Étape 1
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            sys.exit(f"Audio introuvable : {audio_path}")
        log(f"Audio fourni : {audio_path} (étape 1 sautée).")
    else:
        acast_url = bool(args.url) and "acast." in args.url.lower()
        if args.acast or acast_url:
            feed, campaign, episode = args.acast_feed, args.campaign, args.episode
            if acast_url:
                f2, c2, e2 = acast_info_from_url(args.url)
                feed = feed or f2
                campaign = campaign or c2
                episode = episode or e2
            feed = feed or ACAST_FEED_DEFAUT
            campaign = campaign or "Sheol"
            if not episode:
                sys.exit("Mode Acast : précise --episode N (ou donne l'URL de la page Acast d'une partie).")
            audio_path = step1_download_acast(feed, campaign, episode, name, outdir=outdir)
        elif args.url:
            audio_path = step1_download(args.url, name, outdir=outdir)
        else:
            sys.exit("Fournis --url, --acast --episode N, --audio, --transcript (ou --all pour la campagne).")

    # Étape 2
    transcript_txt = step2_transcribe(
        audio_path, args.episode, name, outdir=outdir,
        batch_size=args.batch_size, compute_type=args.compute_type,
        diarize=not args.no_diarize,
        min_speakers=args.min_speakers, max_speakers=args.max_speakers,
    )

    # Étape 3
    step3_summarize(transcript_txt, name, outdir=outdir, model=args.model, num_ctx=args.num_ctx)

    log(f"=== ✅ Terminé en {_fmt_dur(time.time()-_T0)}. ===")
    log(f"   Dossier {outdir}/ : {name}.json · {name}_transcript.txt · "
        f"{name}_resume.md · {name}_resume_court.md")


if __name__ == "__main__":
    main()