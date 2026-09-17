"""Zeichnet eine Greedy-Episode eines Checkpoints als animiertes GIF auf.

Für Präsentation und Dokumentation. Schreibt zusätzlich eine `.txt` mit den
Kennzahlen des aufgezeichneten Ausschnitts, damit später nachvollziehbar ist,
was das Video zeigt.

    python -m flappy_bird_gymnasium.dqn.record `
        runs/final_dqn/hidden_512x512_seed102/best.pt `
        --out docs/dqn_agent_start.gif --seed 90010 --seconds 30

Um eine *gemessene* Episode nachzustellen, `--max-steps` auf deren Limit setzen
und mit `--skip-to-pipe` an ihr Ende springen -- so endet der Clip exakt dort,
wo auch die Messung endete (Beispiel in ARBEITSSTAND 12.4a).

**Eine Aufnahme ist eine Illustration, kein Messwert.** Der hier übliche
Checkpoint ist der stärkste Einzellauf des Projekts; die berichtete Zahl bleibt
der Mittelwert über die Seeds (ARBEITSSTAND 12.4, 8.9). Der Dateiname und die
begleitende `.txt` sagen deshalb, aus welchem Lauf die Aufnahme stammt.

Warum GIF: `imageio`/`ffmpeg` sind keine Abhängigkeit dieses Projekts, Pillow
dagegen ist über pygame ohnehin vorhanden. Ein GIF spielt in PowerPoint und im
Browser ohne Codec-Frage.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from flappy_bird_gymnasium.dqn.agent import DQNAgent
from flappy_bird_gymnasium.dqn.config import DQNConfig
from flappy_bird_gymnasium.dqn.env_utils import make_env

#: Die Umgebung meldet 30 fps; ein GIF in Originalgeschwindigkeit hat also
#: 33 ms je Bild.
RENDER_FPS = 30


def _score_font(size: int):
    """Eine skalierbare Schrift für den Punktestand, mit Rückfallkette.

    Die Standardschrift von Pillow ist eine Bitmap in fester Größe und damit
    auf einem 288-Pixel-Bild unlesbar klein.
    """
    from PIL import ImageFont

    candidates = [
        Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
    ]
    try:  # matplotlib liegt dem Projekt ohnehin bei
        import matplotlib
        candidates.append(
            Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf")
    except Exception:
        pass
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _overlay_score(image, score: int):
    """Zeichnet den Punktestand oben ins Bild.

    Nicht über die Sprites des Spiels: `render()` blendet den Score im
    `rgb_array`-Modus bewusst aus (`flappy_bird_env.render`), und die
    Ziffern-Sprites werden ohne Alpha-Konvertierung als weißer Block geblittet.
    Hier wird er stattdessen nach dem Skalieren gezeichnet -- dadurch bleibt er
    in der Endauflösung scharf.
    """
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    text = f"{score:,}".replace(",", ".")
    font = _score_font(max(14, int(image.width * 0.12)))
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    x = (image.width - (right - left)) / 2 - left
    y = image.height * 0.06 - top
    # Kontur, damit die Zahl ueber hellem Himmel und dunklen Roehren traegt
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx or dy:
                draw.text((x + dx, y + dy), text, font=font, fill=(20, 20, 20))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return image


def record(
    checkpoint: Path,
    out: Path,
    *,
    seconds: float = 30.0,
    seed: int = 90_000,
    every: int = 1,
    scale: float = 1.0,
    skip_to_pipe: int = 0,
    max_steps: int = 2_000_000,
    with_score: bool = True,
) -> dict:
    """Spielt eine Episode greedy und schreibt die Bilder als GIF.

    Args:
        seconds: Länge der Aufnahme in Sekunden Spielzeit.
        every: nur jedes n-te Bild behalten (2 = halbe Dateigröße, doppeltes Tempo).
        scale: Skalierung der Bilder, < 1 verkleinert die Datei.
        skip_to_pipe: erst ab dieser Röhre aufzeichnen -- der Anfang jeder
            Episode sieht bei jedem Agenten gleich aus.
    """
    from PIL import Image  # Pillow kommt mit pygame; kein neues Paket

    agent = DQNAgent.load(checkpoint)
    config = DQNConfig.from_dict(
        json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    )
    # Standardmaessig ein hohes Limit, damit die Episode nicht vor dem Ende des
    # Ausschnitts abbricht. Wer eine *gemessene* Episode nachstellen will, setzt
    # `max_steps` auf deren Limit -- dann endet der Clip exakt dort, wo auch die
    # Messung endete.
    env = make_env(config, render_mode="rgb_array", max_episode_steps=max_steps)

    # `every` ueberspringt Bilder, deckt aber dieselbe Spielzeit ab -- die Zahl
    # der zu behaltenden Bilder muss deshalb durch `every` geteilt werden,
    # sonst meint `--seconds` bei every=2 die doppelte Laenge.
    wanted = max(1, int(seconds * RENDER_FPS / every))
    frames: List[Image.Image] = []
    obs, _ = env.reset(seed=seed)
    score, step, flaps = 0, 0, 0
    start_score: Optional[int] = None

    while len(frames) < wanted:
        action = int(agent.act(obs, epsilon=0.0))
        obs, _, terminated, truncated, info = env.step(action)
        score = int(info.get("score", score))
        flaps += action
        step += 1
        if score >= skip_to_pipe:
            if start_score is None:
                start_score = score
            if step % every == 0:
                image = Image.fromarray(
                    np.asarray(env.render(), dtype=np.uint8))
                if scale != 1.0:
                    image = image.resize(
                        (int(image.width * scale), int(image.height * scale)),
                        Image.LANCZOS,
                    )
                if with_score:
                    image = _overlay_score(image, score)
                frames.append(image)
        if terminated or truncated:
            break
    env.close()

    if not frames:
        raise SystemExit("keine Bilder aufgezeichnet -- Agent sofort gestorben?")

    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=int(1000 / RENDER_FPS * every), loop=0, optimize=True,
    )

    stats = {
        "checkpoint": str(checkpoint),
        "run": checkpoint.parent.name,
        "reward_preset": config.reward_preset,
        "eval_seed": seed,
        "frames": len(frames),
        "sekunden": round(len(frames) * every / RENDER_FPS, 1),
        "roehren_im_ausschnitt": score - (start_score or 0),
        "score_am_ende_des_ausschnitts": score,
        "gestorben": bool(terminated),
        "flap_rate": round(flaps / max(step, 1), 3),
        "megabyte": round(out.stat().st_size / 1e6, 2),
    }
    out.with_suffix(".txt").write_text(
        "Aufnahme eines einzelnen Laufs -- Illustration, kein Messwert.\n"
        "Die berichtete Zahl des Projekts ist der Mittelwert ueber die Seeds\n"
        "(ARBEITSSTAND 8.9), nicht der Wert dieses Laufs.\n\n"
        + "\n".join(f"{k}: {v}" for k, v in stats.items()) + "\n",
        encoding="utf-8",
    )
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Greedy-Episode als GIF aufzeichnen.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/dqn_agent_start.gif"))
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=90_000)
    parser.add_argument("--every", type=int, default=1,
                        help="nur jedes n-te Bild behalten (kleinere Datei)")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--skip-to-pipe", type=int, default=0,
                        help="erst ab dieser Röhre aufzeichnen")
    parser.add_argument("--max-steps", type=int, default=2_000_000,
                        help="Frame-Limit der Episode; auf das Limit einer "
                             "Messung setzen, um sie exakt nachzustellen")
    parser.add_argument("--no-score", action="store_true",
                        help="Punktestand nicht einblenden")
    args = parser.parse_args(argv)

    stats = record(
        args.checkpoint, args.out, seconds=args.seconds, seed=args.seed,
        every=args.every, scale=args.scale, skip_to_pipe=args.skip_to_pipe,
        max_steps=args.max_steps, with_score=not args.no_score,
    )
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
