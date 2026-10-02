"""Preprocessing del dataset: scarica il Berkeley Segmentation Dataset (BSDS500) e prepara
le immagini usate dai benchmark in images/.

Passi di preprocessing (eseguiti una volta sola, fuori dalle misure dei tempi):
  1. crop quadrato centrale (lato = lato minore dell'immagine, 321 px per BSDS500);
  2. ridimensionamento a S x S pixel con filtro Lanczos, per ogni S in SIZES;
  3. salvataggio in PNG (lossless) come images/<id>_<S>.png.

Il ridimensionamento è necessario perché il mean shift è O(N^2) nel numero di pixel:
un'immagine BSDS a piena risoluzione (481x321 = 154k pixel) richiederebbe ore in sequenziale.

Uso:  .venv/bin/python scripts/prepare_images.py
"""

import tarfile
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data"                 # archivio originale (gitignorato)
OUT = ROOT / "images"
URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"

# Immagini scelte dal test set di BSDS500 per la varietà di colori
IMAGES = {
    "118015": "barche in un porto (immagine principale dei benchmark)",
    "108004": "tigre nell'erba",
    "145059": "persone con abiti colorati",
}
SIZES = [64, 96, 128]


def download_bsds():
    archive = CACHE / "BSR_bsds500.tgz"
    if not archive.exists():
        CACHE.mkdir(exist_ok=True)
        print(f"Download di BSDS500 (~70 MB) da {URL} ...")
        urllib.request.urlretrieve(URL, archive)
    return archive


def main():
    archive = download_bsds()
    OUT.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for image_id, description in IMAGES.items():
            member = tar.getmember(f"BSR/BSDS500/data/images/test/{image_id}.jpg")
            img = Image.open(tar.extractfile(member)).convert("RGB")
            side = min(img.size)
            left, top = (img.width - side) // 2, (img.height - side) // 2
            square = img.crop((left, top, left + side, top + side))
            for size in SIZES:
                path = OUT / f"{image_id}_{size}.png"
                square.resize((size, size), Image.LANCZOS).save(path)
                print(f"{path.relative_to(ROOT)}  ({description}, {size}x{size} = {size * size} pixel)")


if __name__ == "__main__":
    main()
