# Bolyai Labs – 4. blokk: Neuronok a gépben (PyTorch alapok)

## Feladatkártya #4

> Fejlesztőként szeretném a hívásjóslót neurális hálóval is megépíteni (PyTorch), hogy később képekre és szövegre is bővíthessük.

| Rész | Feladat |
| --- | --- |
| a) | kézi neuron: `sigmoid(x @ w + b)` |
| b) | a tanítási ciklus négy sora |
| c) | XOR: egy neuron kevés, a kétrétegű háló megoldja |
| d) | hívásjósló háló (3 → 8 → 1) a novemberi naplón |

- [ ] saját repó ebből a sablonból (pl. `neuron-13b-07`, Public)
- [ ] a `neuron_halo.ipynb` TODO-jai kitöltve, az önellenőrzés lefut
- [ ] visszamentve Colabból, commitüzenet: `feat: első neurális háló (feladatkártya #4)`
- [ ] az Actions fülön a **Feladatkártya 4** futás zöld
- [ ] a repó linkje beadva a Teams-feladatba

**Érték:** 5 sprintpont (zöld build 2, határidő 1, commitüzenet 1, kilépőkérdések 1). **Határidő:** a 2. óra vége.

## Megnyitás Colabban

[colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub** fül → felhasználónév → saját repó → `neuron_halo.ipynb`. A Colabban a PyTorch előre telepítve van, GPU nem kell.

## Ha piros a build

Nyisd meg a piros futást az Actions fülön, a tesztek nevéből látod, melyik rész (a–d) hibás. A neurális háló tanítása gépenként kicsit eltérhet, ezért a d) részt a CI küszöbértékekkel ellenőrzi (pontosság ≥ 0,79, felidézés ≥ 0,50).
