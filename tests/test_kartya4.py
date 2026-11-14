"""Feladatkártya #4: a CI lefuttatja a neuron_halo.ipynb-t, és ellenőrzi az eredményeket.

A neurális háló tanítása gépenként kicsit eltérhet, ezért a d) részt küszöbértékekkel ellenőrizzük.
"""
import json
import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def e():
    path = ROOT / "neuron_halo.ipynb"
    if not path.exists():
        pytest.fail("Hiányzik a neuron_halo.ipynb (Colabból ugyanezen a néven mentsd vissza).", pytrace=False)
    nb = nbformat.read(path, as_version=4)
    try:
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    except CellExecutionError as hiba:
        uzenet = re.sub(r"\x1b\[[0-9;]*m", "", str(hiba)).strip().splitlines()[-1]
        pytest.fail("A notebook hibával állt le: " + uzenet, pytrace=False)
    return json.load(open(ROOT / "eredmenyek" / "kartya4.json"))


def test_a_kezi_neuron(e):
    assert e["neuron_ertek"] == pytest.approx(0.52498, abs=1e-4)


def test_b_tanulas(e):
    assert e["loss_vege"] < 0.75 * e["loss_eleje"], "A veszteségnek csökkennie kell."
    assert e["loss_vege"] < 0.40


def test_c_xor(e):
    assert e["xor_egy_neuron"] < 1.0, "Egy neuron nem tudja megoldani a XOR-t."
    assert e["xor_halo"] == 1.0, "A kétrétegű hálónak mind a 4 esetet el kell találnia."


def test_d_halo_szerkezete(e):
    assert e["parameterek"] == 41, "3 → 8 → 1 háló: 3·8+8 + 8·1+1 = 41 paraméter."


def test_d_hivasjoslo(e):
    assert e["accuracy"] >= 0.79, f"Pontosság: {e['accuracy']:.3f}, legalább 0,79 kell."
    assert e["recall_hiba"] >= 0.50, f"Felidézés: {e['recall_hiba']:.3f}, legalább 0,50 kell."
