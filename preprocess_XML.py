import os
import glob
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional

XMILE_NS = {"x": "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"}
BASE_DIR = Path("Vensim_Models")
DOCS_DIR = BASE_DIR / "XML-based-Files"
JSONLD_DIR = BASE_DIR / "JSONLD-Files"


def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return " ".join(s.replace("\t", " ").replace("\n", " ").split()).strip()


def _get_child_text(parent: ET.Element, xpath: str) -> str:
    el = parent.find(xpath, XMILE_NS)
    if el is None:
        return ""
    return _clean_text(el.text)


def _extract_sim_specs(root: ET.Element) -> Dict[str, Any]:
    sim = root.find("x:sim_specs", XMILE_NS)
    if sim is None:
        return {}
    return {
        "method": sim.get("method", ""),
        "time_units": sim.get("time_units", ""),
        "start": _get_child_text(sim, "x:start"),
        "stop": _get_child_text(sim, "x:stop"),
        "dt": _get_child_text(sim, "x:dt"),
    }


def _extract_stock(stock_el: ET.Element) -> Dict[str, Any]:
    inflows = [_clean_text(x.text) for x in stock_el.findall("x:inflow", XMILE_NS)]
    outflows = [_clean_text(x.text) for x in stock_el.findall("x:outflow", XMILE_NS)]

    return {
        "type": "stock",
        "name": stock_el.get("name", ""),
        "units": _get_child_text(stock_el, "x:units"),
        "eqn": _get_child_text(stock_el, "x:eqn"),
        "inflows": [x for x in inflows if x],
        "outflows": [x for x in outflows if x],
    }


def _extract_aux(aux_el: ET.Element) -> Dict[str, Any]:
    return {
        "type": "aux",
        "name": aux_el.get("name", ""),
        "units": _get_child_text(aux_el, "x:units"),
        "eqn": _get_child_text(aux_el, "x:eqn"),
    }


def _extract_variables(root: ET.Element) -> List[Dict[str, Any]]:
    variables_el = root.find("x:model/x:variables", XMILE_NS)
    if variables_el is None:
        return []

    items: List[Dict[str, Any]] = []
    for stock_el in variables_el.findall("x:stock", XMILE_NS):
        items.append(_extract_stock(stock_el))
    for aux_el in variables_el.findall("x:aux", XMILE_NS):
        items.append(_extract_aux(aux_el))
    return items


def xmile_to_jsonld(xmile_path: str) -> Dict[str, Any]:
    tree = ET.parse(xmile_path)
    root = tree.getroot()

    base_name = os.path.basename(xmile_path)
    model_id = os.path.splitext(base_name)[0]

    data = {
        "@type": "XMILEModel",
        "modelId": model_id,
        "sourceFile": base_name,
        "simSpecs": _extract_sim_specs(root),
        "variables": _extract_variables(root),
    }

    return data


def preprocess_xmile_folder_to_jsonld(
    xmile_dir: str | Path = DOCS_DIR,
    out_dir: str | Path = JSONLD_DIR,
    overwrite: bool = True
) -> Dict[str, Any]:
    xmile_dir = str(xmile_dir)
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(xmile_dir, "*.xmile")))
    if not paths:
        raise RuntimeError(f"No .xmile files found in: {xmile_dir}")

    written = 0
    skipped = 0

    for p in paths:
        print("Processing:", p) 
        
        base = os.path.basename(p)
        out_name = os.path.splitext(base)[0] + ".jsonld"
        out_path = os.path.join(out_dir, out_name)

        if (not overwrite) and os.path.exists(out_path):
            skipped += 1
            continue

        data = xmile_to_jsonld(p)

        # minified json (smaller than XML and much smaller than indent=2)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

        written += 1

    return {"written": written, "skipped": skipped, "out_dir": out_dir}

if __name__ == "__main__":
    result = preprocess_xmile_folder_to_jsonld()
    print(json.dumps(result, indent=2, ensure_ascii=False))
