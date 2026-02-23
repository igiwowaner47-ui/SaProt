#!/usr/bin/env python3
"""Parse protein-ligand PDB complexes into LigandMPNN-style JSONL entries.

The output entry includes:
- per-chain sequence and backbone coordinates (N/CA/C/O)
- ligand atom elements and coordinates
- protein<->ligand edges under a CA-ligand distance threshold
- design mask from ligand neighborhood
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import importlib
import importlib.machinery
import math

if importlib.machinery.PathFinder.find_spec("Bio") is not None:
    bio_pdb = importlib.import_module("Bio.PDB")
    PDBParser = bio_pdb.PDBParser
    Polypeptide = bio_pdb.Polypeptide
else:
    PDBParser = None
    Polypeptide = None

BACKBONE_ATOMS = ("N", "CA", "C", "O")


@dataclass
class ResidueRecord:
    chain_id: str
    seq_letter: str
    coord: Dict[str, List[float]]


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


def _to_letter(resname: str) -> str:
    try:
        if Polypeptide is not None:
            return Polypeptide.three_to_one(resname)
        return THREE_TO_ONE[resname]
    except Exception:
        return "X"


def _is_standard_aa(residue) -> bool:
    if residue.id[0] != " ":
        return False
    if Polypeptide is not None:
        return Polypeptide.is_aa(residue, standard=False)
    return _to_letter(residue.get_resname()) != "X"


def _parse_with_biopython(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())

    residues: List[ResidueRecord] = []
    chain_to_residues: Dict[str, List[ResidueRecord]] = {}
    ligand_elements: List[str] = []
    ligand_coords: List[List[float]] = []
    ligand_residues: List[Dict[str, str]] = []

    for chain in model:
        chain_id = chain.id
        for residue in chain:
            rec = _extract_residue_record(residue, chain_id)
            if rec is not None:
                residues.append(rec)
                chain_to_residues.setdefault(chain_id, []).append(rec)
                continue

            hetflag = residue.id[0]
            if not str(hetflag).startswith("H_"):
                continue
            if residue.get_resname().strip() in {"HOH", "WAT"}:
                continue

            residue_key = {
                "chain_id": chain_id,
                "resname": residue.get_resname().strip(),
                "resseq": str(residue.id[1]),
            }

            for atom in residue:
                element = atom.element.strip() if atom.element else atom.get_name()[0]
                if not element:
                    continue
                ligand_elements.append(element.upper())
                c = atom.get_coord()
                ligand_coords.append([float(c[0]), float(c[1]), float(c[2])])
                ligand_residues.append(residue_key)

    return residues, chain_to_residues, ligand_elements, ligand_coords, ligand_residues


def _parse_with_text_fallback(pdb_path: Path):
    residues: List[ResidueRecord] = []
    chain_to_residues: Dict[str, List[ResidueRecord]] = {}
    ligand_elements: List[str] = []
    ligand_coords: List[List[float]] = []
    ligand_residues: List[Dict[str, str]] = []

    residue_atoms: Dict[tuple, Dict[str, List[float]]] = {}
    residue_names: Dict[tuple, str] = {}

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in {"ATOM", "HETATM"}:
                continue

            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            if rec == "ATOM" and resname in THREE_TO_ONE:
                key = (chain_id, resseq)
                residue_names[key] = resname
                residue_atoms.setdefault(key, {})[atom_name] = [x, y, z]
                continue

            if rec == "HETATM" and resname not in {"HOH", "WAT"}:
                element = (line[76:78].strip() or atom_name[0]).upper()
                ligand_elements.append(element)
                ligand_coords.append([x, y, z])
                ligand_residues.append({"chain_id": chain_id, "resname": resname, "resseq": resseq})

    ordered_keys = sorted(residue_atoms.keys(), key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else x[1]))
    for chain_id, resseq in ordered_keys:
        atom_map = residue_atoms[(chain_id, resseq)]
        if not all(k in atom_map for k in BACKBONE_ATOMS):
            continue
        rec = ResidueRecord(
            chain_id=chain_id,
            seq_letter=_to_letter(residue_names[(chain_id, resseq)]),
            coord={k: atom_map[k] for k in BACKBONE_ATOMS},
        )
        residues.append(rec)
        chain_to_residues.setdefault(chain_id, []).append(rec)

    return residues, chain_to_residues, ligand_elements, ligand_coords, ligand_residues


def _extract_residue_record(residue, chain_id: str) -> Optional[ResidueRecord]:
    if not _is_standard_aa(residue):
        return None

    atom_map = {}
    for atom_name in BACKBONE_ATOMS:
        if atom_name not in residue:
            return None
        atom_map[atom_name] = residue[atom_name].get_coord().astype(float).tolist()

    return ResidueRecord(
        chain_id=chain_id,
        seq_letter=_to_letter(residue.get_resname()),
        coord=atom_map,
    )


def parse_complex(pdb_path: Path, edge_cutoff: float, design_radius: float) -> Dict:
    if PDBParser is not None:
        residues, chain_to_residues, ligand_elements, ligand_coords, ligand_residues = _parse_with_biopython(pdb_path)
    else:
        residues, chain_to_residues, ligand_elements, ligand_coords, ligand_residues = _parse_with_text_fallback(pdb_path)

    if not residues:
        raise ValueError(f"No valid amino-acid backbone residues found in {pdb_path}")
    if not ligand_coords:
        raise ValueError(f"No ligand atoms found in {pdb_path}; require HETATM ligand coordinates")

    seq_chain = {}
    coords_chain = {}
    merged_seq = []

    for chain_id in sorted(chain_to_residues):
        chain_res = chain_to_residues[chain_id]
        seq = "".join(r.seq_letter for r in chain_res)
        seq_chain[f"seq_chain_{chain_id}"] = seq
        merged_seq.append(seq)

        coords_chain[f"coords_chain_{chain_id}"] = {
            f"N_chain_{chain_id}": [r.coord["N"] for r in chain_res],
            f"CA_chain_{chain_id}": [r.coord["CA"] for r in chain_res],
            f"C_chain_{chain_id}": [r.coord["C"] for r in chain_res],
            f"O_chain_{chain_id}": [r.coord["O"] for r in chain_res],
        }

    protein_ligand_edges = []
    ligand_protein_edges = []
    design_mask = []

    for i, rec in enumerate(residues):
        ca = rec.coord["CA"]
        min_dist = float("inf")
        for j, lig in enumerate(ligand_coords):
            dist = math.dist(ca, lig)
            if dist <= edge_cutoff:
                protein_ligand_edges.append([i, j])
                ligand_protein_edges.append([j, i])
            if dist < min_dist:
                min_dist = dist
        design_mask.append(1 if min_dist <= design_radius else 0)

    output = {
        "name": pdb_path.stem,
        "seq": "/".join(merged_seq),
        "num_of_chains": len(chain_to_residues),
        **seq_chain,
        **coords_chain,
        "ligand_atoms": {"element": ligand_elements, "coords": ligand_coords},
        "ligand_residues": ligand_residues,
        "protein_ligand_edges": protein_ligand_edges,
        "ligand_protein_edges": ligand_protein_edges,
        "design_mask": design_mask,
    }
    return output


def write_jsonl(records: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse PDB complexes into LigandMPNN input JSONL")
    ap.add_argument("--input_dir", type=Path, required=True, help="Directory with WT complex PDB files")
    ap.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("processed_data/ligandmpnn_inputs/wt_geometry.jsonl"),
        help="Output JSONL path",
    )
    ap.add_argument("--edge_cutoff", type=float, default=4.5, help="CA-ligand edge distance threshold (Å)")
    ap.add_argument(
        "--design_radius",
        type=float,
        default=8.0,
        help="Residues with min(CA, ligand atom) <= radius marked as designable",
    )

    args = ap.parse_args()

    pdb_files = sorted(list(args.input_dir.glob("*.pdb")) + list(args.input_dir.glob("*.ent")))
    if not pdb_files:
        raise FileNotFoundError(f"No pdb files found under {args.input_dir}")

    records = [parse_complex(p, args.edge_cutoff, args.design_radius) for p in pdb_files]
    write_jsonl(records, args.output_jsonl)
    print(f"[OK] wrote {len(records)} records -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
