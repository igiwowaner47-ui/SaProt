# LigandMPNN data preprocessing (local geometry flow)

Directory convention:

- `raw_data/pdb/`: WT protein-ligand complexes (`.pdb`) with protein backbone + HETATM ligand.
- `bin/ligandmpnn_tools/parse_protein_ligand.py`: parser.
- `processed_data/ligandmpnn_inputs/wt_geometry.jsonl`: output JSONL.

## Usage

```bash
python bin/ligandmpnn_tools/parse_protein_ligand.py \
  --input_dir raw_data/pdb \
  --output_jsonl processed_data/ligandmpnn_inputs/wt_geometry.jsonl \
  --edge_cutoff 4.5 \
  --design_radius 8.0
```

Each JSONL line contains sequence, backbone N/CA/C/O coordinates, ligand atom type+coords,
protein-ligand edges, reverse edges, and `design_mask`.
