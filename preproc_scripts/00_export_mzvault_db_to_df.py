"""Export Thermo mzVault plain SQLite libraries to FraGNNet stage-1 CSVs.

Each output row is one spectrum. Column names match what
``preproc_scripts/02_prepare_proc.py`` reads. Structures come from
``CompoundTable.Structure`` (MOL text).

Default adducts match the ICEBERG ``nist23_hr`` export
(``ms-pred/data_scripts/nist23_hr/export_mzvault.py``): positive and negative
precursors, including ``[M+H]+``, ``[M+Na]+``, ``[M-H]-``, and the other
aliases in that set. Instrument filter stays FT + HCD.
"""

import argparse
import csv
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")

# Same set as ms-pred/data_scripts/nist23_hr/export_mzvault.py ADDUCTS.
ICEBERG_ADDUCTS = (
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M+H-H2O]+",
    "[M+H3N+H]+",
    "[M+NH4]+",
    "[M]+",
    "[M-H4O2+H]+",
    "[M+H-2H2O]+",
    "[M+H-H3N]+",
    "[M-H3N+H]+",
    "[M-NH3+H]+",
    "[M+H-NH3]+",
    "[M+H-CH2O2]+",
    "[M+CHO2]-",
    "[M-H]-",
    "[M+Cl]-",
    "[M-H2O-H]-",
    "[M-H-H2O]-",
    "[M-H-CO2]-",
)

COLUMNS = [
    "spec_id",
    "dset",
    "dset_spec_id",
    "name",
    "smiles",
    "peaks",
    "prec_mz",
    "prec_type",
    "spec_type",
    "ion_mode",
    "ion_type",
    "inst_type",
    "inst",
    "col_energy",
    "col_energy_extra_1",
    "col_energy_extra_2",
    "frag_mode",
    "formula",
    "inchikey",
    "exact_mass",
    "ri",
    "col_gas",
    "notes",
]


def repair_molblock(text: str) -> str | None:
    """Pad the V2000 counts line and append ``M  END`` when mzVault omitted them."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "$$$$" in text:
        text = text.split("$$$$", 1)[0]
    lines = text.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 4:
        return None
    lines[3] = lines[3][:33].ljust(33) + " V2000"
    if not any(line.startswith("M  END") for line in lines):
        lines.append("M  END")
    return "\n".join(lines) + "\n"


def parse_compound(structure, library_inchikey):
    """Return smiles/inchikey/formula, or a short rejection reason."""
    if structure is None:
        return None, "empty_structure"
    if isinstance(structure, bytes):
        structure = structure.decode("utf-8", errors="replace")
    structure = str(structure).strip()
    if not structure:
        return None, "empty_structure"
    block = repair_molblock(structure)
    if block is None:
        return None, "mol_parse"
    mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=True)
    if mol is None:
        return None, "mol_parse"
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if not smiles:
        return None, "mol_parse"
    inchikey = Chem.MolToInchiKey(mol)
    if not inchikey or "-" not in inchikey:
        return None, "mol_parse"
    library_key = "" if library_inchikey is None else str(library_inchikey).strip()
    if len(library_key) >= 14 and inchikey[:14] != library_key[:14]:
        return None, "inchikey_mismatch"
    try:
        formula = CalcMolFormula(mol)
    except Exception:
        return None, "formula"
    if not formula:
        return None, "formula"
    return {"smiles": smiles, "inchikey": inchikey, "formula": formula}, None


def peaks_text(blob_mass, blob_intensity) -> str | None:
    if not blob_mass or not blob_intensity:
        return None
    if isinstance(blob_mass, memoryview):
        blob_mass = blob_mass.tobytes()
    if isinstance(blob_intensity, memoryview):
        blob_intensity = blob_intensity.tobytes()
    if len(blob_mass) < 8 or len(blob_mass) % 8 or len(blob_intensity) % 8:
        return None
    masses = np.frombuffer(blob_mass, dtype="<f8")
    intens = np.frombuffer(blob_intensity, dtype="<f8")
    if masses.shape != intens.shape or masses.size == 0:
        return None
    finite = np.isfinite(masses) & np.isfinite(intens) & (intens > 0)
    if not finite.any():
        return None
    lines = [
        f"{mz:.6f} {inten:.4f}"
        for mz, inten in zip(masses[finite], intens[finite], strict=True)
    ]
    return "\n".join(lines)


def format_ce(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower().endswith("ev"):
        return text
    return f"{text} eV"


def load_molecules(con, stats: Counter) -> dict:
    molecules = {}
    rows = con.execute(
        "SELECT CompoundId, Name, Structure, InChiKey FROM CompoundTable"
    )
    for compound_id, name, structure, inchikey in rows:
        parsed, reason = parse_compound(structure, inchikey)
        if parsed is None:
            stats[f"compound_{reason}"] += 1
            continue
        parsed["name"] = "" if name is None else str(name)
        molecules[int(compound_id)] = parsed
    stats["compounds_kept"] = len(molecules)
    return molecules


def export_one(db_path: Path, prefix: str, dset: str, out_fp: Path, args, stats: Counter) -> int:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        molecules = load_molecules(con, stats)
        adducts = tuple(args.adducts)
        placeholders = ",".join("?" * len(adducts))
        query = f"""
            SELECT SpectrumId, CompoundId, PrecursorMass, CollisionEnergy,
                   Polarity, FragmentationMode, IonizationMode, MassAnalyzer,
                   InstrumentName, PrecursorIonType, blobMass, blobIntensity
            FROM SpectrumTable
            WHERE CompoundId > 0
              AND MassAnalyzer = ?
              AND FragmentationMode = ?
              AND PrecursorIonType IN ({placeholders})
        """
        cursor = con.execute(query, (args.mass_analyzer, args.frag_mode, *adducts))
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        inst_type = "QTOF" if args.mass_analyzer == "TOF" else args.mass_analyzer
        with out_fp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            while True:
                batch = cursor.fetchmany(2000)
                if not batch:
                    break
                for row in batch:
                    stats["candidates"] += 1
                    (
                        spectrum_id,
                        compound_id,
                        precursor,
                        collision,
                        _polarity,
                        frag_mode,
                        ionization,
                        _analyzer,
                        instrument,
                        prec_type,
                        blob_mass,
                        blob_intensity,
                    ) = row
                    molecule = molecules.get(int(compound_id))
                    if molecule is None:
                        stats["no_molecule"] += 1
                        continue
                    peaks = peaks_text(blob_mass, blob_intensity)
                    if peaks is None:
                        stats["no_peaks"] += 1
                        continue
                    if precursor is None or not np.isfinite(float(precursor)):
                        stats["no_precursor"] += 1
                        continue
                    if str(prec_type).endswith("+"):
                        ion_mode = "P"
                    elif str(prec_type).endswith("-"):
                        ion_mode = "N"
                    elif _polarity == "+":
                        ion_mode = "P"
                    elif _polarity == "-":
                        ion_mode = "N"
                    else:
                        stats["no_ion_mode"] += 1
                        continue
                    stats[f"adduct:{prec_type}"] += 1
                    writer.writerow(
                        {
                            "spec_id": written,
                            "dset": dset,
                            "dset_spec_id": f"{prefix}_{spectrum_id}",
                            "name": molecule["name"],
                            "smiles": molecule["smiles"],
                            "peaks": peaks,
                            "prec_mz": f"{float(precursor):.6f}",
                            "prec_type": prec_type,
                            "spec_type": "MS2",
                            "ion_mode": ion_mode,
                            "ion_type": "" if ionization is None else str(ionization),
                            "inst_type": inst_type,
                            "inst": "" if instrument is None else str(instrument),
                            "col_energy": format_ce(collision),
                            "col_energy_extra_1": "",
                            "col_energy_extra_2": "",
                            "frag_mode": "" if frag_mode is None else str(frag_mode),
                            "formula": molecule["formula"],
                            "inchikey": molecule["inchikey"],
                            "exact_mass": "",
                            "ri": "",
                            "col_gas": "",
                            "notes": "",
                        }
                    )
                    written += 1
                    if args.limit > 0 and written >= args.limit:
                        stats["written"] += written
                        print(f"{prefix}: stopped at --limit {args.limit}", flush=True)
                        return written
                if written and written % 50000 < 2000:
                    print(f"{prefix}: wrote {written}", flush=True)
        stats["written"] += written
        print(f"{prefix}: wrote {written} -> {out_fp}", flush=True)
        return written
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hr1-db",
        type=Path,
        default=Path("/home/lirui/data/2026.10.08.NIST.MS.db/MSMS_HR1_plain.db"),
    )
    parser.add_argument(
        "--hr2-db",
        type=Path,
        default=Path("/home/lirui/data/2026.10.08.NIST.MS.db/MSMS_HR2_plain.db"),
    )
    parser.add_argument(
        "--output-dp",
        type=Path,
        default=Path("/home/lirui/package/fragnnet/data/df"),
    )
    parser.add_argument(
        "--adducts",
        nargs="+",
        default=list(ICEBERG_ADDUCTS),
        help="Precursor types to keep. Default is the ICEBERG nist23_hr set.",
    )
    parser.add_argument("--mass-analyzer", default="FT")
    parser.add_argument("--frag-mode", default="HCD")
    parser.add_argument("--limit", type=int, default=0, help="max spectra kept per library; 0 = all")
    args = parser.parse_args()

    jobs = [
        (args.hr1_db, "HR1", "nist23_hr1", args.output_dp / "nist23_hr1_df.csv"),
        (args.hr2_db, "HR2", "nist23_hr2", args.output_dp / "nist23_hr2_df.csv"),
    ]
    for db_path, prefix, dset, out_fp in jobs:
        if not db_path.is_file():
            raise FileNotFoundError(db_path)
        stats: Counter = Counter()
        print(f"export {prefix} from {db_path}", flush=True)
        export_one(db_path, prefix, dset, out_fp, args, stats)
        print(f"{prefix} stats: {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
