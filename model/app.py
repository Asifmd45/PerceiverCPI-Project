
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

"""
Streamlit inference app for Perceiver CPI (Davis Novel Pair).
Loads the trained checkpoint and runs REAL model inference.
"""

import os
import sys
import csv

import numpy as np
import torch
import streamlit as st

# -- make sure imports resolve from the model/ directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# -- project imports
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from tape import TAPETokenizer

from chemprop.data import MoleculeDatapoint, MoleculeDataset, MoleculeDataLoader, empty_cache
from chemprop.utils import load_checkpoint, load_scalers
from chemprop.features import set_explicit_h, set_reaction

# -- constants
CHECKPOINT_PATH = os.path.join(
    APP_DIR, "results", "davis_np_fold0", "fold_0", "model_0", "model.pt"
)
TEST_CSV_PATH = os.path.join(
    APP_DIR, "data", "davis_novel_pair", "novel_pair_0_test.csv"
)
SEQUENCE_LENGTH = 500   # from args.json
MORGAN_RADIUS   = 2
MORGAN_BITS     = 2048


# -- Helper: Morgan fingerprint (exact same call as MoleculeDataset.add_features)
def smiles_to_morgan(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS)
    fp_arr = np.zeros((1,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp_vec, fp_arr)
    return fp_arr   # shape (2048,)


# -- Helper: protein sequence -> LongTensor (exact same logic as predict.py)
def sequence_to_tensor(sequence: str, tokenizer) -> torch.LongTensor:
    dummy = [0] * SEQUENCE_LENGTH
    encoded = list(tokenizer.encode(list(sequence))) + dummy
    while len(encoded) > SEQUENCE_LENGTH:
        encoded.pop(len(encoded) - 1)
    arr = np.zeros(SEQUENCE_LENGTH) + np.array(encoded)
    return torch.LongTensor(arr).unsqueeze(0)   # (1, 500)


# -- Helper: build mol batch (graph input for D-MPNN)
def smiles_to_mol_batch(smiles: str):
    dp = MoleculeDatapoint(
        smiles=[smiles],
        sequences=[[""]],
        targets=[None],
    )
    dataset = MoleculeDataset([dp])
    return dataset.batch_graph()


# -- Cached model loader
@st.cache_resource(show_spinner="Loading trained checkpoint...")
def load_model_and_scalers():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_explicit_h(False)
    set_reaction(False, "reac_diff")
    model  = load_checkpoint(CHECKPOINT_PATH, device=device)
    scaler, _, _, _ = load_scalers(CHECKPOINT_PATH)
    tokenizer = TAPETokenizer(vocab="unirep")
    model.eval()
    return model, scaler, tokenizer, device


# -- Core inference for a single (smiles, sequence) pair
def run_single_inference(smiles: str, sequence: str, model, scaler, tokenizer, device) -> float:
    # Clear global SMILES->MolGraph cache so Streamlit reruns don't reuse
    # stale graph objects built under a different featurization state
    empty_cache()
    morgan_fp = smiles_to_morgan(smiles)
    morgan_t  = torch.FloatTensor(morgan_fp).unsqueeze(0).to(device)
    seq_t = sequence_to_tensor(sequence, tokenizer).to(device)
    mol_batch = smiles_to_mol_batch(smiles)
    with torch.no_grad():
        output = model(
            mol_batch,
            seq_t,
            morgan_t,
            features_batch=None,
            atom_descriptors_batch=None,
            atom_features_batch=None,
            bond_features_batch=None,
        )
    pred = output.data.cpu().numpy()
    if scaler is not None:
        pred = scaler.inverse_transform(pred)
    return float(pred[0][0])


# -- Davis test-set evaluation (batch)
def run_test_evaluation(model, scaler, tokenizer, device, max_samples=None):
    from sklearn.metrics import mean_squared_error
    from lifelines.utils import concordance_index

    rows = []
    with open(TEST_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            if max_samples and len(rows) >= max_samples:
                break

    preds_all, labels_all = [], []
    errors = []
    progress = st.progress(0, text="Running inference on test set...")
    for i, row in enumerate(rows):
        try:
            pred = run_single_inference(
                smiles=row["smiles"], sequence=row["sequence"],
                model=model, scaler=scaler, tokenizer=tokenizer, device=device,
            )
            preds_all.append(pred)
            labels_all.append(float(row["label"]))
        except Exception as e:
            import traceback
            errors.append(f"Row {i}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            # Stop collecting errors after 5 to keep logs readable
            if len(errors) >= 5 and not preds_all:
                progress.empty()
                raise RuntimeError(
                    f"All first {len(errors)} inference attempts failed. "
                    f"First error:\n\n{errors[0]}"
                )
        progress.progress((i + 1) / len(rows), text=f"Sample {i+1}/{len(rows)}")
    progress.empty()

    if not preds_all:
        raise RuntimeError(
            f"All {len(rows)} inference attempts failed.\nFirst error:\n\n{errors[0] if errors else 'unknown'}"
        )

    test_mse = mean_squared_error(labels_all, preds_all)
    try:
        test_ci = concordance_index(labels_all, preds_all)
    except ZeroDivisionError:
        # Happens when all sampled labels are identical (e.g. all 5.0)
        # CI requires at least one pair with different label values
        test_ci = None
    return preds_all, labels_all, test_mse, test_ci, errors


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="Perceiver CPI - Binding Affinity Predictor",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Perceiver CPI — Binding Affinity Predictor")
st.caption("Davis Novel Pair baseline · D-MPNN + ECFP + 1D-CNN + Cross-Attention")

model, scaler, tokenizer, device = load_model_and_scalers()
st.success(f"Checkpoint loaded · device: **{device}**")

tab_predict, tab_eval = st.tabs(["🧪 Single Prediction", "📊 Davis Test Evaluation"])

# Tab 1: Single Prediction
with tab_predict:
    st.subheader("Predict Binding Affinity")
    st.markdown("Enter a compound SMILES and a protein amino-acid sequence, then click **Predict**.")

    col1, col2 = st.columns(2)
    with col1:
        smiles_input = st.text_area(
            "Compound SMILES",
            value="CC1=CC2=C(C=C1)N=C(N2)NC(=O)C3=CC=C(C=C3)CN4CCN(CC4)C",
            height=120,
        )
    with col2:
        seq_input = st.text_area(
            "Protein Amino-Acid Sequence",
            value=("MAESAGASSFFPLVVLLLAGSGGSGPRGVQALLCACTSCLQANYTCETDGACMVSIFNLDGMEH"
                   "HVRTCIPKVELVPAGKPFYCLSSEDLRNTHCCYTDYCNRIDLRVPSGHLKEPEHPSMWGPVELV"
                   "GIIAGPVFLLFLIIIIVFLVINYHQRVYHNRQRLDMEDPSCEMCLSKDKTLQDLVYDLSTSGSG"
                   "SGLPLFVQRTVARTIVLQEIIGKGRFGEVWRGRWRGGDVAVKIFSSREERSWFREAEIYQTVML"),
            height=120,
        )

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        smiles_input = smiles_input.strip()
        seq_input    = seq_input.strip()
        if not smiles_input:
            st.error("Please enter a SMILES string.")
        elif not seq_input:
            st.error("Please enter a protein sequence.")
        else:
            with st.spinner("Running inference..."):
                try:
                    pred_val = run_single_inference(
                        smiles=smiles_input, sequence=seq_input,
                        model=model, scaler=scaler, tokenizer=tokenizer, device=device,
                    )
                    st.success("Inference complete!")
                    st.metric(label="Predicted Binding Affinity (pKd)", value=f"{pred_val:.4f}")
                    st.info("Real prediction from trained Perceiver CPI checkpoint — not a mock or random value.")
                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    st.exception(e)

# Tab 2: Davis Test Evaluation
with tab_eval:
    st.subheader("Davis Novel Pair — Test Set Evaluation")
    st.markdown("Runs the trained checkpoint on `novel_pair_0_test.csv` and reports **MSE** and **CI**.")

    max_samples = st.number_input(
        "Max samples to evaluate (0 = all)",
        min_value=0, max_value=50000, value=500, step=50,
        help="Use â‰¥500 samples for a meaningful CI score (needs varied label values). Set to 0 for the full test set.",
    )

    if st.button("Run Test Evaluation", type="primary", use_container_width=True):
        limit = int(max_samples) if int(max_samples) > 0 else None
        with st.spinner("Evaluating..."):
            preds, labels, test_mse, test_ci, errors = run_test_evaluation(
                model, scaler, tokenizer, device, max_samples=limit
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Samples evaluated", len(preds))
        c2.metric("MSE", f"{test_mse:.4f}")
        ci_display = f"{test_ci:.4f}" if test_ci is not None else "N/A"
        c3.metric("CI (concordance index)", ci_display)
        if test_ci is None:
            st.warning(
                "CI could not be computed: all sampled labels are identical. "
                "Increase the sample count so the set contains varied affinity values."
            )

        if errors:
            with st.expander(f"{len(errors)} errors during evaluation"):
                for e in errors:
                    st.text(e)

        import pandas as pd
        n_show = min(50, len(preds))
        df = pd.DataFrame({
            "Actual (pKd)":    labels[:n_show],
            "Predicted (pKd)": [round(p, 4) for p in preds[:n_show]],
            "Error":           [round(abs(p - a), 4) for p, a in zip(preds[:n_show], labels[:n_show])],
        })
        st.dataframe(df, use_container_width=True)
        st.caption(f"Showing first {n_show} of {len(preds)} samples.")

st.divider()
st.caption(
    f"Perceiver CPI · Davis Novel Pair · "
    f"Checkpoint: results/davis_np_fold0/fold_0/model_0/model.pt · Device: {device}"
)

