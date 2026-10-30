"""
ML Model Loader and Inference Service
--------------------------------------
Loads trained joblib pipelines once at startup and exposes prediction methods.
SHAP explanations are computed using TreeExplainer when supported.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _BACKEND_DIR / "models_store"

# Risk level thresholds
_RISK_THRESHOLDS = {"low": 0.35, "moderate": 0.65}


def _risk_level(prob: float) -> str:
    if prob < _RISK_THRESHOLDS["low"]:
        return "low"
    if prob < _RISK_THRESHOLDS["moderate"]:
        return "moderate"
    return "high"


def _load(filename: str) -> Optional[dict]:
    path = _MODELS_DIR / filename
    if not path.exists():
        logger.warning(f"Model file not found: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        logger.error(f"Failed to load {path}: {exc}")
        return None


# ── SHAP ──────────────────────────────────────────────────────────────────────

def _shap_values_for_pipeline(
    artifact: dict, X: pd.DataFrame
) -> Optional[List[Dict[str, Any]]]:
    """Compute per-feature SHAP values for a single sample.
    Returns a list of {feature, shap_value, value} dicts or None on failure.
    """
    try:
        import shap

        pipeline = artifact["pipeline"]
        preprocessor = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["classifier"]

        X_transformed = preprocessor.transform(X)

        # Derive transformed feature names
        feature_names: List[str] = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                try:
                    ohe = trans.named_steps["encoder"]
                    feature_names.extend(ohe.get_feature_names_out(cols).tolist())
                except Exception:
                    feature_names.extend([f"{name}_{i}" for i in range(X_transformed.shape[1] - len(feature_names))])

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_transformed)

        # Binary: sv is (n_samples, n_features); Multiclass: sv is list
        if isinstance(sv, list):
            # Use positive class (index of highest predicted prob)
            proba = clf.predict_proba(X_transformed)
            cls_idx = int(np.argmax(proba[0]))
            sv_row = sv[cls_idx][0] if sv[cls_idx].ndim == 2 else sv[cls_idx]
        else:
            sv_row = sv[0] if sv.ndim == 2 else sv

        result = []
        for i, (feat, val) in enumerate(zip(feature_names, sv_row)):
            original_val = X.iloc[0].get(feat, None) if feat in X.columns else None
            result.append({
                "feature": feat,
                "shap_value": round(float(val), 6),
                "value": float(original_val) if original_val is not None else None,
            })
        # Sort by absolute SHAP value descending
        result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return result[:15]  # Return top-15 features

    except Exception as exc:
        logger.warning(f"SHAP computation failed: {exc}")
        return None


# ── Prediction helpers ─────────────────────────────────────────────────────────

def _binary_predict(
    artifact: Optional[dict],
    features: Dict[str, Any],
    organ_name: str,
    include_shap: bool = True,
) -> Optional[Dict[str, Any]]:
    if artifact is None:
        return None
    try:
        pipeline = artifact["pipeline"]
        num_features = artifact["num_features"]
        cat_features = artifact["cat_features"]
        all_features = num_features + cat_features
        threshold = artifact.get("threshold", 0.5)

        row = {f: features.get(f, None) for f in all_features}
        X = pd.DataFrame([row])

        prob = float(pipeline.predict_proba(X)[0, 1])
        label = int(prob >= threshold)

        shap_vals = None
        if include_shap:
            shap_vals = _shap_values_for_pipeline(artifact, X)

        return {
            "organ": organ_name,
            "probability": round(prob, 4),
            "label": label,
            "threshold": threshold,
            "risk_level": _risk_level(prob),
            "model_type": "LightGBM",
            "shap_available": shap_vals is not None,
            "shap_values": shap_vals,
        }
    except Exception as exc:
        logger.error(f"{organ_name} prediction error: {exc}", exc_info=True)
        return None


# ── Public Inference API ───────────────────────────────────────────────────────

class MLInferenceService:
    """Singleton service — loads models once, exposes predict_* methods."""

    def __init__(self):
        logger.info("Loading ML model artifacts …")
        self._sepsis = _load("sepsis_model.joblib")
        self._aki = _load("aki_model.joblib")
        self._cardio = _load("cardio_model.joblib")
        self._liver = _load("liver_model.joblib")
        self._neuro = _load("neuro_model.joblib")
        self._resp = _load("resp_model.joblib")
        self._antibiotic = _load("antibiotic_model.joblib")
        self._armd_susceptibility = _load("antibiotic_susceptibility/lightgbm_susceptibility.joblib")
        self._armd_explainer = None
        self._log_loaded_models()

    def _log_loaded_models(self):
        loaded = [
            n for n, a in [
                ("sepsis", self._sepsis), ("aki", self._aki), ("cardio", self._cardio),
                ("liver", self._liver), ("neuro", self._neuro), ("resp", self._resp),
                ("synthetic_antibiotic", self._antibiotic),
                ("armd_susceptibility", self._armd_susceptibility),
            ] if a is not None
        ]
        logger.info(f"Models loaded: {loaded}")

    # ── Sepsis ─────────────────────────────────────────────────────────────

    def predict_sepsis(self, features: Dict[str, Any], include_shap: bool = True) -> Optional[Dict[str, Any]]:
        result = _binary_predict(self._sepsis, features, "sepsis", include_shap)
        if result:
            result["model_type"] = "LightGBM"
            result["sepsis_probability"] = result.pop("probability")
            result["sepsis_label"] = result.pop("label")
        return result

    # ── Organ Models ───────────────────────────────────────────────────────

    def predict_aki(self, features: Dict[str, Any], include_shap: bool = True) -> Optional[Dict[str, Any]]:
        return _binary_predict(self._aki, features, "acute_kidney_injury", include_shap)

    def predict_cardio(self, features: Dict[str, Any], include_shap: bool = True) -> Optional[Dict[str, Any]]:
        return _binary_predict(self._cardio, features, "cardiovascular", include_shap)

    def predict_liver(self, features: Dict[str, Any], include_shap: bool = True) -> Optional[Dict[str, Any]]:
        return _binary_predict(self._liver, features, "liver", include_shap)

    def predict_neuro(self, features: Dict[str, Any], include_shap: bool = True) -> Optional[Dict[str, Any]]:
        return _binary_predict(self._neuro, features, "neurological", include_shap)

    def predict_resp(self, features: Dict[str, Any], include_shap: bool = True) -> Optional[Dict[str, Any]]:
        return _binary_predict(self._resp, features, "respiratory", include_shap)

    # ── Antibiotic / Antimicrobial Susceptibility ──────────────────────────

    def predict_antibiotic(
        self, features: Dict[str, Any], include_shap: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Predict antimicrobial susceptibility using the ARMD-ECUH LightGBM model.
        Falls back to legacy synthetic pipeline only if ARMD model is unavailable.
        """
        # 1. Prefer the real ARMD-ECUH model
        if self._armd_susceptibility is not None:
            try:
                bundle = self._armd_susceptibility
                model = bundle["model"]
                feature_columns = bundle["feature_columns"]
                category_mappings = bundle["category_mappings"]
                classes = list(bundle["label_encoder"].classes_)

                # Map input values
                org_in = str(features.get("organism") or "ESCHERICHIA COLI").strip().upper()
                drug_in = str(features.get("antibiotic") or "Ciprofloxacin").strip()
                # Normalize drug title case if needed
                drug_in_title = drug_in.title()
                if drug_in_title in category_mappings.get("antibiotic", {}):
                    drug_in = drug_in_title

                culture_in = str(features.get("culture_description") or "URINE").strip().upper()
                mode_in = str(features.get("ordering_mode") or "Inpatient").strip()

                context_dict = {
                    "organism": org_in,
                    "antibiotic": drug_in,
                    "culture_description": culture_in,
                    "ordering_mode": mode_in,
                }

                # Encode ordinal values (unknown mapped to -1)
                encoded_row = {}
                for col in feature_columns:
                    val = context_dict.get(col, "UNKNOWN")
                    encoded_row[col] = [category_mappings[col].get(val, -1)]

                X = pd.DataFrame(encoded_row)
                proba = model.predict_proba(X)[0]
                pred_idx = int(np.argmax(proba))
                pred_class = classes[pred_idx]

                prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

                # Compute genuine SHAP attribution for ARMD-ECUH model (cached explainer)
                shap_list = None
                if include_shap:
                    try:
                        import shap
                        if self._armd_explainer is None:
                            self._armd_explainer = shap.TreeExplainer(model)
                        sv = self._armd_explainer.shap_values(X)
                        if isinstance(sv, np.ndarray) and sv.ndim == 3:
                            class_sv = sv[0, :, pred_idx]
                        elif isinstance(sv, list):
                            class_sv = sv[pred_idx][0] if sv[pred_idx].ndim == 2 else sv[pred_idx]
                        else:
                            class_sv = sv[0]

                        shap_list = [
                            {
                                "feature": f,
                                "shap_value": round(float(v), 4),
                                "value": context_dict.get(f),
                            }
                            for f, v in zip(feature_columns, class_sv)
                        ]
                        # Sort by absolute SHAP magnitude descending
                        shap_list.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
                    except Exception as shap_exc:
                        logger.warning(f"TreeSHAP calculation for ARMD model failed: {shap_exc}")

                return {
                    "predicted_class": pred_class,
                    "probabilities": prob_dict,
                    "model_type": "LightGBM Multiclass Classifier (ARMD-ECUH)",
                    "model_version": "v1.0-armd",
                    "data_source": "ARMD-ECUH (ECU Health, N=1,931,685 culture tests)",
                    "organism": org_in,
                    "antibiotic": drug_in,
                    "culture_description": culture_in,
                    "ordering_mode": mode_in,
                    "shap_available": shap_list is not None,
                    "shap_values": shap_list,
                    "metrics_summary": {
                        "accuracy": 0.685,
                        "balanced_accuracy": 0.702,
                        "macro_roc_auc": 0.873,
                        "test_records": 390979,
                        "test_patients": 12478,
                    },
                    "data_disclaimer": (
                        "Data source: ARMD-ECUH retrospective research dataset. "
                        "Model outputs are research estimates and have not been prospectively or clinically validated. "
                        "Susceptibility prediction is an in-vitro phenotype classification and does NOT constitute "
                        "an individualized antibiotic recommendation."
                    ),
                }
            except Exception as exc:
                logger.error(f"ARMD susceptibility prediction error: {exc}", exc_info=True)

        # 2. Fallback to legacy synthetic model if ARMD model is not loaded
        artifact = self._antibiotic
        if artifact is None:
            return None
        try:
            pipeline = artifact["pipeline"]
            le = artifact["label_encoder"]
            num_features = artifact["num_features"]
            cat_features = artifact["cat_features"]
            all_features = num_features + cat_features
            classes = artifact["classes"]

            row = {f: features.get(f, None) for f in all_features}
            X = pd.DataFrame([row])

            proba = pipeline.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            pred_class = le.inverse_transform([pred_idx])[0]

            prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
            return {
                "predicted_class": pred_class,
                "probabilities": prob_dict,
                "model_type": "LightGBM (Synthetic Demo)",
                "model_version": "v0.9-synthetic",
                "data_source": "Synthetic Benchmark Prototype",
                "shap_available": False,
                "shap_values": None,
                "data_disclaimer": (
                    "Data source: Benchmark prototype dataset. "
                    "Model outputs are research estimates and have not been prospectively or clinically validated."
                ),
            }
        except Exception as exc:
            logger.error(f"Fallback antibiotic prediction error: {exc}", exc_info=True)
            return None

    def predict_patient_antibiotic_suitability(
        self, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predicts and ranks potentially suitable antimicrobial options using:
        1. ARMD-ECUH microbiology culture predictions (LightGBM in-vitro phenotype)
        2. Patient-specific clinical parameters (Age, Gender, eGFR, Creatinine, Liver, Site)
        3. Documented drug allergies and contraindications
        4. Prior antibiotic exposure and resistance profile (ESBL, MRSA, etc.)
        5. Drug-drug interactions with concurrent medications
        6. TreeSHAP explainability per candidate
        7. Antimicrobial stewardship tiering
        """
        age = float(request_data.get("patient_age", 60))
        gender = str(request_data.get("patient_gender", "M")).upper()
        site = str(request_data.get("infection_site", "URINE")).upper()
        organism = str(request_data.get("identified_organism", "ESCHERICHIA COLI")).upper()

        allergies = [str(a).strip().lower() for a in request_data.get("drug_allergies", []) if a]
        res_profile = [str(r).strip().upper() for r in request_data.get("antibiotic_resistance_profile", []) if r]
        prior_exposure = [str(p).strip().lower() for p in request_data.get("previous_antibiotic_exposure", []) if p]

        creatinine = float(request_data.get("creatinine", 1.0))
        egfr = float(request_data.get("egfr", 90.0))
        total_bilirubin = request_data.get("total_bilirubin")
        total_bilirubin = float(total_bilirubin) if total_bilirubin is not None else None
        ast_alt_elevated = bool(request_data.get("ast_alt_elevated", False))

        medications = [str(m).strip().lower() for m in request_data.get("current_medications", []) if m]

        # Candidate antibiotic panel
        user_candidates = request_data.get("candidate_antibiotics")
        if user_candidates and isinstance(user_candidates, list) and len(user_candidates) > 0:
            candidates = user_candidates
        else:
            candidates = [
                "Ceftriaxone",
                "Ciprofloxacin",
                "Piperacillin/tazobactam",
                "Meropenem",
                "Vancomycin",
                "Levofloxacin",
                "Cefepime",
                "Ampicillin",
                "Gentamicin",
                "Trimethoprim/sulfamethoxazole",
                "Aztreonam",
                "Tobramycin",
            ]

        # Normalized allergy flags
        has_pcn_allergy = any("penicillin" in a or "beta-lactam" in a or "beta_lactam" in a for a in allergies)
        has_ceph_allergy = any("cephalosporin" in a for a in allergies)
        has_quinolone_allergy = any("quinolone" in a or "cipro" in a or "levo" in a for a in allergies)
        has_sulfa_allergy = any("sulfa" in a or "trimethoprim" in a for a in allergies)
        has_glyco_allergy = any("vancomycin" in a or "glycopeptide" in a for a in allergies)
        has_amino_allergy = any("aminoglycoside" in a or "gentamicin" in a or "tobramycin" in a for a in allergies)

        # Stewardship classifications
        stewardship_map = {
            "Ceftriaxone": "First-Line Preferred",
            "Ampicillin": "First-Line Preferred",
            "Ciprofloxacin": "First-Line Preferred",
            "Trimethoprim/sulfamethoxazole": "First-Line Preferred",
            "Piperacillin/tazobactam": "Broad-Spectrum Reserve",
            "Cefepime": "Broad-Spectrum Reserve",
            "Aztreonam": "Broad-Spectrum Reserve",
            "Levofloxacin": "Broad-Spectrum Reserve",
            "Gentamicin": "Broad-Spectrum Reserve",
            "Tobramycin": "Broad-Spectrum Reserve",
            "Meropenem": "Restricted / Escalation",
            "Vancomycin": "Restricted / Escalation",
        }

        evaluated_candidates = []

        for drug in candidates:
            # 1. In-vitro susceptibility prediction via ARMD-ECUH model
            pred = self.predict_antibiotic({
                "organism": organism,
                "antibiotic": drug,
                "culture_description": site,
                "ordering_mode": "Inpatient",
            })

            if pred and "probabilities" in pred:
                probs = pred["probabilities"]
                p_sus = float(probs.get("Susceptible", 0.5))
                p_res = float(probs.get("Resistant", 0.3))
                p_int = float(probs.get("Intermediate", 0.2))
                predicted_class = pred.get("predicted_class", "Susceptible")
                shap_vals = pred.get("shap_values")
            else:
                p_sus, p_res, p_int = 0.5, 0.3, 0.2
                predicted_class = "Susceptible"
                shap_vals = None

            # Base score derived from predicted susceptibility probabilities
            base_score = (p_sus * 80.0) - (p_res * 65.0) - (p_int * 20.0)

            penalties = 0.0
            bonuses = 0.0
            warnings: List[str] = []
            allergy_contraindication = False
            allergy_warning = None
            renal_warning = None
            hepatic_warning = None
            interaction_warning = None

            drug_lower = drug.lower()

            # 2. Allergy & Hypersensitivity Analysis
            if drug in ("Ampicillin", "Piperacillin/tazobactam") and has_pcn_allergy:
                allergy_contraindication = True
                allergy_warning = f"CONTRAINDICATION: Documented penicillin allergy. Severe hypersensitivity/anaphylaxis risk with {drug}."
                warnings.append(allergy_warning)
                penalties += 80.0
            elif drug in ("Ceftriaxone", "Cefepime"):
                if has_ceph_allergy:
                    allergy_contraindication = True
                    allergy_warning = f"CONTRAINDICATION: Documented cephalosporin hypersensitivity. Cross-reactivity with {drug}."
                    warnings.append(allergy_warning)
                    penalties += 80.0
                elif has_pcn_allergy:
                    allergy_warning = "Caution: Documented penicillin allergy. Potential ~2-5% cross-reactivity with cephalosporins; monitor closely."
                    warnings.append(allergy_warning)
                    penalties += 15.0
            elif drug == "Meropenem" and has_pcn_allergy:
                allergy_warning = "Caution: Documented penicillin allergy. Cross-reactivity with carbapenems is low (<1%), but monitor during administration."
                warnings.append(allergy_warning)
                penalties += 8.0
            elif drug == "Aztreonam" and has_pcn_allergy:
                allergy_warning = "✓ Monobactam: Lack of cross-reactivity with penicillins makes Aztreonam a preferred beta-lactam substitute."
                bonuses += 10.0
            elif drug in ("Ciprofloxacin", "Levofloxacin") and has_quinolone_allergy:
                allergy_contraindication = True
                allergy_warning = f"CONTRAINDICATION: Documented fluoroquinolone allergy with {drug}."
                warnings.append(allergy_warning)
                penalties += 80.0
            elif drug == "Trimethoprim/sulfamethoxazole" and has_sulfa_allergy:
                allergy_contraindication = True
                allergy_warning = "CONTRAINDICATION: Documented sulfonamide/sulfa drug allergy (risk of severe cutaneous reactions/SJS)."
                warnings.append(allergy_warning)
                penalties += 80.0
            elif drug == "Vancomycin" and has_glyco_allergy:
                allergy_contraindication = True
                allergy_warning = "CONTRAINDICATION: Documented glycopeptide allergy with Vancomycin."
                warnings.append(allergy_warning)
                penalties += 80.0
            elif drug in ("Gentamicin", "Tobramycin") and has_amino_allergy:
                allergy_contraindication = True
                allergy_warning = f"CONTRAINDICATION: Documented aminoglycoside allergy with {drug}."
                warnings.append(allergy_warning)
                penalties += 80.0

            # 3. Renal Safety Analysis (eGFR & Creatinine)
            if egfr < 30.0:
                if drug in ("Gentamicin", "Tobramycin"):
                    renal_warning = "High Nephrotoxicity & Ototoxicity Risk (eGFR < 30 mL/min): Avoid aminoglycosides unless no viable alternative exists. Strict trough TDM mandatory."
                    warnings.append(renal_warning)
                    penalties += 35.0
                elif drug == "Vancomycin":
                    renal_warning = "Nephrotoxicity & Accumulation Risk (eGFR < 30 mL/min): Vancomycin clearance is markedly impaired. Dose reduction and frequent AUC/trough monitoring required."
                    warnings.append(renal_warning)
                    penalties += 25.0
                elif drug == "Cefepime":
                    renal_warning = "Neurotoxicity Alert (eGFR < 30 mL/min): Risk of cefepime-induced encephalopathy/myoclonus in severe renal impairment. Mandatory dose reduction (1g q24h)."
                    warnings.append(renal_warning)
                    penalties += 20.0
                elif drug in ("Ciprofloxacin", "Levofloxacin"):
                    renal_warning = "Renal Dose Adjustment Required: 50% dose reduction recommended for eGFR < 30 mL/min."
                    warnings.append(renal_warning)
                    penalties += 15.0
                elif drug == "Piperacillin/tazobactam":
                    renal_warning = "Renal Adjustment Required: Adjust dose to 2.25g q6h-q8h. Avoid combining with Vancomycin due to synergistic AKI risk."
                    warnings.append(renal_warning)
                    penalties += 15.0
                elif drug == "Meropenem":
                    renal_warning = "Renal Adjustment Required: Adjust dose to 500mg q12h (risk of seizures if under-dialyzed/unadjusted)."
                    warnings.append(renal_warning)
                    penalties += 15.0
                elif drug == "Ceftriaxone":
                    renal_warning = "✓ Renal Sparing: Primarily biliary excretion. No renal dose adjustment required even in severe renal impairment."
                    bonuses += 12.0
            elif 30.0 <= egfr < 60.0:
                if drug in ("Gentamicin", "Tobramycin"):
                    renal_warning = "Moderate Renal Impairment (eGFR 30-59 mL/min): Extended-interval dosing and therapeutic trough monitoring advised."
                    warnings.append(renal_warning)
                    penalties += 15.0
                elif drug == "Vancomycin":
                    renal_warning = "Renal Caution (eGFR 30-59 mL/min): Monitor serum creatinine and maintain trough levels between 15-20 mcg/mL."
                    warnings.append(renal_warning)
                    penalties += 10.0
                elif drug == "Ceftriaxone":
                    renal_warning = "✓ Renal Sparing: Standard dosing appropriate without renal adjustment."
                    bonuses += 6.0

            # 4. Hepatic Safety Analysis (Bilirubin & AST/ALT)
            if (total_bilirubin and total_bilirubin > 2.5) or ast_alt_elevated:
                if drug == "Ceftriaxone":
                    hepatic_warning = "Hepatic/Biliary Caution: Ceftriaxone biliary excretion may worsen biliary sludging or jaundice in severe hepatic impairment."
                    warnings.append(hepatic_warning)
                    penalties += 15.0
                elif drug in ("Ciprofloxacin", "Levofloxacin"):
                    hepatic_warning = "Hepatic Alert: Fluoroquinolones may cause transaminitis or idiosyncratic hepatic injury. Monitor liver panel."
                    warnings.append(hepatic_warning)
                    penalties += 10.0
                elif drug == "Piperacillin/tazobactam":
                    hepatic_warning = "Hepatic Alert: Rare cholestatic hepatitis and jaundice reported; monitor LFTs."
                    warnings.append(hepatic_warning)
                    penalties += 10.0

            # 5. Drug-Drug Interactions
            has_warfarin = any("warfarin" in m or "coumadin" in m for m in medications)
            has_nsaid = any("nsaid" in m or "ibuprofen" in m or "ketorolac" in m or "naproxen" in m for m in medications)
            has_qt_drug = any("amiodarone" in m or "qt" in m or "sotalol" in m for m in medications)

            if has_warfarin:
                if drug in ("Ciprofloxacin", "Levofloxacin"):
                    interaction_warning = "Drug Interaction (Warfarin): Fluoroquinolones inhibit CYP1A2 and displace warfarin, causing significant INR elevation and bleeding risk."
                    warnings.append(interaction_warning)
                    penalties += 20.0
                elif drug == "Trimethoprim/sulfamethoxazole":
                    interaction_warning = "Severe Drug Interaction (Warfarin): TMP-SMX potently inhibits CYP2C9, triggering acute warfarin toxicity and dangerous hemorrhage risk."
                    warnings.append(interaction_warning)
                    penalties += 25.0

            if has_nsaid:
                if drug in ("Vancomycin", "Gentamicin", "Tobramycin"):
                    interaction_warning = "Synergistic Nephrotoxicity (NSAIDs): Concurrent NSAIDs with nephrotoxic antimicrobials markedly increases acute tubular necrosis and AKI risk."
                    warnings.append(interaction_warning)
                    penalties += 20.0
                elif drug in ("Ciprofloxacin", "Levofloxacin"):
                    interaction_warning = "Interaction (NSAIDs): Co-administration of quinolones with NSAIDs increases CNS excitation and seizure risk."
                    warnings.append(interaction_warning)
                    penalties += 15.0

            if has_qt_drug and drug in ("Ciprofloxacin", "Levofloxacin"):
                interaction_warning = "Cardiac Caution (QTc Prolongation): Combining fluoroquinolones with QT-prolonging drugs increases risk of Torsades de Pointes."
                warnings.append(interaction_warning)
                penalties += 20.0

            # 6. Prior Antibiotic Exposure Selection Penalty
            if any(drug_lower in p or p in drug_lower for p in prior_exposure):
                exp_warning = f"Prior Exposure Alert: Patient received {drug} in past 90 days. High risk of acquired resistant sub-clones."
                warnings.append(exp_warning)
                penalties += 20.0

            # 7. Resistance Profile Flags
            if "ESBL" in res_profile:
                if drug in ("Ceftriaxone", "Cefepime", "Ampicillin"):
                    esbl_warn = "ESBL Resistance Profile: Extended-spectrum beta-lactamase producer. Cephalosporins and penicillins are clinically ineffective."
                    warnings.append(esbl_warn)
                    penalties += 60.0
                elif drug == "Meropenem":
                    bonuses += 15.0
            if "MRSA" in res_profile:
                if drug in ("Ampicillin", "Ceftriaxone", "Cefepime", "Piperacillin/tazobactam"):
                    mrsa_warn = "MRSA Resistance Profile: Methicillin-resistant S. aureus is resistant to standard beta-lactams."
                    warnings.append(mrsa_warn)
                    penalties += 70.0
                elif drug == "Vancomycin":
                    bonuses += 15.0
            if "VRE" in res_profile and drug == "Vancomycin":
                warnings.append("VRE Profile: Vancomycin-resistant enterococci; Vancomycin is clinically ineffective.")
                penalties += 75.0
            if "CRE" in res_profile and drug == "Meropenem":
                warnings.append("CRE Profile: Carbapenem-resistant enterobacterales; Meropenem is clinically ineffective.")
                penalties += 75.0

            # 8. Compute Final Suitability Score (0-100)
            score = base_score + bonuses - penalties
            if allergy_contraindication:
                score = min(score, 12.0)
            suitability_score = round(max(0.0, min(100.0, score)), 1)

            tier = stewardship_map.get(drug, "Broad-Spectrum Reserve")

            evaluated_candidates.append({
                "antibiotic": drug,
                "suitability_score": suitability_score,
                "predicted_susceptibility": predicted_class,
                "resistance_probability": round(p_res, 4),
                "susceptibility_probability": round(p_sus, 4),
                "intermediate_probability": round(p_int, 4),
                "stewardship_tier": tier,
                "allergy_contraindication": allergy_contraindication,
                "allergy_warning": allergy_warning,
                "renal_warning": renal_warning,
                "hepatic_warning": hepatic_warning,
                "interaction_warning": interaction_warning,
                "warnings": warnings,
                "shap_values": shap_vals,
            })

        # Sort candidates descending by suitability_score, then ascending by resistance_probability
        evaluated_candidates.sort(key=lambda x: (-x["suitability_score"], x["resistance_probability"]))

        # Assign ranks 1..N
        for idx, item in enumerate(evaluated_candidates, start=1):
            item["rank"] = idx

        return {
            "identified_organism": organism,
            "infection_site": site,
            "patient_summary": {
                "age": age,
                "gender": gender,
                "egfr": egfr,
                "creatinine": creatinine,
                "total_bilirubin": total_bilirubin,
                "ast_alt_elevated": ast_alt_elevated,
                "allergies": allergies,
                "resistance_profile": res_profile,
                "prior_exposure": prior_exposure,
                "current_medications": medications,
            },
            "ranked_candidates": evaluated_candidates,
            "model_architecture": "LightGBM + Patient-specific feature analysis + TreeSHAP explainability",
            "data_source": "ARMD-ECUH retrospective research dataset (Dryad DOI: 10.5061/dryad.7sqv9s55x)",
            "disclaimer": (
                "Data source: ARMD-ECUH retrospective research dataset (Dryad DOI: 10.5061/dryad.7sqv9s55x). "
                "Predictions are research estimates and require clinical validation. "
                "They are not prescriptions or substitutes for antimicrobial stewardship and clinician judgment."
            ),
        }

    def models_available(self) -> Dict[str, bool]:
        return {
            "sepsis": self._sepsis is not None,
            "aki": self._aki is not None,
            "cardiovascular": self._cardio is not None,
            "liver": self._liver is not None,
            "neurological": self._neuro is not None,
            "respiratory": self._resp is not None,
            "antibiotic": (self._armd_susceptibility is not None or self._antibiotic is not None),
            "armd_susceptibility": self._armd_susceptibility is not None,
        }


@lru_cache(maxsize=1)
def get_ml_service() -> MLInferenceService:
    return MLInferenceService()
