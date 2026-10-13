"""Phase 7 — Streamlit dashboard (architecture Stages 11–12).

Launch from the project root:
    streamlit run app.py

Pages
  1. Predict a project    - form -> EBM prediction, risk level, recommendations
  2. Explainability       - SHAP global importance + explanation of one prediction
  3. Historical insights  - EDA charts and outcome mix by year, country, sector
  4. Model performance    - Phase 4 comparison table, charts, classification reports
  5. Policy decision support - portfolio risk mix by sector / region / instrument (Stage 13)

The app only LOADS saved artifacts and reports. It never trains a model.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.app import data
from src.decision_support import policy
from src.prediction.predictor import InputError, Predictor

st.set_page_config(page_title="Development Program Outcome Prediction", page_icon="📊", layout="wide")

CLASS_COLORS = {"Successful": "#2E7D5B", "Partially Successful": "#D89B2B", "Unsuccessful": "#C0453A"}
RISK_COLORS = {"Low": "#2E7D5B", "Medium": "#D89B2B", "High": "#C0453A"}
ESTIMATE_NOTE = ("Predictions are model estimates based on historical World Bank projects. "
                 "They are not guarantees of project success or failure, and they need expert review.")


# ----------------------------------------------------------------------------
# Loading (cached so it happens once per session, never retrains)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the saved model ...")
def get_predictor():
    return Predictor()


def load_predictor_or_stop():
    try:
        return get_predictor()
    except Exception as e:  # missing artifacts or a model-loading problem
        st.error(f"The saved model could not be loaded: {e}")
        st.info("Run the training steps in the README (Phases 1–5), then reload this page.")
        st.stop()


def show_image(name: str, caption: str | None = None):
    path = data.figure(name)
    if path:
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Chart `{name}` not found. Run the matching phase command from the README.")


# ----------------------------------------------------------------------------
# Page 1 — Project Lifecycle (Before Implementation & During Implementation)
# ----------------------------------------------------------------------------
def page_predict():
    st.title("Project lifecycle prediction & monitoring")
    st.info(ESTIMATE_NOTE)
    p = load_predictor_or_stop()

    from src.stage2 import lifecycle_store
    from stage2_experiments.interim_risk_assessment import assess_interim_risk

    # Step 1: Project Title, Description, and Project Selection
    st.subheader("1. Project identity")
    all_projects = lifecycle_store.load_all_projects()
    project_options = ["-- New Project --"] + [
        f"{pid}: {rec.get('title', 'Untitled')}" for pid, rec in all_projects.items()
    ]

    col_sel, col_lookup = st.columns([2, 1])
    
    # Check if a project was loaded via lookup ID or previous state
    current_pid = st.session_state.get("active_project_id")
    sel_idx = 0
    if current_pid:
        for i, opt in enumerate(project_options):
            if opt.startswith(f"{current_pid}:"):
                sel_idx = i
                break

    selected_option = col_sel.selectbox(
        "Load an existing project or create new",
        project_options,
        index=sel_idx,
        key="lifecycle_project_select",
    )

    lookup_id = col_lookup.text_input(
        "Load by exact Project ID",
        placeholder="e.g. PRJ-E22E84B3",
        key="lifecycle_lookup_id",
        help="Type or paste a Project ID to load its saved details.",
    ).strip().upper()

    # Determine which project to load (exact lookup ID takes precedence if entered)
    loaded_pid = None
    if lookup_id:
        if lookup_id in all_projects:
            loaded_pid = lookup_id
        else:
            st.error(f"Project ID `{lookup_id}` not found in saved records.")
    elif selected_option != "-- New Project --":
        loaded_pid = selected_option.split(":", 1)[0].strip()

    # If project changed or was just loaded, sync active_project_id and loaded data into session state
    if loaded_pid != st.session_state.get("_last_loaded_pid"):
        st.session_state["_last_loaded_pid"] = loaded_pid
        st.session_state["active_project_id"] = loaded_pid
        if loaded_pid and loaded_pid in all_projects:
            proj_data = all_projects[loaded_pid]
            st.session_state["in_proj_title"] = proj_data.get("title", "")
            st.session_state["in_proj_desc"] = proj_data.get("description", "")
            st.session_state["last_result"] = proj_data.get("stage1_prediction")
            s1_in = proj_data.get("stage1_inputs", {})
            if s1_in:
                if s1_in.get("Country Name") in p.levels["Country Name"]:
                    st.session_state["in_country"] = s1_in["Country Name"]
                if s1_in.get("Sector Board") in p.levels["Sector Board"]:
                    st.session_state["in_sector"] = s1_in["Sector Board"]
                if s1_in.get("Agreement Type") in p.levels["Agreement Type"]:
                    st.session_state["in_agreement"] = s1_in["Agreement Type"]
                if s1_in.get("Lending Instrument") in p.levels["Lending Instrument"]:
                    st.session_state["in_instrument"] = s1_in["Lending Instrument"]
        elif loaded_pid is None:
            st.session_state["in_proj_title"] = ""
            st.session_state["in_proj_desc"] = ""
            st.session_state.pop("last_result", None)
            st.session_state.pop("in_country", None)
            st.session_state.pop("in_sector", None)
            st.session_state.pop("in_agreement", None)
            st.session_state.pop("in_instrument", None)

    loaded_rec = all_projects.get(loaded_pid) if loaded_pid else None

    col_title, col_id = st.columns([2, 1])
    project_title = col_title.text_input("Project title", key="in_proj_title")
    project_id_display = loaded_pid or "Auto-generated on save"
    col_id.text_input("Project ID", value=project_id_display, disabled=True)
    project_desc = st.text_area("Project description", key="in_proj_desc", height=80)

    # Step 2: Lifecycle Stage Selection
    st.subheader("2. Project lifecycle stage")
    lifecycle_stage = st.radio(
        "Choose assessment stage",
        ["Before Implementation", "During Implementation"],
        horizontal=True,
        key="lifecycle_stage_radio",
    )

    # ------------------------------------------------------------------------
    # Branch A: Before Implementation (Stage 1 Approval-Time Prediction)
    # ------------------------------------------------------------------------
    if lifecycle_stage == "Before Implementation":
        st.markdown("#### Approval-time project characteristics (Stage 1)")
        saved_inputs = loaded_rec.get("stage1_inputs", {}) if loaded_rec else {}

        with st.form("project_form"):
            c1, c2 = st.columns(2)
            c_idx = p.levels["Country Name"].index(saved_inputs["Country Name"]) if (saved_inputs.get("Country Name") in p.levels["Country Name"] and "in_country" not in st.session_state) else None
            country = c1.selectbox("Country", p.levels["Country Name"], index=c_idx, placeholder="Select a country", key="in_country")
            
            s_idx = p.levels["Sector Board"].index(saved_inputs["Sector Board"]) if (saved_inputs.get("Sector Board") in p.levels["Sector Board"] and "in_sector" not in st.session_state) else None
            sector = c2.selectbox("Sector Board", p.levels["Sector Board"], index=s_idx, placeholder="Select a sector", key="in_sector")
            
            a_idx = p.levels["Agreement Type"].index(saved_inputs["Agreement Type"]) if (saved_inputs.get("Agreement Type") in p.levels["Agreement Type"] and "in_agreement" not in st.session_state) else None
            agreement = c1.selectbox("Agreement Type (financing source)", p.levels["Agreement Type"], index=a_idx, placeholder="e.g. IBRD or IDA", key="in_agreement")
            
            i_idx = p.levels["Lending Instrument"].index(saved_inputs["Lending Instrument"]) if (saved_inputs.get("Lending Instrument") in p.levels["Lending Instrument"] and "in_instrument" not in st.session_state) else None
            instrument = c2.selectbox("Lending Instrument", p.levels["Lending Instrument"], index=i_idx, placeholder="e.g. SIL", key="in_instrument")
            
            fy_val = int(saved_inputs.get("Approval FY", 2010)) if saved_inputs.get("Approval FY") else 2010
            year = c1.number_input("Approval fiscal year", min_value=1940, max_value=2025, value=fy_val, step=1,
                                   help=f"Training data covers FY{p.fy_range[0]}–FY{p.fy_range[1]}.")
            
            saved_cost = saved_inputs.get("Lending Project Cost")
            cost_init = float(saved_cost) / 1_000_000 if (saved_cost is not None and not pd.isna(saved_cost)) else 50.0
            cost_m = c2.number_input("Planned project cost (USD millions)", min_value=0.0, value=cost_init, step=5.0)
            no_cost = c2.checkbox("Planned cost is not available", value=(saved_cost is None and loaded_rec is not None))
            
            submitted = st.form_submit_button("Predict & Save Project", type="primary")

        if submitted:
            inputs = {"Country Name": country, "Sector Board": sector, "Agreement Type": agreement,
                      "Lending Instrument": instrument, "Approval FY": year,
                      "Lending Project Cost": None if no_cost else cost_m * 1_000_000}
            try:
                with st.spinner("Predicting and explaining ..."):
                    pred_res = p.predict(inputs)
                    st.session_state["last_result"] = pred_res

                    # Save project record with unique project ID
                    import datetime
                    curr_date = datetime.date.today().isoformat()
                    new_or_existing_id = lifecycle_store.create_or_update_project(
                        project_id=loaded_pid,
                        title=project_title or "Untitled Project",
                        description=project_desc or "",
                        stage1_inputs=inputs,
                        stage1_prediction=pred_res,
                        created_date=curr_date,
                    )
                    st.session_state["active_project_id"] = new_or_existing_id
                    st.success(f"Project saved under ID: **{new_or_existing_id}**")
            except InputError as e:
                st.session_state.pop("last_result", None)
                st.error("Please fix the following and try again:\n\n" + "\n".join(f"- {m}" for m in e.messages))
                return

        result = st.session_state.get("last_result")
        if not result and loaded_rec and loaded_rec.get("stage1_prediction"):
            result = loaded_rec["stage1_prediction"]
            st.session_state["last_result"] = result

        if result:
            show_result(result)

    # ------------------------------------------------------------------------
    # Branch B: During Implementation (Stage 2 Monitoring & Heuristic Assessment)
    # ------------------------------------------------------------------------
    else:
        st.markdown("#### Implementation-stage monitoring & risk assessment (Stage 2 Prototype)")
        st.warning("**Prototype Notice:** Stage 2 risk rules are transparent, rule-based heuristics for operational "
                   "decision support. They are **NOT** statistically validated predictions or machine learning models.")

        if not loaded_rec:
            st.info("💡 You can select an existing project above to compare against its approval-time baseline, "
                    "or monitor this project directly.")

        # Baseline Status Summary
        st.markdown("##### 1. Approval-time baseline status")
        base_pred = loaded_rec.get("stage1_prediction") if loaded_rec else None
        if base_pred:
            b1, b2, b3 = st.columns(3)
            b1.metric("Baseline Predicted Outcome", base_pred.get("predicted_outcome", "Unavailable"))
            b1_risk = base_pred.get("risk_level", "Unavailable")
            b2.markdown(f"**Baseline Risk Level**<br><span style='font-size:1.4rem;font-weight:600;color:{RISK_COLORS.get(b1_risk, '#555')}'>"
                        f"{b1_risk}</span>", unsafe_allow_html=True)
            b3.metric("Baseline Risk Score", f"{base_pred.get('risk_score', 0.0):.2f}")
        else:
            st.info("Baseline approval prediction: **Unavailable** (No prior Stage 1 estimate recorded for this project).")

        # Six Actual Implementation Indicators Input Form
        st.markdown("##### 2. Enter actual implementation indicators")
        with st.form("monitoring_form"):
            col_m1, col_m2 = st.columns(2)
            budget_util = col_m1.number_input(
                "Budget utilization (%)", min_value=0.0, max_value=200.0, value=25.0, step=1.0,
                help="Cumulative actual expenditure vs. planned allocation."
            )
            ben_part = col_m2.number_input(
                "Beneficiary participation (%)", min_value=0.0, max_value=100.0, value=60.0, step=5.0,
                help="Active participation or feedback engagement rate."
            )
            proj_comp = col_m1.number_input(
                "Project completion (%)", min_value=0.0, max_value=100.0, value=30.0, step=5.0,
                help="Reported physical work or outputs delivered."
            )
            delay_months = col_m2.number_input(
                "Implementation delay (months)", min_value=0.0, max_value=120.0, value=0.0, step=1.0,
                help="Schedule extension or delay beyond original closing date."
            )
            res_avail = col_m1.selectbox(
                "Resource availability",
                ["High / Adequate", "Moderate / Constrained", "Low / Severe Shortage"],
                index=0,
                help="Adequacy of staffing, counterpart funding, and logistical inputs."
            )
            ben_cov = col_m2.number_input(
                "Beneficiary coverage (%)", min_value=0.0, max_value=100.0, value=40.0, step=5.0,
                help="Proportion of target population or region reached."
            )

            import datetime
            obs_date = st.date_input("Monitoring observation date", value=datetime.date.today(), key="in_obs_date")
            mon_notes = st.text_input("Monitoring notes / supervision context", placeholder="e.g. Mid-term review findings", key="in_mon_notes")
            mon_submitted = st.form_submit_button("Assess Implementation Risk & Save Record", type="primary")

        if mon_submitted:
            # Map indicators to rule-based prototype assessment (reusing assess_interim_risk)
            # Budget utilization aligns directly with disbursement percentage
            # Resource availability informs task-team distress flags
            pdo_proxy = "Moderately Unsatisfactory" if res_avail == "Low / Severe Shortage" else (
                "Moderately Satisfactory" if res_avail == "Moderate / Constrained" else "Satisfactory"
            )
            ip_proxy = "Moderately Unsatisfactory" if (proj_comp < 15.0 and delay_months > 6.0) else (
                "Moderately Satisfactory" if proj_comp < 40.0 else "Satisfactory"
            )

            base_risk_level = base_pred.get("risk_level", "Medium") if base_pred else "Medium"
            interim_assessment = assess_interim_risk(
                pdo_rating=pdo_proxy,
                ip_rating=ip_proxy,
                disbursement_pct=float(budget_util),
                delay_months=float(delay_months),
                stage1_risk=base_risk_level,
            )

            indicators_payload = {
                "budget_utilization_pct": float(budget_util),
                "beneficiary_participation_pct": float(ben_part),
                "project_completion_pct": float(proj_comp),
                "implementation_delay_months": float(delay_months),
                "resource_availability": res_avail,
                "beneficiary_coverage_pct": float(ben_cov),
            }

            # Ensure we have a saved project to attach monitoring records to
            active_pid = loaded_pid or st.session_state.get("active_project_id")
            if not active_pid:
                active_pid = lifecycle_store.create_or_update_project(
                    project_id=None,
                    title=project_title or "Untitled Monitored Project",
                    description=project_desc or "",
                    created_date=str(obs_date),
                )
                st.session_state["active_project_id"] = active_pid

            lifecycle_store.add_monitoring_record(
                project_id=active_pid,
                record_date=str(obs_date),
                indicators=indicators_payload,
                interim_assessment=interim_assessment,
                notes=mon_notes,
            )
            st.session_state["last_monitoring_result"] = {
                "indicators": indicators_payload,
                "assessment": interim_assessment,
                "record_date": str(obs_date),
                "project_id": active_pid,
            }
            st.success(f"Monitoring record saved for Project **{active_pid}** on {obs_date}!")

        # Display latest monitoring assessment
        last_mon = st.session_state.get("last_monitoring_result")
        if last_mon:
            st.divider()
            st.subheader("Implementation Risk Stance (Prototype)")
            assess = last_mon["assessment"]
            m_col1, m_col2 = st.columns([1, 2])
            
            stance = assess.get("interim_risk_stance", "Standard Supervision")
            stance_colors = {
                "High Implementation Alert": "#C0453A",
                "Moderate Implementation Attention": "#D89B2B",
                "Standard Supervision": "#2E7D5B",
            }
            color = stance_colors.get(stance, "#555")
            m_col1.markdown(f"**Illustrative Risk Stance**<br><span style='font-size:1.6rem;font-weight:600;color:{color}'>"
                            f"{stance}</span>", unsafe_allow_html=True)
            m_col2.markdown(f"**Supervision Signals:** {assess.get('interim_signals_summary', 'None')}")
            st.caption("Note: This heuristic assessment evaluates documented operational thresholds. "
                       "It does not generate machine-learning probability scores.")

        # Baseline vs. Actual Comparison Table & History
        if loaded_rec or last_mon:
            st.divider()
            st.subheader("3. Baseline vs. Actual Monitoring History")
            target_pid = loaded_pid or (last_mon.get("project_id") if last_mon else None)
            proj_data = lifecycle_store.get_project(target_pid) if target_pid else None
            
            if proj_data and proj_data.get("monitoring_records"):
                recs = proj_data["monitoring_records"]
                comp_rows = []
                for r in recs:
                    inds = r["indicators"]
                    assm = r.get("interim_assessment", {})
                    base_est = proj_data.get("stage1_prediction", {})
                    comp_rows.append({
                        "Date": r["record_date"],
                        "Budget Util (%)": inds.get("budget_utilization_pct"),
                        "Beneficiary Part (%)": inds.get("beneficiary_participation_pct"),
                        "Completion (%)": inds.get("project_completion_pct"),
                        "Delay (mo)": inds.get("implementation_delay_months"),
                        "Resource Avail": inds.get("resource_availability"),
                        "Beneficiary Cov (%)": inds.get("beneficiary_coverage_pct"),
                        "Stage 2 Stance": assm.get("interim_risk_stance", "N/A"),
                        "Stage 1 Baseline": base_est.get("predicted_outcome", "Unavailable") if base_est else "Unavailable",
                    })
                comp_df = pd.DataFrame(comp_rows)
                st.dataframe(comp_df, hide_index=True, use_container_width=True)
            else:
                st.info("No monitoring history recorded for this project yet.")


def show_result(r: dict):
    st.divider()
    st.subheader("Approval-time result (Stage 1)")
    for w in r.get("warnings", []):
        st.warning(w)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted outcome", r["predicted_outcome"])
    c2.markdown(f"**Risk level**<br><span style='font-size:1.8rem;font-weight:600;color:{RISK_COLORS[r['risk_level']]}'>"
                f"{r['risk_level']}</span>", unsafe_allow_html=True)
    c3.metric("Risk score (for ranking only)", f"{r['risk_score']:.2f}")
    st.caption(f"Why this risk level: {r['reason']} Risk rules are provisional and project-defined; "
               "scores are model scores, not calibrated real-world probabilities.")

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("**Model scores per outcome**")
        fig, ax = plt.subplots(figsize=(5, 2.2))
        names = list(r["scores"])
        ax.barh(names, list(r["scores"].values()), color=[CLASS_COLORS[n] for n in names])
        for i, v in enumerate(r["scores"].values()):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
        ax.set_xlim(0, 1.1)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        rec = r["recommendations"]
        st.markdown(f"**Recommended actions ({r['risk_level']} risk)**")
        st.markdown("\n".join(f"- {a}" for a in rec["actions"]))
        if rec["feature_notes"]:
            st.markdown("**Points to review (features that raised this project's risk score)**")
            st.markdown("\n".join(f"- {n['note']}" for n in rec["feature_notes"]))
        st.caption(rec["disclaimer"])

    summary = pd.DataFrame([{**{k: v for k, v in r["inputs"].items()},
                             **{f"score_{k}": v for k, v in r["scores"].items()},
                             "predicted_outcome": r["predicted_outcome"], "risk_level": r["risk_level"],
                             "risk_score": r["risk_score"], "model": r["model"]}])
    st.download_button("Download this result (CSV)", summary.to_csv(index=False), "prediction_result.csv", "text/csv")
    st.caption("See the **Explainability** page for a feature-by-feature explanation of this prediction.")



# ----------------------------------------------------------------------------
# Page 2 — Explainability
# ----------------------------------------------------------------------------
def page_explain():
    st.title("Explainability (SHAP)")
    st.caption("SHAP shows which features the model relies on. It does not show what causes project outcomes.")
    report = data.read_text("shap_explanation.md")

    st.header("1. Which features matter most overall")
    show_image("shap_01_global_importance.png")
    imp = data.shap_importance()
    if imp is not None:
        st.dataframe(imp.round(3), use_container_width=True)
    how = data.section(report, "1. Global importance (`shap_01_global_importance.png`)")
    if how and "**How to read it:**" in how:
        st.markdown("**How to read it:**" + how.split("**How to read it:**", 1)[1])

    st.header("2. Why did the model give this result?")
    r = st.session_state.get("last_result")
    if r and r.get("risk_contributions"):
        st.markdown(f"Your last prediction: **{r['inputs']['Country Name']}**, {r['inputs']['Sector Board']}, "
                    f"FY{r['inputs']['Approval FY']} → **{r['risk_level']} risk** (risk score {r['risk_score']:.2f}).")
        contrib = pd.Series(r["risk_contributions"]).sort_values()
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.barh(contrib.index, contrib.values, color=["#C0453A" if v > 0 else "#3B6EA8" for v in contrib.values])
        ax.axvline(0, color="#555", lw=0.8)
        ax.set_xlabel("Change in risk score  (red = raised it, blue = lowered it)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Make a prediction on the **Predict a project** page to see its explanation here. "
                "Below is a saved example from the test set.")
    st.markdown("**Saved example (one test project)**")
    show_image("shap_02_local_example.png",
               "Start at the average score E[f(X)] at the bottom; each bar pushes P(Unsuccessful) up (red) or down (blue).")

    cautions = data.section(report, "3. Cautions")
    if cautions:
        with st.expander("Cautions when reading these explanations"):
            st.markdown(cautions)


# ----------------------------------------------------------------------------
# Page 3 — Historical insights (training data only, kept apart from predictions)
# ----------------------------------------------------------------------------
def page_history():
    st.title("Historical insights")
    df = data.training_data()
    n_hist = f"{len(df):,} projects" if df is not None else "training data"
    st.info(f"These charts describe past World Bank projects in the training data ({n_hist}). "
            "They are historical patterns, not predictions for any individual project.")

    st.header("Outcome distribution")
    show_image("eda_01_class_distribution.png")

    st.header("By approval year")
    show_image("eda_02_outcome_by_decade.png")
    fy = data.read_text("fy1990_check.md")
    if fy:
        with st.expander("Note on the rating scale before FY1990"):
            st.markdown(fy.split("\n", 1)[1])

    st.header("By country")
    if df is not None:
        mix = data.outcome_mix(df, "Country Name")
        country = st.selectbox("Choose a country", mix.index, index=None, placeholder="Type to search")
        if country:
            row = mix.loc[country]
            cols = st.columns(4)
            cols[0].metric("Projects", int(row["Projects"]))
            for col, cls in zip(cols[1:], data.CLASS_ORDER):
                col.metric(cls, f"{row[cls]:.0%}")
        st.markdown("**Countries with the most projects** (share of each outcome)")
        st.dataframe(mix.head(15).style.format({c: "{:.0%}" for c in data.CLASS_ORDER}), use_container_width=True)
    else:
        st.warning("Training data not found. Run `python -m src.preprocessing.prepare`.")

    st.header("By sector")
    show_image("eda_04_outcome_by_sector.png")

    with st.expander("More breakdowns: region, instrument type, project cost"):
        show_image("eda_03_outcome_by_region.png")
        show_image("eda_05_outcome_by_instrument.png")
        show_image("eda_06_cost_by_outcome.png")


# ----------------------------------------------------------------------------
# Page 4 — Model performance (Phase 4 results, read from disk)
# ----------------------------------------------------------------------------
def page_performance():
    st.title("Model performance")
    meta, comp = data.model_metadata(), data.model_comparison()
    if meta is None or comp is None:
        st.error("Evaluation results not found. Run `python -m src.models.train evaluate` (Phase 4).")
        return

    st.success(f"Selected model: **{meta['selected_model']}**, chosen by 5-fold cross-validated macro-F1 on the "
               "training set before the test set was used.")
    blocked = [m for m, s in meta["model_status"].items() if s != "TRAINED"]
    if blocked:
        reason = (" TabPFN's pre-trained weights could not be obtained in the development environment, "
                  "so no TabPFN results exist." if "TabPFN" in blocked else "")
        st.warning(f"Not available: {', '.join(blocked)}.{reason}")

    n_test = data.held_out_size()
    st.header(f"Comparison table (test set: {n_test:,} projects)" if n_test else "Comparison table (test set)")
    show = comp[["model", "cv_macro_f1", "test_macro_f1", "test_accuracy", "test_macro_precision",
                 "test_macro_recall", "test_weighted_f1", "train_rows_used", "selected"]].rename(columns={
        "model": "Model", "cv_macro_f1": "CV macro-F1 (train)", "test_macro_f1": "Test macro-F1",
        "test_accuracy": "Accuracy", "test_macro_precision": "Macro precision", "test_macro_recall": "Macro recall",
        "test_weighted_f1": "Weighted F1", "train_rows_used": "Training rows", "selected": "Selected"})
    st.dataframe(show, hide_index=True, use_container_width=True)
    notes = data.performance_notes(comp, meta)   # built from the saved results, not hard-coded
    if notes:
        st.caption(" ".join(notes))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Macro-F1: CV vs test")
        show_image("eval_model_comparison.png")
    with c2:
        st.subheader("Confusion matrices (test set)")
        show_image("eval_confusion_matrices.png")

    st.header("Classification reports (test set)")
    md = data.read_text("model_comparison.md")
    for name in comp["model"]:
        text = data.section(md, name)
        if text:
            with st.expander(name, expanded=name == meta["selected_model"]):
                st.markdown(text)


# ----------------------------------------------------------------------------
# Page 5 — Policy decision support (Stage 13, portfolio level)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading the historical portfolio ...")
def get_portfolio():
    return policy.historical_portfolio()


def risk_mix_chart(profile: pd.DataFrame, top: int = 12):
    prof = profile[profile["Projects"] >= policy.MIN_PROJECTS].head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(prof) + 1.2))
    left = pd.Series(0.0, index=prof.index)
    for lvl in ["Low", "Medium", "High"]:
        ax.barh(prof.index.astype(str), prof[lvl], left=left, color=RISK_COLORS[lvl], label=lvl)
        left += prof[lvl]
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of projects")
    ax.legend(ncol=3, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_considerations(cons: dict):
    for title, key in [("Supervision allocation", "supervision_allocation"),
                       ("Risk mitigation", "risk_mitigation"), ("Planning cautions", "planning_cautions")]:
        st.markdown(f"**{title}**")
        st.markdown("\n".join(f"- {t}" for t in cons[key]) or "- No segment stands out.")


def page_policy():
    st.title("Policy decision support")
    st.caption("Portfolio-level view for planning and resource allocation. For a single project, use "
               "**Predict a project**.")
    try:
        pf = get_portfolio()
    except Exception as e:
        st.error(f"The historical portfolio could not be loaded: {e}")
        return
    st.info(f"Historical portfolio: {len(pf):,} training projects. Each was scored by the EBM while it was held "
            "out during cross-validation (verified out-of-fold predictions); the test set is not used. "
            "Risk levels use the existing documented risk rules. Patterns are associations, not causes.")
    overall = policy.overall_mix(pf)
    cols = st.columns(3)
    for col, lvl in zip(cols, ["Low", "Medium", "High"]):
        col.metric(f"{lvl} risk", f"{overall[lvl]:.0%}")

    for tab, (label, column) in zip(st.tabs(list(policy.DIMENSIONS)), policy.DIMENSIONS.items()):
        with tab:
            prof = policy.risk_profile(pf, column)
            st.markdown(f"**Risk mix by {label.lower()}** (segments with at least {policy.MIN_PROJECTS} projects)")
            risk_mix_chart(prof)
            st.dataframe(prof.style.format({c: "{:.0%}" for c in ["Low", "Medium", "High", "Actually Unsuccessful"]}),
                         use_container_width=True)
            show_considerations(policy.policy_considerations(prof, overall, label))
    with st.expander("General cautions"):
        st.markdown("\n".join(f"- {c}" for c in policy.GENERAL_CAUTIONS))

    st.header("Score a portfolio of proposed projects (optional)")
    p = load_predictor_or_stop()
    st.markdown(f"Upload a CSV with columns: **{', '.join(policy.required_columns(p))}**, optional "
                "**Lending Project Cost** (USD) and **Region**. Each row is checked and scored with the same "
                "model and risk rules as the prediction page.")
    st.download_button("Download a template CSV", policy.template_csv(p), "portfolio_template.csv", "text/csv")
    upload = st.file_uploader("Proposed projects (CSV)", type="csv")
    if upload is None:
        return
    try:
        df = pd.read_csv(upload)
        with st.spinner(f"Scoring {len(df)} project(s) ..."):
            scored, rejected = policy.score_portfolio(df, p)
    except (policy.PortfolioError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
        st.error(f"This file could not be used: {e}")
        return
    if not rejected.empty:
        st.warning(f"{len(rejected)} row(s) were not scored because of invalid inputs:")
        st.dataframe(rejected, hide_index=True, use_container_width=True)
    if scored.empty:
        return
    counts = scored["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    cols = st.columns(3)
    for col, lvl in zip(cols, counts.index):
        col.metric(f"{lvl} risk projects", int(counts[lvl]))
    st.dataframe(scored, hide_index=True, use_container_width=True)
    for label, column in [("sector", "Sector Board"), ("lending instrument", "Lending Instrument")]:
        st.markdown(f"**Uploaded portfolio by {label}**")
        st.dataframe(policy.risk_profile(scored, column).style.format({c: "{:.0%}" for c in ["Low", "Medium", "High"]}),
                     use_container_width=True)
    st.download_button("Download scored portfolio (CSV)", scored.to_csv(index=False), "scored_portfolio.csv", "text/csv")
    st.caption("Scores are model estimates for planning discussions, not guarantees; review each project individually.")


# ----------------------------------------------------------------------------
# Page 6 — Implementation-stage risk (Stage 2 Prototype)
# ----------------------------------------------------------------------------
def page_stage2_prototype():
    st.title("Implementation-stage risk assessment (Stage 2 Prototype)")
    st.info("**Prototype Notice:** This view demonstrates how genuine, dated World Bank Implementation Status "
            "and Results Reports (ISRs) can update project risk during execution. Stage 2 risk rules are illustrative "
            "decision-support heuristics—**not statistically validated failure predictions**.")

    from src.utils.config import resolve
    demo_csv = resolve("stage2_experiments/stage2_interim_assessment_demo.csv")
    if not demo_csv.exists():
        st.warning("Stage 2 demonstration data not found. Run `python -m stage2_experiments.interim_risk_assessment`.")
        return

    df = pd.read_csv(demo_csv)

    st.subheader("1. Detailed project inspection (14 pilot operations)")
    selected_pid = st.selectbox("Choose a pilot project", df["project_id"].tolist(), key="s2_pid")
    row = df[df["project_id"] == selected_pid].iloc[0]

    c1, c2, c3 = st.columns(3)
    seq_str = f" (Seq {int(row['sequence'])})" if pd.notna(row["sequence"]) else ""
    c1.metric("Interim Snapshot Date", f"{row['snapshot_date']}{seq_str}")
    c2.metric("Stage 1 Approval-Time Risk", str(row["stage1_baseline_risk"]))

    stance_colors = {
        "High Implementation Alert": "#C0453A",
        "Moderate Implementation Attention": "#D89B2B",
        "Standard Supervision": "#2E7D5B",
    }
    color = stance_colors.get(str(row["interim_risk_stance"]), "#555")
    c3.markdown(f"**Stage 2 Interim Stance**<br><span style='font-size:1.4rem;font-weight:600;color:{color}'>"
                f"{row['interim_risk_stance']}</span>", unsafe_allow_html=True)

    st.markdown(f"**Interim Signals Summary:** {row['interim_signals']}")

    st.markdown("#### Documented supervision signals from cached ISR")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("PDO Rating (Current)", str(row["pdo_rating"]) if pd.notna(row["pdo_rating"]) else "Not stated")
    col_b.metric("IP Rating (Current)", str(row["ip_rating"]) if pd.notna(row["ip_rating"]) else "Not stated")
    col_c.metric("Disbursement %", f"{row['disbursement_pct']:.1f}%" if pd.notna(row["disbursement_pct"]) else "Not stated")
    col_d.metric("Schedule Delay", f"{row['schedule_delay_months']:.0f} mo" if pd.notna(row["schedule_delay_months"]) else "Not stated")

    st.divider()
    st.subheader("2. Full pilot portfolio view (14 operations)")
    st.caption("Signals extracted from official World Bank plain-text ISR documents. Missing ratings or dates are shown explicitly as missing.")

    disp_df = df[["project_id", "snapshot_date", "pdo_rating", "ip_rating", "disbursement_pct",
                  "schedule_delay_months", "stage1_baseline_risk", "interim_risk_stance"]].rename(columns={
        "project_id": "Project ID",
        "snapshot_date": "Snapshot Date",
        "pdo_rating": "PDO Rating",
        "ip_rating": "IP Rating",
        "disbursement_pct": "Disb %",
        "schedule_delay_months": "Delay (mo)",
        "stage1_baseline_risk": "Stage 1 Risk",
        "interim_risk_stance": "Stage 2 Interim Stance",
    })
    st.dataframe(disp_df, hide_index=True, use_container_width=True)
    st.caption("Note: Final outcome labels are excluded from this operational view to prevent target leakage.")


# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------
PAGES = {
    "Predict a project": page_predict,
    "Explainability": page_explain,
    "Historical insights": page_history,
    "Model performance": page_performance,
    "Policy decision support": page_policy,
    "Implementation-stage risk (prototype)": page_stage2_prototype,
}

st.sidebar.title("Development Program Outcome Prediction")
choice = st.sidebar.radio("Go to", list(PAGES), key="page")
st.sidebar.caption("Model: Explainable Boosting Machine (EBM). Approval-time features only.")
PAGES[choice]()

