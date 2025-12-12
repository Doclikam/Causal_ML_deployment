import os
stage = st.selectbox('Stage', ['I','II','III','IV','Missing'], index=2)
t_cat = st.selectbox('T', ['T1','T2','T3','T4','Tx'], index=1)
treatment = st.selectbox('Planned strategy', options=[0,1], format_func=lambda x: 'RT alone' if x==0 else 'Chemo-RT')
submitted = st.form_submit_button('Estimate outcomes')


if submitted:
patient = dict(age=age, sex=sex, primary_site_group=primary_site_group,
pathology_group=pathology_group, hpv_clean=hpv_clean, stage=stage,
t=t_cat, ecog_ps=ecog_ps, smoking_status_clean=smoking_status_clean,
smoking_py_clean=smoking_py_clean, treatment=treatment)


# build canonical Xpatient in-app
Xpatient_app, build_debug = build_canonical_Xpatient(patient, outdir=OUTDIR, base_url=BASE_URL)


# call infer, pass Xpatient_override so infer uses same features
out = infer_new_patient_fixed(patient, return_raw=True, outdir=OUTDIR, base_url=BASE_URL,
max_period_override=int(max_period_months), Xpatient_override=Xpatient_app)
# attach app debug
out['debug'] = out.get('debug', {})
out['debug']['Xpatient_app'] = Xpatient_app
out['debug'].update(build_debug)


dev_show_inference_debug(out)


surv = out.get('survival_curve')
cates = out.get('CATEs', {})


if surv is None or getattr(surv,'empty',True):
st.warning('Survival curve not available')
else:
surv_df = surv.copy()
surv_df['months'] = surv_df['days']/30.0
fig = go.Figure()
fig.add_trace(go.Scatter(x=surv_df['months'], y=surv_df['S_control'], name='RT'))
fig.add_trace(go.Scatter(x=surv_df['months'], y=surv_df['S_treat'], name='Chemo-RT'))
fig.update_layout(xaxis_title='Months', yaxis_title='Probability alive & well')
st.plotly_chart(fig, use_container_width=True)


rmst = compute_rmst_from_survival(surv_df, rmst_horizon_months)
st.metric('Extra time with Chemo-RT (months)', f"{rmst.get('delta', np.nan):+.2f} m")


# show CATE bar if present
if cates:
rows = []
for h,v in cates.items():
try:
mh = float(h)
except Exception:
mh = None
rows.append({'horizon_months': mh, 'CATE': v.get('CATE'), 'error': v.get('error')})
df_c = pd.DataFrame(rows).dropna(subset=['horizon_months']).sort_values('horizon_months')
if not df_c.empty:
df_c['CATE_pct'] = df_c['CATE'] * 100
figc = go.Figure()
figc.add_trace(go.Bar(x=df_c['horizon_months'], y=df_c['CATE_pct']))
figc.update_layout(xaxis_title='Months', yaxis_title='Chemo-RT − RT (% points)')
st.plotly_chart(figc, use_container_width=True)


st.markdown('---')
st.caption('Developer: artifacts are taken from local OUTDIR first, then from BASE_URL if missing.')
