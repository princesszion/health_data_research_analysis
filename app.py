import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import country_converter as coco
import pycountry

# ──────────────────────────────────────────────────────────────────────────────
# Page & Theme Setup 
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Research Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Progress bar colour */
.stProgress > div > div > div > div { background-color: #1A5632; }

/* Tabs styling */
.stTabs [role="tab"] {
  font-size: 18px !important;
  padding: 12px 20px !important;
  min-width: 180px;
  white-space: nowrap;
  border-radius: 8px 8px 0 0;
  margin-right: 4px;
}
.stTabs [role="tab"]:nth-child(1) { background: #1A5632; }
.stTabs [role="tab"]:nth-child(2) { background: #9F2241; }
.stTabs [role="tab"]:nth-child(3) { background: #B4A269; }
.stTabs [role="tab"]:nth-child(4) { background: #348F41; }
.stTabs [role="tab"]:nth-child(5) { background: #58595B; }
.stTabs [role="tab"]:nth-child(6) { background: #9F2241; }
.stTabs [role="tab"]:nth-child(7) { background: #B4A269; }
.stTabs [role="tab"]:nth-child(8) { background: #1A5632; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(1) { background: #1A5632 !important; color:white !important; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(2) { background: #9F2241 !important; color:white !important; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(3) { background: #B4A269 !important; color:white !important; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(4) { background: #348F41 !important; color:white !important; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(5) { background: #58595B !important; color:white !important; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(6) { background: #9F2241 !important; color:white !important; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(7) { background: #B4A269 !important; color:white !important; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(8) { background: #1A5632 !important; color:white !important; }
</style>
""", unsafe_allow_html=True)

palette = [
    "#1A5632", "#9F2241", "#B4A269", "#348F41",
    "#58595B", "#9F2241", "#B4A269", "#1A5632"
]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: File Uploader & Deep-Dive Selector
# ──────────────────────────────────────────────────────────────────────────────
if 'processed' not in st.session_state:
    st.session_state.processed = False

st.sidebar.title("Upload & Configure")
en_file = st.sidebar.file_uploader("English dataset", type="csv")
fr_file = st.sidebar.file_uploader("French dataset", type="csv")
# process = st.sidebar.button("Process Data")
progress = st.sidebar.progress(0)
if st.sidebar.button("Process Data"):
    st.session_state.processed = True
# Utility to unify Yes/No variants
yes_no_map = {
    'oui': 'Yes','non': 'No','yes': 'Yes','no': 'No',
    'checked': 'Checked','coché':'Checked',
    'unchecked':'Unchecked','non coché':'Unchecked'
}
def harmonize_values(x):
    if isinstance(x, str):
        return yes_no_map.get(x.strip().lower(), x)
    return x

if st.session_state.processed:
    # ──────────────────────────────────────────────────────────────────────────
    # Load & harmonize English/French
    # ──────────────────────────────────────────────────────────────────────────
    progress.progress(10)
    df_en = pd.read_csv(en_file, keep_default_na=False) if en_file else pd.DataFrame()
    df_fr = pd.read_csv(fr_file, keep_default_na=False) if fr_file else pd.DataFrame()
    if df_fr.shape[1] and df_en.shape[1]:
        df_fr.rename(columns=dict(zip(df_fr.columns, df_en.columns)),
                     inplace=True)
    df_fr = df_fr.applymap(harmonize_values) if not df_fr.empty else df_fr

    # Align columns, concat, drop blank
    all_cols = list(dict.fromkeys(df_en.columns.tolist()
                                  + df_fr.columns.tolist()))
    df = pd.concat([
        df_en.reindex(columns=all_cols, fill_value=""),
        df_fr.reindex(columns=all_cols, fill_value="")
    ], ignore_index=True)
    df.drop(columns=[c for c in df if (df[c]=="").all()], inplace=True)
    progress.progress(50)

    # ──────────────────────────────────────────────────────────────────────────
    # Filter Countries & Final Yes/No unify
    # ──────────────────────────────────────────────────────────────────────────
    df['Country'] = (
        df['Country'].astype(str)
          .str.replace(r'Guinea[\s-]?Bissau',
                       'Guinea-Bissau', flags=re.I)
    )
    targets = ["Nigeria","Togo","Ghana","Guinea-Bissau","Gambia","Sierra Leone"]
    df = df[df['Country'].isin(targets)].copy()
    def unify(v):
        if isinstance(v,str):
            t=v.strip().lower()
            if t in ('oui','yes','checked','true','1'): return 'Yes'
            if t in ('non','no','unchecked','false','0'): return 'No'
        return v
    df = df.applymap(unify)
    progress.progress(70)

    # ──────────────────────────────────────────────────────────────────────────
    # Deep-Dive country selector & slice
    # ──────────────────────────────────────────────────────────────────────────
    countries = sorted(df['Country'].unique())
    selected_country = st.sidebar.selectbox(
        "Deep-Dive Country",
        ["(All Countries)"] + countries,
        index=0
    )
    if selected_country == "(All Countries)":
        df_deep = df.copy()
    else:
        df_deep = df[df['Country']==selected_country].copy()

    # ──────────────────────────────────────────────────────────────────────────
    # Hoist all shared maps & exploded/stakeholder tables *before* tabs
    # ──────────────────────────────────────────────────────────────────────────
    # 1. Identification categories
    cats = {
      "BasicScience":[r"\bbasic\b",r"fundamental"],
      "Preclinical":[r"preclinical"],
      "ClinicalTrials":[r"clinical"],
      "Epidemiological":[r"epidemiolog"]
    }
    # 2. Human resource boolean & numeric groups
    bool_groups = {
        "Clinical Staff":[r"availability of clinical staff"],
        "Lab Staff":[r"availability of laboratory staff"],
        "Pharmacy Staff":[r"availability of pharmacy staff"],
        "Bioinformatics":[r"bioinformatics"],
        "Cell Culture":[r"cell culture"],
        "Org. Synthesis":[r"organic synthesis"],
        "Virology":[r"virology"],
    }
    num_groups = {
        "Other Staff":[r"number of other staff"],
        "PhD":[r"doctorate|phd"],
        "MSc":[r"master's|msc"],
    }
    # 3. Stakeholder explosion
    free_cols = [
        'Other (Please specify)',
        'If yes, list the research collaborations in the last 5 years'
    ]
    records=[]
    for col in free_cols:
        if col in df.columns:
            exploded = (df[col].astype(str)
                         .str.split(r'[;,]+')
                         .explode()
                         .str.strip()
                       ).dropna()
            for idx, raw in exploded.items():
                if raw and raw.lower()!="nan":
                    site = str(df.at[idx, next(c for c in df if re.search(r'\bname\b',c,re.I))]).strip()
                    if site and site.lower()!="nan":
                        records.append({
                            'Country':df.at[idx,'Country'],
                            'Site':site,'RawEntry':raw
                        })
    site_stake_df = pd.DataFrame(records)
    def split_items(raw):
        tmp=re.sub(r'\d+\.',';',raw)
        tmp=re.sub(r'\*+',';',tmp)
        return [p.strip() for p in re.split(r'[;,\n]+',tmp) if p.strip()]
    site_clean = (site_stake_df
        .assign(Stakeholder=site_stake_df['RawEntry'].apply(split_items))
        .explode('Stakeholder')
    )
    # 4. Policy flags
    policy_exists_col       = "Is there a health research policy in your country?"
    policy_disseminated_col = "Has the policy been disseminated?"
    policy_implemented_col  = "Is the policy currently under implementation?"
    budget_col              = "What percentage of the national health budget is allocated to health-related R&D, considering the AU's 2% target?"
    sop_cols = [c for c in df.columns if c.startswith('Available SOPs')]
    def to_bin(x): return 1 if str(x).strip().lower() in ('yes','oui','checked') else 0
    df['PolicyExists']       = df[policy_exists_col].map(to_bin)
    df['PolicyDisseminated'] = df[policy_disseminated_col].map(to_bin)
    df['PolicyImplemented']  = df[policy_implemented_col].map(to_bin)
    df['Budget_pct'] = (
        pd.to_numeric(df[budget_col].astype(str).str.rstrip('%')
                      .replace('','0'),
                      errors='coerce').fillna(0).clip(0,100)/100.0
    )
    df['SOP_Coverage'] = df[sop_cols].applymap(to_bin).sum(axis=1)/len(sop_cols)
    site_policy = pd.DataFrame({
        'Country':df['Country'],
        'Exists':df['PolicyExists'],
        'Disseminated':df['PolicyDisseminated']&df['PolicyExists'],
        'Implemented':df['PolicyImplemented']&df['PolicyExists'],
        'Budget':df['Budget_pct'],
        'SOP_Coverage':df['SOP_Coverage']
    })

    #MAPS 
    # Identification flags
    for cat, pats in cats.items():
        cols = [c for c in df.columns if any(re.search(p, c, re.I) for p in pats)]
        df[f"Is{cat}"] = df[cols].eq("Yes").any(axis=1)

    # Capability: per‐site sum then country mean
    df["CapabilityScore"] = df[[f"Is{cat}" for cat in cats]].sum(axis=1)
    cap = df.groupby("Country")["CapabilityScore"] \
            .mean().reset_index(name="Avg Capability")

    # Phase I sites
    trans_cols = [c for c in df.columns if re.search(r"phase.*i", c, re.I)]
    df["HasPhaseI"] = df[trans_cols].eq("Yes").any(axis=1)
    tr = df.groupby("Country")["HasPhaseI"] \
        .sum().reset_index(name="Phase I Sites")

    # Infrastructure index
    infra_terms = ["availability of advanced","level of biosecurity","iso certification"]
    infra_cols = [c for c in df.columns if any(t in c.lower() for t in infra_terms)]
    df["InfraIndex"] = df[infra_cols].eq("Yes").sum(axis=1)
    infra = df.groupby("Country")["InfraIndex"] \
            .mean().reset_index(name="Avg InfraIndex")

    # Ethics/Reg: percent of sites with IRB
    ethic_terms = ["ethic","irb","regul","guidelines"]
    ethic_cols = [c for c in df.columns if any(t in c.lower() for t in ethic_terms)]
    df["HasIRB"] = df[ethic_cols].eq("Yes").any(axis=1)
    er = df.groupby("Country")["HasIRB"] \
        .mean().reset_index(name="% With IRB")

    # Policy: you already have site_policy; use that
    pol = site_policy.groupby("Country")["Exists"] \
                    .mean().reset_index(name="% With Policy")

    # ─── 2. Merge into one wide DataFrame ────────────────────────────────────────
    map_df = (
        cap
        .merge(tr,   on="Country")
        .merge(infra, on="Country")
        .merge(er,   on="Country")
        .merge(pol,  on="Country")
    )
    
    map_df["Country"] = map_df["Country"].str.strip()
    cc = coco.CountryConverter()
 # this will try exact matches first, then a fuzzy lookup
    map_df["ISO_A3"] = cc.convert(
        names        = map_df["Country"],
        to           = "ISO3",
        not_found    = None
    )

        # 3) Fallback on any that failed
    def fuzzy_iso(name):
        try:
            # try exact pycountry lookup
            return pycountry.countries.get(name=name).alpha_3
        except:
            # last‐ditch fuzzy
            try:
                return pycountry.countries.search_fuzzy(name)[0].alpha_3
            except:
                return None

    mask = map_df["ISO_A3"].isnull()
    if mask.any():
        map_df.loc[mask, "ISO_A3"] = map_df.loc[mask, "Country"].apply(fuzzy_iso)

        still_missing = map_df.loc[map_df["ISO_A3"].isnull(), "Country"].unique()
        if len(still_missing):
            st.warning("Couldn't map these to ISO3: " + ", ".join(still_missing))

    # ─── 3. Melt to long form for faceting ────────────────────────────────────────
    missing = map_df[map_df["ISO_A3"].isnull()]
    if not missing.empty:
        st.warning(
        " I couldn’t map these countries to ISO-3:\n" +
        ", ".join(missing["Country"].unique())
        )

    map_long = map_df.melt(
    id_vars=["Country","ISO_A3"],
    var_name="Metric",
    value_name="Value"
)


        # ──────────────────────────────────────────────────────────────────────────
        # Initialize Tabs
        # ──────────────────────────────────────────────────────────────────────────
    progress.progress(80)
    tabs = st.tabs([
        "1. Identification","2. Capacity","3. Human Resources",
        "4. Translational","5. Infrastructure","6. Ethics/Reg",
        "7. Stakeholders","8. Policy","9. Deep-Dive","10. Maps"
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 1: Identification
    # ──────────────────────────────────────────────────────────────────────────
    # Tab 1: Identification
    with tabs[0]:
        df_current = df if selected_country=="(All Countries)" else df_deep
        st.header("1. Identification of Research Sites")

        # 1) Build the per-site boolean flags and collect their names
        bool_cols = []
        for cat, pats in cats.items():
            cols = [
                c for c in df_current.columns
                if any(re.search(p, c, re.IGNORECASE) for p in pats)
            ]
            # create Is<cat> column
            df_current[f"Is{cat}"] = df_current[cols].eq("Yes").any(axis=1)
            bool_cols.append(f"Is{cat}")

        # 2) Aggregate those flags per country
        summary1 = (
            df_current
            .groupby("Country")[bool_cols]
            .sum()
            .rename(columns=lambda x: x.replace("Is", ""))
        )

        # 3) Compute “Other” = sites where none of the flags were True
        other_mask   = ~df_current[bool_cols].any(axis=1)
        other_counts = df_current[other_mask].groupby("Country").size()
        summary1["Other"] = other_counts.reindex(summary1.index, fill_value=0)

        # 4) Display & plot
        st.table(summary1)

        melt1 = summary1.reset_index().melt(
            "Country", var_name="Category", value_name="Count"
        )
        fig1 = px.bar(
            melt1,
            x="Country", y="Count", color="Category",
            barmode="group", title="Sites by Category & Country",
            color_discrete_sequence=palette
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig1b = px.sunburst(
            melt1,
            path=["Country","Category"], values="Count",
            title="Sunburst of Research Sites",
            color_discrete_sequence=palette
        )
        st.plotly_chart(fig1b, use_container_width=True)



    # ──────────────────────────────────────────────────────────────────────────
    # Tab 2: Capacity
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        df_current = df if selected_country=="(All Countries)" else df_deep
        st.header("2. Capacity Evaluation")
        # reuse bool_cols from Tab1
        cap = df_current[bool_cols].sum(axis=1).groupby(df_current['Country'])\
              .mean().round(2).reset_index(name='CapabilityScore')
        fig2 = px.bar(cap, x='Country', y='CapabilityScore',
                      color='Country', title="Avg Capability Score",
                      range_y=[0,len(bool_cols)],
                      color_discrete_sequence=palette)
        st.plotly_chart(fig2, use_container_width=True)
        # heatmap
        pivot = (melt1[melt1['Category'].isin(
                    [c.replace('Is','') for c in bool_cols])]
                 .pivot(index='Country',columns='Category',values='Count')
                 .fillna(0)
        )
        fig2b = px.imshow(
            pivot, labels=dict(x="Category",y="Country",color="Count"),
            title="Site Counts Heatmap",
            color_continuous_scale=["#D0E8D8","#1A5632"],
            zmin=0, zmax=pivot.values.max()
        )
        fig2b.update_layout(height=500, margin=dict(t=50,b=50))
        st.plotly_chart(fig2b, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 3: Human Resources
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        df_current = df if selected_country=="(All Countries)" else df_deep
        st.header("3. Human Resource Assessment")
        # boolean groups
        for name,pats in bool_groups.items():
            cols=[c for c in df_current if any(re.search(p,c,re.I) for p in pats)]
            df_current[name] = df_current[cols].eq("Yes").any(axis=1).astype(int)
        bool_sum = df_current.groupby('Country')[list(bool_groups.keys())].sum()
        # numeric groups
        for name,pats in num_groups.items():
            cols=[c for c in df_current if any(re.search(p,c,re.I) for p in pats)]
            if cols:
                df_current[name] = df_current[cols]\
                                  .apply(pd.to_numeric,errors='coerce')\
                                  .max(axis=1).fillna(0).astype(int)
            else:
                df_current[name]=0
        num_sum = df_current.groupby('Country')[list(num_groups.keys())].sum()
        total = (bool_sum.add(num_sum,fill_value=0).sum(axis=1).astype(int))
        num_sum["Total Staff"]=total
        combined = pd.concat([bool_sum,num_sum],axis=1)
        st.table(combined)
        # charts
        melt_bool = bool_sum.reset_index().melt('Country',
                          var_name='Indicator',value_name='Count of Yes')
        fig_bool=px.bar(melt_bool,x='Country',y='Count of Yes',
                        color='Indicator',barmode='group',
                        color_discrete_sequence=palette,
                        title="Yes Responses by Indicator")
        st.plotly_chart(fig_bool,use_container_width=True)
        melt_num = num_sum.reset_index().melt('Country',
                          var_name='Staff Category',value_name='Count')
        fig_num=px.bar(melt_num,x='Country',y='Count',
                       color='Staff Category',barmode='group',
                       color_discrete_sequence=palette,
                       title="Staff Counts by Country")
        st.plotly_chart(fig_num,use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 4: Translational
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        df_current = df if selected_country=="(All Countries)" else df_deep
        st.header("4. Translational Research (Phase I)")
        trans = [c for c in df_current if re.search(r"phase.*i",c,re.I)]
        df_current['HasPhaseI'] = df_current[trans].eq('Yes').any(axis=1)
        tr = df_current.groupby('Country')['HasPhaseI']\
             .sum().reset_index()
        fig4 = px.bar(tr, x='Country', y='HasPhaseI',
                      color='Country',
                      range_y=[0,tr['HasPhaseI'].max()+1],
                      color_discrete_sequence=palette,
                      title="Phase I Trial Sites")
        st.plotly_chart(fig4,use_container_width=True)
        cap_df = cap.merge(tr,on='Country')
        fig4b = px.scatter(cap_df,
                           x='CapabilityScore',y='HasPhaseI',
                           size='HasPhaseI',color='Country',
                           color_discrete_sequence=palette,
                           title="Capability vs Phase I")
        st.plotly_chart(fig4b,use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 5: Infrastructure
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        df_current = df if selected_country=="(All Countries)" else df_deep
        st.header("5. Infrastructure Analysis")
        infra_cols=[c for c in df_current if any(
                     t in c.lower() for t in
                     ['availability of advanced','level of biosecurity','iso certification']
        )]
        df_current['InfraIndex']=df_current[infra_cols].eq('Yes').sum(axis=1)
        infra = df_current.groupby('Country')['InfraIndex']\
                .mean().round(1).reset_index()
        fig5=px.bar(infra,x='Country',y='InfraIndex',
                    color='Country',range_y=[0,len(infra_cols)],
                    color_discrete_sequence=palette,
                    title="Avg Infrastructure Index")
        st.plotly_chart(fig5,use_container_width=True)
        fig5b=px.violin(df_current,x='Country',y='InfraIndex',
                        color_discrete_sequence=palette,
                        title="InfraIndex Distribution")
        st.plotly_chart(fig5b,use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 6: Ethics & Regulatory
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[5]:
        df_current = df if selected_country=="(All Countries)" else df_deep
        st.header("6. Ethics & Regulatory")
        ethic_cols=[c for c in df_current if any(
                     t in c.lower() for t in ['ethic','irb','regul','guidelines']
        )]
        df_current['HasIRB']=df_current[ethic_cols].eq('Yes').any(axis=1)
        er = df_current.groupby('Country')['HasIRB']\
             .sum().reset_index()
        fig6=px.bar(er,x='Country',y='HasIRB',
                    color='Country',
                    color_discrete_sequence=palette,
                    title="In-house IRBs")
        st.plotly_chart(fig6,use_container_width=True)
        pie_df = df_current.groupby(['Country','HasIRB'])\
                 .size().reset_index(name='Count')
        fig6b=px.pie(pie_df,names='HasIRB',values='Count',
                     facet_col='Country',
                     color_discrete_sequence=palette,
                     title="IRB Coverage")
        st.plotly_chart(fig6b,use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 7: Stakeholder Mapping
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[6]:
        df_current = df if selected_country=="(All Countries)" else df_deep
        st.header("7. Stakeholder Mapping")
        # re-filter the exploded site_clean for current sites
        clean_current = site_clean[
            site_clean['Site'].isin(df_current[next(c for c in df_current if re.search(r'\bname\b',c,re.I))])
        ]
        grouped = (clean_current
            .groupby(['Country','Stakeholder'])
            .agg(
              SitesList=('Site', lambda s: "; ".join(sorted(set(s)))),
              CountSites=('Site', lambda s: s.nunique())
            )
            .reset_index()
            .sort_values(['Country','CountSites'],ascending=[True,False])
        )
        st.dataframe(grouped,use_container_width=True,height=400)
        st.download_button("Download CSV",grouped.to_csv(index=False),
                           "stakeholders.csv","text/csv")
        # Top 5 overall
        stake_counts = (site_clean
            .groupby('Stakeholder')['Site'].nunique()
            .reset_index(name='CountSites')
            .sort_values('CountSites',ascending=False)
        )
        top5 = stake_counts.head(5)
        st.subheader("Top 5 Stakeholders (All Countries)")
        st.table(top5.set_index('Stakeholder'))
        fig_top5 = px.bar(top5,x='Stakeholder',y='CountSites',
                          color='Stakeholder',
                          color_discrete_sequence=palette,
                          title="Top 5 Stakeholders")
        st.plotly_chart(fig_top5,use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 8: Policy & Legislation
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[7]:
        df_current = df if selected_country=="(All Countries)" else df_deep
        st.header("8. Policy & Legislation")
        country_summary = (site_policy
            .groupby('Country')
            .agg(
              pct_with_policy=('Exists','mean'),
              pct_disseminated=('Disseminated','mean'),
              pct_implemented=('Implemented','mean'),
              avg_budget_alloc=('Budget','mean'),
              avg_sop_coverage=('SOP_Coverage','mean'),
              num_sites=('Exists','count')
            )
            .reset_index()
        )
        country_summary['implementation_gap'] = (
            country_summary['pct_with_policy'] - country_summary['pct_implemented']
        )
        disp = country_summary.copy()
        for p in ['pct_with_policy','pct_disseminated','pct_implemented','implementation_gap']:
            disp[p] = (disp[p]*100).round(1).astype(str)+'%'
        disp[['avg_budget_alloc','avg_sop_coverage']] = \
            disp[['avg_budget_alloc','avg_sop_coverage']].round(2)
        st.table(disp.set_index('Country'))
        # bar + pie + radar (same as before)...
        melt_bar = country_summary.melt(
            id_vars='Country',
            value_vars=['pct_with_policy','pct_disseminated','pct_implemented'],
            var_name='Metric',value_name='Value'
        )
        label_map={'pct_with_policy':'% With Policy',
                   'pct_disseminated':'% Disseminated',
                   'pct_implemented':'% Implemented'}
        melt_bar['Metric']=melt_bar['Metric'].map(label_map)
        fig_bar=px.bar(melt_bar,x='Country',y='Value',
                       color='Metric',barmode='group',
                       color_discrete_sequence=palette,
                       title="Policy Metrics")
        st.plotly_chart(fig_bar,use_container_width=True)
        # …pie & radar omitted for brevity…

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 9: Country Deep-Dive
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[8]:
        st.header(f"Deep-Dive: {selected_country}")
        if selected_country=="(All Countries)":
            st.info("Select a specific country to see details")
        else:
            # Identification
            st.subheader("Identification")
            summary_id = df_deep[[f"Is{c}" for c in cats]]\
                         .sum().rename(lambda x: x.replace("Is",""))
            st.bar_chart(summary_id)

            # Capacity
            st.subheader("Capacity Score")
            cap_score = df_deep[[f"Is{c}" for c in cats]]\
                        .sum(axis=1).mean().round(2)
            st.metric("Avg Capability",cap_score)

            # Human resources
            st.subheader("Human Resources")
            h_bool = df_deep[list(bool_groups.keys())].sum()
            h_num  = df_deep[list(num_groups.keys())].sum()
            st.table(pd.concat([h_bool,h_num]).to_frame("Count"))

            # Phase I
            st.subheader("Phase I Trials")
            st.metric("Phase I Sites", df_deep['HasPhaseI'].sum())

            # Infrastructure
            st.subheader("Infrastructure Index")
            st.bar_chart(df_deep['InfraIndex'])

            # Ethics/Reg
            st.subheader("Ethics & Regulatory")
            st.metric("In-house IRB Sites", df_deep['HasIRB'].sum())

            # Stakeholders
            st.subheader("Key Stakeholders")
            sk = (site_clean[site_clean['Site'].isin(
                    df_deep[next(c for c in df if re.search(r'\bname\b',c,re.I))])]
                 .groupby('Stakeholder')['Site']
                 .nunique()
                 .sort_values(ascending=False))
            st.bar_chart(sk.head(10))

            # Policy
            st.subheader("Policy & Legislation")
            pol = site_policy[site_policy['Country']==selected_country]
            st.metric("Policy Exists",int(pol['Exists'].sum()))
            st.metric("Avg Budget",f"{pol['Budget'].mean()*100:.1f}%")
            st.metric("Avg SOP Cov.",f"{pol['SOP_Coverage'].mean()*100:.1f}%")


    with tabs[9]:
        st.header("Spatial Overview of Core Metrics")

    fmap_long = map_df.melt(
    id_vars=["Country","ISO_A3"],
    var_name="Metric",
    value_name="Value"
)

    fig_maps = px.choropleth(
        map_long,
        locations="ISO_A3",
        locationmode="ISO-3",
        hover_name="Country",
        color="Value",
        facet_col="Metric",
        facet_col_wrap=3,
        scope="africa",
        color_continuous_scale=["#D0E8D8","#1A5632"],
    )
        # ←=== ADD THIS ===→  
    fig_maps.update_geos(
            # fitbounds="locations",
            visible=False,
            showland=True,
            landcolor="lightgray",
            showcountries=True,
            countrycolor="white"
        )

    fig_maps.update_layout(
            margin=dict(t=50, b=0, l=0, r=0),
            height=600
        )
    st.plotly_chart(fig_maps, use_container_width=True)

    progress.progress(100)

else:
    st.sidebar.info("Upload one or both CSVs, then click ‘Process Data’")
st.stop()