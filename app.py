import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="Student Service Insights", page_icon="🎓", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.6rem} .hero{padding:1.5rem 1.8rem;border-radius:18px;background:linear-gradient(120deg,#102A43,#2563EB);color:white;margin-bottom:1rem}
.hero h1{color:white;margin:0;font-size:2rem}.hero p{color:#DBEAFE;margin:.45rem 0 0}
div[data-testid="stMetric"]{background:white;border:1px solid #E7ECF3;padding:14px;border-radius:14px}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🎓 Student Service Request Intelligence</h1><p>Monitor service activity, explore delay patterns, and test a delay-risk model.</p></div>', unsafe_allow_html=True)
st.caption("Capstone prototype • Synthetic sample data • Decision support only")
DEFAULT = "data/01_student_service_requests_synthetic.csv"
up = st.sidebar.file_uploader("Upload your authorized CSV", type="csv")
try:
    df = pd.read_csv(up if up else DEFAULT)
except Exception as e:
    st.error(f"Could not load dataset: {e}"); st.stop()
required = ["Request_ID","Request_Type","Office","Date_Submitted","Priority","Status","Date_Completed","Processing_Days","Service_Target_Days","Delayed"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing)); st.stop()
df["Date_Submitted"] = pd.to_datetime(df["Date_Submitted"], errors="coerce")
df["Delayed"] = df["Delayed"].astype(str)
df["Processing_Days"] = pd.to_numeric(df["Processing_Days"], errors="coerce")
df["Service_Target_Days"] = pd.to_numeric(df["Service_Target_Days"], errors="coerce")
with st.sidebar:
    st.header("Filters")
    offices = st.multiselect("Office", sorted(df.Office.dropna().unique()), default=sorted(df.Office.dropna().unique()))
    types = st.multiselect("Request type", sorted(df.Request_Type.dropna().unique()), default=sorted(df.Request_Type.dropna().unique()))
    date_min, date_max = df.Date_Submitted.min(), df.Date_Submitted.max()
    if pd.notna(date_min) and pd.notna(date_max):
        dr = st.date_input("Submission date range", (date_min.date(), date_max.date()))
    else: dr = None
f = df[df.Office.isin(offices) & df.Request_Type.isin(types)].copy()
if dr and len(dr)==2:
    f = f[(f.Date_Submitted.dt.date>=dr[0]) & (f.Date_Submitted.dt.date<=dr[1])]
completed = f[f.Delayed.isin(["Yes","No"])].copy()
c1,c2,c3,c4=st.columns(4)
c1.metric("Requests", f"{len(f):,}")
c2.metric("Completed", f"{(f.Status=='Completed').sum():,}")
c3.metric("Pending", f"{(f.Status=='Pending').sum():,}")
delay_rate=(completed.Delayed=="Yes").mean()*100 if len(completed) else 0
c4.metric("Delay rate (completed)", f"{delay_rate:.1f}%")
t1,t2,t3,t4=st.tabs(["Overview","Delay patterns","Model evaluation","Data"])
with t1:
    a,b=st.columns(2)
    with a:
        tmp=f.groupby("Request_Type").size().reset_index(name="Requests").sort_values("Requests",ascending=False)
        st.plotly_chart(px.bar(tmp,x="Request_Type",y="Requests",color="Requests",title="Requests by type",template="plotly_white"),use_container_width=True)
    with b:
        tmp=f.groupby("Status").size().reset_index(name="Count")
        st.plotly_chart(px.pie(tmp,names="Status",values="Count",hole=.55,title="Request status",template="plotly_white"),use_container_width=True)
    monthly=f.dropna(subset=["Date_Submitted"]).assign(Month=lambda x:x.Date_Submitted.dt.to_period("M").astype(str)).groupby("Month").size().reset_index(name="Requests")
    st.plotly_chart(px.line(monthly,x="Month",y="Requests",markers=True,title="Monthly request volume",template="plotly_white"),use_container_width=True)
with t2:
    if len(completed):
        a,b=st.columns(2)
        with a:
            st.plotly_chart(px.histogram(completed,x="Processing_Days",color="Delayed",nbins=20,title="Processing time distribution",template="plotly_white"),use_container_width=True)
        with b:
            by=completed.groupby("Office").Delayed.apply(lambda s:(s=="Yes").mean()*100).reset_index(name="Delay_Rate")
            st.plotly_chart(px.bar(by,x="Office",y="Delay_Rate",title="Completed-request delay rate by office",template="plotly_white"),use_container_width=True)
        st.info("Use these patterns to review workload and service targets. A delay label is only available for completed records.")
    else: st.warning("No completed records with a Yes/No delay label in the selected filters.")
with t3:
    st.subheader("Train and compare classification models")
    st.warning("Leakage prevention: Date_Completed and Processing_Days are excluded from model inputs. Only completed records with known delay labels are used.")
    if len(completed)<20 or completed.Delayed.nunique()<2:
        st.warning("At least 20 labeled completed records and both delay classes are recommended to train and evaluate.")
    else:
        features=["Request_Type","Office","Priority","Service_Target_Days"]
        X=completed[features].copy(); y=completed.Delayed.map({"No":0,"Yes":1})
        strat=y if y.value_counts().min()>=2 else None
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=strat)
        cats=["Request_Type","Office","Priority"]; nums=["Service_Target_Days"]
        prep=ColumnTransformer([("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cats),
                                ("num",Pipeline([("imp",SimpleImputer(strategy="median"))]),nums)])
        models={"Decision Tree":DecisionTreeClassifier(max_depth=5,random_state=42,class_weight="balanced"),
                "Random Forest":RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced")}
        results=[]
        for name,model in models.items():
            pipe=Pipeline([("prep",prep),("model",model)])
            pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
            results.append({"Model":name,"Accuracy":accuracy_score(yte,pred),"Precision":precision_score(yte,pred,zero_division=0),
                            "Recall":recall_score(yte,pred,zero_division=0),"F1":f1_score(yte,pred,zero_division=0)})
        res=pd.DataFrame(results)
        st.dataframe(res.style.format({c:"{:.3f}" for c in ["Accuracy","Precision","Recall","F1"]}),use_container_width=True)
        chosen=st.selectbox("Inspect model",list(models))
        pipe=Pipeline([("prep",prep),("model",models[chosen])]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
        cm=confusion_matrix(yte,pred,labels=[0,1])
        st.plotly_chart(px.imshow(cm,text_auto=True,x=["Predicted on-time","Predicted delayed"],y=["Actual on-time","Actual delayed"],title=f"{chosen} confusion matrix",color_continuous_scale="Blues"),use_container_width=True)
        st.caption("Metrics are based on a held-out test split. Results from synthetic data are for software testing, not evidence of real school performance.")
with t4:
    st.dataframe(f, use_container_width=True, height=420)
    st.download_button("Download filtered records",f.to_csv(index=False).encode("utf-8"),"student_service_filtered.csv","text/csv")
st.caption("Research ethics: replace sample data with anonymized, authorized school records before reporting research findings.")
