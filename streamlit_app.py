import pandas as pd
import streamlit as st
import joblib as jb

st.set_page_config(
    page_icon=":apple:",
    page_title="Apple Classification"
)

model = jb.load('apple_model.joblib')

st.title(":apple: Apple Classification")
st.markdown("Web application machine learning classification untuk memprediksi kualitas apel")

diameter = st.slider("Diameter", 5.0, 8.0, 6.5)
berat = st.slider("Berat", 70.0, 284.0, 125.0)
tebal_kulit = st.slider("Tebal kulit", 0.5, 1.0, 0.7)
kadar_gula = st.slider("Kadar gula", 8.0, 15.0, 10.5)
asal_daerah = st.pills("Asal daerah", ['Malang', 'Garut', 'Boyolali'], default="Malang")
warna = st.pills("Warna", ['hijau', 'kuning kemerahan', 'merah'], default="kuning kemerahan")
musim_panen = st.pills("Musim panen", ['kemarau', 'hujan'], default="kemarau")

if st.button("Prediksi", type="primary"):
    data_baru = pd.DataFrame([[diameter, berat, tebal_kulit, kadar_gula, asal_daerah, warna, musim_panen]], columns=['diameter', 'berat', 'tebal_kulit', 'kadar_gula', 'asal_daerah', 'warna', 'musim_panen'])
    prediksi = model.predict(data_baru)[0]
    presentase = max(model.predict_proba(data_baru)[0])
    st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
    st.balloons()

st.divider()
st.caption("Dibuat dengan :fire: oleh **Hanzz**")
