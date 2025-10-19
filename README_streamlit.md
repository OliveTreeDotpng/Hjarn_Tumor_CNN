# 🧠 Hjärntumör AI-klassificerare med Streamlit

En elegant webbapplikation för att klassificera hjärntumörer med hjälp av artificiell intelligens.

## 🚀 Snabbstart

### 1. Träna modellen först
```bash
# Kör main.ipynb i Jupyter/VS Code för att träna modellen
# Detta skapar brain_tumor_model.h5 och model_config.json
```

### 2. Installera Streamlit-beroenden
```bash
pip install -r requirements_streamlit.txt
```

### 3. Kör Streamlit-appen
```bash
streamlit run streamlit_app.py
```

Appen öppnas automatiskt i din webbläsare på `http://localhost:8501`

## 📁 Filstruktur

```
Hjarn_Tumor_CNN/
├── main.ipynb                    # Träningsnotebook
├── streamlit_app.py              # Streamlit-applikation
├── requirements_streamlit.txt    # Python-beroenden för Streamlit
├── brain_tumor_model.h5          # Tränad modell (skapas efter träning)
├── model_config.json             # Modellkonfiguration (skapas efter träning)
└── brain_tumor_dataset_split/    # Dataset
    ├── train/
    └── val/
```

## 🎯 Funktioner

- **Drag & Drop bilduppladdning**: Enkelt att ladda upp bilder
- **Realtidsanalys**: Direkt AI-prediktion på uppladdade bilder
- **Visuell feedback**: Tydliga resultat med sannolikheter och konfidens
- **Modellinformation**: Visa modellprestanda i sidebar
- **Responsiv design**: Fungerar på desktop och mobil
- **Säkerhetsvarningar**: Tydliga disclaimers för medicinsk användning

## 🔧 Anpassning

Du kan enkelt anpassa appen genom att:
- Ändra teman i `st.set_page_config()`
- Lägga till fler modellinformationsfält
- Anpassa layouten med Streamlit's kolumnsystem
- Lägga till fler bildformat eller preprocessing-steg

## ⚠️ Viktig information

Denna applikation är endast för demonstrationsändamål och ska inte användas för verklig medicinsk diagnos.

## 🚀 Deploy till molnet

För att deployka till Streamlit Cloud:
1. Push koden till GitHub
2. Gå till [share.streamlit.io](https://share.streamlit.io)
3. Koppla ditt GitHub-repo
4. Välj `streamlit_app.py` som main file