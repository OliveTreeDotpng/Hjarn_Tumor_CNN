import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

# Konfigurera sidan
st.set_page_config(
    page_title="Hjärntumör AI-klassificerare",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titel och beskrivning
st.title("🧠 Hjärntumör AI-klassificerare")
st.markdown("""
**Artificiell intelligens för medicinsk bildanalys**

Denna applikation använder en djup neural nätverksmodell (EfficientNetV2B0) för att analysera medicinska bilder 
och detektera förekomsten av hjärntumörer. Modellen är tränad på medicinska bilder och kan klassificera bilder 
som antingen "Tumör" eller "Ingen tumör".

⚠️ **Viktig information**: Detta är endast ett demonstrationssyfte och ska inte användas för verklig medicinsk diagnos.
""")

@st.cache_resource
def load_model_and_config():
    """Ladda modell och konfiguration (cached för prestanda)"""
    try:
        # Ladda modellen
        model = tf.keras.models.load_model('brain_tumor_model.h5')
        
        # Ladda konfiguration
        with open('model_config.json', 'r') as f:
            config = json.load(f)
            
        return model, config
    except FileNotFoundError:
        st.error("❌ Modell eller konfigurationsfiler saknas. Kör först main.ipynb för att träna modellen.")
        return None, None

def preprocess_image(image, img_size):
    """Förprocessa bild för modellen"""
    # Konvertera till RGB om det behövs
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Ändra storlek
    image = image.resize((img_size, img_size))
    
    # Konvertera till numpy array
    img_array = np.array(image)
    
    # Lägg till batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocessing för EfficientNet
    img_array = preprocess_input(img_array)
    
    return img_array

def predict_tumor(model, image, threshold, img_size):
    """Gör prediktion på bilden"""
    # Förprocessa bilden
    processed_img = preprocess_image(image, img_size)
    
    # Gör prediktion
    prediction = model.predict(processed_img, verbose=0)[0][0]
    
    # Klassificera baserat på tröskelvärde
    is_tumor = prediction >= threshold
    confidence = prediction if is_tumor else 1 - prediction
    
    return is_tumor, prediction, confidence

# Ladda modell och konfiguration
model, config = load_model_and_config()

if model is not None and config is not None:
    # Sidebar med modellinformation
    st.sidebar.header("📊 Modellinformation")
    st.sidebar.write(f"**Modell**: EfficientNetV2B0")
    st.sidebar.write(f"**F1-score**: {config['f1_score']:.3f}")
    st.sidebar.write(f"**Precision**: {config['precision']:.3f}")
    st.sidebar.write(f"**Recall**: {config['recall']:.3f}")
    st.sidebar.write(f"**Optimal tröskel**: {config['best_threshold']:.2f}")
    
    # Huvudområde för bilduppladdning
    st.header("📤 Ladda upp medicinsk bild")
    
    uploaded_file = st.file_uploader(
        "Välj en bild (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Ladda upp en medicinsk hjärnbild för analys"
    )
    
    if uploaded_file is not None:
        # Visa den uppladdade bilden
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uppladdad bild")
            image = Image.open(uploaded_file)
            st.image(image, caption="Originalbild", use_column_width=True)
            
            # Visa bildinformation
            st.write(f"**Bildstorlek**: {image.size}")
            st.write(f"**Bildformat**: {image.format}")
            st.write(f"**Färgmodell**: {image.mode}")
        
        with col2:
            st.subheader("🤖 AI-analys")
            
            # Gör prediktion
            with st.spinner("Analyserar bild..."):
                is_tumor, raw_prediction, confidence = predict_tumor(
                    model, image, config['best_threshold'], config['img_size']
                )
            
            # Visa resultat
            if is_tumor:
                st.error("⚠️ **TUMOR DETEKTERAD**")
                st.write(f"Sannolikhet för tumor: **{raw_prediction:.1%}**")
                st.write(f"Konfidensnivå: **{confidence:.1%}**")
            else:
                st.success("✅ **INGEN TUMOR DETEKTERAD**")
                st.write(f"Sannolikhet för tumor: **{raw_prediction:.1%}**")
                st.write(f"Konfidensnivå: **{confidence:.1%}**")
            
            # Progress bar för sannolikhet
            st.write("**Sannolikhetsfördelning:**")
            st.progress(float(raw_prediction), text=f"Tumor: {raw_prediction:.1%}")
            st.progress(float(1-raw_prediction), text=f"Ingen tumor: {(1-raw_prediction):.1%}")
            
            # Förklaring av resultat
            st.info(f"""
            **Hur tolkningar fungerar:**
            - Modellen ger en sannolikhet mellan 0 och 1
            - Tröskelvärde: {config['best_threshold']:.2f}
            - Över tröskeln = Tumor detekterad
            - Under tröskeln = Ingen tumor
            """)
    
    # Exempel på hur man använder appen
    st.header("💡 Hur använder man appen?")
    st.markdown("""
    1. **Ladda upp en bild**: Använd filuppladdaren ovan för att välja en medicinsk hjärnbild
    2. **Vänta på analys**: AI-modellen kommer att analysera bilden automatiskt
    3. **Tolka resultaten**: Se sannolikheten och klassificeringen
    4. **Observera konfidensen**: Högre konfidens betyder säkrare prediktion
    
    **Tips för bästa resultat:**
    - Använd tydliga, högkvalitativa medicinska bilder
    - Se till att bilden visar hjärnvävnad tydligt
    - Undvik bilder med mycket brus eller artefakter
    """)
    
    # Varning och disclaimer
    st.header("⚠️ Viktig information")
    st.warning("""
    **MEDICINSK DISCLAIMER:**
    
    Denna applikation är endast för demonstrationsändamål och forskningssyften. 
    Resultaten ska ALDRIG användas för verklig medicinsk diagnos eller behandlingsbeslut.
    
    - Konsultera alltid kvalificerad medicinsk personal
    - Denna AI-modell är inte certifierad för klinisk användning
    - Resultaten kan vara felaktiga och ska inte förlitas på
    """)

else:
    st.error("""
    ❌ **Kunde inte ladda modellen**
    
    För att använda denna app, behöver du först:
    1. Köra `main.ipynb` för att träna modellen
    2. Se till att `brain_tumor_model.h5` och `model_config.json` finns i samma mapp
    """)

# Footer
st.markdown("---")
st.markdown("🤖 Powered by TensorFlow & Streamlit | EfficientNetV2B0 Architecture")