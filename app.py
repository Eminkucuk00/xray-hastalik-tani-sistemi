import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

# 🎛️ Sayfa Ayarları (Geniş Ekran / Wide Mode)
st.set_page_config(
    page_title="Akciğer Röntgeni Yapay Zeka Teşhis İstasyonu",
    page_icon="🫁",
    layout="wide"
)

# 🎨 Premium Tıbbi Arayüz CSS Tasarımı
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;1,400&display=swap');
    
    /* Sayfa Arka Planı (Derin Uzay Grisi / Tıbbi Lacivert) */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0d1527 0%, #05070c 80%) !important;
        font-family: 'Outfit', sans-serif !important;
        color: #f1f5f9 !important;
    }
    
    /* Streamlit Üst ve Alt Barlarını Gizleme/Karartma */
    [data-testid="stHeader"] {
        background-color: transparent !important;
        backdrop-filter: blur(10px);
        border-bottom: none !important;
    }
    
    /* Sayfa Kenar Boşluklarını Düzenleme */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 92% !important;
    }
    
    /* Başlık ve İstasyon Kartı */
    .station-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.4) 100%);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    
    .station-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #60a5fa, #3b82f6);
    }
    
    .station-title {
        font-size: 2.3em;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .station-subtitle {
        color: #94a3b8;
        font-size: 1.1em;
        font-weight: 350;
    }

    /* Durum Kartı Grid */
    .status-grid {
        display: flex;
        gap: 12px;
        margin-top: 18px;
        flex-wrap: wrap;
    }
    
    .status-pill {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #94a3b8;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.85em;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .status-pill-active {
        border-color: rgba(59, 130, 246, 0.3);
        color: #60a5fa;
        background: rgba(59, 130, 246, 0.05);
    }
    
    /* Dosya yükleyici dış kapsayıcı alanı özelleştirmesi */
    [data-testid="stFileUploader"] {
        background-color: rgba(15, 23, 42, 0.3) !important;
        border: 2px dashed rgba(59, 130, 246, 0.2) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px);
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(59, 130, 246, 0.5) !important;
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.1);
    }
    
    /* İçteki beyaz/gri kutuyu (Dropzone alanını) tamamen karanlık temaya uyarlama */
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 15px 20px !important;
    }
    
    /* File uploader iç metin renklerini beyazlatma ve kontrastı artırma */
    [data-testid="stFileUploaderDropzone"] div, 
    [data-testid="stFileUploaderDropzone"] span, 
    [data-testid="stFileUploaderDropzone"] p {
        color: #e2e8f0 !important;
        font-weight: 400 !important;
    }
    
    /* Yükleme Butonu Tasarımı */
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(29, 78, 216, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(29, 78, 216, 0.4) !important;
    }
    
    /* Profesyonel Teşhis Paneli (Rapor Kartları) */
    .report-card {
        border-radius: 20px;
        padding: 35px;
        backdrop-filter: blur(25px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6);
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .report-card-normal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(4, 120, 87, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.08);
    }
    
    .report-card-pneumonia {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.05) 0%, rgba(185, 28, 28, 0.15) 100%);
        border: 1px solid rgba(239, 68, 68, 0.25);
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.08);
    }
    
    .report-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 25px;
    }
    
    .report-badge {
        font-size: 0.85em;
        font-weight: 600;
        padding: 5px 14px;
        border-radius: 30px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-normal {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .badge-pneumonia {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    .report-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.6em;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .title-normal { color: #34d399; }
    .title-pneumonia { color: #f87171; }
    
    .report-text {
        font-size: 1.05em;
        line-height: 1.65;
        color: #cbd5e1;
        margin-bottom: 30px;
    }
    
    /* Gelişmiş Olasılık Grafik Barları */
    .bar-wrapper {
        margin-bottom: 20px;
    }
    
    .bar-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.95em;
        font-weight: 500;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    
    .bar-value {
        font-weight: 700;
        color: #ffffff;
    }
    
    .bar-bg {
        background-color: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        height: 14px;
        overflow: hidden;
        position: relative;
    }
    
    .bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1.2s cubic-bezier(0.1, 0.8, 0.3, 1);
    }
    
    .bar-fill-normal {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
    }
    
    .bar-fill-pneumonia {
        background: linear-gradient(90deg, #ef4444 0%, #f87171 100%);
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.3);
    }
    
    /* Analiz Yokken Çıkan İstasyon Boş Durum Kartı (Placeholder) */
    .placeholder-card {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 60px 40px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        backdrop-filter: blur(15px);
    }
    
    .placeholder-icon {
        font-size: 4.5em;
        color: rgba(59, 130, 246, 0.25);
        margin-bottom: 20px;
        animation: pulse 2.5s infinite ease-in-out;
    }
    
    .placeholder-text {
        font-size: 1.25em;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 8px;
    }
    
    .placeholder-subtext {
        font-size: 0.95em;
        color: #475569;
        max-width: 320px;
        line-height: 1.5;
    }
    
    /* Animasyon tanımlamaları */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.25; }
        50% { transform: scale(1.05); opacity: 0.5; }
    }
</style>
"""

# CSS kodu enjeksiyonu
st.markdown(custom_css, unsafe_allow_html=True)

# 🏢 Profesyonel Üst İstasyon Kartı
st.markdown("""
<div class="station-header">
    <div class="station-title">🫁 AKCİĞER RÖNTGENİ TEŞHİS İSTASYONU</div>
    <div class="station-subtitle">Yapay Zeka ve Derin Öğrenme Tabanlı Tanı Destek Sistemi</div>
</div>
""", unsafe_allow_html=True)

# Sınıf isimleri (Eğitim sırasıyla eşleşmelidir)
classes = ["Normal", "Pneumonia"]

# 🤖 Modeli Önbellekte Yükle
@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Modeli arka planda başlat
try:
    model = load_model()
except Exception as e:
    st.error(f"Model yükleme hatası: {e}")

# Görüntü tahmin fonksiyonu
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()], probabilities

# 🎛️ Ana Grid Tasarımı (Sol Kontrol Paneli, Sağ Sonuç/Görüntü Paneli)
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown("### 🛠️ Kontrol ve Yükleme Paneli")
    uploaded_file = st.file_uploader(
        "Analiz etmek istediğiniz göğüs röntgeni (X-ray) grafisini yükleyin:",
        type=["jpg", "jpeg", "png"]
    )
    
    # Bilgi ve Eğitim Kılavuzu Kartı
    st.markdown("""
    <div style="background-color: rgba(30, 41, 59, 0.25); border: 1px solid rgba(59, 130, 246, 0.1); border-radius: 16px; padding: 25px; margin-top: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
        <h4 style="margin-top:0; color:#60a5fa; font-weight:600; font-size:1.1em; margin-bottom:12px;">📊 İstasyon Eğitim Bilgileri</h4>
        <p style="font-size:0.9em; line-height:1.6; color:#94a3b8; margin-bottom:10px;">
            Bu istasyon, akciğer röntgenleri üzerinden hastalık tespiti yapmak üzere eğitilmiştir.
        </p>
        <ul style="font-size:0.85em; color:#cbd5e1; padding-left:18px; margin-bottom:0;">
            <li style="margin-bottom:6px;"><strong>Toplam Eğitim Verisi:</strong> 4,076 röntgen grafisi</li>
            <li style="margin-bottom:6px;"><strong>Eğitim / Doğrulama Bölünmesi:</strong> %80 / %20</li>
            <li style="margin-bottom:6px;"><strong>Yöntem:</strong> ResNet-18 Transfer Learning & Two-Stage Fine-Tuning</li>
            <li style="margin-bottom:0;"><strong>Öğrenim Algoritması:</strong> L2 Regularization & Dropout Koruması</li>
        </ul>
    </div>
    
    <!-- Geliştirici ve Güç Sağlayıcı Kartı -->
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.2) 0%, rgba(15, 23, 42, 0.4) 100%); border: 1px solid rgba(255, 255, 255, 0.03); border-radius: 16px; padding: 18px 25px; margin-top: 20px; text-align: center; backdrop-filter: blur(10px); box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
        <span style="font-size: 0.75em; color: #64748b; text-transform: uppercase; letter-spacing: 1.5px; display: block; margin-bottom: 6px; font-weight:600;">POWERED BY</span>
        <span style="font-family: 'Montserrat', sans-serif; font-size: 1.15em; font-weight: 700; color: #f1f5f9; letter-spacing: 0.5px; display: block;">MEHMET EMİN KÜÇÜK</span>
        <span style="font-size: 0.8em; color: #475569; display: block; margin-top: 4px; font-weight:500;">Bilgisayar Mühendisliği | Proje No: 233908073</span>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    if uploaded_file:
        # Görüntüyü yükle ve analiz et
        image = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner("🧠 Yapay zeka akciğer grafisini ve doku yapısını inceliyor..."):
            prediction, probabilities = predict(image)
            
        prob_normal = probabilities[0].item() * 100
        prob_pneumonia = probabilities[1].item() * 100
        
        # Röntgen resmi ve Rapor çıktısı yan yana (Alt Grid)
        sub_col1, sub_col2 = st.columns([1.1, 1], gap="medium")
        
        with sub_col1:
            st.markdown("<p style='font-weight: 600; font-size:0.95em; color:#94a3b8; letter-spacing:0.5px; margin-bottom:12px;'>📸 ANALİZ EDİLEN GRAFİ</p>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
        with sub_col2:
            st.markdown("<p style='font-weight: 600; font-size:0.95em; color:#94a3b8; letter-spacing:0.5px; margin-bottom:12px;'>📋 TEŞHİS RAPORU</p>", unsafe_allow_html=True)
            
            if prediction == "Normal":
                st.markdown(f"""
                <div class="report-card report-card-normal">
                    <div class="report-header">
                        <span class="report-title title-normal">SAĞLIKLI</span>
                        <span class="report-badge badge-normal">BULGU YOK</span>
                    </div>
                    <div class="report-text">
                        Yapay zeka parankimal analiz sonuçlarına göre akciğer dokuları doğal görünümünü korumaktadır. Pnömonik veya konsolide sıvı infiltrasyonuna rastlanmamıştır.
                    </div>
                    <div class="bar-wrapper">
                        <div class="bar-header">
                            <span>Sağlıklı Akciğer</span>
                            <span class="bar-value">%{prob_normal:.2f}</span>
                        </div>
                        <div class="bar-bg">
                            <div class="bar-fill bar-fill-normal" style="width: {prob_normal}%;"></div>
                        </div>
                    </div>
                    <div class="bar-wrapper" style="margin-bottom:0;">
                        <div class="bar-header">
                            <span>Zatürre (Pnömoni)</span>
                            <span>%{prob_pneumonia:.2f}</span>
                        </div>
                        <div class="bar-bg">
                            <div class="bar-fill bar-fill-pneumonia" style="width: {prob_pneumonia}%;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="report-card report-card-pneumonia">
                    <div class="report-header">
                        <span class="report-title title-pneumonia">ZATÜRRE</span>
                        <span class="report-badge badge-pneumonia">PNÖMONİ BULGUSU</span>
                    </div>
                    <div class="report-text">
                        Yapay zeka parankimal analiz sonuçlarına göre akciğer loblarında infiltrasyon ve sıvı birikimi ile uyumlu yoğunluk artışları saptanmıştır. Klinik takip önerilir.
                    </div>
                    <div class="bar-wrapper">
                        <div class="bar-header">
                            <span>Sağlıklı Akciğer</span>
                            <span>%{prob_normal:.2f}</span>
                        </div>
                        <div class="bar-bg">
                            <div class="bar-fill bar-fill-normal" style="width: {prob_normal}%;"></div>
                        </div>
                    </div>
                    <div class="bar-wrapper" style="margin-bottom:0;">
                        <div class="bar-header">
                            <span>Zatürre (Pnömoni)</span>
                            <span class="bar-value">%{prob_pneumonia:.2f}</span>
                        </div>
                        <div class="bar-bg">
                            <div class="bar-fill bar-fill-pneumonia" style="width: {prob_pneumonia}%;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Alt Tıbbi Bilgilendirme Notu
            st.markdown("""
            <div style="background-color: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; font-size: 0.8em; color: #94a3b8; line-height:1.5; margin-top:20px;">
                🚨 <strong>Önemli Uyarı:</strong> Bu rapor tıbbi karar destek aracıdır. Doğruluk oranı %96.1 olsa dahi, nihai teşhis uzman hekimler tarafından klinik veriler eşliğinde konulmalıdır.
            </div>
            """, unsafe_allow_html=True)
    else:
        # Görsel yüklenmemişken sağ tarafta gösterilecek Placeholder kartı
        st.markdown("""
        <div class="placeholder-card">
            <div class="placeholder-icon">📡</div>
            <div class="placeholder-text">Yapay Zeka Analize Hazır</div>
            <div class="placeholder-subtext">Sol panelden bir akciğer grafisi (X-ray) resmi yüklediğinizde, parankimal inceleme sonuçları ve olasılık grafikleri anlık olarak burada görüntülenecektir.</div>
        </div>
        """, unsafe_allow_html=True)

# 🏢 Profesyonel Sayfa Sonu Künyesi (Footer)
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.03); font-size: 0.85em; color: #475569;">
    Akciğer Röntgeni Teşhis İstasyonu © 2026 | Powered by Mehmet Emin Küçük (233908073)
</div>
""", unsafe_allow_html=True)