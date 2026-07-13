<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
</div>

<h1 align="center">X-Ray (Röntgen) Görüntülerinden Zatürre (Pneumonia) Tanı Sistemi 🩻</h1>

<p align="center">
  Derin öğrenme (Deep Learning) teknikleri kullanılarak, akciğer röntgen görüntülerinden hastada <b>Zatürre (Pneumonia)</b> olup olmadığını yüksek doğrulukla tespit eden yapay zeka destekli bir tanı sistemi.
</p>

---

## 🚀 Proje Hakkında

Bu proje, tıp alanındaki uzmanlara destek olmak ve teşhis süreçlerini hızlandırmak amacıyla geliştirilmiştir. Binlerce açık kaynaklı röntgen (X-Ray) görüntüsü kullanılarak eğitilen Convolutional Neural Network (Evrişimli Sinir Ağı - CNN) modeli, sisteme yüklenen yeni röntgen görüntülerini analiz ederek saniyeler içerisinde "Sağlıklı" veya "Zatürre (Pneumonia)" tahmininde bulunur.

Proje, yalnızca bir makine öğrenmesi modeli olmakla kalmayıp, kullanıcı dostu bir arayüz/API ile entegre edilerek son kullanıcıya hitap edecek şekilde tasarlanmıştır.

## ✨ Temel Özellikler

- **Yüksek Doğruluk:** Özel CNN mimarisi sayesinde röntgen görüntülerindeki ince detayları analiz eder.
- **Hızlı Teşhis:** Arayüze yüklenen görüntüyü saniyeler içinde işler ve sonucu döndürür.
- **Kullanıcı Dostu API/Web:** `app.py` üzerinden sunulan yapıyla tıp çalışanları tarafından kolayca kullanılabilir.
- **Büyük Veri Kümesi:** Model eğitimi sırasında overfitting'i engellemek için 4000'den fazla akciğer röntgeni ile eğitilmiş ve doğrulanmıştır.

## 🛠 Kullanılan Teknolojiler

- **Yapay Zeka & Makine Öğrenmesi:** PyTorch / Torchvision
- **Veri İşleme & Görselleştirme:** NumPy, Pandas, Matplotlib, PIL
- **Backend (Sunucu):** Flask (Python)
- **Model Formatı:** `.pth` (PyTorch Model Weights)

## ⚙️ Kurulum ve Çalıştırma

Projeyi kendi yerel bilgisayarınızda (localhost) çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### Ön Koşullar
- Python 3.8 veya üzeri bir sürümün yüklü olması gerekmektedir.

### Adımlar

1. **Projeyi Klonlayın:**
   ```bash
   git clone https://github.com/Eminkucuk00/xray-hastalik-tani-sistemi.git
   cd xray-hastalik-tani-sistemi
   ```

2. **Gerekli Kütüphaneleri Yükleyin:**
   Eğer projenin içinde bir `requirements.txt` bulunuyorsa doğrudan kurabilirsiniz, yoksa temel kütüphaneleri yükleyin:
   ```bash
   pip install torch torchvision flask numpy pillow matplotlib
   ```

3. **Uygulamayı Başlatın:**
   ```bash
   python app.py
   ```

4. **Tarayıcıda Görüntüleyin:**
   Uygulama başarıyla çalıştığında genellikle `http://127.0.0.1:5000` adresinden sisteme erişebilir ve yeni röntgen görüntülerini test edebilirsiniz.

## 📸 Ekran Görüntüleri ve Çıktılar

*(Buraya uygulamanızın arayüzünden ve modelin teşhis sonuçlarından ekran görüntüleri eklenecektir.)*

---
*Bu proje Emin Küçük tarafından geliştirilmiştir.*
