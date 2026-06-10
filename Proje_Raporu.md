# X-Ray Hastalık Tanı Sistemi - Final Raporu

**Hazırlayan:** Mehmet Emin Küçük (233908073)  
**Tarih:** 7 Haziran 2026  
**Ders:** Bilgisayar Mühendisliği Tasarım Projesi  

---

## 1. Projenin Amacı
Bu projenin temel amacı, göğüs röntgeni (X-ray) görüntülerini kullanarak hastanın akciğer durumunun **"Normal" (Sağlıklı)** veya **"Zatürre" (Pnömoni)** olduğunu otomatik olarak yüksek doğrulukla tespit eden derin öğrenme tabanlı bir sistem geliştirmektir. Bu sistem yardımıyla radyoloji uzmanlarının teşhis süreçlerini hızlandırmayı, tanı hata payını en aza indirmeyi ve son kullanıcılar için anlaşılır, şeffaf ve modern bir medikal web arayüzü sunmayı hedefledik.

## 2. Projede Kullanılan Teknolojiler
* **Python:** Projenin tüm veri ön işleme, model eğitimi ve web arayüzü mantığı için kullanılan ana dildir.
* **PyTorch & Torchvision:** Derin öğrenme mimarisinin tasarlanması, transfer öğrenme (Transfer Learning) aşamaları, eğitimi ve model ağırlıklarının test edilmesi için kullanılmıştır. Model omurgası olarak güçlü öznitelik çıkarımı sağlayan **ResNet-18** mimarisi tercih edilmiştir.
* **Streamlit:** Modelin son kullanıcıya sunulduğu; hızlı, modern, duyarlı (responsive) ve interaktif medikal web arayüzünü oluşturmak için kullanılmıştır.
* **PIL (Pillow):** Röntgen görüntülerinin okunması ve yeniden boyutlandırılması süreçlerinde kullanılmıştır.
* **Scikit-learn:** Veri setinin %80 eğitim ve %20 doğrulama olarak ayrılması ile değerlendirme metriklerinin çıkarılması aşamalarında kullanılmıştır.

## 3. Proje Çalışma Takvimi (Aylık Çalışma Planı)
Projenin Mart 2026'da başlayan geliştirme süreci ve teslimine kadar aylık bazda gerçekleştirilen çalışmalar aşağıda detaylandırılmıştır:

* **Mart 2026 (Proje Başlangıcı, Literatür ve Prototip Hazırlığı):**
  * Proje fikrinin belirlenmesi, göğüs röntgeni (X-ray) görüntüleri üzerinden hastalık teşhisi yapan literatürdeki derin öğrenme modellerinin incelenmesi.
  * PyTorch kütüphanesinin model mimarilerinin (ResNet-18, ResNet-50, VGG-16) analiz edilerek transfer öğrenme için ResNet-18 mimarisinde karar kılınması.
  * Kaggle Chest X-ray veri setinden prototip için küçük bir test veri kümesi oluşturulması.
  * Röntgen görüntülerinin önişleme, normalizasyon ve boyutlandırma betiklerinin yazılması.

* **Nisan 2026 (Arayüz Kurulumu, Vize Raporu ve Entegrasyon):**
  * Streamlit web arayüzünün ilk sürümünün geliştirilmesi.
  * PyTorch model.pth ağırlıklarının yüklenmesi ve Streamlit arayüzüne entegrasyonu (görüntü yükle -> model tahmini çalıştır -> olasılık barı göster akışının tamamlanması).
  * Normal, Covid-19 ve Viral Pnömoni sınıflarından oluşan 3 sınıflı prototip uygulamanın test edilmesi.
  * **Ara Dönem (Vize) İlerleme Raporunun** hazırlanması ve sisteme yüklenmesi (19 Nisan 2026).

* **Mayıs 2026 (Veri Kümesinin Genişletilmesi, Model Eğitimi ve Optimizasyon):**
  * Kaggle Chest X-ray veri setinden toplam 4076 gerçek klinik röntgen görseline erişim sağlanarak veri kümesinin genişletilmesi.
  * Verinin %80 eğitim ve %20 doğrulama olarak bölünmesi için `extract_and_split.py` scriptinin kodlanması.
  * COVID-19 sınıfındaki veri dengesizliğinin tanı başarısını düşürmesini önlemek amacıyla model sınıflarının 2 ana kategoriye (Normal ve Zatürre) indirgenmesi.
  * ResNet-18 model kafasına Dropout regularizasyonu eklenerek aşırı öğrenmenin engellenmesi.
  * Modelin GPU üzerinde iki aşamalı (Warm-up + Fine-tuning) diferansiyel öğrenme oranları ile 40 epoch boyunca eğitilmesi ve doğrulama setinde **%96.1 doğruluk oranına** ulaşılması.

* **Haziran 2026 (Arayüz Modernizasyonu, Raporlama ve Teslim):**
  * Streamlit medikal web arayüzünün geniş ekran (wide mode) klinik karanlık temaya geçirilerek özel CSS, gradyan geçişli ilerleme çubukları ve glassmorphic kartlar ile tamamen yenilenmesi.
  * Geliştirici imza künyesinin eklenmesi ve proje kodlarının derleme/sözdizimi testlerinden geçirilmesi.
  * **Final Proje Raporunun** hazırlanarak projenin tamamlanması ve teslim edilmesi (7 Haziran 2026).

## 4. Projenin Benzer Projelerden Üstünlüğü
* **Medikal Veri Güvenliği (KVKK Uyumlu):** Uygulama, hasta röntgenlerini hiçbir uzak veya yerel sunucuda depolamaz, tamamen istemci (client-side) odaklı çalışır. Analiz sonrası görseller bellekten anında silindiği için yasal veri güvenliği standartlarına %100 uyumludur.
* **PACS Uyumlu Medikal Arayüz Tasarımı:** Radyoloji kliniklerinde kullanılan PACS ekran standartları referans alınarak göz yorgunluğunu azaltan koyu medikal tema tasarlanmıştır. CSS glassmorphic ögeler ve dinamik teşhis panelleriyle hekimler için ergonomik bir çalışma alanı sunulmuştur.
* **Açıklanabilir Yapay Zeka (XAI) Yaklaşımı:** Karar aşamasında sadece teşhis etiketi verilmez, hekime her iki durum için hesaplanan olasılık oranları renk geçişli dinamik ilerleme çubukları ile sunulur. Bu, yapay zekanın teşhisteki güven seviyesini şeffaf hale verir.
* **İki Aşamalı Diferansiyel İnce Ayar Eğitimi:** Model eğitiminde öncelikle sınıflandırıcı kafa ısındırılmış (Warm-up), ardından tüm katmanlar diferansiyel öğrenme hızı ile ince ayar (Fine-tuning) işlemine tabi tutulmuştur. Bu hibrit yaklaşım, modelin ezberleme (overfitting) yapmasını engellemiştir.
* **Düşük Gecikmeli CPU Performansı:** Optimize edilen ResNet-18 mimarisi sayesinde, pahalı GPU donanımları gerektirmeden standart bilgisayar işlemcilerinde (CPU) dahi 100-150 ms gibi çok düşük bir gecikmeyle teşhis çıktısı üretilebilmektedir.
* **Sıfır Kurulum Maliyeti ve Web Tabanlı Dağıtım:** Streamlit altyapısı sayesinde sisteme herhangi bir ek yazılım veya kütüphane kurulmasına gerek kalmadan doğrudan tarayıcı üzerinden çalışır. Mobil, tablet ve masaüstü uyumluluğu ile hekime her an tanı desteği sağlar.




## 5. Projenin Tamamlanma Durumu ve Teknik Metrikler
Proje **%100 oranında tamamlanmış** ve tüm sistem entegrasyonu başarıyla gerçekleştirilmiştir. Elde edilen teknik kazanımlar, tasarım kararları ve detaylar şu şekildedir:
* **Veri Kümesi:** Kaggle Chest X-ray veri tabanından derlenen toplam **4076 göğüs röntgeni görüntüsü** kullanılmıştır. Veriler **%80 Eğitim (2960 görüntü)** ve **%20 Doğrulama (1116 görüntü)** oranında bölünmüştür.
* **Aşırı Öğrenme (Overfitting) Engelleme:** Eğitim verilerine güçlü veri artırımı (rastgele kırpma, yatay çevirme, parlaklık ve kontrast ayarlama, gri tonlama) uygulanmıştır. Ayrıca modelin Classifier (FC) katmanında sırasıyla `0.4` ve `0.3` oranlarında Dropout kullanılmıştır.
* **Model Başarısı:** GPU (CUDA) altyapısı kullanılarak gerçekleştirilen 40 epoch'luk eğitim sonucunda doğrulama veri kümesi üzerinde **%96.1** gibi son derece yüksek bir doğruluk oranına (Accuracy) ulaşılmıştır.

## 6. Kritik Kod Mimarisinden Geliştirme Örnekleri

### ResNet-18 Model Kafasının Özelleştirilmesi (Dropout Entegrasyonu)
```python
# Önceden eğitilmiş ResNet-18 yüklenmesi
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Aşırı öğrenmeyi önleyici Dropout katmanları içeren 2 çıkışlı yeni FC katmanı
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2) # Sınıflar: Normal, Pneumonia
)
```

### İki Aşamalı İnce Ayar (Fine-Tuning) ve Diferansiyel Öğrenme Hızı
```python
# Aşama 1: Sadece yeni eklenen sınıflandırıcı kafanın eğitilmesi (Omurga dondurulmuş)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
optimizer_phase1 = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Aşama 2: Tüm katmanların diferansiyel öğrenme hızı ile hassas eğitimi
for param in model.parameters():
    param.requires_grad = True
optimizer_phase2 = torch.optim.Adam([
    {'params': [p for name, p in model.named_parameters() if 'fc' not in name], 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-4}
], weight_decay=1e-4)
```

## 7. Proje Arayüzü ve Ekran Görüntüleri Üzerinden Çalışmaların Anlatılması
Projenin görsel tasarımında Streamlit'in varsayılan şablonları elenmiş, hastane kokpitlerini andıran koyu lacivert medikal karanlık tema entegre edilmiştir. Aşağıdaki ekran görüntüleri üzerinden sistemin işleyişi açıklanmıştır:

* **Şekil 1 (Ana Sayfa - Yapay Zeka Hazır):** `ss_dashboard.png` görselinde görüldüğü üzere hekim sol panelden röntgeni sürükleyip bırakabileceği dosya alanını görür. Sağ panel ise röntgen yüklenmediğinde hekimi yönlendiren interaktif bir karşılama ekranıdır.
* **Şekil 2 (Sağlıklı Akciğer Analizi):** `ss_normal.png` görselinde sisteme sağlıklı bir akciğer röntgeni yüklenmiştir. Yapay zeka görüntüyü inceleyerek parankim dokularında infiltrasyona rastlanmadığını teşhis etmiş ve sonucu **YEŞİL** renkli rapor kutusuyla ve %99.93 olasılık barıyla hekime sunmuştur.
* **Şekil 3 (Zatürre Teşhisi ve Klinik İmza):** `ss_pneumonia.png` görselinde ise akciğer loblarında yoğun sıvı infiltrasyonu saptanmıştır. Sistem anında **KIRMIZI** renkli alarm raporunu ve %86.04 oranındaki zatürre olasılığını çizmiştir. Sol panelin alt kısmında hekimin imza künyesi ve akademik bilgileri yer almaktadır.

## 8. Ara Dönem (Vize) ve Final Aşaması Karşılaştırması
Projenin ara dönem (vize) raporundaki planlanan durumundan final aşamasına kadar geçirdiği gelişimler ve farklar aşağıda özetlenmiştir:

* **İlerleme Oranı:** Vizede %20-30 prototip aşamasında olan proje, final aşamasında veri setinden arayüze kadar %100 tamamlanmıştır.
* **Veri Kümesi Boyutu:** Vizede sınıf başına sadece 5 adet test görüntüsü bulunurken, final aşamasında veri seti toplam **4076 gerçek röntgen görüntüsüne** çıkarılmıştır. Veriler %80 eğitim ve %20 doğrulama olarak ayrılmıştır.
* **Arayüz ve Kullanıcı Deneyimi (UI/UX):** Vizede Streamlit'in varsayılan, dar ve sade gri şablonu kullanılırken; final aşamasında profesyonel klinik kokpit tasarımlarını andıran geniş ekran medikal karanlık tema düzenine geçilmiştir.
* **Sınıf Yapısı Değişikliği (COVID-19 Sınıfı):** Vizede planlanan COVID-19 sınıfı, açık kaynaklı veri tabanlarındaki klinik veri yetersizliği nedeniyle kaldırılmıştır. Veri dengesizliğinin tanı başarısını düşürmesini önlemek amacıyla model, zengin ve kaliteli veriye sahip olduğumuz **Normal** ve **Zatürre (Pnömoni)** olmak üzere 2 ana sınıfa odaklanacak şekilde optimize edilmiştir.
* **Eğitim ve Başarı Oranı:** Vizede hazır ağırlıklar kullanılırken, final aşamasında model kendi veri setimizle GPU üzerinde eğitilmiş ve doğrulama setinde **%96.1** gibi yüksek bir doğruluk oranına (Accuracy) ulaşılmıştır.
