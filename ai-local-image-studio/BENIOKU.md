# 🎨 AI Local Image Studio v1.3.0

> 🇬🇧 English overview: **[README.md](README.md)**

%100 yerel çalışan yapay zeka görsel oluşturucu. Görsel tarz **AI Local Translator** (SRT projesi) ile aynı: koyu tema, numaralı 1–7 bölümler, GPU çubuğu, log ekranı.

## v1.3.0'da yeni — yüz modu

- **👤 Yüz (bir kişi ya da karakter)** türü eklendi. Mekân modunun tersidir: her görselden yalnızca yüz (saçla birlikte) kesilip öğretilir.
  - Yüzler, Yüz ve El Düzeltici'nin YuNet algılayıcısıyla bulunur. Her görselden en büyük yüz alınır. Yüz bulunamayan ya da 40 pikselden küçük yüzü olan görseller atlanır ve log'da sayısı yazar.
  - Kesitte yüz ve saç tam ağırlıkla öğretilir; arka plan, boyun ve kıyafet çok az (0.2) öğretilir.
  - Ayna kullanılmaz: saç ayrımı, ben gibi asimetrik ayrıntılar kimliğin parçasıdır.
  - Eğitim açıklama kullanmaz; her kesit "tetikleyici, close-up face portrait" metniyle öğretilir. Bu yüzden derin modda Florence-2 gerekmez.
  - Derin modda, veri seti başına bir kez sorulur: yüzler sizin, izin veren kişilerin ya da hayali karakterlerin mi? Gerçek kişileri izinleri olmadan taklit etmek için kullanmayın.
  - Hafif modda kelime kartı yüz ve saç özelliklerini toplar (saç rengi, göz rengi, gözlük, çil…). LLM'den kişiyi tanımlaması istenmez.

## v1.2.1'de yeni — mekânda çevre öne çıkar, LoRA'lar korunur

- **Mekân eğitiminde yüzler gizlenir** ("🙈 Yüzleri gizle, çevreyi öne çıkar", Mekân + Derin modda varsayılan açık):
  - Her görseldeki yüzler bulunur (Yüz ve El Düzeltici'nin YuNet algılayıcısı; kurulu değilse onayla kurulur) ve tanınmayacak biçimde bulanıklaştırılır.
  - Yüz bölgeleri eğitimde hiç öğretilmez. Böylece LoRA ne gerçek yüzleri ne de bulanık yüzleri öğrenir.
  - İnsan gövdeleri az öğretilir (ağırlık 0.35), çevre tam ağırlıkla öğretilir. Sonuç: LoRA sokakları, binaları, ışığı öğrenir; yoldan geçenleri öğrenmez.
  - Mekânın kelime kartına insanla ilgili kelimeler (adam, kadın, yürüyen, kalabalık…) girmez.
  - Görsellerin büyük kısmı insan olan kesitler atlanır.
- **Var olan LoRA asla ezilmez.** Bir veri setinin türü değişince ya da aynı etiket başka bir veri setinde kullanılınca eğitim kaldığı yerden devam edemez. Önceki sürüm bu durumda eski LoRA'nın üstüne yazıyordu. Artık eski dosya olduğu gibi kalır, yeni eğitim yeni bir adla kaydedilir (ör. `istanbul_sokak_place.safetensors`) ve log'da yazar.
- **LoRA penceresinde "Sil" kalıcı silmez.** Dosya, bilgi dosyası ve örnek görselleriyle `loras\.trash\<tarih-saat>\` klasörüne taşınır; geri almak için oradan `loras\` içine taşımak yeterli. Onay penceresinde artık silinecek şeyin LoRA mı kelime kartı mı olduğu ve dosya adı yazıyor (kart ve LoRA aynı adı taşıyabildiği için).
- **Gizlenen LoRA'lar belirtilir.** "Bu modele uyanlar" filtresinde seçili modele uymayan LoRA'lar (ör. SDXL seçiliyken SD 1.5 LoRA'ları) gizlenir. Pencere artık kaç tanesinin gizlendiğini söylüyor; silinmiş sanılmasın.

## v1.2.0'da yeni — videodan veri seti, öğretilen tür, sağlam ilk kurulum

- **Ne öğretilecek?** Eğitim penceresinde her veri setinin türü seçilir: 🎨 Stil, 🏙️ Mekân / şehir, 🧸 Konu / nesne / karakter, 🧍 İnsan anatomisi / poz. Artık bir şehir ya da anatomi için "stil" denmez.
  - Tür, eğitimi de değiştirir. Stil ve anatomide görseller aynalanarak da öğretilir. Mekân ve konuda ayna kullanılmaz: tabelalar ve trafik ters dönmez, asimetrik detaylar korunur. Mekânda geniş video kareleri iki uçtan da öğretilir. Anatomi daha düşük öğrenme hızıyla, daha yumuşak öğretilir.
  - Tetikleyici kelime türe göre: stilde `manga_tarzi style`, diğerlerinde `istanbul_sokak`.
  - Hafif mod artık "kelime kartı" üretir ve türe uygun kelimeleri seçer: mekânda mimari, sokak ve ışık; anatomide poz ve bakış açısı (+ kaçınılacak anatomi hataları); konuda ortak görünüm özellikleri.
  - LoRA seçme penceresi ve ana ekran türü gösterir (ör. "🏙️ Program · Mekân", "📝 Kart · Mekân"). Arama kutusu tür adıyla da arar.
- **Videodan veri seti:** "Klasörden al…" ya da "Dosya seç…" ile seçilen yerde video varsa önce sorulur.
  - **Evet** derseniz her videodan en fazla seçilen sayıda (30–400, varsayılan 120) kare alınır. Video eşit parçalara bölünür ve her parçadan en net kare seçilir. Kararmış ya da düz geçişler, bulanık bölümler ve bir öncekiyle neredeyse aynı sahneler atlanır. 4K kareler 1920 piksele küçültülür.
  - **Hayır** derseniz videolar atlanır.
  - Video okumak için OpenCV gerekir (Apache-2.0, ~40 MB). Kurulu değilse onayla kurulur.
- **Dosyalar veri setinin adıyla kaydedilir:** `istanbul_sokak_0001.jpg`, `istanbul_sokak_0002.jpg` … Bilgisayardan, internetten ve videodan gelenlerin hepsi böyle adlandırılır. Var olan dosya asla ezilmez.
- **Otomatik açıklama:** İçe aktarma bitince açıklaması olmayan yeni görseller (video kareleri dahil) hemen Florence-2 ile okunur. Görsel motoru henüz kurulu değilse açıklamalar BAŞLAT'ta yazılır.
- **İlk kurulum (Python'u yeni kurulmuş bilgisayar):**
  - Program kendi ortamını kurmadan önce sorar: **programa özel Python 3.12** (önerilen; ~45 MB: `uv` aracı + Python, lisanslar MIT/Apache-2.0 ve PSF) ya da bilgisayardaki Python. Programa özel Python veri klasörüne (`runtime\python`) kurulur. Windows'a, PATH'e ve kayıt defterine bir şey yazılmaz. Microsoft Store'un Python güncellemeleri ya da çok yeni bir Python sürümü programın ortamını bozamaz.
  - Bilgisayardaki Python çok yeni, çok eski ya da 32 bit ise program bunu söyler ve programa özel Python'u önerir.
  - Bilgisayardaki Python'la kurulum başarısız olursa program programa özel Python'la yeniden denemeyi teklif eder.
  - pip önce güncellenir. Hazır paketler (wheel) tercih edilir, derleme gerektiren kaynak paketlere gidilmez. Antivirüs bir dosyayı kilitlerse bir kez daha denenir.
  - Her adım ve pip'in bütün çıktısı **`logs\setup.log`** dosyasına yazılır. Hata penceresinde pip'in asıl hata satırları ve olası çözüm görünür: uzun dosya yolu, antivirüs kilidi, bu Python için hazır paket olmaması, HTTPS tarama.
  - Hata sonrası konsolda çıkan "application has been destroyed" yazısı giderildi.
  - Veri klasörü taşınırsa programa özel Python'un yolu otomatik düzeltilir. Motorun yeniden indirilmesi gerekmez.
- **"HF_TOKEN" uyarısı gizlendi.** Bu bir hata değil: Hugging Face sunucusu hesapsız indirmelerde bu notu gönderiyor, indirme hesapsız da çalışıyor. Hugging Face hesabınız varsa isterseniz kendi `HF_TOKEN` ortam değişkeninizi tanımlayabilirsiniz; program onu kullanır.
- Ana ekrandaki turuncu buton artık "🧠 EĞİTİM / GÖRSEL ANALİZ". LoRA butonu "🧩 LoRA / Kart" oldu.

## v1.1.0'da yeni — LoRA, görsel okuma ve kendi kendine öğrenme

- **v2 / v3 ifadeleri kaldırıldı.** Görsel Düzenleme Stüdyosu butonları da kaldırıldı; ileride paketleyiciyle eklenecek.
- **🧩 LoRA / Stil** (1. bölüm): seçme penceresi açar.
  - Arama kutusu ve filtre var: bu modele uyanlar, programın ürettikleri, benim eklediklerim, stil kartları.
  - Her LoRA'nın gücü ayrı ayarlanır. En fazla 6 seçim yapılabilir.
  - LoRA'nın hangi modele ait olduğu dosyanın yalnızca başı okunarak anlaşılır (SD 1.5 / SDXL / diğer).
  - Uymayanlar ⚠ ile gösterilir ve üretimde atlanır.
  - Programın ürettiği LoRA'ların tetikleyici kelimesi prompt'a kendiliğinden eklenir. Stil kartlarının tarz kelimeleri de prompt'a, kaçınılacakları negatife eklenir.
  - LoRA kullanmak için küçük **PEFT** paketi (Apache-2.0, ~1 MB) gerekir; ilk kullanımda onayla kurulur. Program açıkken kurulursa bir kez yeniden başlatma ister.
- **Görselleri otomatik okuma:** Açıklaması olmayan görseller **Florence-2** (Microsoft, MIT, ~0,5–0,9 GB, onayla iner) ile yazıya dökülür. Elle açıklama yazmak gerekmez. 6. bölümden de indirilebilir/silinebilir.
- **Stil Öğrenme penceresi artık gerçekten çalışıyor.** Üç mod var:
  - **Hafif — stil kartı:** Görsellerin ortak tarzı kelimelere dökülür. LLM çalışıyorsa o yazar, yoksa en sık geçen tarz kelimeleri seçilir. Her modelle çalışır.
  - **Derin — stil eğitimi (LoRA):** Veri setiyle bir LoRA eğitilir (SD 1.5, 4 GB VRAM'de çalışır). Bitiş için "ben durdurana kadar" ya da süre sınırı seçilir. Her ara kayıtta LoRA kaydedilir ve istenirse örnek görsel galeriye düşer. Durdurunca son hâl kaydedilir; yeniden başlatınca kaldığı yerden devam eder.
  - **🧠 Kendi kendine öğrenme — ödül/ceza:**
    - Galerideki **👍 / 👎** butonlarıyla puan verilir (bir kez daha basınca puan geri alınır).
    - Aynı prompt için 👍 ve 👎 alan görseller **Diffusion-DPO** ile karşılaştırılır: 👍 ödül, 👎 ceza.
    - **Otomatik hakem** açıksa program aynı prompt'la iki görsel üretir, yüz netliği, el algısı, keskinlik ve pozlamaya bakarak iyisini ödüllendirir, kötüsünü cezalandırır.
    - Sonuç birikerek **"AILS tercih" LoRA'sına** kaydedilir. Gelişmiş Ayarlar'dan SD 1.5'e otomatik uygulanır (güç ayarlı).
- 4 GB için bellek dostu eğitim: görüntü kodları ve metinler önceden hesaplanır, VAE ve metin kodlayıcı RAM'e alınır, UNet fp16 + gradyan kontrol noktaları kullanılır. DPO, kazanan ve kaybeden için ayrı geri yayılımla yapılır; gradyan, klasik yöntemle birebir aynıdır (testte fark 0).
- Ekran kartı belleği yetmezse eğitim 384 px'te yeniden denenir. RAM azalırsa bellek koruyucu eğitimi durdurur ve son hâli kaydeder.
- 👍 / 👎 butonları galeride görsel bilgisinin sağında. Galeri butonları dar pencerede ya da uzun dillerde kesilmek yerine iki satıra geçer.

## v1.0.3'te yeni — v2 penceresinde internetten arama

- **🔎 Ara ve ekle** (3. İnternet bölümü): yazdığın konuyu **Openverse**'te arar ve bulunan görselleri veri setine ekler.
  - Openverse, Wikimedia, Flickr, müzeler gibi kaynaklardaki **açık lisanslı** görselleri içerir.
  - Varsayılan lisans filtresi yalnızca ticari kullanıma uygun olanlar: CC0, kamu malı, CC BY. İstenirse CC BY-SA da eklenir.
  - Tür seçilebilir: illüstrasyon, sanat eseri, fotoğraf.
  - Her görselin sahibi, lisansı, kaynak sayfası ve atıf metni `dataset.json`'a kaydedilir.
- Türkçe/Korece/Almanca yazılan arama, LLM **zaten çalışıyor ve kuruluysa** İngilizce anahtar kelimelere çevrilir. Bunun için hiçbir şey indirilmez. LLM yoksa metin olduğu gibi aranır.
- **Ağ kuralları:** tek bağlantı, istekler arasında en az 1 sn. Openverse'ün hesapsız kullanım sınırına uyulur: sayfa başına 20 sonuç, saatte yaklaşık 5 arama sayfası. Sınıra gelinince arama durur ve log'da yazar. Sınırı aşmaya çalışılmaz.
- Neden Google/Bing değil: Bing'in arama API'leri 11 Ağustos 2025'te kapatıldı. Google'ın Custom Search API'si yeni müşteri almıyor ve 1 Ocak 2027'de kapanıyor. Google Görseller'i kazımak hem kurallara aykırı hem de telifli görselleri programa çeker.
- Pencere yeniden düzenlendi. Sol sütunda veri seti, bilgisayardan ekleme, eklentiler ve ANALİZİ BAŞLAT var. Sağ sütunda internet ve analiz ayarları var. Pencere artık 125% ölçekli dizüstü ekranlara sığıyor.
- Derin mod açıklamasına not eklendi: gerçek kişilerle yalnızca kendiniz ya da izni olan kişiler için kullanın.

## v1.0.2'de yeni

**Yüz ve el bozulmalarına karşı:**

- **Adım sayıları yükseltildi** ve yeni **Ultra** kalite eklendi. SD 1.5: 20 / 30 / 40 / 50 adım. SDXL: 25 / 35 / 45 / 60 adım.
- **Elle adım sayısı** (Gelişmiş Ayarlar). 0 girilirse kalite ön ayarı kullanılır, 1–150 arası bir sayı ön ayarın yerine geçer.
- **Örnekleyici (sampler):** SD 1.5 / SDXL için varsayılan artık **DPM++ 2M Karras**. Euler a, Euler, UniPC ve modelin kendi örnekleyicisi de seçilebilir. Turbo modeller (Z-Image, FLUX.2 klein) kendi zamanlayıcısını kullanır. Bu modellerde 8–9 adım idealdir, çok daha fazlası genelde fark etmez.
- **İkinci geçiş (hires fix):** görsel ~1,5 kat büyütülüp aynı modelle düşük güçte yeniden çizilir. Küçük yüzler ve parmaklar çok daha düzgün çıkar. *Otomatik* ayarda SD 1.5'te Dengeli ve üstünde, SDXL'de Ultra'da (ve yeterli VRAM'de) açılır. CPU profilinde otomatik olarak kapalıdır.
- **🩹 Yüz ve el düzeltici (ADetailer benzeri, 1. bölümde onay kutusu):** yüzleri ve elleri bulur, her birini modelin doğal çözünürlüğüne büyütüp yeniden çizer ve yumuşak kenarlı maskeyle yerine koyar.
  - Algılama CPU'da çalışır (1024 px görselde ~0,2 sn), **ek VRAM kullanmaz**.
  - Ek model yüklenmez. Yeniden çizimi aynı görsel modeli yapar.
  - Düzeltilecek parçalar (yüzler / eller) Gelişmiş Ayarlar'dan seçilir.
  - İlk kullanımda onayla kurulur (~50 MB, veri klasörüne): OpenCV + YuNet yüz modeli + MediaPipe avuç içi modeli + anime yüz algılayıcı. Dosyalar sabit sürümden indirilir ve SHA-256 ile doğrulanır.
  - Düzeltme ya da ikinci geçiş için bellek yetmezse o adım atlanır, görsel korunur. Düzeltme sırasında DURDUR'a basılırsa bitmiş son hâl kaydedilir.
- **Anatomi negatifleri:** SD 1.5 / SDXL'de "bad hands, extra fingers, fused fingers, poorly drawn face…" gibi negatifler otomatik eklenir. Gelişmiş Ayarlar'dan kapatılabilir.
- LLM artık insan olan görsellerde yüz ve eller için kalite terimleri de ekliyor.
- **💾 Düzeltme öncesi hâlini de kaydet** seçeneğiyle önce/sonra karşılaştırılabilir. Galeride "düzeltme öncesi" diye işaretlenir.
- Görsel bilgisine (PNG) örnekleyici, ikinci geçiş ve düzeltilen yüz/el sayısı da yazılır.

**v2 hazırlığı: 🎨 Stil Öğrenme / Görsel Analiz penceresi** (turuncu buton):

- **Veri seti + stil etiketi** (ör. "manga tarzı") oluşturma ve silme.
- **Bilgisayardan görsel alma:** klasörden (alt klasörler dahil) ya da dosya seçerek.
  - Aynı görseller içeriğinden tanınır ve bir kez eklenir. Bozuk ve küçük görseller atlanır.
  - Görseller varsayılan olarak `datasets\<ad>\images` içine kopyalanır. İstenirse yalnızca dosya konumu tutulur.
- **İnternet, iki ayrı anahtarla:**
  - 🔌 **Çevrimdışı mod** (varsayılan açık): bu bölüm internete hiç bağlanmaz.
  - 🌐 **İnternetten görsel ekle:** yalnızca yapıştırılan doğrudan görsel bağlantılarını indirir. Web sayfası taramaz. Ağı yormamak için tek bağlantı kullanır, istekler arasında en az 1 sn bekler, bir seferde en fazla 500 bağlantı ve görsel başına en fazla 25 MB indirir. Üst üste 5 bağlantı hatasında durur. İlk kullanımda görsel hakları onayı istenir.
- **Analiz ayarları:** hafif (stil kartı) / derin (LoRA eğitimi), temel model, süre ("ben durdurana kadar" ya da dakika), ara kayıt sıklığı, her ara kayıtta örnek görsel. Ayarlar kaydedilir. Analiz motorunun kendisi v2 ile gelecek.
- **Eklentiler (LoRA) klasörü:** `addons\`. Program buraya hiçbir şey indirmez. Kullanıcının koyduğu dosyaların lisans sorumluluğu kullanıcıdadır.

## Çalıştırma

```
python ai_local_image_studio.py
```

Programı başlatmak için bilgisayarda herhangi bir Python 3.10+ (python.org ya da Microsoft Store) yeterlidir. Programın kendi ortamı için ilk açılışta **programa özel Python 3.12** önerilir (bkz. v1.2.0).

## İlk açılışta ne olur?

1. **"Dosyalar nereye kaydedilsin?" penceresi açılır.** Diskler boş alanlarıyla listelenir, en çok yeri olan disk önerilir (ör. `D:\AI Local Image Studio`). Dil de bu pencerede seçilir.
2. Program o klasörde **kendi Python ortamını** kurar (`runtime\venv`). Önce programa özel Python 3.12 isteyip istemediğinizi sorar (önerilen, ~45 MB). Ardından küçük temel paketleri yükler (~40 MB). **Bilgisayardaki genel Python'a hiçbir şey kurulmaz.** Her adım `logs\setup.log` dosyasına yazılır.
3. Arayüz açılır. **Açılışta büyük bir indirme yapılmaz.**
4. İlk görseli ürettiğinde program neyin indirileceğini boyutu ve konumuyla tek bir onay penceresinde gösterir: görsel motoru (PyTorch + diffusers), seçtiğin model ve gerekiyorsa LLM. Onaylarsan devam eder. Aynı şey 6. bölümden tek tek de yapılabilir.

## Tek konum kuralı

Programın indirdiği her şey seçilen veri klasörüne gider:

```
D:\AI Local Image Studio\
  runtime\venv\     ← görsel motoru (PyTorch, diffusers…)
  runtime\python\   ← programa özel Python 3.12 (seçildiyse)
  runtime\tools\    ← uv aracı (programa özel Python'u kuran araç)
  models\           ← görsel modelleri, Real-ESRGAN ve yüz/el algılayıcı (models\detect)
  datasets\         ← veri setleri (görseller, video kareleri, açıklamalar, tür — dataset.json)
  addons\           ← kullanıcının koyduğu LoRA eklentileri (program indirmez)
  loras\            ← programın ürettiği LoRA'lar, kelime kartları ve örnek görseller
  feedback\         ← 👍 / 👎 puanları (ratings.jsonl)
  outputs\          ← üretilen görseller
  cache\            ← Hugging Face önbelleği
  tmp\              ← pip ve indirmelerin geçici dosyaları
  logs\ails.log     ← ayrıntılı log
  logs\setup.log    ← ilk kurulum günlüğü (pip'in tüm çıktısı)
  settings.json
```

- Program klasörüne sadece küçük bir `location.json` yazılır (veri klasörünün yeri). Program klasörü yazılamıyorsa bu dosya `%LOCALAPPDATA%` içine yazılır.
- Veri klasörünü değiştirmek için **6. bölüm → 📁 Veri Klasörü**. Program yeniden başlar ve tüm verileri ilerleme çubuğuyla taşır. Taşıma yarıda kalırsa eski klasör aynen kalır.
- **Ollama'nın kendi model konumu ayarına uyulur.** Ollama kapalıysa program `ollama serve`'ü değil Ollama uygulamasının kendisini (pencere açmadan) başlatır, böylece "Model location" ayarı geçerli olur. LLM modellerinin indiği klasör ve oradaki boş alan 4. bölümde görünür.
- Ollama hâlâ varsayılan C: konumundaysa ve veri klasörü başka bir diskteyse program bir kez "LLM modelleri de veri klasörüne gitsin mi?" diye sorar.
- Ollama hiç kurulu değilse ilk kullanımda (onayla) veri klasörüne kurulur.

## Kurulum güvenliği

- Kurulum ve indirmeden önce **disk alanı kontrolü** yapılır (motor için ~9 GB).
- **Disk dolu** ya da **internet yok** ise program hemen durur ve başka PyTorch sürümü denemez. Sonraki CUDA sürümü yalnızca "bu bilgisayara uygun paket yok" durumunda denenir.
- PyTorch kartın nesline ve sürücü sürümüne göre seçilir: RTX 20/30/40 için cu126 (sürücü 580 ve üzeriyse cu130), RTX 50 için cu128 ve üzeri.
- Motor her kurulumdan sonra **ayrı bir süreçte doğrulanır**. Ana program bozuk bir PyTorch'u asla yüklemez.
- Yarım kalmış veya bozuk kurulum otomatik temizlenir. Program çalışırken motor bozulursa açılışta sıfırlanır ve tek tıkla yeniden kurulur.
- v1.0.0'ın C:'ye bıraktığı bozuk PyTorch ve pip önbelleği bulunursa program bir kez silmeyi teklif eder.

## Ağdaki Ollama (4. bölüm → LLM sunucusu)

- Seçenekler: **Bu bilgisayar**, **ağda bulunan bilgisayarlar** veya **elle IP**. Seçim hatırlanır.
- LLM başka bilgisayarda çalışınca bu PC'nin VRAM'inin tamamı görsel modeline kalır.
- Ağdaki bilgisayardaki modeller boşaltılmaz. Orada eksik olan bir model indirilecekse önce sorulur.
- Ağdaki bilgisayara ulaşılamazsa iş çökmez: model yerel Ollama'da da varsa ona geçer, yoksa prompt olduğu gibi kullanılır.
- Diğer bilgisayarda yapılması gerekenler **❔** butonunda yazıyor: Ollama → Settings → "Expose Ollama to the network" ve güvenlik duvarında 11434 portu.

**İnterneti koparmayan tarama kuralları:**
- 254 adresin tamamı taranmaz. Sadece daha önce bilinen sunucular ve Windows'un zaten tanıdığı komşu cihazlar (ARP tablosu, en fazla 10) sırayla, aralarında beklenerek denenir.
- Tarama yalnızca "🔍 Ağda ara"ya basınca yapılır. Bir tek istisna var: yerelde Ollama yoksa açılışta bir kez.
- Ollama'ya giden tüm istekler tek bir kalıcı bağlantıdan geçer. Testte 20 istek tek bağlantı kullandı.
- Ulaşılamayan sunucu 2 → 4 → 8 … en fazla 60 saniye aralıklarla yeniden denenir.
- Yalnızca 192.168.x / 10.x / 172.16–31.x adreslerine bağlanılır.

## Bellek Koruyucu (v1.0.0'dan)

- Profiller: 🟢 Tam GPU / 🔵 Normal / 🟡 Dengeli / 🟠 Düşük VRAM / ⚪ Sadece CPU.
- RAM tavanı ve bekçi thread aktif. VRAM yetmezse alt profile inip aynı görseli tekrar dener.
- Yerel LLM ile görsel modeli hiçbir zaman aynı anda VRAM'de durmaz.

## Modeller (hepsi ticari kullanıma uygun, onaysız indirilebilir)

| Model | Lisans | Not |
|---|---|---|
| Stable Diffusion 1.5 | CreativeML OpenRAIL-M | 4 GB'ta rahat |
| Stable Diffusion XL 1.0 | CreativeML OpenRAIL++-M | 4 GB'ta yavaş, 8 GB+ rahat |
| Z-Image-Turbo | Apache-2.0 | En yüksek kalite, 16 GB+ rahat |
| FLUX.2 [klein] 4B | Apache-2.0 | Çok hızlı, 12–14 GB+ rahat |
| Qwen3 1.7B / 4B / 8B, Qwen2.5 7B (Ollama) | Apache-2.0 | Prompt çevirmeni |
| Real-ESRGAN x2plus | BSD-3-Clause | 2x büyütme |
| OpenCV (opencv-python-headless 4.14) | Apache-2.0 | Yüz/el düzeltici — algılama |
| YuNet yüz algılayıcı (opencv_zoo) | MIT | Yüz/el düzeltici |
| MediaPipe avuç içi algılayıcı (opencv_zoo) | Apache-2.0 | Yüz/el düzeltici |
| lbpcascade_animeface (nagadomi) | MIT | Anime/manga yüzleri |
| Florence-2 base (Microsoft, florence-community) | MIT | Görselleri okuma (açıklama) |
| PEFT (Hugging Face) | Apache-2.0 | LoRA yükleme ve eğitimi |
| uv (Astral) | MIT / Apache-2.0 | Programa özel Python'u kurar (isteğe bağlı) |
| CPython 3.12 (python-build-standalone) | PSF-2.0 | Programa özel Python (isteğe bağlı) |
| OpenCV (aynı paket) + içindeki FFmpeg | Apache-2.0 + LGPL-2.1 (dinamik) | Videodan kare çıkarma |

⚠️ OpenRAIL lisanslarının **kullanım kısıtlamaları** (Ek A) son kullanıcıya da geçer: bu modellerle görsel üreten herkes bu kısıtlamalara uymak zorundadır. Programın kaynak kodu **Apache-2.0** lisanslıdır (depodaki [LICENSE](../LICENSE)).

## 4 GB VRAM + 32 GB RAM test listesi

1. İlk açılış: konum penceresinde D: önerildi mi? Seçince program birkaç saniyede açıldı mı?
2. C:'deki boş alanı not et. Testlerin sonunda C: değişmemiş olmalı.
3. SD 1.5 ile ilk görsel. Onay penceresinde motor ve modelin D:'ye ineceği yazmalı. Log'da "✅ Görsel motoru kuruldu: … (CUDA)" görünmeli.
4. SDXL (🟠 Düşük VRAM): sistem donmadan, yavaş ama sonuna kadar gidiyor mu?
5. LLM açıkken log'da "LLM modelleri: D:\…" görünmeli. Qwen3 4B D:'deki Ollama klasörüne inmeli.
6. Başka bir bilgisayarda Ollama varsa: "🔍 Ağda ara" ya da elle IP. İnternet bağlantın kopmamalı.
7. "⬆️ 2x Büyüt", "Bittiğinde bilgisayarı kapat", 4 dil.
8. **v1.0.2 — yüz/el:** SD 1.5, Dengeli, insan olan bir prompt ("sahilde el sallayan bir kadın, tam boy").
   - İlk üretimde onay penceresinde "Yüz ve El Düzeltici (~50 MB)" görünmeli.
   - Log'da sırayla şu satırlar görünmeli: "Kalite: … DPM++ 2M Karras … ikinci geçiş: açık … yüz/el düzeltme: açık", "İkinci geçiş (hires fix): 768×768…", "Yüz/el düzeltici: N yüz, M el bulundu."
   - Gelişmiş Ayarlar'da "Düzeltme öncesi hâlini de kaydet"i açıp önce/sonra karşılaştırın.
9. SDXL + Ultra ve Z-Image ile yüz/el düzeltici. Sistem donmamalı. Bellek yetmezse log'da "ek geçiş atlandı" yazmalı ve görsel yine kaydedilmeli.
10. **Stil Öğrenme penceresi:** bir klasörden görsel alın. Aynı klasörü tekrar alınca hepsi "tekrar" sayılmalı. Çevrimdışı mod açıkken internet bölümü kapalı olmalı.
11. **v1.1.0 — görsel okuma + hafif mod:** 20–30 görsellik bir veri seti → mod "Hafif" → BAŞLAT.
    - Florence-2 onayı çıkmalı.
    - Log'da "Yazılan açıklama: …" ve "🎨 stil kartı kaydedildi" görünmeli.
    - 🧩 LoRA / Stil penceresinde "Stil kartları" filtresinde çıkmalı.
12. **Derin mod (SD 1.5):** Süre sınırı 15–30 dk ile başlatın.
    - Sistem donmamalı.
    - Her ara kayıtta örnek görsel galeriye düşmeli.
    - Durdur'a basınca son hâl kaydedilmeli.
    - LoRA seçme penceresinde "🧠 Program" olarak görünmeli.
13. **Ödül/ceza:** Aynı prompt'la 4–6 görsel üretip bazılarına 👍, bazılarına 👎 verin → mod "🧠 Kendi kendine öğrenme" → BAŞLAT (otomatik hakem açık).
    - "AILS tercih" LoRA'sı oluşmalı.
    - Sonraki SD 1.5 görsellerinde log'da "🧩 LoRA: AILS tercih …" görünmeli.

14. **v1.2.0 — sıfır PC:** Python'u yeni kurulmuş bir bilgisayarda programı açın.
    - "Programa özel Python 3.12" sorusu gelmeli. Evet → `runtime\python` oluşmalı, program açılmalı.
    - Hata olursa pencerede pip'in hata satırları görünmeli. `logs\setup.log` dosyasını gönderin.
15. **Video:** İçinde 1–2 vlog videosu ve birkaç fotoğraf olan bir klasör seçin. Tür: "🏙️ Mekân", etiket: "istanbul sokak".
    - Video sorusu çıkmalı. Evet → log'da "🎬 … N kare eklendi" görünmeli. Dosyalar `istanbul_sokak_0001.jpg` gibi adlandırılmalı.
    - Ardından açıklamalar kendiliğinden yazılmalı.
    - Aynı klasörü tekrar seçip videoya Hayır deyin: "video atlandı" yazmalı.
16. **Türler:** Hafif mod → "🏙️ … kelime kartı (Mekân)" yazmalı. Derin mod → tetikleyici `istanbul_sokak` olmalı (sonunda "style" yok). LoRA penceresinde "🏙️ Program · Mekân" görünmeli.
17. **v1.2.1 — mekân:** İnsanlı sokak görüntülerinden oluşan bir veri seti → tür "🏙️ Mekân", mod "Derin", "🙈 Yüzleri gizle" açık → BAŞLAT.
    - Log'da "🙈 Yüzler gizleniyor: … yüz bulundu" yazmalı.
    - Daha önce aynı etiketle stil LoRA'sı eğittiyseniz log'da "… korunuyor" yazmalı ve eski dosya yerinde durmalı.
18. **Sil:** LoRA penceresinde bir LoRA'yı silin. Dosya `loras\.trash\…` içinde durmalı.
19. **v1.3.0 — yüz modu:** Kendi yüzünüzün 15–30 fotoğrafı → tür "👤 Yüz", mod "Derin" → BAŞLAT.
    - İzin sorusu bir kez çıkmalı.
    - Log'da "👤 N görselde yüz bulunup kesildi" yazmalı. LoRA penceresinde "👤 Program · Yüz" görünmeli.

Sorun çıkarsa veri klasöründeki `logs\ails.log` (ilk kurulumda `logs\setup.log`) dosyasını gönderin.

## v1'in bilinen sınırları

- İndirme sırasında DURDUR'a basınca arayüz hemen serbest kalır, ancak o anki dosya arka planda tamamlanabilir. Yarım kalan indirme bir sonraki denemede kaldığı yerden devam eder.
- Paketleme (exe): exe'nin yanında gömülü bir Python gerekir (`python\python.exe`). Program kaynak koddan (`python ai_local_image_studio.py`) çalıştırılırken buna gerek yoktur.
- AMD/Intel ekran kartları v1'de CPU modunda çalışır.
- Programa özel Python, `uv` aracıyla GitHub'daki python-build-standalone sürümlerinden iner. İnternet ya da GitHub'a erişim yoksa program bilgisayardaki Python'la devam eder.
- Videodan kare çıkarma OpenCV'nin içindeki FFmpeg'i kullanır. Çok yeni ya da nadir codec'lerle kaydedilmiş bir video okunamazsa log'da "video okunamadı" yazar.
- Eğitim (derin mod ve ödül/ceza) bu sürümde yalnızca **SD 1.5** için. SDXL, Z-Image ve FLUX.2 eğitimi 4 GB'a sığmıyor. Bu modeller için dışarıdan eklenen LoRA'lar kullanılabilir.
- Otomatik hakem basit bir ölçüdür: bozuk yüz, bulanıklık, kapkara/bembeyaz gibi bariz hataları cezalandırır, "güzellik" ölçmez. En güçlü sinyal sizin 👍 / 👎 puanlarınızdır.
- Florence-2 için motorun güncel olması gerekir. Eski motorda program uyarır; 6. bölümden motor yeniden kurulabilir.
- Yüz/el düzeltici, algılayıcının yüz ya da el olarak görebildiği bölgeleri düzeltir. Çok bozuk bir el bazen hiç algılanmaz. El algılayıcı yanlışlıkla bir eşyayı el sanarsa oraya el çizmemesi için el bölgelerinde prompt'a "el" kelimesi eklenmez.
