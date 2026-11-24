# good_work — Sun'iy intellektni noldan o'rganish darslari

Bu repo — sun'iy intellektning ichida nima bo'layotganini tushunish uchun bosqichma-bosqich darslar to'plami.
Har bir dars avvalgisidan bitta yangi g'oyani qo'shadi, hammasi faqat `numpy` bilan (kutubxonalarsiz),
shuning uchun kod ichida hech narsa "sehr" bilan ishlamaydi — hammasi ko'rinib turadi.

## Darslar

1. **[`ai-darslar/01_neyron_tarmoq.py`](ai-darslar/01_neyron_tarmoq.py)** — Noldan neyron tarmoq.
   Tarmoq XOR amalini misollardan o'zi topadi. To'rt qadam: taxmin, xato, backpropagation, tuzatish.
2. **[`ai-darslar/02_til_modeli.py`](ai-darslar/02_til_modeli.py)** — Kichik til modeli.
   Harf darajasida ishlaydigan model o'zbekcha ismlardan o'rganib, yangi ismlar o'ylab topadi.
   Overfitting (yodlab olish) nima ekanini ham ko'rsatadi.

Ishga tushirish: `python3 ai-darslar/01_neyron_tarmoq.py` (yoki `02_...`). Kerakli kutubxona: `pip install numpy`.

2-dars o'qitilgan modelni `ai-darslar/model.npz` fayliga saqlaydi va keyingi safar qayta o'qitmasdan
yuklaydi. `--qayta` bilan qaytadan o'qitish, `--soni 50` bilan ko'proq ism yozdirish mumkin.

### Google Colab'da ishlatish

```python
!git clone https://github.com/samadbekxolmamatov2009/good_work
%cd good_work
!python ai-darslar/02_til_modeli.py --soni 10
```

Colab sessiyasi yopilganda `model.npz` ham o'chadi (repo'dagi nusxasidan tashqari). Doimiy saqlash uchun
Google Drive ulang va `--fayl /content/drive/MyDrive/model.npz` bering.

## Keyingi darslar (rejada)

Qaysi birini keyin qurishni session ichida tanlaymiz. G'oyalar:

- Rasmlarni taniydigan tarmoq (qo'lda chizilgan raqamlar)
- E'tibor mexanizmi (attention) — transformer'ning yuragi
- Mustahkamlovchi o'rganish (reinforcement learning) — labirintdan chiqishni o'rganuvchi agent
- So'z vektorlari (embeddings) — so'zlar orasidagi o'xshashlik
