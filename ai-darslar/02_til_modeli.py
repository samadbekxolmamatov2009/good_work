"""
2-dars: Kichik til modeli - o'zbekcha ismlar o'ylab topadi (faqat numpy)

G'oya: model oldingi 3 ta harfga qarab KEYINGI harfni taxmin qiladi.
    "..." -> "s"
    "..s" -> "a"
    ".sa" -> "r"
    "sar" -> "d"   va hokazo, oxirida "." (ism tugadi)

ChatGPT va Claude ham xuddi shu ishni qiladi, faqat harf o'rniga so'z
bo'laklari (token) bilan va 3 ta emas, minglab oldingi tokenga qarab.

1-darsdagi 4 qadam o'zgarmaydi: taxmin -> xato -> backpropagation -> tuzatish.
Yangi narsalar:
    - embedding: har bir harf sonlar ro'yxatiga (vektorga) aylanadi
    - softmax:   chiqishda har bir harf uchun ehtimollik
    - sampling:  ehtimollikka qarab harf tanlab, yangi ism yozish

Saqlash: o'qitishdan keyin model "bilimi" (og'irliklari) model.npz fayliga yoziladi.
Keyingi ishga tushirishda fayl topilsa, model qayta o'qitilmaydi, to'g'ridan-to'g'ri
fayldan yuklanadi.

Ishga tushirish:
    python3 ai-darslar/02_til_modeli.py                  # bor bo'lsa yuklaydi, yo'q bo'lsa o'qitib saqlaydi
    python3 ai-darslar/02_til_modeli.py --qayta          # saqlanganini e'tiborsiz qoldirib, qaytadan o'qitadi
    python3 ai-darslar/02_til_modeli.py --fayl YO'L.npz  # modelni boshqa joyga saqlash/yuklash
    python3 ai-darslar/02_til_modeli.py --soni 50        # 50 ta yangi ism yozdirish
"""

import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--qayta", action="store_true", help="saqlangan modelni ishlatmay, qaytadan o'qitish")
parser.add_argument("--fayl", default=str(Path(__file__).with_name("model.npz")), help="model fayli yo'li")
parser.add_argument("--soni", type=int, default=20, help="nechta yangi ism yozdirish")
args, _ = parser.parse_known_args()   # Colab/Jupyter o'z argumentlarini qo'shsa ham xato bermaydi
MODEL_FAYL = Path(args.fayl)

rng = np.random.default_rng(42)

# --- 1. Ma'lumot ---
ismlar = Path(__file__).with_name("ismlar.txt").read_text(encoding="utf-8").split()
ismlar = sorted(set(ismlar))

# Lug'at: har bir belgiga raqam beramiz. "." - ism boshi va oxiri belgisi.
belgilar = ["."] + sorted(set("".join(ismlar)))
raqam = {b: i for i, b in enumerate(belgilar)}
LUGAT = len(belgilar)

KONTEKST = 3  # model nechta oldingi harfni ko'radi


def misollar_yasash(royxat):
    """Har bir ismni (oldingi 3 harf -> keyingi harf) juftliklariga bo'ladi."""
    X, Y = [], []
    for ism in royxat:
        oyna = [0] * KONTEKST
        for b in ism + ".":
            X.append(oyna)
            Y.append(raqam[b])
            oyna = oyna[1:] + [raqam[b]]
    return np.array(X), np.array(Y)


# Ismlarning 10 foizini model HECH QACHON ko'rmaydi. Ular bilan modelni
# tekshiramiz: u yodlab oldimi yoki qonuniyatni chindan o'rgandimi?
aralash = list(ismlar)
rng.shuffle(aralash)
chegara = int(len(aralash) * 0.9)
X_oqit, Y_oqit = misollar_yasash(aralash[:chegara])
X_tekshir, Y_tekshir = misollar_yasash(aralash[chegara:])

# --- 2. Model ---
EMB = 6        # har bir harf nechta son bilan ifodalanadi
YASHIRIN = 24  # yashirin qatlamdagi neyronlar soni

C = rng.normal(0, 1, (LUGAT, EMB))                    # embedding jadvali
W1 = rng.normal(0, 1, (KONTEKST * EMB, YASHIRIN)) * 0.2
b1 = np.zeros(YASHIRIN)
W2 = rng.normal(0, 1, (YASHIRIN, LUGAT)) * 0.01
b2 = np.zeros(LUGAT)
parametrlar = [C, W1, b1, W2, b2]


def taxmin(X):
    """Kirish: harf raqamlari. Chiqish: keyingi harf uchun ehtimolliklar."""
    emb = C[X].reshape(len(X), -1)          # harflar -> vektorlar
    h = np.tanh(emb @ W1 + b1)              # yashirin qatlam
    z = h @ W2 + b2
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)       # softmax: yig'indisi 1 bo'ladi
    return emb, h, p


def xato(X, Y):
    """To'g'ri harfga model qancha kam ehtimol bergan bo'lsa, xato shuncha katta."""
    p = taxmin(X)[2]
    return -np.log(p[np.arange(len(Y)), Y]).mean()


def ism_yoz(harorat=1.0):
    """Model o'zi harfma-harf ism yozadi."""
    oyna, natija = [0] * KONTEKST, ""
    while len(natija) < 20:
        p = taxmin(np.array([oyna]))[2][0]
        if harorat != 1.0:
            p = p ** (1 / harorat)
            p /= p.sum()
        i = rng.choice(LUGAT, p=p)
        if i == 0:
            break
        natija += belgilar[i]
        oyna = oyna[1:] + [i]
    return natija


soni = sum(p.size for p in parametrlar)
print(f"Ismlar: {len(ismlar)} ta   |   belgilar: {LUGAT} ta   |   misollar: {len(X_oqit)} ta")
print(f"Model og'irliklari: {soni} ta son\n")


def saqlash():
    """Modelning bilimini (og'irliklarini) va lug'atini faylga yozadi."""
    MODEL_FAYL.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_FAYL, belgilar=np.array(belgilar), C=C, W1=W1, b1=b1, W2=W2, b2=b2)


def yuklash():
    """Saqlangan modelni o'qiydi. Muvaffaqiyatli bo'lsa True qaytaradi.
    Lug'at yoki o'lchamlar hozirgi kodga mos kelmasa (masalan, ismlar.txt o'zgargan bo'lsa),
    eski model yaroqsiz - False qaytaradi va qaytadan o'qitiladi."""
    if not MODEL_FAYL.exists():
        return False
    d = np.load(MODEL_FAYL)
    if list(d["belgilar"]) != belgilar or any(d[k].shape != par.shape for k, par in zip(["C", "W1", "b1", "W2", "b2"], parametrlar)):
        print(f"{MODEL_FAYL} hozirgi ma'lumotga mos kelmadi, qaytadan o'qitamiz.\n")
        return False
    for k, par in zip(["C", "W1", "b1", "W2", "b2"], parametrlar):
        par[...] = d[k]       # joyida almashtiramiz, shunda taxmin() ham yangi sonlarni ko'radi
    return True


yuklandi = (not args.qayta) and yuklash()

if yuklandi:
    print(f"Saqlangan model yuklandi: {MODEL_FAYL}")
    print("O'qitish o'tkazib yuborildi.")
    print(f"Hozirgi xato:  o'qitish = {xato(X_oqit, Y_oqit):.3f}   tekshiruv = {xato(X_tekshir, Y_tekshir):.3f}")
else:
    print(f"Tasodifiy taxmin qilgandagi xato: {np.log(LUGAT):.3f}")
    print(f"O'qitishdan oldingi xato:         {xato(X_oqit, Y_oqit):.3f}")
    print("O'qitishdan OLDIN yozgan 'ismlari':", ", ".join(ism_yoz() for _ in range(5)))
    print()

    # --- 3. O'qitish ---
    QADAMLAR = 5000
    TOPLAM = 32  # har qadamda nechta misolga qaraymiz

    for qadam in range(QADAMLAR + 1):
        idx = rng.integers(0, len(X_oqit), TOPLAM)
        Xb, Yb = X_oqit[idx], Y_oqit[idx]

        # 1-2. Taxmin va xato
        emb, h, p = taxmin(Xb)

        # 3. Backpropagation
        dz = p.copy()
        dz[np.arange(TOPLAM), Yb] -= 1
        dz /= TOPLAM
        dW2 = h.T @ dz
        db2 = dz.sum(axis=0)
        dh = dz @ W2.T * (1 - h**2)
        dW1 = emb.T @ dh
        db1 = dh.sum(axis=0)
        demb = (dh @ W1.T).reshape(TOPLAM, KONTEKST, EMB)
        dC = np.zeros_like(C)
        np.add.at(dC, Xb, demb)

        # 4. Tuzatish
        tezlik = 0.1
        for par, grad in zip(parametrlar, [dC, dW1, db1, dW2, db2]):
            par -= tezlik * grad

        if qadam % 1000 == 0:
            print(
                f"qadam {qadam:>5}   o'qitish xatosi = {xato(X_oqit, Y_oqit):.3f}"
                f"   tekshiruv xatosi = {xato(X_tekshir, Y_tekshir):.3f}"
            )

    saqlash()
    print(f"\nModel saqlandi: {MODEL_FAYL}")

# --- 4. Natija ---
print("\nO'qitishdan KEYIN model yozgan ismlar:")
mavjud = set(ismlar)
yozilgan = []
while len(yozilgan) < args.soni:
    ism = ism_yoz(harorat=0.8)
    if len(ism) >= 3 and ism not in yozilgan:
        yozilgan.append(ism)
for ism in yozilgan:
    izoh = "(ro'yxatda bor - yodlagan)" if ism in mavjud else "(yangi - o'zi o'ylab topdi)"
    print(f"  {ism.capitalize():<16}{izoh}")
yangi = sum(ism not in mavjud for ism in yozilgan)
print(f"\n{args.soni} tadan {yangi} tasi yangi ism.")
