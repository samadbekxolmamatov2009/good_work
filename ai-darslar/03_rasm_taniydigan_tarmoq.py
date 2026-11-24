"""
3-dars: Rasm taniydigan tarmoq - qo'lda yozilgan raqamlarni tanish (faqat numpy)

Ma'lumot: 1797 ta qo'lda yozilgan raqam (0-9), har biri 8x8 piksel (raqamlar.npz).
Manba: UCI "Optical Recognition of Handwritten Digits" (E. Alpaydin, C. Kaynak), CC BY 4.0.

Ikkita tarmoqni solishtiramiz:

  A) Oddiy tarmoq (1-dars kabi): rasmni 64 ta sonlar qatoriga yoyib, har bir pikselga alohida qaraydi.
     Muammo: u pikselning QAYERDA turganini yodlaydi. Raqam bir piksel surilsa, deyarli yangi rasm.

  B) Convolution (burama) tarmoq: kichik 3x3 "ko'zoynak" (filtr) butun rasm bo'ylab suriladi.
     Filtr masalan "to'g'ri chiziq" yoki "burchak"ni qidiradi, va u qayerda bo'lsa ham topadi.
     Keyin max-pooling: har 2x2 joydan eng kuchli javobni oladi - bu kichik surilishlarga
     biroz chidamlilik beradi.

Yangi g'oya (1-2 darslarga nisbatan): filtr og'irliklari rasmning HAR BIR joyida takror ishlatiladi.
Shuning uchun parametrlar kam, lekin rasm tuzilishini tushunadi. Zamonaviy rasm AI'larining asosi.

Backpropagation qadamlari o'sha-o'sha: taxmin -> xato -> aybdorni topish -> tuzatish.

Halol xulosa (o'lchab ko'rilgan): oddiy rasmlarda ikkalasi ~97-98% beradi. Rasm 1 piksel surilsa
ikkalasi ham ancha yiqiladi (oddiy ~40%, convolution ~55%): convolution yaxshiroq, lekin sehr emas.
Asosiy davo - modelga surilgan nusxalarni ham ko'rsatish (--surish). Shunda ikkalasi ham ~93%+ ga qaytadi.
AI'da "ma'lumot ko'rsatish" ko'pincha tarmoq tuzilishidan ham muhimroq.

Ishga tushirish:
    python3 ai-darslar/03_rasm_taniydigan_tarmoq.py            # saqlangan model bo'lsa yuklaydi
    python3 ai-darslar/03_rasm_taniydigan_tarmoq.py --qayta    # qaytadan o'qitadi
    python3 ai-darslar/03_rasm_taniydigan_tarmoq.py --surish   # surilgan rasmlar bilan o'qitadi (augmentation)
    python3 ai-darslar/03_rasm_taniydigan_tarmoq.py --fayl YO'L.npz
"""

import argparse
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

parser = argparse.ArgumentParser()
parser.add_argument("--qayta", action="store_true", help="saqlangan modelni ishlatmay, qaytadan o'qitish")
parser.add_argument("--surish", action="store_true", help="o'qitishda rasmlarni tasodifiy 1 piksel surish (augmentation)")
parser.add_argument("--fayl", default=None, help="model fayli yo'li")
args, _ = parser.parse_known_args()
MODEL_FAYL = Path(args.fayl) if args.fayl else Path(__file__).with_name("model_rasm_surish.npz" if args.surish else "model_rasm.npz")

rng = np.random.default_rng(0)

# --- 1. Ma'lumot ---
d = np.load(Path(__file__).with_name("raqamlar.npz"))
X_hammasi = d["X"].astype(float) / 16.0     # piksellar 0..1 oralig'ida
Y_hammasi = d["y"].astype(int)
tartib = rng.permutation(len(X_hammasi))
X_hammasi, Y_hammasi = X_hammasi[tartib], Y_hammasi[tartib]
chegara = int(len(X_hammasi) * 0.8)
X_oqit, Y_oqit = X_hammasi[:chegara], Y_hammasi[:chegara]
X_tekshir, Y_tekshir = X_hammasi[chegara:], Y_hammasi[chegara:]   # model bularni o'qitishda KO'RMAYDI


def surish(X, dx):
    """Rasmlarni dx piksel o'ngga suradi (bo'sh joy 0 bilan to'ladi)."""
    out = np.zeros_like(X)
    if dx > 0:
        out[:, :, dx:] = X[:, :, :-dx]
    elif dx < 0:
        out[:, :, :dx] = X[:, :, -dx:]
    else:
        out = X.copy()
    return out


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def xato_va_gradient(p, Y):
    """Cross-entropy xatosi va uning softmax'dan oldingi gradienti."""
    n = len(Y)
    xato = -np.log(p[np.arange(n), Y] + 1e-12).mean()
    dz = p.copy()
    dz[np.arange(n), Y] -= 1
    return xato, dz / n


# --- 2A. Oddiy tarmoq: 64 piksel -> 32 neyron -> 10 raqam ---
A = {
    "W1": rng.normal(0, 1, (64, 32)) * np.sqrt(2 / 64), "b1": np.zeros(32),
    "W2": rng.normal(0, 1, (32, 10)) * np.sqrt(2 / 32), "b2": np.zeros(10),
}


def A_oldinga(X):
    x = X.reshape(len(X), -1)
    h = np.maximum(0, x @ A["W1"] + A["b1"])        # ReLU: manfiy sonlarni 0 qiladi
    return x, h, softmax(h @ A["W2"] + A["b2"])


def A_qadam(X, Y, tezlik):
    x, h, p = A_oldinga(X)
    xato, dz = xato_va_gradient(p, Y)
    dW2 = h.T @ dz
    db2 = dz.sum(0)
    dh = (dz @ A["W2"].T) * (h > 0)
    dW1 = x.T @ dh
    db1 = dh.sum(0)
    for k, g in (("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2)):
        A[k] -= tezlik * g
    return xato


# --- 2B. Convolution tarmog'i: 8 ta 3x3 filtr -> ReLU -> max-pool -> 10 raqam ---
FILTR = 8
B = {
    "K": rng.normal(0, 1, (FILTR, 3, 3)) * np.sqrt(2 / 9), "bk": np.zeros(FILTR),
    "W": rng.normal(0, 1, (FILTR * 9, 10)) * np.sqrt(2 / (FILTR * 9)), "b": np.zeros(10),
}


def B_oldinga(X):
    n = len(X)
    yamoq = sliding_window_view(X, (3, 3), axis=(1, 2))               # (n, 6, 6, 3, 3): har bir 3x3 joy
    c = np.einsum("nijab,fab->nfij", yamoq, B["K"]) + B["bk"][None, :, None, None]   # filtr javobi (n, F, 6, 6)
    r = np.maximum(0, c)                                              # ReLU
    bo = r.reshape(n, FILTR, 3, 2, 3, 2)                              # 2x2 bloklarga bo'lamiz
    pool = bo.max(axis=(3, 5))                                        # max-pool (n, F, 3, 3)
    flat = pool.reshape(n, -1)
    return yamoq, c, bo, pool, flat, softmax(flat @ B["W"] + B["b"])


def B_qadam(X, Y, tezlik):
    n = len(X)
    yamoq, c, bo, pool, flat, p = B_oldinga(X)
    xato, dz = xato_va_gradient(p, Y)
    dW = flat.T @ dz
    db = dz.sum(0)
    dpool = (dz @ B["W"].T).reshape(n, FILTR, 3, 3)
    # max-pool orqaga: gradient faqat g'olib (eng katta) pikselga boradi
    maska = (bo == pool[:, :, :, None, :, None])
    maska = maska / maska.sum(axis=(3, 5), keepdims=True)             # tenglik bo'lsa teng bo'lamiz
    dr = (maska * dpool[:, :, :, None, :, None]).reshape(n, FILTR, 6, 6)
    dc = dr * (c > 0)                                                 # ReLU orqaga
    dK = np.einsum("nfij,nijab->fab", dc, yamoq)                      # filtr gradienti: barcha joylardan yig'iladi
    dbk = dc.sum(axis=(0, 2, 3))
    for k, g in (("K", dK), ("bk", dbk), ("W", dW), ("b", db)):
        B[k] -= tezlik * g
    return xato


def aniqlik(oldinga, X, Y):
    return (oldinga(X)[-1].argmax(axis=1) == Y).mean()


def soni(model):
    return sum(v.size for v in model.values())


def saqlash():
    MODEL_FAYL.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_FAYL, **{"A_" + k: v for k, v in A.items()}, **{"B_" + k: v for k, v in B.items()})


def yuklash():
    if not MODEL_FAYL.exists():
        return False
    f = np.load(MODEL_FAYL)
    kalitlar = [("A_" + k, A[k]) for k in A] + [("B_" + k, B[k]) for k in B]
    if any(k not in f or f[k].shape != v.shape for k, v in kalitlar):
        print(f"{MODEL_FAYL} hozirgi tuzilishga mos kelmadi, qaytadan o'qitamiz.\n")
        return False
    for k, v in kalitlar:
        v[...] = f[k]
    return True


print(f"Rasmlar: {len(X_hammasi)} ta (o'qitish: {len(X_oqit)}, tekshiruv: {len(X_tekshir)})   |   hajmi 8x8 piksel")
print(f"A) Oddiy tarmoq parametrlari:       {soni(A):>5} ta")
print(f"B) Convolution tarmoq parametrlari: {soni(B):>5} ta   (kamroq, lekin rasm tuzilishini tushunadi)\n")

if (not args.qayta) and yuklash():
    print(f"Saqlangan model yuklandi: {MODEL_FAYL}\nO'qitish o'tkazib yuborildi.\n")
else:
    # --- 3. O'qitish ---
    print(f"{'davr':>5} | {'A xato':>7} {'A aniqlik':>10} | {'B xato':>7} {'B aniqlik':>10}   (aniqlik tekshiruv rasmlarida)")
    DAVR, TOPLAM, TEZLIK = 40, 32, 0.1
    for davr in range(1, DAVR + 1):
        tartib = rng.permutation(len(X_oqit))
        xa = xb = 0.0
        for i in range(0, len(tartib), TOPLAM):
            j = tartib[i:i + TOPLAM]
            xj = X_oqit[j]
            if args.surish:   # har bir rasmni tasodifiy -1, 0 yoki +1 piksel suramiz
                dx = rng.integers(-1, 2, len(j))
                xj = xj.copy()
                for d_ in (-1, 1):
                    xj[dx == d_] = surish(xj[dx == d_], d_)
            xa += A_qadam(xj, Y_oqit[j], TEZLIK) * len(j)
            xb += B_qadam(xj, Y_oqit[j], TEZLIK) * len(j)
        if davr in (1, 2, 5, 10, 20, 30, 40):
            print(f"{davr:>5} | {xa / len(X_oqit):>7.3f} {aniqlik(A_oldinga, X_tekshir, Y_tekshir):>9.1%} "
                  f"| {xb / len(X_oqit):>7.3f} {aniqlik(B_oldinga, X_tekshir, Y_tekshir):>9.1%}")
    saqlash()
    print(f"\nModel saqlandi: {MODEL_FAYL}\n")

# --- 4. Natija: oddiy tekshiruv ---
print("Hech qachon ko'rilmagan rasmlarda aniqlik:")
print(f"  A) Oddiy tarmoq:       {aniqlik(A_oldinga, X_tekshir, Y_tekshir):.1%}")
print(f"  B) Convolution tarmoq: {aniqlik(B_oldinga, X_tekshir, Y_tekshir):.1%}")

# --- 5. Qiyin sinov: rasmlarni 1 piksel suramiz ---
print("\nENDI QIYIN SINOV: o'sha rasmlarni 1 piksel o'ngga surib ko'rsatamiz (odam uchun hech narsa o'zgarmaydi):")
for dx in (1, -1):
    Xs = surish(X_tekshir, dx)
    print(f"  {dx:+d} piksel:  A) oddiy {aniqlik(A_oldinga, Xs, Y_tekshir):>6.1%}    B) convolution {aniqlik(B_oldinga, Xs, Y_tekshir):>6.1%}")
if not args.surish:
    print("  Ikkalasi ham yiqildi, chunki o'qitishda surilgan rasmlarni hech ko'rmagan.")
    print("  Davo: --surish bilan ishga tushiring (--qayta kerak emas, alohida faylga saqlanadi).")
else:
    print("  Bu model o'qitishda surilgan nusxalarni ham ko'rgan, shuning uchun surilishga chidamli.")

# --- 6. Ko'z bilan ko'rish ---
BELGI = " .:-=+*#%@"


def chiz(rasm):
    return ["".join(BELGI[min(9, int(v * 10))] * 2 for v in qator) for qator in rasm]


print("\nTekshiruv rasmlaridan namunalar (tarmoq B taxmini):")
namuna = rng.choice(len(X_tekshir), 5, replace=False)
pred = B_oldinga(X_tekshir[namuna])[-1]
satrlar = [chiz(X_tekshir[i]) for i in namuna]
for q in range(8):
    print("   ".join(s[q] for s in satrlar))
print("   ".join(f"{'taxmin: ' + str(pred[k].argmax()):<16}" for k in range(5)))
print("   ".join(f"{'ishonch: ' + format(pred[k].max(), '.0%'):<16}" for k in range(5)))
print("   ".join(f"{'haqiqiy: ' + str(Y_tekshir[i]):<16}" for i in namuna))

xato_idx = np.where(B_oldinga(X_tekshir)[-1].argmax(1) != Y_tekshir)[0]
print(f"\nTarmoq B xato qilgan rasmlar: {len(xato_idx)} ta ({len(X_tekshir)} tadan).", end="")
if len(xato_idx):
    k = xato_idx[0]
    print(f" Masalan, haqiqiy {Y_tekshir[k]}, taxmin {B_oldinga(X_tekshir[k:k + 1])[-1].argmax()}:")
    for q in chiz(X_tekshir[k]):
        print("   " + q)
else:
    print()

print("\nTarmoq B o'zi o'rgangan 3x3 filtrlar (+ yorqin joyni, - qorong'i joyni qidiradi):")
for f in range(FILTR):
    print(f"  filtr {f + 1}: " + "  |  ".join(" ".join(f"{v:+.1f}" for v in qator) for qator in B["K"][f]))
