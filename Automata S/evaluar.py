
import argparse
import time

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import automata_sentimientos as A


def cargar_kaggle(ruta, muestra=100_000):
    """Lee el CSV de Kaggle y lo prepara igual que el notebook de Colab.

    Devuelve dos listas: los textos de las resenas y su etiqueta (1 = Fresh, 0 = Rotten).
    """
    df = pd.read_csv(ruta, usecols=["review_content", "review_type"])
    df = df.dropna(subset=["review_content"]).drop_duplicates(subset=["review_content"])
    df["sentiment"] = df["review_type"].map({"Fresh": 1, "Rotten": 0})
    if muestra and len(df) > muestra:
        df, _ = train_test_split(df, train_size=muestra, stratify=df["sentiment"], random_state=42)
    return df["review_content"], df["sentiment"]


def cargar_rt_polarity(pos, neg):
    """Lee los dos archivos de frases (positivas y negativas) y les pone etiqueta 1 o 0."""
    def leer(p):
        with open(p, encoding="latin-1") as f:
            return [l.strip() for l in f if l.strip()]
    P, N = leer(pos), leer(neg)
    return pd.Series(P + N), pd.Series([1] * len(P) + [0] * len(N))


def predecir(textos, workers, negacion, lexico):
    """Le pide al automata (en paralelo) que clasifique cada texto.

    El automata tiene 3 respuestas (positivo, negativo, neutral) pero las
    etiquetas reales solo tienen 2. Por eso, "neutral" se cuenta como Fresh
    (la clase que mas abunda en los datos).
    Devuelve (predicciones_0_o_1, respuestas_originales_del_automata).
    """
    res = list(A.granja(list(enumerate(textos)), workers=workers, negacion=negacion, lexico=lexico))
    etiqueta = [r["sentimiento"] if r["valida"] else "neutral" for r in res]
    y_pred = [0 if e == "negativo" else 1 for e in etiqueta]
    return y_pred, etiqueta


def reporte(nombre, y_true, y_pred, etiquetas):
    """Imprime los resultados: precision, reporte por clase y matriz de confusion.

    La matriz de confusion muestra los aciertos y errores:
      [[negativas bien, negativas mal]
       [positivas mal,  positivas bien]]
    """
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    neutros = etiquetas.count("neutral") / len(etiquetas)
    print(f"\n===== {nombre} =====")
    print(f"Precision: {acc:.4f} ({acc*100:.2f} %)   |   respuestas 'neutral': {neutros*100:.1f} %")
    print(classification_report(y_true, y_pred, target_names=["Negativas", "Positivas"], digits=2))
    print("Matriz de confusion [[TN FP] [FN TP]]:\n", confusion_matrix(y_true, y_pred))
    return acc


def main():
    """Carga los datos, los divide, y prueba 4 versiones del automata."""
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", help="rotten_tomatoes_critic_reviews.csv de Kaggle")
    g.add_argument("--rt-polarity", nargs=2, metavar=("POS", "NEG"))
    ap.add_argument("--muestra", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--umbral", type=float, default=0.6)
    ap.add_argument("--max-palabras", type=int, default=3000)
    ap.add_argument("--guardar-lexico", help="guarda el diccionario aprendido en este archivo JSON")
    a = ap.parse_args()

    # 1) cargar datos y dividir en entrenamiento (80%) y prueba (20%)
    X, y = cargar_kaggle(a.csv, a.muestra) if a.csv else cargar_rt_polarity(*a.rt_polarity)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"Entrenamiento: {len(X_tr)} | Prueba: {len(X_te)}")
    y_te = list(y_te)

    # 2) probar el automata con el diccionario hecho a mano
    t0 = time.time()
    p0, e0 = predecir(X_te, a.workers, False, None)
    reporte("Automata con diccionario base (manual)", y_te, p0, e0)
    p1, e1 = predecir(X_te, a.workers, True, None)
    reporte("Automata con diccionario base + negacion", y_te, p1, e1)

    # 3) aprender un diccionario del set de entrenamiento y probarlo en el set de prueba
    lex = A.entrenar_lexico(X_tr, y_tr, a.min_freq, a.umbral, a.max_palabras)
    print(f"\nDiccionario aprendido del set de entrenamiento: {len(lex[0])} positivas, {len(lex[1])} negativas")
    p2, e2 = predecir(X_te, a.workers, False, lex)
    reporte("Automata con diccionario aprendido", y_te, p2, e2)
    p3, e3 = predecir(X_te, a.workers, True, lex)
    reporte("Automata con diccionario aprendido + negacion", y_te, p3, e3)
    print(f"\nTiempo total: {time.time()-t0:.1f}s")
    print("(Referencia: TF-IDF + Regresion Logistica del notebook = 78.47 %)")
    if a.guardar_lexico:
        A.guardar_lexico(a.guardar_lexico, lex)
        print("Diccionario guardado en", a.guardar_lexico)


if __name__ == "__main__":
    main()
