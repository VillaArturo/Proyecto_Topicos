#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from functools import lru_cache, partial
from itertools import islice
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# DICCIONARIOS
# Listas de palabras buenas (POSITIVAS) y malas (NEGATIVAS). Se pueden ampliar.
# Todo va en minusculas y en ingles porque las resenas estan en ingles.
# ---------------------------------------------------------------------------
POSITIVAS = frozenset("""
good great excellent brilliant funny charming moving powerful stunning beautiful
masterpiece delightful entertaining engaging fun wonderful terrific superb
compelling clever smart witty touching gripping thrilling impressive best love
loved loves enjoyable fresh original memorable solid strong perfect hilarious
sharp rich heartfelt inventive riveting captivating triumph gem outstanding
fantastic amazing awesome remarkable satisfying sweet warm vibrant polished
effective worth recommend recommended dazzling haunting absorbing genuine
winning lively splendid sublime stellar exhilarating refreshing rewarding
gorgeous spectacular elegant enjoy enjoyed fine nice
""".split())

NEGATIVAS = frozenset("""
bad worst boring dull tedious awful terrible poor weak flat bland mess messy
disappointing disappointment predictable clumsy stale forgettable lame silly
pointless painful ugly annoying cliched derivative shallow stupid
dreadful horrible mediocre lifeless overlong bloated tiresome unfunny incoherent
confusing hollow trite tired failure fails failed fail waste wasted misfire dud
insipid laughable ridiculous sloppy uninspired unconvincing offensive irritating
plodding pretentious lackluster rubbish trash junk hate hated worse slow cheap
cringe pathetic tepid empty ponderous charmless joyless unwatchable atrocious
abysmal nonsense formulaic
""".split())

# Palabras vacias: no dicen nada sobre el sentimiento, asi que se tiran.
# OJO: aqui NO van las negaciones (not, no, never...) porque cambian el sentido.
STOP_WORDS = frozenset("""
a an the and or but if while of at by for with about against between into through
during before after above below to from up down in out on off over under again
further then once here there when where why how all any both each few more most
other some such only own same so than too very can will just should now i me my
myself we our ours you your yours he him his she her hers it its they them their
theirs what which who whom this that these those am is are was were be been being
have has had having do does did doing would could film movie s t
""".split())

# Palabras de negacion (se usan solo si se activa la opcion de negacion).
NEGACIONES = frozenset("not no never neither nor nothing nobody hardly without cannot".split())


def es_negacion(token: str) -> bool:
    """Dice si una palabra es una negacion ("not", "never", "isn't", "don't"...)."""
    return token in NEGACIONES or token.endswith("n't")


# ---------------------------------------------------------------------------
# 1) RECEPCION DE LA RESENA
# ---------------------------------------------------------------------------
def recibir_resena(crudo) -> Optional[str]:
    """Primer paso: revisa que la resena tenga caracteres validos.

    - Quita caracteres raros (de control o el simbolo de "caracter danado").
    - Si la resena esta vacia o es None, devuelve None (resena invalida).
    - Si esta bien, devuelve el texto listo para limpiar.
    """
    if crudo is None:
        return None
    if not isinstance(crudo, str):
        crudo = str(crudo)
    # \ufffd es el simbolo de "caracter danado" que sale cuando se lee mal un archivo
    texto = "".join(
        c for c in crudo
        if c in "\n\t" or (unicodedata.category(c)[0] != "C" and c != "\ufffd")
    )
    return texto if texto.strip() else None


# ---------------------------------------------------------------------------
# 2) LIMPIEZA DEL TEXTO
# Es un "transductor": lee un caracter a la vez y decide que escribir a la salida.
# Tiene solo 2 estados: TEXTO (normal) y ETIQUETA (dentro de <...>).
# ---------------------------------------------------------------------------
class EstadoLimpieza(Enum):
    TEXTO = "TEXTO"          # leyendo texto normal
    ETIQUETA = "ETIQUETA"    # leyendo dentro de una etiqueta HTML como <b>


# Todos los tipos de apostrofe que existen. \u2019 y \u2018 son las comillas
# "curvas" y \u00b4 es el acento agudo; todos se convierten en un apostrofe normal.
_APOSTROFES = "'\u2019\u2018`\u00b4"


def _espacio(salida: List[str]) -> None:
    """Agrega un espacio a la salida, pero nunca dos seguidos."""
    if salida and salida[-1] != " ":
        salida.append(" ")


def _transductor_limpieza(texto: str) -> str:
    """Recorre el texto caracter por caracter y construye el texto limpio.

    Reglas:
      - Dentro de una etiqueta HTML (<...>) no se escribe nada.
      - Las letras se escriben en minuscula.
      - Los apostrofes se dejan (para palabras como "don't").
      - Todo lo demas (numeros, simbolos, signos) se vuelve un espacio.
    """
    salida: List[str] = []
    estado = EstadoLimpieza.TEXTO
    inicio_etiqueta = 0
    n = len(texto)
    for i, c in enumerate(texto):
        if estado is EstadoLimpieza.TEXTO:
            # Un "<" abre etiqueta solo si le sigue una letra, "/" o "!".
            # Asi "I <3 it" no se toma como HTML.
            if c == "<" and i + 1 < n and (texto[i + 1].isalpha() or texto[i + 1] in "/!"):
                estado = EstadoLimpieza.ETIQUETA
                inicio_etiqueta = i
            elif c.isalpha():
                salida.append(c.lower())
            elif c in _APOSTROFES:
                salida.append("'")
            else:
                _espacio(salida)
        else:  # estamos dentro de una etiqueta: ignorar todo hasta el ">"
            if c == ">":
                estado = EstadoLimpieza.TEXTO
                _espacio(salida)
    if estado is EstadoLimpieza.ETIQUETA:
        # El "<" nunca se cerro: entonces no era HTML. Se procesa lo que sigue como texto.
        _espacio(salida)
        salida.append(_transductor_limpieza(texto[inicio_etiqueta + 1:]))
    return "".join(salida)


def limpiar_texto(texto: str) -> str:
    """Segundo paso: deja el texto "normalizado" (limpio y en minusculas).

    Ejemplo: "A <b>GREAT</b> film!!! 10/10"  ->  "a great film"
    """
    texto = html.unescape(texto)                 # convierte &amp; en &, &lt; en <, etc.
    texto = unicodedata.normalize("NFKC", texto)  # unifica formas raras de escribir un caracter
    return " ".join(_transductor_limpieza(texto).split())


# ---------------------------------------------------------------------------
# 3) TOKENIZACION (partir el texto en palabras)
# Automata de 2 estados: FUERA de una palabra o EN_PALABRA.
# ---------------------------------------------------------------------------
class EstadoToken(Enum):
    FUERA = "FUERA"            # estamos en un espacio entre palabras
    EN_PALABRA = "EN_PALABRA"  # estamos leyendo una palabra


def _normalizar_token(tok: str) -> str:
    """Arregla una palabra: quita apostrofes sueltos y el posesivo 's.

    Ejemplo: "movie's" -> "movie"
    """
    tok = tok.strip("'")
    if tok.endswith("'s"):
        tok = tok[:-2]
    return tok.strip("'")


def tokenizar(texto_limpio: str) -> List[str]:
    """Tercer paso: parte el texto limpio en palabras (tokens).

    Funciona asi: cada vez que lee una letra empieza o sigue una palabra;
    cuando lee un espacio, cierra la palabra y la guarda en la lista.
    """
    tokens: List[str] = []
    actual: List[str] = []
    estado = EstadoToken.FUERA
    for c in texto_limpio + " ":   # el espacio extra al final obliga a cerrar la ultima palabra
        if c == " ":
            if estado is EstadoToken.EN_PALABRA:
                tokens.append("".join(actual))
                actual = []
                estado = EstadoToken.FUERA
        else:
            actual.append(c)
            estado = EstadoToken.EN_PALABRA
    return [t for t in map(_normalizar_token, tokens) if t]


def eliminar_stop_words(tokens: Iterable[str], stop_words=STOP_WORDS) -> List[str]:
    """Tira las palabras vacias (the, of, is...) y las de una sola letra."""
    return [t for t in tokens if t not in stop_words and len(t) > 1]


# ---------------------------------------------------------------------------
# 4) EVALUACION DE SENTIMIENTO (el automata del diagrama)
# ---------------------------------------------------------------------------
class Estado(Enum):
    """Los estados del automata. Son los mismos cuadros del diagrama."""
    ANALISIS_POLARIDAD = "AnalisisPolaridad"  # estado "central": decide a donde ir segun la palabra
    ESTADO_NEUTRAL = "EstadoNeutral"          # la palabra no es ni buena ni mala
    ESTADO_POSITIVO = "EstadoPositivo"        # la palabra es buena
    ESTADO_NEGATIVO = "EstadoNegativo"        # la palabra es mala
    ACEPTACION = "Aceptacion"                 # estado final: ya se leyeron todas las palabras


class Simbolo(Enum):
    """Lo que "lee" el automata para moverse. Son las etiquetas de las flechas."""
    TOKEN_SIN_CARGA = "Token sin carga"
    TOKEN_POSITIVO = "Token en diccionario (+)"
    TOKEN_NEGATIVO = "Token en diccionario (-)"
    SIGUIENTE_TOKEN = "Siguiente token"
    FIN_DE_CADENA = "Fin de cadena"


# Tabla de transiciones: (estado actual, lo que se lee) -> estado siguiente.
# Es exactamente el diagrama: son las 7 flechas que hay en el.
DELTA: Dict[Tuple[Estado, Simbolo], Estado] = {
    (Estado.ANALISIS_POLARIDAD, Simbolo.TOKEN_SIN_CARGA): Estado.ESTADO_NEUTRAL,
    (Estado.ANALISIS_POLARIDAD, Simbolo.TOKEN_POSITIVO): Estado.ESTADO_POSITIVO,
    (Estado.ANALISIS_POLARIDAD, Simbolo.TOKEN_NEGATIVO): Estado.ESTADO_NEGATIVO,
    (Estado.ESTADO_NEUTRAL, Simbolo.SIGUIENTE_TOKEN): Estado.ANALISIS_POLARIDAD,
    (Estado.ESTADO_POSITIVO, Simbolo.SIGUIENTE_TOKEN): Estado.ANALISIS_POLARIDAD,
    (Estado.ESTADO_NEGATIVO, Simbolo.SIGUIENTE_TOKEN): Estado.ANALISIS_POLARIDAD,
    (Estado.ANALISIS_POLARIDAD, Simbolo.FIN_DE_CADENA): Estado.ACEPTACION,
}


def transicion(estado: Estado, simbolo: Simbolo) -> Estado:
    """Mueve el automata: dado el estado actual y lo que lee, devuelve el nuevo estado.

    Si esa flecha no existe en el diagrama, da error (el automata no se puede
    mover de forma que el diagrama no permite).
    """
    try:
        return DELTA[(estado, simbolo)]
    except KeyError:
        raise ValueError(f"Transicion no definida: delta({estado.value}, {simbolo.value})") from None


class AutomataSentimiento:
    """El automata que decide el sentimiento de una resena.

    Como funciona:
      - Empieza en AnalisisPolaridad.
      - Por cada palabra: se mueve a Positivo, Negativo o Neutral segun el
        diccionario, suma 1 al contador de ese estado y regresa a AnalisisPolaridad.
      - Al terminar las palabras pasa a Aceptacion.
      - El sentimiento final es el del contador mas grande (si empatan: neutral).

    negacion=True es una opcion extra (apagada por defecto, porque no esta en el
    diagrama): una palabra como "not" invierte la siguiente palabra buena o mala.
    Ejemplo: "not good" cuenta como negativa.
    """

    def __init__(self, positivas=POSITIVAS, negativas=NEGATIVAS, negacion: bool = False):
        self.positivas = positivas
        self.negativas = negativas
        self.negacion = negacion

    def _simbolo(self, token: str, negar: bool) -> Simbolo:
        """Revisa una palabra en los diccionarios y dice que simbolo es.

        Si 'negar' es True, invierte el resultado (bueno <-> malo).
        """
        if token in self.positivas:
            sim = Simbolo.TOKEN_POSITIVO
        elif token in self.negativas:
            sim = Simbolo.TOKEN_NEGATIVO
        else:
            return Simbolo.TOKEN_SIN_CARGA
        if negar:
            sim = Simbolo.TOKEN_NEGATIVO if sim is Simbolo.TOKEN_POSITIVO else Simbolo.TOKEN_POSITIVO
        return sim

    def evaluar(self, tokens: Iterable[str], traza: bool = False) -> dict:
        """Recorre las palabras por el automata y devuelve el resultado.

        Devuelve un diccionario con: cuantas palabras buenas/malas/neutrales hubo,
        el puntaje (buenas - malas), el sentimiento final y el estado final.
        Si traza=True, tambien devuelve cada paso que dio el automata.
        """
        estado = Estado.ANALISIS_POLARIDAD
        pos = neg = neu = 0
        negar = False
        pasos: List[Tuple[str, str, str]] = []

        def mover(simbolo: Simbolo) -> None:
            """Da un paso del automata (y lo anota si se pidio la traza)."""
            nonlocal estado
            nuevo = transicion(estado, simbolo)
            if traza:
                pasos.append((estado.value, simbolo.value, nuevo.value))
            estado = nuevo

        for token in tokens:
            if self.negacion and es_negacion(token):
                negar = True                      # la siguiente palabra se invertira
                mover(Simbolo.TOKEN_SIN_CARGA)
            else:
                mover(self._simbolo(token, negar))
                negar = False
            # contamos segun el estado al que llegamos
            if estado is Estado.ESTADO_POSITIVO:
                pos += 1
            elif estado is Estado.ESTADO_NEGATIVO:
                neg += 1
            else:
                neu += 1
            mover(Simbolo.SIGUIENTE_TOKEN)        # regresar a AnalisisPolaridad

        mover(Simbolo.FIN_DE_CADENA)              # ya no hay mas palabras
        assert estado is Estado.ACEPTACION

        if pos > neg:
            veredicto = "positivo"
        elif neg > pos:
            veredicto = "negativo"
        else:
            veredicto = "neutral"
        res = {
            "positivos": pos, "negativos": neg, "neutrales": neu,
            "puntaje": pos - neg, "sentimiento": veredicto,
            "estado_final": estado.value,
        }
        if traza:
            res["traza"] = pasos
        return res


@lru_cache(maxsize=None)
def _automata(negacion: bool, positivas=POSITIVAS, negativas=NEGATIVAS) -> AutomataSentimiento:
    """Crea el automata una sola vez y lo reutiliza (para que sea mas rapido)."""
    return AutomataSentimiento(positivas, negativas, negacion=negacion)


# ---------------------------------------------------------------------------
# DICCIONARIO APRENDIDO DE DATOS CON ETIQUETA
# El automata NO cambia; solo se le da un diccionario mejor, sacado de los datos.
# ---------------------------------------------------------------------------
def entrenar_lexico(textos: Iterable[str], etiquetas: Iterable[int], min_freq: int = 5,
                    umbral: float = 0.6, max_palabras: int = 3000) -> Tuple[frozenset, frozenset]:
    """Aprende que palabras son buenas y cuales malas a partir de resenas ya etiquetadas.

    Idea simple: si una palabra sale mucho mas en resenas positivas que en
    negativas, es una palabra "buena" (y al reves).

    - etiquetas: 1 = positiva (Fresh), 0 = negativa (Rotten).
    - min_freq: la palabra debe aparecer al menos esas veces para tomarse en cuenta.
    - umbral: que tan marcada debe ser la diferencia entre positivas y negativas.
    - max_palabras: tope de palabras en total en el diccionario.
    Devuelve (palabras_positivas, palabras_negativas).
    """
    import math
    from collections import Counter
    cpos: Counter = Counter()   # en cuantas resenas positivas sale cada palabra
    cneg: Counter = Counter()   # en cuantas resenas negativas sale cada palabra
    for t, y in zip(textos, etiquetas):
        v = recibir_resena(t)
        if v is None:
            continue
        # set(): cuenta cada palabra una vez por resena (importa si aparece, no cuantas veces)
        toks = set(eliminar_stop_words(tokenizar(limpiar_texto(v))))
        (cpos if y == 1 else cneg).update(toks)
    npos, nneg = sum(cpos.values()) + 1, sum(cneg.values()) + 1
    cand = []
    for w in set(cpos) | set(cneg):
        f = cpos[w] + cneg[w]
        if f < min_freq or es_negacion(w):
            continue
        # lo > 0: la palabra sale mas en positivas; lo < 0: sale mas en negativas
        lo = math.log((cpos[w] + 1) / npos) - math.log((cneg[w] + 1) / nneg)
        if abs(lo) >= umbral:
            cand.append((abs(lo) * math.log(f), lo, w))
    cand.sort(reverse=True)          # primero las palabras mas seguras y frecuentes
    cand = cand[:max_palabras]
    return (frozenset(w for _, lo, w in cand if lo > 0),
            frozenset(w for _, lo, w in cand if lo < 0))


def guardar_lexico(ruta: str, lexico: Tuple[frozenset, frozenset]) -> None:
    """Guarda el diccionario aprendido en un archivo JSON."""
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump({"positivas": sorted(lexico[0]), "negativas": sorted(lexico[1])}, f, ensure_ascii=False)


def cargar_lexico(ruta: str) -> Tuple[frozenset, frozenset]:
    """Lee un diccionario guardado con guardar_lexico()."""
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)
    return frozenset(d["positivas"]), frozenset(d["negativas"])


# ---------------------------------------------------------------------------
# PIPELINE COMPLETO DE UNA RESENA
# ---------------------------------------------------------------------------
def procesar_resena(texto, id_=None, negacion: bool = False, traza: bool = False,
                    lexico: Optional[Tuple[frozenset, frozenset]] = None) -> dict:
    """Pasa UNA resena por todos los pasos: recepcion, limpieza, tokens y sentimiento.

    Devuelve un diccionario con el texto original, el limpio, las palabras y el
    resultado. Si la resena es invalida, devuelve {"valida": False, ...}.
    """
    valido = recibir_resena(texto)
    if valido is None:
        return {"id": id_, "valida": False, "sentimiento": None}
    limpio = limpiar_texto(valido)
    tokens = eliminar_stop_words(tokenizar(limpio))
    pos, neg = lexico if lexico else (POSITIVAS, NEGATIVAS)
    res = _automata(negacion, pos, neg).evaluar(tokens, traza=traza)
    return {"id": id_, "valida": True, "texto_original": valido,
            "texto_limpio": limpio, "tokens": tokens, **res}


# ---------------------------------------------------------------------------
# 5) LA GRANJA: procesar muchas resenas al mismo tiempo
# (reemplaza al bus de mensajes Kafka/RabbitMQ del diseno original)
# ---------------------------------------------------------------------------
def _tarea(par: Tuple[int, str], negacion: bool, lexico=None) -> dict:
    """Trabajo de un proceso de la granja: procesa una resena (id, texto)."""
    return procesar_resena(par[1], id_=par[0], negacion=negacion, lexico=lexico)


def granja(resenas: Iterable[Tuple[int, str]], workers: Optional[int] = None,
           negacion: bool = False, lote: int = 5000, lexico=None) -> Iterator[dict]:
    """Procesa muchas resenas (id, texto) en paralelo y las devuelve en el mismo orden.

    - workers: cuantos procesos usar (por defecto, uno por nucleo del procesador).
      Con workers=1 no se usan procesos extra; sirve para depurar errores.
    - lote: cuantas resenas se mandan juntas, para no llenar la memoria.
    """
    it = iter(resenas)
    if workers == 1:
        for par in it:
            yield _tarea(par, negacion, lexico)
        return
    f = partial(_tarea, negacion=negacion, lexico=lexico)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        while True:
            bloque = list(islice(it, lote))
            if not bloque:
                break
            yield from ex.map(f, bloque, chunksize=max(1, len(bloque) // ((workers or 4) * 4)))


# ---------------------------------------------------------------------------
# 6) GUARDADO DE RESULTADOS
# ---------------------------------------------------------------------------
class AlmacenJSONL:
    """Guarda los resultados en un archivo de texto, una resena por linea (formato JSONL)."""

    def __init__(self, ruta: str):
        self.ruta = ruta
        self._f = open(ruta, "w", encoding="utf-8")

    def guardar_lote(self, docs: List[dict]) -> None:
        """Escribe un grupo de resultados en el archivo."""
        for d in docs:
            self._f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def cerrar(self) -> None:
        """Cierra el archivo."""
        self._f.close()


class AlmacenMongo:
    """Guarda los resultados en MongoDB (necesita: pip install pymongo).

    Usa el id de la resena como llave: si se corre dos veces, actualiza en lugar
    de duplicar.
    """

    def __init__(self, uri: str, base: str = "sentimientos", coleccion: str = "resenas"):
        from pymongo import MongoClient, ReplaceOne  # se importa aqui para que pymongo sea opcional
        self._ReplaceOne = ReplaceOne
        self._cliente = MongoClient(uri)
        self._col = self._cliente[base][coleccion]

    def guardar_lote(self, docs: List[dict]) -> None:
        """Guarda un grupo de resultados en MongoDB."""
        if docs:
            self._col.bulk_write(
                [self._ReplaceOne({"_id": d["id"]}, {**d, "_id": d["id"]}, upsert=True) for d in docs],
                ordered=False)

    def cerrar(self) -> None:
        """Cierra la conexion con MongoDB."""
        self._cliente.close()


def guardar(resultados: Iterable[dict], almacen, tam_lote: int = 1000) -> Dict[str, int]:
    """Guarda los resultados validos (de mil en mil) y devuelve un resumen con los conteos.

    Las resenas invalidas no se guardan, solo se cuentan como "rechazadas".
    """
    resumen = {"positivo": 0, "negativo": 0, "neutral": 0, "rechazadas": 0}
    lote: List[dict] = []
    for r in resultados:
        if not r["valida"]:
            resumen["rechazadas"] += 1
            continue
        resumen[r["sentimiento"]] += 1
        lote.append(r)
        if len(lote) >= tam_lote:
            almacen.guardar_lote(lote)
            lote = []
    if lote:
        almacen.guardar_lote(lote)
    return resumen


# ---------------------------------------------------------------------------
# LECTURA DE DATOS (INGESTA)
# ---------------------------------------------------------------------------
# Nombres de columna que se buscan solos en el CSV si no se indica --col
_COLUMNAS_TEXTO = ("review_content", "review", "text", "content", "comentario", "resena", "phrase")


def leer_csv(ruta: str, columna: Optional[str] = None) -> Iterator[Tuple[int, str]]:
    """Lee un CSV y va entregando (numero_de_fila, texto_de_la_resena).

    Si no se dice cual es la columna del texto, intenta encontrarla sola.
    """
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # permite resenas muy largas
    with open(ruta, newline="", encoding="utf-8", errors="replace") as f:
        lector = csv.DictReader(f)
        if columna is None:
            minus = {c.lower(): c for c in (lector.fieldnames or [])}
            columna = next((minus[c] for c in _COLUMNAS_TEXTO if c in minus), None)
            if columna is None:
                raise SystemExit(f"No encontre la columna de texto. Columnas: {lector.fieldnames}. Usa --col.")
        elif columna not in (lector.fieldnames or []):
            raise SystemExit(f"La columna '{columna}' no existe. Columnas: {lector.fieldnames}")
        for i, fila in enumerate(lector):
            yield i, fila[columna]


# ---------------------------------------------------------------------------
# PROGRAMA PRINCIPAL (linea de comandos)
# ---------------------------------------------------------------------------
# Resenas de ejemplo para --demo (incluye casos raros a proposito).
# "\U0001F60D" es un emoji de carita con corazones en los ojos.
DEMO = [
    "A <b>brilliant</b>, moving and beautifully acted film. 10/10!!!",
    "Boring, predictable and way too long. A total waste of time &amp; money.",
    "It's a movie about a guy who goes to the store.",
    "Not good. Not funny. Never engaging.",
    "Hilarious &lt;i&gt;and&lt;/i&gt; touching, but the ending is a bit clumsy.",
    "   ",
    None,
    "I <3 this charming little gem... \U0001F60D\U0001F60D",
]


def _imprimir_traza(pasos) -> None:
    """Imprime cada paso del automata: de que estado salio, que leyo y a cual llego."""
    for o, s, d in pasos:
        print(f"    {o:>18} --[{s}]--> {d}")


def main(argv=None) -> int:
    """Lee las opciones de la linea de comandos y ejecuta lo que se pidio."""
    ap = argparse.ArgumentParser(description="Granja de Automatas de Sentimientos")
    ap.add_argument("--demo", action="store_true", help="corre resenas de ejemplo")
    ap.add_argument("--texto", help="procesa una sola resena y la muestra")
    ap.add_argument("--traza", action="store_true", help="con --texto/--demo: muestra los pasos del automata")
    ap.add_argument("--csv", help="archivo CSV con resenas")
    ap.add_argument("--col", help="nombre de la columna de texto (si no se da, se busca sola)")
    ap.add_argument("--salida", default="resultados.jsonl", help="archivo JSONL de salida")
    ap.add_argument("--mongo", help="direccion de MongoDB (si se da, guarda ahi en vez del archivo)")
    ap.add_argument("--workers", type=int, default=None, help="procesos en paralelo (por defecto, los nucleos)")
    ap.add_argument("--negacion", action="store_true", help="activa la inversion por negaciones (opcion extra)")
    ap.add_argument("--lexico", help="archivo JSON con diccionario aprendido (lo genera evaluar.py)")
    a = ap.parse_args(argv)
    lexico = cargar_lexico(a.lexico) if a.lexico else None

    # Modo 1: mostrar una resena suelta o la demo
    if a.texto is not None or a.demo:
        textos = [a.texto] if a.texto is not None else DEMO
        for i, t in enumerate(textos):
            r = procesar_resena(t, id_=i, negacion=a.negacion, traza=a.traza, lexico=lexico)
            print(f"\n[{i}] {t!r}")
            if not r["valida"]:
                print("    -> RECHAZADA (caracteres invalidos o vacia)")
                continue
            print(f"    limpio : {r['texto_limpio']}")
            print(f"    tokens : {r['tokens']}")
            print(f"    +{r['positivos']} -{r['negativos']} ={r['neutrales']}  "
                  f"=> {r['sentimiento'].upper()}  (estado final: {r['estado_final']})")
            if a.traza:
                _imprimir_traza(r["traza"])
        return 0

    # Modo 2: procesar un CSV completo con la granja
    if not a.csv:
        ap.error("indica --demo, --texto o --csv")

    almacen = AlmacenMongo(a.mongo) if a.mongo else AlmacenJSONL(a.salida)
    t0 = time.time()
    try:
        resumen = guardar(granja(leer_csv(a.csv, a.col), a.workers, a.negacion, lexico=lexico), almacen)
    finally:
        almacen.cerrar()
    total = sum(resumen.values())
    dt = time.time() - t0
    destino = "MongoDB" if a.mongo else a.salida
    print(f"Listo: {total} resenas en {dt:.1f}s ({total / max(dt, 1e-9):.0f}/s) -> {destino}")
    print(resumen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
