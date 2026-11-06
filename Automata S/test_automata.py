
import json
import os
import random
import tempfile
import unittest

import automata_sentimientos as A
from automata_sentimientos import Estado, Simbolo


class TestRecepcion(unittest.TestCase):
    """Pruebas del paso 1: revisar que la resena sea valida."""

    def test_invalidas(self):
        """Textos vacios, solo espacios o solo caracteres raros se rechazan (None)."""
        for x in (None, "", "   ", "\x00\x01", "\ufffd\ufffd"):
            self.assertIsNone(A.recibir_resena(x), repr(x))

    def test_valida(self):
        """Un texto bueno pasa, pero se le quitan los caracteres raros."""
        self.assertEqual(A.recibir_resena("hola\x00 mundo"), "hola mundo")
        self.assertEqual(A.recibir_resena(123), "123")


class TestLimpieza(unittest.TestCase):
    """Pruebas del paso 2: limpiar el texto."""

    def test_html_numeros_simbolos_minusculas(self):
        """Quita HTML, numeros y simbolos, y pasa a minusculas."""
        self.assertEqual(A.limpiar_texto("A <b>GREAT</b> film!!! 10/10"), "a great film")

    def test_entidades(self):
        """Convierte cosas como &amp; y &lt;i&gt; antes de limpiar."""
        self.assertEqual(A.limpiar_texto("fun &amp; games"), "fun games")
        self.assertEqual(A.limpiar_texto("&lt;i&gt;nice&lt;/i&gt;"), "nice")

    def test_menor_que_no_html(self):
        """Un "<" suelto (como en "<3") no se confunde con una etiqueta HTML."""
        self.assertEqual(A.limpiar_texto("I <3 it"), "i it")
        self.assertEqual(A.limpiar_texto("a <very bad one"), "a very bad one")

    def test_apostrofes(self):
        """El apostrofe curvo (\u2019) se convierte en uno normal (')."""
        self.assertEqual(A.limpiar_texto("It\u2019s NOT good"), "it's not good")

    def test_emojis_y_espacios(self):
        """Los emojis desaparecen y los espacios de sobra se juntan en uno solo."""
        self.assertEqual(A.limpiar_texto("wow \U0001F60D\U0001F60D   nice \n\t"), "wow nice")

    def test_idempotente(self):
        """Limpiar un texto ya limpio no lo cambia."""
        t = A.limpiar_texto("<p>Hello, World 42!</p>")
        self.assertEqual(A.limpiar_texto(t), t)


class TestTokenizacion(unittest.TestCase):
    """Pruebas del paso 3: partir en palabras y quitar palabras vacias."""

    def test_tokens(self):
        """Parte el texto en palabras y quita el posesivo 's."""
        self.assertEqual(A.tokenizar("the movie's plot isn't bad"),
                         ["the", "movie", "plot", "isn't", "bad"])

    def test_stop_words_conservan_negaciones(self):
        """Se quitan "this", "is", "a", pero se conserva "not" porque cambia el sentido."""
        t = A.eliminar_stop_words(A.tokenizar("this is not a good plot"))
        self.assertEqual(t, ["not", "good", "plot"])

    def test_vacio(self):
        """Un texto vacio no da palabras."""
        self.assertEqual(A.tokenizar(""), [])


class TestAutomata(unittest.TestCase):
    """Pruebas del paso 4: el automata de sentimientos."""

    def test_traza_igual_al_diagrama(self):
        """Los pasos que da el automata son exactamente los del diagrama."""
        r = A.AutomataSentimiento().evaluar(["good", "table", "bad"], traza=True)
        esperado = [
            ("AnalisisPolaridad", "Token en diccionario (+)", "EstadoPositivo"),
            ("EstadoPositivo", "Siguiente token", "AnalisisPolaridad"),
            ("AnalisisPolaridad", "Token sin carga", "EstadoNeutral"),
            ("EstadoNeutral", "Siguiente token", "AnalisisPolaridad"),
            ("AnalisisPolaridad", "Token en diccionario (-)", "EstadoNegativo"),
            ("EstadoNegativo", "Siguiente token", "AnalisisPolaridad"),
            ("AnalisisPolaridad", "Fin de cadena", "Aceptacion"),
        ]
        self.assertEqual(r["traza"], esperado)
        self.assertEqual(r["estado_final"], "Aceptacion")

    def test_veredictos(self):
        """Gana el sentimiento con mas palabras; si empatan, es neutral."""
        a = A.AutomataSentimiento()
        self.assertEqual(a.evaluar(["great", "fun", "bad"])["sentimiento"], "positivo")
        self.assertEqual(a.evaluar(["boring", "dull", "good"])["sentimiento"], "negativo")
        self.assertEqual(a.evaluar(["good", "bad"])["sentimiento"], "neutral")
        self.assertEqual(a.evaluar(["table"])["sentimiento"], "neutral")

    def test_cadena_vacia_acepta(self):
        """Sin palabras, el automata igual llega a Aceptacion (con resultado neutral)."""
        r = A.AutomataSentimiento().evaluar([])
        self.assertEqual((r["estado_final"], r["sentimiento"], r["puntaje"]), ("Aceptacion", "neutral", 0))

    def test_transicion_indefinida(self):
        """Una flecha que no existe en el diagrama da error."""
        with self.assertRaises(ValueError):
            A.transicion(Estado.ESTADO_POSITIVO, Simbolo.TOKEN_NEGATIVO)
        with self.assertRaises(ValueError):
            A.transicion(Estado.ACEPTACION, Simbolo.SIGUIENTE_TOKEN)

    def test_tabla_cubre_diagrama(self):
        """La tabla tiene exactamente las 7 flechas del diagrama."""
        self.assertEqual(len(A.DELTA), 7)

    def test_negacion(self):
        """Con negacion activada, "not good" es negativo. Apagada, se sigue el diagrama."""
        a = A.AutomataSentimiento(negacion=True)
        self.assertEqual(a.evaluar(["not", "good"])["sentimiento"], "negativo")
        self.assertEqual(a.evaluar(["isn't", "boring"])["sentimiento"], "positivo")
        # la negacion solo afecta a la palabra siguiente
        self.assertEqual(a.evaluar(["not", "plot", "good"])["sentimiento"], "positivo")
        self.assertEqual(A.AutomataSentimiento().evaluar(["not", "good"])["sentimiento"], "positivo")


class TestPipeline(unittest.TestCase):
    """Pruebas de todo el proceso junto: de la resena cruda al resultado guardado."""

    def test_extremo_a_extremo(self):
        """Una resena cruda con HTML y numeros se clasifica bien."""
        r = A.procesar_resena("A <b>brilliant</b>, moving film. 10/10!", id_=7)
        self.assertEqual(r["tokens"], ["brilliant", "moving"])
        self.assertEqual(r["sentimiento"], "positivo")
        self.assertEqual(r["id"], 7)
        r = A.procesar_resena("Boring and predictable &amp; way too long.")
        self.assertEqual(r["sentimiento"], "negativo")

    def test_invalida(self):
        """Una resena vacia se marca como no valida."""
        r = A.procesar_resena("  ", id_=1)
        self.assertFalse(r["valida"])

    def test_granja_paralela_igual_a_secuencial(self):
        """Usar 4 procesos da EXACTAMENTE el mismo resultado y orden que usar 1."""
        rnd = random.Random(1)
        pal = list(A.POSITIVAS | A.NEGATIVAS) + ["table", "the", "<b>", "42", "not", "story", "&amp;"]
        datos = [(i, " ".join(rnd.choices(pal, k=rnd.randint(0, 25)))) for i in range(3000)]
        sec = list(A.granja(datos, workers=1))
        par = list(A.granja(datos, workers=4, lote=700))
        self.assertEqual(sec, par)
        self.assertEqual([r["id"] for r in par], list(range(3000)))

    def test_guardar_jsonl_y_resumen(self):
        """Se guardan solo las resenas validas y el resumen cuenta bien cada tipo."""
        datos = list(enumerate(["great fun", "awful mess", "table", None, "  "]))
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "o.jsonl")
            alm = A.AlmacenJSONL(ruta)
            resumen = A.guardar(A.granja(datos, workers=1), alm, tam_lote=2)
            alm.cerrar()
            with open(ruta, encoding="utf-8") as fh:
                filas = [json.loads(l) for l in fh]
        self.assertEqual(resumen, {"positivo": 1, "negativo": 1, "neutral": 1, "rechazadas": 2})
        self.assertEqual([f["id"] for f in filas], [0, 1, 2])

    def test_csv_autodeteccion(self):
        """Al leer un CSV encuentra sola la columna de texto; si la columna no existe, avisa."""
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "r.csv")
            with open(ruta, "w", encoding="utf-8", newline="") as f:
                f.write('id,Review_Content,review_type\n1,"great, fun film",Fresh\n2,"dull ""mess""",Rotten\n')
            self.assertEqual(list(A.leer_csv(ruta)), [(0, "great, fun film"), (1, 'dull "mess"')])
            with self.assertRaises(SystemExit):
                list(A.leer_csv(ruta, "nope"))


class TestLexicoAprendido(unittest.TestCase):
    """Pruebas del diccionario aprendido de resenas con etiqueta."""

    def test_entrenar_y_usar(self):
        """Aprende que "zorbo" es buena y "blarg" es mala, e ignora palabras que salen en ambas."""
        textos = ["a splendid zorbo film"] * 6 + ["a dreadful blarg film"] * 6
        pos, neg = A.entrenar_lexico(textos, [1] * 6 + [0] * 6, min_freq=3, umbral=0.5)
        self.assertIn("zorbo", pos)
        self.assertIn("blarg", neg)
        self.assertNotIn("film", pos | neg)
        r = A.procesar_resena("what a zorbo", lexico=(pos, neg))
        self.assertEqual(r["sentimiento"], "positivo")
        r = A.procesar_resena("what a zorbo")   # el diccionario base no conoce "zorbo"
        self.assertEqual(r["sentimiento"], "neutral")

    def test_guardar_cargar(self):
        """Un diccionario guardado en archivo se vuelve a leer igual."""
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "l.json")
            A.guardar_lexico(ruta, (frozenset({"a1"}), frozenset({"b1"})))
            self.assertEqual(A.cargar_lexico(ruta), (frozenset({"a1"}), frozenset({"b1"})))

    def test_granja_con_lexico_paralelo(self):
        """El diccionario aprendido tambien funciona cuando se usan varios procesos."""
        lex = (frozenset({"zorbo"}), frozenset({"blarg"}))
        datos = list(enumerate(["zorbo"] * 50 + ["blarg"] * 50))
        r = list(A.granja(datos, workers=2, lote=30, lexico=lex))
        self.assertEqual([x["sentimiento"] for x in r], ["positivo"] * 50 + ["negativo"] * 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
