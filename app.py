"""
------------------------------------------------------------------
Nombre completo: (completar)
Matricula:       (completar)
Asignatura:      Ciencia de Datos
Fecha:           (completar)
------------------------------------------------------------------
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from linear_model import (
    calcular_pendiente_y_ordenada,
    cargar_puntos,
    generar_tabla_prediccion,
)


def configurar_pagina():
    """Define el titulo y la configuracion general de la pagina."""
    st.set_page_config(
        page_title="Ajuste de Recta y Prediccion",
        page_icon="📈",
        layout="centered",
    )
    st.title("📈 Ajuste de Recta a partir de Dos Puntos")
    st.caption(
        "Ciencia de Datos — 7mo Cuatrimestre — "
        "Carga un archivo .csv con dos puntos (x, y) para comenzar."
    )


def obtener_csv_ejemplo():
    """Genera un .csv de ejemplo en memoria para que el usuario pueda
    descargarlo y probar la aplicacion rapidamente."""
    ejemplo = pd.DataFrame({"x": [1, 8], "y": [3, 14]})
    buffer = io.StringIO()
    ejemplo.to_csv(buffer, index=False)
    return buffer.getvalue()


def graficar_recta(punto_1, punto_2, pendiente_m, ordenada_b, tabla):
    """Construye la figura de Matplotlib con los puntos originales,
    la recta ajustada y las proyecciones extrapoladas."""
    x1, y1 = punto_1
    x2, y2 = punto_2

    figura, eje = plt.subplots(figsize=(7, 5))

    # Puntos originales
    eje.scatter(
        [x1, x2], [y1, y2],
        color="crimson", zorder=5, s=90,
        label="Puntos de referencia (P1, P2)",
    )
    eje.annotate(f"P1({x1:g}, {y1:g})", (x1, y1),
                 textcoords="offset points", xytext=(8, 8))
    eje.annotate(f"P2({x2:g}, {y2:g})", (x2, y2),
                 textcoords="offset points", xytext=(8, 8))

    # Recta ajustada (interpolacion)
    interpolados = tabla[tabla["tipo"] == "interpolado"]
    eje.plot(
        interpolados["x"], interpolados["y_estimado"],
        color="steelblue", linewidth=2,
        label=f"y = {pendiente_m:.4f}x + {ordenada_b:.4f}",
    )

    # Extrapolacion (linea punteada)
    extrapolados = tabla[tabla["tipo"] == "extrapolado"]
    if not extrapolados.empty:
        union = pd.concat([interpolados.tail(1), extrapolados])
        eje.plot(
            union["x"], union["y_estimado"],
            color="steelblue", linewidth=2, linestyle="--",
            label="Prediccion (extrapolacion)",
        )

    eje.set_title("Ajuste Lineal a partir de Dos Puntos")
    eje.set_xlabel("X")
    eje.set_ylabel("Y")
    eje.grid(True, linestyle=":", alpha=0.6)
    eje.legend(loc="best")

    return figura


def main():
    """Punto de entrada principal de la aplicacion Streamlit."""
    configurar_pagina()

    with st.sidebar:
        st.header("Carga de datos")
        archivo_subido = st.file_uploader(
            "Archivo .csv con columnas x, y (2 filas)", type=["csv"]
        )
        st.download_button(
            "⬇️ Descargar .csv de ejemplo",
            data=obtener_csv_ejemplo(),
            file_name="puntos_ejemplo.csv",
            mime="text/csv",
        )
        pasos_extrapolacion = st.slider(
            "Cantidad de valores futuros de x a predecir",
            min_value=1, max_value=20, value=5,
        )

    if archivo_subido is None:
        st.info(
            "Sube un archivo .csv con dos puntos P1(x1, y1) y "
            "P2(x2, y2), o descarga el archivo de ejemplo en la "
            "barra lateral."
        )
        return

    try:
        punto_1, punto_2 = cargar_puntos(archivo_subido)
    except ValueError as error:
        st.error(f"Error en los datos de entrada: {error}")
        return

    pendiente_m, ordenada_b = calcular_pendiente_y_ordenada(punto_1, punto_2)

    tabla_prediccion = generar_tabla_prediccion(
        punto_1, punto_2, pendiente_m, ordenada_b,
        pasos_extrapolacion=pasos_extrapolacion,
    )

    st.subheader("Parametros calculados")
    columna_m, columna_b = st.columns(2)
    columna_m.metric("Pendiente (m)", f"{pendiente_m:.4f}")
    columna_b.metric("Ordenada al origen (b)", f"{ordenada_b:.4f}")
    st.latex(f"y = {pendiente_m:.4f}x + {ordenada_b:.4f}")

    st.subheader("Grafica de la recta ajustada")
    figura = graficar_recta(punto_1, punto_2, pendiente_m, ordenada_b,
                             tabla_prediccion)
    st.pyplot(figura)

    st.subheader("Tabla de interpolacion y prediccion")
    st.dataframe(tabla_prediccion, use_container_width=True)


if __name__ == "__main__":
    main()
