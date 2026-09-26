"""
linear_model.py

Modulo de logica numerica para el ajuste de una recta a partir de dos
puntos de referencia. Se mantiene completamente separado de los
componentes de interfaz web (Streamlit) para cumplir con el criterio
de arquitectura y modularidad de la rubrica.

Autor: (completar nombre completo)
Matricula: (completar matricula)
Asignatura: Ciencia de Datos
Fecha: (completar fecha de entrega)
"""

import numpy as np
import pandas as pd


def cargar_puntos(archivo_csv):
    """Lee un archivo .csv con dos registros (P1 y P2) y los valida.

    Se espera un archivo con columnas 'x' e 'y' y exactamente dos filas.

    Args:
        archivo_csv: ruta o buffer del archivo .csv a leer.

    Returns:
        tuple: ((x1, y1), (x2, y2)) con los dos puntos como flotantes.

    Raises:
        ValueError: si el archivo no contiene exactamente dos puntos,
            si faltan las columnas requeridas, o si x1 es igual a x2
            (lo que impediria calcular la pendiente).
    """
    datos = pd.read_csv(archivo_csv)

    columnas_requeridas = {"x", "y"}
    if not columnas_requeridas.issubset(set(datos.columns.str.lower())):
        raise ValueError(
            "El archivo .csv debe contener las columnas 'x' e 'y'."
        )

    # Normalizar nombres de columnas a minusculas por si vienen en otro formato
    datos.columns = [columna.lower() for columna in datos.columns]

    if len(datos) != 2:
        raise ValueError(
            f"Se esperaban exactamente 2 puntos (P1 y P2), "
            f"pero se encontraron {len(datos)}."
        )

    x1, y1 = float(datos.loc[0, "x"]), float(datos.loc[0, "y"])
    x2, y2 = float(datos.loc[1, "x"]), float(datos.loc[1, "y"])

    if np.isclose(x1, x2):
        raise ValueError(
            "x1 no puede ser igual a x2: no se puede calcular la pendiente "
            "de una recta vertical."
        )

    return (x1, y1), (x2, y2)


def calcular_pendiente_y_ordenada(punto_1, punto_2):
    """Calcula la pendiente (m) y la ordenada al origen (b) de la recta
    que pasa por dos puntos dados.

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    Args:
        punto_1: tupla (x1, y1) del primer punto de referencia.
        punto_2: tupla (x2, y2) del segundo punto de referencia.

    Returns:
        tuple: (m, b) como valores flotantes.
    """
    x1, y1 = punto_1
    x2, y2 = punto_2

    pendiente_m = (y2 - y1) / (x2 - x1)
    ordenada_b = y1 - pendiente_m * x1

    return pendiente_m, ordenada_b


def generar_tabla_prediccion(punto_1, punto_2, pendiente_m, ordenada_b,
                              pasos_extrapolacion=5):
    """Genera un DataFrame con los valores interpolados entre x1 y x2,
    y los valores extrapolados (predichos) para x > x2.

    Args:
        punto_1: tupla (x1, y1) del primer punto de referencia.
        punto_2: tupla (x2, y2) del segundo punto de referencia.
        pendiente_m: pendiente de la recta ajustada.
        ordenada_b: ordenada al origen de la recta ajustada.
        pasos_extrapolacion: cantidad de valores enteros de x a proyectar
            mas alla de x2 (por defecto 5).

    Returns:
        pandas.DataFrame: columnas 'x', 'y_estimado' y 'tipo'
            ('interpolado' o 'extrapolado').
    """
    x1, _ = punto_1
    x2, _ = punto_2

    x_menor, x_mayor = sorted((x1, x2))

    x_interpolados = np.arange(np.floor(x_menor), np.floor(x_mayor) + 1, 1)
    x_extrapolados = np.arange(
        np.floor(x_mayor) + 1,
        np.floor(x_mayor) + 1 + pasos_extrapolacion,
        1,
    )

    valores_x = np.concatenate([x_interpolados, x_extrapolados])
    valores_y = pendiente_m * valores_x + ordenada_b

    etiquetas_tipo = (
        ["interpolado"] * len(x_interpolados)
        + ["extrapolado"] * len(x_extrapolados)
    )

    tabla_resultado = pd.DataFrame({
        "x": valores_x,
        "y_estimado": np.round(valores_y, 4),
        "tipo": etiquetas_tipo,
    })

    return tabla_resultado
