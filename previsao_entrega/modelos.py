"""Os modelos comparados.

- baseline: sempre prevê a mediana. Todo modelo precisa ganhar dele.
- linear: regressão linear com variáveis categóricas em one-hot.
- boosting: gradient boosting em árvores, que captura relações não lineares
  (por exemplo, o pico de horário e a interação chuva × distância).
- faixa: dois modelos de boosting com perda quantílica (10% e 90%), para
  mostrar ao cliente um intervalo ("chega entre 35 e 50 min") em vez de um número só.

Os modelos de boosting são monotônicos: mais distância, mais itens, mais fila ou chuva nunca
diminuem o tempo previsto. Sem isso, o modelo prometia entrega mais rápida com chuva em 1,2%
dos pedidos. A restrição custa 0,03 min de erro médio (média de 5 sorteios; veja o README).
"""

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .dados import COLUNAS_ENTRADA

CATEGORICAS = ["categoria", "dia_semana"]
NUMERICAS = [c for c in COLUNAS_ENTRADA if c not in CATEGORICAS]
# Colunas na ordem em que saem do pré-processamento do boosting: a categoria codificada
# primeiro, depois as outras, na ordem original.
ORDEM_BOOSTING = ["categoria"] + [c for c in COLUNAS_ENTRADA if c != "categoria"]
# +1: quanto maior, maior o tempo. 0: sem restrição (a hora e o dia têm picos, não uma direção).
SEMPRE_AUMENTAM = {"chuva", "distancia_km", "itens", "pedidos_na_fila"}
MONOTONIA = [1 if c in SEMPRE_AUMENTAM else 0 for c in ORDEM_BOOSTING]


def baseline() -> DummyRegressor:
    return DummyRegressor(strategy="median")


def linear() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAS),
            ("num", StandardScaler(), NUMERICAS),
        ]
    )
    return Pipeline([("pre", pre), ("modelo", LinearRegression())])


def boosting(
    perda: str = "squared_error", quantil: float | None = None, semente: int = 0, monotono: bool = True
) -> Pipeline:
    # A categoria vira um código (0, 1, 2...) e o boosting a trata como categórica nativa.
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["categoria"])],
        remainder="passthrough",
    )
    parametros: dict[str, Any] = {
        "max_iter": 400,
        "learning_rate": 0.06,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "categorical_features": [0],
        "random_state": semente,
    }
    if monotono:
        parametros["monotonic_cst"] = MONOTONIA
    if quantil is not None:
        parametros.update(loss="quantile", quantile=quantil)
    else:
        parametros.update(loss=perda)
    return Pipeline([("pre", pre), ("modelo", HistGradientBoostingRegressor(**parametros))])


MODELOS = {"baseline (mediana)": baseline, "regressão linear": linear, "gradient boosting": boosting}
