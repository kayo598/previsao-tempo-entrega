"""Treina, compara e salva os modelos, e gera o relatório."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from . import __version__
from .dados import ALVO, COLUNAS_ENTRADA
from .modelos import MODELOS, boosting
from .prever import FORMATO_ARTEFATO, faixa_mostrada

# Abaixo disso, o treino "terminava" com 10 linhas, gravava um modelo sem sentido (R² de -13)
# por cima do bom e escrevia NaN no metricas.json.
MIN_LINHAS = 2_000
# Colunas numéricas cujo intervalo do treino fica guardado no modelo: fora dele, a previsão é recusada.
COM_LIMITES = ["hora", "distancia_km", "itens", "pedidos_na_fila"]

NOMES_LEGIVEIS = {
    "categoria": "Categoria do restaurante",
    "dia_semana": "Dia da semana",
    "hora": "Hora do pedido",
    "chuva": "Chuva",
    "distancia_km": "Distância (km)",
    "itens": "Número de itens",
    "pedidos_na_fila": "Pedidos na fila",
}


@dataclass
class Resultado:
    modelo: str
    mae_cv: float
    mae_cv_desvio: float
    mae_teste: float
    rmse_teste: float
    r2_teste: float
    dentro_10min: float  # fração das previsões com erro de até 10 minutos


def comparar(treino: pd.DataFrame, teste: pd.DataFrame, semente: int = 0) -> tuple[list[Resultado], dict]:
    x_tr, y_tr = treino[COLUNAS_ENTRADA], treino[ALVO]
    x_te, y_te = teste[COLUNAS_ENTRADA], teste[ALVO]
    folds = KFold(n_splits=5, shuffle=True, random_state=semente)

    resultados, treinados = [], {}
    for nome, fabrica in MODELOS.items():
        cv = -cross_val_score(fabrica(), x_tr, y_tr, cv=folds, scoring="neg_mean_absolute_error")
        modelo = fabrica().fit(x_tr, y_tr)
        previsto = modelo.predict(x_te)
        erro = np.abs(previsto - y_te)
        resultados.append(
            Resultado(
                modelo=nome,
                mae_cv=float(cv.mean()),
                mae_cv_desvio=float(cv.std()),
                mae_teste=float(mean_absolute_error(y_te, previsto)),
                rmse_teste=float(np.sqrt(mean_squared_error(y_te, previsto))),
                r2_teste=float(r2_score(y_te, previsto)),
                dentro_10min=float((erro <= 10).mean()),
            )
        )
        treinados[nome] = modelo
    return resultados, treinados


COBERTURA_ALVO = 0.8


def treinar_faixa(
    treino: pd.DataFrame, teste: pd.DataFrame, pontual_teste: np.ndarray, semente: int = 0
) -> tuple[dict, dict]:
    """Faixa de previsão com quantis 10% e 90%, calibrada por conformal prediction.

    Os modelos de quantil sozinhos cobriam menos do que os 80% prometidos. A calibração
    usa um pedaço do treino que os modelos não viram: mede o quanto as entregas reais
    "escapam" da faixa e alarga a faixa na medida certa para cobrir 80%.

    A cobertura no teste é medida na faixa exatamente como o cliente a vê (`faixa_mostrada`),
    com a previsão do modelo pontual (`pontual_teste`) e as bordas em minutos inteiros.
    """
    ajuste, calib = train_test_split(treino, test_size=0.25, random_state=semente)
    p10 = boosting(quantil=0.1).fit(ajuste[COLUNAS_ENTRADA], ajuste[ALVO])
    p90 = boosting(quantil=0.9).fit(ajuste[COLUNAS_ENTRADA], ajuste[ALVO])

    y_cal = calib[ALVO].to_numpy()
    escape = np.maximum(p10.predict(calib[COLUNAS_ENTRADA]) - y_cal, y_cal - p90.predict(calib[COLUNAS_ENTRADA]))
    n = len(y_cal)
    margem = float(np.quantile(escape, min(1.0, np.ceil((n + 1) * COBERTURA_ALVO) / n), method="higher"))

    y = teste[ALVO].to_numpy()
    baixo, alto = p10.predict(teste[COLUNAS_ENTRADA]), p90.predict(teste[COLUNAS_ENTRADA])

    def medir(b, a):
        b, a = faixa_mostrada(b, a, pontual_teste)
        return {
            "cobertura": float(((y >= b) & (y <= a)).mean()),
            "largura_media_min": float((a - b).mean()),
            "chega_depois": float((y > a).mean()),  # o que o cliente sente: entrega depois do fim da faixa
        }

    metricas = {
        "sem_calibrar": medir(baixo, alto),
        "calibrada": medir(baixo - margem, alto + margem),
        "margem_min": margem,
    }
    return {"p10": p10, "p90": p90, "margem_faixa": margem}, metricas


SEGMENTOS = {
    "todas": lambda d: np.ones(len(d), bool),
    "com chuva": lambda d: d["chuva"].to_numpy() == 1,
    "mais de 5 km": lambda d: d["distancia_km"].to_numpy() > 5,
    "chuva e mais de 5 km": lambda d: (d["chuva"].to_numpy() == 1) & (d["distancia_km"].to_numpy() > 5),
}


def por_segmento(teste: pd.DataFrame, treinados: dict) -> list[dict]:
    """Erro e viés de cada modelo por tipo de entrega.
    Viés negativo = o modelo promete menos tempo do que a entrega leva (cliente espera mais)."""
    y = teste[ALVO].to_numpy()
    previsoes = {nome: m.predict(teste[COLUNAS_ENTRADA]) for nome, m in treinados.items() if "baseline" not in nome}
    linhas = []
    for segmento, filtro in SEGMENTOS.items():
        f = filtro(teste)
        linha: dict = {"segmento": segmento, "entregas": int(f.sum())}
        for nome, prev in previsoes.items():
            # Segmento vazio: None, e não NaN (NaN não é JSON válido).
            linha[f"mae {nome}"] = float(np.abs(prev[f] - y[f]).mean()) if f.any() else None
            linha[f"viés {nome}"] = float((prev[f] - y[f]).mean()) if f.any() else None
        linhas.append(linha)
    return linhas


def importancia(modelo, teste: pd.DataFrame, semente: int = 0) -> pd.Series:
    """Quanto o erro (MAE) piora quando embaralhamos cada coluna."""
    r = permutation_importance(
        modelo,
        teste[COLUNAS_ENTRADA],
        teste[ALVO],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=semente,
    )
    return pd.Series(r.importances_mean, index=COLUNAS_ENTRADA).sort_values()


def _grafico_importancia(imp: pd.Series, caminho: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    nomes = [NOMES_LEGIVEIS[c] for c in imp.index]
    ax.barh(nomes, imp.values, color="#3b6fb6", height=0.6)
    for y, v in enumerate(imp.values):
        ax.text(max(v, 0) + imp.max() * 0.01, y, f"{round(v, 1) or 0.0:.1f}", va="center", fontsize=9, color="#333")
    ax.set_xlabel("Aumento do erro médio ao embaralhar a coluna (minutos)")
    ax.set_title("O que mais influencia o tempo de entrega", loc="left", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)


def _grafico_vies(segmentos: list[dict], caminho: Path) -> None:
    modelos = [("regressão linear", "#b0b7c3"), ("gradient boosting", "#3b6fb6")]
    nomes = [s["segmento"] for s in segmentos][::-1]
    y = np.arange(len(nomes))
    altura = 0.36
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    for i, (modelo, cor) in enumerate(modelos):
        valores = [s[f"viés {modelo}"] or 0.0 for s in segmentos][::-1]  # segmento vazio: barra zero
        pos = y + (0.5 - i) * altura
        ax.barh(pos, valores, height=altura, color=cor, label=modelo)
        for p, v in zip(pos, valores, strict=True):
            ax.text(
                v + (0.08 if v >= 0 else -0.08),
                p,
                f"{v:+.1f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8.5,
                color="#333",
            )
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_yticks(y, nomes)
    lim = max([abs(s[f"viés {m}"] or 0.0) for s in segmentos for m, _ in modelos] + [0.5]) * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Viés médio (min): negativo = promete menos do que a entrega leva")
    ax.set_title("Onde a regressão linear erra: chuva e distância longa", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)


def _gravar_sem_corromper(artefato: dict, caminho: Path) -> None:
    """Grava num arquivo temporário da mesma pasta e só depois troca pelo definitivo (troca atômica).
    Antes, uma gravação que falhava no meio (disco cheio, Ctrl+C) destruía o modelo bom anterior."""
    temporario = caminho.with_name(f".{caminho.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(artefato, temporario)
        os.replace(temporario, caminho)
    finally:
        temporario.unlink(missing_ok=True)


def executar(df: pd.DataFrame, caminho_modelo: Path, pasta_relatorio: Path, semente: int = 0) -> dict:
    """Treina, avalia e salva o modelo em `caminho_modelo` (o nome do arquivo é respeitado:
    antes, --modelo meu.joblib gravava em modelo.joblib e dizia que tinha gravado em meu.joblib)."""
    if len(df) < MIN_LINHAS:
        raise ValueError(
            f"são precisas pelo menos {MIN_LINHAS:,} entregas para treinar; vieram {len(df):,}".replace(",", ".")
        )
    caminho_modelo = Path(caminho_modelo)
    treino, teste = train_test_split(df, test_size=0.2, random_state=semente)
    resultados, treinados = comparar(treino, teste, semente)
    segmentos = por_segmento(teste, treinados)
    melhor = treinados["gradient boosting"]
    faixa, metricas_faixa = treinar_faixa(treino, teste, melhor.predict(teste[COLUNAS_ENTRADA]), semente)
    imp = importancia(melhor, teste, semente)

    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
    pasta_relatorio = Path(pasta_relatorio)
    pasta_relatorio.mkdir(parents=True, exist_ok=True)
    limites = {c: [float(treino[c].min()), float(treino[c].max())] for c in COM_LIMITES}
    artefato = {
        "formato": FORMATO_ARTEFATO,
        "versao": __version__,
        "sklearn": sklearn.__version__,
        "colunas": COLUNAS_ENTRADA,
        "limites": limites,
        "pontual": melhor,
        **faixa,
    }
    _gravar_sem_corromper(artefato, caminho_modelo)
    _grafico_importancia(imp, pasta_relatorio / "importancia.png")
    _grafico_vies(segmentos, pasta_relatorio / "vies_por_segmento.png")
    relatorio = {
        "linhas_treino": len(treino),
        "linhas_teste": len(teste),
        "modelos": [asdict(r) for r in resultados],
        "segmentos": segmentos,
        "faixa_p10_p90": metricas_faixa,
        "importancia_min": {k: round(float(v), 2) for k, v in imp.sort_values(ascending=False).items()},
        "limites_do_treino": limites,
    }
    # allow_nan=False: se algum NaN escapar, o erro aparece aqui, e não num leitor de JSON depois.
    texto = json.dumps(relatorio, ensure_ascii=False, indent=2, allow_nan=False)
    (pasta_relatorio / "metricas.json").write_text(texto, "utf-8")
    return relatorio
