"""Gera um conjunto de dados SINTÉTICO de entregas.

Os dados são simulados, mas seguem regras realistas, e as regras são conhecidas.
Isso permite conferir se o modelo aprendeu o que de fato importa
(por exemplo: chuva aumenta o tempo de deslocamento, não o de preparo).

Regras usadas na simulação:
- preparo = base da categoria + 1,2 min por item + 1,8 min por pedido na fila
- deslocamento = 4 min de coleta + 3 min/km, 35% mais lento com chuva e 25% mais lento no horário de pico
- horário de pico (11h45–13h15 e 19h30–21h30) NÃO é uma coluna: o modelo precisa descobrir pela hora
- ruído assimétrico e 3% de atrasos grandes (10 a 30 min), como acontece na prática
"""

import numpy as np
import pandas as pd

PREPARO_BASE = {"pizzaria": 18.0, "japones": 22.0, "lanches": 12.0, "brasileira": 15.0}
COLUNAS_ENTRADA = ["categoria", "dia_semana", "hora", "chuva", "distancia_km", "itens", "pedidos_na_fila"]
ALVO = "minutos"


def em_pico(hora: np.ndarray) -> np.ndarray:
    return ((hora >= 11.75) & (hora <= 13.25)) | ((hora >= 19.5) & (hora <= 21.5))


def gerar(n: int = 20_000, semente: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(semente)

    categoria = rng.choice(list(PREPARO_BASE), size=n, p=[0.35, 0.20, 0.30, 0.15])
    # 0 = segunda ... 6 = domingo; sexta, sábado e domingo têm mais pedidos
    dia_semana = rng.choice(7, size=n, p=[0.10, 0.11, 0.12, 0.13, 0.18, 0.19, 0.17])
    almoco = rng.random(n) < 0.4
    hora = np.where(almoco, rng.normal(12.5, 0.7, n), rng.normal(20.3, 1.0, n)).clip(10.5, 23.9)
    chuva = rng.random(n) < 0.15
    distancia = rng.gamma(2.2, 1.3, n).clip(0.3, 10.0)
    itens = (rng.poisson(1.8, n) + 1).clip(1, 10)

    pico = em_pico(hora)
    fim_de_semana = dia_semana >= 4
    fila = rng.poisson(2 + 4 * pico + 2 * fim_de_semana + 2 * chuva)

    preparo = np.vectorize(PREPARO_BASE.get)(categoria) + 1.2 * itens + 1.8 * fila
    min_por_km = 3.0 * np.where(chuva, 1.35, 1.0) * np.where(pico, 1.25, 1.0)
    deslocamento = 4.0 + distancia * min_por_km
    ruido = rng.gamma(2.0, 2.5, n) - 5.0  # média zero, cauda para a direita
    atraso_grande = np.where(rng.random(n) < 0.03, rng.uniform(10, 30, n), 0.0)

    minutos = (preparo + deslocamento + ruido + atraso_grande).clip(min=10)

    return pd.DataFrame(
        {
            "categoria": categoria,
            "dia_semana": dia_semana,
            "hora": hora.round(2),
            "chuva": chuva.astype(int),
            "distancia_km": distancia.round(2),
            "itens": itens,
            "pedidos_na_fila": fila,
            ALVO: minutos.round(1),
        }
    )
