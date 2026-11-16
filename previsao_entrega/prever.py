"""Usa o modelo salvo para prever o tempo de uma entrega.

Garantias:
- Entrada inválida (tipo errado, infinito, fora de faixa) vira ValueError com mensagem clara.
- Entrada fora do que o modelo viu no treino é recusada. Antes, 50 km ou 1 bilhão de km
  davam "66 min" com toda a confiança: árvores de decisão repetem o valor da borda.
- Modelo corrompido, de versão antiga ou que não é um modelo vira ErroModelo, sem traceback.
"""

import math
import numbers
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .dados import PREPARO_BASE

FORMATO_ARTEFATO = 2  # muda quando o conteúdo do arquivo do modelo muda
CHAVES_ARTEFATO = {"formato", "colunas", "pontual", "p10", "p90", "margem_faixa", "limites", "sklearn"}

NOMES = {
    "dia_semana": "dia da semana",
    "hora": "hora",
    "distancia_km": "distância",
    "itens": "número de itens",
    "pedidos_na_fila": "pedidos na fila",
}


class ErroModelo(Exception):
    """O arquivo do modelo não pode ser usado (não existe, corrompido, versão antiga)."""


@dataclass
class Previsao:
    minutos: int
    minimo: int
    maximo: int
    avisos: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"Previsão: {self.minutos} min (faixa provável: {self.minimo} a {self.maximo} min)"


def _sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


# numbers.Real e numbers.Integral também aceitam os tipos do numpy. Antes, um valor tirado de uma
# coluna do pandas (numpy.int64) era recusado com "dia_semana vai de 0 a 6", mesmo valendo 4.
def _numero(valor) -> bool:
    return isinstance(valor, numbers.Real) and not isinstance(valor, bool) and math.isfinite(valor)


def _inteiro(valor) -> bool:
    return isinstance(valor, numbers.Integral) and not isinstance(valor, bool)


def formatar_hora(hora: float) -> str:
    minutos = round(hora * 60)
    return f"{minutos // 60}h{minutos % 60:02d}"


def _formatar(campo: str, valor: float) -> str:
    if campo == "hora":
        return formatar_hora(valor)
    if campo == "distancia_km":
        return f"{valor:g} km".replace(".", ",")
    return f"{valor:g}"


def faixa_mostrada(baixo, alto, pontual):
    """A faixa como o cliente vê: contém a previsão, em minutos inteiros e nunca abaixo de zero.
    Vale para um número ou para um array, e o treino mede a cobertura com esta mesma função.

    As bordas vão para o minuto mais PRÓXIMO. A primeira versão arredondava para fora (para baixo
    e para cima): a faixa mostrada ficava 1 min mais larga e continha 85% das entregas, não os 80%
    que a calibração mediu e o README prometia."""
    baixo, alto = np.minimum(baixo, pontual), np.maximum(alto, pontual)
    return np.maximum(0, np.round(baixo)), np.round(alto)


def validar(pedido: dict, limites: dict | None = None) -> dict:
    """Confere o pedido e devolve uma cópia normalizada ("Japonês" -> "japones", chuva True -> 1).

    `limites` são as faixas vistas no treino ({"distancia_km": [0.3, 10.0], ...}). Fora delas,
    o modelo não sabe o que acontece, e a previsão seria um chute com cara de certeza."""
    if not isinstance(pedido, dict):
        raise ValueError("o pedido precisa ser um dicionário")
    faltando = [
        c
        for c in ("categoria", "dia_semana", "hora", "chuva", "distancia_km", "itens", "pedidos_na_fila")
        if c not in pedido
    ]
    if faltando:
        raise ValueError(f"faltam os campos: {', '.join(faltando)}")

    categoria = pedido["categoria"]
    categoria = _sem_acento(categoria.strip().lower()) if isinstance(categoria, str) else categoria
    if categoria not in PREPARO_BASE:
        raise ValueError(f"categoria deve ser uma de: {', '.join(PREPARO_BASE)}")
    if not _inteiro(pedido["dia_semana"]) or not 0 <= pedido["dia_semana"] <= 6:
        raise ValueError("dia_semana vai de 0 (segunda) a 6 (domingo)")
    if not _numero(pedido["hora"]) or not 0 <= pedido["hora"] < 24:
        raise ValueError("hora deve estar entre 0 e 24 (ex.: 20.5 para 20h30)")
    chuva = pedido["chuva"]
    if chuva not in (0, 1):  # True, 1 e 1.0 valem como sim; "1" (texto) não
        raise ValueError("chuva deve ser 0 (não) ou 1 (sim)")
    if not _numero(pedido["distancia_km"]) or pedido["distancia_km"] <= 0:
        raise ValueError("a distância precisa ser um número maior que zero")
    if not _inteiro(pedido["itens"]) or pedido["itens"] < 1:
        raise ValueError("o número de itens precisa ser um inteiro, a partir de 1")
    if not _inteiro(pedido["pedidos_na_fila"]) or pedido["pedidos_na_fila"] < 0:
        raise ValueError("os pedidos na fila precisam ser um inteiro, a partir de 0")

    normalizado = {
        **pedido,
        "categoria": categoria,
        "chuva": int(chuva),
        "dia_semana": int(pedido["dia_semana"]),
        "hora": float(pedido["hora"]),
        "distancia_km": float(pedido["distancia_km"]),
        "itens": int(pedido["itens"]),
        "pedidos_na_fila": int(pedido["pedidos_na_fila"]),
    }
    for campo, (minimo, maximo) in (limites or {}).items():
        valor = normalizado[campo]
        if not minimo <= valor <= maximo:
            raise ValueError(
                f"{NOMES.get(campo, campo)} {_formatar(campo, valor)} está fora do que o modelo conhece "
                f"({_formatar(campo, minimo)} a {_formatar(campo, maximo)}); a previsão não seria confiável"
            )
    return normalizado


def carregar(caminho_modelo: Path) -> dict:
    """Lê o arquivo do modelo. Atenção: um arquivo .joblib executa código ao ser lido.
    Use só modelos que você mesmo treinou."""
    caminho_modelo = Path(caminho_modelo)
    if not caminho_modelo.is_file():
        raise ErroModelo(f"modelo não encontrado em {caminho_modelo}. Rode primeiro: previsao-entrega treinar")
    try:
        artefato = joblib.load(caminho_modelo)
    except Exception as erro:  # o formato pickle pode falhar de muitos jeitos
        raise ErroModelo(
            f"não consegui ler o modelo em {caminho_modelo} ({type(erro).__name__}). Rode o treino de novo."
        ) from erro
    if (
        not isinstance(artefato, dict)
        or artefato.get("formato") != FORMATO_ARTEFATO
        or not set(artefato) >= CHAVES_ARTEFATO
    ):
        raise ErroModelo(f"o modelo em {caminho_modelo} é de uma versão antiga deste projeto. Rode o treino de novo.")
    return artefato


def prever(pedido: dict, modelo: Path | dict) -> Previsao:
    """`modelo` é o caminho do arquivo ou o que `carregar` devolveu. Ler o arquivo leva cerca de
    0,1 s: para várias previsões seguidas (um serviço, por exemplo), carregue uma vez só."""
    artefato = modelo if isinstance(modelo, dict) else carregar(modelo)
    pedido = validar(pedido, artefato["limites"])
    linha = pd.DataFrame([{c: pedido[c] for c in artefato["colunas"]}])
    pontual = float(artefato["pontual"].predict(linha)[0])
    margem = artefato["margem_faixa"]  # calibração da faixa (ver treino.treinar_faixa)
    baixo = float(artefato["p10"].predict(linha)[0]) - margem
    alto = float(artefato["p90"].predict(linha)[0]) + margem
    # Modelos de quantis treinados separadamente podem se cruzar; a faixa sempre contém a previsão.
    minimo, maximo = faixa_mostrada(baixo, alto, pontual)
    avisos = []
    if artefato["sklearn"] != sklearn.__version__:
        avisos.append(
            f"o modelo foi treinado com scikit-learn {artefato['sklearn']} e está rodando com {sklearn.__version__}; "
            "se a previsão parecer estranha, rode o treino de novo"
        )
    return Previsao(int(np.round(pontual)), int(minimo), int(maximo), avisos)
