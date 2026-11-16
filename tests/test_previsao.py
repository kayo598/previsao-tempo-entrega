import json
import random
from pathlib import Path

import joblib
import numpy as np
import pytest
import sklearn
from sklearn.model_selection import train_test_split

from previsao_entrega.dados import ALVO, COLUNAS_ENTRADA, em_pico, gerar
from previsao_entrega.modelos import MONOTONIA, ORDEM_BOOSTING, boosting
from previsao_entrega.prever import ErroModelo, carregar, faixa_mostrada, prever, validar
from previsao_entrega.treino import executar, por_segmento

PEDIDO = {
    "categoria": "pizzaria",
    "dia_semana": 5,
    "hora": 20.5,
    "chuva": 0,
    "distancia_km": 3.0,
    "itens": 2,
    "pedidos_na_fila": 4,
}


@pytest.fixture(scope="module")
def treinado(tmp_path_factory):
    pasta = tmp_path_factory.mktemp("saida")
    relatorio = executar(gerar(5_000, semente=1), pasta / "modelos" / "modelo.joblib", pasta / "relatorios")
    return pasta, relatorio


def test_dados_reprodutiveis_e_dentro_dos_limites():
    a, b = gerar(500, semente=7), gerar(500, semente=7)
    assert a.equals(b)
    assert list(a.columns) == [*COLUNAS_ENTRADA, ALVO]
    assert a["distancia_km"].between(0.3, 10).all()
    assert a["dia_semana"].between(0, 6).all()
    assert (a[ALVO] >= 10).all()


def test_pico_de_almoco_e_jantar():
    assert em_pico(np.array([12.0, 20.0, 16.0, 23.0])).tolist() == [True, True, False, False]


def test_modelos_ganham_do_baseline(treinado):
    _, r = treinado
    mae = {m["modelo"]: m["mae_teste"] for m in r["modelos"]}
    assert mae["gradient boosting"] < 0.5 * mae["baseline (mediana)"]


def test_boosting_nao_subestima_chuva_e_distancia(treinado):
    _, r = treinado
    seg = next(s for s in r["segmentos"] if s["segmento"] == "chuva e mais de 5 km")
    assert abs(seg["viés gradient boosting"]) < abs(seg["viés regressão linear"])


def test_faixa_calibrada_cobre_o_prometido(treinado):
    _, r = treinado
    assert r["faixa_p10_p90"]["calibrada"]["cobertura"] >= 0.75


def test_salva_artefatos(treinado):
    pasta, _ = treinado
    assert (pasta / "modelos" / "modelo.joblib").exists()
    for arquivo in ("importancia.png", "vies_por_segmento.png", "metricas.json"):
        assert (pasta / "relatorios" / arquivo).exists()
    json.loads((pasta / "relatorios" / "metricas.json").read_text("utf-8"))


def test_previsao_reage_como_esperado(treinado):
    pasta, _ = treinado
    modelo = pasta / "modelos" / "modelo.joblib"
    base = prever(PEDIDO, modelo)
    assert 15 < base.minutos < 90
    assert base.minimo <= base.minutos <= base.maximo
    assert prever({**PEDIDO, "distancia_km": 7.0}, modelo).minutos > base.minutos
    assert prever({**PEDIDO, "pedidos_na_fila": 12}, modelo).minutos > base.minutos


@pytest.mark.parametrize(
    "campo, valor",
    [("categoria", "sushi"), ("dia_semana", 7), ("hora", 25), ("distancia_km", 0), ("itens", 0)],
)
def test_validacao_de_entrada(campo, valor):
    with pytest.raises(ValueError):
        validar({**PEDIDO, campo: valor})


# --- Validação da entrada ---


@pytest.mark.parametrize(
    "campo, valor, mensagem",
    [
        ("dia_semana", "5", "dia_semana"),  # era TypeError com traceback
        ("dia_semana", True, "dia_semana"),
        ("dia_semana", 5.0, "dia_semana"),
        ("hora", True, "hora"),  # era aceito
        ("hora", float("nan"), "hora"),
        ("hora", "20", "hora"),
        ("chuva", 7, "chuva"),  # era aceito
        ("chuva", 0.5, "chuva"),
        ("chuva", "1", "chuva"),
        ("distancia_km", float("inf"), "distância"),  # era aceito
        ("distancia_km", -1, "distância"),
        ("itens", 2.5, "itens"),  # era aceito
        ("itens", True, "itens"),
        ("pedidos_na_fila", -1, "fila"),
        ("categoria", None, "categoria"),
    ],
)
def test_tipos_e_valores_invalidos(campo, valor, mensagem):
    with pytest.raises(ValueError, match=mensagem):
        validar({**PEDIDO, campo: valor})


def test_campo_faltando_e_pedido_que_nao_e_dicionario():
    with pytest.raises(ValueError, match="faltam os campos: hora"):
        validar({k: v for k, v in PEDIDO.items() if k != "hora"})
    with pytest.raises(ValueError, match="dicionário"):
        validar(["pizzaria"])


@pytest.mark.parametrize("chuva, esperado", [(True, 1), (False, 0), (1, 1), (0, 0), (1.0, 1), (0.0, 0)])
def test_chuva_e_normalizada(chuva, esperado):
    normalizado = validar({**PEDIDO, "chuva": chuva})
    assert normalizado["chuva"] == esperado and type(normalizado["chuva"]) is int


def test_entrada_e_normalizada():
    normalizado = validar({**PEDIDO, "categoria": " Japonês ", "chuva": True})
    assert normalizado["categoria"] == "japones" and normalizado["chuva"] == 1
    assert PEDIDO["categoria"] == "pizzaria"  # o pedido original não é alterado


@pytest.mark.parametrize(
    "campo, valor, mensagem",
    [
        ("distancia_km", 50.0, "distância 50 km está fora do que o modelo conhece \\(0,3 km a 10 km\\)"),
        ("distancia_km", 0.1, "distância 0,1 km está fora"),
        ("hora", 3.0, "hora 3h00 está fora do que o modelo conhece \\(10h30 a 23h54\\)"),
        ("pedidos_na_fila", 200, "pedidos na fila 200 está fora"),
        ("itens", 80, "número de itens 80 está fora"),
    ],
)
def test_fora_do_que_o_modelo_conhece_e_recusado(campo, valor, mensagem):
    """Antes, 50 km ou 1 bilhão de km davam "66 min": árvores repetem o valor da borda."""
    limites = {"distancia_km": [0.3, 10.0], "hora": [10.5, 23.9], "pedidos_na_fila": [0, 25], "itens": [1, 9]}
    with pytest.raises(ValueError, match=mensagem):
        validar({**PEDIDO, campo: valor}, limites)
    validar(PEDIDO, limites)  # dentro dos limites passa


# --- Modelo salvo ---


def test_modelo_guarda_os_limites_do_treino(treinado):
    pasta, relatorio = treinado
    artefato = carregar(pasta / "modelos" / "modelo.joblib")
    assert artefato["limites"] == relatorio["limites_do_treino"]
    assert artefato["limites"]["distancia_km"][1] <= 10.0 and artefato["limites"]["hora"][0] >= 10.5
    assert artefato["sklearn"] == sklearn.__version__


def test_previsao_fora_dos_limites_e_recusada(treinado):
    pasta, _ = treinado
    with pytest.raises(ValueError, match="fora do que o modelo conhece"):
        prever({**PEDIDO, "distancia_km": 50.0}, pasta / "modelos" / "modelo.joblib")


@pytest.mark.parametrize(
    "conteudo, mensagem",
    [
        (b"lixo", "não consegui ler"),
        (None, "versão antiga"),  # um dicionário sem os campos novos
        ([1, 2], "versão antiga"),
    ],
)
def test_modelo_ruim_vira_erro_claro(tmp_path, conteudo, mensagem):
    caminho = tmp_path / "modelo.joblib"
    if isinstance(conteudo, bytes):
        caminho.write_bytes(conteudo)
    else:
        joblib.dump({"pontual": 1} if conteudo is None else conteudo, caminho)
    with pytest.raises(ErroModelo, match=mensagem):
        prever(PEDIDO, caminho)


def test_modelo_de_outro_formato_com_todos_os_campos(treinado, tmp_path):
    pasta, _ = treinado
    artefato = joblib.load(pasta / "modelos" / "modelo.joblib")
    joblib.dump({**artefato, "formato": 1}, tmp_path / "m.joblib")
    with pytest.raises(ErroModelo, match="versão antiga"):
        prever(PEDIDO, tmp_path / "m.joblib")


class Fixo:
    """Modelo falso que sempre prevê o mesmo valor."""

    def __init__(self, valor):
        self.valor = valor

    def predict(self, linhas):
        return np.full(len(linhas), self.valor)


def artefato_falso(pontual, p10, p90, margem=0.0):
    return {
        "formato": 2,
        "colunas": COLUNAS_ENTRADA,
        "pontual": Fixo(pontual),
        "p10": Fixo(p10),
        "p90": Fixo(p90),
        "margem_faixa": margem,
        "limites": {},
        "sklearn": sklearn.__version__,
    }


def test_quantis_cruzados_ainda_dao_faixa_que_contem_a_previsao():
    """Os modelos de 10% e 90% são treinados separados e podem se cruzar."""
    previsao = prever(PEDIDO, artefato_falso(pontual=40, p10=45, p90=38))
    assert previsao.minimo <= previsao.minutos <= previsao.maximo
    assert (previsao.minimo, previsao.minutos, previsao.maximo) == (40, 40, 40)


def test_faixa_nunca_comeca_abaixo_de_zero():
    previsao = prever(PEDIDO, artefato_falso(pontual=12, p10=3, p90=20, margem=8))
    assert previsao.minimo == 0 and previsao.maximo == 28


def test_modelo_inexistente(tmp_path):
    with pytest.raises(ErroModelo, match="não encontrado"):
        prever(PEDIDO, tmp_path / "nada.joblib")


def test_aviso_quando_o_scikit_learn_mudou(treinado, tmp_path):
    pasta, _ = treinado
    artefato = joblib.load(pasta / "modelos" / "modelo.joblib")
    joblib.dump({**artefato, "sklearn": "0.0.1"}, tmp_path / "m.joblib")
    previsao = prever(PEDIDO, tmp_path / "m.joblib")
    assert len(previsao.avisos) == 1 and "0.0.1" in previsao.avisos[0]
    assert prever(PEDIDO, pasta / "modelos" / "modelo.joblib").avisos == []


# --- Coerência das previsões ---


def test_previsao_nunca_promete_menos_com_mais_motivo_para_demorar(treinado):
    """Sem restrição de monotonicidade, a chuva deixava a entrega mais rápida em 1,2% dos pedidos."""
    pasta, _ = treinado
    artefato = carregar(pasta / "modelos" / "modelo.joblib")  # carregado uma vez só
    limites = artefato["limites"]
    gerador = random.Random(3)
    for _ in range(100):
        pedido = {
            "categoria": gerador.choice(["pizzaria", "japones", "lanches", "brasileira"]),
            "dia_semana": gerador.randint(0, 6),
            "hora": round(gerador.uniform(*limites["hora"]), 2),
            "chuva": 0,
            "distancia_km": round(gerador.uniform(limites["distancia_km"][0], limites["distancia_km"][1] - 1), 1),
            "itens": gerador.randint(1, int(limites["itens"][1]) - 1),
            "pedidos_na_fila": gerador.randint(0, int(limites["pedidos_na_fila"][1]) - 1),
        }
        base = prever(pedido, artefato)
        for campo, mais in (
            ("chuva", 1),
            ("distancia_km", pedido["distancia_km"] + 1),
            ("itens", pedido["itens"] + 1),
            ("pedidos_na_fila", pedido["pedidos_na_fila"] + 1),
        ):
            outra = prever({**pedido, campo: mais}, artefato)
            assert outra.minutos >= base.minutos, (campo, pedido)
        assert 0 <= base.minimo <= base.minutos <= base.maximo


def test_monotonia_aponta_para_as_colunas_certas():
    """As restrições valem por posição: a ordem precisa ser a que sai do pré-processamento."""
    df = gerar(300, semente=2)
    modelo = boosting().fit(df[COLUNAS_ENTRADA], df[ALVO])
    nomes = [n.split("__", 1)[1] for n in modelo.named_steps["pre"].get_feature_names_out()]
    assert nomes == ORDEM_BOOSTING
    assert dict(zip(ORDEM_BOOSTING, MONOTONIA, strict=True)) == {
        "categoria": 0,
        "dia_semana": 0,
        "hora": 0,
        "chuva": 1,
        "distancia_km": 1,
        "itens": 1,
        "pedidos_na_fila": 1,
    }


# --- Treino ---


def test_treino_com_poucas_linhas_e_recusado(tmp_path):
    """Antes, 10 linhas "treinavam" um modelo com R² de -13 e o gravavam por cima do bom."""
    with pytest.raises(ValueError, match=r"pelo menos 2\.000"):
        executar(gerar(500, semente=1), tmp_path / "m.joblib", tmp_path)
    assert not (tmp_path / "m.joblib").exists()


def test_metricas_sao_json_estrito(treinado):
    pasta, _ = treinado
    texto = (pasta / "relatorios" / "metricas.json").read_text("utf-8")

    def recusar(constante):
        raise ValueError(f"{constante} não é JSON válido")

    json.loads(texto, parse_constant=recusar)


def test_segmento_vazio_vira_none_e_nao_nan(treinado):
    pasta, _ = treinado
    teste = gerar(50, semente=3)
    teste["chuva"] = 0  # nenhum pedido com chuva: dois segmentos ficam vazios
    modelos = {"regressão linear": joblib.load(pasta / "modelos" / "modelo.joblib")["pontual"]}
    linhas = por_segmento(teste, modelos)
    vazio = next(s for s in linhas if s["segmento"] == "com chuva")
    assert vazio["entregas"] == 0 and vazio["viés regressão linear"] is None
    json.dumps(linhas, allow_nan=False)


# --- Segunda rodada ---


def test_tipos_do_numpy_sao_aceitos(treinado):
    """Um valor tirado de uma coluna do pandas é numpy.int64: era recusado com "dia_semana vai de
    0 a 6", mesmo valendo 4."""
    pasta, _ = treinado
    pedido = {
        **PEDIDO,
        "dia_semana": np.int64(5),
        "hora": np.float32(20.5),
        "chuva": np.bool_(False),
        "distancia_km": np.float64(3.0),
        "itens": np.int32(2),
        "pedidos_na_fila": np.int64(4),
    }
    assert prever(pedido, pasta / "modelos" / "modelo.joblib") == prever(PEDIDO, pasta / "modelos" / "modelo.joblib")
    normalizado = validar(pedido)
    assert {k: type(v) for k, v in normalizado.items()} == {k: type(v) for k, v in validar(PEDIDO).items()}


@pytest.mark.parametrize(
    "campo, valor", [("itens", np.float64(2.0)), ("dia_semana", np.bool_(True)), ("distancia_km", np.float64("nan"))]
)
def test_tipos_do_numpy_invalidos_continuam_recusados(campo, valor):
    with pytest.raises(ValueError):
        validar({**PEDIDO, campo: valor})


def test_falha_ao_gravar_nao_destroi_o_modelo_anterior(treinado, tmp_path, monkeypatch):
    """Disco cheio no meio da gravação: o modelo antigo precisa continuar inteiro."""
    pasta, _ = treinado
    caminho = tmp_path / "m.joblib"
    caminho.write_bytes((pasta / "modelos" / "modelo.joblib").read_bytes())

    def disco_cheio(objeto, destino, *args, **kwargs):
        Path(destino).write_bytes(b"\\x80\\x04metade")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("previsao_entrega.treino.joblib.dump", disco_cheio)
    with pytest.raises(OSError, match="No space"):
        executar(gerar(2_000, semente=1), caminho, tmp_path / "rel")
    assert carregar(caminho)["formato"] == 2
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []  # sem lixo na pasta


# --- Terceira rodada: a faixa medida é a faixa mostrada ---


def test_faixa_mostrada_arredonda_para_o_minuto_mais_proximo():
    """Arredondar para fora deixava a faixa 1 min mais larga: 85% de cobertura, não os 80% medidos."""
    baixo, alto = faixa_mostrada(np.array([40.4, 40.6]), np.array([50.4, 50.6]), np.array([45.0, 45.0]))
    assert baixo.tolist() == [40, 41] and alto.tolist() == [50, 51]
    assert faixa_mostrada(-3.0, 20.0, 10.0) == (0, 20)  # nunca abaixo de zero
    assert faixa_mostrada(45.0, 38.0, 40.0) == (40, 40)  # quantis cruzados: a faixa contém a previsão


def test_cobertura_relatada_e_a_da_faixa_mostrada(treinado):
    """O relatório mede a faixa como o cliente a vê: refeita sobre todo o teste, a conta dá exatamente
    o número do relatório, e cada faixa que prever() mostra é a mesma que entrou nessa conta."""
    pasta, relatorio = treinado
    artefato = carregar(pasta / "modelos" / "modelo.joblib")
    _, teste = train_test_split(gerar(5_000, semente=1), test_size=0.2, random_state=0)  # o do fixture
    x, y = teste[COLUNAS_ENTRADA], teste[ALVO].to_numpy()
    margem = artefato["margem_faixa"]
    baixo, alto = faixa_mostrada(
        artefato["p10"].predict(x) - margem, artefato["p90"].predict(x) + margem, artefato["pontual"].predict(x)
    )
    calibrada = relatorio["faixa_p10_p90"]["calibrada"]
    assert ((y >= baixo) & (y <= alto)).mean() == calibrada["cobertura"]
    assert (alto - baixo).mean() == pytest.approx(calibrada["largura_media_min"])
    assert (y > alto).mean() == calibrada["chega_depois"]

    conferidas = 0
    for i in random.Random(4).sample(range(len(teste)), 200):
        try:
            previsao = prever(x.iloc[i].to_dict(), artefato)
        except ValueError:  # fora dos limites do treino: prever() recusa
            continue
        assert (previsao.minimo, previsao.maximo) == (baixo[i], alto[i])
        conferidas += 1
    assert conferidas >= 190
