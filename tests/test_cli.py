"""Linha de comando de ponta a ponta, rodando numa pasta vazia."""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from previsao_entrega import __main__ as cli

RAIZ = Path(__file__).resolve().parent.parent
PREVER = ["prever", "--categoria", "pizzaria", "--dia", "sexta", "--hora", "20:30", "--distancia", "3"]


@pytest.fixture(scope="module")
def pasta_treinada(tmp_path_factory):
    pasta = tmp_path_factory.mktemp("cli")
    atual = Path.cwd()
    os.chdir(pasta)
    try:
        assert cli.main(["treinar", "--linhas", "2000"]) == 0
    finally:
        os.chdir(atual)
    return pasta


@pytest.fixture
def na_pasta(pasta_treinada, monkeypatch):
    monkeypatch.chdir(pasta_treinada)
    return pasta_treinada


def test_treinar_cria_os_arquivos_na_pasta_atual(na_pasta):
    """Antes, os arquivos iam para a pasta do código, qualquer que fosse a pasta atual."""
    for arquivo in (
        "modelos/modelo.joblib",
        "relatorios/metricas.json",
        "relatorios/importancia.png",
        "dados/amostra.csv",
    ):
        assert (na_pasta / arquivo).exists(), arquivo


def test_prever(na_pasta, capsys):
    assert cli.main(PREVER) == 0
    assert capsys.readouterr().out.startswith("Previsão: ")


def test_treinar_respeita_o_nome_do_modelo(tmp_path, monkeypatch, capsys):
    """Antes, --modelo meu.joblib gravava em modelo.joblib e dizia que tinha gravado em meu.joblib."""
    monkeypatch.chdir(tmp_path)
    assert cli.main(["treinar", "--linhas", "2000", "--modelo", "outro/meu.joblib", "--relatorios", "rel"]) == 0
    # Path(...) usa o separador do sistema: "outro\\meu.joblib" no Windows. O teste antigo esperava
    # "outro/meu.joblib" e falhou na primeira execução num Windows de verdade.
    esperado = f"Modelo salvo em {Path('outro') / 'meu.joblib'}. Gráficos e métricas na pasta rel."
    assert esperado in capsys.readouterr().out
    assert (tmp_path / "outro" / "meu.joblib").exists() and not (tmp_path / "modelos").exists()
    assert cli.main([*PREVER, "--modelo", "outro/meu.joblib"]) == 0


@pytest.mark.parametrize("linhas", ["0", "-5", "10", "1999", "abc"])
def test_poucas_linhas_sao_recusadas(linhas, capsys):
    with pytest.raises(SystemExit):
        cli.main(["treinar", "--linhas", linhas])
    assert "--linhas" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argumentos, mensagem",
    [
        (["--distancia", "50"], "distância 50 km está fora do que o modelo conhece"),
        (["--hora", "3"], "hora 3h00 está fora"),
        (["--fila", "200"], "pedidos na fila 200 está fora"),
        (["--distancia", "inf"], "maior que zero"),
        (["--categoria", "sushi"], "categoria deve ser uma de"),
        (["--dia", "9"], "dia_semana vai de 0"),
    ],
)
def test_erros_viram_uma_linha(na_pasta, capsys, argumentos, mensagem):
    assert cli.main([*PREVER, *argumentos]) == 1
    erro = capsys.readouterr().err
    assert erro.startswith("Erro: ") and mensagem in erro and "Traceback" not in erro


def test_sem_modelo(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(PREVER) == 1
    assert "modelo não encontrado" in capsys.readouterr().err


def test_modelo_corrompido(na_pasta, tmp_path, capsys):
    (tmp_path / "ruim.joblib").write_bytes(b"lixo")
    assert cli.main([*PREVER, "--modelo", str(tmp_path / "ruim.joblib")]) == 1
    assert "não consegui ler o modelo" in capsys.readouterr().err


@pytest.mark.parametrize(
    "texto, hora",
    [
        ("20.5", 20.5),
        ("20,5", 20.5),
        ("20:30", 20.5),
        ("20h30", 20.5),
        ("20h", 20.0),
        ("9:05", 9 + 5 / 60),
        (" 21H15 ", 21.25),
    ],
)
def test_ler_hora(texto, hora):
    assert cli.ler_hora(texto) == pytest.approx(hora)


@pytest.mark.parametrize("texto, km", [("2,5", 2.5), ("2.5", 2.5), (" 4 ", 4.0), ("10", 10.0)])
def test_ler_distancia(texto, km):
    assert cli.ler_distancia(texto) == km


@pytest.mark.parametrize("texto", ["dois", "2,5,1", ""])
def test_distancia_invalida(texto):
    with pytest.raises(argparse.ArgumentTypeError):
        cli.ler_distancia(texto)


@pytest.mark.parametrize("texto", ["20:75", "vinte", "20:3", ""])
def test_hora_invalida(texto):
    with pytest.raises(argparse.ArgumentTypeError):
        cli.ler_hora(texto)


@pytest.mark.parametrize(
    "texto, dia",
    [
        ("0", 0),
        ("6", 6),
        ("segunda", 0),
        ("segunda-feira", 0),
        ("Terça", 1),
        ("qua", 2),
        ("qui", 3),
        ("sex", 4),
        ("sábado", 5),
        ("SAB", 5),
        ("dom", 6),
    ],
)
def test_ler_dia(texto, dia):
    assert cli.ler_dia(texto) == dia


@pytest.mark.parametrize(
    "texto", ["s", "qu", "d", "t", "do", "feriado", ""]
)  # "d" só bate com domingo, mas é curto demais
def test_dia_invalido(texto):
    with pytest.raises(argparse.ArgumentTypeError):
        cli.ler_dia(texto)


def test_aviso_de_versao_vai_para_o_erro_padrao(na_pasta, monkeypatch, capsys):
    monkeypatch.setattr("previsao_entrega.prever.sklearn.__version__", "9.9.9")
    assert cli.main(PREVER) == 0
    saida = capsys.readouterr()
    assert saida.out.startswith("Previsão: ") and "Aviso: o modelo foi treinado com scikit-learn" in saida.err


def _rodar(argumentos, pasta, **kw):
    ambiente = {**os.environ, "PYTHONPATH": str(RAIZ), **kw.pop("env", {})}
    return subprocess.run(
        [sys.executable, "-m", "previsao_entrega", *argumentos],
        cwd=pasta,
        env=ambiente,
        capture_output=True,
        check=False,
        **kw,
    )


def test_terminal_do_windows_nao_quebra(pasta_treinada):
    """Com cp1252, o "≤" da tabela derrubava o programa depois de o modelo já ter sido salvo."""
    r = _rodar(
        ["treinar", "--linhas", "2000", "--modelo", "w.joblib", "--relatorios", "w"],
        pasta_treinada,
        env={"PYTHONIOENCODING": "cp1252"},
    )
    assert r.returncode == 0, r.stderr.decode("cp1252", "replace")
    assert b"Traceback" not in r.stderr and b"Modelo salvo em w.joblib" in r.stdout


def test_ctrl_c_de_verdade(tmp_path):
    if sys.platform == "win32":
        pytest.skip("SIGINT só no Unix")
    processo = subprocess.Popen(
        [sys.executable, "-m", "previsao_entrega", "treinar", "--linhas", "200000"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(RAIZ)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)
    processo.send_signal(signal.SIGINT)
    _, erro = processo.communicate(timeout=60)
    assert processo.returncode == 130 and b"Traceback" not in erro and b"Interrompido." in erro


def test_segmento_sem_entregas_no_teste_aparece_escrito(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    modelo = {
        "modelo": "m",
        "mae_cv": 1.0,
        "mae_cv_desvio": 0.1,
        "mae_teste": 1.0,
        "r2_teste": 0.5,
        "dentro_10min": 0.9,
    }
    faixa = {"cobertura": 0.8, "largura_media_min": 10.0, "chega_depois": 0.1}
    relatorio = {
        "linhas_treino": 1600,
        "linhas_teste": 400,
        "modelos": [modelo],
        "segmentos": [
            {
                "segmento": "chuva e mais de 5 km",
                "entregas": 0,
                "viés regressão linear": None,
                "viés gradient boosting": None,
            }
        ],
        "faixa_p10_p90": {"sem_calibrar": faixa, "calibrada": faixa},
        "importancia_min": {"distancia_km": 5.0},
    }
    monkeypatch.setattr(cli, "executar", lambda *a, **k: relatorio)
    assert cli.main(["treinar", "--linhas", "2000"]) == 0
    assert "chuva e mais de 5 km" in capsys.readouterr().out.split("sem entregas no teste")[0]
