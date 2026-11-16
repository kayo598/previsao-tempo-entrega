"""Linha de comando.

previsao-entrega treinar
previsao-entrega prever --categoria pizzaria --dia sexta --hora 20:30 --distancia 4 --itens 3 --fila 6 --chuva
(sem instalar: python -m previsao_entrega ...)

Os arquivos ficam na pasta atual: modelos/, relatorios/ e dados/.
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

from .dados import gerar
from .prever import ErroModelo, prever
from .treino import MIN_LINHAS, executar

MODELO_PADRAO = Path("modelos") / "modelo.joblib"
DIAS = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]


def tolerar_terminal_sem_acentos() -> None:
    """No Windows, com a saída redirecionada, o terminal usa cp1252: o "≤" da tabela de resultados
    derrubava o programa depois de o modelo já ter sido salvo. Os caracteres que não cabem viram "?"."""
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(errors="replace")


def ler_hora(texto: str) -> float:
    """Aceita "20.5", "20,5", "20:30", "20h30" e "20h"."""
    limpo = texto.strip().lower().replace(",", ".")
    formato = re.fullmatch(r"(\d{1,2})\s*[:h]\s*(\d{2})?", limpo)
    if formato:
        horas, minutos = int(formato.group(1)), int(formato.group(2) or 0)
        if minutos >= 60:
            raise argparse.ArgumentTypeError(f"'{texto}': os minutos vão de 00 a 59")
        return horas + minutos / 60
    try:
        return float(limpo)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{texto}' não é uma hora; use 20:30, 20h30 ou 20.5") from None


def ler_distancia(texto: str) -> float:
    """Aceita vírgula decimal, como se escreve no Brasil: "2,5" ou "2.5"."""
    try:
        return float(texto.strip().replace(",", "."))
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{texto}' não é uma distância; use km, como 2,5 ou 2.5") from None


def ler_dia(texto: str) -> int:
    """Aceita o número (0 = segunda ... 6 = domingo) ou o nome: "sexta", "sex", "sábado", "sab"."""
    limpo = "".join(c for c in unicodedata.normalize("NFKD", texto.strip().lower()) if not unicodedata.combining(c))
    limpo = limpo.removesuffix("-feira")
    if limpo.isdigit():
        return int(limpo)
    candidatos = [i for i, dia in enumerate(DIAS) if len(limpo) >= 3 and dia.startswith(limpo)]
    if len(candidatos) == 1:
        return candidatos[0]
    raise argparse.ArgumentTypeError(f"'{texto}' não é um dia; use 0 a 6 (0 = segunda) ou o nome, como 'sexta'")


def ler_linhas(texto: str) -> int:
    try:
        n = int(texto)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{texto}' não é um número inteiro") from None
    if n < MIN_LINHAS:
        raise argparse.ArgumentTypeError(f"são precisas pelo menos {MIN_LINHAS} entregas para treinar")
    return n


def cmd_treinar(args) -> int:
    df = gerar(args.linhas, args.semente)
    Path("dados").mkdir(exist_ok=True)
    df.head(200).to_csv(Path("dados") / "amostra.csv", index=False)
    print(f"Treinando com {args.linhas:,} entregas simuladas...".replace(",", "."), flush=True)
    r = executar(df, args.modelo, args.relatorios)
    print(f"Treino: {r['linhas_treino']} entregas · teste: {r['linhas_teste']}\n")
    print(f"{'modelo':22} {'MAE CV':>12} {'MAE teste':>10} {'R²':>6} {'erro ≤10 min':>13}")
    for m in r["modelos"]:
        print(
            f"{m['modelo']:22} {m['mae_cv']:6.2f} ± {m['mae_cv_desvio']:.2f} {m['mae_teste']:10.2f}"
            f" {m['r2_teste']:6.2f} {m['dentro_10min']:13.0%}"
        )
    print(f"\n{'segmento':22} {'entregas':>8} {'viés linear':>12} {'viés boosting':>14}")
    for s in r["segmentos"]:
        linear, boosting = s["viés regressão linear"], s["viés gradient boosting"]
        if linear is None or boosting is None:
            print(f"{s['segmento']:22} {s['entregas']:8} {'sem entregas no teste':>27}")
            continue
        print(f"{s['segmento']:22} {s['entregas']:8} {linear:+12.2f} {boosting:+14.2f}")
    f = r["faixa_p10_p90"]
    for nome in ("sem_calibrar", "calibrada"):
        print(
            f"Faixa {nome.replace('_', ' '):13}: contém {f[nome]['cobertura']:.0%} das entregas"
            f" (meta 80%), largura média {f[nome]['largura_media_min']:.1f} min,"
            f" chega depois da faixa {f[nome]['chega_depois']:.0%}"
        )
    print(
        "Importância (min de erro a mais ao embaralhar):",
        ", ".join(f"{k} {v}" for k, v in r["importancia_min"].items()),
    )
    print(f"\nModelo salvo em {args.modelo}. Gráficos e métricas na pasta {args.relatorios}.")
    return 0


def cmd_prever(args) -> int:
    pedido = {
        "categoria": args.categoria,
        "dia_semana": args.dia,
        "hora": args.hora,
        "chuva": int(args.chuva),
        "distancia_km": args.distancia,
        "itens": args.itens,
        "pedidos_na_fila": args.fila,
    }
    try:
        previsao = prever(pedido, args.modelo)
    except (ValueError, ErroModelo) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    print(previsao)
    for aviso in previsao.avisos:
        print(f"Aviso: {aviso}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    tolerar_terminal_sem_acentos()
    p = argparse.ArgumentParser(prog="previsao-entrega", description="Previsão de tempo de entrega.")
    sub = p.add_subparsers(dest="comando", required=True)

    t = sub.add_parser("treinar", help="gera os dados, compara os modelos e salva o gradient boosting")
    t.add_argument("--linhas", type=ler_linhas, default=20_000, help=f"entregas simuladas (mínimo {MIN_LINHAS})")
    t.add_argument("--semente", type=int, default=42)
    t.add_argument("--relatorios", type=Path, default=Path("relatorios"), help="pasta dos gráficos e métricas")

    pr = sub.add_parser("prever", help="prevê o tempo de uma entrega")
    pr.add_argument("--categoria", required=True, help="pizzaria, japones, lanches ou brasileira")
    pr.add_argument("--dia", type=ler_dia, required=True, help="0 a 6 (0 = segunda) ou o nome, como 'sexta'")
    pr.add_argument("--hora", type=ler_hora, required=True, help="ex.: 20:30, 20h30 ou 20.5")
    pr.add_argument("--distancia", type=ler_distancia, required=True, help="km (ex.: 4, 2,5 ou 2.5)")
    pr.add_argument("--itens", type=int, default=2)
    pr.add_argument("--fila", type=int, default=3, help="pedidos na fila do restaurante")
    pr.add_argument("--chuva", action="store_true")

    for s in (t, pr):
        s.add_argument("--modelo", type=Path, default=MODELO_PADRAO, help="arquivo do modelo")
    args = p.parse_args(argv)

    try:
        return cmd_treinar(args) if args.comando == "treinar" else cmd_prever(args)
    except KeyboardInterrupt:
        print("\nInterrompido.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
