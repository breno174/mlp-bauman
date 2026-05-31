import re
import os
from datetime import datetime, timedelta

# Diretório onde os experimentos são salvos
EXP_DIR = os.path.join("src", "experiments", "multi")

if not os.path.exists(EXP_DIR):
    print(f"Diretório não encontrado: {EXP_DIR}")
else:
    # Definir o intervalo: de ontem para hoje (desde 00:00:00 de ontem)
    hoje = datetime.now()
    ontem = hoje - timedelta(days=1)
    ontem_zero_hora = ontem.replace(hour=0, minute=0, second=0, microsecond=0)

    print(
        f"Filtrando experimentos desde: {ontem_zero_hora.strftime('%d/%m/%Y %H:%M:%S')}"
    )
    print(f"{'ARQUIVO':<50} | {'DATA':<12} | {'HORA':<10}")
    print("-" * 80)

    # Lista todos os arquivos .json no diretório
    files = [f for f in os.listdir(EXP_DIR) if f.endswith(".json")]

    # Ordena para mostrar os mais recentes primeiro
    files.sort(reverse=True)

    count = 0
    for filename in files:
        # Expressão regular para encontrar a sequência numérica (8 dígitos + 6 dígitos)
        match = re.search(r"(\d{8})_(\d{6})", filename)

        if match:
            date_part = match.group(1)
            time_part = match.group(2)

            # Converte para um objeto datetime
            try:
                dt_obj = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")

                # Filtro: Apenas de ontem para hoje
                if dt_obj >= ontem_zero_hora:
                    full_path = os.path.abspath(os.path.join(EXP_DIR, filename))
                    print(
                        f"{filename:<50} | {dt_obj.strftime('%d/%m/%Y'):<12} | {dt_obj.strftime('%H:%M:%S'):<10}"
                    )
                    # print(f"  Caminho: {full_path}")
                    count += 1
            except ValueError:
                continue

    if count == 0:
        print("\nNenhum experimento encontrado desde ontem.")
    else:
        print(f"\nTotal encontrado: {count} arquivos.")
