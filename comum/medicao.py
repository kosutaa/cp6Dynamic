"""Medição de tempo (time.perf_counter) e memória de pico (tracemalloc)."""

import time
import tracemalloc


def medir(funcao, *args, **kwargs):
    """Executa funcao(*args, **kwargs) e retorna (resultado, segundos, pico_memoria_KiB)."""
    tracemalloc.start()
    inicio = time.perf_counter()
    try:
        resultado = funcao(*args, **kwargs)
        segundos = time.perf_counter() - inicio
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return resultado, segundos, pico / 1024


def medir_tempo(funcao, *args, **kwargs):
    """Somente tempo (sem overhead do tracemalloc), para comparações de desempenho."""
    inicio = time.perf_counter()
    resultado = funcao(*args, **kwargs)
    return resultado, time.perf_counter() - inicio
