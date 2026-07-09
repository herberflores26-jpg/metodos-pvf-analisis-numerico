
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy import symbols, sympify, lambdify
import numpy as np
# ============================================================
# 1. SOLVERS PARA EL PROBLEMA DE VALOR INICIAL (PVI)
# ============================================================
def euler_step(f, x, y, h):
    k = f(x, y)
    return [y[i] + h * k[i] for i in range(len(y))]

def heun_step(f, x, y, h):
    k1 = f(x, y)
    y_pred = [y[i] + h * k1[i] for i in range(len(y))]
    k2 = f(x + h, y_pred)
    return [y[i] + (h/2) * (k1[i] + k2[i]) for i in range(len(y))]

def rk4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, [y[i] + (h/2) * k1[i] for i in range(len(y))])
    k3 = f(x + h/2, [y[i] + (h/2) * k2[i] for i in range(len(y))])
    k4 = f(x + h,   [y[i] + h * k3[i] for i in range(len(y))])
    return [y[i] + (h/6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(len(y))]

SOLVERS = {
    "1": ("Euler", euler_step),
    "2": ("Heun", heun_step),
    "3": ("RK4", rk4_step)
}

# ============================================================
# 2. ALGORITMO DEL DISPARO LINEAL
# ============================================================
def linear_shooting(a, b, alpha, beta, n, p_func, q_func, r_func, solver):
    h = (b - a) / n
    x_vals = [a + i * h for i in range(n + 1)]

    state = [alpha, 0.0, 0.0, 1.0]
    y1_vals = [state[0]]
    y2_vals = [state[2]]

    def system(x, s):
        y1, y1p, y2, y2p = s
        dy1 = y1p
        dy1p = p_func(x) * y1p + q_func(x) * y1 + r_func(x)
        dy2 = y2p
        dy2p = p_func(x) * y2p + q_func(x) * y2
        return [dy1, dy1p, dy2, dy2p]

    x_actual = a
    for _ in range(n):
        state = solver(system, x_actual, state, h)
        x_actual += h
        y1_vals.append(state[0])
        y2_vals.append(state[2])

    y1_b = y1_vals[-1]
    y2_b = y2_vals[-1]

    if abs(y2_b) < 1e-12:
        raise ValueError("ERROR: y2(b) es prácticamente cero. El método falla.")

    C = (beta - y1_b) / y2_b
    y_vals = [y1_vals[i] + C * y2_vals[i] for i in range(len(x_vals))]

    return x_vals, y_vals, y1_vals, y2_vals, C

# ============================================================
# 3. VERIFICAR EXISTENCIA (Mejora 7)
# ============================================================
def verificar_existencia(a, b, p_func, q_func, r_func, n=100):
    advertencias = []
    x_sample = np.linspace(a, b, n)

    q_vals = [q_func(x) for x in x_sample]
    if any(q < -10 for q in q_vals):
        advertencias.append("⚠️ q(x) es muy negativa en algunos puntos. Puede haber oscilaciones rápidas.")

    p_vals = [p_func(x) for x in x_sample]
    if any(abs(p) > 100 for p in p_vals) or any(abs(q) > 100 for q in q_vals):
        advertencias.append("⚠️ Coeficientes muy grandes. El problema puede ser rígido.")

    r_vals = [r_func(x) for x in x_sample]
    if any(abs(r) > 100 for r in r_vals):
        advertencias.append("⚠️ Término fuente r(x) muy grande. Solución puede variar rápidamente.")

    if all(q >= 0 for q in q_vals):
        advertencias.append("✅ Condición suficiente de unicidad: q(x) >= 0 en todo el intervalo.")
    else:
        advertencias.append("ℹ️ q(x) toma valores negativos. La unicidad no está garantizada; verifique y2(b).")

    return advertencias

# ============================================================
# 4. MOSTRAR ECUACIÓN Y COMBINACIÓN LINEAL (Mejora 2)
# ============================================================
def mostrar_ecuacion(p_str, q_str, r_str, a, b, alpha, beta, C_opt):
    print("\n" + "="*60)
    print("📐 PROBLEMA DE VALOR DE FRONTERA RESUELTO")
    print("="*60)
    print(f"EDO: y'' = ({p_str}) y' + ({q_str}) y + ({r_str})")
    print(f"Condiciones: y({a}) = {alpha},  y({b}) = {beta}")
    print("-"*60)
    print("🔑 SOLUCIÓN COMO COMBINACIÓN LINEAL:")
    print(f"   y(x) = y1(x) + C * y2(x)")
    print(f"   con C = {C_opt:.8f}")
    print(f"   donde:")
    print(f"      y1(a) = {alpha}, y1'(a) = 0")
    print(f"      y2(a) = 0,     y2'(a) = 1")
    print("="*60)

# ============================================================
# 5. FUNCIONES DE GRÁFICAS (llamadas automáticamente)
# ============================================================
def graficar_solucion_y_disparos(x_sol, y_sol, y1_sol, y2_sol, C_opt, a, alpha, b, beta, nombre_metodo, n):
    plt.figure(figsize=(12, 6))
    plt.plot(x_sol, y1_sol, 'g--', linewidth=2, label=f'Disparo 1: y1(x) [y1(a)={alpha}, y1\'(a)=0]')
    plt.plot(x_sol, y2_sol, 'r--', linewidth=2, label=f'Disparo 2: y2(x) [y2(a)=0, y2\'(a)=1]')
    plt.plot(x_sol, y_sol, 'b-', linewidth=3, label=f'Solución final: y1(x) + {C_opt:.4f}·y2(x)')
    plt.plot(a, alpha, 'ko', markersize=8, label=f'Frontera ({a},{alpha})')
    plt.plot(b, beta, 'ko', markersize=8, label=f'Frontera ({b},{beta})')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'Disparo Lineal - {nombre_metodo} (n={n})')
    plt.grid(True); plt.legend()
    plt.show()

def graficar_comparacion_solvers(a, b, alpha, beta, n, p_func, q_func, r_func, x_sol):
    print("\n🔬 Comparando solvers con n =", n)
    resultados = {}
    for key, (nombre, solver) in SOLVERS.items():
        try:
            _, y_temp, _, _, _ = linear_shooting(
                a, b, alpha, beta, n, p_func, q_func, r_func, solver
            )
            resultados[nombre] = y_temp
        except Exception as e:
            print(f"   {nombre} falló: {e}")
            resultados[nombre] = None

    plt.figure(figsize=(12, 6))
    colores = {'Euler': 'orange', 'Heun': 'green', 'RK4': 'blue'}
    for nombre, y_vals in resultados.items():
        if y_vals is not None:
            plt.plot(x_sol, y_vals, 'o-', color=colores.get(nombre, 'gray'),
                     label=nombre, markersize=3, linewidth=1.5)

    plt.plot(a, alpha, 'ko', markersize=8, label=f'Frontera ({a},{alpha})')
    plt.plot(b, beta, 'ko', markersize=8, label=f'Frontera ({b},{beta})')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'Comparación de solvers - n={n}')
    plt.grid(True); plt.legend()
    plt.show()

    print("\n📊 Precisión en x = b (valor esperado β =", beta, "):")
    for nombre, y_vals in resultados.items():
        if y_vals is not None:
            error = abs(y_vals[-1] - beta)
            rel_error = error / (abs(beta) + 1e-12)
            print(f"   {nombre:6} -> Error absoluto: {error:.2e},  Error relativo: {rel_error:.2e}")

def animar_disparo(x_sol, y1_sol, y2_sol, C_opt, a, alpha, b, beta):
    try:
        rango = max(abs(C_opt), 1.0) * 2.5
        C_min = C_opt - rango
        C_max = C_opt + rango
        C_values = np.linspace(C_min, C_max, 70)

        fig_anim, ax_anim = plt.subplots(figsize=(12, 6))
        ax_anim.plot(x_sol, y1_sol, 'g--', linewidth=1.5, alpha=0.6, label='y1(x) (disparo fijo)')
        ax_anim.plot(x_sol, y2_sol, 'r--', linewidth=1.5, alpha=0.6, label='y2(x) (disparo fijo)')
        ax_anim.scatter([a], [alpha], color='k', s=100, zorder=5, label=f'Inicio ({a}, {alpha})')
        ax_anim.scatter([b], [beta], color='k', s=100, zorder=5, label=f'Objetivo ({b}, {beta})', marker='*')

        line, = ax_anim.plot([], [], 'b-', linewidth=2.5, label='Disparo actual')
        ax_anim.legend(loc='upper left')
        ax_anim.grid(True)
        ax_anim.set_xlabel('x')
        ax_anim.set_ylabel('y')
        ax_anim.set_xlim(a - 0.1*(b-a), b + 0.1*(b-a))

        def init():
            line.set_data([], [])
            return line,

        def update(C_val):
            y_shot = [y1_sol[i] + C_val * y2_sol[i] for i in range(len(x_sol))]
            line.set_data(x_sol, y_shot)

            if abs(C_val - C_opt) < 0.01 * max(1, abs(C_opt)):
                line.set_color('magenta')
                line.set_linewidth(4)
            else:
                line.set_color('blue')
                line.set_linewidth(2)

            max_y = max(y_shot + [alpha, beta])
            min_y = min(y_shot + [alpha, beta])
            margen = (max_y - min_y) * 0.3 + 1.0
            ax_anim.set_ylim(min_y - margen, max_y + margen)
            ax_anim.set_title(f'Disparo con pendiente inicial y\'(a) = C = {C_val:.4f}  (Óptimo: {C_opt:.4f})')
            return line,

        ani = FuncAnimation(fig_anim, update, frames=C_values, init_func=init, interval=80, repeat=True, blit=False)
        plt.show()
        print("\n✅ Animación finalizada.")
        print(f"🔍 La línea se vuelve MAGENTA cuando C se acerca al valor óptimo {C_opt:.4f}.")
    except Exception as e:
        print(f"\n❌ Error al generar la animación: {e}")

# ============================================================
# 6. PROGRAMA PRINCIPAL (INTERACTIVO SOLO PARA DATOS)
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("   MÉTODO DEL DISPARO LINEAL CON ANIMACIÓN")
    print("   EDO: y'' = p(x)y' + q(x)y + r(x)")
    print("="*50 + "\n")

    # --- 1. DEFINIR EL PVF ---
    print("Ingrese las funciones en términos de 'x' (ej: 2*x, sin(x), exp(x), -1):")
    p_str = input("  p(x) = ")
    q_str = input("  q(x) = ")
    r_str = input("  r(x) = ")

    def make_func(expr):
        return lambda x: eval(expr, {"x": x, "math": math})

    p_func = make_func(p_str)
    q_func = make_func(q_str)
    r_func = make_func(r_str)

    # --- 2. CONDICIONES DE FRONTERA ---
    a = float(input("Ingrese el extremo izquierdo a: "))
    b = float(input("Ingrese el extremo derecho  b: "))
    alpha = float(input("Ingrese y(a) = alpha: "))
    beta = float(input("Ingrese y(b) = beta: "))

    if a >= b:
        print("Error: a debe ser menor que b.")
        exit()

    # --- 3. VERIFICAR EXISTENCIA (Mejora 7) ---
    advertencias = verificar_existencia(a, b, p_func, q_func, r_func)
    print("\n" + "="*60)
    print("🔍 VERIFICACIÓN DE CONDICIONES DE EXISTENCIA Y UNICIDAD")
    print("="*60)
    for msg in advertencias:
        print(msg)
    print("="*60)

    # --- 4. TAMAÑO DE PASO ---
    n = int(input("\nIngrese el número de subintervalos n (tamaño de paso h = (b-a)/n): "))
    if n <= 0:
        print("Error: n debe ser un entero positivo.")
        exit()

    # --- 5. ELEGIR EL MÉTODO (solo para la solución principal) ---
    print("\nSeleccione el método para resolver el Problema de Valor Inicial:")
    print("  1. Euler (Orden 1)")
    print("  2. Heun  (Orden 2)")
    print("  3. RK4   (Orden 4)")
    opcion = input("Opción (1/2/3): ")

    if opcion not in SOLVERS:
        print("Opción no válida. Saliendo...")
        exit()

    nombre_metodo, solver_func = SOLVERS[opcion]

    # --- 6. EJECUTAR ---
    print(f"\nResolviendo con {nombre_metodo} y n = {n}...")
    try:
        x_sol, y_sol, y1_sol, y2_sol, C_opt = linear_shooting(
            a, b, alpha, beta, n, p_func, q_func, r_func, solver_func
        )
    except ValueError as e:
        print(f"\n{e}")
        exit()

    # --- 7. MOSTRAR ECUACIÓN Y COMBINACIÓN LINEAL (Mejora 2) ---
    mostrar_ecuacion(p_str, q_str, r_str, a, b, alpha, beta, C_opt)

    # --- 8. TABLA DE RESULTADOS ---
    print("\n" + "-"*60)
    print(f"{'  x  ':^10} | {'y(x) aproximada':^20} | {'Error (estimado)':^20}")
    print("-"*60)

    step_display = max(1, n // 20)
    for i in range(0, n+1, step_display):
        if n >= 4:
            try:
                _, y_fine, _, _, _ = linear_shooting(
                    a, b, alpha, beta, n*2, p_func, q_func, r_func, solver_func
                )
                idx_fine = i * 2
                if idx_fine < len(y_fine):
                    error_est = abs(y_sol[i] - y_fine[idx_fine])
                else:
                    error_est = 0.0
            except:
                error_est = 0.0
        else:
            error_est = 0.0

        print(f"{x_sol[i]:^10.6f} | {y_sol[i]:^20.10f} | {error_est:^20.2e}")

    print("-"*60)
    print(f"Paso h = {(b-a)/n:.6f}")
    print(f"Método usado: {nombre_metodo}")

    # ============================================================
    # 9. SALIDAS AUTOMÁTICAS (SIN PREGUNTAR)
    # ============================================================
    print("\n" + "="*60)
    print("📊 MOSTRANDO GRÁFICAS DE FORMA AUTOMÁTICA")
    print("   Cierra cada ventana para ver la siguiente.")
    print("="*60)

    # Gráfica 1: Solución + disparos
    graficar_solucion_y_disparos(x_sol, y_sol, y1_sol, y2_sol, C_opt, a, alpha, b, beta, nombre_metodo, n)

    # Gráfica 2: Comparación de solvers (Mejora 3)
    graficar_comparacion_solvers(a, b, alpha, beta, n, p_func, q_func, r_func, x_sol)

    # Gráfica 3: Animación del disparo (sugerencia)
    animar_disparo(x_sol, y1_sol, y2_sol, C_opt, a, alpha, b, beta)

    print("\n✅ Todas las gráficas han sido mostradas. Programa finalizado.")


