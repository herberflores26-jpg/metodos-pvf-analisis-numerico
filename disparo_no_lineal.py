


import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# 1. SOLVERS PARA EL PROBLEMA DE VALOR INICIAL (PVI)
# ============================================================
def euler_step(f, x, y, h):
    """Euler explícito para sistemas"""
    k = f(x, y)
    return [y[i] + h * k[i] for i in range(len(y))]

def heun_step(f, x, y, h):
    """Método de Heun (RK2)"""
    k1 = f(x, y)
    y_pred = [y[i] + h * k1[i] for i in range(len(y))]
    k2 = f(x + h, y_pred)
    return [y[i] + (h/2) * (k1[i] + k2[i]) for i in range(len(y))]

def rk4_step(f, x, y, h):
    """Runge-Kutta 4to orden"""
    k1 = f(x, y)
    k2 = f(x + h/2, [y[i] + (h/2) * k1[i] for i in range(len(y))])
    k3 = f(x + h/2, [y[i] + (h/2) * k2[i] for i in range(len(y))])
    k4 = f(x + h,   [y[i] + h * k3[i] for i in range(len(y))])
    return [y[i] + (h/6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(len(y))]

SOLVERS_PVI = {
    "1": ("Euler", euler_step),
    "2": ("Heun", heun_step),
    "3": ("RK4", rk4_step)
}

# ============================================================
# 2. FUNCIÓN QUE RESUELVE EL PVI DADO UN PARÁMETRO t = y'(a)
# ============================================================
def resolver_pvi(a, b, alpha, t, n, f_func, solver):
    """
    Resuelve el PVI: y' = z, z' = f(x, y, z)
    con y(a) = alpha, z(a) = t
    Retorna: x_vals, y_vals, z_vals
    """
    h = (b - a) / n
    x_vals = [a + i * h for i in range(n + 1)]
    y_vals = [alpha]
    z_vals = [t]
    
    # Estado inicial: [y, z]
    state = [alpha, t]
    
    def system(x, s):
        y, z = s
        dy = z
        dz = f_func(x, y, z)   # f(x, y, z) = y''
        return [dy, dz]
    
    x_actual = a
    for _ in range(n):
        state = solver(system, x_actual, state, h)
        x_actual += h
        y_vals.append(state[0])
        z_vals.append(state[1])
    
    return x_vals, y_vals, z_vals

# ============================================================
# 3. FUNCIÓN DE DISPARO (error en la frontera)
# ============================================================
def disparo(t, a, b, alpha, beta, n, f_func, solver):
    """Devuelve phi(t) = y(b; t) - beta"""
    _, y_vals, _ = resolver_pvi(a, b, alpha, t, n, f_func, solver)
    return y_vals[-1] - beta

# ============================================================
# 4. MÉTODOS DE BÚSQUEDA DE RAÍCES
# ============================================================
def biseccion(f, t_a, t_b, tol=1e-8, max_iter=100):
    """Método de bisección para encontrar raíz de f(t)=0"""
    fa = f(t_a)
    fb = f(t_b)
    if fa * fb >= 0:
        raise ValueError("La función debe cambiar de signo en el intervalo.")
    
    iteraciones = 0
    historial = []  # guarda (t, f(t)) para graficar
    for _ in range(max_iter):
        t_c = (t_a + t_b) / 2
        fc = f(t_c)
        historial.append((t_c, fc))
        if abs(fc) < tol or (t_b - t_a)/2 < tol:
            return t_c, iteraciones, historial
        if fa * fc < 0:
            t_b = t_c
            fb = fc
        else:
            t_a = t_c
            fa = fc
        iteraciones += 1
    return (t_a + t_b)/2, iteraciones, historial

def newton(f, t0, df=None, tol=1e-8, max_iter=100, h=1e-6):
    """Método de Newton con derivada numérica si no se provee"""
    iteraciones = 0
    historial = []
    t = t0
    for _ in range(max_iter):
        ft = f(t)
        historial.append((t, ft))
        if abs(ft) < tol:
            return t, iteraciones, historial
        # Derivada numérica
        if df is None:
            df_t = (f(t + h) - f(t - h)) / (2*h)
        else:
            df_t = df(t)
        if abs(df_t) < 1e-12:
            raise ValueError("Derivada cercana a cero. Newton falla.")
        t = t - ft / df_t
        iteraciones += 1
    return t, iteraciones, historial

# ============================================================
# 5. VERIFICACIÓN DE CONDICIONES (advertencias)
# ============================================================
def verificar_existencia_no_lineal(a, b, f_func, n=100):
    """Verifica algunas condiciones básicas (no riguroso)"""
    advertencias = []
    x_sample = np.linspace(a, b, n)
    
    # Evaluamos f en algunos puntos con y=0, z=0 para tener una idea
    try:
        f_vals = [f_func(x, 0, 0) for x in x_sample]
        if any(abs(fv) > 100 for fv in f_vals):
            advertencias.append("⚠️ f(x,0,0) es muy grande en algunos puntos. Posible rigidez.")
    except:
        advertencias.append("⚠️ No se pudo evaluar f en (x,0,0). Verifique la función.")
    
    # Sugerencia sobre unicidad: para problemas no lineales no hay criterio simple
    advertencias.append("ℹ️ Para EDOs no lineales, la unicidad no está garantizada globalmente.")
    advertencias.append("   El método del disparo puede encontrar una solución, pero pueden existir múltiples.")
    return advertencias

# ============================================================
# 6. MOSTRAR ECUACIÓN Y CONDICIONES
# ============================================================
def mostrar_ecuacion_no_lineal(f_str, a, b, alpha, beta):
    print("\n" + "="*60)
    print("📐 PROBLEMA DE VALOR DE FRONTERA NO LINEAL")
    print("="*60)
    print(f"EDO: y'' = f(x, y, y')  con  f(x,y,z) = {f_str}")
    print(f"Condiciones: y({a}) = {alpha},  y({b}) = {beta}")
    print("="*60)

# ============================================================
# 7. FUNCIONES DE GRÁFICAS (automáticas)
# ============================================================
def graficar_solucion_final(x_sol, y_sol, a, alpha, b, beta, nombre_metodo, n, t_opt):
    plt.figure(figsize=(10, 5))
    plt.plot(x_sol, y_sol, 'b-', linewidth=2, label='Solución final')
    plt.plot(a, alpha, 'ko', markersize=8, label=f'Frontera ({a},{alpha})')
    plt.plot(b, beta, 'ko', markersize=8, label=f'Frontera ({b},{beta})')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'Solución del PVF no lineal - {nombre_metodo} (n={n}, t*={t_opt:.6f})')
    plt.grid(True); plt.legend()
    plt.show()

def graficar_comparacion_busqueda(t_vals, errores_hist, metodos):
    """Grafica la evolución de la raíz para cada método"""
    plt.figure(figsize=(12, 5))
    for nombre, historial in metodos.items():
        if historial:
            ts = [p[0] for p in historial]
            fs = [p[1] for p in historial]
            plt.semilogy(range(len(ts)), [abs(f) for f in fs], 'o-', label=nombre)
    plt.xlabel('Iteración'); plt.ylabel('|φ(t)|')
    plt.title('Convergencia de los métodos de búsqueda de raíz')
    plt.grid(True); plt.legend()
    plt.show()

def animar_disparo_no_lineal(a, b, alpha, beta, n, f_func, solver, t_opt, t_min, t_max, nombre_metodo):
    """Animación variando t = y'(a) y mostrando las trayectorias"""
    try:
        t_values = np.linspace(t_min, t_max, 60)
        
        fig_anim, ax_anim = plt.subplots(figsize=(12, 6))
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
        
        def update(t_val):
            x_vals, y_vals, _ = resolver_pvi(a, b, alpha, t_val, n, f_func, solver)
            line.set_data(x_vals, y_vals)
            
            if abs(t_val - t_opt) < 0.01 * max(1, abs(t_opt)):
                line.set_color('magenta')
                line.set_linewidth(4)
            else:
                line.set_color('blue')
                line.set_linewidth(2)
            
            max_y = max(max(y_vals), alpha, beta)
            min_y = min(min(y_vals), alpha, beta)
            margen = (max_y - min_y) * 0.3 + 1.0
            ax_anim.set_ylim(min_y - margen, max_y + margen)
            ax_anim.set_title(f'Disparo con t = y\'(a) = {t_val:.4f}  (Óptimo: {t_opt:.4f})')
            return line,
        
        ani = FuncAnimation(fig_anim, update, frames=t_values, init_func=init, interval=80, repeat=True, blit=False)
        plt.show()
        print("\n✅ Animación finalizada.")
        print(f"🔍 La línea se vuelve MAGENTA cuando t se acerca al valor óptimo {t_opt:.4f}.")
    except Exception as e:
        print(f"\n❌ Error en la animación: {e}")

# ============================================================
# 8. PROGRAMA PRINCIPAL (INTERACTIVO)
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("   MÉTODO DEL DISPARO NO LINEAL")
    print("   EDO: y'' = f(x, y, y')")
    print("="*50 + "\n")
    
    # --- 1. DEFINIR LA EDO NO LINEAL ---
    print("Ingrese la función f(x, y, y') en términos de 'x', 'y' y 'z' (donde z = y'):")
    print("Ejemplo: -y + sin(x)  o  -z**2 + y  o  exp(x)*z + y**2")
    f_str = input("  f(x, y, z) = ")
    
    def make_func(expr):
        # Definimos la función que evalúa la expresión con math disponible
        return lambda x, y, z: eval(expr, {"x": x, "y": y, "z": z, "math": math})
    
    f_func = make_func(f_str)
    
    # --- 2. CONDICIONES DE FRONTERA ---
    a = float(input("Ingrese el extremo izquierdo a: "))
    b = float(input("Ingrese el extremo derecho  b: "))
    alpha = float(input("Ingrese y(a) = alpha: "))
    beta = float(input("Ingrese y(b) = beta: "))
    
    if a >= b:
        print("Error: a debe ser menor que b.")
        exit()
    
    # --- 3. VERIFICAR EXISTENCIA (advertencias) ---
    advertencias = verificar_existencia_no_lineal(a, b, f_func)
    print("\n" + "="*60)
    print("🔍 VERIFICACIÓN DE CONDICIONES")
    print("="*60)
    for msg in advertencias:
        print(msg)
    print("="*60)
    
    # --- 4. TAMAÑO DE PASO ---
    n = int(input("\nIngrese el número de subintervalos n (tamaño de paso h = (b-a)/n): "))
    if n <= 0:
        print("Error: n debe ser un entero positivo.")
        exit()
    
    # --- 5. ELEGIR EL MÉTODO PARA EL PVI ---
    print("\nSeleccione el método para resolver el Problema de Valor Inicial:")
    print("  1. Euler (Orden 1)")
    print("  2. Heun  (Orden 2)")
    print("  3. RK4   (Orden 4)")
    opcion_pvi = input("Opción (1/2/3): ")
    
    if opcion_pvi not in SOLVERS_PVI:
        print("Opción no válida. Saliendo...")
        exit()
    
    nombre_pvi, solver_pvi = SOLVERS_PVI[opcion_pvi]
    
    # --- 6. ELEGIR MÉTODO DE BÚSQUEDA DE RAÍZ ---
    print("\nSeleccione el método para encontrar la pendiente inicial t = y'(a):")
    print("  1. Bisección (requiere intervalo con cambio de signo)")
    print("  2. Newton   (requiere una estimación inicial)")
    opcion_raiz = input("Opción (1/2): ")
    
    if opcion_raiz not in ["1", "2"]:
        print("Opción no válida. Saliendo...")
        exit()
    
    # Definimos la función de disparo (con los parámetros fijos)
    def phi(t):
        return disparo(t, a, b, alpha, beta, n, f_func, solver_pvi)
    
    # --- 7. EJECUTAR LA BÚSQUEDA ---
    t_opt = None
    historial_metodos = {}
    
    if opcion_raiz == "1":
        # Bisección: pedir intervalo
        print("\nPara bisección, necesita un intervalo [t0, t1] donde phi cambie de signo.")
        t0 = float(input("Ingrese t0 (extremo izquierdo): "))
        t1 = float(input("Ingrese t1 (extremo derecho): "))
        try:
            t_opt, iter_bisec, hist_bisec = biseccion(phi, t0, t1)
            historial_metodos["Bisección"] = hist_bisec
            print(f"\n✅ Raíz encontrada por bisección: t* = {t_opt:.8f} en {iter_bisec} iteraciones.")
        except ValueError as e:
            print(f"\n❌ Error en bisección: {e}")
            exit()
    else:
        # Newton: pedir estimación inicial
        t0 = float(input("Ingrese la estimación inicial t0: "))
        try:
            t_opt, iter_newton, hist_newton = newton(phi, t0)
            historial_metodos["Newton"] = hist_newton
            print(f"\n✅ Raíz encontrada por Newton: t* = {t_opt:.8f} en {iter_newton} iteraciones.")
        except ValueError as e:
            print(f"\n❌ Error en Newton: {e}")
            exit()
    
    # --- 8. RESOLVER EL PVI CON EL t ÓPTIMO ---
    x_sol, y_sol, z_sol = resolver_pvi(a, b, alpha, t_opt, n, f_func, solver_pvi)
    
    # --- 9. MOSTRAR ECUACIÓN Y CONDICIONES ---
    mostrar_ecuacion_no_lineal(f_str, a, b, alpha, beta)
    print(f"Pendiente inicial óptima: y'({a}) = t* = {t_opt:.8f}")
    
    # --- 10. TABLA DE RESULTADOS ---
    print("\n" + "-"*60)
    print(f"{'  x  ':^10} | {'y(x) aproximada':^20} | {'z(x) = y\'':^20}")
    print("-"*60)
    step_display = max(1, n // 20)
    for i in range(0, n+1, step_display):
        print(f"{x_sol[i]:^10.6f} | {y_sol[i]:^20.10f} | {z_sol[i]:^20.10f}")
    print("-"*60)
    print(f"Paso h = {(b-a)/n:.6f}")
    print(f"Método PVI: {nombre_pvi}")
    
    # --- 11. GRÁFICAS AUTOMÁTICAS ---
    print("\n" + "="*60)
    print("📊 MOSTRANDO GRÁFICAS DE FORMA AUTOMÁTICA")
    print("   Cierra cada ventana para ver la siguiente.")
    print("="*60)
    
    # Gráfica 1: Solución final
    graficar_solucion_final(x_sol, y_sol, a, alpha, b, beta, nombre_pvi, n, t_opt)
    
    # Gráfica 2: Convergencia del método de búsqueda (si hay historial)
    if historial_metodos:
        graficar_comparacion_busqueda(None, None, historial_metodos)
    
    # Gráfica 3: Animación del disparo
    # Determinamos un rango para t alrededor del óptimo
    rango_t = max(abs(t_opt), 1.0) * 2.0
    t_min = t_opt - rango_t
    t_max = t_opt + rango_t
    animar_disparo_no_lineal(a, b, alpha, beta, n, f_func, solver_pvi, t_opt, t_min, t_max, nombre_pvi)
    
    print("\n✅ Programa finalizado.")
