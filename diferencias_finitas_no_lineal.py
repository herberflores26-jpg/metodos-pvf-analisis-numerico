
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy import symbols, sympify, lambdify
import numpy as np

# ============================================================
# 1. FUNCIONES PARA DISCRETIZACIÓN
# ============================================================
def construir_sistema_no_lineal(a, b, alpha, beta, n, f_func):
    """
    Construye el sistema no lineal F(y) = 0 para el problema:
    y'' = f(x, y, y'),  y(a)=alpha, y(b)=beta
    n = número de puntos interiores.
    Retorna:
        x_interior: lista de puntos interiores
        F: función que evalúa el sistema (vector de longitud n)
        J: función que evalúa el Jacobiano (opcional, usaremos numérico)
    """
    h = (b - a) / (n + 1)
    x_interior = [a + (i+1)*h for i in range(n)]
    
    def F(y):
        """Evalúa el sistema F(y) = 0. y es un array de longitud n."""
        F_vec = np.zeros(n)
        for i in range(n):
            xi = x_interior[i]
            # Aproximación de y'' y y' usando diferencias centradas
            if i == 0:
                y_menos = alpha
                y_mas = y[i+1]
            elif i == n-1:
                y_menos = y[i-1]
                y_mas = beta
            else:
                y_menos = y[i-1]
                y_mas = y[i+1]
            
            y_actual = y[i]
            yprima = (y_mas - y_menos) / (2*h)
            ybi = (y_menos - 2*y_actual + y_mas) / (h*h)
            
            # Ecuación: y'' - f(x, y, y') = 0
            F_vec[i] = ybi - f_func(xi, y_actual, yprima)
        return F_vec
    
    # Para el Jacobiano usaremos diferenciación numérica en Newton
    return x_interior, F

# ============================================================
# 2. MÉTODO DE NEWTON CON JACOBIANO NUMÉRICO
# ============================================================
def newton_sistema(F, y0, tol=1e-8, max_iter=100, relajacion=1.0, verbose=False):
    """
    Resuelve F(y) = 0 usando Newton con Jacobiano numérico.
    y0: estimación inicial (array)
    relajacion: factor de relajación (1.0 = Newton puro, <1.0 para problemas difíciles)
    Retorna: (y_sol, iteraciones, historial)
    """
    n = len(y0)
    y = np.array(y0, dtype=float)
    historial = [y.copy()]  # Guardamos todas las iteraciones para animación
    
    for k in range(max_iter):
        Fy = F(y)
        norm_F = np.linalg.norm(Fy)
        if verbose:
            print(f"Iteración {k}: ||F|| = {norm_F:.2e}")
        if norm_F < tol:
            return y, k+1, historial
        
        # Calcular Jacobiano numérico
        J = np.zeros((n, n))
        epsilon = 1e-8
        for i in range(n):
            y_plus = y.copy()
            y_plus[i] += epsilon
            F_plus = F(y_plus)
            J[:, i] = (F_plus - Fy) / epsilon
        
        # Resolver J * delta = -F
        try:
            delta = np.linalg.solve(J, -Fy)
        except np.linalg.LinAlgError:
            raise ValueError("Jacobiano singular. Newton falló.")
        
        # Actualizar con relajación
        y = y + relajacion * delta
        historial.append(y.copy())
    
    raise ValueError(f"No convergió en {max_iter} iteraciones.")

# ============================================================
# 3. FUNCIÓN PRINCIPAL DE DIFERENCIAS FINITAS NO LINEAL
# ============================================================
def diferencias_finitas_no_lineal(a, b, alpha, beta, n, f_func, 
                                   y0=None, tol=1e-8, max_iter=100, 
                                   relajacion=1.0, verbose=False):
    """
    Resuelve el PVF no lineal usando diferencias finitas + Newton.
    Retorna: x_vals (incluye fronteras), y_vals (solución), historial_Newton
    """
    x_interior, F = construir_sistema_no_lineal(a, b, alpha, beta, n, f_func)
    
    # Estimación inicial por defecto: interpolación lineal entre fronteras
    if y0 is None:
        h = (b - a) / (n + 1)
        y0 = np.linspace(alpha, beta, n+2)[1:-1]  # puntos interiores
    else:
        y0 = np.array(y0)
        if len(y0) != n:
            raise ValueError(f"y0 debe tener longitud {n}")
    
    # Resolver con Newton
    y_interior, iteraciones, historial = newton_sistema(F, y0, tol, max_iter, relajacion, verbose)
    
    # Construir vectores completos
    x_vals = [a] + x_interior + [b]
    y_vals = [alpha] + list(y_interior) + [beta]
    
    return x_vals, y_vals, historial, iteraciones

# ============================================================
# 4. VERIFICACIÓN DE CONDICIONES (advertencias)
# ============================================================
def verificar_existencia_no_lineal(a, b, f_func, n=100):
    advertencias = []
    x_sample = np.linspace(a, b, n)
    
    # Evaluamos f en algunos puntos para tener una idea
    try:
        f_vals = [f_func(x, 0, 0) for x in x_sample]
        if any(abs(fv) > 100 for fv in f_vals):
            advertencias.append("⚠️ f(x,0,0) es muy grande en algunos puntos. Posible rigidez.")
    except:
        advertencias.append("⚠️ No se pudo evaluar f en (x,0,0). Verifique la función.")
    
    advertencias.append("ℹ️ Para EDOs no lineales, la unicidad no está garantizada globalmente.")
    advertencias.append("   El método de Newton puede converger a una solución, pero pueden existir múltiples.")
    return advertencias

# ============================================================
# 5. MOSTRAR ECUACIÓN
# ============================================================
def mostrar_ecuacion_no_lineal(f_str, a, b, alpha, beta):
    print("\n" + "="*60)
    print(" PROBLEMA DE VALOR DE FRONTERA NO LINEAL (DIFERENCIAS FINITAS)")
    print("="*60)
    print(f"EDO: y'' = f(x, y, y')  con  f(x,y,z) = {f_str}")
    print(f"Condiciones: y({a}) = {alpha},  y({b}) = {beta}")
    print("="*60)

# ============================================================
# 6. FUNCIONES DE GRÁFICAS
# ============================================================
def graficar_solucion(x_vals, y_vals, a, alpha, b, beta, n, iteraciones):
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, y_vals, 'b-o', linewidth=2, markersize=4, label='Solución aproximada')
    plt.plot(a, alpha, 'ko', markersize=8, label=f'Frontera ({a},{alpha})')
    plt.plot(b, beta, 'ko', markersize=8, label=f'Frontera ({b},{beta})')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'Diferencias Finitas No Lineal (n={n} puntos, {iteraciones} iteraciones Newton)')
    plt.grid(True); plt.legend()
    plt.show()

def graficar_error(x_vals, y_vals, y_exact_vals, n):
    error = [abs(y_vals[i] - y_exact_vals[i]) for i in range(len(x_vals))]
    plt.figure(figsize=(10, 5))
    plt.semilogy(x_vals, error, 'r-', linewidth=2)
    plt.xlabel('x'); plt.ylabel('Error absoluto')
    plt.title(f'Error de la solución - n={n} puntos interiores')
    plt.grid(True)
    plt.show()

def graficar_convergencia_newton(historial, n):
    """Grafica la evolución de la norma del residuo en las iteraciones de Newton"""
    if len(historial) < 2:
        return
    # Calcular norma de F en cada iteración (necesitamos re-evaluar)
    # Pero podemos usar la norma de la diferencia entre iteraciones
    diffs = [np.linalg.norm(historial[i+1] - historial[i]) for i in range(len(historial)-1)]
    plt.figure(figsize=(10, 5))
    plt.semilogy(range(1, len(diffs)+1), diffs, 'bo-', linewidth=2)
    plt.xlabel('Iteración de Newton')
    plt.ylabel('||Δy||')
    plt.title(f'Convergencia de Newton - n={n} puntos')
    plt.grid(True)
    plt.show()

def animar_newton(a, b, alpha, beta, n, f_func, historial):
    """Animación que muestra cómo evoluciona la solución en cada iteración de Newton"""
    if len(historial) < 2:
        print("No hay suficientes iteraciones para animar.")
        return
    
    x_interior = [a + (i+1)*(b-a)/(n+1) for i in range(n)]
    x_vals = [a] + x_interior + [b]
    
    fig_anim, ax_anim = plt.subplots(figsize=(10, 5))
    ax_anim.scatter([a], [alpha], color='k', s=100, zorder=5, label=f'Frontera ({a},{alpha})')
    ax_anim.scatter([b], [beta], color='k', s=100, zorder=5, label=f'Frontera ({b},{beta})')
    line, = ax_anim.plot([], [], 'b-', linewidth=2, label='Solución actual')
    ax_anim.legend(loc='upper left')
    ax_anim.grid(True)
    ax_anim.set_xlabel('x'); ax_anim.set_ylabel('y')
    ax_anim.set_xlim(a - 0.1*(b-a), b + 0.1*(b-a))
    
    def init():
        line.set_data([], [])
        return line,
    
    def update(k):
        y_interior = historial[k]
        y_vals = [alpha] + list(y_interior) + [beta]
        line.set_data(x_vals, y_vals)
        max_y = max(max(y_vals), alpha, beta)
        min_y = min(min(y_vals), alpha, beta)
        margen = (max_y - min_y) * 0.3 + 1.0
        ax_anim.set_ylim(min_y - margen, max_y + margen)
        ax_anim.set_title(f'Iteración Newton {k+1}/{len(historial)}')
        return line,
    
    ani = FuncAnimation(fig_anim, update, frames=range(len(historial)), 
                        init_func=init, interval=500, repeat=True, blit=False)
    plt.show()

def animar_convergencia_malla(a, b, alpha, beta, f_func, n_values):
    """Animación que muestra cómo la solución se refina al aumentar n"""
    fig_anim, ax_anim = plt.subplots(figsize=(10, 5))
    ax_anim.scatter([a], [alpha], color='k', s=100, zorder=5, label=f'Frontera ({a},{alpha})')
    ax_anim.scatter([b], [beta], color='k', s=100, zorder=5, label=f'Frontera ({b},{beta})')
    line, = ax_anim.plot([], [], 'b-', linewidth=2, label='Solución aproximada')
    ax_anim.legend(loc='upper left')
    ax_anim.grid(True)
    ax_anim.set_xlabel('x'); ax_anim.set_ylabel('y')
    ax_anim.set_xlim(a - 0.1*(b-a), b + 0.1*(b-a))
    
    def init():
        line.set_data([], [])
        return line,
    
    def update(n_actual):
        try:
            x_vals, y_vals, _, _ = diferencias_finitas_no_lineal(
                a, b, alpha, beta, n_actual, f_func, 
                tol=1e-8, max_iter=50, verbose=False
            )
            line.set_data(x_vals, y_vals)
            ax_anim.set_title(f'Convergencia con n = {n_actual} puntos interiores')
            max_y = max(max(y_vals), alpha, beta)
            min_y = min(min(y_vals), alpha, beta)
            margen = (max_y - min_y) * 0.3 + 1.0
            ax_anim.set_ylim(min_y - margen, max_y + margen)
        except:
            pass
        return line,
    
    ani = FuncAnimation(fig_anim, update, frames=n_values, init_func=init, interval=1000, repeat=True, blit=False)
    plt.show()

# ============================================================
# 7. PROGRAMA PRINCIPAL
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("   DIFERENCIAS FINITAS PARA PVF NO LINEALES")
    print("   EDO: y'' = f(x, y, y')")
    print("="*50 + "\n")
    
    # --- 1. DEFINIR LA EDO NO LINEAL ---
    print("Ingrese la función f(x, y, y') en términos de 'x', 'y' y 'z' (donde z = y'):")
    print("Ejemplo: -y + sin(x)  o  -z**2 + y  o  exp(x)*z + y**2")
    f_str = input("  f(x, y, z) = ")
    
    def make_func(expr):
        return lambda x, y, z: eval(expr, {"x": x, "y": y, "z": z, "math": math})
    
    f_func = make_func(f_str)
    
    # --- 2. CONDICIONES DE FRONTERA ---
    a = float(input("Ingrese el extremo izquierdo a: "))
    b = float(input("Ingrese el extremo derecho  b: "))
    alpha = float(input("Ingrese y(a) = alpha: "))
    beta = float(input("Ingrese y(b) = beta: "))
    
    if a >= b:
        print("Error: a debe ser mayor que b.")
        exit()
    
    # --- 3. VERIFICAR EXISTENCIA ---
    advertencias = verificar_existencia_no_lineal(a, b, f_func)
    print("\n" + "="*60)
    print(" VERIFICACIÓN DE CONDICIONES")
    print("="*60)
    for msg in advertencias:
        print(msg)
    print("="*60)
    
    # --- 4. NÚMERO DE PUNTOS INTERIORES ---
    n = int(input("\nIngrese el número de puntos interiores n (mayor que 0): "))
    if n <= 0:
        print("Error: n debe ser positivo.")
        exit()
    
    # --- 5. PARÁMETROS DE NEWTON ---
    print("\nOpciones para el método de Newton:")
    tol = float(input("Tolerancia (ej. 1e-8): ") or "1e-8")
    max_iter = int(input("Número máximo de iteraciones (ej. 100): ") or "100")
    relajacion = float(input("Factor de relajación (1.0 = Newton puro, <1.0 más estable): ") or "1.0")
    
    # --- 6. (OPCIONAL) INGRESAR SOLUCIÓN EXACTA ---
    print("\n¿Desea ingresar una solución exacta para comparar? (s/n)")
    exacta_opcion = input().strip().lower()
    y_exacta_func = None
    if exacta_opcion == 's':
        expr_exacta = input("Ingrese la solución exacta y(x) en términos de 'x': ")
        y_exacta_func = make_func(expr_exacta)
    
    # --- 7. RESOLVER ---
    print(f"\nResolviendo con n = {n} puntos interiores...")
    try:
        x_vals, y_vals, historial, iteraciones = diferencias_finitas_no_lineal(
            a, b, alpha, beta, n, f_func, 
            tol=tol, max_iter=max_iter, relajacion=relajacion, verbose=False
        )
    except Exception as e:
        print(f"\n Error al resolver: {e}")
        exit()
    
    # --- 8. MOSTRAR ECUACIÓN ---
    mostrar_ecuacion_no_lineal(f_str, a, b, alpha, beta)
    print(f"Paso h = {(b-a)/(n+1):.6f}")
    print(f"Iteraciones de Newton: {iteraciones}")
    print(f"Tolerancia: {tol}")
    print(f"Relajación: {relajacion}")
    
    # --- 9. TABLA DE RESULTADOS ---
    print("\n" + "-"*70)
    print(f"{'  x  ':^12} | {'y(x) aproximada':^22} | {'Error (si exacta)':^22}")
    print("-"*70)
    
    for i in range(len(x_vals)):
        if y_exacta_func is not None:
            y_exact = y_exacta_func(x_vals[i])
            error = abs(y_vals[i] - y_exact)
            print(f"{x_vals[i]:^12.6f} | {y_vals[i]:^22.10f} | {error:^22.2e}")
        else:
            print(f"{x_vals[i]:^12.6f} | {y_vals[i]:^22.10f} | {'---':^22}")
    
    print("-"*70)
    
    # --- 10. GRÁFICAS AUTOMÁTICAS ---
    print("\n" + "="*60)
    print(" MOSTRANDO GRÁFICAS DE FORMA AUTOMÁTICA")
    print("   Cierra cada ventana para ver la siguiente.")
    print("="*60)
    
    # Gráfica 1: Solución aproximada
    graficar_solucion(x_vals, y_vals, a, alpha, b, beta, n, iteraciones)
    
    # Gráfica 2: Error (si se proporcionó exacta)
    if y_exacta_func is not None:
        y_exact_vals = [y_exacta_func(x) for x in x_vals]
        graficar_error(x_vals, y_vals, y_exact_vals, n)
    
    # Gráfica 3: Convergencia de Newton
    if len(historial) > 1:
        graficar_convergencia_newton(historial, n)
    
    # Gráfica 4: Animación de Newton
    print("\n¿Desea ver la animación de las iteraciones de Newton? (s/n)")
    anim_newton = input().strip().lower()
    if anim_newton == 's':
        animar_newton(a, b, alpha, beta, n, f_func, historial)
    
    # Gráfica 5: Animación de convergencia de malla
    print("\n¿Desea ver la animación de convergencia al aumentar n? (s/n)")
    anim_malla = input().strip().lower()
    if anim_malla == 's':
        n_values = [2, 4, 8, 16, 32, 64]
        n_values = [ni for ni in n_values if ni <= 100]  # limitar para no tardar
        animar_convergencia_malla(a, b, alpha, beta, f_func, n_values)
    
    print("\n Programa finalizado.")
    
