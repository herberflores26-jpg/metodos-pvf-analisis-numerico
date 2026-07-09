
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy import symbols, sympify, lambdify
import numpy as np

# ============================================================
# 1. FUNCIONES AUXILIARES PARA LA DISCRETIZACIÓN
# ============================================================
def construir_sistema(a, b, alpha, beta, n, p_func, q_func, r_func):
    """
    Construye el sistema lineal A * y = b para el problema de diferencias finitas.
    Retorna: A (matriz tridiagonal como tres diagonales), d (vector RHS)
    """
    h = (b - a) / (n + 1)   # n = número de puntos interiores
    x = [a + (i+1)*h for i in range(n)]  # puntos interiores
    
    # Diagonales de la matriz tridiagonal
    # A = diag(inferior), diag(principal), diag(superior)
    low = [0.0] * (n-1)    # subdiagonal (tamaño n-1)
    diag = [0.0] * n       # diagonal principal
    sup = [0.0] * (n-1)    # superdiagonal (tamaño n-1)
    d = [0.0] * n          # vector de términos independientes
    
    for i in range(n):
        xi = x[i]
        # Coeficientes de la ecuación en diferencias:
        # y'' ≈ (y_{i-1} - 2y_i + y_{i+1})/h^2
        # y' ≈ (y_{i+1} - y_{i-1})/(2h)
        # Sustituyendo en y'' = p*y' + q*y + r, y reordenando:
        # Coeficiente para y_{i-1}: (1/h^2) - p(xi)/(2h)
        # Coeficiente para y_i: -2/h^2 - q(xi)
        # Coeficiente para y_{i+1}: (1/h^2) + p(xi)/(2h)
        # Término independiente: r(xi)
        coef_izq = 1.0/(h*h) - p_func(xi)/(2*h)
        coef_cent = -2.0/(h*h) - q_func(xi)
        coef_der = 1.0/(h*h) + p_func(xi)/(2*h)
        
        diag[i] = coef_cent
        
        # Término independiente incluye r(xi) y las condiciones de frontera si corresponden
        rhs = r_func(xi)
        
        # Ajuste para la primera ecuación (i=0): incluye alpha en la izquierda
        if i == 0:
            rhs -= coef_izq * alpha
        else:
            low[i-1] = coef_izq
        
        # Ajuste para la última ecuación (i=n-1): incluye beta en la derecha
        if i == n-1:
            rhs -= coef_der * beta
        else:
            sup[i] = coef_der
        
        d[i] = rhs
    
    return low, diag, sup, d, x

# ============================================================
# 2. RESOLVER SISTEMA TRIDIAGONAL (ALGORITMO DE THOMAS)
# ============================================================
def thomas(low, diag, sup, d):
    """
    Resuelve sistema tridiagonal A*y = d.
    low: subdiagonal (n-1)
    diag: diagonal principal (n)
    sup: superdiagonal (n-1)
    d: vector RHS (n)
    Retorna: vector solución y
    """
    n = len(diag)
    # Copiamos para no modificar los originales
    c = sup[:]   # superdiagonal
    b = diag[:]  # diagonal
    a = low[:]   # subdiagonal
    r = d[:]     # RHS
    
    # Eliminación hacia adelante
    for i in range(1, n):
        m = a[i-1] / b[i-1]
        b[i] = b[i] - m * c[i-1]
        r[i] = r[i] - m * r[i-1]
        if i < n-1:
            a[i-1] = 0.0  # ya no se necesita
    
    # Sustitución hacia atrás
    y = [0.0] * n
    y[-1] = r[-1] / b[-1]
    for i in range(n-2, -1, -1):
        y[i] = (r[i] - c[i] * y[i+1]) / b[i]
    
    return y

# ============================================================
# 3. RESOLVER POR GAUSS (usando numpy.linalg.solve)
# ============================================================
def resolver_gauss(low, diag, sup, d):
    """
    Construye la matriz completa y resuelve usando numpy.
    """
    n = len(diag)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = diag[i]
        if i > 0:
            A[i, i-1] = low[i-1]
        if i < n-1:
            A[i, i+1] = sup[i]
    b = np.array(d)
    y = np.linalg.solve(A, b)
    return y.tolist()

# ============================================================
# 4. FUNCIÓN PRINCIPAL DE DIFERENCIAS FINITAS
# ============================================================
def diferencias_finitas(a, b, alpha, beta, n, p_func, q_func, r_func, metodo='thomas'):
    """
    Resuelve el PVF lineal usando diferencias finitas.
    metodo: 'thomas' o 'gauss'
    Retorna: x_vals (incluye fronteras), y_vals (solución en todos los puntos)
    """
    low, diag, sup, d, x_interior = construir_sistema(a, b, alpha, beta, n, p_func, q_func, r_func)
    
    if metodo == 'thomas':
        y_interior = thomas(low, diag, sup, d)
    elif metodo == 'gauss':
        y_interior = resolver_gauss(low, diag, sup, d)
    else:
        raise ValueError("Método no reconocido. Use 'thomas' o 'gauss'.")
    
    # Construir vectores completos incluyendo fronteras
    x_vals = [a] + x_interior + [b]
    y_vals = [alpha] + y_interior + [beta]
    
    return x_vals, y_vals

# ============================================================
# 5. VERIFICACIÓN DE CONDICIONES (existencia y unicidad)
# ============================================================
def verificar_existencia_lineal(a, b, p_func, q_func, r_func, n=100):
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
        advertencias.append("ℹ️ q(x) toma valores negativos. La unicidad no está garantizada; verifique la solución obtenida.")
    
    return advertencias

# ============================================================
# 6. MOSTRAR ECUACIÓN Y CONDICIONES
# ============================================================
def mostrar_ecuacion(p_str, q_str, r_str, a, b, alpha, beta):
    print("\n" + "="*60)
    print("📐 PROBLEMA DE VALOR DE FRONTERA LINEAL (DIFERENCIAS FINITAS)")
    print("="*60)
    print(f"EDO: y'' = ({p_str}) y' + ({q_str}) y + ({r_str})")
    print(f"Condiciones: y({a}) = {alpha},  y({b}) = {beta}")
    print("="*60)

# ============================================================
# 7. FUNCIONES DE GRÁFICAS
# ============================================================
def graficar_solucion(x_vals, y_vals, a, alpha, b, beta, n, metodo):
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, y_vals, 'b-o', linewidth=2, markersize=4, label='Solución aproximada')
    plt.plot(a, alpha, 'ko', markersize=8, label=f'Frontera ({a},{alpha})')
    plt.plot(b, beta, 'ko', markersize=8, label=f'Frontera ({b},{beta})')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'Diferencias Finitas - {metodo.upper()} (n={n} puntos interiores)')
    plt.grid(True); plt.legend()
    plt.show()

def graficar_error(x_vals, y_vals, y_exact_vals, n):
    """Grafica el error absoluto si se proporciona solución exacta"""
    error = [abs(y_vals[i] - y_exact_vals[i]) for i in range(len(x_vals))]
    plt.figure(figsize=(10, 5))
    plt.semilogy(x_vals, error, 'r-', linewidth=2)
    plt.xlabel('x'); plt.ylabel('Error absoluto')
    plt.title(f'Error de la solución - n={n} puntos interiores')
    plt.grid(True)
    plt.show()

def animar_convergencia(a, b, alpha, beta, p_func, q_func, r_func, metodo):
    """Animación que muestra cómo la solución se refina al aumentar n"""
    n_values = [2, 4, 8, 16, 32, 64]  # números de puntos interiores
    fig_anim, ax_anim = plt.subplots(figsize=(10, 5))
    
    # Puntos frontera
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
        x_vals, y_vals = diferencias_finitas(a, b, alpha, beta, n_actual, p_func, q_func, r_func, metodo)
        line.set_data(x_vals, y_vals)
        ax_anim.set_title(f'Convergencia con n = {n_actual} puntos interiores')
        max_y = max(max(y_vals), alpha, beta)
        min_y = min(min(y_vals), alpha, beta)
        margen = (max_y - min_y) * 0.3 + 1.0
        ax_anim.set_ylim(min_y - margen, max_y + margen)
        return line,
    
    ani = FuncAnimation(fig_anim, update, frames=n_values, init_func=init, interval=1000, repeat=True, blit=False)
    plt.show()

# ============================================================
# 8. PROGRAMA PRINCIPAL (INTERACTIVO)
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("   MÉTODO DE DIFERENCIAS FINITAS PARA PVF LINEALES")
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
        print("Error: a debe ser mayor que b.")
        exit()
    
    # --- 3. VERIFICAR EXISTENCIA ---
    advertencias = verificar_existencia_lineal(a, b, p_func, q_func, r_func)
    print("\n" + "="*60)
    print("VERIFICACIÓN DE CONDICIONES DE EXISTENCIA Y UNICIDAD")
    print("="*60)
    for msg in advertencias:
        print(msg)
    print("="*60)
    
    # --- 4. NÚMERO DE PUNTOS INTERIORES ---
    n = int(input("\nIngrese el número de puntos interiores n (mayor que 0): "))
    if n <= 0:
        print("Error: n debe ser positivo.")
        exit()
    
    # --- 5. ELEGIR MÉTODO DE RESOLUCIÓN ---
    print("\nSeleccione el método para resolver el sistema lineal:")
    print("  1. Thomas (tridiagonal - recomendado)")
    print("  2. Gauss (numpy.linalg.solve)")
    opcion_sist = input("Opción (1/2): ")
    if opcion_sist == "1":
        metodo = 'thomas'
    elif opcion_sist == "2":
        try:
            import numpy as np
            metodo = 'gauss'
        except ImportError:
            print(" numpy no está instalado. Usando Thomas por defecto.")
            metodo = 'thomas'
    else:
        print("Opción no válida. Usando Thomas por defecto.")
        metodo = 'thomas'
    
    # --- 6. (OPCIONAL) INGRESAR SOLUCIÓN EXACTA ---
    print("\n¿Desea ingresar una solución exacta para comparar? (s/n)")
    exacta_opcion = input().strip().lower()
    y_exacta_func = None
    if exacta_opcion == 's':
        expr_exacta = input("Ingrese la solución exacta y(x) en términos de 'x': ")
        y_exacta_func = make_func(expr_exacta)
    
    # --- 7. RESOLVER ---
    print(f"\nResolviendo con {metodo.upper()} y n = {n} puntos interiores...")
    try:
        x_vals, y_vals = diferencias_finitas(a, b, alpha, beta, n, p_func, q_func, r_func, metodo)
    except Exception as e:
        print(f"\n Error al resolver: {e}")
        exit()
    
    # --- 8. MOSTRAR ECUACIÓN ---
    mostrar_ecuacion(p_str, q_str, r_str, a, b, alpha, beta)
    print(f"Paso h = {(b-a)/(n+1):.6f}")
    print(f"Método usado: {metodo.upper()}")
    
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
    graficar_solucion(x_vals, y_vals, a, alpha, b, beta, n, metodo)
    
    # Gráfica 2: Error (si se proporcionó exacta)
    if y_exacta_func is not None:
        y_exact_vals = [y_exacta_func(x) for x in x_vals]
        graficar_error(x_vals, y_vals, y_exact_vals, n)
    
    # Gráfica 3: Animación de convergencia (opcional)
    print("\n¿Desea ver una animación de convergencia al aumentar n? (s/n)")
    anim_opcion = input().strip().lower()
    if anim_opcion == 's':
        animar_convergencia(a, b, alpha, beta, p_func, q_func, r_func, metodo)
    
    print("\n Programa finalizado.")

