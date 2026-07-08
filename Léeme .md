Métodos Numéricos para Problemas de Valor de Frontera (PVF)

Este repositorio contiene la implementación en Python de cuatro métodos clásicos para resolver problemas de valor de frontera de segundo orden. Los códigos fueron desarrollados como parte del curso de Análisis Numérico II.

Contenido del repositorio

- disparo_lineal.py – Método del disparo lineal.
- disparo_no_lineal.py – Método del disparo no lineal.
- diferencias_finitas_lineal.py – Método de diferencias finitas para PVF lineales.
- diferencias_finitas_no_lineal.py – Método de diferencias finitas para PVF no lineales.

Requisitos

Para ejecutar los programas necesitas tener instalado:

- Python 3.6 o superior.
- Las librerías numpy y matplotlib. Puedes instalarlas con:

pip install numpy matplotlib

--------------------------------

Ejemplos de uso

Cada programa es interactivo y guía al usuario en la entrada de datos. A continuación se muestran las configuraciones exactas utilizadas en los problemas ilustrativos del trabajo académico.

1. Disparo lineal – Deflexión de una viga

Contexto: Viga simplemente apoyada con carga uniforme.
Ecuación: y'' = -x(1-x), con y(0)=0, y(1)=0.

Ejecuta:

python disparo_lineal.py

Ingresa los siguientes valores:

Parámetro | Valor
p(x) | 0
q(x) | 0
r(x) | -x*(1-x)
a | 0
b | 1
alpha | 0
beta | 0
n (subintervalos) | 20
Método PVI | 3 (RK4)

2. Disparo no lineal – Trayectoria de un proyectil con arrastre

Contexto: Movimiento vertical con resistencia del aire cuadrática.
Ecuación: y'' = -1 - (y')^2, con y(0)=0, y(1)=0.5.

Ejecuta:

python disparo_no_lineal.py

Parámetro | Valor
f(x, y, z) | -1 - z**2
a | 0
b | 1
alpha | 0
beta | 0.5
n (subintervalos) | 50
Método PVI | 3 (RK4)
Método de búsqueda de raíz | 2 (Newton)
Estimación inicial t0 | 0.5

3. Diferencias finitas lineal – Aleta con convección

Contexto: Distribución de temperatura en una aleta recta con convección.
Ecuación: y'' - 4y = -4, con y(0)=2, y(1)=1.

Ejecuta:

python diferencias_finitas_lineal.py

Parámetro | Valor
p(x) | 0
q(x) | -4
r(x) | -4
a | 0
b | 1
alpha | 2
beta | 1
n (puntos interiores) | 20
Método de solución | 1 (Thomas)

4. Diferencias finitas no lineal – Reacción en catalizador poroso

Contexto: Difusión con reacción química no lineal (cinética de Michaelis-Menten).
Ecuación: y'' = 10*y/(1+y), con y(0)=1, y(1)=0.1.

Ejecuta:

python diferencias_finitas_no_lineal.py

Parámetro | Valor
f(x, y, z) | 10*y/(1+y)
a | 0
b | 1
alpha | 1
beta | 0.1
n (puntos interiores) | 30
Tolerancia de Newton | 1e-8
Máximo de iteraciones | 100
Factor de relajación | 1.0

Características destacadas

- Solvers de PVI: Euler, Heun y RK4 (seleccionables en los métodos de disparo).
- Búsqueda de raíces: Bisección y Newton (para disparo no lineal).
- Solución de sistemas lineales: Algoritmo de Thomas (tridiagonal) y numpy.linalg.solve.
- Visualización: Gráficas de las soluciones y animaciones interactivas que muestran:
  - La combinación lineal en el disparo lineal.
  - La variación de la pendiente inicial en el disparo no lineal.
  - La convergencia de Newton en diferencias finitas no lineales.
  - El refinamiento de la malla al aumentar el número de puntos.

Autor:

Desarrollado por herberflores26-jpg para el curso de Análisis Numérico II.

Licencia

Este material es de uso educativo y puede ser utilizado libremente con fines académicos. 
