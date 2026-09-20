# Quimiotaxis de C. elegans — agente bioinspirado

Simulación de un agente que localiza una fuente química siguiendo un gradiente,
gobernado por una red neuronal de pesos fijos inspirada en el circuito de
quimiotaxis del nematodo *Caenorhabditis elegans*. Sin dependencias: JavaScript
puro, módulos ES, Canvas 2D. El banco de pruebas corre en Node.

Proyecto académico de Computación Bioinspirada (UNIMINUTO, 2026).

## El sistema biológico

*C. elegans* es el único organismo con el sistema nervioso reconstruido por
completo: 302 neuronas mapeadas neurona por neurona. Detecta gradientes con un
par bilateral de quimiorreceptores (ASEL/ASER) y navega con dos estrategias
documentadas que operan en paralelo:

- **Pirueta** — giro brusco y aleatorio cuando la concentración lleva varios
  pasos bajando. Reorientación estocástica.
- **Veleta** (*weathervane*) — corrección continua del rumbo hacia el lado más
  concentrado. Dirección propiamente dicha.

Ambas están implementadas y son desactivables por separado, que es lo que
permite hacer ablaciones.

## Arquitectura

| Archivo | Qué hace |
|---|---|
| `world.js` | Campo de concentración gaussiano, fuente reubicable |
| `nn.js` | Red 2→3→2, tanh, pesos fijos, adaptación sensorial |
| `agent.js` | Cuerpo, sensores, los dos mecanismos, ablaciones |
| `sim.js` | Un ensayo con sus métricas |
| `bench.js` | 200 ensayos × 7 condiciones, RNG con semilla |
| `ui.js` + `index.html` | Interfaz interactiva |
| `spec.md` | Especificación que sirvió de contrato |
| `results.json` | Salida del banco de pruebas |

## Resultados medidos

200 ensayos por condición, mismos estados iniciales en todas, semilla 12345.

| Condición | Éxito | Pasos medios |
|---|---|---|
| intact | 1.000 | 309.1 |
| no_pirouette | 0.995 | 328.5 |
| no_weathervane | 0.875 | 1094.3 |
| ablate_ASEL | 0.000 | — |
| noise_0.1 | 0.715 | 1112.9 |
| moving_source | 1.000 | 309.4 |
| random_walk (control) | 0.050 | 1341.5 |

Dos lecturas que importan. **`ablate_ASEL` = 0.000**: cortar un solo sensor
destruye la navegación por completo, porque el canal de dirección es una
*diferencia* entre las dos lecturas — con una sola, no codifica dirección.
**`moving_source` = 1.000**: reubicar la fuente no degrada nada, porque el
agente no guarda representación interna que invalidar.

## El hallazgo: por qué hubo que rediseñarlo

La primera versión alimentaba los dos mecanismos con la misma señal temporal
(lectura menos promedio móvil). El banco de pruebas dio esto:

| Condición | Éxito (versión inicial) |
|---|---|
| intact | 0.990 |
| no_weathervane | **1.000** |
| no_pirouette | 0.015 |

Apagar la red neuronal **mejoraba** el resultado. La única lectura coherente es
que la red no estaba dirigiendo: toda la navegación la hacía la pirueta, y la
salida de la red era ruido encima.

La causa: en un gradiente suave la variación por paso es despreciable, así que
el canal temporal entregaba a la red un valor próximo a cero. La diferencia
espacial entre sensores sí tenía señal, pero la arquitectura la descartaba antes
de llegar a la red.

El arreglo fue separar los canales — diferencia espacial para dirigir, variación
temporal para decidir la pirueta — que es como está cableado el animal real. Los
pesos y la topología no se tocaron. `intact` pasó de 604.5 pasos a 309.1.

El defecto no se vio leyendo el código. Se vio porque una medición sistemática
dio un resultado imposible de reconciliar con la hipótesis de partida.

## Cómo correrlo

```bash
node bench.js                 # banco de pruebas, escribe results.json
python3 -m http.server 8000   # y abrir localhost:8000 para la interfaz
```

Los módulos ES no cargan desde `file://`, por eso hace falta el servidor.

## Referencias

- Pierce-Shimomura, Morse y Lockery (1999). *The fundamental role of pirouettes
  in C. elegans chemotaxis*. J Neurosci 19(21), 9557-9569.
- Iino y Yoshida (2009). *Parallel use of two behavioral mechanisms for
  chemotaxis in C. elegans*. J Neurosci 29(17), 5370-5380.
- Suzuki et al. (2008). *Functional asymmetry in C. elegans taste neurons*.
  Nature 454(7200), 114-117.
- Izquierdo y Beer (2013). *Connecting a connectome to behavior*. PLoS Comput
  Biol 9(2), e1002890.
- White, Southgate, Thomson y Brenner (1986). *The structure of the nervous
  system of the nematode C. elegans*. Phil Trans R Soc B 314(1165), 1-340.
