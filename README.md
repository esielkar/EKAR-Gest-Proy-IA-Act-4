# Clasificador Automático de Incidencias de Desarrollo con Hugging Face

## Descripción del Proyecto
Prototipo técnico desarrollado para la clasificación automática de reportes de errores (*bugs*), incidencias de infraestructura en servidores propios (Synology NAS / Docker) y solicitudes de nuevas funcionalidades en proyectos de desarrollo de software. Utiliza modelos preentrenados del ecosistema de **Hugging Face** ejecutados en un entorno de prototipado rápido con GPU.

## Estructura del Repositorio
- `data/`: Contiene el dataset estructurado de 50 registros de incidencias técnicas.
- `notebooks/`: Cuaderno de Google Colab utilizado para la experimentación y evaluación.
- `src/`: Scripts ejecutables de preprocesamiento e inferencia modular.
- `results/`: Tablas de resultados comparativos y métricas de rendimiento.

## Entorno de Ejecución
- Python 3.10 o superior.
- PyTorch con soporte CUDA (recomendado GPU NVIDIA T4 o superior).
- Librerías clave: `transformers`, `datasets`, `scikit-learn`, `pandas`.

## Pasos de Instalación y Ejecución
1. Clona el repositorio:
   ```bash
   git clone [https://github.com/esielkar/EKAR-Gest-Proy-IA-Act-4.git](https://github.com/esielkar/EKAR-Gest-Proy-IA-Act-4.git)
   cd EKAR-Gest-Proy-IA-Act-4

## Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Ejecuta el script de inferencia local:
```bash
python src/inference.py
```

   
