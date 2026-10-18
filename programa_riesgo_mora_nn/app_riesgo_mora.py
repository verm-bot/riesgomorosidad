"""
Sistema de predicción de riesgo de mora con red neuronal feedforward.
Autor: Adaptable para proyecto académico / empresarial.

Funcionamiento:
1. Carga datos históricos en CSV.
2. Procesa variables numéricas y categóricas.
3. Entrena una red neuronal feedforward usando MLPClassifier.
4. Evalúa el modelo con accuracy, matriz de confusión y reporte de clasificación.
5. Predice riesgo de incumplimiento para nuevos clientes.
"""

import os
import sys
import shutil
from pathlib import Path
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_TITLE = "Predicción de Riesgo de Mora - Red Neuronal Feedforward"
APP_SHORT_NAME = "RiesgoMoraNN"
MODEL_FILE_DEFAULT = "modelo_riesgo_mora.joblib"
SAMPLE_TRAIN_FILE = "datos_historicos_clientes.csv"
SAMPLE_PREDICT_FILE = "clientes_nuevos_para_predecir.csv"
APP_ICON_FILE = os.path.join("assets", "riesgo_mora.ico")


def resource_path(relative_path: str) -> str:
    """
    Devuelve una ruta válida tanto cuando el programa se ejecuta como .py
    como cuando se empaqueta en .exe con PyInstaller.
    """
    base_path = getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)
    return str(Path(base_path) / relative_path)


def get_app_data_dir() -> str:
    """
    Carpeta fija y escribible para guardar modelos, ejemplos y resultados.
    Así el ejecutable puede abrirse desde un icono del escritorio sin depender
    de la carpeta donde estaba el código fuente.
    """
    if os.name == "nt":
        base_dir = os.environ.get("LOCALAPPDATA", str(Path.home()))
        app_dir = Path(base_dir) / APP_SHORT_NAME
    else:
        app_dir = Path.home() / f".{APP_SHORT_NAME.lower()}"
    app_dir.mkdir(parents=True, exist_ok=True)
    return str(app_dir)


class RiesgoMoraApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("980x680")
        self.root.minsize(900, 600)

        self.app_data_dir = get_app_data_dir()
        self._set_window_icon()
        self.sample_train_path = self._ensure_sample_file(SAMPLE_TRAIN_FILE)
        self.sample_predict_path = self._ensure_sample_file(SAMPLE_PREDICT_FILE)

        self.train_df = None
        self.predict_df = None
        self.model_bundle = None
        self.train_path = None
        self.predict_path = None

        self._build_ui()

    def _set_window_icon(self):
        """Coloca el icono de la ventana cuando existe el archivo .ico."""
        try:
            icon_path = resource_path(APP_ICON_FILE)
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except Exception:
            # El icono no es crítico para el funcionamiento del sistema.
            pass

    def _ensure_sample_file(self, filename: str) -> str:
        """Copia los archivos de ejemplo a una carpeta fija del usuario."""
        destination = os.path.join(self.app_data_dir, filename)
        source = resource_path(filename)
        try:
            if os.path.exists(source) and not os.path.exists(destination):
                shutil.copy2(source, destination)
        except Exception:
            pass
        return destination if os.path.exists(destination) else source

    def _load_training_csv_from_path(self, path: str):
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("El archivo CSV está vacío.")

        self.train_df = df
        self.train_path = path
        self.train_file_label.config(text=os.path.basename(path))

        columns = list(df.columns)
        self.target_combo["values"] = columns
        preferred_targets = [
            "riesgo_mora",
            "mora",
            "riesgo",
            "incumplimiento",
            "default",
            "target",
        ]
        selected = None
        for candidate in preferred_targets:
            for col in columns:
                if col.strip().lower() == candidate:
                    selected = col
                    break
            if selected:
                break
        self.target_combo.set(selected or columns[-1])

        preview = df.head(8).to_string(index=False)
        self._write_train_output(
            f"Archivo cargado correctamente: {path}\n"
            f"Filas: {len(df)} | Columnas: {len(df.columns)}\n\n"
            f"Vista previa:\n{preview}\n"
        )

    def _load_prediction_csv_from_path(self, path: str):
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("El archivo CSV está vacío.")
        self.predict_df = df
        self.predict_path = path
        self.predict_file_label.config(text=os.path.basename(path))

        preview = df.head(8).to_string(index=False)
        self._write_predict_output(
            f"Archivo de clientes cargado correctamente: {path}\n"
            f"Filas: {len(df)} | Columnas: {len(df.columns)}\n\n"
            f"Vista previa:\n{preview}\n"
        )

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        title = ttk.Label(
            self.root,
            text="Sistema de Predicción de Riesgo de Mora",
            font=("Arial", 18, "bold"),
        )
        title.pack(pady=(12, 4))

        subtitle = ttk.Label(
            self.root,
            text="Red neuronal feedforward para clasificar clientes según su riesgo de incumplimiento",
            font=("Arial", 10),
        )
        subtitle.pack(pady=(0, 10))

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=12, pady=8)

        self.tab_train = ttk.Frame(notebook)
        self.tab_predict = ttk.Frame(notebook)
        self.tab_help = ttk.Frame(notebook)

        notebook.add(self.tab_train, text="1. Entrenar modelo")
        notebook.add(self.tab_predict, text="2. Predecir clientes")
        notebook.add(self.tab_help, text="Guía rápida")

        self._build_train_tab()
        self._build_predict_tab()
        self._build_help_tab()

    def _build_train_tab(self):
        frame_top = ttk.LabelFrame(self.tab_train, text="Ingreso de datos históricos")
        frame_top.pack(fill="x", padx=10, pady=10)

        ttk.Button(frame_top, text="Cargar CSV histórico", command=self.load_training_csv).pack(
            side="left", padx=8, pady=10
        )
        ttk.Button(frame_top, text="Usar datos de ejemplo", command=self.load_sample_training_csv).pack(
            side="left", padx=8, pady=10
        )

        self.train_file_label = ttk.Label(frame_top, text="Ningún archivo cargado")
        self.train_file_label.pack(side="left", padx=8)

        frame_options = ttk.LabelFrame(self.tab_train, text="Configuración del entrenamiento")
        frame_options.pack(fill="x", padx=10, pady=8)

        ttk.Label(frame_options, text="Columna objetivo / riesgo:").grid(
            row=0, column=0, padx=8, pady=10, sticky="w"
        )
        self.target_combo = ttk.Combobox(frame_options, state="readonly", width=35)
        self.target_combo.grid(row=0, column=1, padx=8, pady=10, sticky="w")

        ttk.Label(frame_options, text="Tamaño de prueba:").grid(
            row=0, column=2, padx=8, pady=10, sticky="w"
        )
        self.test_size_var = tk.StringVar(value="0.20")
        ttk.Entry(frame_options, textvariable=self.test_size_var, width=8).grid(
            row=0, column=3, padx=8, pady=10, sticky="w"
        )

        ttk.Button(frame_options, text="Entrenar red neuronal", command=self.train_model).grid(
            row=0, column=4, padx=8, pady=10, sticky="w"
        )

        ttk.Button(frame_options, text="Guardar modelo", command=self.save_model).grid(
            row=0, column=5, padx=8, pady=10, sticky="w"
        )

        frame_output = ttk.LabelFrame(self.tab_train, text="Resultado del entrenamiento y evaluación")
        frame_output.pack(fill="both", expand=True, padx=10, pady=10)

        self.train_output = tk.Text(frame_output, wrap="word", height=20)
        self.train_output.pack(fill="both", expand=True, padx=8, pady=8)
        self._write_train_output(
            "Pasos:\n"
            "1. Cargue un archivo CSV con datos históricos de clientes.\n"
            "2. Seleccione la columna objetivo, por ejemplo: riesgo_mora.\n"
            "3. Entrene la red neuronal y revise la evaluación.\n"
            "4. Guarde el modelo para usarlo en predicciones futuras.\n"
        )

    def _build_predict_tab(self):
        frame_model = ttk.LabelFrame(self.tab_predict, text="Modelo entrenado")
        frame_model.pack(fill="x", padx=10, pady=10)

        ttk.Button(frame_model, text="Cargar modelo guardado", command=self.load_model).pack(
            side="left", padx=8, pady=10
        )
        self.model_label = ttk.Label(frame_model, text="Puede entrenar un modelo o cargar uno existente")
        self.model_label.pack(side="left", padx=8)

        frame_data = ttk.LabelFrame(self.tab_predict, text="Clientes para predicción")
        frame_data.pack(fill="x", padx=10, pady=8)

        ttk.Button(frame_data, text="Cargar CSV de clientes", command=self.load_prediction_csv).pack(
            side="left", padx=8, pady=10
        )
        ttk.Button(frame_data, text="Usar clientes de ejemplo", command=self.load_sample_prediction_csv).pack(
            side="left", padx=8, pady=10
        )
        self.predict_file_label = ttk.Label(frame_data, text="Ningún archivo cargado")
        self.predict_file_label.pack(side="left", padx=8)

        ttk.Button(frame_data, text="Predecir riesgo", command=self.predict_risk).pack(
            side="right", padx=8, pady=10
        )

        frame_output = ttk.LabelFrame(self.tab_predict, text="Resultados de predicción")
        frame_output.pack(fill="both", expand=True, padx=10, pady=10)

        self.predict_output = tk.Text(frame_output, wrap="word", height=20)
        self.predict_output.pack(fill="both", expand=True, padx=8, pady=8)
        self._write_predict_output(
            "Para predecir:\n"
            "1. Use el modelo entrenado o cargue un archivo .joblib.\n"
            "2. Cargue un CSV de clientes nuevos con las mismas variables usadas en el entrenamiento.\n"
            "3. Presione 'Predecir riesgo'. El sistema generará un CSV con los resultados.\n"
        )

    def _build_help_tab(self):
        help_frame = ttk.Frame(self.tab_help)
        help_frame.pack(fill="both", expand=True, padx=12, pady=12)

        help_text = tk.Text(help_frame, wrap="word")
        help_text.pack(fill="both", expand=True)
        help_text.insert(
            "1.0",
            "GUÍA RÁPIDA\n\n"
            "Estructura sugerida del CSV histórico:\n"
            "- id_cliente: identificador del cliente.\n"
            "- edad: edad del cliente.\n"
            "- ingresos_mensuales: ingresos aproximados.\n"
            "- saldo_pendiente: valor pendiente de pago.\n"
            "- dias_atraso_promedio: promedio de días de atraso.\n"
            "- historial_pagos_puntual: cantidad o porcentaje de pagos puntuales.\n"
            "- deuda_total: deuda total del cliente.\n"
            "- num_creditos: número de créditos activos o históricos.\n"
            "- antiguedad_cliente_meses: tiempo del cliente con la empresa.\n"
            "- uso_credito_pct: porcentaje de uso del cupo o crédito.\n"
            "- riesgo_mora: variable objetivo. Ejemplo: Bajo, Medio, Alto.\n\n"
            "La red neuronal feedforward aprende patrones a partir de los datos históricos. "
            "El modelo no reemplaza el criterio financiero humano, pero ayuda a priorizar clientes "
            "para cobranza preventiva, seguimiento y control de pérdidas.\n\n"
            "Recomendación: use datos reales, limpios y autorizados. No incluya información sensible "
            "innecesaria. Mientras más datos históricos de buena calidad tenga, mejor será la capacidad "
            "predictiva del modelo.\n",
        )
        help_text.config(state="disabled")

    def _write_train_output(self, text: str, clear: bool = True):
        if clear:
            self.train_output.delete("1.0", "end")
        self.train_output.insert("end", text)
        self.train_output.see("end")

    def _write_predict_output(self, text: str, clear: bool = True):
        if clear:
            self.predict_output.delete("1.0", "end")
        self.predict_output.insert("end", text)
        self.predict_output.see("end")

    def load_training_csv(self):
        path = filedialog.askopenfilename(
            title="Seleccione el CSV histórico",
            initialdir=self.app_data_dir,
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        try:
            self._load_training_csv_from_path(path)
        except Exception as exc:
            messagebox.showerror("Error al cargar CSV", str(exc))

    def load_sample_training_csv(self):
        try:
            self._load_training_csv_from_path(self.sample_train_path)
        except Exception as exc:
            messagebox.showerror("Error al cargar datos de ejemplo", str(exc))

    def train_model(self):
        if self.train_df is None:
            messagebox.showwarning("Falta información", "Primero cargue un CSV histórico.")
            return

        target_col = self.target_combo.get().strip()
        if not target_col or target_col not in self.train_df.columns:
            messagebox.showwarning("Columna objetivo", "Seleccione una columna objetivo válida.")
            return

        try:
            test_size = float(self.test_size_var.get())
            if not 0.05 <= test_size <= 0.50:
                raise ValueError("El tamaño de prueba debe estar entre 0.05 y 0.50.")

            df = self.train_df.copy()
            df = df.dropna(subset=[target_col])

            # Se elimina el identificador del cliente si existe, porque no aporta patrón general.
            id_like_cols = [
                col
                for col in df.columns
                if col.lower() in {"id", "id_cliente", "cliente_id", "codigo_cliente", "código_cliente"}
            ]

            X = df.drop(columns=[target_col] + id_like_cols, errors="ignore")
            y = df[target_col].astype(str)

            if X.empty:
                raise ValueError("No hay variables de entrada para entrenar el modelo.")
            if y.nunique() < 2:
                raise ValueError("La columna objetivo debe tener al menos 2 clases diferentes.")
            if len(df) < 20:
                raise ValueError("Se recomienda tener al menos 20 registros para entrenar.")

            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = [col for col in X.columns if col not in numeric_features]

            transformers = []
            if numeric_features:
                numeric_pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                )
                transformers.append(("num", numeric_pipeline, numeric_features))

            if categorical_features:
                categorical_pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                )
                transformers.append(("cat", categorical_pipeline, categorical_features))

            preprocessor = ColumnTransformer(transformers=transformers)

            classifier = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                alpha=0.0005,
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                random_state=42,
            )

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", classifier),
                ]
            )

            stratify = y if y.value_counts().min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=stratify,
            )

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)
            matrix = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
            labels = sorted(y.unique())

            self.model_bundle = {
                "pipeline": pipeline,
                "target_col": target_col,
                "feature_columns": X.columns.tolist(),
                "ignored_columns": id_like_cols,
                "classes": list(pipeline.named_steps["model"].classes_),
            }

            matrix_df = pd.DataFrame(matrix, index=[f"Real {l}" for l in labels], columns=[f"Pred {l}" for l in labels])

            output = (
                "ENTRENAMIENTO COMPLETADO\n"
                "=" * 70
                + "\n"
                f"Registros usados: {len(df)}\n"
                f"Variables de entrada: {len(X.columns)}\n"
                f"Columnas ignoradas por ser identificadores: {id_like_cols if id_like_cols else 'Ninguna'}\n"
                f"Variable objetivo: {target_col}\n"
                f"Clases detectadas: {', '.join(map(str, self.model_bundle['classes']))}\n"
                f"Exactitud / Accuracy: {acc:.2%}\n\n"
                "REPORTE DE CLASIFICACIÓN\n"
                + report
                + "\nMATRIZ DE CONFUSIÓN\n"
                + matrix_df.to_string()
                + "\n\nInterpretación rápida:\n"
                "- Accuracy indica el porcentaje de clientes clasificados correctamente en la prueba.\n"
                "- La matriz de confusión muestra aciertos y errores por nivel de riesgo.\n"
                "- Para un uso real, valide el modelo con datos recientes y revise falsos negativos.\n"
            )
            self._write_train_output(output)
            self.model_label.config(text="Modelo entrenado en memoria. Puede usarlo para predecir o guardarlo.")
            messagebox.showinfo("Entrenamiento finalizado", "La red neuronal fue entrenada correctamente.")

        except Exception as exc:
            self._write_train_output("ERROR DURANTE EL ENTRENAMIENTO\n\n" + str(exc) + "\n\n" + traceback.format_exc())
            messagebox.showerror("Error durante entrenamiento", str(exc))

    def save_model(self):
        if self.model_bundle is None:
            messagebox.showwarning("No hay modelo", "Primero entrene un modelo.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar modelo",
            initialdir=self.app_data_dir,
            initialfile=MODEL_FILE_DEFAULT,
            defaultextension=".joblib",
            filetypes=[("Modelo Joblib", "*.joblib"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        try:
            joblib.dump(self.model_bundle, path)
            messagebox.showinfo("Modelo guardado", f"Modelo guardado en:\n{path}")
        except Exception as exc:
            messagebox.showerror("Error al guardar modelo", str(exc))

    def load_model(self):
        path = filedialog.askopenfilename(
            title="Seleccione el modelo guardado",
            initialdir=self.app_data_dir,
            filetypes=[("Modelo Joblib", "*.joblib"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        try:
            bundle = joblib.load(path)
            required_keys = {"pipeline", "target_col", "feature_columns", "classes"}
            if not required_keys.issubset(bundle.keys()):
                raise ValueError("El archivo seleccionado no tiene la estructura de modelo esperada.")
            self.model_bundle = bundle
            self.model_label.config(text=f"Modelo cargado: {os.path.basename(path)}")
            self._write_predict_output(
                f"Modelo cargado correctamente: {path}\n"
                f"Variable objetivo original: {bundle['target_col']}\n"
                f"Variables requeridas: {', '.join(bundle['feature_columns'])}\n"
                f"Clases: {', '.join(map(str, bundle['classes']))}\n"
            )
        except Exception as exc:
            messagebox.showerror("Error al cargar modelo", str(exc))

    def load_prediction_csv(self):
        path = filedialog.askopenfilename(
            title="Seleccione el CSV de clientes a predecir",
            initialdir=self.app_data_dir,
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        try:
            self._load_prediction_csv_from_path(path)
        except Exception as exc:
            messagebox.showerror("Error al cargar CSV", str(exc))

    def load_sample_prediction_csv(self):
        try:
            self._load_prediction_csv_from_path(self.sample_predict_path)
        except Exception as exc:
            messagebox.showerror("Error al cargar clientes de ejemplo", str(exc))

    def predict_risk(self):
        if self.model_bundle is None:
            messagebox.showwarning("No hay modelo", "Primero entrene o cargue un modelo.")
            return
        if self.predict_df is None:
            messagebox.showwarning("Falta CSV", "Primero cargue un CSV de clientes para predecir.")
            return

        try:
            feature_columns = self.model_bundle["feature_columns"]
            pipeline = self.model_bundle["pipeline"]
            df = self.predict_df.copy()

            missing = [col for col in feature_columns if col not in df.columns]
            if missing:
                raise ValueError(
                    "El CSV de predicción no contiene todas las variables requeridas. "
                    f"Faltan: {', '.join(missing)}"
                )

            X_new = df[feature_columns]
            predictions = pipeline.predict(X_new)
            result = df.copy()
            result["Prediccion_Riesgo_Mora"] = predictions

            if hasattr(pipeline.named_steps["model"], "predict_proba"):
                probabilities = pipeline.predict_proba(X_new)
                result["Confianza_Prediccion"] = probabilities.max(axis=1).round(4)

                classes = list(pipeline.named_steps["model"].classes_)
                for idx, class_name in enumerate(classes):
                    result[f"Prob_{class_name}"] = probabilities[:, idx].round(4)

            default_output = "clientes_prediccion_riesgo.csv"
            initial_dir = os.path.dirname(self.predict_path) if self.predict_path else self.app_data_dir
            save_path = filedialog.asksaveasfilename(
                title="Guardar resultados de predicción",
                initialdir=initial_dir,
                initialfile=default_output,
                defaultextension=".csv",
                filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")],
            )
            if not save_path:
                return

            result.to_csv(save_path, index=False, encoding="utf-8-sig")

            summary = result["Prediccion_Riesgo_Mora"].value_counts().to_string()
            preview_cols = ["Prediccion_Riesgo_Mora"]
            if "Confianza_Prediccion" in result.columns:
                preview_cols.append("Confianza_Prediccion")
            preview = result[preview_cols].head(12).to_string(index=False)

            self._write_predict_output(
                "PREDICCIÓN COMPLETADA\n"
                "=" * 70
                + "\n"
                f"Clientes analizados: {len(result)}\n"
                f"Archivo generado: {save_path}\n\n"
                "Resumen por nivel de riesgo:\n"
                f"{summary}\n\n"
                "Vista previa de resultados:\n"
                f"{preview}\n\n"
                "Sugerencia operativa:\n"
                "- Riesgo alto: priorizar contacto preventivo y revisión de condiciones de pago.\n"
                "- Riesgo medio: seguimiento controlado y alertas tempranas.\n"
                "- Riesgo bajo: monitoreo normal de cartera.\n"
            )
            messagebox.showinfo("Predicción finalizada", f"Resultados guardados en:\n{save_path}")

        except Exception as exc:
            self._write_predict_output("ERROR DURANTE LA PREDICCIÓN\n\n" + str(exc) + "\n\n" + traceback.format_exc())
            messagebox.showerror("Error durante predicción", str(exc))


def main():
    root = tk.Tk()
    app = RiesgoMoraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
