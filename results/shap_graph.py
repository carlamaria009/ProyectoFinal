def generate_shap_outputs(model, x_data, feature_names, output_dir="results"):
    import shap
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    class_names = np.load("src/class_labels.npy", allow_pickle=True)
    
    os.makedirs(output_dir, exist_ok=True)
    print("🔍 Generando explicaciones SHAP ...")

    # Aplanar si es necesario
    if len(x_data.shape) > 2:
        x_data = x_data.reshape(x_data.shape[0], -1)

    if len(feature_names) != x_data.shape[1]:
        raise ValueError(f"❌ Longitud de feature_names ({len(feature_names)}) no coincide con columnas de X ({x_data.shape[1]})")

    background = shap.sample(x_data, 100, random_state=42)
    x_sample = x_data[:100]

    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(x_sample)

    # ⚠️ Corregir si viene en forma (100, 364, 8)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        print("⚠️ Convertido shap_values de array 3D a lista de arrays 2D (por clase)")
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    # Calcular media absoluta de SHAP
    shap_abs_mean = np.mean([np.abs(val).mean(axis=0) for val in shap_values], axis=0)

    # Plot
    shap.initjs()
    plt.figure()
    #shap.summary_plot(shap_values, x_sample, feature_names=feature_names, plot_type='bar', show=False)
    shap.summary_plot(
    shap_values,
    x_sample,
    feature_names=feature_names,
    class_names=class_names,  # 👈 Esto es lo nuevo
    plot_type='bar',
    show=False
    )
    png_path = os.path.join(output_dir, "shap_summary_plot_svm.png")
    plt.savefig(png_path)
    plt.show()
    plt.close()
    print(f"🖼️ Gráfico SHAP guardado en: {png_path}")

    # CSV
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP_Value": shap_abs_mean
    }).sort_values(by="Mean_Abs_SHAP_Value", ascending=False)

    csv_path = os.path.join(output_dir, "shap_feature_importance_svm.csv")
    shap_df.to_csv(csv_path, index=False)
    print(f"📄 Importancias SHAP guardadas en CSV: {csv_path}")
