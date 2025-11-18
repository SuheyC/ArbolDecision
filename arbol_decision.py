# =============================================
# 1. Importación de librerías necesarias
# =============================================
from sklearn.datasets import load_wine          # Cargar el dataset Wine
from sklearn.tree import DecisionTreeClassifier # Crear el modelo de árbol
from sklearn.tree import export_text, plot_tree # Exportar reglas y graficar árbol
from sklearn.model_selection import train_test_split # Dividir datos
import matplotlib.pyplot as plt                 # Para mostrar el gráfico

# =============================================
# 2. Cargar el dataset del vino
# =============================================
wine = load_wine()
X, y = wine.data, wine.target      # X = características químicas, y = clases de vino

print("Características:", wine.feature_names)
print("Clases:", wine.target_names)
print("Shape de X:", X.shape)

# =============================================
# 3. Dividir los datos en entrenamiento (80%) y prueba (20%)
# =============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.80,
    test_size=0.20,
    random_state=10,
    shuffle=True
)

# =============================================
# 4. Crear el clasificador (primero con max_depth = 2)
# =============================================
tree_depth2 = DecisionTreeClassifier(max_depth=2)
tree_depth2.fit(X_train, y_train)

# =============================================
# 5. Mostrar precisión del modelo
# =============================================
accuracy_depth2 = tree_depth2.score(X_test, y_test)
print("\nPrecisión con max_depth=2:", accuracy_depth2)

# =============================================
# 6. Exportar y mostrar reglas del árbol
# =============================================
rules_depth2 = export_text(tree_depth2, feature_names=wine.feature_names)
print("\nReglas del árbol con max_depth=2:\n")
print(rules_depth2)

# =============================================
# 7. Graficar el árbol de decisión con max_depth=2
# =============================================
plt.figure(figsize=(10, 7), dpi=100)  # Tamaño para que se vea completo
plot_tree(
    tree_depth2,
    filled=True,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    fontsize=8  # Texto más pequeño para mejor visualización
)
plt.title("Árbol de Decisión (max_depth=2)")
plt.tight_layout()  # Evita que se corte
plt.show()

# ============================================================
# 8. ENTRENAR MODELO SIN LÍMITE DE PROFUNDIDAD (max_depth=None)
# ============================================================
tree_full = DecisionTreeClassifier(max_depth=None)
tree_full.fit(X_train, y_train)

# Precisión sin límite
accuracy_full = tree_full.score(X_test, y_test)
print("\nPrecisión con max_depth=None:", accuracy_full)

# Reglas
rules_full = export_text(tree_full, feature_names=wine.feature_names)
print("\nReglas del árbol completo (sin límite):\n")
print(rules_full)

# ============================================================
# 9. Graficar el árbol COMPLETO — versión resumida (para que se vea)
# ============================================================
plt.figure(figsize=(10, 7), dpi=100)
plot_tree(
    tree_full,
    filled=True,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    fontsize=6,
    max_depth=4  # Solo mostramos primeros niveles para no saturar
)
plt.title("Árbol de decisión completo (vista reducida)")
plt.tight_layout()
plt.show()

# ============================================================
# 10. (Opcional) Graficar TODO el árbol completo real
# ============================================================
# ⚠ Este árbol puede ser muy grande, se verá pero con letra más pequeña

plt.figure(figsize=(10, 7), dpi=100)
plot_tree(
    tree_full,
    filled=True,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    fontsize=4
)
plt.title("Árbol de decisión COMPLETO (max_depth=None)")
plt.tight_layout()
plt.show()
