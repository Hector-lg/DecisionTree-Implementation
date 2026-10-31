import pandas as pd

from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


def imprimir_arbol(nodo, columnas, nivel=0):
    pad = "    " * nivel
    if nodo.is_leaf:
        print(f"{pad}Hoja: {nodo.value}")
        return
    print(f"{pad}¿{columnas[nodo.feature_index]} <= {nodo.threshold}?")
    print(f"{pad}  Si:")
    imprimir_arbol(nodo.left, columnas, nivel + 1)
    print(f"{pad}  No:")
    imprimir_arbol(nodo.right, columnas, nivel + 1)



data = {
    "ID": [1, 2, 3, 4, 5, 6, 7],
    "STREAM": [False, True, True, False, False, True, True],
    "SLOPE": ["steep", "moderate", "steep", "steep", "flat", "steep", "steep"],
    "ELEVATION": ["high", "low", "medium", "medium", "high", "highest", "high"],
    "VEGETATION": ["chapparal", "riparian", "riparian", "chapparal", "conifer", "conifer", "chapparal"],
}
df = pd.DataFrame(data)


X = pd.DataFrame({
    "STREAM": df["STREAM"].astype(int),
    "SLOPE": df["SLOPE"].map({"flat": 0, "moderate": 1, "steep": 2}),
    "ELEVATION": df["ELEVATION"].map({"low": 0, "medium": 1, "high": 2, "highest": 3}),
})
y = df["VEGETATION"]

clf = DecisionTreeClassifier(criterion="gini")
clf.fit(X, y)

print("Clasificador (vegetacion)")
imprimir_arbol(clf.root, list(X.columns))
print("Prediccion:", [str(p) for p in clf.predict(X)])
print("Real:      ", list(y))
print("Accuracy:  ", clf.eval(X, y))
print()

casas = pd.DataFrame({
    "METROS": [50, 60, 70, 80, 90, 100, 120, 150],
    "CUARTOS": [1, 2, 2, 2, 3, 3, 4, 4],
    "PRECIO": [100, 120, 130, 150, 200, 210, 300, 320],
})
X_casas = casas[["METROS", "CUARTOS"]]
y_casas = casas["PRECIO"]

reg = DecisionTreeRegressor(max_depth=2, criterion="mse")
reg.fit(X_casas, y_casas)

print("Regresor (precio de casas")
imprimir_arbol(reg.root, list(X_casas.columns))
print("Prediccion:", [round(float(p), 1) for p in reg.predict(X_casas)])
print("Real:      ", list(y_casas))
print("R^2:       ", round(reg.eval(X_casas, y_casas), 3))
