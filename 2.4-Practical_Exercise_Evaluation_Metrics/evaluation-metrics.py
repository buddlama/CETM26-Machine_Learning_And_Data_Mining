from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

y_true = [1, 1, 1, 0, 0, 1, 0, 1]
y_pred = [1, 1, 0, 0, 1, 0, 0, 1]

print(confusion_matrix(y_true, y_pred))

r,s,t,u = confusion_matrix(y_true,y_pred).ravel()
print(r)
print(s)
print(u)
print(t)

tn,fp, fn,tp = confusion_matrix(y_true,y_pred).ravel()

print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"TN: {tn}")
print(f"FN: {fn}")

ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
