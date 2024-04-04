import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Plot hyperparameter performance
def plot_hyperparameter_performance(grid_result):
    params = grid_result.param_grid
    scores = grid_result.cv_results_['mean_test_score']
    params_str = [str(param) for param in params['neurons_layer1']]
    sns.barplot(x=params_str, y=scores)
    plt.xlabel('Neurons Layer 1')
    plt.ylabel('Mean F1 Score')
    plt.title('Performance du modèle avec différentes hyperparamètres')
    plt.show()

# Plot learning curves
def plot_learning_curves(history):
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Courbes d\'apprentissage')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies étiquettes')
    plt.title('Matrice de confusion')
    plt.show()

# Plot ROC curve and AUC
def plot_roc_curve(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.show()