import numpy as np

def sigmoid(x, lam=1):
    """Fonction de transfert sigmoide."""
    return 1 / (1 + np.exp(-lam * x))

def simulate_fcm(W, A0, max_iter=50, threshold=0.001, 
                 transfer='sigmoid', lam=1):
    """
    Simule une FCM jusqu'a convergence ou max_iter.
    
    Parametres:
    -----------
    W : np.array - Matrice de poids (n x n)
    A0 : np.array - Vecteur d'activation initial
    max_iter : int - Nombre maximum d'iterations
    threshold : float - Seuil de convergence
    transfer : str - 'sigmoid' ou 'tanh'
    lam : float - Parametre lambda pour sigmoid
    
    Retourne:
    ---------
    history : list - Historique des etats
    converged : bool - True si convergence atteinte
    """
    n = len(A0)
    A = A0.copy()
    history = [A.copy()]
    
    # Choix de la fonction de transfert
    if transfer == 'sigmoid':
        f = lambda x: sigmoid(x, lam)
    else:
        f = np.tanh
    
    for iteration in range(max_iter):
        # Calcul du nouvel etat (regle de Kosko modifiee)
        A_new = f(W @ A + A)
        history.append(A_new.copy())
        
        # Verification de convergence
        if np.max(np.abs(A_new - A)) < threshold:
            return history, True
        
        A = A_new
    
    return history, False

# Exemple d'utilisation
concepts = ['Stress', 'Sommeil', 'Cafe', 'Productivite']
W = np.array([
    [0,    0,     0,    0],
    [-0.7, 0,    -0.6,  0],
    [0.4,  0,     0,    0],
    [0,    0.8,   0.5,  0]
])

# Etat initial: stress eleve
A0 = np.array([0.8, 0.5, 0.3, 0.5])

# Simulation
history, converged = simulate_fcm(W, A0, lam=1)

print(f"Convergence: {converged}")
print(f"Iterations: {len(history)-1}")
print(f"\nEtat final:")
for i, c in enumerate(concepts):
    print(f"  {c}: {history[-1][i]:.3f}")