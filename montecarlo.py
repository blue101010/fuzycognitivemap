from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class SimpleFCM:
    """
    Classe de base pour une Carte Cognitive Floue (FCM)
    """
    def __init__(self, n_concepts):
        """
        Initialise une FCM avec n_concepts concepts
        
        Parametres:
        -----------
        n_concepts : int
            Nombre de concepts dans la FCM
        """
        self.n_concepts = n_concepts
        # Matrice des poids (n x n)
        self.weights = np.zeros((n_concepts, n_concepts))
        # Vecteur d'etat initial (valeurs des concepts)
        self.state = np.zeros(n_concepts)
        # Historique des etats
        self.history = []
        
    def set_weight(self, from_concept, to_concept, weight):
        """
        Definit le poids de la relation entre deux concepts
        
        Parametres:
        -----------
        from_concept : int
            Indice du concept source
        to_concept : int
            Indice du concept cible
        weight : float
            Poids de la relation (entre -1 et 1)
        """
        if not -1 <= weight <= 1:
            raise ValueError("Le poids doit etre entre -1 et 1")
        self.weights[to_concept, from_concept] = weight
        
    def set_initial_state(self, state):
        """
        Definit l'etat initial des concepts
        
        Parametres:
        -----------
        state : array-like
            Vecteur des valeurs initiales
        """
        self.state = np.array(state)
        self.history = [self.state.copy()]
        
    def sigmoid(self, x, lambda_param=1):
        """
        Fonction de transfert sigmoide
        
        Parametres:
        -----------
        x : array-like
            Valeurs d'entree
        lambda_param : float
            Parametre de raideur de la sigmoide
        """
        return 1 / (1 + np.exp(-lambda_param * x))
    
    def tanh_transfer(self, x):
        """
        Fonction de transfert tangente hyperbolique
        """
        return np.tanh(x)
    
    def bivalent(self, x):
        """
        Fonction de transfert bivalente
        """
        return (x > 0).astype(float)

    def kosko_inference(self, transfer_func='sigmoid', lambda_param=1):
        """
        Regle d'inference de Kosko originale (sans auto-influence)
        """
        total_influence = np.dot(self.weights, self.state)

        if transfer_func == 'sigmoid':
            return self.sigmoid(total_influence, lambda_param)
        if transfer_func == 'tanh':
            return self.tanh_transfer(total_influence)
        if transfer_func == 'bivalent':
            return self.bivalent(total_influence)
        raise ValueError(f"Fonction inconnue: {transfer_func}")

    def modified_kosko_inference(self, transfer_func='sigmoid', lambda_param=1):
        """
        Regle d'inference de Kosko modifiee (avec auto-influence)
        """
        total_influence = self.state + np.dot(self.weights, self.state)

        if transfer_func == 'sigmoid':
            return self.sigmoid(total_influence, lambda_param)
        if transfer_func == 'tanh':
            return self.tanh_transfer(total_influence)
        if transfer_func == 'bivalent':
            return self.bivalent(total_influence)
        raise ValueError(f"Fonction inconnue: {transfer_func}")

    def rescaled_inference(self, transfer_func='sigmoid', lambda_param=1):
        """
        Regle d'inference rescalee
        """
        scaled_state = 2 * self.state - 1
        scaled_influence = np.dot(self.weights, scaled_state)
        total = scaled_state + scaled_influence

        if transfer_func == 'sigmoid':
            return self.sigmoid(total, lambda_param)
        if transfer_func == 'tanh':
            return self.tanh_transfer(total)
        raise ValueError(f"Fonction inconnue: {transfer_func}")

    def simulate(self, iterations=50, inference='kosko', transfer_func='sigmoid',
                 lambda_param=1, threshold=0.001, verbose=True):
        """
        Simule l'evolution de la FCM jusqu'a convergence ou iteration max.
        """
        if inference == 'kosko':
            inference_func = self.kosko_inference
        elif inference == 'modified':
            inference_func = self.modified_kosko_inference
        elif inference == 'rescaled':
            inference_func = self.rescaled_inference
        else:
            raise ValueError(f"Inference inconnue: {inference}")

        self.history = [self.state.copy()]
        converged = False

        for iteration in range(iterations):
            new_state = inference_func(transfer_func, lambda_param)
            diff = np.abs(new_state - self.state)

            self.state = new_state
            self.history.append(self.state.copy())

            if np.all(diff < threshold):
                converged = True
                if verbose:
                    print(f"Convergence atteinte a l'iteration {iteration}")
                break

        if not converged and verbose:
            print(f"Pas de convergence apres {iterations} iterations")

        return {
            'converged': converged,
            'iterations': len(self.history) - 1,
            'final_state': self.state,
            'history': np.array(self.history)
        }

    def print_results(self):
        """Affiche l'etat final des concepts."""
        print("\nEtat final des concepts:")
        print("-" * 40)
        for i, value in enumerate(self.state):
            print(f"Concept C{i+1}: {value:.4f}")
    
class FCMMonteCarloAnalysis:
    """
    Classe pour l'analyse Monte Carlo des FCM
    """
    def __init__(self, base_fcm):
        """
        Parametres:
        -----------
        base_fcm : SimpleFCM
            FCM de base pour l'analyse
        """
        self.base_fcm = base_fcm
        self.results: Optional[np.ndarray] = None
        
    def run_monte_carlo(self, n_simulations=1000, 
                       weight_std=0.1,
                       initial_state_std=0.05):
        """
        Execute une analyse Monte Carlo
        
        Parametres:
        -----------
        n_simulations : int
            Nombre de simulations Monte Carlo
        weight_std : float
            Ecart-type pour la perturbation des poids
        initial_state_std : float
            Ecart-type pour la perturbation de l'etat initial
        """
        results = []
        base_weights = self.base_fcm.weights.copy()
        base_initial = self.base_fcm.state.copy()
        
        print(f"Execution de {n_simulations} simulations Monte Carlo...")
        
        for sim in range(n_simulations):
            # Perturber les poids (distribution normale)
            perturbed_weights = base_weights + np.random.normal(
                0, weight_std, base_weights.shape
            )
            # Garder les poids dans [-1, 1]
            perturbed_weights = np.clip(perturbed_weights, -1, 1)
            
            # Perturber l'etat initial
            perturbed_initial = base_initial + np.random.normal(
                0, initial_state_std, base_initial.shape
            )
            # Garder dans [0, 1]
            perturbed_initial = np.clip(perturbed_initial, 0, 1)
            
            # Creer une FCM avec parametres perturbes
            temp_fcm = SimpleFCM(self.base_fcm.n_concepts)
            temp_fcm.weights = perturbed_weights
            temp_fcm.set_initial_state(perturbed_initial)
            
            # Simuler
            result = temp_fcm.simulate(
                iterations=30,
                inference='kosko',
                transfer_func='sigmoid',
                threshold=0.001,
                verbose=False
            )
            
            results.append(result['final_state'])
            
            if (sim + 1) % 100 == 0:
                print(f"  {sim + 1}/{n_simulations} simulations completees")
        
        self.results = np.array(results)
        return self.results
    
    def analyze_results(self):
        """
        Analyse statistique des resultats Monte Carlo
        """
        print("\n=== Analyse Statistique Monte Carlo ===\n")
        if self.results is None or len(self.results) == 0:
            raise ValueError("Aucun resultat Monte Carlo disponible. Appelez run_monte_carlo d'abord.")

        for concept_id in range(self.results.shape[1]):
            values = self.results[:, concept_id]
            mean_val = np.mean(values)
            std_val = np.std(values)
            if len(values) > 1:
                sem = stats.sem(values)
                ci_low, ci_high = stats.t.interval(
                    0.95,
                    len(values) - 1,
                    loc=mean_val,
                    scale=sem
                )
            else:
                ci_low = ci_high = mean_val
            
            print(f"Concept C{concept_id + 1}:")
            print(f"  Moyenne      : {mean_val:.4f}")
            print(f"  Ecart-type   : {std_val:.4f}")
            print(f"  Min          : {np.min(values):.4f}")
            print(f"  Max          : {np.max(values):.4f}")
            print(f"  Quartile 25% : {np.percentile(values, 25):.4f}")
            print(f"  Mediane      : {np.median(values):.4f}")
            print(f"  Quartile 75% : {np.percentile(values, 75):.4f}")
            print(f"  IC 95% Moy.  : [{ci_low:.4f}, {ci_high:.4f}]")
            print()
    
    def plot_distributions(self, concept_names=None):
        """
        Visualise les distributions des resultats
        """
        if self.results is None or len(self.results) == 0:
            raise ValueError("Aucun resultat Monte Carlo disponible. Appelez run_monte_carlo d'abord.")

        n_concepts = self.results.shape[1]
        if concept_names is None:
            concept_names = [f"C{i+1}" for i in range(n_concepts)]
        n_cols = min(3, n_concepts)
        n_rows = int(np.ceil(n_concepts / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()
        
        for i in range(n_concepts):
            ax = axes[i]
            values = self.results[:, i]
            
            # Histogramme
            ax.hist(values, bins=30, alpha=0.7, 
                   color='steelblue', edgecolor='black')
            
            # Ligne verticale pour la moyenne
            mean_val = np.mean(values)
            ax.axvline(mean_val, color='red', 
                      linestyle='--', linewidth=2,
                      label=f'Moyenne: {mean_val:.3f}')
            
            # Intervalles de confiance (95%)
            ci_low = np.percentile(values, 2.5)
            ci_high = np.percentile(values, 97.5)
            ax.axvline(ci_low, color='green', 
                      linestyle=':', linewidth=1.5)
            ax.axvline(ci_high, color='green', 
                      linestyle=':', linewidth=1.5)
            
            ax.set_title(f'{concept_names[i]}')
            ax.set_xlabel('Valeur finale')
            ax.set_ylabel('Frequence')
            ax.legend()
            ax.grid(True, alpha=0.3)
        for j in range(n_concepts, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig('monte_carlo_distributions.png', dpi=300)
        plt.close(fig)
        print("Graphique sauvegarde: monte_carlo_distributions.png")


# Creation de la FCM avec 5 concepts
health_fcm = SimpleFCM(n_concepts=5)

# Definition des poids (matrice d'adjacence)
# C1 (Exercice) -> C5 (Sante) : +0.7
health_fcm.set_weight(from_concept=0, to_concept=4, weight=0.7)

# C2 (Alimentation) -> C5 (Sante) : +0.8
health_fcm.set_weight(from_concept=1, to_concept=4, weight=0.8)

# C3 (Stress) -> C5 (Sante) : -0.6
health_fcm.set_weight(from_concept=2, to_concept=4, weight=-0.6)

# C4 (Sommeil) -> C5 (Sante) : +0.7
health_fcm.set_weight(from_concept=3, to_concept=4, weight=0.7)

# C3 (Stress) -> C4 (Sommeil) : -0.5
health_fcm.set_weight(from_concept=2, to_concept=3, weight=-0.5)

# C1 (Exercice) -> C3 (Stress) : -0.4
health_fcm.set_weight(from_concept=0, to_concept=2, weight=-0.4)

# Definition de l'etat initial
# [Exercice, Alimentation, Stress, Sommeil, Sante]
initial_state = [0.8, 0.6, 0.7, 0.5, 0.4]
health_fcm.set_initial_state(initial_state)

# Simulation
print("=== Simulation du Systeme de Sante ===\n")
print("Etat initial:")
print("Exercice    : 0.8 (eleve)")
print("Alimentation: 0.6 (moyen)")
print("Stress      : 0.7 (eleve)")
print("Sommeil     : 0.5 (moyen)")
print("Sante       : 0.4 (faible)")

results = health_fcm.simulate(
    iterations=30,
    inference='kosko',
    transfer_func='sigmoid',
    lambda_param=1,
    threshold=0.001
)

health_fcm.print_results()

# Analyse des resultats
print("\n=== Analyse ===")
print(f"Convergence: {'Oui' if results['converged'] else 'Non'}")
print(f"Iterations: {results['iterations']}")

# Analyse Monte Carlo a partir du scenario de base
print("\n=== Analyse Monte Carlo (incertitudes) ===")
health_fcm.set_initial_state(initial_state)
monte_carlo = FCMMonteCarloAnalysis(health_fcm)
concept_names = ["Exercice", "Alimentation", "Stress", "Sommeil", "Sante"]
monte_carlo.run_monte_carlo(
    n_simulations=500,
    weight_std=0.05,
    initial_state_std=0.02
)
monte_carlo.analyze_results()
monte_carlo.plot_distributions(concept_names)