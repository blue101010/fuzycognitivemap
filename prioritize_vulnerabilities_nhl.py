#!/usr/bin/env python3
"""
Priorisation de Vulnérabilités avec FCM et Apprentissage NHL
"""

import numpy as np


class FCM:
    """Carte Cognitive Floue avec apprentissage NHL."""
    
    def __init__(self, n_concepts, lambda_=5.0):
        self.n = n_concepts
        self.W = np.zeros((n_concepts, n_concepts))
        self.lambda_ = lambda_
        self.A = np.zeros(n_concepts)
    
    def sigmoid(self, x):
        """Fonction de transfert sigmoide."""
        return 1 / (1 + np.exp(-self.lambda_ * x))
    
    def set_weights(self, weight_matrix):
        """Definit la matrice des poids."""
        self.W = np.array(weight_matrix)
    
    def infer(self, initial_state, max_iter=50, threshold=1e-4,
              clamp_indices=None):
        """
        Inference FCM avec support du verrouillage de noeuds.
        """
        self.A = np.array(initial_state, dtype=float)
        history = [self.A.copy()]
        
        if clamp_indices is None:
            clamp_indices = []
        clamped_values = {i: self.A[i] for i in clamp_indices}
        
        for iteration in range(max_iter):
            A_new = self.sigmoid(np.dot(self.W.T, self.A))
            
            # Restaurer les noeuds verrouilles
            for idx, val in clamped_values.items():
                A_new[idx] = val
            
            history.append(A_new.copy())
            
            if np.max(np.abs(A_new - self.A)) < threshold:
                self.A = A_new
                break
            self.A = A_new
        
        return self.A, np.array(history)
    
    def nhl_learning(self, data_sequences, eta=0.1, gamma=0.9, epochs=100):
        """
        Apprentissage Hebbien Non-Lineaire (NHL).
        Reference: Section 13.5, Equation 4 de l'article
        
        w_ij^(k+1) = gamma * w_ij^(k) + eta * A_i^(k) * (A_j^(k) - A_j^(k-1))
        
        Parametres:
        -----------
        data_sequences : list of np.array
            Sequences temporelles d'activations observees
            Chaque sequence est une matrice (T x n_concepts)
        eta : float
            Taux d'apprentissage (defaut: 0.1)
        gamma : float
            Facteur de decroissance/memoire (defaut: 0.9)
        epochs : int
            Nombre d'epoques d'entrainement
        """
        print(f"\n[NHL] Debut de l'apprentissage ({epochs} epoques)...")
        
        for epoch in range(epochs):
            total_error = 0
            
            for sequence in data_sequences:
                for t in range(1, len(sequence)):
                    A_prev = sequence[t - 1]
                    A_curr = sequence[t]
                    delta_A = A_curr - A_prev
                    
                    # Mise a jour NHL (Equation 4)
                    for i in range(self.n):
                        for j in range(self.n):
                            if i != j:
                                delta_w = eta * A_prev[i] * delta_A[j]
                                self.W[i, j] = gamma * self.W[i, j] + delta_w
                                # Contrainte [-1, 1]
                                self.W[i, j] = np.clip(self.W[i, j], -1, 1)
                    
                    # Calcul erreur de prediction
                    A_pred = self.sigmoid(np.dot(self.W.T, A_prev))
                    total_error += np.mean((A_curr - A_pred) ** 2)
            
            if epoch % 20 == 0:
                print(f"    Epoch {epoch:3d}, MSE: {total_error:.6f}")
        
        print(f"[NHL] Apprentissage termine. MSE finale: {total_error:.6f}")
        return self.W


def generate_training_data(n_samples=50):
    """
    Genere des donnees d'entrainement synthetiques.
    Simule des cas historiques de vulnerabilites avec priorites connues.
    
    En pratique, ces donnees proviendraient de:
    - Historique de remediations
    - Jugements d'experts sur des cas passes
    - Incidents de securite documentes
    """
    print("\n[DATA] Generation de donnees d'entrainement...")
    
    sequences = []
    
    for i in range(n_samples):
        # Etat initial aleatoire (caracteristiques de vulnerabilite)
        cvss = np.random.uniform(0.3, 1.0)
        epss = np.random.uniform(0.0, 1.0)
        exploit = np.random.uniform(0.0, 1.0)
        complexity = np.random.uniform(0.2, 0.8)
        chaining = np.random.uniform(0.0, 0.7)
        criticality = np.random.uniform(0.3, 1.0)
        exposure = np.random.uniform(0.1, 1.0)
        controls = np.random.uniform(0.1, 0.9)
        
        # Priorite cible (simulee selon une logique metier)
        # Haute priorite si: CVSS eleve + EPSS eleve + exploit mature + criticite elevee
        priority_target = (
            0.25 * cvss +
            0.25 * epss +
            0.15 * exploit +
            0.20 * criticality +
            0.10 * exposure -
            0.15 * controls +
            np.random.normal(0, 0.05)  # Bruit
        )
        priority_target = np.clip(priority_target, 0, 1)
        
        # Sequence: etat initial -> etat final (convergence simulee)
        state_initial = np.array([
            cvss, epss, exploit, complexity, chaining,
            criticality, exposure, controls, 0.0
        ])
        
        state_final = np.array([
            cvss, epss, exploit, complexity, chaining,
            criticality, exposure, controls, priority_target
        ])
        
        # Creer une sequence de transition (3 etapes)
        sequence = np.array([
            state_initial,
            (state_initial + state_final) / 2,  # Etape intermediaire
            state_final
        ])
        
        sequences.append(sequence)
    
    print(f"[DATA] {n_samples} sequences generees.")
    return sequences


def prioritize_vulnerabilities(vulnerabilities, fcm_model):
    """
    Priorise une liste de vulnerabilites avec FCM.
    """
    results = []
    
    for vuln in vulnerabilities:
        initial_state = np.array([
            vuln['cvss'] / 10.0,
            vuln['epss'],
            vuln['exploit_mature'],
            0.5,
            vuln.get('chaining', 0.3),
            vuln['asset_criticality'],
            vuln['exposure'],
            vuln.get('controls', 0.5),
            0.0
        ])
        
        # Verrouillage des noeuds d'entree (C0-C7)
        clamped_nodes = list(range(len(initial_state) - 1))
        final_state, history = fcm_model.infer(
            initial_state,
            clamp_indices=clamped_nodes
        )
        priority_score = final_state[-1]
        
        results.append({
            'cve_id': vuln['cve_id'],
            'priority_score': priority_score,
            'final_state': final_state,
            'iterations': len(history) - 1
        })
    
    results.sort(key=lambda x: x['priority_score'], reverse=True)
    return results


def main():
    """
    Pipeline complet: Apprentissage NHL + Priorisation
    """
    print("=" * 65)
    print("  FCM avec Apprentissage NHL pour Priorisation de Vulnerabilites")
    print("=" * 65)
    
    # 1. Creation du modele FCM
    print("\n[1] Creation du modele FCM (9 concepts)...")
    fcm = FCM(n_concepts=9, lambda_=5.0)
    
    # 2. Initialisation avec poids experts (point de depart)
    print("[2] Initialisation avec matrice de poids experte...")
    W_init = np.array([
        [0.0, 0.3, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.7],
        [0.0, 0.0, 0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 0.8],
        [0.0, 0.0, 0.0,-0.3, 0.5, 0.0, 0.0, 0.0, 0.6],
        [0.0, 0.0, 0.0, 0.0,-0.2, 0.0, 0.0, 0.0,-0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
        [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0,-0.2, 0.6],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.4],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    fcm.set_weights(W_init)
    
    # 3. Generer donnees d'entrainement
    training_data = generate_training_data(n_samples=100)
    
    # 4. Apprentissage NHL
    print("\n[3] Apprentissage NHL des poids...")
    W_learned = fcm.nhl_learning(
        training_data,
        eta=0.05,      # Taux d'apprentissage
        gamma=0.95,    # Facteur de memoire
        epochs=100
    )
    
    # 5. Afficher les poids appris vs initiaux
    print("\n[4] Comparaison des poids (colonne Priorite C8):")
    print("-" * 50)
    concepts = ["CVSS", "EPSS", "Exploit", "Complx", 
                "Chain", "Critic", "Expos", "Ctrl"]
    print(f"{'Concept':<10} {'Initial':>10} {'Appris NHL':>12}")
    print("-" * 50)
    for i, name in enumerate(concepts):
        print(f"{name:<10} {W_init[i, 8]:>10.3f} {W_learned[i, 8]:>12.3f}")
    
    # 6. Vulnerabilites a prioriser
    print("\n[5] Priorisation des vulnerabilites...")
    vulns = [
        {'cve_id': 'CVE-2024-0001', 'cvss': 9.8, 'epss': 0.95,
         'exploit_mature': 1.0, 'asset_criticality': 0.9,
         'exposure': 0.8, 'controls': 0.2},
        {'cve_id': 'CVE-2024-0002', 'cvss': 7.5, 'epss': 0.15,
         'exploit_mature': 0.0, 'asset_criticality': 0.6,
         'exposure': 0.3, 'controls': 0.7},
        {'cve_id': 'CVE-2024-0003', 'cvss': 8.1, 'epss': 0.72,
         'exploit_mature': 0.8, 'asset_criticality': 0.95,
         'exposure': 0.9, 'controls': 0.3},
        {'cve_id': 'CVE-2024-0004', 'cvss': 5.5, 'epss': 0.05,
         'exploit_mature': 0.2, 'asset_criticality': 0.4,
         'exposure': 0.2, 'controls': 0.9},
    ]
    
    ranked = prioritize_vulnerabilities(vulns, fcm)
    
    # 7. Resultats
    print("\n" + "=" * 65)
    print("       RESULTATS DE PRIORISATION (FCM + NHL)")
    print("=" * 65)
    
    for i, v in enumerate(ranked, 1):
        score = v['priority_score']
        if score > 0.8:
            niveau = "CRITIQUE"
        elif score > 0.6:
            niveau = "ELEVE"
        elif score > 0.4:
            niveau = "MOYEN"
        else:
            niveau = "FAIBLE"
        
        print(f"\n{i}. {v['cve_id']}")
        print(f"   Score: {score:.4f} | Niveau: {niveau}")
    
    print("\n" + "=" * 65)
    return ranked


if __name__ == "__main__":
    main()