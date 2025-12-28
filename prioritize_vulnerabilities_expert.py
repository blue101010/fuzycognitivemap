#!/usr/bin/env python3
"""
Vulnerability Prioritization with Fuzzy Cognitive Maps (FCM)
"""

import numpy as np


class FCM:
    """
    Fuzzy Cognitive Map with inference support.
    """
    
    def __init__(self, n_concepts, lambda_=5.0):
        """
        Initialize an FCM instance.
        
        Parameters:
        -----------
        n_concepts : int
            Number of concepts in the FCM
        lambda_ : float
            Steepness parameter for the sigmoid
        """
        self.n = n_concepts
        self.W = np.zeros((n_concepts, n_concepts))
        self.lambda_ = lambda_
        self.A = np.zeros(n_concepts)
    
    def sigmoid(self, x):
        """
        Sigmoid transfer function.
        
        f(x) = 1 / (1 + e^(-lambda * x))
        """
        return 1 / (1 + np.exp(-self.lambda_ * x))
    
    def set_weights(self, weight_matrix):
        """
        Set the weight matrix.
        
        Parameters:
        -----------
        weight_matrix : np.array
            Causal weight matrix (n x n)
        """
        self.W = np.array(weight_matrix)
    
    def infer(self, initial_state, max_iter=50, threshold=1e-4, clamp_indices=None):
        """
        Run FCM inference until convergence.
        Reference: Equation 2 (Kosko rule)
        
        A_i^(t+1) = f(sum_j(w_ji * A_j^(t)))
        
        Parameters:
        -----------
        initial_state : array-like
            Initial activation vector
        max_iter : int
            Maximum number of iterations
        threshold : float
            Convergence tolerance
        clamp_indices : sequence[int] or None
            Concept indices that must stay clamped to their initial values
            (data nodes)
            
        Returns:
        ---------
        tuple : (final_state, history)
        """
        initial_state = np.array(initial_state, dtype=float)
        self.A = initial_state.copy()
        clamp_indices = (np.array(clamp_indices, dtype=int)
                          if clamp_indices is not None else None)
        history = [self.A.copy()]
        
        for iteration in range(max_iter):
            # Kosko rule: A_new = f(W^T * A)
            A_new = self.sigmoid(np.dot(self.W.T, self.A))
            if clamp_indices is not None:
                A_new[clamp_indices] = initial_state[clamp_indices]
            history.append(A_new.copy())
            delta = np.max(np.abs(A_new - self.A))
            self.A = A_new
            
            # Convergence check
            if delta < threshold:
                break
        
        return self.A, np.array(history)


def prioritize_vulnerabilities(vulnerabilities, fcm_model):
    """
    Prioritize a list of vulnerabilities.
    
    Parameters:
    -----------
    vulnerabilities : list of dict
        Entries containing: 'cve_id', 'cvss', 'epss',
        'exploit_mature', 'asset_criticality', 'exposure'
    fcm_model : FCM
        Trained FCM model
    
    Returns:
    ---------
    list : Vulnerabilities sorted by descending priority
    """
    results = []
    
    for vuln in vulnerabilities:
        # Build the initial activation vector
        # Concepts:
        # C0: CVSS_Base (normalized [0,1])
        # C1: Exploitability (EPSS)
        # C2: Exploit_Maturity
        # C3: Complexity (default 0.5)
        # C4: Chaining potential
        # C5: Asset_Criticality
        # C6: Exposure
        # C7: Controls
        # C8: Priority (output)
        
        initial_state = np.array([
            vuln['cvss'] / 10.0,              # C0: Normalized CVSS
            vuln['epss'],                      # C1: EPSS
            vuln['exploit_mature'],            # C2: Exploit maturity
            0.5,                               # C3: Default complexity
            vuln.get('chaining', 0.3),         # C4: Chaining potential
            vuln['asset_criticality'],         # C5: Asset criticality
            vuln['exposure'],                  # C6: Exposure
            vuln.get('controls', 0.5),         # C7: Compensating controls
            0.0                                # C8: Priority (output)
        ])
        
        # FCM inference with clamped input nodes (C0-C7)
        clamped_nodes = list(range(len(initial_state) - 1))
        final_state, history = fcm_model.infer(
            initial_state,
            clamp_indices=clamped_nodes
        )
        priority_score = final_state[-1]  # Last concept = Priority
        
        results.append({
            'cve_id': vuln['cve_id'],
            'priority_score': priority_score,
            'final_state': final_state,
            'iterations': len(history) - 1
        })
    
    # Sort by decreasing priority
    results.sort(key=lambda x: x['priority_score'], reverse=True)
    return results


def create_expert_weight_matrix():
    """
    Build the expert-provided initial weight matrix.
    
    Concepts:
        0: CVSS, 1: EPSS, 2: Exploit, 3: Complexity, 4: Chaining,
        5: Criticality, 6: Exposure, 7: Controls, 8: Priority
    """
    W_expert = np.array([
        # CVSS  EPSS  Expl  Cmpl  Chain Crit  Expo  Ctrl  Prio
        [0.0,  0.3,  0.2,  0.0,  0.1,  0.0,  0.0,  0.0,  0.7],   # CVSS
        [0.0,  0.0,  0.4,  0.0,  0.2,  0.0,  0.0,  0.0,  0.8],   # EPSS
        [0.0,  0.0,  0.0, -0.3,  0.5,  0.0,  0.0,  0.0,  0.6],   # Exploit
        [0.0,  0.0,  0.0,  0.0, -0.2,  0.0,  0.0,  0.0, -0.3],   # Complexity
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.5],   # Chaining
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.9],   # Criticality
        [0.0,  0.0,  0.3,  0.0,  0.0,  0.0,  0.0, -0.2,  0.6],   # Exposure
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.4],   # Controls
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],   # Priority
    ])
    return W_expert


def print_results(ranked_vulns):
    """
    Display prioritization results.
    """
    print("\n" + "=" * 60)
    print("       PRIORISATION DES VULNÉRABILITÉS (FCM)")
    print("=" * 60)
    
    for i, v in enumerate(ranked_vulns, 1):
        print(f"\n{i}. {v['cve_id']}")
        print(f"   Score de priorité : {v['priority_score']:.4f}")
        print(f"   Convergence en    : {v['iterations']} itérations")
        
        # Score interpretation
        score = v['priority_score']
        if score > 0.8:
            niveau = "CRITIQUE - Action immédiate requise"
        elif score > 0.6:
            niveau = "ÉLEVÉ - Traitement prioritaire"
        elif score > 0.4:
            niveau = "MOYEN - Planifier correction"
        else:
            niveau = "FAIBLE - Surveillance"
        print(f"   Niveau            : {niveau}")
    
    print("\n" + "=" * 60)


def main():
    """
    Full usage example.
    """
    print("=" * 60)
    print("  FCM pour la Priorisation de Vulnérabilités")

    print("=" * 60)
    
    # 1. Create the FCM model (9 concepts)
    print("\n[1] Création du modèle FCM avec 9 concepts...")
    fcm = FCM(n_concepts=9, lambda_=5.0)
    
    # 2. Configure the expert weight matrix
    print("[2] Chargement de la matrice de poids experte...")
    W_expert = create_expert_weight_matrix()
    fcm.set_weights(W_expert)
    
    print("\n    Concepts du modèle:")
    concepts = [
        "C0: CVSS_Base", "C1: Exploitabilité (EPSS)", "C2: Maturité_Exploit",
        "C3: Complexité", "C4: Chaînage", "C5: Criticité_Actif",
        "C6: Exposition", "C7: Contrôles", "C8: Priorité (sortie)"
    ]
    for c in concepts:
        print(f"    - {c}")
    
    # 3. Define the vulnerabilities to prioritize
    print("\n[3] Vulnérabilités à analyser:")
    
    vulns = [
        {
            'cve_id': 'CVE-2024-0001',
            'cvss': 9.8,
            'epss': 0.95,
            'exploit_mature': 1.0,
            'asset_criticality': 0.9,
            'exposure': 0.8,
            'chaining': 0.4,
            'controls': 0.2
        },
        {
            'cve_id': 'CVE-2024-0002',
            'cvss': 7.5,
            'epss': 0.15,
            'exploit_mature': 0.0,
            'asset_criticality': 0.6,
            'exposure': 0.3,
            'chaining': 0.1,
            'controls': 0.7
        },
        {
            'cve_id': 'CVE-2024-0003',
            'cvss': 8.1,
            'epss': 0.72,
            'exploit_mature': 0.8,
            'asset_criticality': 0.95,
            'exposure': 0.9,
            'chaining': 0.6,
            'controls': 0.3
        },
        {
            'cve_id': 'CVE-2024-0004',
            'cvss': 5.5,
            'epss': 0.05,
            'exploit_mature': 0.2,
            'asset_criticality': 0.4,
            'exposure': 0.2,
            'chaining': 0.1,
            'controls': 0.9
        },
    ]
    
    for vuln in vulns:
        print(f"\n    {vuln['cve_id']}:")
        print(f"      CVSS: {vuln['cvss']}, EPSS: {vuln['epss']}, "
              f"Exploit: {vuln['exploit_mature']}")
        print(f"      Criticité: {vuln['asset_criticality']}, "
              f"Exposition: {vuln['exposure']}, Contrôles: {vuln['controls']}")
    
    # 4. Prioritization
    print("\n[4] Exécution de la priorisation FCM...")
    ranked = prioritize_vulnerabilities(vulns, fcm)
    
    # 5. Display results
    print_results(ranked)
    
    # 6. Extra details
    print("\n[5] Détail des états finaux (valeurs des concepts):")
    print("-" * 60)
    concept_names = ["CVSS", "EPSS", "Exploit", "Complx", 
                     "Chain", "Critic", "Expos", "Ctrl", "PRIO"]
    
    print(f"{'CVE':<15}", end="")
    for name in concept_names:
        print(f"{name:>7}", end="")
    print()
    print("-" * 60)
    
    for v in ranked:
        print(f"{v['cve_id']:<15}", end="")
        for val in v['final_state']:
            print(f"{val:>7.3f}", end="")
        print()
    
    return ranked


if __name__ == "__main__":
    results = main()