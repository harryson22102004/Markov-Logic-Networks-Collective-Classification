import numpy as np
from collections import defaultdict
 
class MarkovLogicNetwork:
    def __init__(self):
        self.formulas = []  # (weight, formula_fn)
        self.groundings = {}
 
    def add_formula(self, weight, formula_fn, description=""):
        self.formulas.append((weight, formula_fn, description))
 
    def ground(self, constants):
        """Ground all formulas with given constants."""
        self.constants = constants
 
    def feature_vector(self, assignment):
        """Count satisfied groundings for each formula."""
        return np.array([sum(1 for c1 in self.constants
                             for c2 in self.constants
        if c1 != c2 and fn(assignment, c1, c2))
                          for _, fn, _ in self.formulas])
 
    def log_likelihood(self, assignment):
        w = np.array([wt for wt, _, _ in self.formulas])
        f = self.feature_vector(assignment)
        return np.dot(w, f)
 
def run_inference_mcmc(mln, init_assignment, n_iter=1000, T=1.0):
    """MC-SAT style inference."""
    assignment = dict(init_assignment)
    nodes = list(assignment.keys())
    best = dict(assignment); best_ll = mln.log_likelihood(assignment)
    for _ in range(n_iter):
        node = np.random.choice(nodes)
        old_val = assignment[node]
        assignment[node] = 1 - old_val
        ll_new = mln.log_likelihood(assignment)
        ll_old = mln.log_likelihood({**assignment, node: old_val})
        if np.random.rand() < min(1, np.exp((ll_new - ll_old)/T)):
            if ll_new > best_ll: best = dict(assignment); best_ll = ll_new
        else:
            assignment[node] = old_val
    return best
 
# Citation graph: papers and authors
constants = ['p1','p2','p3','a1','a2']
assignment = {'p1_ml':1,'p2_ml':0,'p3_ml':1,'p1_p2_cite':1,'p2_p3_cite':1}
mln = MarkovLogicNetwork()
mln.add_formula(2.0, lambda a,c1,c2: a.get(f'{c1}_ml',0) and a.get(f'{c1}_{c2}_cite',0), "Same topic propagates")
mln.add_formula(1.5, lambda a,c1,c2: a.get(f'{c1}_ml',0) == a.get(f'{c2}_ml',0), "Cited papers same topic")
mln.ground(constants)
result = run_inference_mcmc(mln, assignment, n_iter=500)
print("MLN inference result (sampled assignments):")
for k,v in result.items(): print(f"  {k}: {v}")
