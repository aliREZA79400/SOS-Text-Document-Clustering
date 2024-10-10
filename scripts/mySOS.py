from mealpy.optimizer import Optimizer


class OriginalSOS(Optimizer):


    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.is_parallelizable = False
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            ## Mutualism Phase
            jdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            mutual_vector = (self.pop[idx].solution + self.pop[jdx].solution) / 2
            bf1, bf2 = self.generator.integers(1, 3, 2)
            xi_new = self.pop[idx].solution + self.generator.random() * (self.g_best.solution - bf1 * mutual_vector)
            xj_new = self.pop[jdx].solution + self.generator.random() * (self.g_best.solution - bf2 * mutual_vector)
            xi_new = self.correct_solution(xi_new)
            xj_new = self.correct_solution(xj_new)
            xi_target = self.get_target(xi_new)
            xj_target = self.get_target(xj_new)
            if self.compare_target(xi_target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=xi_new, target=xi_target)
            if self.compare_target(xj_target, self.pop[jdx].target, self.problem.minmax):
                self.pop[jdx].update(solution=xj_new, target=xj_target)
            ## Commensalism phase
            jdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            xi_new = self.pop[idx].solution + self.generator.uniform(-1, 1) * (self.g_best.solution - self.pop[jdx].solution)
            xi_new = self.correct_solution(xi_new)
            xi_target = self.get_target(xi_new)
            if self.compare_target(xi_target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=xi_new, target=xi_target)
            ## Parasitism phase
            jdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            temp_idx = self.generator.integers(0, self.problem.n_dims)
            xi_new = self.pop[jdx].solution.copy()
            xi_new[temp_idx] = self.problem.generate_solution()[temp_idx]
            xi_new = self.correct_solution(xi_new)
            xi_target = self.get_target(xi_new)
            ##changed here from idx to jdx
            if self.compare_target(xi_target, self.pop[jdx].target, self.problem.minmax):
                self.pop[idx].update(solution=xi_new, target=xi_target)
