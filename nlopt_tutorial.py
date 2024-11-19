import nlopt
import numpy as np
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from copy import deepcopy


class Geom:
    side: float
    curr_coor: np.ndarray = None
    history: list = []
    color: None

    def __init__(self, side):
        self.side = side

    def set_coor(self, coor):
        if self.curr_coor is not None:
            self.history.append(deepcopy(self.curr_coor))
        self.curr_coor = deepcopy(coor)

    def curr_min(self):
        return self.min(self.curr_coor)

    def min(self, coor: np.ndarray):
        return coor - np.array([self.side/2, self.side/2])

    def max(self, coor: np.ndarray):
        return coor + np.array([self.side/2, self.side/2])

    def get_mpatches(self):
        # anchor = self.curr_min()
        patches = []

        # print(len(self.history))
        cmap = plt.get_cmap('OrRd')
        cnt = 0
        # for h in self.history[::int(len(self.history)/100)]:
        #     anchor = self.min(h)
        #     patch = mpatches.Rectangle(
        #         anchor, self.side, self.side, color=cmap(cnt),  fill=None)
        #     patches.append(patch)
        #     cnt += 0.01

        anchor = self.curr_min()
        patch = mpatches.Rectangle(
            anchor, self.side, self.side, color="#FF0000",  fill=None)
        patches.append(patch)

        return patches


class SearchSpace:
    x_space: np.ndarray
    y_space: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray

    lower_bounds: np.ndarray
    upper_bounds: np.ndarray

    placements = []

    def __init__(self,
                 min_x: float,
                 min_y: float,
                 max_x: float,
                 max_y: float,
                 x_resolution: float,
                 y_resolution: float):

        self.lower_bounds = np.array([min_x, min_y])
        self.upper_bounds = np.array([max_x, max_y])

        self.x_space = np.linspace(min_x, max_x, x_resolution)
        self.y_space = np.linspace(min_y, max_y, y_resolution)
        self.X, self.Y = np.meshgrid(self.x_space, self.y_space)
        self.Z = (1 - self.X/2 + self.X**5 + self.Y**3) * \
            np.exp(-self.X**2 - self.Y**2)

    def get_cost(self, obj: Geom, coor: np.ndarray = None) -> float:
        if coor is None:
            coor = obj.curr_coor

        btm_lft_corner = obj.min(coor)
        tp_rt_corner = obj.max(coor)

        x_indx = np.argwhere(np.logical_and(
            self.x_space >= btm_lft_corner[0], self.x_space <= tp_rt_corner[0]))
        y_indx = np.argwhere(np.logical_and(
            self.y_space >= btm_lft_corner[1], self.y_space <= tp_rt_corner[1]))
        from itertools import product

        res = list(product(x_indx.T[0].tolist(), y_indx.T[0].tolist()))
        vals = []
        for i in res:
            vals.append(self.Z[i[::-1]])
        if len(vals):
            return np.sum(vals)
        else:
            return 9999.9

    def place(self, geom, coor):
        geom.set_coor(coor)
        cost = self.get_cost(geom)
        entry = {"coor": coor,
                 "obj": geom,
                 "cost": cost}
        self.placements.append(entry)

    def solve_optimization_problem(self):
        # algorithm argument doesnt really matter
        # number of dimensions is 2 for the x and y axes
        opt = nlopt.opt(nlopt.GN_MLSL_LDS, 2)
        # lower bound dictated by the space parameters
        opt.set_lower_bounds(self.lower_bounds)
        opt.set_upper_bounds(self.upper_bounds)

        # opt.add_equality_constraint(self.collision_constraint, tol=0)

        opt.set_min_objective(self.min_objective)
        opt.set_xtol_rel(1e-4)
        opt.set_maxeval(1000)
        random_start = np.random.uniform(
            self.lower_bounds, self.upper_bounds, None)
        min_coor = opt.optimize(random_start)

        dx = opt.get_initial_step(random_start)
        minf = opt.last_optimum_value()

        self.placements[0]["obj"].set_coor(min_coor)
        print("Solution :", min_coor)

    def collision_constraint(self, params, grad):

        pass

    def min_objective(self, params, grad):
        # a = deepcopy(np.array(params))
        # self.placements[0]["obj"].set_coor(a)
        cost = search_space.get_cost(self.placements[0]["obj"],
                                     np.array(params))
        return cost

    def plot(self):
        fig, ax = plt.subplots()
        h = plt.contourf(self.x_space, self.y_space, self.Z)

        for entry in self.placements:
            mpatches = entry["obj"].get_mpatches()
            for artist in mpatches:
                ax.add_artist(artist)

        plt.axis('scaled')
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    # create the space/map/whatever
    # grid spanning between -3 and 3
    search_space = SearchSpace(-3.0, -3.0,
                               3.0, 3.0,
                               256, 256)  # 128, 128)

    geom = Geom(0.5)  # square of size 0.25m
    # place the object in the space
    search_space.place(geom, np.array([-2.0, -1.0]))
    # get the cost of the object with respect to its placement
    search_space.get_cost(geom)
    search_space.solve_optimization_problem()
    search_space.plot()
