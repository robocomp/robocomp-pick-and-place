import matplotlib.pyplot as plt
import numpy as np

class Graphics():
    def __init__(self):
        # Lists for plotting
        self.exploration = []
        self.arrivals = []
        self.farAway = []
        self.finalDist = []
        
        # Temporary lists
        self.exploration_ = []
        self.arrivals_ = []
        self.farAway_ = []
        self.finalDist_ = []

    def insertData(self):
        self.exploration.append(np.mean(self.exploration_))
        self.arrivals.append(np.sum(self.arrivals_) * 2)
        self.farAway.append(np.sum(self.farAway_) * 2)
        self.finalDist.append(np.mean(self.finalDist_) * 1000)

        self.exploration_ = []
        self.arrivals_ = []
        self.farAway_ = []
        self.finalDist_ = []

    def storeData(self, ex, ar, fa, fd):
        self.exploration_.append(ex)
        self.arrivals_.append(ar)
        self.farAway_.append(fa)
        self.finalDist_.append(fd)
        
    def show(self, epoch, period):
        t = np.arange(0, epoch, period)
        
        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.title.set_text('Exploration probability (%) vs Epoch')
        ax.plot(t, self.exploration)

        ax = fig.add_subplot(222)
        ax.title.set_text('Arrivals(%) vs Epoch')
        ax.plot(t, self.arrivals)

        ax = fig.add_subplot(223)
        ax.title.set_text('Goes away(%) vs Epoch')
        ax.plot(t, self.farAway)

        ax = fig.add_subplot(224)
        ax.title.set_text('Dist error (mm) vs Epoch')
        ax.plot(t, self.finalDist)

        plt.show()
