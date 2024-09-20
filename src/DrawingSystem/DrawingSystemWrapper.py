
from DrawingSystem.MultiChartDrawer import MultiChartDrawer
from DrawingSystem.ForecastDrawer import ForecastDrawer
from DrawingSystem.CombineDrawer import CombineDrawer
from DrawingSystem.ResidueDrawer import ResidueDrawer

class DrawingSystemWrapper(object):
    def __init__(self, mode : str) -> None:
        if mode == 'multi-chart':
            self.drawer = MultiChartDrawer()
        elif mode == 'forecast':
            self.drawer = ForecastDrawer()
        elif mode == 'combine':
            self.drawer = CombineDrawer()
        elif mode == 'residue':
            self.drawer = ResidueDrawer()
        else:
            raise ValueError('Bad mode')
        # end if
    # end def

    def save(self, path : str) -> None:
        self.drawer.save(path = path)
    # end def

    def plot(self) -> None:
        self.drawer.plot()
    # end def
# end class
