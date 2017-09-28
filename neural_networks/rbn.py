from neurolab.core import Net
from neurolab.core import Layer
from neurolab import layer
import numpy as np
import neurolab as nl


class Radial_Basis:
    
    out_minmax = [-1, 1] # min and max values for output
    
    inp_active = [-4, 4] # range of function
    
    """
        Actual functionality, how the rbf convert values
    """
    def __call__(self, x):
        return 1/np.exp(x*x)
    
    
    def deriv(self, x, y):
        """
        Derivative of transfer function RB

        """
        return -2*x/np.exp(x*x)

    
"""
    RBN Layer class
"""
    
class RBN(Layer):
    
    def __init__(self, ci, cn):
        
        Layer.__init__(self, ci, cn, cn, {'w': (cn, ci), 'b': cn})
        
        
        # Activation function
        self.transf = Radial_Basis()
        
        # Output range
        self.out_minmax[:] = np.asfarray([Radial_Basis.out_minmax] * self.cn)
        
        # Step values
        self.s = np.zeros(self.cn)
        
        
    # What happens with values on each step
    def _step(self, inp):
        self.s = layer.euclidean(self.np['w'], inp.reshape([1, len(inp)]))
        self.s *= self.np['b']
        return self.transf(self.s)
    

class Linear_Sum(Layer):
    
    def __init__(self, ci, cn):

        Layer.__init__(self, ci, cn, cn, {'w': (cn, ci), 'b': cn})

        self.transf = nl.trans.PureLin()
        
        self.out_minmax = np.asfarray([nl.trans.PureLin.out_minmax] * self.co)
        #self.initf = init.initwb_nw
        self.s = np.zeros(self.cn)

    def _step(self, inp):
        self.s = np.sum(self.np['w'] * inp, axis=1)
        self.s += self.np['b']
        return self.transf(self.s)

"""
    New network with 2 layers
"""

def newpnn(minmax, cn0, cn1, bias):
    
    # Input size for RBN layer
    ci = len(minmax)

    # Create RBN layer
    layer_inp = RBN(ci, cn0)
    
    # Create competitive layer
    layer_out = layer.Competitive(cn0, cn1)

    layer_inp.np['b'].fill(bias)
    layer_inp.np['w'].fill(0.0)

    layer_out.np['w'].fill(0.0)

    net = Net(minmax, cn1, [layer_inp, layer_out], [[-1], [0], [1]], None, None)

    return net


def newpnn_linear(minmax, cn0, cn1, bias):
    
    # Input size for RBN layer
    ci = len(minmax)
    # Create RBN layer
    layer_inp = RBN(ci, cn0)
    
    # Create competitive layer
    layer_out = Linear_Sum(cn0, cn1)

    layer_inp.np['b'].fill(bias)
    layer_inp.np['w'].fill(0.0)

    layer_out.np['w'].fill(0.0)

    net = Net(minmax, cn1, [layer_inp, layer_out], [[-1], [0], [1]], None, None)

    return net

