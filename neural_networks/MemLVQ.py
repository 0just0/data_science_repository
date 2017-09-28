from neurolab.core import Net
from neurolab import trans
from neurolab import layer
from neurolab import train
from neurolab import error


def newlvq_for_memory(minmax, cn0, cn1):

    ci = len(minmax)

    layer_inp = layer.Competitive(ci, cn0)
    layer_out = layer.Perceptron(cn0, cn1, trans.PureLin())
    layer_out.initf = None
    layer_out.np['b'].fill(0.0)
    layer_out.np['w'].fill(0.0)

    net = Net(minmax, cn1, [layer_inp, layer_out],
                            [[-1], [0], [1]], train.train_lvq, error.MSE())

    return net
