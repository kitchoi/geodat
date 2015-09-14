import pylab

def default():
    ''' Default settings:
    All lines with linewidth=2 
    '''
    lines = pylab.gca().lines
    for line in lines:
        line.set(linewidth=2.)
    pylab.draw()

