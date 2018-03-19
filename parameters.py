TEST_SET_RATIO = 0.1
BALANCE = False
REG_DRAW_MIN = 0.47
REG_DRAW_MAX = 0.53

CLASSES_NAMES = ['Away win', 'Draw', 'Home win']


########## NMF parameters ##########
NMF_INITS = ['random', 'nndsvd', 'nndsvda', 'nndsvdar']
NMF_SOLVERS = ['cd']
NMF_BETA_LOSSES = ['frobenius']
NMF_MAX_ITERS = [200, 500]
NMF_ALPHAS = [0, 0.01, 0.1]
NMF_L1_RATIOS = [0, 0.5, 1]
NMF_SHUFFLES = [False, True]


########## MLPRegressor parameters ##########
MLP_REG_LAYERS_SIZES = [(100,50), (100,50,10), (200,100,50)]
MLP_REG_ACTIVATIONS = ['tanh', 'relu']
MLP_REG_ALPHAS = [10**x for x in range(-3, 1)]
MLP_REG_LEARNING_RATES = ['constant']#, 'invscaling', 'adaptive']
MLP_REG_MOMENTUMS = [0.01, 0.1, 1]


########## SVM parameters ##########
SVM_CS = [10**x for x in range(2, 4)]
SVM_KERNELS = ['poly', 'rbf']
SVM_SHRINKINGS = [True, False]


########## MLPClassifier parameters ##########
MLP_CLS_LAYERS_SIZES = [(50,), (100,50), (100,50,10), (200,100,50)]
MLP_CLS_ACTIVATIONS = ['tanh', 'relu']
MLP_CLS_ALPHAS = [10**x for x in range(-3, 1)]
#MLP_CLS_LEARNING_RATES = ['constant', 'invscaling', 'adaptive']
