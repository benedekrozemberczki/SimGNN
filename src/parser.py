import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description = "Run SimGNN.")

    parser.add_argument("--training-graphs",
                        nargs = "?",
                        default = "./dataset/train/",
	                help = "Edge list csv.")

    parser.add_argument("--testing-graphs",
                        nargs = "?",
                        default = "./dataset/test/",
	                help = "Feature json.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 5,
	                help = "Number of training epochs. Default is 300.")

    parser.add_argument("--filters-1",
                        type = int,
                        default = 128,
	                help = "Filters (neurons) in convolution. Default is 16.")

    parser.add_argument("--filters-2",
                        type = int,
                        default = 64,
	                help = "Filters (neurons) in convolution. Default is 16.")

    parser.add_argument("--filters-3",
                        type = int,
                        default = 32,
	                help = "Filters (neurons) in convolution. Default is 16.")

    parser.add_argument("--tensor-neurons",
                        type = int,
                        default = 16,
	                help = "Filters (neurons) in convolution. Default is 16.")

    parser.add_argument("--bottle-neck-neurons",
                        type = int,
                        default = 16,
	                help = "Order of Chebyshev polynomial. Default is 8.")

    parser.add_argument("--batch-size",
                        type = int,
                        default = 128,
	                help = "Order of Chebyshev polynomial. Default is 20.")

    parser.add_argument("--bins",
                        type = int,
                        default = 16,
	                help = "Filters (neurons) in convolution. Default is 16.")

    parser.add_argument("--test-size",
                        type = float,
                        default = 0.2,
	                help = "Ratio of training samples. Default is 0.2.")

    parser.add_argument("--dropout",
                        type = float,
                        default = 0.5,
	                help = "Dropout probability. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.001,
	                help = "Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type = float,
                        default = 5*10**-4,
	                help = "Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--histogram",
                        dest="histogram",
                        action="store_true")

    parser.set_defaults(histogram=False)

    return parser.parse_args()
