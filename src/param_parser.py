import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the Bitcoin OTC dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """
    parser = argparse.ArgumentParser(description="Run SGCN.")

    parser.add_argument("--edge-path",
                        nargs="?",
                        default="./input/bitcoin_otc.csv",
	                help="Edge list csv.")

    parser.add_argument("--features-path",
                        nargs="?",
                        default="./input/bitcoin_otc.csv",
	                help="Edge list csv.")

    parser.add_argument("--embedding-path",
                        nargs="?",
                        default="./output/embedding/bitcoin_otc_sgcn.csv",
	                help="Target embedding csv.")

    parser.add_argument("--regression-weights-path",
                        nargs="?",
                        default="./output/weights/bitcoin_otc_sgcn.csv",
	                help="Regression weights csv.")

    parser.add_argument("--log-path",
                        nargs="?",
                        default="./logs/bitcoin_otc_logs.json",
	                help="Log json.")

    parser.add_argument("--epochs",
                        type=int,
                        default=100,
	                help="Number of training epochs. Default is 100.")

    parser.add_argument("--reduction-iterations",
                        type=int,
                        default=30,
	                help="Number of SVD iterations. Default is 30.")

    parser.add_argument("--reduction-dimensions",
                        type=int,
                        default=64,
	                help="Number of SVD feature extraction dimensions. Default is 64.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
	                help="Random seed for sklearn pre-training. Default is 42.")

    parser.add_argument("--lamb",
                        type=float,
                        default=1.0,
	                help="Embedding regularization parameter. Default is 1.0.")

    parser.add_argument("--test-size",
                        type=float,
                        default=0.2,
	                help="Test dataset size. Default is 0.2.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
	                help="Learning rate. Default is 0.01.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10**-5,
	                help="Learning rate. Default is 10^-5.")

    parser.add_argument("--layers",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space. E.g. 32 32.")

    parser.add_argument("--spectral-features",
                        dest="spectral_features",
                        action="store_true")

    parser.add_argument("--general-features",
                        dest="spectral_features",
                        action="store_false")

    parser.set_defaults(spectral_features=True)

    parser.set_defaults(layers=[32, 32])

    return parser.parse_args()
