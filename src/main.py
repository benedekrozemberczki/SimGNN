from simgnn import SimGNNTrainer
from parser import parameter_parser
from utils import tab_printer

def main():
    """
    Parsing command line parameters, reading data, doing sparsification, fitting a GWNN and saving the logs.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = SimGNNTrainer(args)
    trainer.fit()
    trainer.score()
    

if __name__ == "__main__":
    main()
