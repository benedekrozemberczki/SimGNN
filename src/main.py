from utils import tab_printer
from simgnn import SimGNNTrainer
from parser import parameter_parser

def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = SimGNNTrainer(args)
    trainer.fit()
    trainer.score()
    
if __name__ == "__main__":
    main()
