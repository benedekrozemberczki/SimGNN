"""SGCN runner."""

from sgcn import SignedGCNTrainer
from param_parser import parameter_parser
from utils import tab_printer, read_graph, score_printer, save_logs

def main():
    """
    Parsing command line parameters.
    Creating target matrix.
    Fitting an SGCN.
    Predicting edge signs and saving the embedding.
    """
    args = parameter_parser()
    tab_printer(args)
    edges = read_graph(args)
    trainer = SignedGCNTrainer(args, edges)
    trainer.setup_dataset()
    trainer.create_and_train_model()
    if args.test_size > 0:
        trainer.save_model()
        score_printer(trainer.logs)
        save_logs(args, trainer.logs)

if __name__ == "__main__":
    main()
