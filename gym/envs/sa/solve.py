#!/usr/bin/python

import argparse
import copy

import ising
import simulated_annealing

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Ising Solvers")
    subparsers = parser.add_subparsers(dest="solver", help="solver selection")

    # parser for config file option
    parser_file = subparsers.add_parser("file", help="read from file")
    parser_file.add_argument("filepath", type=str, help="a file describing all parameters")

    # parent parser for solvers
    parser_solver = argparse.ArgumentParser(add_help=False)
    parser_solver.add_argument("-problem", type=str, default="sample_data/graph187",
                               help="a file containing problem data in three column format, see README for details")
    parser_solver.add_argument("-steps", type=int, default=10000,
                               help="number of monte carlo steps (MCS) used, good values depend on problem size")
    parser_solver.add_argument("-dump", type=str, default=None,
                               help="a file to dump move acceptances to, if desired -- slows down performance")
    parser_solver.add_argument("-spins", type=str, default=None, help="initial spin configuration, if desired")

    # parser for simulated annealing
    parser_sa = subparsers.add_parser("sa", parents=[parser_solver],
                                      help="use simulated annealing with commandline arguments")
    parser_sa.add_argument("-T0", type=float, default=3, help="initial temperature in K, 3 to 5 is typical")
    parser_sa.add_argument("-Tf", type=float, default=0.001, help="final temperature in K, typically near zero")

    args = parser.parse_args()

    vargs = copy.deepcopy(vars(args))
    solver = vargs.pop("solver")

    if solver == "file":
        print("File based input is not yet implemented, use commandline =)")
        exit(1)

    problem = ising.SpinGlass(vargs.pop("problem"), vargs.pop("spins"))

    if solver == "sa":
        solver = simulated_annealing.SimulatedAnnealing(problem, vargs)
    else:
        print("Unknown solver: {}".format(solver))

    E_final, configuration = solver.solve()
    print(problem)
    print(solver)
    vargs = vars(args)
    arglist = "{}".format(vargs.pop("solver"))
    for key in vars(args):
        if vargs[key] is not None:
            arglist += " -{} {}".format(key, vargs[key])
    arglist += " -spins {}".format(problem.spins_to_hex(problem.spins_initial))
    print("To replicate starting conditions, run with arguments:\n\t{}\n".format(arglist))