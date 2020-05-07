#coding: utf-8
import argparse
from functools import partial
from time import time

from branch_bounds import branch_and_bounds
from brute_force import brute_force
from dynamic_programming import dynamic_programming
from fptas import FPTAS
from sa import annealing_algorithm
from ratio_greedy import ratio_greedy

BRUTE_FORCE_METHOD = "brute"
RATIO_GREEDY_METHOD = "ratio"
DYNAMIC_PROGRAMMING_METHOD = "dynamic"
BRANCH_AND_BOUNDS_METHOD = "bandb"
FPTAS_METHOD = "fptas"
GENETIC_METHOD = "sa"


def parse_line(line):
    """Line parser method
    :param line: line from input file
    :return: tuple like: (instance id, number of items, knapsack capacity,
                            list of tuples like: [(weight, cost), (weight, cost), ...])
    """
    parts = [int(value) for value in line.split()]
    inst_id, number, capacity = parts[0:3]
    weight_cost = [(parts[i], parts[i + 1]) for i in range(3, len(parts), 2)]
    return inst_id, number, capacity, weight_cost


def solver(method, inst_file_path, solution_file_path):
    """Main method that solves knapsack problem using one of the existing methods

    :param method: knapsack problem solving method
    :param inst_file_path: path to file with input instances
    :param solution_file_path: path to file where solver should write output data
    """
    inst_file = open(inst_file_path, "r")
    sol_file = open(solution_file_path, "w")

    for line in inst_file:
        inst_id, number, capacity, weight_cost = parse_line(line)
        # get best cost and variables combination
        best_cost, best_combination = method(number, capacity, weight_cost)
        best_combination_str = " ".join("%s" % i for i in best_combination)
        # write best result to file
        sol_file.write("%s %s %s  %s\n" % (inst_id, number, best_cost, best_combination_str))

    inst_file.close()
    sol_file.close()

def solverV(method, number, capacity, weight_cost):
    """Main method that solves knapsack problem using one of the existing methods
    """

    # get best cost and variables combination
    best_cost, best_combination = method(number, capacity, weight_cost)
    best_combination_str = " ".join("%s" % i for i in best_combination)


    cap=0
    index=0
    for i in best_combination:
        if i == 1:
            cap += weight_cost[index][0]
        index += 1


    # return best results
    return best_cost, best_combination_str, cap


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Script solving the 0/1 knapsack problem')
    # parser.add_argument('-f', '--inst-file', required=True, type=str, dest="inst_file_path",
    #                     help='Path to inst *.dat file')
    # parser.add_argument('-o', type=str, dest="solution_file_path", default="output.sol.dat",
    #                     help='Path to file where solutions will be saved. Default value: output.sol.dat')
    # parser.add_argument('-r', type=int, dest="repeat", default=1,
    #                     help='Number of repetitions. Default value: 1')
    # parser.add_argument("-m", default=BRUTE_FORCE_METHOD, type=str, dest="method",
    #                     choices=[BRUTE_FORCE_METHOD, RATIO_GREEDY_METHOD, DYNAMIC_PROGRAMMING_METHOD,
    #                              BRANCH_AND_BOUNDS_METHOD, FPTAS_METHOD, GENETIC_METHOD],
    #                     help="Solving method. Default value: brute force method")
    # parser.add_argument('-s', type=float, dest="scaling_factor", default=4.0,
    #                     help='Scaling factor for FPTAS algorithm. Default value: 4.0')
    # parser.add_argument('-t', type=int, dest="temperature", default=100,
    #                     help='Initial temperature for annealing approach. Default value: 100')
    # parser.add_argument('-n', type=int, dest="steps", default=100,
    #                     help='Number of steps for annealing approach iteration. Default value: 100')
    # args = parser.parse_args()
    #
    # # selecting knapsack problem solving method
    # if args.method == BRUTE_FORCE_METHOD:
    #     method = brute_force
    # elif args.method == RATIO_GREEDY_METHOD:
    #     method = ratio_greedy
    # elif args.method == DYNAMIC_PROGRAMMING_METHOD:
    #     method = dynamic_programming
    # elif args.method == BRANCH_AND_BOUNDS_METHOD:
    #     method = branch_and_bounds
    # elif args.method == FPTAS_METHOD:
    #     if args.scaling_factor <= 1:
    #         raise Exception("Scaling factor for FPTAS must be greater than 1")
    #     method = partial(FPTAS, scaling_factor=args.scaling_factor)
    # elif args.method == GENETIC_METHOD:
    #     if args.temperature < 1:
    #         raise Exception("Initial temperature for annealing approach must be greater than 0")
    #     if args.steps < 1:
    #         raise Exception("Number of steps for annealing approach iteration must be greater than 0")
    #     method = partial(annealing_algorithm, init_temp=args.temperature, steps=args.steps)
    # else:
    #     raise Exception("Unknown solving method")


    #1
    # inst_file = "inst/knap_30.inst.dat"
    # sol_file = "sol/knap_30.sol.dat"
    # method = brute_force
    #
    # solving_time = 0
    # t_start = time()
    # solver(method, inst_file, sol_file)
    # t_finish = time()
    # solving_time += (t_finish - t_start)
    # print (solving_time )


    #4 items
    number=4
    capacity= 100
    weight_cost = [(18, 114), (42, 136), (88, 192), (3, 223)]
    method = brute_force

    solving_time = 0
    t_start = time()
    best_cost,best_config, best_weight = solverV(method, number, capacity, weight_cost)
    t_finish = time()
    solving_time += (t_finish - t_start)
    print(solving_time)
    print ("The Best Cost is: %s" % best_cost)
    print ("The Best Configuration is: %s" % best_config)
    print ("The Best Weight is: %s" % best_weight)

    #
    #
    # #20 items
    # number=20
    # capacity= 250
    # weight_cost = [(22, 175), (4, 131), (2, 30), (7, 11), (26, 135), (6, 71), (1, 249), (16, 141), (43, 138),
    #                (15, 164), (40, 252), (21, 172), (3, 9), (19, 88), (48, 70), (18, 42), (49, 146), (8, 182),
    #                (41, 68), (27, 67)]
    # method = brute_force
    #
    # solving_time = 0
    # t_start = time()
    # best_cost,best_config, best_weight = solverV(method, number, capacity, weight_cost)
    # t_finish = time()
    # solving_time += (t_finish - t_start)
    # print(solving_time)
    # print ("The Best Cost is: %s" % best_cost)
    # print ("The Best Configuration is: %s" % best_config)
    # print ("The Best Weight is: %s" % best_weight)
    #
    #
    # #25 items
    # number=25
    # capacity= 300
    # weight_cost = [(2, 132), (1, 67), (4, 101), (35, 181), (10, 106), (3, 46), (22, 179), (19, 139), (48, 196),
    #                (15, 55), (31, 32),(28, 107), (14, 248), (7, 61), (27, 0), (6, 66), (36, 43), (8, 100), (46, 0),
    #                (13, 195), (41, 210), (30, 248),(9, 39), (39, 186), (32, 16)]
    # method = brute_force
    #
    # solving_time = 0
    # t_start = time()
    # best_cost,best_config, best_weight = solverV(method, number, capacity, weight_cost)
    # t_finish = time()
    # solving_time += (t_finish - t_start)
    # print(solving_time)
    # print ("The Best Cost is: %s" % best_cost)
    # print ("The Best Configuration is: %s" % best_config)
    # print ("The Best Weight is: %s" % best_weight)
    #
    #
    # #27 items
    # number=27
    # capacity= 350
    # weight_cost = [(17, 254), (11, 89), (1, 22), (19, 131), (6, 243), (4, 70), (49, 200), (45, 240), (10, 155), (2, 2),
    #                (18, 59),(3, 109), (8, 114), (13, 220), (15, 180), (21, 210), (16, 249), (12, 135), (22, 6),
    #                (35, 244), (34, 232), (20, 249), (33, 201), (36, 74), (42, 164), (38, 186), (5, 130)]
    # method = brute_force
    #
    # solving_time = 0
    # t_start = time()
    # best_cost,best_config, best_weight = solverV(method, number, capacity, weight_cost)
    # t_finish = time()
    # solving_time += (t_finish - t_start)
    # print(solving_time)
    # print ("The Best Cost is: %s" % best_cost)
    # print ("The Best Configuration is: %s" % best_config)
    # print ("The Best Weight is: %s" % best_weight)
    #


    #30 items
    number=30
    capacity= 400
    weight_cost =  [(22, 61), (28, 24), (1, 31), (6, 73), (38, 92), (5, 168), (11, 65), (20, 4), (46, 54), (3, 165), (32, 17),
     (14, 251), (42, 146), (35, 45), (33, 147), (21, 108), (4, 211), (15, 78), (8, 216), (40, 59), (39, 235), (2, 152),
     (17, 187), (9, 9), (44, 3), (16, 40), (12, 72), (43, 67), (7, 175), (25, 126)]

    method = brute_force

    solving_time = 0
    t_start = time()
    best_cost,best_config, best_weight = solverV(method, number, capacity, weight_cost)
    t_finish = time()
    solving_time += (t_finish - t_start)
    print(solving_time)
    print ("The Best Cost is: %s" % best_cost)
    print ("The Best Configuration is: %s" % best_config)
    print ("The Best Weight is: %s" % best_weight)



    # repeating "repeat" time to get average solving time
    # for i in range(args.repeat):
    #     t_start = time()
    #     solver(method, args.inst_file_path, args.solution_file_path)
    #     t_finish = time()
    #     solving_time += (t_finish - t_start)
    #
    # print "Average solving time: %ss (repetitions count %s)" % (solving_time / args.repeat, args.repeat)