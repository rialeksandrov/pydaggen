import sys
import numpy as np
import argparse
import math

# TODO Move this to file with all constants / supporting file.
from enum import Enum
class complexity(Enum):
  MIXED = 0
  N_2 = 1
  N_LOG_N = 2 # (n2 log(n2)) indeed
  N_3 = 3

def make_complexity(ccr):
  if ccr == 0:
    return complexity.MIXED
  elif ccr == 1:
    return complexity.N_2
  elif ccr == 2:
    return complexity.N_LOG_N
  elif ccr == 3:
    return complexity.N_3
  else:
    print("ccr args is out of expected range")
    exit(0)

def positive_integer(str_number):
  number = int(str_number)
  if number < 0:
    raise argparse.ArgumentTypeError("%r is negative integer" % (str_number,))
  return number

def restricted_float(str_number):
  number = float(str_number)
  if number < 0.0 or number > 1.0:
    raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (str_number,))
  return number

def positive_float(str_number):
  number = float(str_number)
  if number < 0.0:
    raise argparse.ArgumentTypeError("%r is negative float" % (str_number,))
  return number

def argument_parser():
  parser = argparse.ArgumentParser(description='Daggen', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--n", type = positive_integer, default = 100,
                      help = "Positive integer. "
                             "Number of computation nodes in the DAG (i.e., application 'tasks')")
  parser.add_argument("--mindata", type = positive_integer, default = 2048,
                      help = "Positive integer. "
                             "Minimum size of data processed by a task")
  parser.add_argument("--maxdata", type = positive_integer, default = 11264,
                      help = "Positive integer. "
                           "Maximum size of data processed by a task")
  parser.add_argument("--minalpha", type = positive_float, default = 0.0,
                      help = "Positive integer. "
                             "Minimum value for extra parameter (e.g., Amdahl's law parameter)")
  parser.add_argument("--maxalpha", type = positive_float, default = 0.2,
                      help = "Positive integer. "
                             "Maximum value for extra parameter (e.g., Amdahl's law parameter)")
  parser.add_argument("--fat", type = restricted_float, default = 0.5,
                      help = "Float between [0.0, 1.0]. "
                             "Width of the DAG, that is maximum number of tasks that can be "
                             "executed concurrently. A small value will lead to a thin DAG "
                             "(e.g., chain) with a low task parallelism, while a large value "
                             "induces a fat DAG (e.g., fork-join) with a high degree of "
                             "parallelism")
  parser.add_argument("--density", type = restricted_float, default = 0.5,
                      help="Float between [0.0, 1.0]. "
                           "Determines the numbers of dependencies between taks of two "
                           "consecutive DAG levels.")
  parser.add_argument("--regular", type = restricted_float, default = 0.9,
                      help = "Float between [0.0, 1.0]. "
                             "Regularity of the distribution of tasks between the different "
                             "levels of the DAG.")
  parser.add_argument("--ccr", type = int, default = 0, choices = [0, 1, 2, 3],
                      help = "Options: [0, 1, 2, 3]. "
                             "Communication to computation ratio. In the current version this "
                             "parameter in fact merely encodes the complexity of the computation "
                             "of a task depending on the number of elements in the dataset if "
                             "processes, n. This number of elements depend on the amount of data "
                             "processed by a task. The encoding is as follows: "
                             "[1 : a . n (a is a constant picked randomly between 26 and 29)], "
                             "[2 : a . n log n], "
                             "[3 : n3/2], "
                             "[0 : Random choice among the three complexities)].")
  parser.add_argument("--jump", type = positive_integer, default = 1,
                      help = "Positive integer. "
                             "Maximum number of levels spanned by inter-task communications. "
                             "This allows to generate DAGs with execution paths of different "
                             "lengths.")
  parser.add_argument("--dot", help = "Output DOT format", action = "store_true")

  parser.add_argument("--seed", type = positive_integer, default = 179,
                      help = "Seed to use for generating random numbers")
  parser.add_argument("--output", type = str, default = "output.txt",
                      help="File where to output result")
  args = parser.parse_args()
  if args.dot:
    #print("wow!")
    t = 10
  return args

def get_int_random_number_around(val, percent):
  radius = - percent + 2.0 * percent * np.random.uniform(0.0, 1.0)
  return int(max(1.0, val * (1.0 + radius / 100.0)))


class DAG:
    class Task:
        def __init__(self):
            self.tag = 0
            self.cost = 0.0
            self.data_size = 0
            self.alpha = 0.0
            self.children = []
            self.task_indexes = []
            self.complexity = complexity.MIXED
    class Edge:
      def __init__(self, Task):
        self.to_task = Task
        self.transfer_tag = 0
        self.communication_cost = 0.0

    def __init__(self):
        self.number_of_levels = 0
        self.number_of_tasks_in_level = []
        self.levels = []


def generate_dag(args):
  np.random.seed(args.seed)
  graph = DAG()
  generate_tasks(graph, args)
  generate_dependencies(graph, args)
  return graph


def generate_tasks(graph, args):
  integral_parts = 0.0
  unused = 0.0
  (unused, integral_parts) = np.modf(np.exp(args.fat * np.log(args.n)))
  nb_tasks_per_level = int(integral_parts)
  print(integral_parts, unused)
  total_tasks = 0.0
  while total_tasks < args.n:
    tmp = get_int_random_number_around(nb_tasks_per_level, 100.00 - 100.0 * args.regular)
    tmp = int(min(tmp, args.n - total_tasks))
    tmp_array = []
    graph.levels.append(tmp_array)
    graph.number_of_levels += 1
    graph.number_of_tasks_in_level.append(tmp)
    total_tasks += tmp
  for idx in range(graph.number_of_levels):
    for jdx in range(graph.number_of_tasks_in_level[idx]):
      task = DAG.Task()
      task.data_size = 1024 * int(np.random.uniform(args.mindata, args.maxdata) / 1024.0)
      op = np.random.uniform(64.0, 512.0)

      if make_complexity(args.ccr) == complexity.MIXED:
        task.complexity = make_complexity(np.random.random_integers(1, 3))
      else:
        task.complexity = make_complexity(args.ccr)

      if task.complexity == complexity.N_2:
        task.cost = op * (task.data_size ** 2)
      elif task.complexity == complexity.N_LOG_N:
        task.cost = 2 * op * (task.data_size ** 2) * math.log2(task.data_size)
      elif task.complexity == complexity.N_3:
        task.cost = (task.data_size ** 3)
      else:
        print("Modulo error in complexity function\n")

      task.alpha = np.random.uniform(args.minalpha, args.maxalpha)
      graph.levels[idx].append(task)

def generate_dependencies(graph, args):
  for idx in range(1, graph.number_of_levels):
    for task in graph.levels[idx]:
      number_of_parents = min(graph.number_of_tasks_in_level[idx - 1], 1 +
                              np.random.random_integers(0, int(args.density * graph.number_of_tasks_in_level[idx - 1])))
      for jdx in range(number_of_parents):
        parent_level = idx - np.random.random_integers(1, min(args.jump + 1, idx))
        parent_index = np.random.random_integers(0, graph.number_of_tasks_in_level[parent_level] - 1)

        find_parent_on_this_level = False
        for ptr in range(graph.number_of_tasks_in_level[parent_level]):
          # print(parent_level, parent_index, len(graph.levels[parent_level]), len(graph.levels))
          parent = graph.levels[parent_level][parent_index]
          find_child = False
          for children in parent.children:
            if children == task:
              parent_index = (parent_index + 1) % graph.number_of_tasks_in_level[parent_level]
              find_child = True
              break

          if find_child == False:
            find_parent_on_this_level = True
            edge = DAG.Edge(task)
            edge.communication_cost = 8 * (parent.data_size ** 2)
            parent.children.append(edge)

          if find_parent_on_this_level:
            break

def dag_print(graph, args):
  f = open(args.output, 'w')
  node_count = 1

  for level in graph.levels:
    for task in level:
      task.tag = node_count
      node_count += 1
    for task in level:
      for edge in task.children:
        edge.transfer_tag = node_count
        node_count += 1
  f.write("NODE_COUNT {0:d}\n".format(node_count + 1))
  f.write("NODE 0 ")

  for i in range(graph.number_of_tasks_in_level[0] - 1):
    f.write("{0:d},".format(graph.levels[0][i].tag))
  if graph.number_of_tasks_in_level[0]:
    f.write("{0:d} ROOT 0.0 0.0\n".format(graph.levels[0][graph.number_of_tasks_in_level[0] - 1].tag))
  else:
    f.write("{0:d} ROOT 0.0 0.0\n".format(node_count))

  # Creating the regular nodes until next to last level
  for i in range(graph.number_of_levels - 1):
    for task in graph.levels[i]:
      # do the COMPUTATION
      f.write("NODE {0:d} ".format(task.tag))
      number_of_children = len(task.children)
      for i in range(number_of_children - 1):
        f.write("{0:d},".format(task.children[i].transfer_tag))
      if number_of_children:
        f.write("{0:d} COMPUTATION {1:.0f} {2:.2f}\n".format(
          task.children[number_of_children - 1].transfer_tag,
          task.cost,
          task.alpha)
        )
      else:
        f.write("{0:d} COMPUTATION {1:.0f} {2:.2f}\n".format(
          node_count,
          task.cost,
          task.alpha)
        )
      # do the TRANSFER
      for edge in task.children:
        f.write("NODE {0:d} ".format(edge.transfer_tag))
        f.write("{0:d} TRANSFER {1:.0f} 0.0\n".format(
          edge.to_task.tag,
          edge.communication_cost
          )
        )
  # Do the last level
  for task in graph.levels[graph.number_of_levels - 1]:
    f.write("NODE {0:d} {1:d} COMPUTATION {2:.0f} {3:.2f}\n".format(
      task.tag,
      node_count,
      task.cost,
      task.alpha
    ))
  # Do the end node
  f.write("NODE {0:d} - END 0.0 0.0".format(node_count))

def dot_print(graph, args):
  f = open(args.output, 'w')
  node_count = 1
  for level in graph.levels:
    for task in level:
      task.tag = node_count
      node_count += 1
  f.write("digraph G {\n")
  # Creating the regular nodes until next to last level
  for level in graph.levels:
    for task in level:
      # do the COMPUTATION
      f.write(' {0:d} [size="{1:.0f}", alpha="{2:.2f}"]\n'.format(
        task.tag, task.cost, task.alpha))
      # do the TRANSFER
      for edge in task.children:
        f.write(' {0:d} -> {1:d} [size ="{2:.0f}"]\n'.format(
          task.tag, edge.to_task.tag, edge.communication_cost))

  f.write("}\n")

def main():
  args = argument_parser()
  graph = generate_dag(args)
  if (args.dot):
    dot_print(graph, args)
  else:
    dag_print(graph, args)

if __name__ == "__main__":
  main()

