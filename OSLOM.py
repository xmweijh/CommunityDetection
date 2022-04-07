#!/usr/bin/env python

import os
import sys
import argparse
import subprocess
import shutil
import uuid


def _parse_arguments(desc, args):
    """
    Parses command line arguments
    :param desc:
    :param args:
    :return:
    """
    help_fm = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=help_fm)
    parser.add_argument('input',
                        help='Edge file in tab delimited format')
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='If set, then treat input as a directed graph')
    parser.set_defaults(directed=False)
    parser.add_argument('--singlet', dest='singlet', action='store_true',
                        help='If set, do NOT merge singlet with existing modules')
    parser.set_defaults(nosinglet=False)
    parser.add_argument('--seed', default=-1, type=int,
                        help='Seed for random generator')
    parser.add_argument('--p_val', default=0.1, type=float,
                        help='p-value: increase to get more module')
    parser.add_argument('--cp', default=0.5, type=float,
                        help='coverage parameter: Bigger value leads to bigger clusters')
    parser.add_argument('--oslomdirected', default='/oslom/OSLOM2/oslom_dir',
                        help='Full path to oslom_dir binary')
    parser.add_argument('--oslomundirected', default='/oslom/OSLOM2/oslom_undir',
                        help='Full path to oslom_undir binary')
    parser.add_argument('--tempdir', default='/tmp',
                        help='Directory needed to hold files temporarily for processing')

    return parser.parse_args(args)


def run_oslom_cmd(cmd):
    """
    Runs docker
    :param cmd_to_run: command to run as list
    :return:
    """
    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    out, err = p.communicate()

    return p.returncode, out, err


def create_tmpdir(theargs):
    """
    :param theargs:
    :return:
    """
    tmpdir = os.path.join(theargs.tempdir, 'cdoslom_' + str(uuid.uuid4()))
    os.makedirs(tmpdir, mode=0o755)
    return tmpdir


def run_oslom(graph, theargs):
    """
    :param outdir: the output directory to comprehend the output link file
    :param graph: input file
    :param directed: whether to treat input file as directed
    :param nosinglet: whether to merge singlets
    :param seed: int
    :param p_val: greater for more clusters
    :param_cp: greater for larger communities
    :return
    """

    seed = theargs.seed
    p_val = theargs.p_val
    cp = theargs.cp

    if theargs.directed is True:
        cmdargs = [theargs.oslomdirected]
    else:
        cmdargs = [theargs.oslomundirected]

    if graph is None or not os.path.isfile(graph):
        sys.stderr.write(str(graph) + ' is not a file')
        return 3

    if os.path.getsize(graph) == 0:
        sys.stderr.write(str(graph) + ' is an empty file')
        return 4

    olddir = os.getcwd()
    tmpdir = create_tmpdir(theargs)
    oldgraph = graph
    graph = os.path.join(tmpdir, os.path.basename(oldgraph))
    shutil.copyfile(oldgraph, graph)
    os.chdir(tmpdir)
    cmdargs.extend(['-f', graph, '-fast'])

    try:
        with open(graph, 'r') as file:
            lines = file.read().splitlines()
            while lines[0][0] == '#':
                lines.pop(0)
            if len(lines[0].split()) >= 3:
                weight = '-w'
            else:
                weight = '-uw'
            cmdargs.append(weight)
        if theargs.singlet is True:
            cmdargs.append('-singlet')
        if isinstance(seed, int) and seed >= 1:
            cmdargs.append('-seed')
            cmdargs.append(str(seed))
        cmdargs.extend(['-t', str(p_val), '-cp', str(cp)])
        sys.stderr.write('Running ' + str(cmdargs) + '\n')
        sys.stderr.flush()
        cmdecode, cmdout, cmderr = run_oslom_cmd(cmdargs)

        if cmdecode != 0:
            sys.stderr.write('Command failed with non-zero exit code: ' +
                             str(cmdecode) + ' : ' + str(cmderr) + '\n')
            return 1

        if len(cmdout) > 0:
            sys.stderr.write('Output from cmd: ' + str(cmdout) + '\n')

        if len(cmderr) > 0:
            sys.stderr.write('Error output from cmd: ' + str(cmderr) + '\n')

        outfolder = graph + '_oslo_files'
        clusts_layers = []
        clusts_layers.append([])
        sys.stderr.write('Attempting to open ' + outfolder + '\n')
        with open(os.path.join(outfolder, 'tp'), 'r') as cfile:
            lines = cfile.read().splitlines()
            for i in range(len(lines) // 2):
                clusts_layers[0].append([])
                members = lines[2 * i + 1].split()
                for m in members:
                    clusts_layers[0][i].append(m)
        cfile.close()
        i = 1
        sys.stderr.write('Opening something else\n')
        sys.stderr.flush()
        while os.path.isfile(os.path.join(outfolder, 'tp' + str(i))):
            with open(os.path.join(outfolder, 'tp' + str(i)), 'r') as cfile:
                clusts_layers.append([])
                lines = cfile.read().splitlines()
                for j in range(len(lines) // 2):
                    clusts_layers[i].append([])
                    members = lines[2 * j + 1].split()
                    for m in members:
                        clusts_layers[i][j].append(m)
            cfile.close()
            i = i + 1

        lines = []
        maxNode = 0
        for clust in clusts_layers[0]:
            maxNode = max(maxNode, max(list(map(int, clust))))
        for i in range(len(clusts_layers[0])):
            for n in clusts_layers[0][i]:
                lines.append(str(maxNode + i + 1) + '\t' + str(n))
        maxNode = maxNode + len(clusts_layers[0])
        for i in range(1, len(clusts_layers)):
            for j in range(len(clusts_layers[i - 1])):
                for k in range(len(clusts_layers[i])):
                    if all(x in clusts_layers[i][k] for x in clusts_layers[i - 1][j]):
                        lines.append(str(maxNode + k + 1) + '\t' + str(maxNode - len(clusts_layers[i - 1]) + j + 1))
                        break
            maxNode = maxNode + len(clusts_layers[i])
        for i in range(len(clusts_layers[-1])):
            lines.append(str(maxNode + 1) + '\t' + str(maxNode - len(clusts_layers[-1]) + i + 1))

        # trim the hierarchy to remove contigs
        up_tree = {}
        down_tree = {}
        for line in lines:
            elts = line.split()
            down_tree.setdefault(elts[0], [])
            down_tree[elts[0]].append(elts[1])
            up_tree.setdefault(elts[1], [])
            up_tree[elts[1]].append(elts[0])

        # store root and leaves
        set1 = set(down_tree.keys())
        set2 = set(up_tree.keys())
        root_l = list(set1.difference(set2))
        leaf_l = list(set2.difference(set1))
        node_l = list(set1.union(set2))

        # find all contigs in the DAG
        Contigs = []
        work_list = root_l
        visited = {}
        for node in node_l:
            visited[node] = 0
        work_path = []
        new_path = False
        while work_list:
            key = work_list.pop(0)
            if new_path == False:
                work_path.append(key)
            else:
                work_path.append(up_tree[key][visited[key]])
                work_path.append(key)
            if key in leaf_l:
                new_path = True
                Contigs.append(work_path)
                work_path = []
            elif len(down_tree[key]) > 1 or visited[key] > 0:
                new_path = True
                Contigs.append(work_path)
                work_path = []
            if visited[key] == 0 and key not in leaf_l:
                work_list = down_tree[key] + work_list
            visited[key] += 1

        # write trimmed DAG
        for path in Contigs[1:]:
            sys.stdout.write(path[0] + ',' + path[-1] + ',')
            if path[-1] in leaf_l:
                sys.stdout.write('c-m' + ';')
            else:
                sys.stdout.write('c-c' + ';')

        sys.stdout.flush()
        return 0
    finally:
        os.chdir(olddir)
        shutil.rmtree(tmpdir)


def main(args):
    """
    Main entry point for program
    :param args: command line arguments usually :py:const:`sys.argv`
    :return: 0 for success otherwise failure
    :rtype: int
    """
    desc = """
    Runs oslom on command line, sending output to standard out 
    """

    theargs = _parse_arguments(desc, args[1:])

    try:
        inputfile = os.path.abspath(theargs.input)

        return run_oslom(inputfile, theargs)

    except Exception as e:
        sys.stderr.write('Caught exception: ' + str(e) + '\n')
        return 2


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv))