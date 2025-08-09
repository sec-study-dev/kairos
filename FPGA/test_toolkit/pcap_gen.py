from cgi import print_exception
from util.graph import Edge, Graph, createGraph, isNegCycleBellmanFord
from util.pcap import PcapGen
import numpy as np
import pandas as pd
import argparse



exch_index2id = np.array([[1, 3], [3, 1], [0, 4], [4, 0], [3, 2], [2, 3], [1, 2], [2, 1], [
                         0, 2], [2, 0], [3, 0], [0, 3], [1, 0], [0, 1], [1, 4], [4, 1], [4, 2], [2, 4]], dtype=int)

id2secid = np.array([(i+1)*1024 for i in range(18)])


def random_data_gen(require_arb, no_arb):
    rates = 10**((np.random.random(18) - 0.5)*8)
    if(require_arb):
        while(not check_cycle(rates)):
            rates = 10**((np.random.random(18) - 0.5)*8)
    elif(no_arb):
        while(check_cycle(rates)):
            rates = 10**((np.random.random(18) - 0.5)*8)
    time_stamp_start = 1643350419136975104

    timpe_stamps = (np.arange(0, 18)*10**8 + (np.random.random(18) - 0.5)
                    * 10**7).astype(int) + time_stamp_start

    entry_type = np.array([48, 49]*9)
    d = {
        "Timestamp": timpe_stamps,
        "MDEntryType": entry_type,
        "SecurityID": np.array(id2secid),

        "MDEntryPx": (rates*10000000).astype(int),
    }
    df = pd.DataFrame(d)


    print()
    print(9)
    for i in range(9):
        print(rates[i], rates[i+1])

    return df, rates


def check_cycle(rates):

    V = 5
    E = 18
    graph = createGraph(V, E)
    logged_rates = np.log(rates)
    for i in range(18):
        graph.edge[i].src = exch_index2id[i][0]
        graph.edge[i].dest = exch_index2id[i][0]
        graph.edge[i].weight = -logged_rates[i]
    return isNegCycleBellmanFord(graph, 0)


def parse_arg():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="selected_sub")
    parser_csv = subparsers.add_parser('r', help='read a exsisting csv file')
    parser_csv.add_argument(
        '-c', '--csv', help="path of csv file", required=True)
    parser_random = subparsers.add_parser('g', help='generate random data')
    required_arbitrage_group = parser_random.add_mutually_exclusive_group()
    required_arbitrage_group.add_argument(
        '--no_arb', action="store_true", help="required the output data to have no arbitrage route")
    required_arbitrage_group.add_argument(
        '--req_arb', action="store_true", help="required the output data to have at least one arbitrage route")
    parser_random.add_argument(
        '--output_csv', help="also output the csv file of generated data to a designated path")
    parser.add_argument(
        '-o', '--output', help="output path and name of pcap, default: cme_input_gen.pcap", default='cme_input_gen.pcap')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    data, rates = None, None
    if(args.selected_sub == 'r'):

        data = pd.read_csv(args.csv)
    elif(args.selected_sub == 'g'):
        data, rates = random_data_gen(args.req_arb, args.no_arb)
        import sys
        if(check_cycle(rates)):
            print("The generated Data has a cycle.", file=sys.stderr)
        else:
            print("The generated Data don't have a cycle.", file=sys.stderr)
    else:
        exit()
    pcap_gen = PcapGen(pcap_path=args.output)

    for index, row in data.iterrows():
        pcap_gen.construct_payload(
            row['Timestamp'], row['MDEntryType'], row['SecurityID'], row['MDEntryPx'])
        pcap_gen.write_pcap()

    if(vars(args).get("output_csv") != None):
        data.to_csv(args.output_csv, index=False)
