import numpy as np

from arguments import parser
from utils import load_pickle_object


def compute_auc(alist):
    return np.sum(np.cumsum(alist))


def compute_rauc(rank, ideal, worst):
    rank_auc = compute_auc(rank)
    ideal_auc = compute_auc(ideal)
    worst_auc = compute_auc(worst)
    rauc = 100. * (rank_auc - worst_auc) / (ideal_auc - worst_auc)
    return rauc


def measure_rauc(ranked_lists):
    rank = ranked_lists['rank']
    ideal = ranked_lists['ideal']
    worst = ranked_lists['worst']

    result = {}
    for k in (100, 200, 300, 500):
        rauc = compute_rauc(rank[:k], ideal[:k], worst[:k])
        print('rauc for top {} elements is {:.4f}%'.format(k, rauc))
        result[f'{k}'] = rauc
    for a in (10, 20, 30, 50):
        k = int(rank.shape[0] * 0.01 * a)
        rauc = compute_rauc(rank[:k], ideal[:k], worst[:k])
        print('rauc for top {}% elements is {:.4f}%'.format(a, rauc))
        result[f'{a}%'] = rauc
    rauc = compute_rauc(rank, ideal, worst)
    print('rauc for all elements is {:.4f}%'.format(rauc))
    result['all'] = rauc

    return result


def main():
    parser.add_argument('method', type=str, choices=('random', 'pmt', 'gini', 'dissector'))
    ctx = parser.parse_args()
    print(ctx)
    ranked_lists = load_pickle_object(ctx, f'{ctx.method}_list.pkl')
    assert(ranked_lists is not None)
    num_ranked_lists = sum([1 for key in ranked_lists if key.startswith('rank')])
    if num_ranked_lists == 1:
        measure_rauc(ranked_lists)
    else:
        results = { sub: [] for sub in ('100', '200', '300', '500', \
                                        '10%', '20%', '30%', '50%', 'all')}
        for i in range(num_ranked_lists):
            rl = {
                'ideal': ranked_lists['ideal'],
                'worst': ranked_lists['worst'],
                'rank': ranked_lists[f'rank{i}']
            }
            result = measure_rauc(rl)
            for sub, rauc in result.items():
                results[sub].append(rauc)

        for sub, raucs in results.items():
            avg_rauc = sum(raucs) / len(raucs)
            print('average rauc for (top) {} elements is {:.4f}%'.format(sub, avg_rauc))


if __name__ == "__main__":
    main()
