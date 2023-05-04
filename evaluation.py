import re
from collections import Counter

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

def parse_data(data):
    queries = re.findall(r'Query Term \d+\nWeighting Scheme: (.+?)\n(.+?)\n\n', data, re.DOTALL)
    query_results = []
    for query in queries:
        weighting_scheme = query[0]
        results = query[1].strip().split('\n')
        initials = [result[0] for result in results]
        most_occurred_initial = Counter(initials).most_common(1)[0][0]
        query_results.append((weighting_scheme, initials, most_occurred_initial))
    return query_results

def calculate_metrics(query_results):
    metrics = []
    for query_result in query_results:
        weighting_scheme = query_result[0]
        initials = query_result[1]
        most_occurred_initial = query_result[2]
        tp = initials.count(most_occurred_initial)
        fp = len(initials) - tp
        fn = 10 - tp
        tn = 90 - fp
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        ap = 0
        num_correct = 0
        for i in range(len(initials)):
            if initials[i] == most_occurred_initial:
                num_correct += 1
                ap += num_correct / (i + 1)
        if num_correct > 0:
            ap /= num_correct
        metrics.append((weighting_scheme, p, r, f1, ap))
    return metrics

def write_results(filename, metrics, map):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('Query Terms\tWeighting\tP\tR\tF\tAP\n')
        for i, query_metric in enumerate(metrics):
            query_term = f'QueryTerm-{i+1}'
            weighting_scheme = query_metric[0]
            p = query_metric[1]
            r = query_metric[2]
            f1 = query_metric[3]
            ap = query_metric[4]
            f.write(f'{query_term}\t{weighting_scheme}\t{p:.3f}\t{r:.3f}\t{f1:.3f}\t{ap:.3f}\n')
        f.write(f'Mean Average Precision for 10 queries = {map:.3f}')

data = read_file('top10tfidf.txt')
query_results = parse_data(data)
metrics = calculate_metrics(query_results)
map = sum(metric[4] for metric in metrics) / len(metrics)
write_results('resultstfidf.txt', metrics, map)

data = read_file('top10bm25.txt')
query_results = parse_data(data)
metrics = calculate_metrics(query_results)
map = sum(metric[4] for metric in metrics) / len(metrics)
write_results('resultsbm25.txt', metrics, map)