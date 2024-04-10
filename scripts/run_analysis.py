import json
import os
import copy
from functools import reduce
import math
import argparse

import numpy as np
import pandas as pd


pretty_map = {
    'meta-llama_Llama-2-13b-chat-hf': 'Llama2-13B-Chat',
    'meta-llama_Llama-2-7b-chat-hf': 'Llama2-7B-Chat',
    'tiiuae_falcon-7b-instruct': 'Falcon-7B-Instruct',
    'mistralai_Mistral-7B-Instruct-v0.1': 'Mistral-7B-Instruct',
    'data_bbn-mvp2-single-kdma': 'MVP2',
    'data_bbn-llm-ethics-single-kdma': 'Ethics',
    'align-selfconsistency': 'Aligned + Self-Consistency',
    'align': 'Aligned',
    'baseline': 'Unaligned',
    'high': 'High',
    'low': 'Low',
    'utilitarianism': 'utilitarianism',
    'fairness': 'fairness',
    'protocol_focus': 'protocol focus',
    'risk_aversion': 'risk aversion',
    'moral_deservingness': 'moral desert',
    'continuation_of_care': 'continuing care'
}

class ExperimentResults:

    @staticmethod
    def is_correct(iol_sample, alignment):
        chosen_idx = iol_sample['output']['choice']
        labels = iol_sample['label']

        label_names = reduce(set.union, (lab.keys() for lab in labels), set())
        assert len(label_names) == 1, "Assuming we only have a single KDMA value in labels"
        label_name = label_names.pop()

        # label_kdma_values_only = list(map(lambda x: next(iter(x.values())), labels)) # TODO not every label has a kdma value which would goof up the indexing later
        label_kdma_values_only = [lab.get(label_name, -math.inf) for lab in labels]

        assert alignment in ['high', 'low'], 'alignment must be either "high" or "low"'

        assert len(label_kdma_values_only) == len(labels), 'label_kdma_values_only and labels must be the same length' + str(labels)
        superlative_fn = max if alignment == 'high' else min

        return chosen_idx < len(label_kdma_values_only) and label_kdma_values_only[chosen_idx] == superlative_fn(label_kdma_values_only)


    def __init__(self, input_output_label, meta_data):
        assert input_output_label is not None, 'input_output_label cannot be None'
        self.input_output_label = input_output_label
        self.meta_data = meta_data

    def meta_str(self):
        sorted_keys = sorted(self.meta_data.keys())
        return ' '.join([f'{key}={self.meta_data[key]}' for key in sorted_keys])

    def merge(self, other, new_meta_data):
        return ExperimentResults(self.input_output_label + other.input_output_label, new_meta_data)

    def to_json(self):
        return {
            'input_output_label': self.input_output_label,
            'meta_data': self.meta_data
        }

    def get_accuracy(self, is_correct_fn: 'callable[int, list[dict[str, int]]]'):
        correct = 0
        total = 0
        for iol_sample in self.input_output_label:
            correct += is_correct_fn(iol_sample)
            total += 1

        if total == 0 and correct == 0:
            return 0

        return correct / total

    def get_high_accuracy_fn(self):
        return lambda iol_sample: ExperimentResults.is_correct(iol_sample, 'high')

    def get_low_accuracy_fn(self):
        return lambda iol_sample: ExperimentResults.is_correct(iol_sample, 'low')

    def get_suspect_fn(self):
        return lambda iol_sample: ExperimentResults.is_suspect(iol_sample, lambda chosen_idx, reasoning: chosen_idx == 0 or reasoning is None)

    def filter_samples(self, filter_fn: 'callable[dict]'):
        return ExperimentResults(list(filter(filter_fn, self.input_output_label)), copy.deepcopy(self.meta_data))

    def filter_by_kdma(self, kdma):
        def kdma_filter(sample):
            kdma_set = set()
            labels = sample['label']
            for label in labels:
                kdma_set = kdma_set.union(label.keys())
            return kdma in kdma_set

        return self.filter_samples(kdma_filter)

    def get_unique_kdmas(self):
        kdma_set = set()
        for sample in self.input_output_label:
            labels = sample['label']
            for label in labels:
                kdma_set = kdma_set.union(label.keys())
        return kdma_set

    def filter_by_meaningful_labels(self):
        def meaningful_labels_filter(sample):
            # labels doesn't contain an empty dict
            # there are more than one unique values in labels
            unique_values = set()
            labels = sample['label']
            for label in labels:
                if len(label) == 0:
                    return False
                unique_values = unique_values.union(label.values())
            return len(unique_values) > 1

        return self.filter_samples(meaningful_labels_filter)

    def recount_votes(self, voteing_fn):
        new_input_output_label = []
        for sample in self.input_output_label:
            new_sample = copy.deepcopy(sample)
            positive_votes = []
            negative_votes = []
            for response in sample['output']['info']['responses']:
                if response['answer_idx'] is None or response['answer_idx'] >= len(response['shuffle_indecies']):
                    print('bad response: ' + str(response))
                    continue

                vote = response['shuffle_indecies'][response['answer_idx']]
                if response['aligned']:
                    positive_votes.append(vote)
                else:
                    negative_votes.append(vote)

            new_sample['output']['choice'] = voteing_fn(positive_votes, negative_votes, n_choices=len(sample['input']['choices']))

            new_input_output_label.append(new_sample)

        return ExperimentResults(new_input_output_label, copy.deepcopy(self.meta_data))

    # define equality for ExperimentResults
    def __eq__(self, other):
        # return self.input_output_label == other.input_output_label and self.meta_data == other.meta_data
        # check deep equality
        return str(self) == str(other)

    def __str__(self):
        return json.dumps(self.to_json(), indent=4)

    def __len__(self):
        return len(self.input_output_label)


class ExperimentResultsGroup():

    def __init__(self, experiment_results):
        self.experiment_results = experiment_results

    def __eq__(self, other):
        return self.experiment_results == other.experiment_results


    def get_accuracy_mean_std(self, is_correct_fn):
        accuracies = [
            experiment_result.get_accuracy(is_correct_fn)
            for experiment_result in self.experiment_results
        ]

        return np.mean(accuracies), np.std(accuracies), sum(map(lambda x: len(x.input_output_label), self.experiment_results))

    def get_high_accuracy_mean_std(self):
        return self.get_accuracy_mean_std(ExperimentResults.get_high_accuracy_fn(self))

    def get_low_accuracy_mean_std(self):
        return self.get_accuracy_mean_std(ExperimentResults.get_low_accuracy_fn(self))

    # def get_suspect_rate_mean_std(self, is_suspect_fn):


    def recount_votes(self, voteing_fn):
        return ExperimentResultsGroup([
            experiment_result.recount_votes(voteing_fn)
            for experiment_result in self.experiment_results
        ])

    def filter_by_kdma(self, kdma):
        return ExperimentResultsGroup([
            experiment_result.filter_by_kdma(kdma)
            for experiment_result in self.experiment_results
        ])

    def filter_by_meaningful_labels(self):
        return ExperimentResultsGroup([
            experiment_result.filter_by_meaningful_labels()
            for experiment_result in self.experiment_results
        ])

    def filter_by_meta(self, meta_data_key, meta_data_value):
        try:
            return ExperimentResultsGroup([
                experiment_result
                for experiment_result in self.experiment_results
                if experiment_result.meta_data[meta_data_key] == meta_data_value
            ])
        except StopIteration:
            raise ValueError('no experiment results with meta_data_key: ' + meta_data_key + ' and meta_data_value: ' + meta_data_value)

    def filter_out_meta(self, meta_data_key, meta_data_value):
        try:
            return ExperimentResultsGroup([
                experiment_result
                for experiment_result in self.experiment_results
                if experiment_result.meta_data[meta_data_key] != meta_data_value
            ])
        except StopIteration:
            raise ValueError('no experiment results with meta_data_key: ' + meta_data_key + ' and meta_data_value: ' + meta_data_value)

    def filter_empty(self):
        return ExperimentResultsGroup([
            experiment_result
            for experiment_result in self.experiment_results
            if len(experiment_result.input_output_label) > 0
        ])

    def get_unique_kdmas(self):
        kdma_set = set()
        for experiment_result in self.experiment_results:
            kdma_set = kdma_set.union(experiment_result.get_unique_kdmas())
        return kdma_set

    def get_meta_values(self):
        meta_values = {}
        for experiment_result in self.experiment_results:
            for key, value in experiment_result.meta_data.items():
                if key not in meta_values:
                    meta_values[key] = set()
                meta_values[key].add(value)
        return meta_values

    def merge(self, other):
        return ExperimentResultsGroup(self.experiment_results + other.experiment_results)

    def add_experiment_result(self, experiment_result):
        return ExperimentResultsGroup(self.experiment_results + [experiment_result])

    def merge_all(self):
        meta_data = {
            key: next(iter(value))
            for key, value in self.get_meta_values().items()
            if len(value) == 1
        }
        return ExperimentResults(sum([
            experiment_result.input_output_label
            for experiment_result in self.experiment_results
        ], []), meta_data)

    def merge_same_meta(self):
        sub_ergs = {}
        for experiment_result in self.experiment_results:
            meta_data_str = experiment_result.meta_str()
            erg = sub_ergs.get(meta_data_str, ExperimentResultsGroup([]))
            sub_ergs[meta_data_str] = erg.add_experiment_result(experiment_result)

        return ExperimentResultsGroup([
            erg.merge_all()
            for erg in sub_ergs.values()
        ])

    def remove_meta(self, meta_data_key):
        return ExperimentResultsGroup([
            ExperimentResults(experiment_result.input_output_label, {k: v for k, v in experiment_result.meta_data.items() if k != meta_data_key})
            for experiment_result in self.experiment_results
        ])

    def merge_on_meta(self, meta_data_key):
        return self.remove_meta(meta_data_key).merge_same_meta()


    def to_json(self):
        return [
            {
                'meta_data': experiment_result.meta_data,
                'experiment_results': experiment_result.input_output_label
            }
            for experiment_result in self.experiment_results
        ]

    def __str__(self):
        return json.dumps(self.to_json(), indent=4)

    def __len__(self):
        return len(self.experiment_results)

    def add_kdma_radar(self, ax, label, alignment=True):
        kdmas = list(self.get_unique_kdmas())
        # sort kdmas
        kdmas.sort()

        high_values = {}
        low_values = {}

        for kdma in kdmas:
            high_erg = self.filter_by_kdma(kdma)
            low_erg = self.filter_by_kdma(kdma)
            if alignment:
                high_erg = high_erg.filter_by_meta('alignment', 'high')
                low_erg = low_erg.filter_by_meta('alignment', 'low')

            high_mean, high_std, n_samples = high_erg.filter_empty().get_high_accuracy_mean_std()
            low_mean, low_std, n_samples = low_erg.filter_empty().get_low_accuracy_mean_std()

            high_values[kdma] = high_mean

            low_values[kdma] = low_mean


        if high_values.keys() != low_values.keys():
            raise ValueError("All dictionaries must have the same keys.")

        # Extracting labels and stats from the dictionaries
        labels = np.array(list(high_values.keys()))

        labels = [
            pretty_map[x]
            for x in labels
        ]

        high_stats = np.array(list(high_values.values()))
        low_stats = np.array(list(low_values.values()))

        # Combining the high and low stats into one array, and their errors

        combined_stats = np.concatenate((high_stats, low_stats))

        # Creating a set of labels for each point
        combined_labels = np.array([f"High {label}" for label in labels] + [f"Low {label}" for label in labels])

        # Number of variables we're plotting.
        num_vars = len(combined_labels)

        # Compute angle each bar is centered on:
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        # angles = angles + angles[1]/2
        angles = angles.tolist()

        # The plot is circular, so we need to "complete the loop" and append the start to the end.
        combined_stats = np.concatenate((combined_stats, [combined_stats[0]]))

        angles += angles[:1]

        # Plot
        # ax.fill(angles, combined_stats, alpha=0.1)
        # plot with dashed lines
        # can't contain red or green
        # cmap = plt.cm.inferno

        # # Select a range of colors from the colormap, excluding red and green
        # colors = [cmap(i) for i in np.linspace(0, 1, 4)]
        # colors = ['#346beb', '#eba134', '#eb34e1', '#34ebe8', '#eb34e1', '#9e34eb', '#e5eb34']
        colors = ['black', 'red', 'blue', 'green']
        colors = ['#377eb8', '#ff7f00',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
        line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
        line_styles = ['solid']


        print(list(zip(combined_labels, combined_stats)))
        ax.plot(angles, combined_stats, linewidth=3, label=label, color=colors[len(ax.get_lines()) % len(colors)], linestyle=line_styles[len(ax.get_lines()) % len(line_styles)])
        # ax.plot([], [], color=ax.get_lines()[-1].get_color(), linewidth=3, label=label)  # For legend color

        # Setting the range of axes and labels
        ax.set_ylim(0, 1)
        ax.set_yticklabels([20, 40, 60, 80, 100], fontsize=20)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([''] * len(combined_labels))


        # Set x-axis tick labels with different colors for 'high' and 'low'
        xticklabels = []
        for label in combined_labels:
            if 'High' in label:
                 # underline high labels
                xticklabels.append({'label': label, 'color': '#119911', 'underline': True})
            elif 'Low' in label:
                xticklabels.append({'label': label, 'color': '#cc3333'})  # Soft blue
            else:
                xticklabels.append({'label': label, 'color': 'black'})


        # Apply the labels and colors
        for label_dict, angle in zip(xticklabels, angles):
            label = ax.text(angle, ax.get_ylim()[1] + 0.075, label_dict['label'],
                            horizontalalignment='center', color=label_dict['color'], fontsize=20)
            # Adjust alignment if needed
            if angle in [1/2 * np.pi, 3/2 * np.pi]:
                label.set_horizontalalignment('center')
            elif 1/2 * np.pi < angle < 3/2 * np.pi:
                label.set_horizontalalignment('right')
            else:
                label.set_horizontalalignment('left')


# has the potential to filter on multiple meta data values
# returns a dataframe similar to the one above
# dataframe has columns where the argument passed to the funciton is True
# for example, if model=True then the dataframe will have a 'Model' column and the accuracies will be grouped by model
# if model=True and augment=True then the dataframe will have a 'Model' and 'Augment' column and the accuracies will be grouped by model and augment
def get_accuracy_df(erg, dataset=False, model=False, augment=False, alignment=False, kdma=False, run=False):
    # Create an empty dataframe
    columns = []
    ergs = [erg]

    meta_bool_map = {
        'dataset': dataset,
        'model': model,
        'augment': augment,
        'alignment': alignment,
        'run': run
    }

    for meta, bool_val in meta_bool_map.items():
        if bool_val:
            columns.append(meta)
            ergs = sum([
                [
                    super_erg.filter_by_meta(meta, meta_val).filter_empty()
                    for meta_val in super_erg.get_meta_values()[meta]
                ]
                for super_erg in ergs
            ], [])

    if kdma:
        columns.append('kdma')
        ergs = sum([
            [
                super_erg.filter_by_kdma(kdma).filter_empty()
                for kdma in super_erg.get_unique_kdmas()
            ]
            for super_erg in ergs
        ], [])

    df = pd.DataFrame(columns=columns + ['High Accuracy', 'High Accuracy Std', 'Low Accuracy', 'Low Accuracy Std', 'Number of Samples'])

    for erg in ergs:
        meta_data = next(iter(erg.experiment_results)).meta_data
        if kdma:
            assert len(erg.get_unique_kdmas()) == 1
            meta_data['kdma'] = next(iter(erg.get_unique_kdmas()))


        if meta_data['augment'] == 'baseline':
            high_acc, high_std, n = erg.get_high_accuracy_mean_std()
            low_acc, low_std, n = erg.get_low_accuracy_mean_std()
            row = {
                col: [meta_data[col]]
                for col in columns
            }
            row.update({
                'High Accuracy': [high_acc],
                'High Accuracy Std': [high_std],
                'Low Accuracy': [low_acc],
                'Low Accuracy Std': [low_std],
                'Number of Samples': [n]
            })
            df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)

        else:
            high_acc, high_std, n = erg.get_high_accuracy_mean_std()
            low_acc, low_std, n = erg.get_low_accuracy_mean_std()
            row = {
                col: [meta_data[col]]
                for col in columns
            }
            row.update({
                'High Accuracy': [high_acc],
                'High Accuracy Std': [high_std],
                'Low Accuracy': [low_acc],
                'Low Accuracy Std': [low_std],
                'Number of Samples': [n]
            })
            df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)

    return df


def aggregate_experiments(exp_dir):
    experiments_aggregate = {
        'input_output_labels': [],
        'metrics': []
    }
    for experiment in os.listdir(exp_dir):
        path = os.path.join(exp_dir, experiment)

        # if path is not a directory, skip
        if not os.path.isdir(path):
            continue

        metrics = None
        if 'metrics.json' in os.listdir(path):
            with open(os.path.join(path, 'metrics.json')) as f:
                metrics = json.load(f)
        experiments_aggregate['metrics'].append(metrics)

        input_output_labels = None
        if 'input_output_labels.json' in os.listdir(path):
            with open(os.path.join(path, 'input_output_labels.json')) as f:
                input_output_labels = json.load(f)

        if input_output_labels is None:
            print('input_output_labels.json not found in {}'.format(path))
        experiments_aggregate['input_output_labels'].append(input_output_labels)

    return experiments_aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ADM experiment analysis")

    parser.add_argument('results_base_directory',
                        type=str,
                        help="Directory of results")
    parser.add_argument('-o', '--outdir',
                        type=str,
                        required=True,
                        help="Output directory for results")

    args = parser.parse_args()

    results_dir = args.results_base_directory
    outdir = args.outdir

    os.makedirs(outdir, exist_ok=True)

    aggregate_results_outpath = os.path.join(outdir, 'all_results.json')
    if os.path.isfile(aggregate_results_outpath):
        with open(aggregate_results_outpath) as f:
            results = json.load(f)
    else:
        results = {}
        path = results_dir
        for dataset in os.listdir(path):
            path = os.path.join(results_dir, dataset)
            if not os.path.isdir(path):
                continue
            results[dataset] = {}
            for model in os.listdir(path):
                path = os.path.join(results_dir, dataset, model)
                if not os.path.isdir(path):
                    continue
                results[dataset][model] = {}
                for augment in os.listdir(path):
                    path = os.path.join(results_dir, dataset, model, augment)
                    if not os.path.isdir(path):
                        continue
                    results[dataset][model][augment] = {}
                    if augment == 'baseline':
                        if 'latest' in path:
                            continue
                        results[dataset][model][augment] = aggregate_experiments(path)
                    else:
                        for alignment in os.listdir(path):
                            path = os.path.join(results_dir, dataset, model, augment, alignment)
                            if 'latest' in path:
                                continue
                            results[dataset][model][augment][alignment] = aggregate_experiments(path)

        with open(aggregate_results_outpath, 'w') as f:
            json.dump(results, f, indent=4)

    experiment_groups = []
    for dataset in results:
        for model in results[dataset]:
            for augment in results[dataset][model]:
                if augment == 'baseline':
                    meta_data = {
                        'dataset': dataset,
                        'model': model,
                        'augment': augment,
                        'alignment': 'baseline'
                    }
                    experiment_groups.extend([
                            ExperimentResults(x, {**meta_data, 'run': i})
                            for i, x in enumerate(results[dataset][model][augment]['input_output_labels'])
                        ])
                else:
                    for alignment in results[dataset][model][augment]:
                        meta_data = {
                            'dataset': dataset,
                            'model': model,
                            'augment': augment,
                            'alignment': alignment
                        }
                        experiment_groups.extend([
                            ExperimentResults(x, {**meta_data, 'run': i})
                            for i, x in enumerate(results[dataset][model][augment][alignment]['input_output_labels'])
                            if x is not None
                        ])

    erg = ExperimentResultsGroup(experiment_groups)

    df = get_accuracy_df(erg, model=True, augment=True, kdma=True, alignment=True, run=True)
    # sort on model
    df = df.sort_values(by=['model', 'augment', 'alignment', 'kdma'])
    # reindex
    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(outdir, 'accuracy_results.csv'), index=False)


    # group by model, augment, alignment and aggregate by mean and std
    # return a dataframe similar to the one above
    grouped_df = df.groupby(['model', 'augment', 'alignment']).agg({'High Accuracy': ['mean', 'std'], 'Low Accuracy': ['mean', 'std'], 'Number of Samples': ['sum']})
    # flatten the multi-index
    grouped_df.columns = [' '.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    grouped_df.to_csv(os.path.join(outdir, 'accuracy_results_grouped.csv'), index=False)

    x = math.sqrt(10)

    # Aggregate accuracy and std by alignment. Display both high and low accuracy for baseline.
    summary_results_rows = [['model', 'augment', 'alignment', 'error', 'num_samples']]
    for i, row in grouped_df.iterrows():
        if row['alignment'] == 'high' or row['alignment'] == 'baseline':
            summary_results_rows.append([
                row['model'],
                row['augment'],
                'high',
                str(round(row['High Accuracy mean']*100, 1)),
                str(round(row['High Accuracy std']*100/x, 1)),
                str(int(row['Number of Samples sum']))
            ])
        if row['alignment'] == 'low':
            summary_results_rows.append([
                row['model'],
                row['augment'],
                'low',
                str(round(row['Low Accuracy mean']*100, 1)),
                str(round(row['Low Accuracy std']*100/x, 1)),
                str(int(row['Number of Samples sum']))
            ])
        if row['alignment'] == 'baseline':
            summary_results_rows.append([
                row['model'],
                row['augment'],
                'low',
                str(round(row['Low Accuracy mean']*100, 1)),
                str(round(row['Low Accuracy std']*100/x, 1)),
                str(int(row['Number of Samples sum']))
            ])

    with open(os.path.join(outdir, 'summary_results.csv'), 'w') as f:
        for r in summary_results_rows:
            r_str = ','.join(r)
            print(r_str, file=f)
            # Print to stdout as well
            print(r_str)
