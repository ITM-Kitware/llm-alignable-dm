import json
import os
import torch
import yaml
import sys
import traceback
from datetime import datetime
import argparse

import align_system.evaluation.adm_evaluator as adm_evaluator
from align_system.algorithms.abstracts import AlignedDecisionMaker


class DummyADM(AlignedDecisionMaker):

    def __init__(self, default_choice, predict_kdma_values):
        self.default_choice = default_choice
        self.predict_kdma_values = predict_kdma_values

    def __call__(self, sample, target_kdma_values, **kwargs):
        response = {
            'choice': self.default_choice
        }

        if self.predict_kdma_values:
            response['predicted_kdma_values'] = [
                {
                    kdma_name: 0
                    for kdma_name in target_kdma_values
                }
                for _ in sample['choices']
            ]

        return response


class OracleADM(AlignedDecisionMaker):

    def __init__(self, flip_alignment, predict_kdma_values):
        self.flip_alignment = flip_alignment
        self.predict_kdma_values = predict_kdma_values

    def __call__(self, sample, target_kdma_values, labels, **kwargs):

        eval_fn = max
        if self.flip_alignment:
            eval_fn = min

        choice_idx = eval_fn(range(len(labels)),
            key=lambda i: adm_evaluator.kitware_similarity_score(target_kdma_values, labels[i])
        )

        response = {'choice': choice_idx}

        if self.predict_kdma_values:
            response['predicted_kdma_values'] = labels

        return response


def chat_kdma_predicting_adm(config):
    from align_system.algorithms.chat_kdma_predicting_adm import ChatKDMAPredictingADM

    algorithm = ChatKDMAPredictingADM.load_model(
        device=config['language_model']['device'],
        hf_model_name=config['language_model']['model_name'],
        precision={
            'half': torch.float16,
            'full': torch.float32
        }[config['language_model']['precision']],
    )

    return algorithm, config


def llama_2_single_kdma_adm(config):
    from align_system.algorithms.llama_2_single_kdma_adm import Llama2SingleKDMAADM

    algorithm = Llama2SingleKDMAADM(**config)
    algorithm.load_model()
    return algorithm, config


def llama_2_single_kdma_adm_with_rag(config):
    from align_system.algorithms.llama_2_single_kdma_adm import Llama2SingleKDMAADM
    from align_system.algorithms.llama_index_retriever import LlamaIndexRetrieverBackend

    init_keys = {'device', 'hf_model', 'precision', 'temperature'}
    filtered_config = {k: config[k] for k in init_keys}

    if config.get('retrieval', False):
        retriever_backend = LlamaIndexRetrieverBackend(
            config['retrieval.document_or_dir'],
            config.get('retrieval.model', 'tiiuae/falcon-7b-instruct'),
            config.get('retrieval.chunk_size', 128))

        retriever = retriever_backend.build_retriever(
            config.get('retrieval.num_results', 4))

        inference_config = {'retriever': retriever}

        # Default is to summarize
        if config.get('retrival.summarize', True):
            summarizer = retriever_backend.build_response_synthesizer(
                config.get('retrieval.response_mode', 'tree_summarize'))

            inference_config['summarizer'] = summarizer
    else:
        inference_config = {}

    algorithm = Llama2SingleKDMAADM(**filtered_config)
    algorithm.load_model()

    return algorithm, inference_config


def llama_index_adm(config):
    from align_system.algorithms.llama_index import LlamaIndex

    algorithm = LlamaIndex(**config)
    algorithm.load_model()
    return algorithm, config


def kaleido_adm(config):
    from align_system.algorithms.kaleido_adm import KaleidoADM
    # May want to seperate initialization config kwargs vs. inference
    # kwargs
    init_keys = {'model_name', 'use_tqdm'}
    filtered_config = {k: config[k] for k in init_keys}

    algorithm = KaleidoADM(**filtered_config)

    return algorithm, config


def dummy_adm(config):
    return DummyADM(**config), config


def oracle_adm(config):
    return OracleADM(**config), config


def pulse_tagging_adm(config):
    from align_system.algorithms.pulse_tagging_adm import PulseTaggingADM

    return PulseTaggingADM.load_model(
        device=config['device'],
        hf_model_name=config['model_name'],
        precision={
            'half': torch.float16,
            'full': torch.float32
        }[config['precision']]
    ), config


def multi_comparison_adm(config):
    from align_system.algorithms.multi_comparison_adm import MultiComparisonADM

    return MultiComparisonADM.load_model(**config['language_model']), config


eval_fns = [
    chat_kdma_predicting_adm,
    llama_2_single_kdma_adm,
    llama_2_single_kdma_adm_with_rag,
    llama_index_adm,
    dummy_adm,
    oracle_adm,
    pulse_tagging_adm,
    multi_comparison_adm,
    kaleido_adm,
]


def save_metrics(dataset, generated_outputs, target_kdma_values, results_dir):
    metrics = adm_evaluator.evaluate(dataset, generated_outputs, target_kdma_values)
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)


def main(config_file, cuda_idx=None):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    with open(config['dataset'], 'r') as f:
        dataset = json.load(f)
        dataset = dataset

    if 'cache_dir' in config:
        cache_dir = config['cache_dir']
        with open(os.path.join(cache_dir, 'input_output_labels.json')) as f:
            in_out_labels = json.load(f)

        generated_output = [
            in_out_label['output']
            for in_out_label in in_out_labels
        ]

        save_metrics(dataset, generated_output, config['target_kdma_values'], cache_dir)

    for eval_fn in eval_fns:
        experiment_name = config['name']

        if eval_fn.__name__ in config:
            if cuda_idx is not None:
                config[eval_fn.__name__]['device'] = f'cuda:{cuda_idx}'

            timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

            results_basename = f"{eval_fn.__name__}__{timestamp}"

            results_dir = os.path.join('results', experiment_name, results_basename)
            os.makedirs(results_dir, exist_ok=True)

            # Make a symlink to latest results for easier terminal access
            latest_results_dir = os.path.join('results', experiment_name, f"{eval_fn.__name__}__latest")

            if os.path.islink(latest_results_dir):
                os.remove(latest_results_dir)

            os.symlink(os.path.realpath(results_dir), latest_results_dir)

            with open(os.path.join(results_dir, os.path.basename(config_file)), 'w') as f:
                yaml.dump(config, f)

            eval_config = config[eval_fn.__name__]

            algorithm, inference_config = eval_fn(eval_config)

            with open(os.path.join(results_dir, 'log.txt'), 'w') as log_file:
                generated_outputs = adm_evaluator.generate_outputs(
                    dataset=dataset,
                    adm=algorithm,
                    target_kdma_values=config['target_kdma_values'],
                    log_file=log_file,
                    **inference_config,
                )

            in_out_labels = []
            for generated_output, (input_, label) in zip(generated_outputs, dataset):
                in_out_labels.append({
                    'input': input_,
                    'label': label,
                    'output': generated_output,
                })

            with open(os.path.join(results_dir, 'input_output_labels.json'), 'w') as f:
                json.dump(in_out_labels, f, indent=4)

            save_metrics(dataset, generated_outputs, config['target_kdma_values'], results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run ADM experiment")

    parser.add_argument('config_files',
                        type=str,
                        nargs='+',
                        help="Path to ADM configuration YAML file")
    parser.add_argument('--cuda-idx',
                        type=int,
                        help="CUDA device index (optional)")

    args = parser.parse_args()

    config_files = args.config_files
    cuda_idx = args.cuda_idx

    if cuda_idx is not None:
        print(f'Running on cuda:{cuda_idx}')

    failed_configs = []
    for config_file in config_files:
        try:
            main(config_file, cuda_idx)
        except Exception as e:
            print(f'Failed to run config {config_file}')
            traceback.print_exc()  # Print the full stack trace
            failed_configs.append(config_file)

    if len(failed_configs) > 0:
        # save failed configs to a file
        with open('failed_configs.txt', 'w') as f:
            f.write('\n'.join(failed_configs))
