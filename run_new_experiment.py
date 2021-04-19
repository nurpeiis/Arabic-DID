# -*- coding: utf-8 -*-

# MIT License
#
# Copyright 2018-2021 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Fine-tuning code for text classification tasks
Heavily adapted from: https://github.com/huggingface/transformers/blob/
                      v2.5.1/examples/run_glue.py
"""

import argparse
import glob
import json
import logging
import os
import random
import data_utils

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

from utils.metrics import compute_metrics, write_predictions
from utils.data_utils import output_modes
from utils.data_utils import processors
from utils.data_utils import convert_examples_to_features
from data_new_utils import DIDProcesser
from finetuning_utils import get_labels, set_seed

logger = logging.getLogger(__name__)


def train(args, train_dataset, model, tokenizer):
    ''' Train the model '''

    args.train_batch_size = args.per_device_train_batch_size * \
        max(1, args.n_gpu)
    train_sampler = (RandomSampler(train_dataset) if args.local_rank == -1
                     else DistributedSampler(train_dataset))
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (args.max_steps //
                                 (len(train_dataloader) //
                                  args.gradient_accumulation_steps) + 1)
    else:
        t_total = (len(train_dataloader) //
                   args.gradient_accumulation_steps *
                   args.num_train_epochs)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if (os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt'))
       and os.path.isfile(os.path.join(args.model_name_or_path, 'scheduler.pt')
                          )):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(
                                  os.path.join(args.model_name_or_path,
                                               'optimizer.pt')))
        scheduler.load_state_dict(torch.load(
                                  os.path.join(args.model_name_or_path,
                                               'scheduler.pt')))

    # Train!
    logger.info('***** Running training *****')
    logger.info('  Num examples = %d', len(train_dataset))
    logger.info('  Num Epochs = %d', args.num_train_epochs)
    logger.info('  Instantaneous batch size per GPU = %d',
                args.per_device_train_batch_size)
    logger.info(
        '  Total train batch size '
        '(w. parallel, distributed & accumulation) = %d',
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info('  Gradient Accumulation steps = %d',
                args.gradient_accumulation_steps)
    logger.info('  Total optimization steps = %d', t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint
        # from model path
        try:
            global_step = int(
                args.model_name_or_path.split('-')[-1].split('/')[0])
        except ValueError:
            global_step = 0
        epochs_trained = (global_step //
                          (len(train_dataloader) //
                           args.gradient_accumulation_steps))
        steps_trained_in_current_epoch = (global_step %
                                          (len(train_dataloader) //
                                           args.gradient_accumulation_steps))

        logger.info('  Continuing training from checkpoint, '
                    'will skip to saved global_step')
        logger.info('  Continuing training from epoch %d', epochs_trained)
        logger.info('  Continuing training from global step %d', global_step)
        logger.info('  Will skip the first %d steps in the first epoch',
                    steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained,
                            int(args.num_train_epochs), desc='Epoch',
                            disable=args.local_rank not in [-1, 0],)
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc='Iteration',
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                      'labels': batch[3]}
            # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = (
                    batch[2] if args.model_type in ['bert', 'xlnet', 'albert']
                    else None)
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel training
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (args.local_rank in [-1, 0] and args.logging_steps > 0 and
                        global_step % args.logging_steps == 0):
                    logs = {}
                    # Only evaluate when single GPU otherwise metrics may not
                    # average well
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = 'eval_{}'.format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    print(json.dumps({**logs, **{'step': global_step}}))

                if (args.local_rank in [-1, 0] and args.save_steps > 0 and
                        global_step % args.save_steps == 0):
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir,
                                              'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, 'module') else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir,
                               'training_args.bin'))
                    logger.info('Saving model checkpoint to %s', output_dir)

                    torch.save(optimizer.state_dict(),
                               os.path.join(output_dir, 'optimizer.pt'))
                    torch.save(scheduler.state_dict(),
                               os.path.join(output_dir, 'scheduler.pt'))
                    logger.info('Saving optimizer and scheduler states to %s',
                                output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, mode='', prefix=''):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (('mnli', 'mnli-mm') if args.task_name == 'mnli'
                       else (args.task_name,))
    eval_outputs_dirs = ((args.output_dir, args.output_dir + '-MM')
                         if args.task_name == 'mnli' else (args.output_dir,))

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task,
                                               tokenizer, mode)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_device_eval_batch_size * \
            max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        # Eval!
        logger.info('***** Running evaluation {} *****'.format(prefix))
        logger.info('  Num examples = %d', len(eval_dataset))
        logger.info('  Batch size = %d', args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader,  desc='Evaluating'):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                          'labels': batch[3]}
                # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use
                # segment_ids
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = (
                        batch[2] if args.model_type in ['bert', 'xlnet', 'albert']
                        else None
                    )
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids,
                    inputs['labels'].detach().cpu().numpy(),
                    axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == 'classification':
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == 'regression':
            preds = np.squeeze(preds)

        if args.write_preds:
            output_path_file = os.path.join(eval_output_dir, prefix,
                                            'predictions.txt')
            logger.info('***** Writing Predictions to '
                        '{} *****'.format(output_path_file))
            write_predictions(output_path_file, eval_task, preds)

        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix,
                                        'eval_results.txt')
        with open(output_eval_file, 'w') as writer:
            logger.info('***** Eval results {} *****'.format(prefix))
            for key in sorted(result.keys()):
                logger.info('  %s = %s', key, str(result[key]))
                writer.write('%s = %s\n' % (key, str(result[key])))

    return results


def load_and_cache_examples(args, files, label_list, level, tokenizer, mode):

    processor = DIDProcesser()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}'.format(
            mode,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info('Loading features from cached file %s',
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info('Creating features from dataset file at %s', args.data_dir)

        examples = processor.get_examples(files, level, mode)

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            pad_on_left=bool(args.model_type in ['xlnet']),
            pad_token=tokenizer.convert_tokens_to_ids(
                [tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        if args.local_rank in [-1, 0]:
            logger.info('Saving features into cached file %s',
                        cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features],
                                      dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features],
                                      dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features],
                              dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        '--data_dir',
        default=None,
        type=str,
        required=True,
        help='The input data dir. Should contain the .tsv files '
             '(or other data files) for the task.',
    )
    parser.add_argument(
        '--model_type',
        default=None,
        type=str,
        required=True,
        help='Model type selected in the list: ',
    )
    parser.add_argument(
        '--model_name_or_path',
        default='CAMeL-Lab/bert-base-camelbert-mix',
        type=str,
        required=True,
        help='Path to pretrained model or model identifier '
             'from huggingface.co/models',
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        type=str,
        required=True,
        help='The output directory where the model predictions and '
        'checkpoints will be written.',
    )

    # Other parameters
    parser.add_argument(
        '--config_name',
        default='CAMeL-Lab/bert-base-camelbert-mix',
        type=str,
        help='Pretrained config name or path if not the same as model_name',
    )
    parser.add_argument(
        '--tokenizer_name',
        default='CAMeL-Lab/bert-base-camelbert-mix',
        type=str,
        help='Pretrained tokenizer name or path if not the same as model_name',
    )
    parser.add_argument(
        '--cache_dir',
        default='',
        type=str,
        help='Where do you want to store the pre-trained models '
             'downloaded from s3',
    )
    parser.add_argument(
        '--label_space_file',
        default='',
        type=str,
        help='Label space file',
    )
    parser.add_argument(
        '--level',
        default='',
        type=str,
        help='Level of dialect: city, country or region',
    )
    parser.add_argument(
        '--max_seq_length',
        default=128,
        type=int,
        help='The maximum total input sequence length after tokenization. '
             'Sequences longer than this will be truncated, sequences shorter '
             'will be padded.',
    )
    parser.add_argument(
        '--do_train',
        action='store_true',
        help='Whether to run training.'
    )
    parser.add_argument(
        '--do_eval',
        action='store_true',
        help='Whether to run eval on the dev set.'
    )
    parser.add_argument(
        '--do_pred',
        action='store_true',
        help='Whether to run eval on the test set.'
    )
    parser.add_argument(
        '--evaluate_during_training',
        action='store_true',
        help='Run evaluation during training at each logging step.',
    )
    parser.add_argument(
        '--per_device_train_batch_size',
        default=8,
        type=int,
        help='Batch size per GPU/CPU for training.',
    )
    parser.add_argument(
        '--per_device_eval_batch_size',
        default=8, type=int,
        help='Batch size per GPU/CPU for evaluation.',
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of updates steps to accumulate '
             'before performing a backward/update pass.',
    )
    parser.add_argument(
        '--learning_rate',
        default=3e-5,
        type=float,
        help='The initial learning rate for Adam.'
    )
    parser.add_argument(
        '--weight_decay',
        default=0.0,
        type=float,
        help='Weight decay if we apply some.'
    )
    parser.add_argument(
        '--adam_epsilon',
        default=1e-8,
        type=float,
        help='Epsilon for Adam optimizer.'
    )
    parser.add_argument(
        '--max_grad_norm',
        default=1.0,
        type=float,
        help='Max gradient norm.'
    )
    parser.add_argument(
        '--num_train_epochs',
        default=10,
        type=float,
        help='Total number of training epochs to perform.',
    )
    parser.add_argument(
        '--max_steps',
        default=-1,
        type=int,
        help='If > 0: set total number of training steps to perform. '
             'Override num_train_epochs.',
    )
    parser.add_argument(
        '--warmup_steps',
        default=0,
        type=int,
        help='Linear warmup over warmup_steps.'
    )
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=500,
        help='Log every X updates steps.'
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=500,
        help='Save checkpoint every X updates steps.'
    )
    parser.add_argument(
        '--eval_all_checkpoints',
        action='store_true',
        help='Evaluate all checkpoints starting with the same prefix as '
             'model_name ending and ending with step number',
    )
    parser.add_argument(
        '--write_preds',
        action='store_true',
        help='Write predictions to a file'
    )
    parser.add_argument(
        '--overwrite_output_dir',
        action='store_true',
        help='Overwrite the content of the output directory',
    )
    parser.add_argument(
        '--overwrite_cache',
        action='store_true',
        help='Overwrite the cached training and evaluation sets',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help='random seed for initialization')

    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            'Output directory ({}) already exists and is not empty. '
            'Use --overwrite_output_dir to overcome.'.format(
                args.output_dir
            )
        )

    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.no_cuda else 'cpu')

    args.device = device

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    # Set seed
    set_seed(args)
    label_list, label2id, id2label = get_labels(args.label_space_file)
    args.train_files = []
    args.dev_files = []
    args.test_files = []
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        finetuning_task='arabic_did',
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model.to(args.device)

    logger.info('Training/evaluation parameters %s', args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.train_files, label_list,
                                                tokenizer, mode='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(' global_step = %s, average loss = %s',
                    global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model,
    # you can reload it using from_pretrained()
    if (args.do_train
            and (args.local_rank == -1 or torch.distributed.get_rank() == 0)):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info('Saving model checkpoint to %s', args.output_dir)
        # Save a trained model, configuration and tokenizer
        # using `save_pretrained()`. They can then be reloaded using
        # `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, 'module') else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments
        # together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForSequenceClassification.from_pretrained(
            args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    best_f1_eval = 0
    best_model_checkpoint = args.output_dir
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME,
                                recursive=True))
            )
            # Reduce logging
            logging.getLogger(
                'transformers.modeling_utils').setLevel(logging.WARN)
        logger.info('Evaluate the following checkpoints: %s', checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                '-')[-1] if len(checkpoints) > 1 else ''
            prefix = checkpoint.split(
                '/')[-1] if checkpoint.find('checkpoint') != -1 else ''

            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer,
                              mode='dev', prefix=prefix)
            # getting the best model checkpoint
            if result['f1'] > best_f1_eval:
                best_f1_eval = result['f1']
                best_model_checkpoint = checkpoint
            result = dict((k + '_{}'.format(global_step), v)
                          for k, v in result.items())
            results.update(result)
        # renaming the best model checkpoint folder
        # os.rename(best_model_checkpoint, best_model_checkpoint + '-best')

    if args.do_pred and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.output_dir
        )
        model.to(args.device)
        result = evaluate(args, model, tokenizer, mode='test')
        result = dict((k, v) for k, v in result.items())
        results.update(result)

    return results


if __name__ == '__main__':
    main()
