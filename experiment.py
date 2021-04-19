from classifier import Classifier
import torch
import random
import logging
import data_utils
import did_dataset
import finetuning_utils
import torch.nn as nn
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import RandomSampler, DataLoader

from transformers import (WEIGHTS_NAME, AutoModel, AutoTokenizer, AdamW, AutoModelForSequenceClassification, AutoConfig,
                          get_linear_schedule_with_warmup)

# manual seed random number generator
torch.manual_seed(999)
random.seed(999)


def run_train(params):
    """
    Run training on the model and get best validation model
    Args:
      params: all parameters to run the experiment
    Returns:
      results: dictionary that contains all data
    """
    # Initialize variables
    print('Initializing variables')
    bert = AutoModel.from_pretrained(params['bert'])
    tokenizer = AutoTokenizer.from_pretrained(params['tokenizer'])
    config = AutoConfig.from_pretrained(
        params['bert'],
        num_labels=params['num_classes'],
        label2id=params['label2id'],
        id2label=params['id2label'],
        finetuning_task='arabic_did'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        params['tokenizer']
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        params['bert'],
        config=config
    )
    seed = params['seed']
    level = params['level']
    train_batch_size = params['train_batch_size']
    val_batch_size = params['val_batch_size']
    max_seq_length = params['max_seq_length']
    epochs = params['epochs']
    learning_rate = params['learning_rate']
    adam_epsilon = params['adam_epsilon']
    early_stop_patience = params['early_stop_patience']
    metric = params['metric']
    train_files = params['train_files']
    val_files = params['val_files']
    label_space_file = params['label_space_file']
    model_folder = params['model_folder']
    weight_decay = params['weight_decay']
    warmup_steps = params['warmup_steps']
    gradient_accumulation_steps = params['gradient_accumulation_steps']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get data
    print('Getting Data')
    train_df = data_utils.get_df_from_files(train_files)
    val_df = data_utils.get_df_from_files(val_files)

    train_data = did_dataset.DIDDataset(
        train_df, tokenizer, level, label_space_file, max_seq_length)
    val_data = did_dataset.DIDDataset(
        val_df, tokenizer, level, label_space_file, max_seq_length)

    train_sampler = RandomSampler(train_data)
    val_sampler = RandomSampler(val_data)

    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=train_batch_size, num_workers=0)
    val_dataloader = DataLoader(
        val_data, sampler=val_sampler, batch_size=val_batch_size, num_workers=0)

    t_total = (len(train_dataloader) // gradient_accumulation_steps*epochs)
    # define the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=learning_rate,
                      eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained,
                            epochs, desc='Epoch')
    finetuning_utils.set_seed(seed)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)

            outputs = model(**batch)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_grad_norm)

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
    """
    cross_entropy = nn.CrossEntropyLoss()

    best_valid_metric = 0
    best_valid_epoch = 0
    if metric == 'loss':
        best_valid_metric = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_metrics_list = []
    valid_metrics_list = []
    train_predictions_list = []
    valid_predictions_list = []
    n_no_improvement = 0
    early_stop = False
    model_file = ''
    print('Starting training')

    # for each epoch
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        # train model
        train_predictions, train_metrics = finetuning_utils.tr
        
        ain(model, train_dataloader,
                                                                  cross_entropy, optimizer, device)

        # evaluate model
        valid_predictions, valid_metrics = finetuning_utils.evaluate(model, val_dataloader,
                                                                     cross_entropy, device)
        # get best metric
        if metric == 'loss' and valid_metrics[metric] < best_valid_metric:
            best_valid_epoch = epoch
            best_valid_metric = valid_metrics[metric]
            model_file = f'{model_folder}/{epoch}.pt'
            torch.save(model.state_dict(), model_file)
            n_no_improvement = 0

        elif metric != 'loss' and valid_metrics[metric] > best_valid_metric:
            best_valid_epoch = epoch
            best_valid_metric = valid_metrics[metric]
            model_file = f'{model_folder}/{epoch}.pt'
            torch.save(model.state_dict(), model_file)
            n_no_improvement = 0
        else:
            n_no_improvement += 1

        train_metrics_list.append(train_metrics)
        valid_metrics_list.append(valid_metrics)
        train_predictions_list.append(train_predictions)
        valid_predictions_list.append(valid_predictions)

        print(
            f'Training Metrics: {train_metrics}')
        print(
            f'Validation Metrics: {valid_metrics}')
        if n_no_improvement > early_stop_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            early_stop = True
            break

    # logging.info best metrics
    print(
        f'Best validation epoch is {best_valid_epoch + 1} with metrics: {valid_metrics_list[epoch]}\nModel can be found here saved_weights_early_stop_{best_valid_epoch}_{metric}.pt')

    results = {}
    results['best_valid_epoch'] = best_valid_epoch
    results['train_metrics_list'] = train_metrics_list
    results['valid_metrics_list'] = valid_metrics_list
    results['best_valid_epoch'] = best_valid_epoch
    results['best_valid_metric'] = valid_metrics_list[best_valid_epoch]
    results['early_stop'] = early_stop
    results['model_file'] = model_file
    return results
    """


def run_test(params):
    """
    Run test on the model
    Args:
      params: all parameters to run the experiment
    Returns:
      results: dictionary that contains all data
      test_predictions: list with test_predictions
    """
    bert = AutoModel.from_pretrained(params['bert'])
    tokenizer = AutoTokenizer.from_pretrained(params['tokenizer'])
    level = params['level']
    test_batch_size = params['test_batch_size']
    max_seq_length = params['max_seq_length']
    dropout_prob = params['dropout_prob']
    hidden_size = bert.config.hidden_size
    num_classes = params['num_classes']
    test_files = params['test_files']
    label_space_file = params['label_space_file']
    model_file = params['model_file']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Classifier(bert, dropout_prob, hidden_size, num_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_file))

    # Get Data
    print('Getting Test Data')
    test_df = data_utils.get_df_from_files(test_files)

    test_data = did_dataset.DIDDataset(
        test_df, tokenizer, level, label_space_file, max_seq_length)

    test_dataloader = DataLoader(
        test_data, batch_size=test_batch_size, num_workers=0)

    # Get cross entropy
    cross_entropy = nn.CrossEntropyLoss()
    print('Starting Test')

    test_predictions, test_metrics = finetuning_utils.test(model, test_dataloader,
                                                           cross_entropy, device)
    print(f'Test Metrics: {test_metrics}')
    results = {}
    for k in test_metrics.keys():
        results[f'test_{k}'] = test_metrics[k]

    return results, test_predictions
