import torch
import argparse
import dataset_utils
import models_utils
import utils
import run_mode_utils
import handlers
import clipping



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model-level CNN hardening")

    parser.add_argument("--model", type=str, required=True, help="name of the CNN")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "imagenet"], required=True, help="name of the dataset")
    parser.add_argument("--batch-size", type=int, default=128, help="an integer value for bach-size")
    parser.add_argument("--is-ranking", action='store_true', help="set the flag, if you want to rank the channels in conv layers")
    parser.add_argument("--is-pruning", action='store_true', help="set the flag, if you want to prune the CNN")
    parser.add_argument("--is-pruned", action='store_true', help="set the flag, if you want to load a pruned CNN")
    parser.add_argument("--pruning-method", type=str, default="hm", help="pruning method, homogeneous or heterogeneous")
    parser.add_argument("--pruning-ratio", type=float, default=None, help="a float value for pruning conv layers")
    parser.add_argument("--pruned-checkpoint", type=str, default=None, help="directory to the pruned model checkpoint")
    parser.add_argument("--is-hardening", action='store_true', help="set the flag, if you want to harden the CNN")
    parser.add_argument("--is-hardened", action='store_true', help="set the flag, if you want to load a hardened CNN")
    parser.add_argument("--hardening-ratio", type=float, default=None, help="a float value for hardening conv layers")
    parser.add_argument("--hardened-checkpoint", type=str, default=None, help="directory to the hardened model checkpoint")
    parser.add_argument("--importance", type=str, choices=["l1-norm", "vul-gain", "salience", "deepvigor", "channel-FI"], default=None, help="method for importance analysis either in pruning or hardening")
    parser.add_argument("--clipping", type=str, choices=["ranger"], default=None, help="method for clipping ReLU in hardening")
    parser.add_argument("--is-FI", action="store_true", help="set the flag, for performing fault simulation in weights")
    parser.add_argument("--BER", type=float, default=None, help="a float value for Bit Error Rate")
    parser.add_argument("--repeat", type=int, default=None, help="number of fault simulation experiments")


    # setting up the arguments values
    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    batch_size = args.batch_size
    is_ranking = args.is_ranking
    is_pruning = args.is_pruning
    is_pruned = args.is_pruned
    pruning_method = args.pruning_method
    pruning_ratio = args.pruning_ratio
    pruned_checkpoint = args.pruned_checkpoint
    is_hardening = args.is_hardening
    is_hardened = args.is_hardened
    hardening_ratio = args.hardening_ratio
    hardened_checkpoint = args.hardened_checkpoint
    importance_command = args.importance 
    clipping_command = args.clipping
    is_FI = args.is_FI
    BER = args.BER
    repetition_count = args.repeat


    #is_hardening and is_pruning should not be True at a same time
    assert not (is_hardening and is_pruning) == True

    # create log file
    run_mode = "test"
    run_mode += "".join([part for part, condition in [("_channel_ranking", is_ranking), ("_pruning", is_pruning), ("_hardening", is_hardening), ("_FI", is_FI)] if condition])
    setup_logger = handlers.LogHandler(run_mode, model_name, dataset_name)   
    logger = setup_logger.getLogger()
    setup_logger_info = ""
    setup_logger_info += "".join(f"{i}: {args.__dict__[i]}, " for i in args.__dict__ if \
                                    (type(args.__dict__[i]) is bool and args.__dict__[i] is True) or \
                                    (type(args.__dict__[i]) is not bool and args.__dict__[i] is not None))
    logger.info(f"args: {setup_logger_info}")

    # set the device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # load dataset and CNN model
    trainloader, classes_count, dummy_input = dataset_utils.load_dataset(dataset_name, batch_size, is_train=True)
    testloader, classes_count, dummy_input = dataset_utils.load_dataset(dataset_name, batch_size, is_train=False)
    model = models_utils.load_model(model_name, dataset_name, device)

    if is_pruned:
        assert pruning_ratio is not None
        assert pruned_checkpoint is not None
        pu = utils.prune_utils(model, pruning_method, classes_count, pruning_method)
        pu.set_pruning_ratios(pruning_ratio)
        model = pu.homogeneous_prune(model)
        models_utils.load_params(model, pruned_checkpoint, device)

    if is_hardened:
        assert hardening_ratio is not None
        assert hardened_checkpoint is not None
        assert clipping_command is not None

        clippingHandler = handlers.ClippingHandler(logger)
        clippingHandler.register("ranger", clipping.Ranger_thresholds)

        hr = utils.hardening_utils(hardening_ratio, clipping_command)
        hr.thresholds_extraction(model, clippingHandler, clipping_command, trainloader, device, logger)
        hardened_model = hr.relu_replacement(model)
        model = hr.conv_replacement(model)
        models_utils.load_params(model, hardened_checkpoint, device)


    dummy_input = dummy_input.to(device)
    model = model.to(device)

    runModeHandler = handlers.RunModeHandler(logger)
    runModeHandler.register("test", run_mode_utils.test_func)
    runModeHandler.register("test_pruning", run_mode_utils.pruning_func)
    runModeHandler.register("test_hardening", run_mode_utils.hardening_func)
    runModeHandler.register("test_FI", run_mode_utils.weights_FI_simulation)
    runModeHandler.register("test_channel_ranking", run_mode_utils.channel_ranking_func)

    if run_mode == "test":
        runModeHandler.execute(run_mode, model, testloader, device, dummy_input, logger)
    
    elif run_mode == "test_pruning":
        assert importance_command is not None
        assert pruning_ratio is not None
        runModeHandler.execute(run_mode, model, trainloader, testloader, classes_count, dummy_input, 
                               pruning_method, device, pruning_ratio, importance_command, logger)
    
    elif run_mode == "test_hardening":
        assert importance_command is not None
        assert hardening_ratio is not None
        assert clipping_command is not None
        runModeHandler.execute(run_mode, model, trainloader, testloader, dummy_input, classes_count, 
                               pruning_method, hardening_ratio, importance_command, clipping_command, device, logger)
    
    elif run_mode == "test_FI":
       assert BER is not None
       assert repetition_count is not None
       runModeHandler.execute(run_mode, model, testloader, repetition_count, BER, classes_count, device, logger)

    elif run_mode == "test_channel_ranking":
        assert importance_command is not None
        runModeHandler.execute(run_mode, model, trainloader, importance_command, classes_count, logger, device)

    #TODO: iterative pruning + refining

