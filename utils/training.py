import torch
import torch.nn.functional as F

from .timer import SynchronizedWallClockTimer


def interpolate(img, shape):
    return F.interpolate(img, size=shape, mode='bilinear', align_corners=True)


def mean(v):
    return sum(v) / len(v)


def combined_loss(evaluator,
                  flows,
                  image1,
                  image2,
                  features,
                  weights=[0.5, 1, 1]):
    arths = (features[f'dec_flow_arth_{i}'] for i in range(len(flows)))
    terms = evaluator(flows, image1, image2, arths)
    loss = sum(map(lambda v, w: w*mean(v), terms, weights))
    return loss, terms


def make_hook_periodic(hook, checkpointing_interval):
    return lambda step, *args: (None
                                if step % checkpointing_interval
                                else hook(step, *args))


def predictions2tag(predictions):
    return (f'{x.shape[-2]}x{x.shape[-1]}' for x in predictions)


def train(model,
          device,
          train_loader,
          generated_train_loader,
          optimizer,
          num_steps: int,
          scheduler,
          logger,
          evaluator,
          weights=[0.5, 1, 1],
          is_raw=True,
          accumulation_steps=1,
          timers=SynchronizedWallClockTimer(),
          hooks={},
          init_step=0,
          init_samples_passed=0):
    ''' Performs training

    Args:
        model (nn.Module): a model to train
        device (torch.device): a device used for training
        train_loader (torch.utils.data.DataLoader): a training data loader
        generated_train_loader (torch.utils.data.DataLoader): a training data loader
        optimzer (torch.optim.optimizer.Optimizer): a used optimizer
        num_steps (int): number of training steps
        scheduler (torch.optimi.lr_scheduler): scheduler that updates lr
        logger (class): a class to store logs
        evaluator (class): a class to compute loss
        weights (float, float, float): weights of the loss functions
        is_raw (bool): does use raw event stream
        accumulation_steps (int): gradient accumulation steps
        hooks (dict): hooks that should be called after each step of optimizer.
                      Each hook is a Callable(steps, samples_passed)->None
    '''

    model.train()

    samples_passed = init_samples_passed
    loss_sum = 0
    smooth_sum = []
    photo_sum = []
    out_reg_sum = []
    optimizer.zero_grad()
    timers('batch_construction').start()
    for global_step, loaded_data in enumerate(zip(train_loader, train_loader), init_step * accumulation_steps):
        timers('batch_construction').stop()
        if global_step == 2 * num_steps * accumulation_steps:
            break

        for is_generated, (events, start, stop, image1, image2) in enumerate(loaded_data):
            samples_passed += start.numel()
            events, start, stop, image1, image2 = send_data_on_device(device, events, start, stop, image1, image2, timers)
            prediction, features, tags = forward_pass(model, events, start, stop, image1, image2, timers, is_raw)
            loss, terms = compute_losses(evaluator, prediction, accumulation_steps, image1, image2, features, weights, timers)
            backward_prop(loss, timers)

            loss_sum += loss.item()
            photo_sum, smooth_sum, out_reg_sum = update_losses(loss, terms, photo_sum, smooth_sum, out_reg_sum, hooks, timers)

            if (global_step + 1) % accumulation_steps == 0:
                do_optimization_step(optimizer, scheduler, timers)
                true_step = (global_step + 1) // accumulation_steps
                dump_losses(loss_sum, smooth_sum, photo_sum, out_reg_sum, samples_passed, tags, optimizer, scheduler, logger, timers)
                run_hooks(model, hooks, true_step, samples_passed, timers)
                loss_sum = 0
                smooth_sum = []
                photo_sum = []
                out_reg_sum = []


        timers.log(names=['batch_construction',
                          'batch2gpu',
                          'forward',
                          'loss',
                          'grid_construction',
                          'photometric_loss',
                          'smoothness_loss',
                          'outborder_loss',
                          'backprop',
                          'optimizer_step',
                          'free',
                          'logging'] + list(hooks))
        timers('batch_construction').start()
    timers('batch_construction').stop()


def send_data_on_device(device, events, start, stop, image1, image2, timers):
    timers('batch2gpu').start()
    events, start, stop, image1, image2 = map(lambda x: x.to(device),
                                            (events,
                                             start,
                                             stop,
                                             image1,
                                             image2))
    timers('batch2gpu').stop()
    return events, start, stop, image1, image2

def forward_pass(model, events, start, stop, image1, image2, timers, is_raw):
        shape = image1.size()[-2:]
        timers('forward').start()
        prediction, features = model(events,
                                     start,
                                     stop,
                                     shape,
                                     raw=is_raw,
                                     intermediate=True)
        tags = predictions2tag(prediction)
        timers('forward').stop()
        return prediction, features, tags


def compute_losses(evaluator, prediction, accumulation_steps, image1, image2, features, weights, timers):
    timers('loss').start()
    loss, terms = combined_loss(evaluator,
                                prediction,
                                image1,
                                image2,
                                features,
                                weights=weights)
    loss = loss / accumulation_steps
    terms = [[term_part / accumulation_steps for term_part in term] for term in terms]
    timers('loss').stop()
    return loss, terms


def backward_prop(loss, timers):
    timers('backprop').start()
    loss.backward()
    timers('backprop').stop()


def do_optimization_step(optimizer, scheduler, timers):
    timers('optimizer_step').start()
    optimizer.step()
    optimizer.zero_grad()
    timers('optimizer_step').stop()
    scheduler.step()


def dump_losses(loss_sum, smooth_sum, photo_sum, out_reg_sum, samples_passed, tags, optimizer, scheduler, logger, timers):
    timers('logging').start()
    for tag, s, p, o in zip(tags, smooth_sum, photo_sum, out_reg_sum):
        logger.add_scalar(f'Train/photometric loss/{tag}',
                          p,
                          samples_passed)
        logger.add_scalar(f'Train/smoothness loss/{tag}',
                          s,
                          samples_passed)
        logger.add_scalar(f'Train/out regularization/{tag}',
                          o,
                          samples_passed)
    logger.add_scalar('General/Train loss',
                      loss_sum,
                      samples_passed)

    for i, lr in enumerate([p['lr']
                            for p in optimizer.param_groups]):
        logger.add_scalar(f'General/learning rate/{i}',
                          lr,
                          samples_passed)

    timers('logging').stop()


def run_hooks(model, hooks, true_step, samples_passed, timers):
    for k, hook in hooks.items():
        timers(k).start()
        hook(true_step, samples_passed)
        timers(k).stop()
    # make sure to return to train after all hooks
    model.train()


def update_losses(loss, terms, photo_sum, smooth_sum, out_reg_sum, hooks, timers):
    timers('optimizer_step').start()
    timers('optimizer_step').stop()
    for k, hook in hooks.items():
        timers(k).start()
        timers(k).stop()
    # losses for logging
    timers('logging').start()
    smoothness, photometric, out_reg = terms

    photo_sum = add_loss(photo_sum, photometric)
    smooth_sum = add_loss(smooth_sum, smoothness)
    out_reg_sum = add_loss(out_reg_sum, out_reg)
    timers('logging').stop()
    return photo_sum, smooth_sum, out_reg_sum


def add_loss(loss_sum, loss_values):
    if len(loss_sum) == 0:
        return [x.item() if type(x) != int else x for x in loss_values]
    return [x + (y.item() if type(y) != int else y) for x, y in zip(loss_sum, loss_values)]


def validate(model, device, loader, samples_passed,
             logger, evaluator, weights=[0.5, 1, 1], is_raw=True):
    model.eval()
    n = len(loader)
    photo_sum = []
    smooth_sum = []
    out_reg_sum = []
    loss_sum = 0
    with torch.no_grad():
        for data, start, stop, image1, image2 in loader:
            data, start, stop, image1, image2 = map(lambda x: x.to(device),
                                                    (data,
                                                     start,
                                                     stop,
                                                     image1,
                                                     image2))
            shape = image1.size()[-2:]
            prediction, features = model(data,
                                         start,
                                         stop,
                                         shape,
                                         raw=is_raw,
                                         intermediate=True)
            tags = predictions2tag(prediction)
            loss, terms = combined_loss(evaluator,
                                        prediction,
                                        image1,
                                        image2,
                                        features,
                                        weights=weights)
            smoothness, photometric, out_reg = terms
            photo_sum = add_loss(photo_sum, photometric)
            smooth_sum = add_loss(smooth_sum, smoothness)
            out_reg_sum = add_loss(out_reg_sum, out_reg)
            loss_sum += loss.item()
    logger.add_scalar('General/Validation loss', loss_sum/n, samples_passed)
    for tag, s, p, o in zip(tags, smooth_sum, photo_sum, out_reg_sum):
        logger.add_scalar(f'Validation/smoothness loss/{tag}',
                          s/n,
                          samples_passed)
        logger.add_scalar(f'Validation/photometric loss/{tag}',
                          p/n,
                          samples_passed)
        logger.add_scalar(f'Validation/out regularization loss/{tag}',
                          o/n,
                          samples_passed)
