import logging

import torch
from .detect import detect
from tqdm import tqdm
from metric import roc_auc
import numpy as np

logger = logging.getLogger('separation_training')
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def train_model(
        x,
        model,
        criterion,
        cri_kwargs,
        epochs,
        optimizer,
        verbose,
        adaptive_lambda=None,
        epoch_offset=0,
        callback=None):
    """
    Train `model` for `epochs` steps on the fixed input `x`.

    Phase 2 change
    --------------
    An optional `adaptive_lambda` controller (see `adaptive_lambda.py`) can
    be supplied. When given, after every `loss.backward()` call (so
    gradients are populated, which the 'gradient_norm' strategy needs) it
    computes a new lambda value from the tensors already produced this
    epoch (`x`, `y`, `mask`, `model`) - no extra forward pass - and pushes
    it into `criterion` via `criterion.set_lmda(...)` before
    `optimizer.step()` is called. If `adaptive_lambda` is None (the
    default), this function is byte-for-byte equivalent to the Phase 1
    implementation.

    Returns
    -------
    loss_history : list[float]
        The total loss value recorded at every epoch of this call, used
        by `separation_training` to build the 'reconstruction loss vs
        epochs' plot.
    """

    loss_history = []

    epoch_iter = iter(_ for _ in range(epochs))
    if verbose:
        epoch_iter = tqdm(list(epoch_iter))
    for e in epoch_iter:

        # Clear gradient information
        optimizer.zero_grad()

        # Forward propagation
        y = model(x)

        # Calculate loss
        loss = criterion(x=x, y=y, **cri_kwargs)

        # Backward propagation
        loss.backward()

        # --- Phase 2: adaptive lambda update -----------------------
        # Runs after backward() (gradients are populated) and before
        # optimizer.step(), so the *next* forward/backward pass uses the
        # updated lambda. Skipped entirely when adaptive_lambda is None,
        # which reproduces the original fixed-lambda behaviour exactly.
        if adaptive_lambda is not None:
            # `y` here is `decoder_outputs`, the full tuple of multiscale
            # decoder heads MSNet returns. SeTLoss internally uses only
            # the finest-scale output (decoder_outputs[0]) for the
            # residual/entropy computations that adaptive_lambda mirrors -
            # pass that same tensor, not the raw tuple.
            new_lmda = adaptive_lambda.step(
                epoch=epoch_offset + e,
                x=x,
                y=y[0],
                mask=cri_kwargs.get('mask'),
                model=model,
            )
            criterion.set_lmda(new_lmda)
        # --------------------------------------------------------------

        # Update network parameters
        optimizer.step()

        loss_val = float(loss.detach().cpu())
        loss_history.append(loss_val)

        if verbose:
            postfix = {'loss': '{0:.4f}'.format(loss)}
            if adaptive_lambda is not None:
                postfix['lmda'] = '{0:.2e}'.format(adaptive_lambda.get())
            epoch_iter.set_postfix(postfix)

        # Optional hook for live progress reporting (e.g. the Streamlit
        # app). Not used anywhere in the CLI path, so it adds no overhead
        # there beyond a single `is None` check per epoch.
        if callback is not None:
            current_lmda = adaptive_lambda.get() if adaptive_lambda is not None else criterion.get_lmda()
            callback(epoch_offset + e, loss_val, current_lmda)

    return loss_history


def separation_training(
        x: torch.Tensor,
        gt: np.ndarray,
        model,
        loss,
        mask,
        optimizer,
        epochs,
        output_iter,
        max_iter,
        verbose,
        adaptive_lambda=None,
        callback=None,
        iter_callback=None,
        ig_steps=20,
        early_stop_patience=None,
        early_stop_tol=None):
    """
    The main process of the separation training algorithm.

    Phase 2 change
    ---------------
    Accepts an optional `adaptive_lambda` controller (an
    `adaptive_lambda.AdaptiveLambda` instance, or None). When supplied,
    it is threaded through to `train_model` every outer iteration so
    lambda is recomputed every epoch. Three additional histories are
    collected purely for logging/plotting and do not influence the
    detection result in any way:

        * loss_history    - total loss at every epoch, whole run.
        * lambda_history  - lambda value at every epoch, whole run
                             (empty list if adaptive_lambda is None).
        * auc_per_epoch   - the outer-iteration AUC broadcast across
                             every epoch of that iteration, so it can be
                             plotted on the same x-axis as loss/lambda.

    Backward compatibility
    -----------------------
    When `adaptive_lambda=None` (the default), the returned `output_dm`
    and `history` (per-iteration AUC list) are numerically identical to
    Phase 1 - the extra `extras` dict is purely additional information,
    it never changes control flow, RNG usage, or numerical results.

    `callback(epoch, loss_val, lmda_val)` and `iter_callback(iter_idx, auc)`
    are optional hooks (both default to `None`, zero overhead when unused)
    consumed by the Streamlit app in `streamlit_app/` for live plots; they
    do not affect training in any way.

    Returns
    -------
    (output_dm, history, extras) : (np.ndarray, list, dict)
    """

    history = []
    loss_history = []
    lambda_history = []
    auc_per_epoch = []

    output_dm = np.zeros_like(gt)

    for i in range(1, max_iter + 1):
        if verbose:
            print('Iter {0}'.format(i))

        # Feed the model with x
        model_input = x

        # Train the model for some epochs
        epoch_offset = (i - 1) * epochs
        _hist_start = len(adaptive_lambda.history) if adaptive_lambda is not None else 0
        iter_loss_history = train_model(
            model_input,
            model,
            loss,
            {'mask': mask},
            epochs,
            optimizer,
            verbose,
            adaptive_lambda=adaptive_lambda,
            epoch_offset=epoch_offset,
            callback=callback,
        )
        loss_history.extend(iter_loss_history)
        _hist_end = len(adaptive_lambda.history) if adaptive_lambda is not None else 0

        # Update the mask using detection map obtained in this iteration
        dm = detect(x, model(model_input))
        mask.update(dm.detach())

        # --- Phase 2: XAI-guided lambda recalibration at mask boundary --
        # This is the one point in the loop where the anomaly/background
        # assignment discontinuously changes, which is where the naive
        # residual-based lambda strategy was observed to spike. For
        # strategy='xai_alignment' this runs Integrated Gradients against
        # the just-updated mask and recalibrates lambda_base for the next
        # block of epochs; for every other strategy (including None) this
        # is a no-op, so existing behaviour is unaffected.
        if adaptive_lambda is not None and i < max_iter:
            # Skip on the final iteration: there is no next block of
            # epochs left for a recalibrated lambda to govern, so paying
            # for another ~ig_steps forward/backward passes here would be
            # pure waste - it changes nothing about output_dm or history.
            adaptive_lambda.boundary_update(iteration=i, x=x, model=model, mask=mask, ig_steps=ig_steps)
            if getattr(adaptive_lambda, 'layer_gates', None) is not None:
                # Governs the decoder-side skip-connection accumulation
                # (MSNet.forward) for the NEXT block of epochs, same
                # timing as the lambda recalibration above - both derive
                # from the same IG pass just computed in boundary_update.
                model.layer_gates = adaptive_lambda.layer_gates
        # ------------------------------------------------------------------

        # Evaluation
        np_dm = dm.cpu().detach().numpy()
        fpr, tpr, auc = roc_auc(np_dm, gt)
        if verbose:
            print('Current AUC score: {0:.4f}'.format(auc))
        logger.info('iter=%d auc=%.4f', i, auc)

        # Record history
        history.append(auc)
        auc_per_epoch.extend([auc] * epochs)

        if iter_callback is not None:
            iter_callback(i, auc)

        if adaptive_lambda is not None:
            # Bug fix (found during final audit): this used to slice
            # lambda_history() with a computed [epoch_offset:epoch_offset
            # +epochs] window, which silently assumed exactly `epochs`
            # entries get appended to self.history per outer iteration.
            # That's true for every strategy except 'xai_alignment',
            # whose boundary_update() appends one extra entry per
            # iteration (the recalibration event itself) - which
            # progressively desynced the offset arithmetic, worse each
            # iteration (verified: 0 wrong entries at iter 1, growing to
            # 4 wrong/misplaced entries by iter 5 in a 5-iteration run).
            # Using the actual before/after list length instead of a
            # computed offset is correct regardless of how many entries
            # any given strategy's boundary_update() adds - it can never
            # desync, because it isn't assuming a count, it's measuring one.
            lambda_history.extend(adaptive_lambda.lambda_history()[_hist_start:_hist_end])

        # Record the output detection map of the algorithm
        if i == output_iter:
            output_dm = np_dm

        # --- Optimization: early stopping on AUC convergence -----------
        # Opt-in only (both args default to None => byte-for-byte Phase 1
        # behaviour, runs the full max_iter every time, same as before).
        # This is a training-procedure optimization, not a model change -
        # it never touches MSNet, the loss, or the mask logic, only when
        # the outer loop decides to stop early. Only meaningful once the
        # designated output_iter has already been captured above, so a
        # premature stop can never leave output_dm as the initial
        # np.zeros_like(gt) placeholder.
        # v14 WARNING: `history` holds AUC computed against `gt`. Stopping on
        # it is model selection using ground-truth labels - i.e. leakage - even
        # though it only shortens the run. It is off by default and MUST stay
        # off for any number that goes in the paper. Enabling it is logged
        # loudly so it cannot happen silently.
        if (early_stop_patience is not None and early_stop_tol is not None
                and i >= output_iter and len(history) >= early_stop_patience):
            logger.warning('EARLY STOPPING ON GROUND-TRUTH AUC IS ENABLED. '
                           'Results from this run are NOT leakage-free and '
                           'must not be reported.')
            recent = history[-early_stop_patience:]
            if max(recent) - min(recent) < early_stop_tol:
                if verbose:
                    print('Early stopping at iter {0}: AUC stable within {1} '
                          'over last {2} iterations.'.format(i, early_stop_tol, early_stop_patience))
                logger.info('early_stop iter=%d recent_auc=%s', i, recent)
                early_stopped_at = i
                break
    else:
        early_stopped_at = None

    extras = {
        'loss_history': loss_history,
        'lambda_history': lambda_history,
        'auc_per_epoch': auc_per_epoch,
        'early_stopped_at': early_stopped_at,
    }

    return output_dm, history, extras
