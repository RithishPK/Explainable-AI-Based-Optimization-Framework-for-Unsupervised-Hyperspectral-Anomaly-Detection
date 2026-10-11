"""
streamlit_app/app.py
======================

Phase 2 - Adaptive Lambda Optimization Framework - interactive dashboard.

Run with:
    streamlit run streamlit_app/app.py

This app is a thin UI layer over the project's existing modules (OPBS band
selection, Z-score normalization, MSNet, SeT separation training) plus the
new `adaptive_lambda.AdaptiveLambda` controller. It does not reimplement
any research logic - everything is delegated to `app_utils.run_pipeline`,
which in turn calls the exact same functions `main.py` uses.
"""

import os
import sys

import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(__file__))
import app_utils as au
import xai_utils as xu
from adaptive_lambda import VALID_STRATEGIES


st.set_page_config(
    page_title='Adaptive Lambda - Hyperspectral Anomaly Detection',
    layout='wide',
)

st.title('🛰️ Explainable Adaptive Lambda Optimization for Hyperspectral Anomaly Detection')
st.caption(
    'Phase 2 dashboard - OPBS band selection, MSNet, and Separation Training '
    'are unchanged; Adaptive Lambda is the new research contribution.'
)

# ------------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------------
st.session_state.setdefault('results', None)
st.session_state.setdefault('strategy_used', None)
st.session_state.setdefault('dataset_name', None)
st.session_state.setdefault('xai_results', None)

# ------------------------------------------------------------------
# Sidebar - dataset upload + all hyperparameters
# ------------------------------------------------------------------
with st.sidebar:
    st.header('1. Dataset')
    uploaded = st.file_uploader(
        'Upload hyperspectral .mat file(s)',
        type=['mat'],
        accept_multiple_files=True,
        help=(
            "Upload 1 file if the cube and ground truth are together "
            "(or if there's no ground truth), or 2 files if they're "
            "separate (e.g. a data cube + a standalone _gt.mat file, "
            "like Indian Pines). Cube/mask are auto-detected by shape "
            "(3-D = cube, 2-D = mask), not by variable name. "
            "Up to 1GB per file. .mat v7.3/HDF5 format is supported "
            "(needs h5py installed) as well as older MATLAB formats. "
            "If nothing is uploaded, the bundled 'coast' dataset is used."
        ),
    )

    # Resolved here (not later in the main body) so dataset_name is
    # available to the "Adaptive Lambda" section below, which suggests a
    # dataset-specific lambda_base. Single source of truth - the main
    # body just uses these, it does not reload them.
    if uploaded:  # non-empty list from accept_multiple_files
        data, gt, raw_multiclass, dataset_name = au.load_uploaded_mat(uploaded)
    else:
        _bundled_data, _bundled_gt, _bundled_name = au.load_bundled_coast()
        data, gt, raw_multiclass, dataset_name = _bundled_data, _bundled_gt, None, _bundled_name

    st.header('2. Preprocessing')
    num_bs = st.slider('OPBS: number of selected bands', 8, 128, 64, step=8)
    data_norm = st.checkbox('Apply Z-score normalization', value=True)

    st.header('3. Model / Optimizer')
    num_layers = st.slider('MSNet layers', 1, 5, 3)
    lr = st.select_slider(
        'Learning rate',
        options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        value=1e-3,
        format_func=lambda v: f'{v:.0e}',
    )
    epochs = st.slider('Epochs per iteration', 5, 300, 50, step=5)
    max_iter = st.slider('Outer training iterations', 1, 15, 5)
    output_iter = st.slider(
        "Output iteration (which iteration's map is kept)", 1, max_iter, min(3, max_iter)
    )

    early_stop_on = st.checkbox(
        'Stop early once AUC stabilizes (optional, saves compute)', value=False,
        help='Disabled with no ground truth uploaded, since AUC has no meaning there. '
             'When enabled, stops once AUC has changed by less than the tolerance over '
             'the last N iterations - never before the "Output iteration" above has '
             'already been captured.',
    )
    early_stop_patience, early_stop_tol = None, None
    if early_stop_on:
        early_stop_patience = st.slider('Patience (iterations)', 2, max_iter, min(3, max_iter))
        early_stop_tol = st.number_input('AUC tolerance', min_value=1e-5, max_value=0.1,
                                          value=0.001, format='%.5f')

    st.header('4. Adaptive Lambda')
    adaptive_enabled = st.toggle('Enable Adaptive Lambda', value=True)
    strategy = st.selectbox(
        'Strategy',
        options=list(VALID_STRATEGIES),
        index=list(VALID_STRATEGIES).index('residual'),
        disabled=not adaptive_enabled,
        help='Ignored (treated as "fixed") when Adaptive Lambda is disabled.',
    )

    # Paper's own optimal lambda is dataset-specific (Section IV-B), not
    # a universal constant - defaulting everything to Coast's value (1e-3)
    # silently mistunes every other dataset. Suggest the right starting
    # point instead of leaving it to guesswork.
    _PAPER_OPTIMAL_LAMBDA = {
        'coast': (1e-3, 'paper-confirmed'),
        'texas-goast': (1e-3, 'same file as coast - see caption'),
        'gulfport': (1e-4, 'paper-confirmed'),
        'pavia': (1e-3, 'paper-confirmed'),
        'hydice': (1e-4, 'paper-confirmed'),
        'aviris': (1.0, 'inferred: shape (100,100,189) matches the paper\'s San Diego spec exactly, not confirmed by filename'),
        'los-angeles-1': (None, 'same file as abu_airport - not a paper benchmark, no known-optimal value; start around 1e-3 and sweep'),
        'abu_airport': (None, 'not a paper benchmark - no known-optimal value; start around 1e-3 and sweep'),
        'los-angeles-2': (None, 'not a paper benchmark - no known-optimal value; start around 1e-3 and sweep'),
        'cat-island': (None, 'not a paper benchmark - no known-optimal value; start around 1e-3 and sweep'),
        'sandiego': (None, 'no ground truth in this file - lambda tuning is unverifiable here regardless of value'),
    }
    _ds_key = (dataset_name or '').lower().strip()
    if _ds_key in _PAPER_OPTIMAL_LAMBDA:
        _sugg_val, _sugg_note = _PAPER_OPTIMAL_LAMBDA[_ds_key]
        if _sugg_val is not None:
            st.caption(f'💡 Suggested lambda_base for **{dataset_name}**: `{_sugg_val:g}` ({_sugg_note})')
        else:
            st.caption(f'💡 **{dataset_name}**: {_sugg_note}')

    lambda_base = st.number_input('lambda_base', min_value=1e-6, max_value=1.0,
                                   value=1e-3, format='%.6f')
    lambda_min, lambda_max = st.slider(
        'lambda_min / lambda_max range', min_value=1e-6, max_value=1.0,
        value=(1e-5, 1e-1), format='%.5f',
    )
    alpha = st.slider('alpha (sensitivity)', 0.0, 2.0, 0.5, step=0.05)
    ema_beta = st.slider('ema_beta (smoothing, EMA strategy only)', 0.0, 0.999, 0.9, step=0.01)

    train_ig_steps = 20
    gating_alpha, gating_min, gating_max = 0.5, 0.5, 1.5
    if adaptive_enabled and strategy == 'xai_alignment':
        st.caption(
            '⚠️ Unlike the other strategies, `xai_alignment` does NOT update '
            'every epoch. It recomputes Integrated Gradients once per outer '
            'iteration, right after the mask update, and holds lambda fixed '
            'in between - so the lambda-vs-epoch chart will look like flat '
            'steps with jumps at iteration boundaries. That is expected, '
            'not a bug (boundaries are marked on the chart).'
        )
        train_ig_steps = st.slider(
            'Training-time IG steps (xai_alignment only)', min_value=5, max_value=50,
            value=20, step=5,
            help='Runs once per outer iteration (max_iter times total), not every '
                 'epoch - unlike the XAI tab\'s IG, which runs once on demand. '
                 'Higher = more faithful alignment signal, linearly slower.',
        )

        st.markdown("**\u26a0\ufe0f EXPERIMENTAL: Multiscale Gating**")
        enable_gating = st.checkbox(
            'Enable IG-guided skip-connection gating', value=False,
            help='Reuses the same IG saliency map computed above, downsampled to '
                 'each of the model\'s scales, to gate how much each scale '
                 'contributes to the final reconstruction (Eq. 8\'s decoder-side '
                 'skip-connection sum) - based on whether that scale\'s attribution '
                 'still agrees with the mask once downsampled to it. This changes '
                 'the model\'s forward pass, not just the loss - unlike every other '
                 'feature in this app, this is new and not yet validated on real '
                 'training runs. Disabled = the paper\'s original unweighted '
                 'summation, byte-for-byte, guaranteed by the model defaulting '
                 '`layer_gates=None`.',
        )
        gating_alpha, gating_min, gating_max = 0.5, 0.5, 1.5
        if enable_gating:
            st.caption(
                'First run with this ON should be compared against the same run '
                'with it OFF, on a dataset you already trust (e.g. Coast) - not '
                'assumed to help without checking.'
            )
            gating_alpha = st.slider('Gating sensitivity (alpha_g)', 0.0, 1.0, 0.5, step=0.05)
            gating_min, gating_max = st.slider(
                'Gate value range', min_value=0.0, max_value=2.0, value=(0.5, 1.5),
                help='Gates are clipped to this range. Keep the minimum > 0 - a gate '
                     'of exactly 0 would completely zero a scale\'s contribution and '
                     'a negative one would flip its sign, neither of which is a '
                     'sensible "how trustworthy is this scale" value.',
            )
    else:
        enable_gating = False

    seed = st.number_input(
        'Random seed (0 = unseeded, non-reproducible)', min_value=0, max_value=2**31 - 1,
        value=0, step=1,
        help='Set a fixed seed for reproducible runs - needed if you plan to '
             'report mean/std across repeated runs of the same configuration.',
    )

    st.divider()
    run_clicked = st.button('▶ Run Detection', type='primary', use_container_width=True)

# ------------------------------------------------------------------
# Dataset was already loaded inside the sidebar (see "1. Dataset" above) -
# data/gt/raw_multiclass/dataset_name are already defined at this point.
# ------------------------------------------------------------------
rows, cols, bands = data.shape

# ------------------------------------------------------------------
# Detect a dataset switch and invalidate stale results. Without this,
# refreshing mid-run (which kills the in-progress pipeline - Streamlit
# does not persist background jobs across a reconnect) leaves whatever
# OLDER completed run is still in session_state on screen, with nothing
# indicating it belongs to a different dataset than the one now loaded.
# ------------------------------------------------------------------
_current_dataset_key = f'{dataset_name}:{data.shape}'
if (st.session_state.get('_dataset_key') is not None
        and st.session_state['_dataset_key'] != _current_dataset_key
        and st.session_state['results'] is not None):
    st.session_state['results'] = None
    st.session_state['xai_results'] = None
    st.session_state['strategy_used'] = None
    st.session_state['dataset_name'] = None
    st.info('New dataset detected - previous results cleared. Click **Run Detection** to run on this one.')
st.session_state['_dataset_key'] = _current_dataset_key

_gpu_available = torch.cuda.is_available()
_device_label = f'cuda ({torch.cuda.get_device_name(0)})' if _gpu_available else 'cpu'
_size_mb = data.nbytes / (1024 ** 2)
st.info(
    f'Dataset **{dataset_name}**: {rows} x {cols} pixels, {bands} bands ({_size_mb:.0f} MB in memory) '
    f'| Device: **{_device_label}**'
    + ('' if (gt is not None or raw_multiclass is not None) else ' - no ground truth provided')
)
if not _gpu_available:
    st.caption(
        '⚠️ Running on CPU - training will be noticeably slower, especially on the larger '
        '`sandiego` scene or with `xai_alignment`/high IG-step settings. If this machine has '
        'an NVIDIA GPU, run `nvidia-smi` in a terminal to check, then reinstall torch with a '
        'CUDA build if `torch.cuda.is_available()` currently returns False.'
    )

# ------------------------------------------------------------------
# Large-scene runtime warning. Keyed off total data volume (pixels x
# bands), not just spatial size - a scene can be large from either
# dimension (sandiego.mat's 400x400 vs. a hypothetical high-band-count
# upload with modest spatial size), and with the upload limit raised to
# 1GB, both are now realistic rather than hypothetical.
# ------------------------------------------------------------------
if _size_mb > 150:
    st.warning(
        f'This scene is {_size_mb:.0f} MB in memory ({rows}x{cols}x{bands}) - substantially '
        f'larger than the other bundled datasets. This affects **every** run regardless of '
        f'strategy: OPBS band selection itself scales with pixels x bands and adds real '
        f'up-front delay before training even starts, and every training epoch afterward '
        f'processes proportionally more spatial pixels too. Integrated Gradients (XAI tab, '
        f'`xai_alignment` strategy) is slower still on top of that. For a first test on a scene '
        f'this size, consider dropping epochs/max_iter well below the defaults to get a result '
        f'in a reasonable time, and avoid refreshing the page mid-run - Streamlit does not '
        f'persist an in-progress run across a page reload, so refreshing kills it outright.'
    )

# ------------------------------------------------------------------
# Multi-class ground truth (e.g. Indian Pines' 16 land-cover classes):
# never silently pick "the anomaly" - the user must choose explicitly,
# and every candidate they select is shown with its pixel count so the
# choice is informed, not a guess.
# ------------------------------------------------------------------
if raw_multiclass is not None:
    st.warning(
        'This ground-truth mask has more than 2 classes, so it looks like a '
        'land-cover/classification map, not a ready-made binary anomaly mask. '
        'Select which class(es) should count as the anomaly below - class 0 '
        'is conventionally unlabeled/background and rarely a sensible choice.'
    )
    class_counts = raw_multiclass['classes']
    total_px = rows * cols
    class_options = sorted(class_counts.keys())
    st.dataframe(
        {
            'class': class_options,
            'pixel_count': [class_counts[c] for c in class_options],
            'pct_of_scene': [f'{100 * class_counts[c] / total_px:.2f}%' for c in class_options],
        },
        use_container_width=True,
        hide_index=True,
    )
    selected_classes = st.multiselect(
        'Anomaly class(es)', options=class_options,
        help='Typically the rarest class(es), since anomalies are by definition uncommon. '
             'Selecting a large/common class will not produce a meaningful anomaly-detection result.',
    )
    if selected_classes:
        gt = au.binarize_multiclass_gt(raw_multiclass, selected_classes)
        n_anomaly = int(gt.sum())
        st.caption(f'Anomaly mask: {n_anomaly} pixels ({100 * n_anomaly / total_px:.2f}% of scene) '
                   f'from class(es) {selected_classes}.')
    else:
        gt = None
        st.caption('No class selected yet - detection will run without AUC/ROC until you pick at least one.')

# ------------------------------------------------------------------
# Single, persistent tab layout
# ------------------------------------------------------------------
tab_progress, tab_results, tab_roc, tab_xai, tab_downloads = st.tabs(
    ['📈 Live Training', '🗺️ Detection Maps', '📊 ROC & AUC', '🔬 Explainability (XAI)', '⬇️ Downloads']
)

# ------------------------------------------------------------------
# Tab 1: run the pipeline (if requested) and show live progress
# ------------------------------------------------------------------
with tab_progress:
    if run_clicked:
        cfg = au.RunConfig(
            num_bs=min(num_bs, bands),
            num_layers=num_layers,
            lr=lr,
            epochs=epochs,
            output_iter=output_iter,
            max_iter=max_iter,
            data_norm=data_norm,
            adaptive_lambda_enabled=adaptive_enabled,
            strategy=strategy if adaptive_enabled else 'fixed',
            lambda_base=lambda_base,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            alpha=alpha,
            ema_beta=ema_beta,
            ig_steps=train_ig_steps,
            seed=(seed if seed != 0 else None),
            early_stop_patience=early_stop_patience,
            early_stop_tol=early_stop_tol,
            enable_multiscale_gating=enable_gating,
            gating_alpha=gating_alpha,
            gating_min=gating_min,
            gating_max=gating_max,
        )

        st.subheader('Training progress')
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        col_l, col_r = st.columns(2)
        with col_l:
            st.caption('Lambda (live)')
            lambda_placeholder = st.empty()
        with col_r:
            st.caption('Reconstruction loss (live)')
            loss_placeholder = st.empty()
        st.caption('AUC per outer iteration (live)')
        auc_placeholder = st.empty()

        total_epochs = cfg.epochs * cfg.max_iter
        redraw_every = max(1, total_epochs // 150)  # keep the UI responsive
        gt_missing = gt is None

        lambda_buf, loss_buf, auc_buf = [], [], []

        def epoch_callback(epoch, loss_val, lmda_val):
            loss_buf.append(loss_val)
            lambda_buf.append(lmda_val)
            frac = min(1.0, (epoch + 1) / total_epochs)
            progress_bar.progress(frac)
            status_text.text(
                f'Epoch {epoch + 1}/{total_epochs} | loss={loss_val:.4f} | lambda={lmda_val:.2e}'
            )
            if (epoch + 1) % redraw_every == 0 or (epoch + 1) == total_epochs:
                lambda_placeholder.line_chart(lambda_buf)
                loss_placeholder.line_chart(loss_buf)

        def iter_callback(iter_idx, auc):
            auc_buf.append(auc)
            if not gt_missing:
                auc_placeholder.line_chart(auc_buf)

        with st.spinner('Running Separation Training...'):
            results = au.run_pipeline(
                data, gt, cfg,
                epoch_callback=epoch_callback,
                iter_callback=iter_callback,
            )

        st.session_state['results'] = results
        st.session_state['strategy_used'] = cfg.strategy
        st.session_state['dataset_name'] = dataset_name
        progress_bar.progress(1.0)
        status_text.text('Training complete. See the other tabs for results.')
        if results.get('early_stopped_at') is not None:
            st.info(
                f"Stopped early at iteration {results['early_stopped_at']} of {cfg.max_iter} "
                f"- AUC had stabilized within tolerance."
            )
        if results.get('final_layer_gates') is not None:
            _gates_str = ', '.join(f'layer {i}: {g:.3f}' for i, g in enumerate(results['final_layer_gates']))
            st.info(f"Multiscale gating (experimental) - final gate values: {_gates_str}")

    elif st.session_state['results'] is not None:
        st.success('Showing results from the last completed run. Click "Run Detection" to run again.')
    else:
        st.write('Configure your run in the sidebar, then click **Run Detection**.')

# ------------------------------------------------------------------
# Tabs 2-4: render persisted results (survive Streamlit reruns via
# session_state so the user can freely switch tabs afterwards)
# ------------------------------------------------------------------
results = st.session_state['results']

if results is not None:
    strategy_used = st.session_state['strategy_used']
    dataset_name_disp = st.session_state['dataset_name']

    with tab_results:
        st.subheader('Input (before detection)')
        st.pyplot(au.fig_pseudocolor(results['pseudocolor_img'], results['pseudocolor_bands']))
        st.caption(
            '3-band pseudocolor composite from the full raw cube (before OPBS/normalization), '
            'percentile-stretched per channel for contrast - not true color (no wavelength '
            'metadata is available to pick literal RGB bands), same spirit as Fig. 4 in the '
            'MSNet paper. Shown for visual contrast against the cleaned-up detection map below; '
            'the detection map is the model\'s output, not another view of this image.'
        )

        st.subheader('Detection Maps')
        n_cols = 2 if results['gt_missing'] else 3
        cols_ = st.columns(n_cols)
        with cols_[0]:
            st.pyplot(au.fig_detection_map(results['pr_dm'], 'MSNet + SeT Detection Map'))
        with cols_[1]:
            st.pyplot(au.fig_detection_map(results['rx_dm'], 'RX Baseline Detection Map'))
        if not results['gt_missing']:
            with cols_[2]:
                st.pyplot(au.fig_ground_truth(results['gt']))
        else:
            st.caption('No ground truth was provided with this dataset - only detection maps are shown.')

        with st.expander('Selected OPBS band indices'):
            st.write(results['band_idx'])

    with tab_roc:
        st.subheader('ROC Curve')
        st.pyplot(au.fig_roc(results))

        st.subheader('Histories')
        c1, c2, c3 = st.columns(3)
        with c1:
            st.pyplot(au.fig_lambda_history(results, strategy_used))
        with c2:
            st.pyplot(au.fig_loss_history(results))
        with c3:
            st.pyplot(au.fig_auc_history(results))

        if not results['gt_missing']:
            m1, m2 = st.columns(2)
            m1.metric('Final AUC (MSNet+SeT)', f"{results['auc_pr']:.4f}")
            m2.metric('RX baseline AUC', f"{results['auc_rx']:.4f}")

    with tab_xai:
        st.subheader('Explainability: Plain Gradient vs Integrated Gradients vs GradCAM')
        st.caption(
            'Runs on the trained model from the last completed run above. '
            'Integrated Gradients does a full forward+backward pass per step, '
            'so cost scales with both step count and image size - a 400x400 '
            'scene will take noticeably longer than a 100x100 one at the same '
            'step count.'
        )
        ig_steps = st.slider(
            'Integrated Gradients steps', min_value=10, max_value=100, value=30, step=10,
            help='Higher = smoother/more faithful attribution, but linearly slower. '
                 '50 was the original notebook default; 30 is a faster default here.',
        )
        xai_clicked = st.button('🔬 Run XAI Analysis', use_container_width=True)

        if xai_clicked:
            xai_progress = st.progress(0.0)
            xai_status = st.empty()

            def ig_progress_callback(step, total):
                xai_progress.progress(step / total)
                xai_status.text(f'Integrated Gradients: step {step}/{total}')

            with st.spinner('Computing plain gradient + GradCAM (fast)...'):
                xai_results = xu.run_xai(
                    results['model'], results['x_bs'], results['band_idx'],
                    results['rows'], results['cols'],
                    ig_steps=ig_steps, ig_progress_callback=ig_progress_callback,
                )
            st.session_state['xai_results'] = xai_results
            xai_status.text('XAI analysis complete.')

        xai_results = st.session_state['xai_results']
        if xai_results is not None:
            st.pyplot(xu.fig_band_importance_comparison(xai_results))
            st.pyplot(xu.fig_spatial_comparison(
                xai_results, results['gt'], results['pr_dm'],
                results.get('auc_pr', float('nan')), results['gt_missing'],
            ))
            st.pyplot(xu.fig_xai_dashboard(
                xai_results, results['gt'], results['pr_dm'],
                results.get('auc_pr', float('nan')), results['gt_missing'],
            ))

            with st.expander('Top attributed bands (raw values)'):
                c1, c2 = st.columns(2)
                with c1:
                    st.write('**Plain gradient**')
                    st.write(dict(zip(
                        [f'Band {b}' for b in xai_results['top_plain_bands']],
                        [round(float(v), 4) for v in xai_results['top_plain_vals']],
                    )))
                with c2:
                    st.write(f"**Integrated Gradients ({xai_results['ig_steps']} steps)**")
                    st.write(dict(zip(
                        [f'Band {b}' for b in xai_results['top_ig_bands']],
                        [round(float(v), 4) for v in xai_results['top_ig_vals']],
                    )))
        else:
            st.info('Click **Run XAI Analysis** above to compute attributions for the current model.')


        st.subheader('Download results')

        st.download_button(
            'Download metrics CSV (epoch, loss, lambda, auc)',
            data=au.metrics_csv_bytes(results),
            file_name=f'{dataset_name_disp}_phase2_history.csv',
            mime='text/csv',
        )

        st.download_button(
            'Download trained model weights (.pt)',
            data=au.model_state_bytes(results['model']),
            file_name=f'{dataset_name_disp}_{results["model"].name}.pt',
            mime='application/octet-stream',
        )

        st.markdown('**Plots (PNG)**')
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button(
                'ROC curve', data=au.fig_to_png_bytes(au.fig_roc(results)),
                file_name='roc.png', mime='image/png',
            )
        with d2:
            st.download_button(
                'Lambda history', data=au.fig_to_png_bytes(au.fig_lambda_history(results, strategy_used)),
                file_name='lambda_history.png', mime='image/png',
            )
        with d3:
            st.download_button(
                'Loss history', data=au.fig_to_png_bytes(au.fig_loss_history(results)),
                file_name='loss_history.png', mime='image/png',
            )
        with d4:
            st.download_button(
                'AUC history', data=au.fig_to_png_bytes(au.fig_auc_history(results)),
                file_name='auc_history.png', mime='image/png',
            )
