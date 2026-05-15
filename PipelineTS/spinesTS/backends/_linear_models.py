import math

import numpy as np


def _as_2d_float(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x.astype(np.float32, copy=False)


def _batch_size(n, batch_size):
    if batch_size == 'auto':
        log_n = max(4, min(9, int(np.log2(max(n, 16)))))
        return min(int(2 ** log_n), n)
    return int(batch_size)


class MLXLinearModel:
    def __init__(self, kind, in_features, out_features, use_revin=True, kernel_size=None,
                 loss_fn='huber', learning_rate=0.001, random_seed=42, weight_decay=1e-4,
                 **kwargs):
        self.kind = kind
        self.in_features = in_features
        self.out_features = out_features
        self.use_revin = use_revin
        self.kernel_size = kernel_size
        self.loss_fn_name = loss_fn
        self.learning_rate = learning_rate
        self.random_seed = 42 if random_seed is None else random_seed
        self.weight_decay = weight_decay
        self.backend = 'mlx'
        self.training_logs = {'epochs': [], 'train_loss': [], 'test_loss': [], 'lrs': []}
        self._cqr_enabled = False
        self.__spinesTS_is_fitted__ = False
        self._params = None

    def _enable_cqr(self, alpha=0.1):
        self._cqr_enabled = True
        self._cqr_alpha = alpha

    def _enable_residual_gate(self, *args, **kwargs):
        raise NotImplementedError("The residual gate is currently only available with the torch backend.")

    def _init_params(self, mx, output_dim):
        rng = np.random.RandomState(self.random_seed)
        scale = math.sqrt(6.0 / max(self.in_features + output_dim, 1))
        if self.kind == 'dlinear':
            return {
                'trend_w': mx.array(rng.uniform(-scale, scale, (self.in_features, output_dim)).astype(np.float32)),
                'trend_b': mx.zeros((output_dim,)),
                'seasonal_w': mx.array(rng.uniform(-scale, scale, (self.in_features, output_dim)).astype(np.float32)),
                'seasonal_b': mx.zeros((output_dim,)),
            }
        return {
            'w': mx.array(rng.uniform(-scale, scale, (self.in_features, output_dim)).astype(np.float32)),
            'b': mx.zeros((output_dim,)),
        }

    def _tree_map(self, fn, *trees):
        keys = trees[0].keys()
        return {k: fn(*(tree[k] for tree in trees)) for k in keys}

    def _copy_params(self, mx, params):
        return {k: mx.array(np.array(v)) for k, v in params.items()}

    def _moving_average(self, mx, x):
        kernel_size = self.kernel_size
        if kernel_size is None:
            kernel_size = max(3, self.in_features // 4)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size <= 1:
            return x
        pad = kernel_size // 2
        left = mx.broadcast_to(x[:, :1], (x.shape[0], pad))
        right = mx.broadcast_to(x[:, -1:], (x.shape[0], pad))
        padded = mx.concatenate([left, x, right], axis=1)
        parts = [padded[:, i:i + self.in_features] for i in range(kernel_size)]
        return mx.mean(mx.stack(parts, axis=0), axis=0)

    def fit(self, X, y, epochs=1000, batch_size='auto', eval_set=None, loss_type='min',
            metrics_name='score', monitor='val_loss', min_delta=0, patience=10,
            lr_scheduler='CosineAnnealingLR', lr_scheduler_patience=10, lr_factor=0.7,
            restore_best_weights=True, verbose=True, **kwargs):
        import mlx.core as mx

        X_np = _as_2d_float(X)
        y_np = _as_2d_float(y)
        n = X_np.shape[0]
        bs = _batch_size(n, batch_size)
        output_dim = self.out_features * (3 if self._cqr_enabled else 1)
        params = self._init_params(mx, output_dim)
        m = self._tree_map(lambda z: mx.zeros_like(z), params)
        v = self._tree_map(lambda z: mx.zeros_like(z), params)
        t = 0
        X_mx = mx.array(X_np)
        y_mx = mx.array(y_np)
        eval_xy = None
        if eval_set is not None:
            if isinstance(eval_set, list):
                eval_set = eval_set[0]
            eval_xy = (mx.array(_as_2d_float(eval_set[0])), mx.array(_as_2d_float(eval_set[1])))

        def forward(p, xb):
            x = xb
            if self.use_revin:
                mean = mx.mean(x, axis=1, keepdims=True)
                std = mx.sqrt(mx.mean((x - mean) ** 2, axis=1, keepdims=True)) + 1e-5
                x = (x - mean) / std
            else:
                mean = mx.zeros((x.shape[0], 1))
                std = mx.ones((x.shape[0], 1))

            if self.kind == 'nlinear':
                last_val = x[:, -1:]
                centered = x - last_val
                out = centered @ p['w'] + p['b']
                out = out + mx.broadcast_to(last_val, (x.shape[0], output_dim))
            else:
                trend = self._moving_average(mx, x)
                seasonal = x - trend
                out = trend @ p['trend_w'] + p['trend_b'] + seasonal @ p['seasonal_w'] + p['seasonal_b']

            out = out * mx.broadcast_to(std, (x.shape[0], output_dim)) + mx.broadcast_to(mean, (x.shape[0], output_dim))
            if self._cqr_enabled:
                f = self.out_features
                q = mx.stack([out[:, :f], out[:, f:2 * f], out[:, 2 * f:]], axis=-1)
                q = mx.sort(q, axis=-1)
                out = mx.concatenate([q[..., 0], q[..., 1], q[..., 2]], axis=-1)
            return out

        def point_loss(pred, target):
            err = pred - target
            if self.loss_fn_name == 'mse':
                return mx.mean(err ** 2)
            if self.loss_fn_name == 'mae':
                return mx.mean(mx.abs(err))
            abs_err = mx.abs(err)
            return mx.mean(mx.where(abs_err <= 1.0, 0.5 * err ** 2, abs_err - 0.5))

        def pinball(pred, target, tau):
            err = target - pred
            return mx.mean(mx.maximum(tau * err, (tau - 1.0) * err))

        def loss_fn_inner(p, xb, yb):
            pred = forward(p, xb)
            if self._cqr_enabled:
                f = yb.shape[-1]
                alpha = self._cqr_alpha
                return (
                    pinball(pred[:, :f], yb, alpha / 2.0) +
                    pinball(pred[:, f:2 * f], yb, 0.5) +
                    pinball(pred[:, 2 * f:], yb, 1.0 - alpha / 2.0)
                ) / 3.0
            return point_loss(pred, yb)

        value_and_grad = mx.value_and_grad(loss_fn_inner)
        best_params = self._copy_params(mx, params)
        best_loss = float('inf')
        wait = 0
        rng = np.random.RandomState(self.random_seed)

        for epoch in range(int(epochs)):
            if lr_scheduler == 'CosineAnnealingLR':
                lr = self.learning_rate * (0.5 * (1.0 + math.cos(math.pi * epoch / max(int(epochs), 1))))
                lr = max(lr, 1e-7)
            else:
                lr = self.learning_rate
            order = rng.permutation(n)
            epoch_losses = []
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                idx_mx = mx.array(idx)
                xb = X_mx[idx_mx]
                yb = y_mx[idx_mx]
                loss, grads = value_and_grad(params, xb, yb)
                t += 1
                m = self._tree_map(lambda a, g: 0.9 * a + 0.1 * g, m, grads)
                v = self._tree_map(lambda a, g: 0.999 * a + 0.001 * (g * g), v, grads)
                m_hat = self._tree_map(lambda a: a / (1.0 - 0.9 ** t), m)
                v_hat = self._tree_map(lambda a: a / (1.0 - 0.999 ** t), v)
                params = self._tree_map(
                    lambda w, mh, vh: w - lr * (mh / (mx.sqrt(vh) + 1e-8) + self.weight_decay * w),
                    params, m_hat, v_hat
                )
                mx.eval(params)
                epoch_losses.append(float(np.array(loss)))
            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float('inf')
            if eval_xy is not None and monitor == 'val_loss':
                val_loss_arr = loss_fn_inner(params, eval_xy[0], eval_xy[1])
                mx.eval(val_loss_arr)
                val_loss = float(np.array(val_loss_arr))
                current = val_loss
            else:
                val_loss = None
                current = train_loss

            self.training_logs['epochs'].append(epoch)
            self.training_logs['train_loss'].append(train_loss)
            self.training_logs['lrs'].append(lr)
            if val_loss is not None:
                self.training_logs['test_loss'].append(val_loss)
            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f" - val_loss: {val_loss:.4f}"
                print(msg)

            if current < best_loss - min_delta:
                best_loss = current
                best_params = self._copy_params(mx, params)
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if restore_best_weights:
            params = best_params
        self._params = params
        self._forward = forward
        self.__spinesTS_is_fitted__ = True
        return self

    def predict(self, X):
        if self._params is None:
            raise RuntimeError("This model is not fitted yet.")
        import mlx.core as mx
        X_mx = mx.array(_as_2d_float(X))
        pred = self._forward(self._params, X_mx)
        mx.eval(pred)
        return np.array(pred)
