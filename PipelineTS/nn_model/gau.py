from PipelineTS.spinesTS.nn import GAUNet
from spinesUtils.asserts import generate_function_kwargs
from PipelineTS.base.spines_base import SpinesNNModelMixin
from PipelineTS.spinesTS.preprocessing import GAUDataPreprocessor


class GAUModel(SpinesNNModelMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=30,
            quantile=0.9,
            random_state=None,
            level=3,
            learning_rate=0.001,
            accelerator='auto',
            verbose=False,
            epochs=1000,
            batch_size='auto',
            patience=20,
            min_delta=0,
            lr_scheduler='OneCycleLR',
            lr_scheduler_patience=10,
            lr_factor=0.5,
            restore_best_weights=True,
            loss_type='min',
            use_features=True,
            use_augmentation=True,
            augmentation_ratio=0.3,
            dropout=0.2,
            weight_decay=1e-4,
            query_key_dim=512,
            expansion_factor=4.0
    ):
        """
        GAUModel: A wrapper for the GAUNet neural network model with additional features.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 30
            The number of lagged values to use as input features for training and prediction.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        random_state : int or None, optional, default: None
            The random seed for reproducibility.
        level : int, optional, default: 3
            The level parameter for the GAUNet model.
        learning_rate : float, optional, default: 0.001
            The learning rate for training the GAUNet model.
        accelerator : str, optional, default: 'auto'
            The PyTorch Lightning accelerator to use during training.
        verbose : bool, optional, default: False
            Whether to print detailed information during training.
        epochs : int, optional, default: 1000
            The number of epochs for training the GAUNet model.
        batch_size : int or 'auto', optional, default: 'auto'
            The batch size used during training. If set to 'auto', it is determined automatically.
        patience : int, optional, default: 20
            The patience parameter for early stopping during training.
        min_delta : float, optional, default: 0
            The minimum change in the monitored quantity to qualify as an improvement during early stopping.
        lr_scheduler : str, optional, default: 'OneCycleLR'
            The learning rate scheduler used during training.
        lr_scheduler_patience : int, optional, default: 10
            The patience parameter for the learning rate scheduler.
        lr_factor : float, optional, default: 0.5
            The factor by which the learning rate is reduced during training.
        restore_best_weights : bool, optional, default: True
            Whether to restore the model weights from the epoch with the best value of the monitored quantity.
        loss_type : str, optional, default: 'min'
            The type of loss function to use during training.
        use_features : bool, optional, default: True
            Whether to use feature engineering.
        use_augmentation : bool, optional, default: True
            Whether to use data augmentation.
        augmentation_ratio : float, optional, default: 0.3
            The ratio of data to augment.
        dropout : float, optional, default: 0.2
            The dropout rate for the GAUNet model.
        weight_decay : float, optional, default: 1e-4
            The weight decay for the optimizer.
        query_key_dim : int, optional, default: 512
            The query key dimension for the GAU layer.
        expansion_factor : float, optional, default: 4.0
            The expansion factor for the GAU layer.

        Attributes
        ----------
        model : spinesTS.nn.GAUNet
            The GAUNet neural network model.
        preprocessor : GAUDataPreprocessor
            The data preprocessor for feature engineering and augmentation.
        """
        super().__init__(time_col=time_col, target_col=target_col, accelerator=accelerator)

        self.all_configs['model_configs'] = generate_function_kwargs(
            GAUNet,
            in_features=lags,
            out_features=lags,
            level=level,
            learning_rate=learning_rate,
            random_seed=random_state,
            device=self.accelerator,
            loss_fn='huber',
            dropout=dropout,
            weight_decay=weight_decay,
            query_key_dim=query_key_dim,
            expansion_factor=expansion_factor
        )

        self.last_dt = None
        
        self.preprocessor = GAUDataPreprocessor()

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': 0,
                'verbose': verbose,
                'epochs': epochs,
                'batch_size': batch_size,
                'patience': patience,
                'min_delta': min_delta,
                'lr_scheduler': lr_scheduler,
                'lr_scheduler_patience': lr_scheduler_patience,
                'lr_factor': lr_factor,
                'restore_best_weights': restore_best_weights,
                'loss_type': loss_type,
                'use_features': use_features,
                'use_augmentation': use_augmentation,
                'augmentation_ratio': augmentation_ratio
            }
        )

        self.x = None

        self.model = self._define_model()

    def _define_model(self):
        """
        Define the GAUNet neural network model.

        Returns
        -------
        spinesTS.nn.GAUNet
            The GAUNet neural network model.
        """
        return GAUNet(**self.all_configs['model_configs'])
        
    def _data_preprocess(self, data, mode='train'):
        """
        重写数据预处理方法，使用高级特征工程
        
        Parameters
        ----------
        data : pd.DataFrame
            输入数据
        mode : str, default='train'
            模式，'train'或'predict'
            
        Returns
        -------
        x : numpy.ndarray
            处理后的特征
        y : numpy.ndarray, optional
            目标值，仅在mode='train'时返回
        """
        if mode == 'train':
            if self.all_configs['use_features']:
                X, y = self.preprocessor.fit_transform(
                    data, 
                    self.all_configs['time_col'], 
                    self.all_configs['target_col'], 
                    self.all_configs['lags']
                )
                
                if self.all_configs['use_augmentation'] and X.shape[0] > 0:
                    X, y = self.preprocessor.augment(
                        X, y, augment_ratio=self.all_configs['augmentation_ratio']
                    )
                    
                return X, y
            else:
                return super()._data_preprocess(data, mode)
        else:
            if self.all_configs['use_features']:
                X, _ = self.preprocessor.transform(
                    data, 
                    self.all_configs['time_col'], 
                    self.all_configs['target_col'], 
                    self.all_configs['lags']
                )
                return X
            else:
                return super()._data_preprocess(data, mode)
