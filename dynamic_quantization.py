import torch
import copy
from torch.quantization import PlaceholderObserver, MinMaxObserver, QConfig, HistogramObserver, PerChannelMinMaxObserver

def quantize_model_dynamic(model, precision):

    if precision == 8:

        quantization_config = QConfig(
            activation=PlaceholderObserver.with_args(
                                                dtype=torch.quint8,
                                                quant_min=0,
                                                quant_max=255,
                                                is_dynamic=True,
                                            ),
            weight=MinMaxObserver.with_args(
                                            dtype=torch.qint8,
                                            qscheme=torch.per_tensor_symmetric,
                                            quant_min=-128,
                                            quant_max=127,
                                        ),
        )

        embedding_qconfig = QConfig(
            activation=PlaceholderObserver,
            weight=PerChannelMinMaxObserver.with_args(
                                                dtype=torch.quint8,
                                                qscheme=torch.per_channel_affine_float_qparams,
                                                ch_axis=0,
                                                quant_min=0,
                                                quant_max=255,
                                            ),
        )

    elif precision == 4:
        quantization_config = QConfig(
            activation=PlaceholderObserver.with_args(
                                                dtype=torch.quint8,
                                                quant_min=0,
                                                quant_max=15,
                                                is_dynamic=True,
                                            ),
            weight=MinMaxObserver.with_args(
                                            dtype=torch.qint8,
                                            qscheme=torch.per_tensor_symmetric,
                                            quant_min=-8,
                                            quant_max=7,
                                        ),
        )

        embedding_qconfig = QConfig(
            activation=PlaceholderObserver,
            weight=PerChannelMinMaxObserver.with_args(
                                                dtype=torch.quint8,
                                                qscheme=torch.per_channel_affine_float_qparams,
                                                ch_axis=0,
                                                quant_min=0,
                                                quant_max=15,
                                            ),
        )
    else:
        raise ValueError("Precision not supported")


    _model = copy.deepcopy(model)

    print("======== Quantizing the model DYNAMIC =========")
    
    torch.quantization.quantize_dynamic(
        _model, {torch.nn.Embedding: embedding_qconfig}, dtype=torch.quint8, inplace=True
    )

    torch.quantization.quantize_dynamic(
        _model, {torch.nn.Linear: quantization_config, torch.nn.LayerNorm: quantization_config}, dtype=torch.qint8, inplace=True
    )

    return _model